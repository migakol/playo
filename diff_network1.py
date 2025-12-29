import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from tqdm import tqdm
import math


class SinusoidalPositionEmbedding(nn.Module):
    """Sinusoidal position embedding for time steps"""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ResidualBlock(nn.Module):
    """Residual block with time embedding"""

    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.residual_conv = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, time_emb):
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)

        # Add time embedding
        time_emb = self.time_mlp(time_emb)
        h = h + time_emb[:, :, None, None]

        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)

        return h + self.residual_conv(x)


class AttentionBlock(nn.Module):
    """Self-attention block for better global coherence"""

    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.norm = nn.GroupNorm(8, channels)
        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(channels, channels, 1)
        self.v = nn.Conv2d(channels, channels, 1)
        self.proj_out = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        residual = x
        x = self.norm(x)

        q = self.q(x).view(b, c, h * w).transpose(1, 2)
        k = self.k(x).view(b, c, h * w)
        v = self.v(x).view(b, c, h * w).transpose(1, 2)

        # Compute attention
        attn = torch.bmm(q, k) / math.sqrt(c)
        attn = F.softmax(attn, dim=-1)

        out = torch.bmm(attn, v).transpose(1, 2).view(b, c, h, w)
        out = self.proj_out(out)

        return out + residual


class UNet(nn.Module):
    """U-Net architecture for diffusion model"""

    def __init__(self, in_channels=1, model_channels=64, num_res_blocks=2,
                 channel_mult=(1, 2, 4, 8), attention_levels=(2, 3)):
        super().__init__()

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.num_res_blocks = num_res_blocks

        # Time embedding
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbedding(model_channels),
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        # Initial convolution
        self.input_blocks = nn.ModuleList([
            nn.Conv2d(in_channels, model_channels, 3, padding=1)
        ])

        # Downsampling path
        ch = model_channels
        input_block_channels = [ch]
        ds = 1

        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [ResidualBlock(ch, mult * model_channels, time_embed_dim)]
                ch = mult * model_channels
                if level in attention_levels:
                    layers.append(AttentionBlock(ch))
                self.input_blocks.append(nn.Sequential(*layers))
                input_block_channels.append(ch)

            if level != len(channel_mult) - 1:  # Don't downsample on last level
                self.input_blocks.append(nn.Conv2d(ch, ch, 3, stride=2, padding=1))
                input_block_channels.append(ch)
                ds *= 2

        # Middle blocks
        self.middle_block = nn.Sequential(
            ResidualBlock(ch, ch, time_embed_dim),
            AttentionBlock(ch),
            ResidualBlock(ch, ch, time_embed_dim)
        )

        # Upsampling path
        self.output_blocks = nn.ModuleList()
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                layers = [ResidualBlock(ch + input_block_channels.pop(),
                                        mult * model_channels, time_embed_dim)]
                ch = mult * model_channels
                if level in attention_levels:
                    layers.append(AttentionBlock(ch))
                if level and i == num_res_blocks:
                    layers.append(nn.ConvTranspose2d(ch, ch, 4, stride=2, padding=1))
                self.output_blocks.append(nn.Sequential(*layers))

        # Final output
        self.out = nn.Sequential(
            nn.GroupNorm(8, ch),
            nn.SiLU(),
            nn.Conv2d(ch, in_channels, 3, padding=1)
        )

    def forward(self, x, timesteps):
        # Time embedding
        time_emb = self.time_embed(timesteps)

        # Downsampling
        hs = []
        h = x
        for module in self.input_blocks:
            if isinstance(module, nn.Sequential) and any(isinstance(m, ResidualBlock) for m in module):
                # Apply ResidualBlock with time embedding
                for m in module:
                    if isinstance(m, ResidualBlock):
                        h = m(h, time_emb)
                    else:
                        h = m(h)
            else:
                h = module(h)
            hs.append(h)

        # Middle
        for module in self.middle_block:
            if isinstance(module, ResidualBlock):
                h = module(h, time_emb)
            else:
                h = module(h)

        # Upsampling
        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            for m in module:
                if isinstance(m, ResidualBlock):
                    h = m(h, time_emb)
                else:
                    h = m(h)

        return self.out(h)


class DDPMScheduler:
    """DDPM noise scheduler"""

    def __init__(self, num_train_timesteps=1000, beta_start=0.0001, beta_end=0.02):
        self.num_train_timesteps = num_train_timesteps

        # Linear beta schedule
        self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

        # Calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)

    def add_noise(self, original_samples, noise, timesteps):
        """Add noise to samples"""
        sqrt_alpha_prod = self.sqrt_alphas_cumprod[timesteps]
        sqrt_one_minus_alpha_prod = self.sqrt_one_minus_alphas_cumprod[timesteps]

        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples

    def step(self, model_output, timestep, sample):
        """Single denoising step"""
        t = timestep

        # Current and previous alpha values
        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod_prev[t] if t > 0 else torch.tensor(1.0)

        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        current_beta_t = 1 - current_alpha_t

        # Compute predicted original sample from model output
        pred_original_sample = (sample - beta_prod_t.sqrt() * model_output) / alpha_prod_t.sqrt()

        # Compute coefficients for pred_original_sample and current sample
        pred_original_sample_coeff = (alpha_prod_t_prev.sqrt() * current_beta_t) / beta_prod_t
        current_sample_coeff = current_alpha_t.sqrt() * beta_prod_t_prev / beta_prod_t

        # Compute predicted previous sample
        pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * sample

        # Add noise if not final step
        variance = 0
        if t > 0:
            device = model_output.device
            variance = (self.posterior_variance[t].sqrt() * torch.randn_like(sample)).to(device)

        pred_prev_sample = pred_prev_sample + variance

        return pred_prev_sample


class DDIMScheduler:
    """DDIM (Denoising Diffusion Implicit Models) scheduler - Much faster sampling"""

    def __init__(self, num_train_timesteps=1000, num_inference_timesteps=50,
                 beta_start=0.0001, beta_end=0.02, eta=0.0):
        self.num_train_timesteps = num_train_timesteps
        self.num_inference_timesteps = num_inference_timesteps
        self.eta = eta  # 0.0 = deterministic, 1.0 = stochastic like DDPM

        # Linear beta schedule
        self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        # Calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod).to('mps')
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod).to('mps')

        # Set inference timesteps
        self.set_timesteps(num_inference_timesteps)

    def set_timesteps(self, num_inference_timesteps):
        """Set the discrete timesteps used for the diffusion chain"""
        self.num_inference_timesteps = num_inference_timesteps

        # Create evenly spaced timesteps
        step = self.num_train_timesteps // self.num_inference_timesteps
        self.timesteps = torch.arange(0, self.num_train_timesteps, step)
        self.timesteps = torch.flip(self.timesteps, dims=(0,))  # Reverse for denoising

        # Add final timestep if needed
        if self.timesteps[0] != self.num_train_timesteps - 1:
            self.timesteps = torch.cat([torch.tensor([self.num_train_timesteps - 1]), self.timesteps])

    def add_noise(self, original_samples, noise, timesteps):
        """Add noise to samples (same as DDPM)"""
        sqrt_alpha_prod = self.sqrt_alphas_cumprod[timesteps]
        sqrt_one_minus_alpha_prod = self.sqrt_one_minus_alphas_cumprod[timesteps]

        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples

    def step(self, model_output, timestep, sample, prev_timestep=None):
        """Single DDIM denoising step"""
        if prev_timestep is None:
            # Find the previous timestep
            timestep_idx = (self.timesteps == timestep).nonzero(as_tuple=True)[0].item()
            prev_timestep = self.timesteps[timestep_idx + 1] if timestep_idx < len(self.timesteps) - 1 else 0

        # Get alpha values
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else torch.tensor(1.0)

        beta_prod_t = 1 - alpha_prod_t

        # Predict original sample (x_0)
        pred_original_sample = (sample - beta_prod_t.sqrt() * model_output) / alpha_prod_t.sqrt()
        pred_original_sample = torch.clamp(pred_original_sample, -1, 1)  # Clamp for stability

        # Compute variance
        variance = self._get_variance(timestep, prev_timestep)
        std_dev_t = self.eta * variance.sqrt()

        # Compute predicted previous sample
        pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t ** 2).sqrt() * model_output
        pred_prev_sample = alpha_prod_t_prev.sqrt() * pred_original_sample + pred_sample_direction

        # Add noise if eta > 0
        if self.eta > 0:
            noise = torch.randn_like(model_output)
            pred_prev_sample = pred_prev_sample + std_dev_t * noise

        return pred_prev_sample

    def _get_variance(self, timestep, prev_timestep=None):
        if prev_timestep is None:
            timestep_idx = (self.timesteps == timestep).nonzero(as_tuple=True)[0].item()
            prev_timestep = self.timesteps[timestep_idx + 1] if timestep_idx < len(self.timesteps) - 1 else 0

        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else torch.tensor(1.0)
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev

        variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)
        return variance


class DPMSolverScheduler:
    """DPM-Solver++ scheduler - Even faster, high-order solver"""

    def __init__(self, num_train_timesteps=1000, num_inference_timesteps=20,
                 beta_start=0.0001, beta_end=0.02, solver_order=2):
        self.num_train_timesteps = num_train_timesteps
        self.num_inference_timesteps = num_inference_timesteps
        self.solver_order = min(solver_order, 3)  # Max order 3

        # Linear beta schedule
        self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0).to('mps')

        # Lambda values for DPM-Solver
        self.lambdas = 0.5 * torch.log(self.alphas_cumprod / (1 - self.alphas_cumprod))

        self.set_timesteps(num_inference_timesteps)
        self.model_outputs = []

    def set_timesteps(self, num_inference_timesteps):
        """Set the discrete timesteps used for the diffusion chain"""
        self.num_inference_timesteps = num_inference_timesteps

        # Use uniform spacing in lambda space for better performance
        lambda_min = self.lambdas[0].item()
        lambda_max = self.lambdas[-1].item()

        inference_lambdas = torch.linspace(lambda_max, lambda_min, num_inference_timesteps)

        # Convert back to timesteps
        timesteps = []
        for lam in inference_lambdas:
            # Find closest lambda value
            idx = torch.argmin(torch.abs(self.lambdas - lam))
            timesteps.append(idx.item())

        self.timesteps = torch.tensor(timesteps, dtype=torch.long)
        self.lambdas_inference = inference_lambdas

    def add_noise(self, original_samples, noise, timesteps):
        """Add noise to samples"""
        sqrt_alpha_prod = torch.sqrt(self.alphas_cumprod[timesteps])
        sqrt_one_minus_alpha_prod = torch.sqrt(1.0 - self.alphas_cumprod[timesteps])

        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples

    def step(self, model_output, timestep, sample, timestep_idx=None):
        """Single DPM-Solver step"""
        if timestep_idx is None:
            timestep_idx = (self.timesteps == timestep).nonzero(as_tuple=True)[0].item()

        # Store model output
        self.model_outputs.append(model_output)

        # Only keep outputs we need
        if len(self.model_outputs) > self.solver_order:
            self.model_outputs = self.model_outputs[-self.solver_order:]

        # Get lambda values
        lambda_t = self.lambdas[timestep]
        lambda_prev = self.lambdas_inference[timestep_idx + 1] if timestep_idx < len(self.lambdas_inference) - 1 else \
        self.lambdas[0]

        # Convert to alpha values
        alpha_t = torch.sigmoid(lambda_t)
        sigma_t = torch.sigmoid(-lambda_t)
        alpha_prev = torch.sigmoid(lambda_prev)
        sigma_prev = torch.sigmoid(-lambda_prev)

        h = lambda_prev - lambda_t

        if len(self.model_outputs) == 1 or self.solver_order == 1:
            # First-order (Euler)
            x_prev = (alpha_prev / alpha_t) * sample - (sigma_prev * torch.expm1(h)) * model_output
        elif len(self.model_outputs) == 2 or self.solver_order == 2:
            # Second-order
            m0 = model_output
            m1 = self.model_outputs[-2]

            x_prev = (alpha_prev / alpha_t) * sample - (sigma_prev * torch.expm1(h)) * m0
            x_prev = x_prev - (sigma_prev / (2 * h)) * torch.expm1(h) ** 2 * (m0 - m1)
        else:
            # Third-order
            m0 = model_output
            m1 = self.model_outputs[-2]
            m2 = self.model_outputs[-3]

            x_prev = (alpha_prev / alpha_t) * sample - (sigma_prev * torch.expm1(h)) * m0
            x_prev = x_prev - (sigma_prev / (2 * h)) * torch.expm1(h) ** 2 * (m0 - m1)
            x_prev = x_prev - (sigma_prev / (6 * h ** 2)) * (
                        torch.expm1(h) ** 3 - torch.expm1(h) - h * torch.expm1(h)) * (2 * m0 - 3 * m1 + m2)

        return x_prev


class ImageDataset(Dataset):
    """Dataset for loading 128x128 B&W images"""

    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.image_files = [f for f in os.listdir(image_dir)
                            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_path).convert('L')  # Convert to grayscale
        # image = image.resize((128, 128), Image.Resampling.LANCZOS)

        # Convert to tensor and normalize to [-1, 1]
        image = torch.from_numpy(np.array(image)).float() / 127.5 - 1.0
        image = image.unsqueeze(0)  # Add channel dimension

        if self.transform:
            image = self.transform(image)

        return image


class DiffusionModel:
    """Complete diffusion model wrapper"""

    def __init__(self, device='cuda' if torch.cuda.is_available() else 'mps', scheduler_type='ddim'):
        self.device = device
        self.model = UNet().to(device)

        # Initialize scheduler based on type
        if scheduler_type == 'ddpm':
            self.scheduler = DDPMScheduler()
            self.inference_steps = 1000
        elif scheduler_type == 'ddim':
            self.scheduler = DDIMScheduler(num_inference_timesteps=50)  # 20x faster
            self.inference_steps = 50
        elif scheduler_type == 'dpm':
            self.scheduler = DPMSolverScheduler(num_inference_timesteps=20)  # 50x faster
            self.inference_steps = 20
        else:
            raise ValueError("scheduler_type must be 'ddpm', 'ddim', or 'dpm'")

        self.scheduler_type = scheduler_type
        self.optimizer = optim.AdamW(self.model.parameters(), lr=2e-4, weight_decay=1e-4)

    def train_step(self, batch):
        """Single training step"""
        batch = batch.to(self.device)
        batch_size = batch.shape[0]

        # Sample random timesteps
        timesteps = torch.randint(0, self.scheduler.num_train_timesteps, (batch_size,), device=self.device)

        # Sample noise
        noise = torch.randn_like(batch)

        # Add noise to images
        noisy_images = self.scheduler.add_noise(batch, noise, timesteps)

        # Predict noise
        noise_pred = self.model(noisy_images, timesteps)

        # Compute loss
        loss = F.mse_loss(noise_pred, noise)

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        return loss.item()

    def train(self, dataloader, num_epochs=100):
        """Train the model"""
        self.model.train()

        for epoch in range(num_epochs):
            epoch_loss = 0
            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")

            for batch in progress_bar:
                loss = self.train_step(batch)
                epoch_loss += loss
                progress_bar.set_postfix({'loss': f'{loss:.4f}'})

            avg_loss = epoch_loss / len(dataloader)
            print(f"Epoch {epoch + 1}, Average Loss: {avg_loss:.4f}")

            # Generate sample every 10 epochs
            if (epoch + 1) % 10 == 0:
                self.generate_samples(num_samples=4, save_path=f'samples_epoch_{epoch + 1}.png')

    @torch.no_grad()
    def generate_samples(self, num_samples=4, save_path=None):
        """Generate new samples using the configured scheduler"""
        self.model.eval()

        # Start from pure noise
        samples = torch.randn(num_samples, 1, 16, 16, device=self.device)

        # Move scheduler tensors to device
        if hasattr(self.scheduler, 'betas'):
            self.scheduler.betas = self.scheduler.betas.to(self.device)
        if hasattr(self.scheduler, 'alphas_cumprod'):
            self.scheduler.alphas_cumprod = self.scheduler.alphas_cumprod.to(self.device)
        if hasattr(self.scheduler, 'alphas_cumprod_prev'):
            self.scheduler.alphas_cumprod_prev = self.scheduler.alphas_cumprod_prev.to(self.device)
        if hasattr(self.scheduler, 'lambdas'):
            self.scheduler.lambdas = self.scheduler.lambdas.to(self.device)
        if hasattr(self.scheduler, 'lambdas_inference'):
            self.scheduler.lambdas_inference = self.scheduler.lambdas_inference.to(self.device)

        # Clear model outputs for DPM solver
        if self.scheduler_type == 'dpm':
            self.scheduler.model_outputs = []

        # Get timesteps based on scheduler type
        if self.scheduler_type == 'ddpm':
            timesteps = reversed(range(self.scheduler.num_train_timesteps))
        else:
            timesteps = self.scheduler.timesteps

        # Denoising loop
        progress_bar = tqdm(enumerate(timesteps), total=len(timesteps),
                            desc=f"Generating samples ({self.scheduler_type.upper()})")

        for i, t in progress_bar:
            # Create timestep tensor
            if isinstance(t, torch.Tensor):
                timestep = t.item()
                timesteps_tensor = torch.full((num_samples,), timestep, device=self.device, dtype=torch.long)
            else:
                timestep = t
                timesteps_tensor = torch.full((num_samples,), timestep, device=self.device, dtype=torch.long)

            # Predict noise
            noise_pred = self.model(samples, timesteps_tensor)

            # Remove noise based on scheduler type
            if self.scheduler_type == 'ddpm':
                samples = self.scheduler.step(noise_pred, timestep, samples)
            elif self.scheduler_type == 'ddim':
                samples = self.scheduler.step(noise_pred, timestep, samples)
            elif self.scheduler_type == 'dpm':
                samples = self.scheduler.step(noise_pred, timestep, samples, timestep_idx=i)

        # Convert to images and display
        samples = (samples + 1) * 127.5  # Denormalize
        samples = samples.clamp(0, 255).cpu().numpy().astype(np.uint8)

        # Plot samples
        fig, axes = plt.subplots(2, 2, figsize=(8, 8))
        axes = axes.ravel()

        for i in range(num_samples):
            axes[i].imshow(samples[i, 0], cmap='gray')
            axes[i].axis('off')
            axes[i].set_title(f'Generated Sample {i + 1} ({self.scheduler_type.upper()})')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()

        self.model.train()
        return samples

    def benchmark_schedulers(self, num_samples=4):
        """Compare generation speed across different schedulers"""
        import time

        schedulers = ['ddpm', 'ddim', 'dpm']
        results = {}

        for scheduler_name in schedulers:
            print(f"\nTesting {scheduler_name.upper()} scheduler...")

            # Temporarily switch scheduler
            original_scheduler = self.scheduler_type
            self.__init__(self.device, scheduler_name)

            # Measure generation time
            start_time = time.time()
            samples = self.generate_samples(num_samples, save_path=f'{scheduler_name}_samples.png')
            end_time = time.time()

            generation_time = end_time - start_time
            steps = self.inference_steps

            results[scheduler_name] = {
                'time': generation_time,
                'steps': steps,
                'time_per_step': generation_time / steps,
                'speedup': None  # Will calculate relative to DDPM
            }

            print(
                f"{scheduler_name.upper()}: {generation_time:.2f}s for {steps} steps ({generation_time / steps:.3f}s per step)")

        # Calculate speedups relative to DDPM
        ddpm_time = results['ddpm']['time']
        for scheduler_name in results:
            if scheduler_name != 'ddpm':
                results[scheduler_name]['speedup'] = ddpm_time / results[scheduler_name]['time']

        print("\n" + "=" * 50)
        print("SCHEDULER COMPARISON SUMMARY:")
        print("=" * 50)
        for scheduler_name, metrics in results.items():
            speedup_str = f"{metrics['speedup']:.1f}x faster" if metrics['speedup'] else "baseline"
            print(
                f"{scheduler_name.upper():>8}: {metrics['time']:>6.2f}s | {metrics['steps']:>4d} steps | {speedup_str}")

        # Restore original scheduler
        self.__init__(self.device, original_scheduler)

        return results

    def save_model(self, path):
        """Save model checkpoint"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)

    def load_model(self, path):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


# Example usage
if __name__ == "__main__":
    # Initialize model with faster scheduler
    device = 'cuda' if torch.cuda.is_available() else 'mps'
    print(f"Using device: {device}")

    # Try different schedulers:
    # 'ddpm' - Original DDPM (1000 steps, slow but high quality)
    # 'ddim' - DDIM (50 steps, 20x faster, deterministic)
    # 'dpm'  - DPM-Solver++ (20 steps, 50x faster, best quality/speed trade-off)

    print("Initializing model with DDIM scheduler (20x faster than DDPM)...")
    diffusion_model = DiffusionModel(device=device, scheduler_type='dpm')

    # Create dataset (replace with your image directory)
    # dataset = ImageDataset('path/to/your/images')
    # dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)

    # For testing without actual images, create synthetic data
    print("Creating synthetic dataset for demonstration...")

    dataset = ImageDataset('/Users/michaelkolomenkin/Data/playo/smaller_images')
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    diffusion_model.train(dataloader, num_epochs=5)


    # Benchmark different schedulers
    print("\nBenchmarking scheduler performance...")
    # diffusion_model.benchmark_schedulers(num_samples=4)

    # Generate final samples with fastest scheduler
    print("\nGenerating samples with DPM-Solver++ (fastest)...")
    # diffusion_model = DiffusionModel(device=device, scheduler_type='dpm')
    diffusion_model.generate_samples(num_samples=4, save_path='final_samples_fast.png')

    # Save the trained model
    diffusion_model.save_model('diffusion_model_checkpoint.pth')
    print("Model saved to 'diffusion_model_checkpoint.pth'")

    print("\nSPEED COMPARISON:")
    print("DDPM:  1000 steps (baseline)")
    print("DDIM:    50 steps (20x faster, deterministic)")
    print("DPM++:   20 steps (50x faster, best quality/speed)")