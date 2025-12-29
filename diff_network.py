from __future__ import annotations

import os
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm


# -------------------------
# Config
# -------------------------

@dataclass
class TrainConfig:
    data_dir: str = "./data_images"   # folder with png/jpg/etc
    out_dir: str = "./out_diffusion"
    image_size: int = 128
    batch_size: int = 32
    num_workers: int = 4
    lr: float = 2e-4
    epochs: int = 30
    grad_clip: float = 1.0
    device: str = "cuda" if torch.cuda.is_available() else "mps"

    # diffusion
    timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 2e-2

    # training behavior
    use_random_crop: bool = True      # if source images larger than 128
    crop_ink_bias: float = 0.7        # probability to try finding a crop that contains ink
    crop_tries: int = 20              # tries to find an "inky" crop

    # logging / sampling
    save_every_epochs: int = 5
    sample_every_epochs: int = 5
    sample_n: int = 16
    seed: int = 42


# -------------------------
# Utilities
# -------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_image_grid(tensor_bchw: torch.Tensor, path: str, nrow: int = 4) -> None:
    """
    tensor_bchw: float in [0,1], shape [B,1,H,W]
    """
    b, c, h, w = tensor_bchw.shape
    assert c == 1
    ncol = nrow
    nrow_count = (b + ncol - 1) // ncol

    grid = torch.ones((1, nrow_count * h, ncol * w), dtype=tensor_bchw.dtype)
    for i in range(b):
        r = i // ncol
        col = i % ncol
        grid[:, r*h:(r+1)*h, col*w:(col+1)*w] = tensor_bchw[i]

    img = (grid.clamp(0, 1) * 255.0).byte().squeeze(0).cpu().numpy()
    Image.fromarray(img, mode="L").save(path)


# -------------------------
# Dataset (folder of images)
# -------------------------

class BWFolderDataset(Dataset):
    def __init__(self, data_dir: str, image_size: int, use_random_crop: bool,
                 crop_ink_bias: float, crop_tries: int):
        super().__init__()
        self.paths: List[Path] = []
        d = Path(data_dir)
        if not d.exists():
            raise FileNotFoundError(f"data_dir not found: {data_dir}")

        exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
        for p in d.rglob("*"):
            if p.suffix.lower() in exts:
                self.paths.append(p)
        if not self.paths:
            raise RuntimeError(f"No images found under: {data_dir}")

        self.image_size = image_size
        self.use_random_crop = use_random_crop
        self.crop_ink_bias = crop_ink_bias
        self.crop_tries = crop_tries

        # Convert to 1-channel, then to tensor
        # We'll normalize to [-1, 1] for diffusion.
        self.to_tensor = transforms.ToTensor()  # gives [0,1] float

    def __len__(self) -> int:
        return len(self.paths)

    @staticmethod
    def _random_crop(img: Image.Image, size: int, ink_bias: float, tries: int) -> Image.Image:
        w, h = img.size
        if w < size or h < size:
            # pad to fit then crop
            pad_w = max(0, size - w)
            pad_h = max(0, size - h)
            img = transforms.Pad((0, 0, pad_w, pad_h), fill=255)(img)
            w, h = img.size

        # If ink_bias enabled, try to find a crop with darker pixels (ink)
        want_ink = (random.random() < ink_bias)
        best = None
        best_score = -1.0

        for _ in range(max(1, tries if want_ink else 1)):
            left = random.randint(0, w - size)
            top = random.randint(0, h - size)
            crop = img.crop((left, top, left + size, top + size))

            if not want_ink:
                return crop

            # score: fraction of pixels darker than a threshold
            # (Image is L mode 0=black, 255=white)
            arr = torch.from_numpy(
                # fast-ish conversion without numpy import:
                # PIL -> bytes -> tensor (fallback: use crop.getdata())
                # We'll do simpler: crop.getdata() is ok for 128x128
                torch.tensor(list(crop.getdata()), dtype=torch.uint8).view(size, size).numpy()
            )  # this line is clunky; we’ll avoid it below
        # That approach is messy; do it properly with crop.getdata():
        # (Re-implement loop cleanly)

    def __getitem__(self, idx: int) -> torch.Tensor:
        path = self.paths[idx]
        img = Image.open(path).convert("L")  # grayscale

        # Optional: if your dataset is true B/W already, keep it.
        # If not, you can hard-threshold to force binary look.
        # We'll keep grayscale during training, then binarize samples at the end.

        if self.use_random_crop:
            img = self._random_crop_with_ink_bias(img, self.image_size, self.crop_ink_bias, self.crop_tries)
        else:
            img = img.resize((self.image_size, self.image_size), Image.BICUBIC)

        x = self.to_tensor(img)            # [1,H,W] in [0,1]
        x = x * 2.0 - 1.0                  # [-1,1]
        return x

    @staticmethod
    def _random_crop_with_ink_bias(img: Image.Image, size: int, ink_bias: float, tries: int) -> Image.Image:
        w, h = img.size
        if w < size or h < size:
            pad_w = max(0, size - w)
            pad_h = max(0, size - h)
            img = transforms.Pad((0, 0, pad_w, pad_h), fill=255)(img)
            w, h = img.size

        want_ink = (random.random() < ink_bias)
        best_crop = None
        best_score = -1.0

        attempts = tries if want_ink else 1
        for _ in range(max(1, attempts)):
            left = random.randint(0, w - size)
            top = random.randint(0, h - size)
            crop = img.crop((left, top, left + size, top + size))

            if not want_ink:
                return crop

            # score ink presence: fraction of pixels below 220
            data = list(crop.getdata())  # length size*size
            ink = sum(1 for v in data if v < 220)
            score = ink / (size * size)

            if score > best_score:
                best_score = score
                best_crop = crop

        return best_crop if best_crop is not None else img.crop((0, 0, size, size))


# -------------------------
# Diffusion schedule (DDPM)
# -------------------------

class Diffusion:
    def __init__(self, timesteps: int, beta_start: float, beta_end: float, device: str):
        self.T = timesteps
        self.device = device

        betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)

        self.betas = betas
        self.alphas = alphas
        self.alpha_bars = alpha_bars

        self.sqrt_alpha_bars = torch.sqrt(alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - alpha_bars)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x0: [B,1,H,W] in [-1,1]
        t:  [B] int64
        """
        if noise is None:
            noise = torch.randn_like(x0)

        a = self.sqrt_alpha_bars[t].view(-1, 1, 1, 1)
        b = self.sqrt_one_minus_alpha_bars[t].view(-1, 1, 1, 1)
        return a * x0 + b * noise

    @torch.no_grad()
    def p_sample_loop(self, model: nn.Module, shape: Tuple[int, int, int, int]) -> torch.Tensor:
        """
        shape: (B,1,H,W)
        returns x0-ish in [-1,1]
        """
        x = torch.randn(shape, device=self.device)
        for i in reversed(range(self.T)):
            t = torch.full((shape[0],), i, device=self.device, dtype=torch.long)
            eps_pred = model(x, t)

            alpha = self.alphas[i]
            alpha_bar = self.alpha_bars[i]
            beta = self.betas[i]

            # DDPM mean
            coef1 = 1.0 / torch.sqrt(alpha)
            coef2 = (1.0 - alpha) / torch.sqrt(1.0 - alpha_bar)
            mean = coef1 * (x - coef2 * eps_pred)

            if i > 0:
                noise = torch.randn_like(x)
                var = torch.sqrt(beta)
                x = mean + var * noise
            else:
                x = mean
        return x


# -------------------------
# Simple U-Net
# -------------------------

def sinusoidal_time_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """
    t: [B] int64
    returns: [B, dim]
    """
    device = t.device
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(0, half, device=device).float() / (half - 1)
    )
    args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, tdim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.time_proj = nn.Linear(tdim, out_ch)
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        h = self.norm1(h)
        h = F.silu(h)

        # add time embedding
        temb = self.time_proj(t_emb).view(-1, h.shape[1], 1, 1)
        h = h + temb

        h = self.conv2(h)
        h = self.norm2(h)
        h = F.silu(h)

        return h + self.skip(x)


class SimpleUNet(nn.Module):
    def __init__(self, base_ch: int = 64, tdim: int = 256):
        super().__init__()
        self.tdim = tdim
        self.time_mlp = nn.Sequential(
            nn.Linear(tdim, tdim),
            nn.SiLU(),
            nn.Linear(tdim, tdim),
        )

        self.in_conv = nn.Conv2d(1, base_ch, 3, padding=1)

        # Down
        self.d1 = ResBlock(base_ch, base_ch, tdim)
        self.down1 = nn.Conv2d(base_ch, base_ch * 2, 4, stride=2, padding=1)  # 128->64
        self.d2 = ResBlock(base_ch * 2, base_ch * 2, tdim)
        self.down2 = nn.Conv2d(base_ch * 2, base_ch * 4, 4, stride=2, padding=1)  # 64->32
        self.d3 = ResBlock(base_ch * 4, base_ch * 4, tdim)
        self.down3 = nn.Conv2d(base_ch * 4, base_ch * 4, 4, stride=2, padding=1)  # 32->16

        # Bottleneck
        self.mid1 = ResBlock(base_ch * 4, base_ch * 4, tdim)
        self.mid2 = ResBlock(base_ch * 4, base_ch * 4, tdim)

        # Up
        self.up3 = nn.ConvTranspose2d(base_ch * 4, base_ch * 4, 4, stride=2, padding=1)  # 16->32
        self.u3 = ResBlock(base_ch * 8, base_ch * 4, tdim)
        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 4, stride=2, padding=1)  # 32->64
        self.u2 = ResBlock(base_ch * 4, base_ch * 2, tdim)
        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch, 4, stride=2, padding=1)      # 64->128
        self.u1 = ResBlock(base_ch * 2, base_ch, tdim)

        self.out_norm = nn.GroupNorm(8, base_ch)
        self.out_conv = nn.Conv2d(base_ch, 1, 3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = sinusoidal_time_embedding(t, self.tdim)
        t_emb = self.time_mlp(t_emb)

        x0 = self.in_conv(x)

        h1 = self.d1(x0, t_emb)
        x1 = self.down1(h1)

        h2 = self.d2(x1, t_emb)
        x2 = self.down2(h2)

        h3 = self.d3(x2, t_emb)
        x3 = self.down3(h3)

        m = self.mid1(x3, t_emb)
        m = self.mid2(m, t_emb)

        u3 = self.up3(m)
        u3 = torch.cat([u3, h3], dim=1)
        u3 = self.u3(u3, t_emb)

        u2 = self.up2(u3)
        u2 = torch.cat([u2, h2], dim=1)
        u2 = self.u2(u2, t_emb)

        u1 = self.up1(u2)
        u1 = torch.cat([u1, h1], dim=1)
        u1 = self.u1(u1, t_emb)

        out = self.out_norm(u1)
        out = F.silu(out)
        out = self.out_conv(out)
        return out  # predicted noise ε


# -------------------------
# Train / Sample
# -------------------------

def train(cfg: TrainConfig) -> None:
    set_seed(cfg.seed)
    os.makedirs(cfg.out_dir, exist_ok=True)

    ds = BWFolderDataset(
        data_dir=cfg.data_dir,
        image_size=cfg.image_size,
        use_random_crop=cfg.use_random_crop,
        crop_ink_bias=cfg.crop_ink_bias,
        crop_tries=cfg.crop_tries,
    )
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True)

    model = SimpleUNet(base_ch=64, tdim=256).to(cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    diffusion = Diffusion(cfg.timesteps, cfg.beta_start, cfg.beta_end, cfg.device)

    step = 0
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        pbar = tqdm(dl, desc=f"epoch {epoch}/{cfg.epochs}")
        for x0 in pbar:
            x0 = x0.to(cfg.device)  # [-1,1]

            b = x0.shape[0]
            t = torch.randint(0, cfg.timesteps, (b,), device=cfg.device, dtype=torch.long)
            noise = torch.randn_like(x0)

            xt = diffusion.q_sample(x0, t, noise=noise)
            noise_pred = model(xt, t)

            loss = F.mse_loss(noise_pred, noise)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()

            step += 1
            pbar.set_postfix(loss=float(loss.item()))

        # Save checkpoint
        if epoch % cfg.save_every_epochs == 0 or epoch == cfg.epochs:
            ckpt_path = os.path.join(cfg.out_dir, f"ckpt_epoch_{epoch}.pt")
            torch.save({"model": model.state_dict(), "cfg": cfg.__dict__}, ckpt_path)

        # Sample
        if epoch % cfg.sample_every_epochs == 0 or epoch == cfg.epochs:
            model.eval()
            with torch.no_grad():
                samples = diffusion.p_sample_loop(model, (cfg.sample_n, 1, cfg.image_size, cfg.image_size))
                # back to [0,1]
                samples01 = (samples.clamp(-1, 1) + 1.0) / 2.0

                # binarize for clean black/white look
                bw = (samples01 > 0.5).float()

                out_path = os.path.join(cfg.out_dir, f"samples_epoch_{epoch}.png")
                save_image_grid(bw, out_path, nrow=int(math.sqrt(cfg.sample_n)))

    print(f"Done. Outputs in: {cfg.out_dir}")


# Prepare images
def prepare_images():
    for f_cnt in range(100000):

        if f_cnt % 8 != 0:
            continue

        filename_source = f'/Users/michaelkolomenkin/Data/playo/images_for_yaron/room_{f_cnt}.png'
        filename_dest = f'/Users/michaelkolomenkin/Data/playo/diff_images/room_{f_cnt}.png'

        shutil.copy2(filename_source, filename_dest)

if __name__ == "__main__":
    # prepare_images()
    cfg = TrainConfig(
        data_dir="/Users/michaelkolomenkin/Data/playo/smaller_images",   # change me
        out_dir="/Users/michaelkolomenkin/Data/playo/out_diff",
        image_size=16,
        batch_size=32,
        epochs=30,
        use_random_crop=False,       # IMPORTANT for "not separated characters"
    )
    train(cfg)
