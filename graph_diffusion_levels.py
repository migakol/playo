import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GAE, VGAE
from torch_geometric.data import Data, DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import os
import random
from tqdm import tqdm
import math

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


class GraphDataProcessor:
    """Process game level data into PyTorch Geometric format"""

    def __init__(self):
        self.node_type_to_idx = {'center': 0, 'entrance': 1, 'corridor': 2}
        self.idx_to_node_type = {v: k for k, v in self.node_type_to_idx.items()}

    def parse_graph_file(self, filepath):
        """Parse a single graph file in the specified format"""
        with open(filepath, 'r') as f:
            lines = f.readlines()

        # Skip header lines (---)
        data_lines = [line.strip() for line in lines if line.strip() and not line.strip().startswith('---')]

        nodes = []
        edges = []
        node_features = []

        import csv
        import io

        for line in data_lines:
            # Skip header line if present
            if 'node_type,pos,connections' in line:
                continue

            # Parse the CSV line properly
            csv_reader = csv.reader(io.StringIO(line))
            try:
                parts = next(csv_reader)
                if len(parts) < 4:
                    continue

                node_id = int(parts[0])
                node_type = parts[1]
                position = eval(parts[2])  # Parse position list
                connections = eval(parts[3])  # Parse connections list
            except (ValueError, IndexError, SyntaxError) as e:
                print(f"Error parsing line: {line.strip()} - {e}")
                continue

            # Node features: [node_type_one_hot, x, y, z]
            type_one_hot = [0, 0, 0]
            if node_type in self.node_type_to_idx:
                type_one_hot[self.node_type_to_idx[node_type]] = 1

            features = type_one_hot + position
            node_features.append(features)

            # Create edges
            for connected_node in connections:
                edges.append([node_id, connected_node])

        return torch.tensor(node_features, dtype=torch.float), torch.tensor(edges, dtype=torch.long).t()

    def load_dataset(self, data_dir):
        """Load all graph files from directory"""
        graphs = []

        for filename in os.listdir(data_dir):
            if filename.endswith('.txt'):
                filepath = os.path.join(data_dir, filename)
                try:
                    node_features, edge_index = self.parse_graph_file(filepath)
                    graph_data = Data(x=node_features, edge_index=edge_index)
                    graphs.append(graph_data)
                except Exception as e:
                    print(f"Error processing {filename}: {e}")

        return graphs


class GraphEncoder(nn.Module):
    """Graph encoder using GCN layers"""

    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(GraphEncoder, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv_mu = GCNConv(hidden_dim, latent_dim)
        self.conv_logvar = GCNConv(hidden_dim, latent_dim)

    def forward(self, x, edge_index):
        h = F.relu(self.conv1(x, edge_index))
        h = F.relu(self.conv2(h, edge_index))
        mu = self.conv_mu(h, edge_index)
        logvar = self.conv_logvar(h, edge_index)
        return mu, logvar


class GraphDecoder(nn.Module):
    """Graph decoder to reconstruct adjacency matrix"""

    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(GraphDecoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        h = F.relu(self.fc1(z))
        h = F.relu(self.fc2(h))
        return torch.sigmoid(self.fc3(h))


class VariationalGraphAutoEncoder(nn.Module):
    """Variational Graph Auto Encoder"""

    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VariationalGraphAutoEncoder, self).__init__()
        self.encoder = GraphEncoder(input_dim, hidden_dim, latent_dim)
        self.decoder = GraphDecoder(latent_dim, hidden_dim, input_dim)
        self.latent_dim = latent_dim

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, edge_index):
        mu, logvar = self.encoder(x, edge_index)
        z = self.reparameterize(mu, logvar)

        # Global mean pooling for graph-level representation
        batch_size = 1  # Assuming single graph
        graph_embedding = torch.mean(z, dim=0, keepdim=True)

        reconstructed = self.decoder(graph_embedding)
        return reconstructed, mu, logvar, z


class SinusoidalPositionEmbedding(nn.Module):
    """Sinusoidal position embedding for time steps in diffusion"""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, timesteps):
        device = timesteps.device

        # Ensure timesteps has batch dimension
        if timesteps.dim() == 0:
            timesteps = timesteps.unsqueeze(0)

        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = timesteps[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class GraphDiffusionModel(nn.Module):
    """Diffusion model for graph generation"""

    def __init__(self, graph_dim, hidden_dim, time_dim=128):
        super(GraphDiffusionModel, self).__init__()
        self.graph_dim = graph_dim
        self.time_embedding = SinusoidalPositionEmbedding(time_dim)

        self.input_projection = nn.Linear(graph_dim, hidden_dim)
        self.time_projection = nn.Linear(time_dim, hidden_dim)

        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=hidden_dim * 2,
                dropout=0.1,
                batch_first=True
            ) for _ in range(6)
        ])

        self.output_projection = nn.Linear(hidden_dim, graph_dim)

    def forward(self, x, timesteps):
        # x: [batch_size, graph_dim]
        # timesteps: [batch_size]

        time_emb = self.time_embedding(timesteps)

        h = self.input_projection(x)
        h = h + self.time_projection(time_emb)

        # Add sequence dimension for transformer
        h = h.unsqueeze(1)  # [batch_size, 1, hidden_dim]

        for layer in self.layers:
            h = layer(h)

        h = h.squeeze(1)  # [batch_size, hidden_dim]
        return self.output_projection(h)


class DDIM:
    """Denoising Diffusion Implicit Models"""

    def __init__(self, model, num_timesteps=1000, beta_start=0.0001, beta_end=0.02):
        self.model = model
        self.num_timesteps = num_timesteps

        # Linear schedule for betas
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.alphas_cumprod)

    def q_sample(self, x_start, t, noise=None):
        """Forward diffusion process"""
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1)  # Reshape to [batch_size, 1]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1,
                                                                                     1)  # Reshape to [batch_size, 1]

        return (sqrt_alphas_cumprod_t * x_start +
                sqrt_one_minus_alphas_cumprod_t * noise)

    def p_sample(self, x, t, eta=0.0):
        """Reverse diffusion process (DDIM sampling)"""
        # Convert scalar timestep to tensor for the model
        if isinstance(t, int):
            t_tensor = torch.tensor([t] * x.shape[0], device=x.device, dtype=torch.float32)
        else:
            t_tensor = t.float()

        pred_noise = self.model(x, t_tensor)

        # Use scalar t for indexing
        if isinstance(t, int):
            t_idx = t
        else:
            t_idx = int(t.item())

        alpha_cumprod_t = self.alphas_cumprod[t_idx]
        alpha_cumprod_t_prev = self.alphas_cumprod[t_idx - 1] if t_idx > 0 else torch.tensor(1.0, device=x.device)

        # Predict x_0
        pred_x_start = (x - torch.sqrt(1 - alpha_cumprod_t) * pred_noise) / torch.sqrt(alpha_cumprod_t)

        # DDIM sampling
        sigma = eta * torch.sqrt((1 - alpha_cumprod_t_prev) / (1 - alpha_cumprod_t)) * torch.sqrt(
            1 - alpha_cumprod_t / alpha_cumprod_t_prev)

        pred_x_prev = (torch.sqrt(alpha_cumprod_t_prev) * pred_x_start +
                       torch.sqrt(1 - alpha_cumprod_t_prev - sigma ** 2) * pred_noise)

        if t_idx > 0:
            noise = torch.randn_like(x)
            pred_x_prev = pred_x_prev + sigma * noise

        return pred_x_prev

    def sample(self, shape, device, num_inference_steps=50):
        """Generate samples using DDIM"""
        x = torch.randn(shape, device=device)

        timesteps = torch.linspace(self.num_timesteps - 1, 0, num_inference_steps, dtype=torch.long, device=device)

        for i, t in enumerate(tqdm(timesteps, desc="DDIM Sampling")):
            # Convert to scalar for p_sample
            t_scalar = int(t.item())
            x = self.p_sample(x, t_scalar)

        return x


class GraphLevelGenerator:
    """Main class for training and generating game levels"""

    def __init__(self, input_dim=6, hidden_dim=128, latent_dim=64,
                 device='cuda' if torch.cuda.is_available() else 'mps'):
        self.device = device
        self.input_dim = input_dim  # 3 (node types) + 3 (position)
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # Initialize models
        self.vae = VariationalGraphAutoEncoder(input_dim, hidden_dim, latent_dim).to(device)
        self.diffusion_model = GraphDiffusionModel(latent_dim, hidden_dim).to(device)
        self.ddim = DDIM(self.diffusion_model)

        # Move DDIM tensors to device
        self.ddim.betas = self.ddim.betas.to(device)
        self.ddim.alphas = self.ddim.alphas.to(device)
        self.ddim.alphas_cumprod = self.ddim.alphas_cumprod.to(device)
        self.ddim.sqrt_alphas_cumprod = self.ddim.sqrt_alphas_cumprod.to(device)
        self.ddim.sqrt_one_minus_alphas_cumprod = self.ddim.sqrt_one_minus_alphas_cumprod.to(device)

        # Data processor
        self.data_processor = GraphDataProcessor()

    def train_vae(self, graphs, epochs=100, lr=0.001):
        """Train the Variational Graph Auto Encoder"""
        optimizer = torch.optim.Adam(self.vae.parameters(), lr=lr)

        self.vae.train()
        losses = []

        for epoch in tqdm(range(epochs), desc="Training VAE"):
            epoch_loss = 0

            for k, graph_data in enumerate(graphs):
                if k % 1000 == 0:
                    print(f"Processing graph {k}...")
                graph_data = graph_data.to(self.device)
                optimizer.zero_grad()

                reconstructed, mu, logvar, z = self.vae(graph_data.x, graph_data.edge_index)

                # Reconstruction loss (MSE)
                recon_loss = F.mse_loss(reconstructed, torch.mean(graph_data.x, dim=0, keepdim=True))

                # KL divergence loss
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / mu.size(0)

                loss = recon_loss + 0.1 * kl_loss
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(graphs)
            losses.append(avg_loss)

            # if epoch % 20 == 0:
            print(f"Epoch {epoch}, Loss: {avg_loss:.4f}")

        return losses

    def extract_graph_embeddings(self, graphs):
        """Extract graph embeddings using trained VAE"""
        self.vae.eval()
        embeddings = []

        with torch.no_grad():
            for graph_data in graphs:
                graph_data = graph_data.to(self.device)
                mu, logvar = self.vae.encoder(graph_data.x, graph_data.edge_index)
                # Global mean pooling for graph-level representation
                graph_embedding = torch.mean(mu, dim=0)
                embeddings.append(graph_embedding.cpu())

        return torch.stack(embeddings)

    def train_diffusion(self, embeddings, epochs=200, lr=0.0001, batch_size=32):
        """Train the diffusion model"""
        embeddings = embeddings.to(self.device)
        optimizer = torch.optim.Adam(self.diffusion_model.parameters(), lr=lr)

        self.diffusion_model.train()
        losses = []

        for epoch in tqdm(range(epochs), desc="Training Diffusion"):
            epoch_loss = 0
            num_batches = 0

            # Create batches
            indices = torch.randperm(len(embeddings))

            for i in range(0, len(embeddings), batch_size):
                batch_indices = indices[i:i + batch_size]
                batch_embeddings = embeddings[batch_indices]

                optimizer.zero_grad()

                # Sample random timesteps
                t = torch.randint(0, self.ddim.num_timesteps, (batch_embeddings.size(0),), device=self.device)

                # Add noise
                noise = torch.randn_like(batch_embeddings)
                noisy_embeddings = self.ddim.q_sample(batch_embeddings, t, noise)

                # Predict noise
                pred_noise = self.diffusion_model(noisy_embeddings, t.float())

                # Compute loss
                loss = F.mse_loss(pred_noise, noise)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
            losses.append(avg_loss)

            if epoch % 50 == 0:
                print(f"Epoch {epoch}, Loss: {avg_loss:.4f}")

        return losses

    def generate_graphs(self, num_graphs=10, num_inference_steps=50):
        """Generate new graphs"""
        self.diffusion_model.eval()
        self.vae.eval()

        with torch.no_grad():
            # Sample from diffusion model
            embeddings = self.ddim.sample(
                shape=(num_graphs, self.latent_dim),
                device=self.device,
                num_inference_steps=num_inference_steps
            )

            # Decode embeddings to graph features
            generated_graphs = []
            for embedding in embeddings:
                # Use VAE decoder to get node features
                reconstructed = self.vae.decoder(embedding.unsqueeze(0))
                generated_graphs.append(reconstructed.cpu().numpy())

        return generated_graphs

    def save_models(self, path_prefix="models/graph_diffusion"):
        """Save trained models"""
        os.makedirs(os.path.dirname(path_prefix), exist_ok=True)
        torch.save(self.vae.state_dict(), f"{path_prefix}_vae.pt")
        torch.save(self.diffusion_model.state_dict(), f"{path_prefix}_diffusion.pt")
        print(f"Models saved to {path_prefix}_*.pt")

    def load_models(self, path_prefix="models/graph_diffusion"):
        """Load trained models"""
        self.vae.load_state_dict(torch.load(f"{path_prefix}_vae.pt"))
        self.diffusion_model.load_state_dict(torch.load(f"{path_prefix}_diffusion.pt"))
        print(f"Models loaded from {path_prefix}_*.pt")


def create_sample_data(num_graphs=50, save_dir="sample_data"):
    """Create sample graph data for testing"""
    import csv
    import io

    os.makedirs(save_dir, exist_ok=True)

    node_types = ['center', 'entrance', 'corridor']

    for i in range(num_graphs):
        num_nodes = np.random.randint(10, 30)

        lines = ["---", "node_type,pos,connections"]

        for j in range(num_nodes):
            node_type = np.random.choice(node_types)
            pos = [np.random.uniform(-10, 10), -1.0, np.random.uniform(-10, 10)]

            # Create random connections (ensuring some connectivity)
            num_connections = np.random.randint(1, min(4, num_nodes - 1))
            possible_connections = [k for k in range(num_nodes) if k != j]
            connections = np.random.choice(possible_connections, num_connections, replace=False).tolist()

            # Use CSV writer to properly quote the fields
            output = io.StringIO()
            writer = csv.writer(output)
            writer.writerow([j, node_type, str(pos), str(connections)])
            line = output.getvalue().strip()
            lines.append(line)

        lines.append("---")

        with open(os.path.join(save_dir, f"level_{i:03d}.txt"), 'w') as f:
            f.write('\n'.join(lines))


# Example usage
if __name__ == "__main__":
    # Create sample data if needed
    create_sample_data(num_graphs=100, save_dir="sample_data")

    # Initialize generator
    generator = GraphLevelGenerator()

    need_train = False
    # Load and process data
    if need_train:
        graphs = generator.data_processor.load_dataset("sample_data")
        print(f"Loaded {len(graphs)} graphs")

        # Train VAE
        print("Training VAE...")
        vae_losses = generator.train_vae(graphs, epochs=100)

        # Extract embeddings
        print("Extracting embeddings...")
        embeddings = generator.extract_graph_embeddings(graphs)
        print(f"Extracted embeddings shape: {embeddings.shape}")

        # Train diffusion model
        print("Training diffusion model...")
        diffusion_losses = generator.train_diffusion(embeddings, epochs=200)
    else:
        generator.load_models('/Users/michaelkolomenkin/Code/playo/models')


    # Generate new graphs
    print("Generating new graphs...")
    new_graphs = generator.generate_graphs(num_graphs=10)

    print("Graph generation completed!")
    print(f"Generated {len(new_graphs)} new graphs")

    # Save models
    generator.save_models()

    # Plot training losses
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(vae_losses)
    plt.title('VAE Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.subplot(1, 2, 2)
    plt.plot(diffusion_losses)
    plt.title('Diffusion Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.tight_layout()
    plt.savefig('training_losses.png')
    plt.show()