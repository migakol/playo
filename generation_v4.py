import numpy as np
import glob
import ast
import os
import pickle, csv

import torch
import torch.nn as nn
import torch.nn.functional as F
from visualize_levels import visualize_sample_level
from torch.utils.data import Dataset, DataLoader, TensorDataset


def parse_level_to_matrix(file_path, max_rooms=5):
    nodes = {}
    center_ids = []

    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            n_id = int(row[0])
            n_type = row[1]
            pos = ast.literal_eval(row[2])
            conns = ast.literal_eval(row[3])
            nodes[n_id] = {'type': n_type, 'pos': pos, 'conns': conns}
            if n_type == "center":
                center_ids.append(n_id)

    # Matrix: 5 rooms x (5 connectivity + 2 coords (x,z) + 1 size)
    matrix = np.zeros((max_rooms, 8))
    center_ids = sorted(center_ids)[:max_rooms]

    for i, c_id in enumerate(center_ids):
        # 1. Connectivity (Cols 0-4)
        my_entrances = nodes[c_id]['conns']
        for j, other_c_id in enumerate(center_ids):
            if i == j: continue
            other_entrances = nodes[other_c_id]['conns']
            for node_data in nodes.values():
                if node_data['type'] == 'corridor':
                    if any(e in node_data['conns'] for e in my_entrances) and \
                            any(e in node_data['conns'] for e in other_entrances):
                        matrix[i, j] = 1
                        break

        # 2. Coordinates (Cols 5-6)
        matrix[i, 5] = nodes[c_id]['pos'][0]  # X
        matrix[i, 6] = nodes[c_id]['pos'][2]  # Z

        # 3. Room Size (Col 7)
        # Average distance to its entrances
        dists = [np.linalg.norm(np.array(nodes[c_id]['pos']) - np.array(nodes[e_id]['pos']))
                 for e_id in my_entrances]
        matrix[i, 7] = np.mean(dists)

    return matrix

def parse_level_to_matrix1(file_path, max_rooms=5):
    nodes = {}
    center_ids = []

    with open(file_path, 'r') as f:
        # Use csv.reader to handle the complex quoting/comma behavior
        reader = csv.reader(f)
        header = next(reader)  # Skip node_type,pos,connections

        for row in reader:
            # row[0] = id, row[1] = type, row[2] = pos, row[3] = connections
            n_id = int(row[0])
            n_type = row[1]

            # Use ast.literal_eval safely on clean strings
            pos = ast.literal_eval(row[2])
            conns = ast.literal_eval(row[3])

            nodes[n_id] = {'type': n_type, 'pos': pos, 'conns': conns}
            if n_type == "center":
                center_ids.append(n_id)

    # ... remaining logic for connectivity matrix ...
    matrix = np.zeros((max_rooms, max_rooms + 2))
    center_ids = sorted(center_ids)[:max_rooms]
    id_to_idx = {c_id: i for i, c_id in enumerate(center_ids)}

    for i, c_id in enumerate(center_ids):
        # Fill X and Z coords
        matrix[i, 5] = nodes[c_id]['pos'][0]
        matrix[i, 6] = nodes[c_id]['pos'][2]

        my_entrances = nodes[c_id]['conns']
        for j, other_c_id in enumerate(center_ids):
            if i == j: continue

            other_entrances = nodes[other_c_id]['conns']
            for node_data in nodes.values():
                if node_data['type'] == 'corridor':
                    # Check if corridor links any of my entrances to any of theirs
                    has_me = any(e in node_data['conns'] for e in my_entrances)
                    has_them = any(e in node_data['conns'] for e in other_entrances)
                    if has_me and has_them:
                        matrix[i, j] = 1
                        break
    return matrix

def parse_level_to_matrix_old(file_path, max_rooms=5):
    with open(file_path, 'r') as f:
        lines = f.readlines()[1:]  # Skip header

    # Extract nodes
    centers = []
    for line in lines:
        parts = line.strip().split(',', 2)
        node_id = int(parts[0])
        node_type = parts[1]

        if node_type == "center":
            # Parse position and connections
            # Format: id, type, "[x, y, z]", "[c1, c2...]"
            remaining = parts[2].split('","')
            pos = ast.literal_eval(remaining[0].strip('"'))
            # We only need X and Z as Y is constant
            centers.append({
                'id': node_id,
                'pos': [pos[0], pos[2]],
                'conns': ast.literal_eval(remaining[1].strip('"'))
            })

    # Initialize 5x7 Matrix (5 rooms x (5 connectivity + 2 coords))
    matrix = np.zeros((max_rooms, max_rooms + 2))

    # Map old IDs to 0-4 index
    id_map = {c['id']: i for i, c in enumerate(centers)}

    for i, c in enumerate(centers):
        # Fill Coordinates
        matrix[i, 5:7] = c['pos']

        # Fill Connectivity (Look through entrances to find neighboring centers)
        # Note: In your data, centers connect to entrances, which connect to corridors
        # This logic simplifies "Center -> Entrance -> Corridor -> Entrance -> Center"
        # to a direct "Center A -> Center B" link.
        for other_idx, other_c in enumerate(centers):
            if i == other_idx: continue
            # Check if there's a path between these centers in the file
            # (Simplified check for this example)
            matrix[i, other_idx] = 1

    return matrix


class LevelVAE1(nn.Module):
    def __init__(self, num_rooms=5, latent_dim=16):
        super(LevelVAE1, self).__init__()
        self.num_rooms = num_rooms
        input_dim = num_rooms * 7  # 5x7 flattened

        # Encoder
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2_mu = nn.Linear(64, latent_dim)
        self.fc2_logvar = nn.Linear(64, latent_dim)

        # Decoder Shared
        self.dec_fc1 = nn.Linear(latent_dim, 64)

        # Head A: Connectivity (Symmetric Adjacency)
        self.head_conn = nn.Linear(64, num_rooms * num_rooms)

        # Head B: Coordinates (X, Z)
        self.head_coords = nn.Linear(64, num_rooms * 2)

    def encode(self, x):
        h = F.relu(self.fc1(x))
        return self.fc2_mu(h), self.fc2_logvar(h)

    def decode(self, z):
        h = F.relu(self.dec_fc1(z))

        # Get Probabilities for connections
        conn = torch.sigmoid(self.head_conn(h)).view(-1, self.num_rooms, self.num_rooms)
        # Symmetrize: (A + A^T) / 2
        conn = (conn + conn.transpose(1, 2)) / 2

        # Get Coordinates
        coords = self.head_coords(h).view(-1, self.num_rooms, 2)
        return conn, coords

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.num_rooms * 7))
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return self.decode(z), mu, logvar


class LevelVAE(nn.Module):
    def __init__(self, num_rooms=5, latent_dim=16):
        super(LevelVAE, self).__init__()
        self.num_rooms = num_rooms
        input_dim = num_rooms * 8

        self.enc_fc = nn.Linear(input_dim, 128)
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)

        self.dec_fc = nn.Linear(latent_dim, 128)
        self.head_conn = nn.Linear(128, num_rooms * num_rooms)
        self.head_coords = nn.Linear(128, num_rooms * 2)  # X, Z
        self.head_size = nn.Linear(128, num_rooms * 1)  # Size

    def decode(self, z):
        h = F.relu(self.dec_fc(z))
        conn = torch.sigmoid(self.head_conn(h)).view(-1, 5, 5)
        conn = (conn + conn.transpose(1, 2)) / 2  # Force symmetry
        coords = self.head_coords(h).view(-1, 5, 2)
        size = self.head_size(h).view(-1, 5, 1)
        return conn, coords, size

    def forward(self, x):
        h = F.relu(self.enc_fc(x.view(-1, 40)))
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        return self.decode(z), mu, logvar


def vae_loss(conn_pred, coords_pred, conn_target, coords_target, mu, logvar):
    # BCE for connections
    bce = F.binary_cross_entropy(conn_pred, conn_target, reduction='sum')
    # MSE for coordinates
    mse = F.mse_loss(coords_pred, coords_target, reduction='sum')
    # Orthogonality Penalty: (dx^2 * dz^2) for connected rooms
    # This enforces straight lines
    dx = coords_pred[:, :, 0].unsqueeze(2) - coords_pred[:, :, 0].unsqueeze(1)
    dz = coords_pred[:, :, 1].unsqueeze(2) - coords_pred[:, :, 1].unsqueeze(1)
    ortho_penalty = torch.sum(conn_target * (dx ** 2 * dz ** 2))

    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return bce + mse + kld + (ortho_penalty * 0.1)


def vae_loss_with_size(conn_pred, coords_pred, size_pred, conn_gt, coords_gt, size_gt, mu, logvar):
    bce = F.binary_cross_entropy(conn_pred, conn_gt, reduction='sum')
    mse_coords = F.mse_loss(coords_pred, coords_gt, reduction='sum')
    mse_size = F.mse_loss(size_pred, size_gt, reduction='sum')

    # Orthogonality penalty
    dx = coords_pred[:, :, 0].unsqueeze(2) - coords_pred[:, :, 0].unsqueeze(1)
    dz = coords_pred[:, :, 1].unsqueeze(2) - coords_pred[:, :, 1].unsqueeze(1)
    ortho = torch.sum(conn_gt * (dx ** 2 * dz ** 2))

    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return bce + mse_coords + mse_size + kld + (ortho * 0.1)


def generate_level(model, latent_dim=16):
    model.eval()
    with torch.no_grad():
        z = torch.randn(1, latent_dim)
        conn, coords = model.decode(z)

        conn = (conn > 0.5).float().squeeze(0).numpy()
        coords = coords.squeeze(0).numpy()

        # Orthogonal Snapping Logic
        for i in range(len(conn)):
            for j in range(i + 1, len(conn)):
                if conn[i, j] == 1:
                    dx = abs(coords[i, 0] - coords[j, 0])
                    dz = abs(coords[i, 1] - coords[j, 1])

                    if dx > dz:  # Horizontal Corridor
                        coords[j, 1] = coords[i, 1]
                    else:  # Vertical Corridor
                        coords[j, 0] = coords[i, 0]
        return conn, coords


class RoomGraphDataset(Dataset):
    def __init__(self, file_paths):
        self.data = []
        for f in file_paths:
            # Using the parser from the previous step
            matrix = parse_level_to_matrix(f)
            self.data.append(matrix)

        self.data = np.array(self.data, dtype=np.float32)

        # Simple Normalization for coordinates (Cols 5 & 6)
        # Assuming max coordinate range is roughly -15 to 15
        self.coords_mean = self.data[:, :, 5:7].mean()
        self.coords_std = self.data[:, :, 5:7].std()
        self.data[:, :, 5:7] = (self.data[:, :, 5:7] - self.coords_mean) / self.coords_std

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Return connectivity matrix and coordinate matrix separately
        target = torch.from_numpy(self.data[idx])
        conn_target = target[:, :5]  # 5x5
        coords_target = target[:, 5:7]  # 5x2
        return conn_target, coords_target

def main_train(all_data, model_pickle):
    # 1. Prepare Data
    # Convert list of matrices to a single torch tensor (100000, 5, 7)
    data_tensor = torch.tensor(np.array(all_data), dtype=torch.float32)

    # Separate Connectivity (cols 0-4) and Coordinates (cols 5-6) for loss calculation
    # dataset = TensorDataset(data_tensor[:, :, :5], data_tensor[:, :, 5:8])
    dataset = TensorDataset(
        data_tensor[:, :, :5],  # conn_batch
        data_tensor[:, :, 5:7],  # coords_batch
        data_tensor[:, :, 7:8]  # size_batch
    )
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    # 2. Initialize Model
    latent_dim = 16
    model = LevelVAE(num_rooms=5, latent_dim=latent_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 3. Training Loop
    model.train()
    for epoch in range(100):
        epoch_loss = 0
        for conn_batch, coords_batch, size_batch in loader:  # Assuming size is in your loader
            optimizer.zero_grad()

            # Prepare input (Flattened 5x8 = 40)
            input_data = torch.cat([conn_batch, coords_batch, size_batch], dim=2)

            # Forward pass with the new size_pred
            (conn_pred, coords_pred, size_pred), mu, logvar = model(input_data)

            # Compute the updated loss
            loss = vae_loss_with_size(
                conn_pred, coords_pred, size_pred,
                conn_batch, coords_batch, size_batch,
                mu, logvar
            )

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if epoch % 1 == 0:
            print(f"Epoch {epoch}, Loss: {epoch_loss / len(loader):.4f}")

    with open(model_pickle, 'wb') as f:
        pickle.dump(model, f)

def prepare_training_data(samples_dir, matrix_pickle_path):
    # Collect all .txt files from the directory and prepare the data
    all_data = []
    for k, f in enumerate(glob.glob(samples_dir + "*.txt")):
        if k % 1000 == 0:
            print('Processing file ', k)
        all_data.append(parse_level_to_matrix(f))

    with open(matrix_pickle_path, 'wb') as f:
        pickle.dump(all_data, f)


def compute_statistics(all_data):
    # Assuming all_data is already populated as a list of 5x7 matrices
    # all_data = [parse_level_to_matrix(f) for f in ...]

    # 1. Convert the list into a single 3D NumPy array: (N, 5, 7)
    # where N is the number of files (e.g., 100,000)
    data_stack = np.array(all_data)

    # 2. Extract the center coordinates (columns 5 and 6)
    # Shape will be (N, 5, 2)
    # coords = data_stack[:, :, 5:7]
    #
    # # 3. Compute global mean and standard deviation
    # # This treats all X and Z values as one distribution
    # global_mean = np.mean(coords)
    # global_std = np.std(coords)
    #
    # # 4. Alternatively, compute mean and std per axis (X and Z separately)
    # # This is often better for VAE normalization
    # per_axis_mean = np.mean(coords, axis=(0, 1))  # Result: [mean_x, mean_z]
    # per_axis_std = np.std(coords, axis=(0, 1))  # Result: [std_x, std_z]

    data_stack = np.array(all_data)  # (N, 5, 8)
    coords_raw = data_stack[:, :, 5:7]
    size_raw = data_stack[:, :, 7:8]

    stats = {
        'coords_mean': coords_raw.mean(axis=(0, 1)),
        'coords_std': coords_raw.std(axis=(0, 1)),
        'size_mean': size_raw.mean(),
        'size_std': size_raw.std()
    }

    print(stats)
    return stats

    # print(f"Global Stats - Mean: {global_mean:.4f}, Std: {global_std:.4f}")
    # print(f"Per-Axis Mean (X, Z): {per_axis_mean}")
    # print(f"Per-Axis Std (X, Z): {per_axis_std}")


def generate_level_v2(model, stats, latent_dim=16):
    model.eval()
    with torch.no_grad():
        z = torch.randn(1, latent_dim)
        conn, coords, size = model.decode(z)

        cluster_center = np.mean(coords.squeeze().numpy(), axis=0)  # [mean_x, mean_z]
        # Shift all coordinates so the cluster center is 0,0
        coords = coords.numpy() - cluster_center

        # Denormalize
        conn = (conn > 0.5).float().squeeze().numpy()
        # coords = (coords.squeeze().numpy() * stats['coords_std'] / 4) + stats['coords_mean']
        coords = (coords.squeeze() * stats['coords_std'] / 2) # + stats['coords_mean']
        room_sizes = (size.squeeze().numpy() * stats['size_std']) + stats['size_mean']

        # Apply Snap (Stage 1)
        for i in range(5):
            for j in range(i + 1, 5):
                if conn[i, j] == 1:
                    if abs(coords[i, 0] - coords[j, 0]) > abs(coords[i, 1] - coords[j, 1]):
                        coords[j, 1] = coords[i, 1]
                    else:
                        coords[j, 0] = coords[i, 0]

        return conn, coords, room_sizes

def generate_new_level(model, mean, std, latent_dim=16):
    model.eval()
    with torch.no_grad():
        # 1. Sample a random point in the 'style' space
        z = torch.randn(1, latent_dim)

        # 2. Decode into raw matrix data
        conn_pred, coords_pred = model.decode(z)

        # 3. Process connectivity (binary)
        # We use a 0.5 threshold to decide if a connection exists
        conn = (conn_pred.squeeze(0) > 0.5).float().cpu().numpy()

        # 4. Process coordinates and denormalize
        coords = coords_pred.squeeze(0).cpu().numpy()
        coords = (coords * std) + mean

        # 5. Apply Orthogonal "Snap"
        # This ensures corridors are perfectly straight based on the predicted connections
        for i in range(5):
            for j in range(i + 1, 5):
                if conn[i, j] == 1:
                    dx = abs(coords[i, 0] - coords[j, 0])
                    dz = abs(coords[i, 1] - coords[j, 1])

                    if dx > dz:
                        coords[j, 1] = coords[i, 1]  # Make perfectly horizontal
                    else:
                        coords[j, 0] = coords[i, 0]  # Make perfectly vertical

        return conn, coords


def export_v2(conn, coords, sizes, filename="generated_level.txt"):
    """
    conn: (5, 5) adjacency matrix (binary)
    coords: (5, 2) x and z positions (denormalized)
    sizes: (5,) room size values (denormalized)
    """
    nodes = []

    # 1. Create Centers (Nodes 0-4)
    # Each center links to 4 specific entrance IDs
    for i in range(5):
        entrance_ids = [5 + (i * 4) + j for j in range(4)]
        nodes.append({
            "id": i,
            "type": "center",
            "pos": [float(coords[i, 0]), -1.0, float(coords[i, 1])],
            "conns": entrance_ids
        })

    # 2. Create Entrances (Nodes 5-24)
    # Mapping directions: 0:+Z, 1:-Z, 2:+X, 3:-X
    # We also track which entrances are used by corridors
    entrance_to_corridor = {}  # maps entrance_id -> corridor_id

    # Logic to identify which entrances connect to corridors
    current_corridor_id = 25
    corridor_data = []

    for i in range(5):
        for j in range(i + 1, 5):
            if conn[i, j] == 1:
                # Determine relative direction to find the correct entrance
                dx = coords[j, 0] - coords[i, 0]
                dz = coords[j, 1] - coords[i, 1]

                # Pick entrance for Room i based on direction to Room j
                if abs(dx) > abs(dz):
                    ent_i = 5 + (i * 4) + (2 if dx > 0 else 3)
                    ent_j = 5 + (j * 4) + (3 if dx > 0 else 2)
                else:
                    ent_i = 5 + (i * 4) + (0 if dz > 0 else 1)
                    ent_j = 5 + (j * 4) + (1 if dz > 0 else 0)

                entrance_to_corridor[ent_i] = current_corridor_id
                entrance_to_corridor[ent_j] = current_corridor_id

                # Corridor position is the midpoint between centers
                mid_pos = [(coords[i, 0] + coords[j, 0]) / 2, -1.0, (coords[i, 1] + coords[j, 1]) / 2]
                corridor_data.append({
                    "id": current_corridor_id,
                    "type": "corridor",
                    "pos": mid_pos,
                    "conns": [ent_i, ent_j]
                })
                current_corridor_id += 1

    # Generate Entrance Nodes with their linked IDs
    offsets = [[0, 1], [0, -1], [1, 0], [-1, 0]]  # +Z, -Z, +X, -X
    for i in range(5):
        s = sizes[i]
        for j in range(4):
            ent_id = 5 + (i * 4) + j
            x = coords[i, 0] + (offsets[j][0] * s)
            z = coords[i, 1] + (offsets[j][1] * s)

            linked_ids = [i]  # Always linked to its center
            if ent_id in entrance_to_corridor:
                linked_ids.append(entrance_to_corridor[ent_id])

            nodes.append({
                "id": ent_id,
                "type": "entrance",
                "pos": [float(x), -1.0, float(z)],
                "conns": linked_ids
            })

    # Add the Corridor nodes to the final list
    nodes.extend(corridor_data)

    # 3. Write to CSV/Text file
    with open(filename, mode='w', newline='') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["node_type", "pos", "connections"])  # Header matches your sample

        # Sort by ID to maintain original structure
        nodes.sort(key=lambda x: x["id"])

        for n in nodes:
            # Format: ID, type, "[x, y, z]", "[c1, c2]"
            # Using f-strings to match the specific string formatting in your file
            pos_str = f"[{n['pos'][0]}, {n['pos'][1]}, {n['pos'][2]}]"
            conns_str = f"{n['conns']}"
            writer.writerow([n["id"], n["type"], pos_str, conns_str])

    print(f"Level successfully exported to {filename}")

def main_generation(model_pickle_path, stats):
    with open(model_pickle_path, 'rb') as f:
        model = pickle.load(f)

    # new_conn, new_coords = generate_new_level(model, data_mean, data_std, latent_dim=16)
    conn, coords, room_sizes = generate_level_v2(model, stats, latent_dim=16)
    print(conn)
    print(coords)
    print(room_sizes)

    export_v2(conn, coords, room_sizes, '/Users/michaelkolomenkin/Data/playo/output/test_level.txt')

if __name__ == "__main__":
    # main()
    samples_dir = '/Users/michaelkolomenkin/Data/playo/files_for_yaron/'
    matrix_pickle_path = '/Users/michaelkolomenkin/Data/playo/matrix_pickle.pkl'
    model_pickle_path = '/Users/michaelkolomenkin/Data/playo/vae_model.pkl'

    # # Step 1
    # prepare_training_data(samples_dir, matrix_pickle_path)
    # #
    # # # Step 2
    with open(matrix_pickle_path, 'rb') as f:
        all_data = pickle.load(f)
    stats = compute_statistics(all_data)
    # main_train(all_data, model_pickle_path)
    #
    # # Step 3
    main_generation(model_pickle_path, stats)

    # visualize
    sample_file = "/Users/michaelkolomenkin/Data/playo/output/test_level.txt"  # Change this to your file
    nodes = visualize_sample_level(sample_file, "/Users/michaelkolomenkin/Data/playo/output")


# Example usage for 100k files
# all_data = [parse_level_to_matrix(f) for f in glob.glob("data/*.txt")]
# training_data = np.array(all_data)