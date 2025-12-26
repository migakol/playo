import torch
import numpy as np
import os
from graph_diffusion_levels import GraphLevelGenerator, GraphDataProcessor
import networkx as nx
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform


class LevelPostProcessor:
    """Post-process generated levels and convert to game format"""

    def __init__(self):
        self.node_types = ['center', 'entrance', 'corridor']

    def embedding_to_level(self, embedding, num_nodes=25):
        """Convert embedding back to structured level"""
        # The embedding contains features for a typical node
        # We need to create multiple nodes with variations

        # Extract base features from embedding
        embedding = embedding.squeeze()

        # Generate nodes with spatial distribution
        nodes = []

        # Determine number of each node type based on typical level structure
        num_centers = max(1, num_nodes // 8)
        num_entrances = max(4, num_nodes // 2)
        num_corridors = num_nodes - num_centers - num_entrances

        node_id = 0

        # Generate center nodes (main hubs)
        centers = []
        for i in range(num_centers):
            pos = [
                np.random.uniform(-8, 8),
                -1.0,
                np.random.uniform(-8, 8)
            ]
            centers.append(pos)
            nodes.append({
                'id': node_id,
                'type': 'center',
                'pos': pos,
                'connections': []
            })
            node_id += 1

        # Generate entrance nodes around centers
        entrances = []
        for i in range(num_entrances):
            # Choose a center to be near
            center_idx = i % len(centers)
            center_pos = centers[center_idx]

            # Create entrance near center with some randomness
            angle = np.random.uniform(0, 2 * np.pi)
            distance = np.random.uniform(2, 4)

            pos = [
                center_pos[0] + distance * np.cos(angle),
                -1.0,
                center_pos[2] + distance * np.sin(angle)
            ]
            entrances.append(pos)
            nodes.append({
                'id': node_id,
                'type': 'entrance',
                'pos': pos,
                'connections': []
            })
            node_id += 1

        # Generate corridor nodes (connectors)
        for i in range(num_corridors):
            pos = [
                np.random.uniform(-6, 6),
                -1.0,
                np.random.uniform(-6, 6)
            ]
            nodes.append({
                'id': node_id,
                'type': 'corridor',
                'pos': pos,
                'connections': []
            })
            node_id += 1

        # Generate connections based on proximity and node types
        self._generate_connections(nodes)

        return nodes

    def _generate_connections(self, nodes):
        """Generate realistic connections between nodes"""
        # Calculate distance matrix
        positions = np.array([node['pos'] for node in nodes])
        dist_matrix = squareform(pdist(positions))

        # Connect each center to nearby entrances
        centers = [node for node in nodes if node['type'] == 'center']
        entrances = [node for node in nodes if node['type'] == 'entrance']
        corridors = [node for node in nodes if node['type'] == 'corridor']

        # Centers connect to nearby entrances
        for center in centers:
            center_pos = np.array(center['pos'])
            distances_to_entrances = []

            for entrance in entrances:
                entrance_pos = np.array(entrance['pos'])
                dist = np.linalg.norm(center_pos - entrance_pos)
                distances_to_entrances.append((dist, entrance['id']))

            # Connect to 3-5 nearest entrances
            distances_to_entrances.sort()
            num_connections = min(5, max(3, len(entrances)))

            for dist, entrance_id in distances_to_entrances[:num_connections]:
                if dist < 6.0:  # Max connection distance
                    center['connections'].append(entrance_id)
                    # Find entrance and add reverse connection
                    for entrance in entrances:
                        if entrance['id'] == entrance_id:
                            entrance['connections'].append(center['id'])
                            break

        # Connect some entrances to corridors
        for entrance in entrances:
            entrance_pos = np.array(entrance['pos'])

            for corridor in corridors:
                corridor_pos = np.array(corridor['pos'])
                dist = np.linalg.norm(entrance_pos - corridor_pos)

                if dist < 3.0 and len(entrance['connections']) < 3:  # Limit connections
                    entrance['connections'].append(corridor['id'])
                    corridor['connections'].append(entrance['id'])

        # Connect corridors to each other to ensure connectivity
        for i, corridor1 in enumerate(corridors):
            corridor1_pos = np.array(corridor1['pos'])

            for j, corridor2 in enumerate(corridors[i + 1:], i + 1):
                corridor2_pos = np.array(corridor2['pos'])
                dist = np.linalg.norm(corridor1_pos - corridor2_pos)

                if dist < 2.5 and len(corridor1['connections']) < 2 and len(corridor2['connections']) < 2:
                    corridor1['connections'].append(corridor2['id'])
                    corridor2['connections'].append(corridor1['id'])

        # Remove duplicates and ensure no self-connections
        for node in nodes:
            node['connections'] = list(set([conn for conn in node['connections'] if conn != node['id']]))

    def save_level_to_file(self, nodes, filename):
        """Save level in the original format"""
        import csv
        import io

        lines = ["---", "node_type,pos,connections"]

        for node in nodes:
            # Use CSV writer to properly quote the fields
            output = io.StringIO()
            writer = csv.writer(output)
            writer.writerow([node["id"], node["type"], str(node["pos"]), str(node["connections"])])
            line = output.getvalue().strip()
            lines.append(line)

        lines.append("---")

        with open(filename, 'w') as f:
            f.write('\n'.join(lines))


class LevelVisualizer:
    """Visualize generated levels"""

    def __init__(self):
        self.colors = {'center': 'red', 'entrance': 'blue', 'corridor': 'green'}

    def visualize_level(self, nodes, title="Generated Level", save_path=None):
        """Create a 2D visualization of the level"""
        fig, ax = plt.subplots(figsize=(12, 10))

        # Plot nodes
        for node in nodes:
            x, y, z = node['pos']
            color = self.colors[node['type']]
            ax.scatter(x, z, c=color, s=100, alpha=0.7, label=node['type'])
            ax.annotate(f"{node['id']}", (x, z), xytext=(5, 5),
                        textcoords='offset points', fontsize=8)

        # Plot connections
        for node in nodes:
            x1, y1, z1 = node['pos']
            for conn_id in node['connections']:
                # Find connected node
                for other_node in nodes:
                    if other_node['id'] == conn_id:
                        x2, y2, z2 = other_node['pos']
                        ax.plot([x1, x2], [z1, z2], 'k-', alpha=0.5, linewidth=1)
                        break

        ax.set_xlabel('X Position')
        ax.set_ylabel('Z Position')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

        # Remove duplicate labels
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()


def generate_new_levels(model_path_prefix="models/graph_diffusion", num_levels=5, output_dir="generated_levels"):
    """Generate new levels using trained model"""

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Initialize generator and load models
    generator = GraphLevelGenerator()

    try:
        generator.load_models(model_path_prefix)
        print("Models loaded successfully!")
    except Exception as e:
        print(f"Error loading models: {e}")
        print("Make sure you've trained the models first by running the main script.")
        return

    # Initialize post-processor and visualizer
    post_processor = LevelPostProcessor()
    visualizer = LevelVisualizer()

    print(f"Generating {num_levels} new levels...")

    # Generate levels
    generated_embeddings = generator.generate_graphs(num_graphs=num_levels)

    for i, embedding in enumerate(generated_embeddings):
        print(f"Processing level {i + 1}/{num_levels}")

        # Convert embedding to level structure
        nodes = post_processor.embedding_to_level(embedding)

        # Save to file
        filename = os.path.join(output_dir, f"generated_level_{i:03d}.txt")
        post_processor.save_level_to_file(nodes, filename)

        # Visualize
        visualizer.visualize_level(
            nodes,
            title=f"Generated Level {i + 1}",
            save_path=os.path.join(output_dir, f"level_{i:03d}_visualization.png")
        )

        print(f"Saved: {filename}")

    print(f"All levels generated and saved to {output_dir}/")


def analyze_level_statistics(data_dir):
    """Analyze statistics of existing levels"""
    import csv
    import io

    processor = GraphDataProcessor()

    stats = {
        'num_nodes': [],
        'num_centers': [],
        'num_entrances': [],
        'num_corridors': [],
        'avg_connections': [],
        'max_connections': []
    }

    for filename in os.listdir(data_dir):
        if filename.endswith('.txt'):
            filepath = os.path.join(data_dir, filename)

            try:
                with open(filepath, 'r') as f:
                    lines = f.readlines()

                # Skip header lines
                data_lines = [line.strip() for line in lines if line.strip() and not line.strip().startswith('---')]

                node_types = {'center': 0, 'entrance': 0, 'corridor': 0}
                connections_count = []
                valid_lines = 0

                for line in data_lines:
                    # Skip header line if present
                    if 'node_type,pos,connections' in line:
                        continue

                    # Use CSV parsing to handle quoted fields properly
                    csv_reader = csv.reader(io.StringIO(line))
                    try:
                        parts = next(csv_reader)
                        if len(parts) < 4:
                            continue

                        node_type = parts[1]
                        connections = eval(parts[3])

                        if node_type in node_types:
                            node_types[node_type] += 1

                        connections_count.append(len(connections))
                        valid_lines += 1

                    except (ValueError, IndexError, SyntaxError) as e:
                        print(f"Error parsing line in {filename}: {line.strip()} - {e}")
                        continue

                if valid_lines > 0:
                    stats['num_nodes'].append(valid_lines)
                    stats['num_centers'].append(node_types['center'])
                    stats['num_entrances'].append(node_types['entrance'])
                    stats['num_corridors'].append(node_types['corridor'])
                    stats['avg_connections'].append(np.mean(connections_count) if connections_count else 0)
                    stats['max_connections'].append(max(connections_count) if connections_count else 0)

            except Exception as e:
                print(f"Error processing {filename}: {e}")

    # Print statistics
    print("Level Statistics:")
    print(f"Number of levels analyzed: {len(stats['num_nodes'])}")
    for key, values in stats.items():
        if values:
            print(f"{key}: mean={np.mean(values):.2f}, std={np.std(values):.2f}, min={min(values)}, max={max(values)}")


# Example usage and testing
if __name__ == "__main__":
    # Analyze existing data (if available)
    if os.path.exists("sample_data"):
        print("Analyzing existing level statistics...")
        analyze_level_statistics("sample_data")
        print()

    # Generate new levels
    generate_new_levels(
        model_path_prefix="models/graph_diffusion",
        num_levels=5,
        output_dir="generated_levels"
    )

    print("\nGeneration complete! Check the 'generated_levels' directory for:")
    print("- .txt files with level data in your format")
    print("- .png files with visualizations")

    # Example of loading and verifying a generated level
    if os.path.exists("generated_levels"):
        print("\nVerifying generated level format...")

        # Load first generated level
        processor = GraphDataProcessor()
        try:
            node_features, edge_index = processor.parse_graph_file("generated_levels/generated_level_000.txt")
            print(f"Successfully parsed generated level:")
            print(f"- Nodes: {node_features.shape[0]}")
            print(f"- Edges: {edge_index.shape[1]}")
            print(f"- Node feature dimensions: {node_features.shape[1]}")
        except Exception as e:
            print(f"Error verifying level: {e}")