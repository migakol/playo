import os
import csv
import io
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
import networkx as nx


class LevelLoader:
    """Load and parse game level files"""

    def __init__(self):
        self.node_types = {'center': 0, 'entrance': 1, 'corridor': 2}
        self.colors = {'center': '#FF4444', 'entrance': '#4444FF', 'corridor': '#44FF44'}
        self.sizes = {'center': 200, 'entrance': 100, 'corridor': 80}

    def load_level(self, filepath):
        """Load a level file and return structured data"""
        with open(filepath, 'r') as f:
            lines = f.readlines()

        # Skip header lines (---)
        data_lines = [line.strip() for line in lines if line.strip() and not line.strip().startswith('---')]

        nodes = []

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

                nodes.append({
                    'id': node_id,
                    'type': node_type,
                    'pos': position,
                    'connections': connections
                })

            except (ValueError, IndexError, SyntaxError) as e:
                print(f"Error parsing line: {line.strip()} - {e}")
                continue

        return nodes


class LevelVisualizer:
    """Visualize game levels as PNG files"""

    def __init__(self):
        self.colors = {'center': '#FF4444', 'entrance': '#4444FF', 'corridor': '#44FF44'}
        self.sizes = {'center': 200, 'entrance': 100, 'corridor': 80}

    def create_2d_visualization(self, nodes, title="Game Level", save_path=None, figsize=(12, 10)):
        """Create a 2D top-down view of the level"""
        fig, ax = plt.subplots(figsize=figsize)

        # Extract positions (use X and Z coordinates, ignore Y)
        positions = {}
        node_types = {}

        for node in nodes:
            x, y, z = node['pos']
            positions[node['id']] = (x, z)  # Top-down view (X, Z)
            node_types[node['id']] = node['type']

        # Plot connections first (so they appear behind nodes)
        for node in nodes:
            x1, z1 = positions[node['id']]
            for conn_id in node['connections']:
                if conn_id in positions:  # Make sure connected node exists
                    x2, z2 = positions[conn_id]
                    ax.plot([x1, x2], [z1, z2], 'k-', alpha=0.5, linewidth=1.5, zorder=1)

        # Plot nodes
        plotted_types = set()
        for node in nodes:
            x, z = positions[node['id']]
            node_type = node['type']
            color = self.colors[node_type]
            size = self.sizes[node_type]

            # Add label only for first occurrence of each type
            label = node_type if node_type not in plotted_types else None
            if label:
                plotted_types.add(node_type)

            ax.scatter(x, z, c=color, s=size, alpha=0.8,
                       edgecolors='black', linewidth=1, label=label, zorder=2)

            # Add node ID annotation
            ax.annotate(f"{node['id']}", (x, z), xytext=(3, 3),
                        textcoords='offset points', fontsize=8,
                        fontweight='bold', color='white', zorder=3)

        ax.set_xlabel('X Position', fontsize=12)
        ax.set_ylabel('Z Position', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10, loc='upper right')
        ax.set_aspect('equal', adjustable='box')

        # Add some padding around the level
        if positions:
            x_coords = [pos[0] for pos in positions.values()]
            z_coords = [pos[1] for pos in positions.values()]
            x_range = max(x_coords) - min(x_coords)
            z_range = max(z_coords) - min(z_coords)
            padding = max(x_range, z_range) * 0.1

            ax.set_xlim(min(x_coords) - padding, max(x_coords) + padding)
            ax.set_ylim(min(z_coords) - padding, max(z_coords) + padding)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f" Visualization saved to: {save_path}")

        plt.show()
        return fig, ax

    def create_network_graph(self, nodes, title="Game Level Network", save_path=None, figsize=(12, 10)):
        """Create a network graph visualization using NetworkX"""
        # Create NetworkX graph
        G = nx.Graph()

        # Add nodes
        for node in nodes:
            G.add_node(node['id'],
                       node_type=node['type'],
                       pos=(node['pos'][0], node['pos'][2]))  # Use X, Z coordinates

        # Add edges
        for node in nodes:
            for conn_id in node['connections']:
                if conn_id < len(nodes):  # Make sure connection is valid
                    G.add_edge(node['id'], conn_id)

        # Set up the plot
        fig, ax = plt.subplots(figsize=figsize)

        # Use the actual positions from the level
        pos = {node['id']: (node['pos'][0], node['pos'][2]) for node in nodes}

        # Draw edges
        nx.draw_networkx_edges(G, pos, alpha=0.5, width=1.5, edge_color='gray')

        # Draw nodes by type
        for node_type in ['center', 'entrance', 'corridor']:
            node_list = [node['id'] for node in nodes if node['type'] == node_type]
            if node_list:
                nx.draw_networkx_nodes(G, pos, nodelist=node_list,
                                       node_color=self.colors[node_type],
                                       node_size=self.sizes[node_type],
                                       alpha=0.8, label=node_type)

        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold', font_color='white')

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend()
        ax.set_aspect('equal')
        plt.axis('off')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f" Network visualization saved to: {save_path}")

        plt.show()
        return fig, ax

    def create_3d_visualization(self, nodes, title="Game Level 3D", save_path=None, figsize=(12, 10)):
        """Create a 3D visualization of the level"""
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')

        # Plot connections
        for node in nodes:
            x1, y1, z1 = node['pos']
            for conn_id in node['connections']:
                # Find connected node
                for other_node in nodes:
                    if other_node['id'] == conn_id:
                        x2, y2, z2 = other_node['pos']
                        ax.plot([x1, x2], [y1, y2], [z1, z2], 'k-', alpha=0.5, linewidth=1)
                        break

        # Plot nodes by type
        for node_type in ['center', 'entrance', 'corridor']:
            type_nodes = [node for node in nodes if node['type'] == node_type]
            if type_nodes:
                xs = [node['pos'][0] for node in type_nodes]
                ys = [node['pos'][1] for node in type_nodes]
                zs = [node['pos'][2] for node in type_nodes]

                ax.scatter(xs, ys, zs, c=self.colors[node_type],
                           s=self.sizes[node_type], alpha=0.8, label=node_type)

                # Add node IDs
                for node in type_nodes:
                    ax.text(node['pos'][0], node['pos'][1], node['pos'][2],
                            f"  {node['id']}", fontsize=8)

        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_zlabel('Z Position')
        ax.set_title(title, fontweight='bold')
        ax.legend()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f" 3D visualization saved to: {save_path}")

        plt.show()
        return fig, ax


def visualize_sample_level(filepath, output_dir="visualizations"):
    """Complete function to load and visualize a sample level"""

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load the level
    loader = LevelLoader()
    try:
        nodes = loader.load_level(filepath)
        print(f" Loaded level from: {filepath}")
        print(f"   Total nodes: {len(nodes)}")

        # Count nodes by type
        type_counts = {}
        for node in nodes:
            node_type = node['type']
            type_counts[node_type] = type_counts.get(node_type, 0) + 1

        for node_type, count in type_counts.items():
            print(f"   {node_type}: {count}")

    except Exception as e:
        print(f" Error loading level: {e}")
        return None

    # Create visualizer
    visualizer = LevelVisualizer()

    # Get base filename for output
    base_name = os.path.splitext(os.path.basename(filepath))[0]

    # Create different visualizations
    print("\nCreating visualizations...")

    # 2D Top-down view
    visualizer.create_2d_visualization(
        nodes,
        title=f"Level: {base_name} (Top View)",
        save_path=os.path.join(output_dir, f"{base_name}_2d.png")
    )

    # Network graph view
    visualizer.create_network_graph(
        nodes,
        title=f"Level: {base_name} (Network)",
        save_path=os.path.join(output_dir, f"{base_name}_network.png")
    )

    # 3D view
    visualizer.create_3d_visualization(
        nodes,
        title=f"Level: {base_name} (3D)",
        save_path=os.path.join(output_dir, f"{base_name}_3d.png")
    )

    print(f"\n🎉 All visualizations saved to: {output_dir}/")
    return nodes


def visualize_multiple_samples(data_dir, output_dir="visualizations", max_files=5):
    """Visualize multiple sample files"""

    if not os.path.exists(data_dir):
        print(f" Data directory not found: {data_dir}")
        return

    # Find all .txt files
    txt_files = [f for f in os.listdir(data_dir) if f.endswith('.txt')]
    txt_files.sort()

    if not txt_files:
        print(f" No .txt files found in {data_dir}")
        return

    print(f"📁 Found {len(txt_files)} level files")

    # Process up to max_files
    for i, filename in enumerate(txt_files[:max_files]):
        print(f"\n🔄 Processing {i + 1}/{min(len(txt_files), max_files)}: {filename}")
        filepath = os.path.join(data_dir, filename)

        try:
            visualize_sample_level(filepath, output_dir)
        except Exception as e:
            print(f"❌ Error processing {filename}: {e}")


# Example usage
if __name__ == "__main__":

    # Example 1: Visualize a single level file
    print("Example 1: Visualizing a single level")

    # Replace with your actual file path
    sample_file = "/Users/michaelkolomenkin/Data/playo/files_for_yaron/sample_77356.txt"  # Change this to your file

    if os.path.exists(sample_file):
        nodes = visualize_sample_level(sample_file, "my_visualizations")

        # Print some details about the level
        if nodes:
            print(f"\n Level Details:")
            print(f"   Total nodes: {len(nodes)}")
            print(f"   Node types: {set(node['type'] for node in nodes)}")
            print(f"   Total connections: {sum(len(node['connections']) for node in nodes) // 2}")

            # Show first few nodes
            print(f"\n First 3 nodes:")
            for node in nodes[:3]:
                print(f"   Node {node['id']}: {node['type']} at {node['pos']} -> {node['connections']}")
    else:
        print(f" Sample file not found: {sample_file}")
        print(" You can create sample data by running:")
        print("   from graph_diffusion_levels import create_sample_data")
        print("   create_sample_data()")

    print("\n" + "=" * 50)

    # Example 2: Visualize multiple levels
    print("Example 2: Visualizing multiple levels")

    data_directory = "sample_data"  # Change this to your data directory

    if os.path.exists(data_directory):
        visualize_multiple_samples(data_directory, "batch_visualizations", max_files=3)
    else:
        print(f" Data directory not found: {data_directory}")
        print("💡 Create sample data first!")