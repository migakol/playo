import os
import csv
import io
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle, Polygon
import matplotlib.patches as patches
import networkx as nx


class LevelLoader:
    """Load and parse game level files"""

    def __init__(self):
        self.node_types = {'center': 0, 'entrance': 1, 'corridor': 2}

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
    """Visualize game levels with a specific custom style"""

    def __init__(self):
        # Specific color mapping based on requirements
        self.colors = {
            'center': 'blue',
            'entrance': 'green',
            'corridor': 'orange'
        }
        self.marker_sizes = {
            'center': 150,
            'entrance': 100,
            'corridor': 120
        }

    def create_2d_visualization(self, nodes, title="Game Level Visualization", save_path=None, figsize=(14, 8)):
        """Create a 2D top-down view with specific custom markers and styling"""
        fig, ax = plt.subplots(figsize=figsize)

        positions = {}
        node_data = {}

        # Store positions and data
        for node in nodes:
            x, _, z = node['pos']
            positions[node['id']] = (x, z)
            node_data[node['id']] = node

        # Draw connections first in red
        for node in nodes:
            x1, z1 = positions[node['id']]
            for conn_id in node['connections']:
                if conn_id in positions:
                    x2, z2 = positions[conn_id]
                    ax.plot([x1, x2], [z1, z2], color='red', linewidth=2, zorder=1)

        # Plot room borders (grey rectangles around centers)
        # Assuming a default room size for visual representation
        for node in nodes:
            if node['type'] == 'center':
                x, z = positions[node['id']]
                # Estimate room boundaries based on associated entrances
                entrance_coords = [positions[cid] for cid in node['connections'] if
                                   node_data[cid]['type'] == 'entrance']
                if entrance_coords:
                    min_x = min(c[0] for c in entrance_coords) - 0.5
                    max_x = max(c[0] for c in entrance_coords) + 0.5
                    min_z = min(c[1] for c in entrance_coords) - 0.5
                    max_z = max(c[1] for c in entrance_coords) + 0.5
                    rect = patches.Rectangle((min_x, min_z), max_x - min_x, max_z - min_z,
                                             linewidth=2, edgecolor='grey', facecolor='none', alpha=0.5, zorder=0)
                    ax.add_patch(rect)

        # Plot nodes with custom markers
        for node in nodes:
            x, z = positions[node['id']]
            node_type = node['type']

            if node_type == 'center':
                # Blue circle with black outline
                circle = patches.Circle((x, z), radius=0.15, facecolor='blue', edgecolor='black', linewidth=1.5,
                                        zorder=3)
                ax.add_patch(circle)
            elif node_type == 'entrance':
                # Green rectangle with black outline
                rect = patches.Rectangle((x - 0.1, z - 0.1), 0.2, 0.2, facecolor='green', edgecolor='black',
                                         linewidth=1.5, zorder=3)
                ax.add_patch(rect)
            elif node_type == 'corridor':
                # Orange triangle with black outline
                triangle = patches.Polygon([[x, z + 0.15], [x - 0.15, z - 0.1], [x + 0.15, z - 0.1]],
                                           facecolor='orange', edgecolor='black', linewidth=1.5, zorder=3)
                ax.add_patch(triangle)

            # Add ID in black text inside a small black-outlined box
            ax.text(x + 0.2, z + 0.2, str(node['id']), fontsize=9, fontweight='bold',
                    bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.2'),
                    zorder=4)

        # Final plot styling
        ax.set_title(title, fontsize=15, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.set_aspect('equal')

        # Set limits with some padding
        if positions:
            x_vals = [p[0] for p in positions.values()]
            z_vals = [p[1] for p in positions.values()]
            ax.set_xlim(min(x_vals) - 2, max(x_vals) + 2)
            ax.set_ylim(min(z_vals) - 2, max(z_vals) + 2)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

# Update rest of file utility functions similarly to use this new styling...