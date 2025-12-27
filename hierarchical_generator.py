import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
import random
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import csv
import io
import os
from enum import Enum


class WallSide(Enum):
    NORTH = "north"
    SOUTH = "south"
    EAST = "east"
    WEST = "west"


@dataclass
class RoomEntrance:
    """Represents an entrance on a room wall"""
    position: Tuple[float, float, float]  # 3D position
    wall_side: WallSide
    room_id: int
    entrance_id: int
    connected_to: Optional[int] = None  # ID of connected entrance


@dataclass
class Room:
    """Represents a rectangular room with center and entrances"""
    room_id: int
    center: Tuple[float, float, float]  # 3D position of room center
    size: float  # Room is square, this is the side length
    floor: int  # Which floor the room is on
    entrances: List[RoomEntrance]

    def get_corner_positions(self):
        """Get the four corner positions of the room"""
        cx, cy, cz = self.center
        half_size = self.size / 2
        return [
            (cx - half_size, cy, cz - half_size),  # SW corner
            (cx + half_size, cy, cz - half_size),  # SE corner
            (cx + half_size, cy, cz + half_size),  # NE corner
            (cx - half_size, cy, cz + half_size),  # NW corner
        ]


class LevelLayout:
    """Represents the overall layout of rooms in a level"""

    def __init__(self, room_size=10.0, floor_height=5.0):
        self.rooms: List[Room] = []
        self.connections: List[Tuple[int, int]] = []  # Pairs of connected entrance IDs
        self.room_size = room_size
        self.floor_height = floor_height
        self.next_room_id = 0
        self.next_entrance_id = 0

    def add_room(self, center: Tuple[float, float, float], floor: int) -> Room:
        """Add a new room to the layout"""
        room = Room(
            room_id=self.next_room_id,
            center=center,
            size=self.room_size,
            floor=floor,
            entrances=[]
        )

        # Create entrances in the middle of each wall
        cx, cy, cz = center
        half_size = self.room_size / 2

        # North entrance (positive Z)
        north_entrance = RoomEntrance(
            position=(cx, cy, cz + half_size),
            wall_side=WallSide.NORTH,
            room_id=self.next_room_id,
            entrance_id=self.next_entrance_id
        )
        room.entrances.append(north_entrance)
        self.next_entrance_id += 1

        # South entrance (negative Z)
        south_entrance = RoomEntrance(
            position=(cx, cy, cz - half_size),
            wall_side=WallSide.SOUTH,
            room_id=self.next_room_id,
            entrance_id=self.next_entrance_id
        )
        room.entrances.append(south_entrance)
        self.next_entrance_id += 1

        # East entrance (positive X)
        east_entrance = RoomEntrance(
            position=(cx + half_size, cy, cz),
            wall_side=WallSide.EAST,
            room_id=self.next_room_id,
            entrance_id=self.next_entrance_id
        )
        room.entrances.append(east_entrance)
        self.next_entrance_id += 1

        # West entrance (negative X)
        west_entrance = RoomEntrance(
            position=(cx - half_size, cy, cz),
            wall_side=WallSide.WEST,
            room_id=self.next_room_id,
            entrance_id=self.next_entrance_id
        )
        room.entrances.append(west_entrance)
        self.next_entrance_id += 1

        self.rooms.append(room)
        self.next_room_id += 1
        return room

    def connect_rooms_adjacent(self, room1_id: int, room2_id: int, wall_side: WallSide):
        """Connect two adjacent rooms through their facing entrances"""
        room1 = self.rooms[room1_id]
        room2 = self.rooms[room2_id]

        # Find the appropriate entrances to connect
        entrance1 = None
        entrance2 = None

        for entrance in room1.entrances:
            if entrance.wall_side == wall_side:
                entrance1 = entrance
                break

        # Find opposite wall entrance in room2
        opposite_walls = {
            WallSide.NORTH: WallSide.SOUTH,
            WallSide.SOUTH: WallSide.NORTH,
            WallSide.EAST: WallSide.WEST,
            WallSide.WEST: WallSide.EAST
        }

        for entrance in room2.entrances:
            if entrance.wall_side == opposite_walls[wall_side]:
                entrance2 = entrance
                break

        if entrance1 and entrance2:
            entrance1.connected_to = entrance2.entrance_id
            entrance2.connected_to = entrance1.entrance_id
            self.connections.append((entrance1.entrance_id, entrance2.entrance_id))

    def connect_rooms_corridor(self, entrance1_id: int, entrance2_id: int,
                               corridor_nodes: List[Tuple[float, float, float]]):
        """Connect two rooms through a corridor with intermediate nodes"""
        # Find the entrances
        entrance1 = None
        entrance2 = None

        for room in self.rooms:
            for entrance in room.entrances:
                if entrance.entrance_id == entrance1_id:
                    entrance1 = entrance
                if entrance.entrance_id == entrance2_id:
                    entrance2 = entrance

        if entrance1 and entrance2:
            entrance1.connected_to = entrance2.entrance_id
            entrance2.connected_to = entrance1.entrance_id
            self.connections.append((entrance1_id, entrance2_id))

            # Store corridor nodes for later use in graph generation
            if not hasattr(self, 'corridor_paths'):
                self.corridor_paths = {}
            self.corridor_paths[(entrance1_id, entrance2_id)] = corridor_nodes


class RoomBasedLevelGenerator:
    """Generate levels using room-based approach"""

    def __init__(self, room_size=10.0, floor_height=5.0):
        self.room_size = room_size
        self.floor_height = floor_height

    def generate_simple_layout(self, num_rooms=4, floors=1) -> LevelLayout:
        """Generate a simple grid-based layout"""
        layout = LevelLayout(self.room_size, self.floor_height)

        # Calculate grid dimensions
        rooms_per_floor = num_rooms // floors
        grid_size = int(np.ceil(np.sqrt(rooms_per_floor)))

        room_id = 0
        for floor in range(floors):
            floor_y = floor * self.floor_height

            for i in range(grid_size):
                for j in range(grid_size):
                    if room_id >= num_rooms:
                        break

                    # Position rooms in a grid
                    x = i * (self.room_size + 2)  # 2 unit spacing
                    z = j * (self.room_size + 2)

                    layout.add_room((x, floor_y, z), floor)
                    room_id += 1

                if room_id >= num_rooms:
                    break

        # Connect adjacent rooms
        self._connect_adjacent_rooms(layout, grid_size, rooms_per_floor)

        return layout

    def generate_complex_layout(self, num_rooms=8, floors=2) -> LevelLayout:
        """Generate a more complex layout with varied room positions"""
        layout = LevelLayout(self.room_size, self.floor_height)

        rooms_per_floor = num_rooms // floors

        for floor in range(floors):
            floor_y = floor * self.floor_height

            # Generate rooms with some randomness but maintaining structure
            for i in range(rooms_per_floor):
                # Base position with some variation
                angle = (i / rooms_per_floor) * 2 * np.pi
                radius = 15 + random.uniform(-3, 3)

                x = radius * np.cos(angle) + random.uniform(-2, 2)
                z = radius * np.sin(angle) + random.uniform(-2, 2)

                layout.add_room((x, floor_y, z), floor)

        # Connect rooms with more interesting patterns
        self._connect_rooms_intelligently(layout)

        return layout

    def _connect_adjacent_rooms(self, layout: LevelLayout, grid_size: int, rooms_per_floor: int):
        """Connect rooms that are adjacent in the grid"""
        for floor in range(len(layout.rooms) // rooms_per_floor):
            floor_offset = floor * rooms_per_floor

            for i in range(grid_size):
                for j in range(grid_size):
                    room_idx = i * grid_size + j
                    if room_idx >= rooms_per_floor:
                        continue

                    room_id = floor_offset + room_idx
                    if room_id >= len(layout.rooms):
                        continue

                    # Connect to right neighbor
                    if j < grid_size - 1:
                        right_idx = i * grid_size + (j + 1)
                        if right_idx < rooms_per_floor:
                            right_room_id = floor_offset + right_idx
                            if right_room_id < len(layout.rooms):
                                layout.connect_rooms_adjacent(room_id, right_room_id, WallSide.EAST)

                    # Connect to bottom neighbor
                    if i < grid_size - 1:
                        bottom_idx = (i + 1) * grid_size + j
                        if bottom_idx < rooms_per_floor:
                            bottom_room_id = floor_offset + bottom_idx
                            if bottom_room_id < len(layout.rooms):
                                layout.connect_rooms_adjacent(room_id, bottom_room_id, WallSide.SOUTH)

    def _connect_rooms_intelligently(self, layout: LevelLayout):
        """Connect rooms based on distance and create interesting paths"""
        # Connect each room to 1-3 nearest rooms on the same floor
        for room in layout.rooms:
            same_floor_rooms = [r for r in layout.rooms if r.floor == room.floor and r.room_id != room.room_id]

            # Sort by distance
            distances = []
            for other_room in same_floor_rooms:
                dist = np.sqrt(
                    (room.center[0] - other_room.center[0]) ** 2 +
                    (room.center[2] - other_room.center[2]) ** 2
                )
                distances.append((dist, other_room))

            distances.sort()

            # Connect to 1-2 nearest rooms
            num_connections = random.randint(1, min(2, len(distances)))
            for i in range(num_connections):
                other_room = distances[i][1]

                # Find best entrances to connect (shortest path)
                best_dist = float('inf')
                best_entrance_pair = None

                for ent1 in room.entrances:
                    for ent2 in other_room.entrances:
                        if ent1.connected_to is None and ent2.connected_to is None:
                            dist = np.sqrt(
                                sum((a - b) ** 2 for a, b in zip(ent1.position, ent2.position))
                            )
                            if dist < best_dist:
                                best_dist = dist
                                best_entrance_pair = (ent1, ent2)

                if best_entrance_pair:
                    ent1, ent2 = best_entrance_pair
                    # Create corridor path between entrances
                    corridor_nodes = self._create_corridor_path(ent1.position, ent2.position)
                    layout.connect_rooms_corridor(ent1.entrance_id, ent2.entrance_id, corridor_nodes)

        # Connect between floors if multiple floors exist
        floors = set(room.floor for room in layout.rooms)
        if len(floors) > 1:
            self._connect_floors(layout)

    def _create_corridor_path(self, pos1: Tuple[float, float, float], pos2: Tuple[float, float, float]) -> List[
        Tuple[float, float, float]]:
        """Create a straight corridor path between two positions"""
        # For now, create a simple straight line with a few intermediate points
        x1, y1, z1 = pos1
        x2, y2, z2 = pos2

        # If different Y levels, create a vertical connection
        if abs(y1 - y2) > 0.1:
            mid_y = (y1 + y2) / 2
            return [
                (x1, mid_y, z1),
                (x2, mid_y, z2)
            ]
        else:
            # Same level, create direct path
            num_points = max(2, int(np.sqrt((x2 - x1) ** 2 + (z2 - z1) ** 2) / 3))
            points = []
            for i in range(1, num_points):
                t = i / num_points
                x = x1 + t * (x2 - x1)
                z = z1 + t * (z2 - z1)
                points.append((x, y1, z))
            return points

    def _connect_floors(self, layout: LevelLayout):
        """Connect different floors with vertical connections"""
        floors = sorted(set(room.floor for room in layout.rooms))

        for i in range(len(floors) - 1):
            current_floor = floors[i]
            next_floor = floors[i + 1]

            # Find rooms on each floor
            current_rooms = [r for r in layout.rooms if r.floor == current_floor]
            next_rooms = [r for r in layout.rooms if r.floor == next_floor]

            # Connect closest rooms between floors
            if current_rooms and next_rooms:
                best_dist = float('inf')
                best_pair = None

                for room1 in current_rooms:
                    for room2 in next_rooms:
                        dist = np.sqrt(
                            (room1.center[0] - room2.center[0]) ** 2 +
                            (room1.center[2] - room2.center[2]) ** 2
                        )
                        if dist < best_dist:
                            best_dist = dist
                            best_pair = (room1, room2)

                if best_pair:
                    room1, room2 = best_pair

                    # Find unused entrances
                    unused_ent1 = [e for e in room1.entrances if e.connected_to is None]
                    unused_ent2 = [e for e in room2.entrances if e.connected_to is None]

                    if unused_ent1 and unused_ent2:
                        ent1 = random.choice(unused_ent1)
                        ent2 = random.choice(unused_ent2)

                        # Create vertical corridor
                        corridor_nodes = self._create_corridor_path(ent1.position, ent2.position)
                        layout.connect_rooms_corridor(ent1.entrance_id, ent2.entrance_id, corridor_nodes)


class LayoutToGraphConverter:
    """Convert room layout to graph format"""

    def __init__(self):
        self.node_id_counter = 0

    def convert_layout_to_graph(self, layout: LevelLayout) -> List[Dict]:
        """Convert room layout to your graph node format"""
        nodes = []
        self.node_id_counter = 0

        # Track entrance ID to node ID mapping
        entrance_to_node = {}

        # Add room centers
        for room in layout.rooms:
            center_node = {
                'id': self.node_id_counter,
                'type': 'center',
                'pos': list(room.center),
                'connections': []
            }

            # Connect center to all its entrances
            entrance_node_ids = []
            for entrance in room.entrances:
                self.node_id_counter += 1
                entrance_node = {
                    'id': self.node_id_counter,
                    'type': 'entrance',
                    'pos': list(entrance.position),
                    'connections': [center_node['id']]  # Connect to center
                }

                entrance_to_node[entrance.entrance_id] = self.node_id_counter
                entrance_node_ids.append(self.node_id_counter)
                nodes.append(entrance_node)

            # Connect center to all entrances
            center_node['connections'] = entrance_node_ids
            nodes.insert(-4, center_node)  # Insert center before its entrances
            self.node_id_counter += 1

        # Add corridor connections
        if hasattr(layout, 'corridor_paths'):
            for (ent1_id, ent2_id), corridor_points in layout.corridor_paths.items():
                if corridor_points:
                    # Add corridor nodes
                    corridor_node_ids = []
                    for point in corridor_points:
                        self.node_id_counter += 1
                        corridor_node = {
                            'id': self.node_id_counter,
                            'type': 'corridor',
                            'pos': list(point),
                            'connections': []
                        }
                        corridor_node_ids.append(self.node_id_counter)
                        nodes.append(corridor_node)

                    # Connect entrances to corridor
                    if ent1_id in entrance_to_node and ent2_id in entrance_to_node:
                        entrance1_node_id = entrance_to_node[ent1_id]
                        entrance2_node_id = entrance_to_node[ent2_id]

                        # Find the nodes and update connections
                        for node in nodes:
                            if node['id'] == entrance1_node_id:
                                if corridor_node_ids:
                                    node['connections'].append(corridor_node_ids[0])
                                else:
                                    node['connections'].append(entrance2_node_id)
                            elif node['id'] == entrance2_node_id:
                                if corridor_node_ids:
                                    node['connections'].append(corridor_node_ids[-1])
                                else:
                                    node['connections'].append(entrance1_node_id)

                        # Connect corridor nodes in sequence
                        for i, corridor_id in enumerate(corridor_node_ids):
                            for node in nodes:
                                if node['id'] == corridor_id:
                                    if i == 0:  # First corridor node
                                        node['connections'] = [entrance1_node_id]
                                        if len(corridor_node_ids) > 1:
                                            node['connections'].append(corridor_node_ids[i + 1])
                                    elif i == len(corridor_node_ids) - 1:  # Last corridor node
                                        node['connections'] = [corridor_node_ids[i - 1], entrance2_node_id]
                                    else:  # Middle corridor node
                                        node['connections'] = [corridor_node_ids[i - 1], corridor_node_ids[i + 1]]
                                    break

        # Handle direct room connections (no corridors)
        for entrance1_id, entrance2_id in layout.connections:
            if not hasattr(layout, 'corridor_paths') or (entrance1_id, entrance2_id) not in layout.corridor_paths:
                if entrance1_id in entrance_to_node and entrance2_id in entrance_to_node:
                    node1_id = entrance_to_node[entrance1_id]
                    node2_id = entrance_to_node[entrance2_id]

                    # Add direct connection between entrances
                    for node in nodes:
                        if node['id'] == node1_id:
                            node['connections'].append(node2_id)
                        elif node['id'] == node2_id:
                            node['connections'].append(node1_id)

        return nodes


class LevelVisualizer:
    """Visualize room-based levels"""

    def __init__(self):
        self.colors = {'center': '#FF4444', 'entrance': '#4444FF', 'corridor': '#44FF44'}

    def visualize_layout(self, layout: LevelLayout, title="Room Layout", save_path=None):
        """Visualize the room layout"""
        fig, ax = plt.subplots(figsize=(12, 10))

        # Plot rooms
        for room in layout.rooms:
            cx, cy, cz = room.center

            # Draw room as rectangle
            rect = Rectangle((cx - room.size / 2, cz - room.size / 2),
                             room.size, room.size,
                             fill=False, edgecolor='black', linewidth=2)
            ax.add_patch(rect)

            # Draw center
            ax.scatter(cx, cz, c=self.colors['center'], s=100, zorder=3)
            ax.annotate(f"R{room.room_id}", (cx, cz), xytext=(5, 5),
                        textcoords='offset points', fontsize=10, fontweight='bold')

            # Draw entrances
            for entrance in room.entrances:
                ex, ey, ez = entrance.position
                ax.scatter(ex, ez, c=self.colors['entrance'], s=60, zorder=2)
                ax.annotate(f"E{entrance.entrance_id}", (ex, ez), xytext=(3, 3),
                            textcoords='offset points', fontsize=8)

        # Draw connections
        for entrance1_id, entrance2_id in layout.connections:
            # Find entrance positions
            pos1 = pos2 = None
            for room in layout.rooms:
                for entrance in room.entrances:
                    if entrance.entrance_id == entrance1_id:
                        pos1 = entrance.position
                    elif entrance.entrance_id == entrance2_id:
                        pos2 = entrance.position

            if pos1 and pos2:
                ax.plot([pos1[0], pos2[0]], [pos1[2], pos2[2]], 'g-', alpha=0.7, linewidth=2)

        ax.set_xlabel('X Position')
        ax.set_ylabel('Z Position')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Layout visualization saved to: {save_path}")

        # plt.show()

    def visualize_graph(self, nodes, title="Generated Graph", save_path=None):
        """Visualize the generated graph"""
        fig, ax = plt.subplots(figsize=(12, 10))

        # Plot connections first
        for node in nodes:
            x1, y1, z1 = node['pos']
            for conn_id in node['connections']:
                # Find connected node
                for other_node in nodes:
                    if other_node['id'] == conn_id:
                        x2, y2, z2 = other_node['pos']
                        ax.plot([x1, x2], [z1, z2], 'k-', alpha=0.5, linewidth=1)
                        break

        # Plot nodes
        for node_type in ['center', 'entrance', 'corridor']:
            type_nodes = [node for node in nodes if node['type'] == node_type]
            if type_nodes:
                xs = [node['pos'][0] for node in type_nodes]
                zs = [node['pos'][2] for node in type_nodes]
                sizes = {'center': 200, 'entrance': 100, 'corridor': 60}

                ax.scatter(xs, zs, c=self.colors[node_type],
                           s=sizes[node_type], alpha=0.8, label=node_type)

                # Add node IDs
                for node in type_nodes:
                    ax.annotate(f"{node['id']}", (node['pos'][0], node['pos'][2]),
                                xytext=(3, 3), textcoords='offset points', fontsize=8)

        ax.set_xlabel('X Position')
        ax.set_ylabel('Z Position')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Graph visualization saved to: {save_path}")

        # plt.show()


def save_graph_to_file(nodes, filename):
    """Save graph nodes to file in your format"""
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

    print(f"✅ Graph saved to: {filename}")


# Example usage and testing
if __name__ == "__main__":
    print("🏠 Room-Based Level Generator")
    print("=" * 40)

    # Create generator
    generator = RoomBasedLevelGenerator(room_size=8.0, floor_height=4.0)

    # Generate different types of layouts
    print("🎯 Generating simple grid layout...")
    simple_layout = generator.generate_simple_layout(num_rooms=6, floors=1)

    print("🎯 Generating complex layout...")
    complex_layout = generator.generate_complex_layout(num_rooms=8, floors=2)

    # Convert to graph format
    converter = LayoutToGraphConverter()

    simple_graph = converter.convert_layout_to_graph(simple_layout)
    complex_graph = converter.convert_layout_to_graph(complex_layout)

    print(f"✅ Simple layout: {len(simple_layout.rooms)} rooms, {len(simple_graph)} nodes")
    print(f"✅ Complex layout: {len(complex_layout.rooms)} rooms, {len(complex_graph)} nodes")

    # Save graphs
    os.makedirs("generated_room_levels", exist_ok=True)
    save_graph_to_file(simple_graph, "generated_room_levels/simple_layout.txt")
    save_graph_to_file(complex_graph, "generated_room_levels/complex_layout.txt")

    # Visualize
    visualizer = LevelVisualizer()

    print("\n🎨 Creating visualizations...")
    visualizer.visualize_layout(simple_layout, "Simple Grid Layout",
                                "generated_room_levels/simple_layout_rooms.png")
    visualizer.visualize_graph(simple_graph, "Simple Layout Graph",
                               "generated_room_levels/simple_layout_graph.png")

    visualizer.visualize_layout(complex_layout, "Complex Multi-Floor Layout",
                                "generated_room_levels/complex_layout_rooms.png")
    visualizer.visualize_graph(complex_graph, "Complex Layout Graph",
                               "generated_room_levels/complex_layout_graph.png")

    print("\n🎉 Room-based level generation completed!")
    print("Check 'generated_room_levels/' directory for outputs")

    # Print sample of generated graph
    print(f"\n📝 Sample nodes from simple layout:")
    for node in simple_graph[:5]:
        print(f"   Node {node['id']}: {node['type']} at {node['pos']} -> {node['connections']}")