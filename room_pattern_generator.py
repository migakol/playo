import numpy as np
import random
from typing import List, Tuple, Dict
from hierarchical_generator import (
    LevelLayout, RoomBasedLevelGenerator, LayoutToGraphConverter,
    LevelVisualizer, save_graph_to_file, WallSide
)
import os


class AdvancedRoomPatterns:
    """Generate levels with specific architectural patterns"""

    def __init__(self, room_size=10.0, floor_height=5.0):
        self.room_size = room_size
        self.floor_height = floor_height
        self.generator = RoomBasedLevelGenerator(room_size, floor_height)

    def generate_linear_corridor(self, num_rooms=6, floors=1) -> LevelLayout:
        """Generate rooms connected in a straight line"""
        layout = LevelLayout(self.room_size, self.floor_height)

        rooms_per_floor = num_rooms // floors

        for floor in range(floors):
            floor_y = floor * self.floor_height

            # Create rooms in a line
            for i in range(rooms_per_floor):
                x = i * (self.room_size + 3)  # 3 unit spacing for corridors
                z = 0
                layout.add_room((x, floor_y, z), floor)

        # Connect adjacent rooms in the line
        for floor in range(floors):
            floor_offset = floor * rooms_per_floor
            for i in range(rooms_per_floor - 1):
                room1_id = floor_offset + i
                room2_id = floor_offset + i + 1
                if room2_id < len(layout.rooms):
                    layout.connect_rooms_adjacent(room1_id, room2_id, WallSide.EAST)

        return layout

    def generate_cross_pattern(self, arm_length=3) -> LevelLayout:
        """Generate rooms in a cross/plus pattern"""
        layout = LevelLayout(self.room_size, self.floor_height)

        # Center room
        layout.add_room((0, 0, 0), 0)
        center_room_id = 0

        # Four arms of the cross
        spacing = self.room_size + 2

        # North arm
        for i in range(1, arm_length + 1):
            layout.add_room((0, 0, i * spacing), 0)
            # Connect to previous room
            if i == 1:
                layout.connect_rooms_adjacent(center_room_id, i, WallSide.NORTH)
            else:
                layout.connect_rooms_adjacent(i - 1, i, WallSide.NORTH)

        # South arm
        for i in range(1, arm_length + 1):
            room_id = arm_length + i
            layout.add_room((0, 0, -i * spacing), 0)
            # Connect to previous room
            if i == 1:
                layout.connect_rooms_adjacent(center_room_id, room_id, WallSide.SOUTH)
            else:
                layout.connect_rooms_adjacent(room_id - 1, room_id, WallSide.SOUTH)

        # East arm
        for i in range(1, arm_length + 1):
            room_id = 2 * arm_length + i
            layout.add_room((i * spacing, 0, 0), 0)
            # Connect to previous room
            if i == 1:
                layout.connect_rooms_adjacent(center_room_id, room_id, WallSide.EAST)
            else:
                layout.connect_rooms_adjacent(room_id - 1, room_id, WallSide.EAST)

        # West arm
        for i in range(1, arm_length + 1):
            room_id = 3 * arm_length + i
            layout.add_room((-i * spacing, 0, 0), 0)
            # Connect to previous room
            if i == 1:
                layout.connect_rooms_adjacent(center_room_id, room_id, WallSide.WEST)
            else:
                layout.connect_rooms_adjacent(room_id - 1, room_id, WallSide.WEST)

        return layout

    def generate_spiral_pattern(self, spiral_size=5) -> LevelLayout:
        """Generate rooms in a spiral pattern"""
        layout = LevelLayout(self.room_size, self.floor_height)

        spacing = self.room_size + 1.5

        # Generate spiral coordinates
        positions = []
        x, z = 0, 0
        dx, dz = 1, 0  # Start moving east

        for i in range(spiral_size * spiral_size):
            positions.append((x * spacing, 0, z * spacing))

            # Check if we need to turn
            next_x, next_z = x + dx, z + dz

            # Turn conditions for spiral
            if (dx == 1 and (
                    next_x >= spiral_size // 2 + 1 or (next_x, next_z) in [(p[0] // spacing, p[2] // spacing) for p in
                                                                           positions])) or \
                    (dz == 1 and (
                            next_z >= spiral_size // 2 + 1 or (next_x, next_z) in [(p[0] // spacing, p[2] // spacing)
                                                                                   for p in positions])) or \
                    (dx == -1 and (
                            next_x <= -spiral_size // 2 or (next_x, next_z) in [(p[0] // spacing, p[2] // spacing) for p
                                                                                in positions])) or \
                    (dz == -1 and (
                            next_z <= -spiral_size // 2 or (next_x, next_z) in [(p[0] // spacing, p[2] // spacing) for p
                                                                                in positions])):
                # Turn 90 degrees clockwise
                dx, dz = -dz, dx

            x += dx
            z += dz

        # Create rooms
        for pos in positions[:spiral_size * spiral_size]:
            layout.add_room(pos, 0)

        # Connect adjacent rooms in spiral order
        for i in range(len(positions) - 1):
            if i + 1 < len(layout.rooms):
                # Determine connection direction
                curr_pos = positions[i]
                next_pos = positions[i + 1]

                dx = next_pos[0] - curr_pos[0]
                dz = next_pos[2] - curr_pos[2]

                if dx > 0:
                    wall_side = WallSide.EAST
                elif dx < 0:
                    wall_side = WallSide.WEST
                elif dz > 0:
                    wall_side = WallSide.NORTH
                else:
                    wall_side = WallSide.SOUTH

                layout.connect_rooms_adjacent(i, i + 1, wall_side)

        return layout

    def generate_tower_floors(self, rooms_per_floor=4, num_floors=3) -> LevelLayout:
        """Generate a multi-floor tower with rooms around central cores"""
        layout = LevelLayout(self.room_size, self.floor_height)

        for floor in range(num_floors):
            floor_y = floor * self.floor_height

            # Central core room
            core_room = layout.add_room((0, floor_y, 0), floor)
            floor_start_id = len(layout.rooms) - 1

            # Surrounding rooms
            angles = np.linspace(0, 2 * np.pi, rooms_per_floor + 1)[:-1]  # Exclude last point (same as first)
            radius = self.room_size + 3

            for i, angle in enumerate(angles):
                x = radius * np.cos(angle)
                z = radius * np.sin(angle)
                layout.add_room((x, floor_y, z), floor)

                # Connect to core
                outer_room_id = len(layout.rooms) - 1

                # Determine which wall to connect through
                if abs(x) > abs(z):
                    wall_side = WallSide.EAST if x > 0 else WallSide.WEST
                else:
                    wall_side = WallSide.NORTH if z > 0 else WallSide.SOUTH

                layout.connect_rooms_adjacent(floor_start_id, outer_room_id, wall_side)

        # Connect floors vertically
        if num_floors > 1:
            self._connect_tower_floors(layout, rooms_per_floor, num_floors)

        return layout

    def _connect_tower_floors(self, layout: LevelLayout, rooms_per_floor: int, num_floors: int):
        """Connect different floors in the tower"""
        rooms_per_complete_floor = rooms_per_floor + 1  # Including core room

        for floor in range(num_floors - 1):
            # Connect core rooms of adjacent floors
            core_room_1 = floor * rooms_per_complete_floor
            core_room_2 = (floor + 1) * rooms_per_complete_floor

            if core_room_2 < len(layout.rooms):
                # Find unused entrances in both core rooms
                room1 = layout.rooms[core_room_1]
                room2 = layout.rooms[core_room_2]

                unused_ent1 = [e for e in room1.entrances if e.connected_to is None]
                unused_ent2 = [e for e in room2.entrances if e.connected_to is None]

                if unused_ent1 and unused_ent2:
                    ent1 = random.choice(unused_ent1)
                    ent2 = random.choice(unused_ent2)

                    # Create vertical corridor
                    corridor_nodes = [(ent1.position[0], (ent1.position[1] + ent2.position[1]) / 2, ent1.position[2])]
                    layout.connect_rooms_corridor(ent1.entrance_id, ent2.entrance_id, corridor_nodes)

    def generate_maze_like(self, grid_size=4) -> LevelLayout:
        """Generate a maze-like structure with rooms"""
        layout = LevelLayout(self.room_size, self.floor_height)

        spacing = self.room_size + 2

        # Create grid of rooms
        for i in range(grid_size):
            for j in range(grid_size):
                x = i * spacing
                z = j * spacing
                layout.add_room((x, 0, z), 0)

        # Create maze connections (ensure connectivity but not full grid)
        connected = set()
        to_connect = [(0, 0)]  # Start from corner

        while to_connect:
            current = to_connect.pop(0)
            if current in connected:
                continue

            connected.add(current)
            i, j = current
            current_room_id = i * grid_size + j

            # Get neighbors
            neighbors = []
            if i > 0:
                neighbors.append(((i - 1, j), WallSide.WEST))
            if i < grid_size - 1:
                neighbors.append(((i + 1, j), WallSide.EAST))
            if j > 0:
                neighbors.append(((i, j - 1), WallSide.SOUTH))
            if j < grid_size - 1:
                neighbors.append(((i, j + 1), WallSide.NORTH))

            # Randomly connect to some neighbors
            random.shuffle(neighbors)
            num_connections = random.randint(1, min(3, len(neighbors)))

            for neighbor_coord, wall_side in neighbors[:num_connections]:
                ni, nj = neighbor_coord
                neighbor_room_id = ni * grid_size + nj

                if neighbor_room_id < len(layout.rooms):
                    layout.connect_rooms_adjacent(current_room_id, neighbor_room_id, wall_side)

                    if neighbor_coord not in connected:
                        to_connect.append(neighbor_coord)

        return layout


class BatchLevelGenerator:
    """Generate multiple levels with different patterns"""

    def __init__(self, output_dir="generated_architectural_levels"):
        self.output_dir = output_dir
        self.patterns = AdvancedRoomPatterns()
        self.converter = LayoutToGraphConverter()
        self.visualizer = LevelVisualizer()

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

    def generate_pattern_set(self, num_each=3):
        """Generate multiple levels of each pattern type"""

        pattern_generators = [
            ("linear", lambda: self.patterns.generate_linear_corridor(
                num_rooms=random.randint(4, 8), floors=random.randint(1, 2))),
            ("cross", lambda: self.patterns.generate_cross_pattern(
                arm_length=random.randint(2, 4))),
            ("spiral", lambda: self.patterns.generate_spiral_pattern(
                spiral_size=random.randint(3, 5))),
            ("tower", lambda: self.patterns.generate_tower_floors(
                rooms_per_floor=random.randint(3, 6), num_floors=random.randint(2, 4))),
            ("maze", lambda: self.patterns.generate_maze_like(
                grid_size=random.randint(3, 5))),
        ]

        generated_levels = []

        for pattern_name, generator_func in pattern_generators:
            print(f"\n🏗️  Generating {num_each} {pattern_name} patterns...")

            for i in range(num_each):
                try:
                    # Generate layout
                    layout = generator_func()

                    # Convert to graph
                    graph_nodes = self.converter.convert_layout_to_graph(layout)

                    # Create filenames
                    base_name = f"{pattern_name}_{i + 1:02d}"

                    # Save graph
                    graph_file = os.path.join(self.output_dir, f"{base_name}.txt")
                    save_graph_to_file(graph_nodes, graph_file)

                    # Create visualizations
                    layout_vis = os.path.join(self.output_dir, f"{base_name}_layout.png")
                    graph_vis = os.path.join(self.output_dir, f"{base_name}_graph.png")

                    self.visualizer.visualize_layout(layout, f"{pattern_name.title()} Pattern {i + 1}", layout_vis)
                    self.visualizer.visualize_graph(graph_nodes, f"{pattern_name.title()} Graph {i + 1}", graph_vis)

                    generated_levels.append({
                        'pattern': pattern_name,
                        'layout': layout,
                        'graph': graph_nodes,
                        'files': {
                            'graph': graph_file,
                            'layout_vis': layout_vis,
                            'graph_vis': graph_vis
                        }
                    })

                    print(f"  ✅ {base_name}: {len(layout.rooms)} rooms, {len(graph_nodes)} nodes")

                except Exception as e:
                    print(f"  ❌ Error generating {pattern_name} {i + 1}: {e}")

        return generated_levels

    def analyze_generated_levels(self, levels):
        """Analyze the generated levels"""
        print(f"\n📊 Analysis of {len(levels)} generated levels:")
        print("=" * 50)

        by_pattern = {}
        for level in levels:
            pattern = level['pattern']
            if pattern not in by_pattern:
                by_pattern[pattern] = []
            by_pattern[pattern].append(level)

        for pattern, pattern_levels in by_pattern.items():
            print(f"\n{pattern.upper()} PATTERN:")

            room_counts = [len(level['layout'].rooms) for level in pattern_levels]
            node_counts = [len(level['graph']) for level in pattern_levels]
            connection_counts = [len(level['layout'].connections) for level in pattern_levels]

            print(f"  Levels: {len(pattern_levels)}")
            print(f"  Rooms: avg={np.mean(room_counts):.1f}, range={min(room_counts)}-{max(room_counts)}")
            print(f"  Nodes: avg={np.mean(node_counts):.1f}, range={min(node_counts)}-{max(node_counts)}")
            print(
                f"  Connections: avg={np.mean(connection_counts):.1f}, range={min(connection_counts)}-{max(connection_counts)}")

            # Analyze node type distribution
            all_nodes = []
            for level in pattern_levels:
                all_nodes.extend(level['graph'])

            type_counts = {}
            for node in all_nodes:
                node_type = node['type']
                type_counts[node_type] = type_counts.get(node_type, 0) + 1

            total_nodes = len(all_nodes)
            print(f"  Node distribution:", end="")
            for node_type, count in type_counts.items():
                percentage = (count / total_nodes) * 100
                print(f" {node_type}={percentage:.1f}%", end="")
            print()


def create_dataset_for_training(output_dir="architectural_dataset", num_levels_per_pattern=10):
    """Create a large dataset suitable for training ML models"""

    print(f"🎯 Creating architectural dataset with {num_levels_per_pattern} levels per pattern")

    generator = BatchLevelGenerator(output_dir)
    levels = generator.generate_pattern_set(num_each=num_levels_per_pattern)

    # Analyze the dataset
    generator.analyze_generated_levels(levels)

    # Create a summary file
    summary_file = os.path.join(output_dir, "dataset_summary.txt")
    with open(summary_file, 'w') as f:
        f.write(f"Architectural Level Dataset\n")
        f.write(f"Generated: {len(levels)} levels\n")
        f.write(f"Patterns: linear, cross, spiral, tower, maze\n")
        f.write(f"Files per level: .txt (graph), _layout.png, _graph.png\n\n")

        by_pattern = {}
        for level in levels:
            pattern = level['pattern']
            if pattern not in by_pattern:
                by_pattern[pattern] = []
            by_pattern[pattern].append(level)

        for pattern, pattern_levels in by_pattern.items():
            f.write(f"{pattern.upper()}: {len(pattern_levels)} levels\n")

    print(f"\n🎉 Dataset created successfully!")
    print(f"📁 Location: {output_dir}/")
    print(f"📊 Summary: {summary_file}")

    return levels


# Example usage
if __name__ == "__main__":
    print("🏛️  Advanced Room Pattern Generator")
    print("=" * 50)

    # Generate individual patterns for testing
    patterns = AdvancedRoomPatterns(room_size=8.0, floor_height=4.0)
    converter = LayoutToGraphConverter()
    visualizer = LevelVisualizer()

    # Test each pattern
    test_patterns = [
        ("Linear Corridor", lambda: patterns.generate_linear_corridor(num_rooms=5, floors=1)),
        ("Cross Pattern", lambda: patterns.generate_cross_pattern(arm_length=3)),
        ("Spiral Pattern", lambda: patterns.generate_spiral_pattern(spiral_size=4)),
        ("Tower Pattern", lambda: patterns.generate_tower_floors(rooms_per_floor=4, num_floors=3)),
        ("Maze Pattern", lambda: patterns.generate_maze_like(grid_size=4)),
    ]

    os.makedirs("test_patterns", exist_ok=True)

    for pattern_name, generator_func in test_patterns:
        print(f"\n🔧 Testing {pattern_name}...")

        try:
            layout = generator_func()
            graph = converter.convert_layout_to_graph(layout)

            base_name = pattern_name.lower().replace(" ", "_")

            # Save files
            save_graph_to_file(graph, f"test_patterns/{base_name}.txt")
            visualizer.visualize_layout(layout, pattern_name, f"test_patterns/{base_name}_layout.png")
            visualizer.visualize_graph(graph, f"{pattern_name} Graph", f"test_patterns/{base_name}_graph.png")

            print(f"  ✅ {len(layout.rooms)} rooms, {len(graph)} nodes, {len(layout.connections)} connections")

        except Exception as e:
            print(f"  ❌ Error: {e}")

    print(f"\n🎯 Creating full dataset...")

    # Create full dataset
    levels = create_dataset_for_training("architectural_dataset", num_levels_per_pattern=5)

    print(f"\n🎉 All done! Check the directories:")
    print(f"  📁 test_patterns/ - Individual pattern tests")
    print(f"  📁 architectural_dataset/ - Full dataset for training")