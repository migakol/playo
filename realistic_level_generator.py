import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
import csv
import io
import os
import random
from typing import List, Dict, Tuple, Optional
import json
from dataclasses import dataclass
from scipy import stats
import networkx as nx
from visualize_levels import visualize_sample_level
from itertools import combinations


@dataclass
class RoomData:
    """Data about a single room"""
    center_id: int
    center_pos: Tuple[float, float, float]
    entrance_ids: List[int]
    entrance_positions: List[Tuple[float, float, float]]
    room_size: float  # Average distance from center to entrances
    connections: List[int]  # IDs of connected rooms


@dataclass
class DataDistributions:
    """Learned distributions from data"""
    room_size_distribution: List[float]
    connections_per_room_distribution: Dict[int, float]
    total_rooms_distribution: Dict[int, float]
    spatial_patterns: Dict[str, any]
    entrance_patterns: Dict[str, any]


class DataAnalyzer:
    """Analyze existing data to learn realistic distributions"""

    def __init__(self):
        self.room_data = []
        self.distributions = None

    def analyze_data_directory(self, data_dir: str) -> DataDistributions:
        """Analyze all level files to extract realistic distributions"""
        print("📊 Analyzing data distributions...")

        all_level_rooms = []  # List of lists - each inner list is rooms from one level
        level_room_counts = []

        txt_files = [f for f in os.listdir(data_dir) if f.endswith('.txt')]

        for filename in txt_files:
            filepath = os.path.join(data_dir, filename)
            try:
                level_rooms = self._analyze_single_level(filepath)
                if level_rooms:  # Only add non-empty levels
                    all_level_rooms.append(level_rooms)
                    level_room_counts.append(len(level_rooms))
                    print(f"  📁 {filename}: {len(level_rooms)} rooms")
            except Exception as e:
                print(f"  ❌ Error analyzing {filename}: {e}")

        if not all_level_rooms:
            raise ValueError("No valid room data found in any files")

        total_rooms = sum(level_room_counts)
        print(f"✅ Analyzed {len(txt_files)} files, {len(all_level_rooms)} valid levels, {total_rooms} total rooms")

        # Extract distributions - now passing list of levels, not flat list of rooms
        distributions = self._extract_distributions(all_level_rooms, level_room_counts)

        # Save distributions for later use
        self._save_distributions(distributions, data_dir)

        # Store for visualization (flatten for backward compatibility)
        self.room_data = [room for level_rooms in all_level_rooms for room in level_rooms]
        self.distributions = distributions

        return distributions

    def _analyze_single_level(self, filepath: str) -> List[RoomData]:
        """Analyze a single level file to extract room data"""
        # Load nodes
        nodes = self._load_level_nodes(filepath)

        # Identify rooms (centers and their entrances)
        centers = [node for node in nodes if node['type'] == 'center']
        entrances = [node for node in nodes if node['type'] == 'entrance']

        rooms = []

        for center in centers:
            # Find entrances connected to this center
            connected_entrances = []
            for entrance in entrances:
                if (center['id'] in entrance['connections'] or
                        entrance['id'] in center['connections']):
                    connected_entrances.append(entrance)

            if connected_entrances:
                # Calculate room size (average distance from center to entrances)
                center_pos = np.array(center['pos'])
                distances = []
                entrance_positions = []

                for entrance in connected_entrances:
                    entrance_pos = np.array(entrance['pos'])
                    entrance_positions.append(tuple(entrance_pos))
                    dist = np.linalg.norm(entrance_pos - center_pos)
                    distances.append(dist)

                avg_room_size = np.mean(distances)

                # Find connections to other rooms (through entrance-to-entrance connections)
                room_connections = self._find_room_connections(
                    center['id'], connected_entrances, centers
                )

                room_data = RoomData(
                    center_id=center['id'],
                    center_pos=tuple(center['pos']),
                    entrance_ids=[e['id'] for e in connected_entrances],
                    entrance_positions=entrance_positions,
                    room_size=avg_room_size,
                    connections=room_connections
                )

                rooms.append(room_data)

        return rooms

    def _load_level_nodes(self, filepath: str) -> List[Dict]:
        """Load nodes from a level file"""
        with open(filepath, 'r') as f:
            lines = f.readlines()

        data_lines = [line.strip() for line in lines if line.strip() and not line.strip().startswith('---')]

        nodes = []
        for line in data_lines:
            if 'node_type,pos,connections' in line:
                continue

            csv_reader = csv.reader(io.StringIO(line))
            try:
                parts = next(csv_reader)
                if len(parts) < 4:
                    continue

                node = {
                    'id': int(parts[0]),
                    'type': parts[1],
                    'pos': eval(parts[2]),
                    'connections': eval(parts[3])
                }
                nodes.append(node)
            except:
                continue

        return nodes

    def _find_room_connections(self, center_id: int, entrances: List[Dict],
                               all_centers: List[Dict]) -> List[int]:
        """Find which other rooms this room connects to"""
        connected_room_ids = []

        for entrance in entrances:
            for conn_id in entrance['connections']:
                if conn_id != center_id:  # Not connected to own center
                    # Find which room this connection belongs to
                    for other_center in all_centers:
                        if other_center['id'] != center_id:
                            # Check if this connection is to an entrance of another room
                            other_center_id = other_center['id']
                            if (conn_id in other_center.get('connections', []) or
                                    any(conn_id == ent_conn for ent_conn in other_center.get('connections', []))):
                                if other_center_id not in connected_room_ids:
                                    connected_room_ids.append(other_center_id)

        return connected_room_ids

    def _extract_distributions(self, all_level_rooms: List[List[RoomData]],
                               level_room_counts: List[int]) -> DataDistributions:
        """Extract probability distributions from room data - FIXED VERSION"""

        # Flatten room data for individual room statistics
        all_rooms = [room for level_rooms in all_level_rooms for room in level_rooms]

        # 1. Room size distribution
        room_sizes = [room.room_size for room in all_rooms]

        # 2. Connections per room distribution
        connections_per_room = [len(room.connections) for room in all_rooms]
        connections_dist = {}
        for count in range(max(connections_per_room) + 1):
            connections_dist[count] = connections_per_room.count(count) / len(connections_per_room)

        # 3. Total rooms per level distribution
        total_rooms_dist = {}
        for count in range(min(level_room_counts), max(level_room_counts) + 1):
            total_rooms_dist[count] = level_room_counts.count(count) / len(level_room_counts)

        # 4. Spatial patterns (room positions relative to each other WITHIN levels)
        spatial_patterns = self._analyze_spatial_patterns_per_level(all_level_rooms)

        # 5. Entrance patterns (how entrances are positioned around centers)
        entrance_patterns = self._analyze_entrance_patterns(all_rooms)  # This one is fine as-is

        distributions = DataDistributions(
            room_size_distribution=room_sizes,
            connections_per_room_distribution=connections_dist,
            total_rooms_distribution=total_rooms_dist,
            spatial_patterns=spatial_patterns,
            entrance_patterns=entrance_patterns
        )

        return distributions

    def _analyze_spatial_patterns_per_level(self, all_level_rooms: List[List[RoomData]]) -> Dict[str, any]:
        """Analyze spatial relationships between rooms WITHIN each level - CORRECTED VERSION"""

        all_distances = []
        alignments = {'horizontal': 0, 'vertical': 0, 'diagonal': 0}

        # Process each level separately
        for cnt_k, level_rooms in enumerate(all_level_rooms):
            if len(level_rooms) < 2:
                continue

            if cnt_k % 1000 == 0:
                print('Spatial positions ', cnt_k, ' out of ', len(all_level_rooms))

            # Convert room positions for this level only
            room_positions = np.array([room.center_pos for room in level_rooms])
            n_rooms = len(room_positions)

            # Vectorized analysis within this level only
            pos_diff = room_positions[:, np.newaxis, :] - room_positions[np.newaxis, :, :]
            distance_matrix = np.linalg.norm(pos_diff, axis=2)

            # Extract upper triangular part (excluding diagonal)
            upper_tri_indices = np.triu_indices(n_rooms, k=1)
            level_distances = distance_matrix[upper_tri_indices]
            all_distances.extend(level_distances.tolist())

            # Alignment analysis for this level
            dx = np.abs(pos_diff[:, :, 0])  # X differences
            dz = np.abs(pos_diff[:, :, 2])  # Z differences

            # Get upper triangular parts for this level
            dx_pairs = dx[upper_tri_indices]
            dz_pairs = dz[upper_tri_indices]

            # Count alignments for this level
            vertical_mask = dx_pairs < 1.0
            horizontal_mask = dz_pairs < 1.0
            diagonal_mask = ~(vertical_mask | horizontal_mask)

            alignments['vertical'] += np.sum(vertical_mask)
            alignments['horizontal'] += np.sum(horizontal_mask)
            alignments['diagonal'] += np.sum(diagonal_mask)

        # Normalize alignment counts to probabilities
        total_pairs = sum(alignments.values())
        if total_pairs > 0:
            alignments = {key: count / total_pairs for key, count in alignments.items()}

        return {
            'room_distances': all_distances,
            'alignment_preferences': alignments,
            'avg_distance': float(np.mean(all_distances)) if all_distances else 0,
            'distance_std': float(np.std(all_distances)) if all_distances else 0
        }

    def _analyze_entrance_patterns(self, room_data: List[RoomData]) -> Dict[str, any]:
        """Analyze how entrances are positioned around centers - OPTIMIZED VERSION"""

        if not room_data:
            return {}

        # Pre-allocate lists to avoid repeated append operations
        entrance_distances = []
        entrance_angles = []
        entrances_per_room = []

        # Vectorized processing for each room
        for cnt_k, room in enumerate(room_data):
            if not room.entrance_positions:
                continue

            if cnt_k % 1000 == 0:
                print('Entrance ', cnt_k, ' out of ', len(room_data))

            center_pos = np.array(room.center_pos)
            entrance_positions = np.array(room.entrance_positions)
            entrances_per_room.append(len(room.entrance_positions))

            # Vectorized distance calculation for all entrances in this room
            entrance_vectors = entrance_positions - center_pos
            distances = np.linalg.norm(entrance_vectors, axis=1)
            entrance_distances.extend(distances.tolist())

            # Vectorized angle calculation (in XZ plane)
            dx = entrance_vectors[:, 0]  # X differences
            dz = entrance_vectors[:, 2]  # Z differences
            angles = np.arctan2(dz, dx)
            entrance_angles.extend(angles.tolist())

        # Convert to numpy arrays for efficient statistics
        entrance_distances = np.array(entrance_distances)
        entrance_angles = np.array(entrance_angles)
        entrances_per_room = np.array(entrances_per_room)

        return {
            'entrance_distances': entrance_distances.tolist(),
            'entrance_angles': entrance_angles.tolist(),
            'entrances_per_room': entrances_per_room.tolist(),
            'avg_entrances_per_room': float(np.mean(entrances_per_room)) if len(entrances_per_room) > 0 else 4
        }

    def _save_distributions(self, distributions: DataDistributions, data_dir: str):
        """Save learned distributions to file"""
        output_file = os.path.join(data_dir, "learned_distributions.json")

        # Convert numpy arrays and complex objects to serializable format
        serializable_data = {
            'room_size_distribution': distributions.room_size_distribution,
            'connections_per_room_distribution': distributions.connections_per_room_distribution,
            'total_rooms_distribution': distributions.total_rooms_distribution,
            'spatial_patterns': {
                'room_distances': distributions.spatial_patterns.get('room_distances', []),
                'alignment_preferences': distributions.spatial_patterns.get('alignment_preferences', {}),
                'avg_distance': float(distributions.spatial_patterns.get('avg_distance', 0)),
                'distance_std': float(distributions.spatial_patterns.get('distance_std', 0))
            },
            'entrance_patterns': {
                'entrance_distances': distributions.entrance_patterns.get('entrance_distances', []),
                'entrance_angles': distributions.entrance_patterns.get('entrance_angles', []),
                'entrances_per_room': distributions.entrance_patterns.get('entrances_per_room', []),
                'avg_entrances_per_room': float(distributions.entrance_patterns.get('avg_entrances_per_room', 4))
            }
        }

        with open(output_file, 'w') as f:
            json.dump(serializable_data, f, indent=2)

        print(f"💾 Distributions saved to: {output_file}")

    def visualize_distributions(self, save_dir: str = "distribution_analysis"):
        """Create visualizations of learned distributions"""
        if not self.distributions:
            print("❌ No distributions to visualize. Run analyze_data_directory first.")
            return

        os.makedirs(save_dir, exist_ok=True)

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # Room size distribution
        if self.distributions.room_size_distribution:
            axes[0, 0].hist(self.distributions.room_size_distribution, bins=20, alpha=0.7, edgecolor='black')
            axes[0, 0].set_title('Room Size Distribution')
            axes[0, 0].set_xlabel('Room Size')
            axes[0, 0].set_ylabel('Frequency')

        # Connections per room
        conn_counts = list(self.distributions.connections_per_room_distribution.keys())
        conn_probs = list(self.distributions.connections_per_room_distribution.values())
        if conn_counts:
            axes[0, 1].bar(conn_counts, conn_probs, alpha=0.7, edgecolor='black')
            axes[0, 1].set_title('Connections per Room Distribution')
            axes[0, 1].set_xlabel('Number of Connections')
            axes[0, 1].set_ylabel('Probability')

        # Total rooms per level
        room_counts = list(self.distributions.total_rooms_distribution.keys())
        room_probs = list(self.distributions.total_rooms_distribution.values())
        if room_counts:
            axes[0, 2].bar(room_counts, room_probs, alpha=0.7, edgecolor='black')
            axes[0, 2].set_title('Rooms per Level Distribution')
            axes[0, 2].set_xlabel('Number of Rooms')
            axes[0, 2].set_ylabel('Probability')

        # Room distances (now correctly computed within levels only)
        room_distances = self.distributions.spatial_patterns.get('room_distances', [])
        if room_distances:
            axes[1, 0].hist(room_distances, bins=20, alpha=0.7, edgecolor='black')
            axes[1, 0].set_title('Inter-Room Distance Distribution (Within Levels)')
            axes[1, 0].set_xlabel('Distance')
            axes[1, 0].set_ylabel('Frequency')

        # Entrance distances
        entrance_distances = self.distributions.entrance_patterns.get('entrance_distances', [])
        if entrance_distances:
            axes[1, 1].hist(entrance_distances, bins=20, alpha=0.7, edgecolor='black')
            axes[1, 1].set_title('Center-to-Entrance Distance Distribution')
            axes[1, 1].set_xlabel('Distance')
            axes[1, 1].set_ylabel('Frequency')

        # Alignment preferences (now correctly computed within levels only)
        align_prefs = self.distributions.spatial_patterns.get('alignment_preferences', {})
        if align_prefs and sum(align_prefs.values()) > 0:
            labels = list(align_prefs.keys())
            values = list(align_prefs.values())
            axes[1, 2].pie(values, labels=labels, autopct='%1.1f%%')
            axes[1, 2].set_title('Room Alignment Preferences (Within Levels)')

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'data_distributions.png'), dpi=300, bbox_inches='tight')
        plt.show()

        print(f"📊 Distribution visualizations saved to: {save_dir}/")

        # Print summary stats
        print(f"\n📈 Analysis Summary:")
        print(f"  Total room pairs analyzed: {len(room_distances)}")
        print(f"  Average room distance: {self.distributions.spatial_patterns.get('avg_distance', 0):.2f}")
        print(f"  Alignment preferences: {align_prefs}")
        total_levels = len(self.distributions.total_rooms_distribution)
        total_rooms_analyzed = sum(k * v * total_levels for k, v in self.distributions.total_rooms_distribution.items())
        print(f"  Estimated rooms analyzed: {int(total_rooms_analyzed)}")
        print(f"  Actual room pairs (within levels): {len(room_distances)}")


class RealisticLevelGenerator:
    """Generate levels based on learned data distributions"""

    def __init__(self, distributions: DataDistributions):
        self.distributions = distributions

    def generate_realistic_level(self) -> List[Dict]:
        """Generate a single realistic level based on learned distributions"""

        # 1. Sample number of rooms from distribution
        num_rooms = self._sample_total_rooms()

        # 2. Generate room positions ensuring straight-line connections
        room_positions = self._generate_room_positions(num_rooms)

        # 3. Create room connection graph based on learned distribution
        connection_graph = self._create_connection_graph(num_rooms)

        # 4. Generate rooms with realistic sizes and entrance patterns
        nodes = self._generate_level_nodes(room_positions, connection_graph)

        return nodes

    def _sample_total_rooms(self) -> int:
        """Sample number of rooms from learned distribution"""
        room_counts = list(self.distributions.total_rooms_distribution.keys())
        probabilities = list(self.distributions.total_rooms_distribution.values())

        if not room_counts:
            return random.randint(3, 8)  # Fallback

        return np.random.choice(room_counts, p=probabilities)

    def _sample_connections_for_room(self) -> int:
        """Sample number of connections for a room"""
        conn_counts = list(self.distributions.connections_per_room_distribution.keys())
        probabilities = list(self.distributions.connections_per_room_distribution.values())

        if not conn_counts:
            return random.randint(1, 3)  # Fallback

        return np.random.choice(conn_counts, p=probabilities)

    def _sample_room_size(self) -> float:
        """Sample room size from learned distribution"""
        sizes = self.distributions.room_size_distribution

        if not sizes:
            return random.uniform(3, 8)  # Fallback

        return random.choice(sizes) + random.uniform(-0.5, 0.5)  # Add small variation

    def _generate_room_positions(self, num_rooms: int) -> List[Tuple[float, float, float]]:
        """Generate room positions that allow straight-line connections"""

        # Use grid-based approach with variation to ensure straight connections
        spatial_info = self.distributions.spatial_patterns
        avg_distance = spatial_info.get('avg_distance', 12)
        distance_std = spatial_info.get('distance_std', 3)

        # Determine grid size
        grid_size = max(2, int(np.ceil(np.sqrt(num_rooms * 1.2))))

        positions = []
        used_positions = set()

        for i in range(num_rooms):
            # Try to place room on grid with some variation
            attempts = 0
            while attempts < 50:  # Prevent infinite loop

                if attempts < 20:
                    # Grid-based placement with small variations
                    grid_x = (i % grid_size)
                    grid_z = (i // grid_size)

                    # Add controlled variation
                    x = grid_x * avg_distance + random.uniform(-2, 2)
                    z = grid_z * avg_distance + random.uniform(-2, 2)
                    y = 0  # Single floor for now, can be extended

                else:
                    # Random placement if grid fails
                    x = random.uniform(-avg_distance * grid_size / 2, avg_distance * grid_size / 2)
                    z = random.uniform(-avg_distance * grid_size / 2, avg_distance * grid_size / 2)
                    y = 0

                # Round to ensure straight-line connections possible
                x = round(x / 2) * 2  # Round to even numbers
                z = round(z / 2) * 2

                pos = (x, y, z)

                # Check minimum distance
                if all(np.linalg.norm(np.array(pos) - np.array(existing)) >= avg_distance * 0.7
                       for existing in positions):
                    positions.append(pos)
                    used_positions.add((int(x), int(z)))
                    break

                attempts += 1

            # If we couldn't place after many attempts, place randomly
            if len(positions) <= i:
                x = i * avg_distance
                z = 0
                positions.append((x, 0, z))

        return positions

    def _create_connection_graph(self, num_rooms: int) -> nx.Graph:
        """Create room connection graph based on learned distributions and straight-line constraint"""

        G = nx.Graph()
        G.add_nodes_from(range(num_rooms))

        # Sample desired connections for each room
        desired_connections = []
        for i in range(num_rooms):
            desired_conns = self._sample_connections_for_room()
            desired_connections.append(desired_conns)

        # Create connections preferring straight lines (horizontal/vertical)
        alignment_prefs = self.distributions.spatial_patterns.get('alignment_preferences', {
            'horizontal': 0.4, 'vertical': 0.4, 'diagonal': 0.2
        })

        # First ensure connectivity
        self._ensure_connectivity(G, num_rooms)

        # Add additional connections based on desired distribution
        for room_id in range(num_rooms):
            current_connections = len(list(G.neighbors(room_id)))
            needed_connections = max(0, desired_connections[room_id] - current_connections)

            # Find potential targets (prefer aligned rooms)
            potential_targets = []
            for other_room in range(num_rooms):
                if other_room != room_id and not G.has_edge(room_id, other_room):
                    potential_targets.append(other_room)

            # Sort by alignment preference (would need room positions, simplified here)
            random.shuffle(potential_targets)

            # Add connections up to desired amount
            for target in potential_targets[:needed_connections]:
                G.add_edge(room_id, target)

        return G

    def _ensure_connectivity(self, G: nx.Graph, num_rooms: int):
        """Ensure all rooms are connected in a single component"""

        # Create minimum spanning tree for connectivity
        for i in range(1, num_rooms):
            # Connect to previous room (simple chain)
            G.add_edge(i - 1, i)

        # Add some additional connections for variety
        additional_edges = max(1, num_rooms // 4)
        for _ in range(additional_edges):
            room1 = random.randint(0, num_rooms - 1)
            room2 = random.randint(0, num_rooms - 1)
            if room1 != room2:
                G.add_edge(room1, room2)

    def _generate_level_nodes(self, room_positions: List[Tuple[float, float, float]],
                              connection_graph: nx.Graph) -> List[Dict]:
        """Generate the actual level nodes (centers, entrances, corridors)"""

        nodes = []
        node_id_counter = 0

        # Generate rooms
        room_data = {}

        for room_id, center_pos in enumerate(room_positions):
            # Generate center node
            center_node = {
                'id': node_id_counter,
                'type': 'center',
                'pos': list(center_pos),
                'connections': []
            }

            room_size = self._sample_room_size()

            # Generate entrances around the center
            entrance_nodes = []
            avg_entrances = self.distributions.entrance_patterns.get('avg_entrances_per_room', 4)
            num_entrances = max(2, int(avg_entrances + random.uniform(-1, 1)))

            # Place entrances at cardinal directions with room size distance
            directions = [
                (room_size, 0, 0),  # East
                (-room_size, 0, 0),  # West
                (0, 0, room_size),  # North
                (0, 0, -room_size)  # South
            ]

            # Select directions for entrances
            selected_directions = random.sample(directions, min(num_entrances, len(directions)))

            for direction in selected_directions:
                node_id_counter += 1
                entrance_pos = [
                    center_pos[0] + direction[0],
                    center_pos[1] + direction[1],
                    center_pos[2] + direction[2]
                ]

                entrance_node = {
                    'id': node_id_counter,
                    'type': 'entrance',
                    'pos': entrance_pos,
                    'connections': [center_node['id']]  # Connect to center
                }

                entrance_nodes.append(entrance_node)
                center_node['connections'].append(entrance_node['id'])

            nodes.append(center_node)
            nodes.extend(entrance_nodes)

            room_data[room_id] = {
                'center': center_node,
                'entrances': entrance_nodes
            }

            node_id_counter += 1

        # Create connections between rooms
        for room1, room2 in connection_graph.edges():
            self._connect_rooms_with_corridors(room1, room2, room_data, nodes, node_id_counter)

        return nodes

    def _connect_rooms_with_corridors(self, room1_id: int, room2_id: int,
                                      room_data: Dict, nodes: List[Dict],
                                      node_id_counter: int):
        """Connect two rooms with straight-line corridors"""

        room1_entrances = room_data[room1_id]['entrances']
        room2_entrances = room_data[room2_id]['entrances']

        # Find best entrance pair for straight-line connection
        best_pair = None
        min_bends = float('inf')

        for ent1 in room1_entrances:
            for ent2 in room2_entrances:
                pos1 = np.array(ent1['pos'])
                pos2 = np.array(ent2['pos'])

                # Count "bends" needed (prefer horizontal/vertical lines)
                dx = abs(pos2[0] - pos1[0])
                dz = abs(pos2[2] - pos1[2])

                bends = 0
                if dx > 1 and dz > 1:  # Diagonal connection
                    bends = 1  # One bend to make L-shape

                # Prefer unused entrances
                if len(ent1['connections']) > 1:  # Already connected to something other than center
                    bends += 2
                if len(ent2['connections']) > 1:
                    bends += 2

                if bends < min_bends:
                    min_bends = bends
                    best_pair = (ent1, ent2)

        if best_pair:
            ent1, ent2 = best_pair
            pos1 = np.array(ent1['pos'])
            pos2 = np.array(ent2['pos'])

            # Create straight-line connection (L-shaped if needed)
            dx = pos2[0] - pos1[0]
            dz = pos2[2] - pos1[2]

            if abs(dx) < 1:  # Vertical line
                # Direct vertical connection
                ent1['connections'].append(ent2['id'])
                ent2['connections'].append(ent1['id'])

            elif abs(dz) < 1:  # Horizontal line
                # Direct horizontal connection
                ent1['connections'].append(ent2['id'])
                ent2['connections'].append(ent1['id'])

            else:
                # L-shaped connection with corridor node
                # Place corridor at the corner
                if random.choice([True, False]):
                    corner_pos = [pos1[0], pos1[1], pos2[2]]  # Horizontal first, then vertical
                else:
                    corner_pos = [pos2[0], pos1[1], pos1[2]]  # Vertical first, then horizontal

                corridor_node = {
                    'id': len(nodes),  # Use current length as ID
                    'type': 'corridor',
                    'pos': corner_pos,
                    'connections': [ent1['id'], ent2['id']]
                }

                ent1['connections'].append(corridor_node['id'])
                ent2['connections'].append(corridor_node['id'])

                nodes.append(corridor_node)


def load_distributions(data_dir: str) -> DataDistributions:
    """Load previously saved distributions"""
    dist_file = os.path.join(data_dir, "learned_distributions.json")

    if not os.path.exists(dist_file):
        raise FileNotFoundError(f"Distributions file not found: {dist_file}")

    with open(dist_file, 'r') as f:
        data = json.load(f)

    return DataDistributions(
        room_size_distribution=data['room_size_distribution'],
        connections_per_room_distribution=data['connections_per_room_distribution'],
        total_rooms_distribution=data['total_rooms_distribution'],
        spatial_patterns=data['spatial_patterns'],
        entrance_patterns=data['entrance_patterns']
    )


def save_graph_to_file(nodes, filename):
    """Save graph nodes to file in the required format"""
    import csv
    import io

    lines = ["---", "node_type,pos,connections"]

    for node in nodes:
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow([node["id"], node["type"], str(node["pos"]), str(node["connections"])])
        line = output.getvalue().strip()
        lines.append(line)

    lines.append("---")

    with open(filename, 'w') as f:
        f.write('\n'.join(lines))


def generate_realistic_levels(data_dir: str, num_levels: int, output_dir: str = "realistic_levels"):
    """Complete pipeline: analyze data and generate realistic levels"""

    print("🎯 Realistic Level Generation Pipeline")
    print("=" * 50)

    # Step 1: Analyze data distributions
    analyzer = DataAnalyzer()
    distributions = analyzer.analyze_data_directory(data_dir)

    # Create visualizations
    analyzer.visualize_distributions(os.path.join(output_dir, "analysis"))

    print("\n📊 Learned Distributions Summary:")
    print(f"  Room sizes: {len(distributions.room_size_distribution)} samples")
    print(f"  Avg room size: {np.mean(distributions.room_size_distribution):.2f}")
    print(f"  Connection distribution: {distributions.connections_per_room_distribution}")
    print(f"  Rooms per level: {distributions.total_rooms_distribution}")

    # Step 2: Generate levels
    print(f"\n🏗️ Generating {num_levels} realistic levels...")

    os.makedirs(output_dir, exist_ok=True)
    generator = RealisticLevelGenerator(distributions)

    generated_levels = []
    for i in range(num_levels):
        try:
            nodes = generator.generate_realistic_level()

            # Save level
            filename = os.path.join(output_dir, f"realistic_level_{i + 1:03d}.txt")
            save_graph_to_file(nodes, filename)

            nodes = visualize_sample_level(filename, "/Users/michaelkolomenkin/Data/playo/output")

            generated_levels.append({
                'id': i + 1,
                'nodes': nodes,
                'filename': filename
            })

            # Count room types
            centers = len([n for n in nodes if n['type'] == 'center'])
            entrances = len([n for n in nodes if n['type'] == 'entrance'])
            corridors = len([n for n in nodes if n['type'] == 'corridor'])

            print(f"  ✅ Level {i + 1}: {centers} rooms, {entrances} entrances, {corridors} corridors")

        except Exception as e:
            print(f"  ❌ Error generating level {i + 1}: {e}")

    # Step 3: Create summary
    summary_file = os.path.join(output_dir, "generation_summary.txt")
    with open(summary_file, 'w') as f:
        f.write("REALISTIC LEVEL GENERATION SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Source data: {data_dir}\n")
        f.write(f"Generated levels: {len(generated_levels)}\n")
        f.write(f"Output directory: {output_dir}\n\n")

        f.write("LEARNED PATTERNS:\n")
        f.write(
            f"  Room size range: {min(distributions.room_size_distribution):.2f} - {max(distributions.room_size_distribution):.2f}\n")
        f.write(f"  Avg room size: {np.mean(distributions.room_size_distribution):.2f}\n")
        f.write(f"  Connection preferences: {distributions.connections_per_room_distribution}\n")
        f.write(f"  Level size preferences: {distributions.total_rooms_distribution}\n")

    print(f"\n🎉 Generation completed!")
    print(f"📁 Files saved to: {output_dir}/")
    print(f"📊 Summary: {summary_file}")

    return generated_levels


# Example usage
if __name__ == "__main__":

    # Example with sample data
    sample_data = '/Users/michaelkolomenkin/Data/playo/files_for_yaron'
    if os.path.exists(sample_data):
        print("🧪 Testing with sample_data...")
        try:
            levels = generate_realistic_levels(sample_data, num_levels=5, output_dir="/Users/michaelkolomenkin/Data/playo/output")
            print(f"✅ Generated {len(levels)} test levels")
        except Exception as e:
            print(f"❌ Error: {e}")

    print("\n💡 Usage examples:")
    print("  # Generate 20 realistic levels from your data")
    print("  levels = generate_realistic_levels('your_data_dir', 20, 'realistic_output')")
    print("\n  # Just analyze distributions")
    print("  analyzer = DataAnalyzer()")
    print("  distributions = analyzer.analyze_data_directory('your_data')")
    print("  analyzer.visualize_distributions()")
