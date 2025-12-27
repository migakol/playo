import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import seaborn as sns
from collections import defaultdict, Counter
import os
import csv
import io
from dataclasses import dataclass
from typing import List, Dict, Tuple, Any
import networkx as nx
from scipy.spatial.distance import pdist, squareform
from sklearn.manifold import TSNE
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


@dataclass
class LevelFeatures:
    """Features extracted from a level"""
    # Basic statistics
    num_nodes: int
    num_centers: int
    num_entrances: int
    num_corridors: int
    num_connections: int

    # Spatial features
    bounding_box_volume: float
    spatial_spread: float
    avg_node_distance: float
    floor_distribution: List[int]  # Nodes per floor

    # Graph topology features
    avg_degree: float
    max_degree: int
    clustering_coefficient: float
    diameter: int
    num_components: int

    # Room structure features
    rooms_detected: int
    avg_entrances_per_room: float
    room_sizes: List[float]
    room_connectivity: float

    # Geometric patterns
    linearity_score: float
    symmetry_score: float
    compactness_score: float
    multi_floor_score: float


class LevelAnalyzer:
    """Analyze individual levels and extract features"""

    def __init__(self):
        self.node_type_to_idx = {'center': 0, 'entrance': 1, 'corridor': 2}

    def load_level(self, filepath):
        """Load a level file"""
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

    def extract_features(self, nodes) -> LevelFeatures:
        """Extract comprehensive features from a level"""
        if not nodes:
            return self._empty_features()

        # Basic counts
        type_counts = Counter(node['type'] for node in nodes)
        num_centers = type_counts.get('center', 0)
        num_entrances = type_counts.get('entrance', 0)
        num_corridors = type_counts.get('corridor', 0)

        # Connection analysis
        total_connections = sum(len(node['connections']) for node in nodes) // 2

        # Spatial analysis
        positions = np.array([node['pos'] for node in nodes])
        spatial_features = self._analyze_spatial_distribution(positions)

        # Graph topology analysis
        graph_features = self._analyze_graph_topology(nodes)

        # Room structure analysis
        room_features = self._analyze_room_structure(nodes)

        # Geometric pattern analysis
        pattern_features = self._analyze_geometric_patterns(positions, nodes)

        return LevelFeatures(
            num_nodes=len(nodes),
            num_centers=num_centers,
            num_entrances=num_entrances,
            num_corridors=num_corridors,
            num_connections=total_connections,
            **spatial_features,
            **graph_features,
            **room_features,
            **pattern_features
        )

    def _empty_features(self):
        """Return empty features for invalid levels"""
        return LevelFeatures(0, 0, 0, 0, 0, 0, 0, 0, [], 0, 0, 0, 0, 0, 0, 0, [], 0, 0, 0, 0, 0)

    def _analyze_spatial_distribution(self, positions):
        """Analyze spatial distribution of nodes"""
        if len(positions) == 0:
            return {'bounding_box_volume': 0, 'spatial_spread': 0, 'avg_node_distance': 0, 'floor_distribution': []}

        # Bounding box
        min_coords = np.min(positions, axis=0)
        max_coords = np.max(positions, axis=0)
        dimensions = max_coords - min_coords
        volume = np.prod(dimensions + 1e-6)  # Add small value to avoid zero

        # Spatial spread (standard deviation of positions)
        spread = np.mean(np.std(positions, axis=0))

        # Average distance between nodes
        if len(positions) > 1:
            distances = pdist(positions)
            avg_distance = np.mean(distances)
        else:
            avg_distance = 0

        # Floor distribution (Y coordinate)
        floors = positions[:, 1]
        unique_floors = np.unique(floors)
        floor_dist = [np.sum(floors == floor) for floor in unique_floors]

        return {
            'bounding_box_volume': volume,
            'spatial_spread': spread,
            'avg_node_distance': avg_distance,
            'floor_distribution': floor_dist
        }

    def _analyze_graph_topology(self, nodes):
        """Analyze graph topology properties"""
        if not nodes:
            return {'avg_degree': 0, 'max_degree': 0, 'clustering_coefficient': 0, 'diameter': 0, 'num_components': 0}

        # Create NetworkX graph
        G = nx.Graph()
        for node in nodes:
            G.add_node(node['id'])
            for conn_id in node['connections']:
                if any(n['id'] == conn_id for n in nodes):  # Ensure connection exists
                    G.add_edge(node['id'], conn_id)

        if G.number_of_nodes() == 0:
            return {'avg_degree': 0, 'max_degree': 0, 'clustering_coefficient': 0, 'diameter': 0, 'num_components': 0}

        # Calculate metrics
        degrees = [G.degree(node) for node in G.nodes()]
        avg_degree = np.mean(degrees) if degrees else 0
        max_degree = max(degrees) if degrees else 0

        # Clustering coefficient
        try:
            clustering = nx.average_clustering(G)
        except:
            clustering = 0

        # Diameter (longest shortest path)
        try:
            if nx.is_connected(G):
                diameter = nx.diameter(G)
            else:
                # For disconnected graphs, use the largest component
                largest_cc = max(nx.connected_components(G), key=len, default=set())
                if len(largest_cc) > 1:
                    diameter = nx.diameter(G.subgraph(largest_cc))
                else:
                    diameter = 0
        except:
            diameter = 0

        # Number of connected components
        num_components = nx.number_connected_components(G)

        return {
            'avg_degree': avg_degree,
            'max_degree': max_degree,
            'clustering_coefficient': clustering,
            'diameter': diameter,
            'num_components': num_components
        }

    def _analyze_room_structure(self, nodes):
        """Analyze room-based structure"""
        centers = [node for node in nodes if node['type'] == 'center']
        entrances = [node for node in nodes if node['type'] == 'entrance']

        if not centers:
            return {'rooms_detected': 0, 'avg_entrances_per_room': 0, 'room_sizes': [], 'room_connectivity': 0}

        # Detect rooms by finding centers and their connected entrances
        rooms = []
        for center in centers:
            center_pos = np.array(center['pos'])

            # Find entrances connected to this center
            connected_entrances = []
            for entrance in entrances:
                if center['id'] in entrance['connections'] or entrance['id'] in center['connections']:
                    connected_entrances.append(entrance)

            if connected_entrances:
                # Estimate room size based on entrance distances from center
                distances = []
                for entrance in connected_entrances:
                    entrance_pos = np.array(entrance['pos'])
                    dist = np.linalg.norm(entrance_pos - center_pos)
                    distances.append(dist)

                avg_distance = np.mean(distances)
                rooms.append({
                    'center': center,
                    'entrances': connected_entrances,
                    'size': avg_distance * 2,  # Approximate room size
                    'connectivity': len([e for e in connected_entrances if len(e['connections']) > 1])
                })

        # Calculate metrics
        avg_entrances = np.mean([len(room['entrances']) for room in rooms]) if rooms else 0
        room_sizes = [room['size'] for room in rooms]
        avg_connectivity = np.mean([room['connectivity'] for room in rooms]) if rooms else 0

        return {
            'rooms_detected': len(rooms),
            'avg_entrances_per_room': avg_entrances,
            'room_sizes': room_sizes,
            'room_connectivity': avg_connectivity
        }

    def _analyze_geometric_patterns(self, positions, nodes):
        """Analyze geometric patterns in the level"""
        if len(positions) < 3:
            return {'linearity_score': 0, 'symmetry_score': 0, 'compactness_score': 0, 'multi_floor_score': 0}

        # Linearity score (how much the layout resembles a line)
        pca = PCA(n_components=min(3, positions.shape[1]))
        pca.fit(positions)
        explained_variance_ratio = pca.explained_variance_ratio_
        linearity = explained_variance_ratio[0] if len(explained_variance_ratio) > 0 else 0

        # Symmetry score (analyze symmetry around center)
        center = np.mean(positions, axis=0)
        centered_positions = positions - center

        # Check reflection symmetry across multiple axes
        symmetry_scores = []
        for axis in range(min(3, positions.shape[1])):
            reflected = centered_positions.copy()
            reflected[:, axis] *= -1

            # Find closest matches
            distances = squareform(pdist(np.vstack([centered_positions, reflected])))
            n = len(centered_positions)
            cross_distances = distances[:n, n:]
            min_distances = np.min(cross_distances, axis=1)
            symmetry_score = np.exp(-np.mean(min_distances))
            symmetry_scores.append(symmetry_score)

        avg_symmetry = np.mean(symmetry_scores)

        # Compactness score (how tightly packed the nodes are)
        if len(positions) > 1:
            distances = pdist(positions)
            avg_distance = np.mean(distances)
            max_distance = np.max(distances)
            compactness = 1 - (avg_distance / (max_distance + 1e-6))
        else:
            compactness = 1

        # Multi-floor score
        floors = positions[:, 1]
        unique_floors = len(np.unique(floors))
        multi_floor = min(1.0, (unique_floors - 1) / 3)  # Normalize to 0-1

        return {
            'linearity_score': linearity,
            'symmetry_score': avg_symmetry,
            'compactness_score': compactness,
            'multi_floor_score': multi_floor
        }


class PatternLearner:
    """Learn patterns from a collection of levels"""

    def __init__(self):
        self.analyzer = LevelAnalyzer()
        self.features_df = None
        self.cluster_labels = None
        self.pattern_prototypes = None

    def analyze_dataset(self, data_dir):
        """Analyze all levels in a directory"""
        print(f"📊 Analyzing levels in {data_dir}...")

        level_features = []
        level_names = []

        txt_files = [f for f in os.listdir(data_dir) if f.endswith('.txt')]

        for filename in txt_files:
            filepath = os.path.join(data_dir, filename)
            try:
                nodes = self.analyzer.load_level(filepath)
                features = self.analyzer.extract_features(nodes)

                # Convert to dict for DataFrame
                feature_dict = {
                    'filename': filename,
                    'num_nodes': features.num_nodes,
                    'num_centers': features.num_centers,
                    'num_entrances': features.num_entrances,
                    'num_corridors': features.num_corridors,
                    'num_connections': features.num_connections,
                    'bounding_box_volume': features.bounding_box_volume,
                    'spatial_spread': features.spatial_spread,
                    'avg_node_distance': features.avg_node_distance,
                    'num_floors': len(features.floor_distribution),
                    'avg_degree': features.avg_degree,
                    'max_degree': features.max_degree,
                    'clustering_coefficient': features.clustering_coefficient,
                    'diameter': features.diameter,
                    'num_components': features.num_components,
                    'rooms_detected': features.rooms_detected,
                    'avg_entrances_per_room': features.avg_entrances_per_room,
                    'avg_room_size': np.mean(features.room_sizes) if features.room_sizes else 0,
                    'room_connectivity': features.room_connectivity,
                    'linearity_score': features.linearity_score,
                    'symmetry_score': features.symmetry_score,
                    'compactness_score': features.compactness_score,
                    'multi_floor_score': features.multi_floor_score
                }

                level_features.append(feature_dict)
                level_names.append(filename)

            except Exception as e:
                print(f"❌ Error analyzing {filename}: {e}")

        self.features_df = pd.DataFrame(level_features)
        print(f"✅ Analyzed {len(self.features_df)} levels")

        return self.features_df

    def discover_patterns(self, n_clusters=None, method='kmeans'):
        """Discover patterns using clustering"""
        if self.features_df is None:
            raise ValueError("Must analyze dataset first")

        print(f"🔍 Discovering patterns using {method}...")

        # Prepare feature matrix (exclude filename and derived features)
        feature_columns = [col for col in self.features_df.columns
                           if col not in ['filename', 'num_floors']]  # num_floors is derived

        X = self.features_df[feature_columns].values

        # Handle any NaN or infinite values
        X = np.nan_to_num(X)

        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Apply clustering
        if method == 'kmeans':
            if n_clusters is None:
                # Find optimal number of clusters using elbow method
                n_clusters = self._find_optimal_clusters(X_scaled)

            clusterer = KMeans(n_clusters=n_clusters, random_state=42)
            labels = clusterer.fit_predict(X_scaled)

        elif method == 'dbscan':
            clusterer = DBSCAN(eps=0.5, min_samples=3)
            labels = clusterer.fit_predict(X_scaled)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

        elif method == 'hierarchical':
            if n_clusters is None:
                n_clusters = 5
            clusterer = AgglomerativeClustering(n_clusters=n_clusters)
            labels = clusterer.fit_predict(X_scaled)

        self.cluster_labels = labels
        self.features_df['cluster'] = labels

        # Calculate silhouette score
        if len(set(labels)) > 1:
            silhouette = silhouette_score(X_scaled, labels)
            print(f"✅ Found {len(set(labels))} patterns (silhouette score: {silhouette:.3f})")
        else:
            print(f"⚠️  Only found 1 pattern")

        return labels

    def _find_optimal_clusters(self, X, max_clusters=10):
        """Find optimal number of clusters using elbow method"""
        inertias = []
        K_range = range(2, min(max_clusters + 1, len(X)))

        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(X)
            inertias.append(kmeans.inertia_)

        # Simple elbow detection
        if len(inertias) > 2:
            diffs = np.diff(inertias)
            second_diffs = np.diff(diffs)
            elbow_idx = np.argmax(second_diffs) + 2  # +2 because we start from k=2
            optimal_k = K_range[elbow_idx] if elbow_idx < len(K_range) else K_range[-1]
        else:
            optimal_k = K_range[0] if K_range else 3

        print(f"💡 Optimal number of clusters: {optimal_k}")
        return optimal_k

    def analyze_patterns(self):
        """Analyze discovered patterns"""
        if self.cluster_labels is None:
            raise ValueError("Must discover patterns first")

        print("\n🎯 Pattern Analysis:")
        print("=" * 50)

        pattern_analysis = {}

        for cluster_id in sorted(set(self.cluster_labels)):
            if cluster_id == -1:  # DBSCAN noise
                continue

            cluster_data = self.features_df[self.features_df['cluster'] == cluster_id]

            print(f"\nPATTERN {cluster_id} ({len(cluster_data)} levels):")
            print("-" * 30)

            # Key characteristics
            key_features = [
                'num_nodes', 'num_centers', 'rooms_detected',
                'linearity_score', 'compactness_score', 'multi_floor_score'
            ]

            analysis = {}
            for feature in key_features:
                if feature in cluster_data.columns:
                    values = cluster_data[feature].values
                    analysis[feature] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'min': np.min(values),
                        'max': np.max(values)
                    }
                    print(f"  {feature}: {np.mean(values):.2f} ± {np.std(values):.2f}")

            # Example files
            example_files = cluster_data['filename'].head(3).tolist()
            print(f"  Examples: {', '.join(example_files)}")

            pattern_analysis[cluster_id] = {
                'size': len(cluster_data),
                'statistics': analysis,
                'examples': example_files
            }

        self.pattern_prototypes = pattern_analysis
        return pattern_analysis

    def visualize_patterns(self, save_dir="pattern_analysis"):
        """Create comprehensive visualizations of discovered patterns"""
        if self.features_df is None:
            raise ValueError("Must analyze dataset first")

        os.makedirs(save_dir, exist_ok=True)

        # 1. Feature correlation heatmap
        self._plot_feature_correlations(save_dir)

        # 2. Cluster visualization in 2D (PCA/t-SNE)
        self._plot_cluster_visualization(save_dir)

        # 3. Pattern characteristics radar chart
        self._plot_pattern_radar_charts(save_dir)

        # 4. Feature distributions by pattern
        self._plot_feature_distributions(save_dir)

        # 5. Interactive 3D scatter plot
        self._plot_interactive_3d(save_dir)

        print(f"📊 Visualizations saved to {save_dir}/")

    def _plot_feature_correlations(self, save_dir):
        """Plot feature correlation heatmap"""
        numeric_cols = self.features_df.select_dtypes(include=[np.number]).columns
        correlation_matrix = self.features_df[numeric_cols].corr()

        plt.figure(figsize=(12, 10))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                    square=True, fmt='.2f', cbar_kws={"shrink": .8})
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'feature_correlations.png'), dpi=300)
        plt.close()

    def _plot_cluster_visualization(self, save_dir):
        """Plot cluster visualization using PCA and t-SNE"""
        if self.cluster_labels is None:
            return

        feature_columns = [col for col in self.features_df.columns
                           if col not in ['filename', 'cluster', 'num_floors']]

        X = self.features_df[feature_columns].values
        X = np.nan_to_num(X)
        X_scaled = StandardScaler().fit_transform(X)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)

        scatter1 = ax1.scatter(X_pca[:, 0], X_pca[:, 1], c=self.cluster_labels,
                               cmap='tab10', alpha=0.7)
        ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        ax1.set_title('PCA Cluster Visualization')
        ax1.grid(True, alpha=0.3)

        # t-SNE (if we have enough samples)
        if len(X_scaled) > 5:
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X_scaled) - 1))
            X_tsne = tsne.fit_transform(X_scaled)

            scatter2 = ax2.scatter(X_tsne[:, 0], X_tsne[:, 1], c=self.cluster_labels,
                                   cmap='tab10', alpha=0.7)
            ax2.set_xlabel('t-SNE 1')
            ax2.set_ylabel('t-SNE 2')
            ax2.set_title('t-SNE Cluster Visualization')
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'cluster_visualization.png'), dpi=300)
        plt.close()

    def _plot_pattern_radar_charts(self, save_dir):
        """Plot radar charts showing pattern characteristics"""
        if self.cluster_labels is None:
            return

        key_features = [
            'linearity_score', 'compactness_score', 'multi_floor_score',
            'symmetry_score', 'clustering_coefficient', 'room_connectivity'
        ]

        # Normalize features to 0-1 scale
        feature_data = self.features_df[key_features].copy()
        for col in key_features:
            if feature_data[col].max() > feature_data[col].min():
                feature_data[col] = (feature_data[col] - feature_data[col].min()) / \
                                    (feature_data[col].max() - feature_data[col].min())

        # Create radar chart for each pattern
        unique_clusters = sorted([c for c in set(self.cluster_labels) if c != -1])
        n_clusters = len(unique_clusters)

        if n_clusters == 0:
            return

        fig, axes = plt.subplots(1, n_clusters, figsize=(5 * n_clusters, 5),
                                 subplot_kw=dict(projection='polar'))

        if n_clusters == 1:
            axes = [axes]

        angles = np.linspace(0, 2 * np.pi, len(key_features), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle

        for i, cluster_id in enumerate(unique_clusters):
            cluster_data = feature_data[self.cluster_labels == cluster_id]
            mean_values = cluster_data.mean().tolist()
            mean_values += mean_values[:1]  # Complete the circle

            axes[i].plot(angles, mean_values, 'o-', linewidth=2, label=f'Pattern {cluster_id}')
            axes[i].fill(angles, mean_values, alpha=0.25)
            axes[i].set_xticks(angles[:-1])
            axes[i].set_xticklabels(key_features)
            axes[i].set_ylim(0, 1)
            axes[i].set_title(f'Pattern {cluster_id}', size=16)
            axes[i].grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'pattern_radar_charts.png'), dpi=300)
        plt.close()

    def _plot_feature_distributions(self, save_dir):
        """Plot feature distributions by pattern"""
        if self.cluster_labels is None:
            return

        key_features = ['num_nodes', 'num_centers', 'linearity_score', 'compactness_score']

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()

        for i, feature in enumerate(key_features):
            for cluster_id in sorted(set(self.cluster_labels)):
                if cluster_id == -1:
                    continue
                cluster_data = self.features_df[self.features_df['cluster'] == cluster_id]
                axes[i].hist(cluster_data[feature], alpha=0.6,
                             label=f'Pattern {cluster_id}', bins=10)

            axes[i].set_xlabel(feature)
            axes[i].set_ylabel('Frequency')
            axes[i].set_title(f'Distribution of {feature}')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'feature_distributions.png'), dpi=300)
        plt.close()

    def _plot_interactive_3d(self, save_dir):
        """Create interactive 3D scatter plot"""
        if self.cluster_labels is None:
            return

        # Use top 3 most variable features
        feature_columns = [col for col in self.features_df.columns
                           if col not in ['filename', 'cluster', 'num_floors']]

        feature_vars = self.features_df[feature_columns].var()
        top_features = feature_vars.nlargest(3).index.tolist()

        fig = go.Figure()

        for cluster_id in sorted(set(self.cluster_labels)):
            if cluster_id == -1:
                continue

            cluster_data = self.features_df[self.features_df['cluster'] == cluster_id]

            fig.add_trace(go.Scatter3d(
                x=cluster_data[top_features[0]],
                y=cluster_data[top_features[1]],
                z=cluster_data[top_features[2]],
                mode='markers',
                marker=dict(size=8, opacity=0.7),
                name=f'Pattern {cluster_id}',
                text=cluster_data['filename'],
                hovertemplate='<b>%{text}</b><br>' +
                              f'{top_features[0]}: %{{x:.2f}}<br>' +
                              f'{top_features[1]}: %{{y:.2f}}<br>' +
                              f'{top_features[2]}: %{{z:.2f}}<extra></extra>'
            ))

        fig.update_layout(
            title='3D Pattern Visualization',
            scene=dict(
                xaxis_title=top_features[0],
                yaxis_title=top_features[1],
                zaxis_title=top_features[2]
            ),
            width=800,
            height=600
        )

        fig.write_html(os.path.join(save_dir, 'interactive_3d_patterns.html'))

    def generate_pattern_rules(self):
        """Generate human-readable rules for each pattern"""
        if self.pattern_prototypes is None:
            raise ValueError("Must analyze patterns first")

        rules = {}

        for pattern_id, pattern_info in self.pattern_prototypes.items():
            rules_text = []
            stats = pattern_info['statistics']

            # Analyze key characteristics
            if 'linearity_score' in stats:
                linearity = stats['linearity_score']['mean']
                if linearity > 0.7:
                    rules_text.append("Highly linear layout (corridor-like)")
                elif linearity < 0.3:
                    rules_text.append("Non-linear layout (clustered or complex)")

            if 'compactness_score' in stats:
                compactness = stats['compactness_score']['mean']
                if compactness > 0.7:
                    rules_text.append("Compact arrangement")
                elif compactness < 0.4:
                    rules_text.append("Spread out arrangement")

            if 'multi_floor_score' in stats:
                multi_floor = stats['multi_floor_score']['mean']
                if multi_floor > 0.5:
                    rules_text.append("Multi-floor structure")
                else:
                    rules_text.append("Single floor structure")

            if 'num_centers' in stats:
                avg_centers = stats['num_centers']['mean']
                if avg_centers > 5:
                    rules_text.append("Many room centers (large level)")
                elif avg_centers < 2:
                    rules_text.append("Few room centers (small level)")

            rules[pattern_id] = {
                'description': "; ".join(rules_text),
                'size': pattern_info['size'],
                'examples': pattern_info['examples']
            }

        return rules


def learn_patterns_from_data(data_dir, output_dir="learned_patterns"):
    """Complete pipeline to learn patterns from architectural data"""

    print("🧠 Learning Architectural Patterns from Data")
    print("=" * 50)

    # Initialize learner
    learner = PatternLearner()

    # Analyze dataset
    features_df = learner.analyze_dataset(data_dir)

    if features_df.empty:
        print("❌ No valid levels found in dataset")
        return None

    # Discover patterns
    labels = learner.discover_patterns(method='kmeans')

    # Analyze patterns
    pattern_analysis = learner.analyze_patterns()

    # Generate rules
    pattern_rules = learner.generate_pattern_rules()

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Save analysis results
    features_df.to_csv(os.path.join(output_dir, 'level_features.csv'), index=False)

    # Save pattern rules
    with open(os.path.join(output_dir, 'pattern_rules.txt'), 'w') as f:
        f.write("LEARNED ARCHITECTURAL PATTERNS\n")
        f.write("=" * 50 + "\n\n")

        for pattern_id, rule_info in pattern_rules.items():
            f.write(f"PATTERN {pattern_id} ({rule_info['size']} levels):\n")
            f.write(f"Description: {rule_info['description']}\n")
            f.write(f"Examples: {', '.join(rule_info['examples'])}\n\n")

    # Create visualizations
    learner.visualize_patterns(output_dir)

    print(f"\n🎉 Pattern learning completed!")
    print(f"📁 Results saved to: {output_dir}/")
    print(f"📊 Found {len(pattern_rules)} distinct patterns")
    print(f"📈 Check visualizations and pattern_rules.txt for details")

    return learner, pattern_rules


# Example usage
if __name__ == "__main__":

    # Example with sample data
    if os.path.exists("sample_data"):
        print("📚 Learning patterns from sample_data...")
        learner, rules = learn_patterns_from_data("sample_data", "learned_patterns")

        if rules:
            print(f"\n🎯 Discovered Pattern Rules:")
            for pattern_id, rule_info in rules.items():
                print(f"Pattern {pattern_id}: {rule_info['description']}")

    # Example with architectural dataset if available
    if os.path.exists("architectural_dataset"):
        print(f"\n📚 Learning patterns from architectural_dataset...")
        learner, rules = learn_patterns_from_data("architectural_dataset", "learned_architectural_patterns")

    else:
        print("💡 To learn from your own data:")
        print("   learner, rules = learn_patterns_from_data('your_data_directory')")