#!/usr/bin/env python3
"""
Complete Pattern Learning + Hierarchical Generation System

This system:
1. Learns architectural patterns from existing data
2. Applies hierarchical room generation to create new levels matching learned patterns
3. Generates X new levels based on discovered patterns

Usage: python complete_system.py --input_dir your_data --output_dir generated --num_levels 20
"""

import argparse
import os
import numpy as np
import random
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
import json

# Import our previous modules
from pattern_learn import PatternLearner, LevelAnalyzer, LevelFeatures
from hierarchical_generator import (
    LevelLayout, RoomBasedLevelGenerator, LayoutToGraphConverter,
    LevelVisualizer, save_graph_to_file, WallSide
)
from room_pattern_generator import AdvancedRoomPatterns


class PatternBasedGenerator:
    """Generate new levels based on learned patterns"""

    def __init__(self):
        self.pattern_learner = PatternLearner()
        self.room_generator = RoomBasedLevelGenerator()
        self.advanced_patterns = AdvancedRoomPatterns()
        self.converter = LayoutToGraphConverter()
        self.visualizer = LevelVisualizer()

        # Pattern-specific generators
        self.pattern_generators = {
            'linear': self._generate_linear_pattern,
            'compact': self._generate_compact_pattern,
            'multi_floor': self._generate_multi_floor_pattern,
            'complex': self._generate_complex_pattern,
            'symmetric': self._generate_symmetric_pattern,
        }

    def learn_patterns(self, data_dir: str, output_dir: str = "learned_patterns"):
        """Learn patterns from existing data"""
        print("🧠 Learning patterns from existing data...")

        # Analyze dataset and discover patterns
        features_df = self.pattern_learner.analyze_dataset(data_dir)

        if features_df.empty:
            print("❌ No valid levels found in dataset")
            return None

        # Discover patterns
        labels = self.pattern_learner.discover_patterns(method='kmeans')
        pattern_analysis = self.pattern_learner.analyze_patterns()
        pattern_rules = self.pattern_learner.generate_pattern_rules()

        # Save analysis results
        os.makedirs(output_dir, exist_ok=True)
        self.pattern_learner.visualize_patterns(output_dir)

        # Save pattern information for generation
        self._save_pattern_info(pattern_analysis, pattern_rules, output_dir)

        print(f"✅ Learned {len(pattern_rules)} patterns from {len(features_df)} levels")
        return pattern_analysis, pattern_rules

    def _save_pattern_info(self, pattern_analysis: Dict, pattern_rules: Dict, output_dir: str):
        """Save pattern information for later use"""
        pattern_info = {
            'patterns': {}
        }

        for pattern_id, analysis in pattern_analysis.items():
            pattern_info['patterns'][str(pattern_id)] = {
                'size': analysis['size'],
                'statistics': {k: {
                    'mean': float(v['mean']),
                    'std': float(v['std']),
                    'min': float(v['min']),
                    'max': float(v['max'])
                } for k, v in analysis['statistics'].items()},
                'examples': analysis['examples'],
                'rules': pattern_rules.get(pattern_id, {})
            }

        with open(os.path.join(output_dir, 'pattern_info.json'), 'w') as f:
            json.dump(pattern_info, f, indent=2)

    def generate_levels_from_patterns(self, pattern_info_file: str, num_levels: int, output_dir: str):
        """Generate new levels based on learned patterns"""
        print(f" Generating {num_levels} levels based on learned patterns...")

        # Load pattern information
        with open(pattern_info_file, 'r') as f:
            pattern_info = json.load(f)

        patterns = pattern_info['patterns']

        if not patterns:
            print(" No patterns found in pattern info file")
            return []

        os.makedirs(output_dir, exist_ok=True)
        generated_levels = []

        # Generate levels distributed across patterns
        pattern_ids = list(patterns.keys())
        pattern_weights = [patterns[pid]['size'] for pid in pattern_ids]  # Weight by original frequency

        for i in range(num_levels):
            print(f"\n Generating level {i + 1}/{num_levels}...")

            # Choose pattern based on original frequency
            pattern_id = np.random.choice(pattern_ids, p=np.array(pattern_weights) / sum(pattern_weights))
            pattern_data = patterns[pattern_id]

            try:
                # Generate level based on pattern characteristics
                layout = self._generate_level_from_pattern(pattern_id, pattern_data)

                # Convert to graph format
                graph_nodes = self.converter.convert_layout_to_graph(layout)

                # Create filenames
                base_name = f"generated_pattern_{pattern_id}_{i + 1:03d}"

                # Save level
                level_file = os.path.join(output_dir, f"{base_name}.txt")
                save_graph_to_file(graph_nodes, level_file)

                # Create visualizations
                layout_vis = os.path.join(output_dir, f"{base_name}_layout.png")
                graph_vis = os.path.join(output_dir, f"{base_name}_graph.png")

                self.visualizer.visualize_layout(
                    layout, f"Generated Level {i + 1} (Pattern {pattern_id})", layout_vis
                )
                self.visualizer.visualize_graph(
                    graph_nodes, f"Graph {i + 1} (Pattern {pattern_id})", graph_vis
                )

                generated_levels.append({
                    'id': i + 1,
                    'pattern_id': pattern_id,
                    'layout': layout,
                    'graph': graph_nodes,
                    'files': {
                        'level': level_file,
                        'layout_vis': layout_vis,
                        'graph_vis': graph_vis
                    }
                })

                print(
                    f"  Generated level {i + 1}: {len(layout.rooms)} rooms, {len(graph_nodes)} nodes (Pattern {pattern_id})")

            except Exception as e:
                print(f"  Error generating level {i + 1}: {e}")

        # Create generation summary
        self._create_generation_summary(generated_levels, output_dir)

        print(f"\n🎉 Generated {len(generated_levels)} levels successfully!")
        return generated_levels

    def _generate_level_from_pattern(self, pattern_id: str, pattern_data: Dict) -> LevelLayout:
        """Generate a level that matches a specific learned pattern"""
        stats = pattern_data['statistics']

        # Extract key parameters from pattern statistics
        target_rooms = max(1, int(stats.get('num_centers', {}).get('mean', 4)))
        target_nodes = max(5, int(stats.get('num_nodes', {}).get('mean', 15)))
        linearity = stats.get('linearity_score', {}).get('mean', 0.5)
        compactness = stats.get('compactness_score', {}).get('mean', 0.5)
        multi_floor = stats.get('multi_floor_score', {}).get('mean', 0.0)
        symmetry = stats.get('symmetry_score', {}).get('mean', 0.3)

        # Determine number of floors
        num_floors = 1
        if multi_floor > 0.3:
            num_floors = max(1, min(3, int(multi_floor * 4)))

        # Choose generation strategy based on pattern characteristics
        if linearity > 0.7:
            # High linearity -> Linear corridor pattern
            layout = self._generate_linear_pattern(target_rooms, num_floors)
        elif compactness > 0.7 and target_rooms <= 6:
            # High compactness + few rooms -> Compact/cross pattern
            layout = self._generate_compact_pattern(target_rooms, num_floors)
        elif multi_floor > 0.5:
            # Multi-floor -> Tower pattern
            layout = self._generate_multi_floor_pattern(target_rooms, num_floors)
        elif symmetry > 0.5:
            # High symmetry -> Symmetric pattern
            layout = self._generate_symmetric_pattern(target_rooms, num_floors)
        else:
            # Complex/irregular pattern
            layout = self._generate_complex_pattern(target_rooms, num_floors)

        return layout

    def _generate_linear_pattern(self, num_rooms: int, num_floors: int) -> LevelLayout:
        """Generate linear corridor-like pattern"""
        return self.advanced_patterns.generate_linear_corridor(
            num_rooms=num_rooms,
            floors=num_floors
        )

    def _generate_compact_pattern(self, num_rooms: int, num_floors: int) -> LevelLayout:
        """Generate compact cross-like pattern"""
        if num_rooms <= 5:
            arm_length = max(1, (num_rooms - 1) // 4)
            return self.advanced_patterns.generate_cross_pattern(arm_length=arm_length)
        else:
            # Use spiral for larger compact layouts
            spiral_size = max(3, int(np.sqrt(num_rooms)))
            return self.advanced_patterns.generate_spiral_pattern(spiral_size=spiral_size)

    def _generate_multi_floor_pattern(self, num_rooms: int, num_floors: int) -> LevelLayout:
        """Generate multi-floor tower pattern"""
        rooms_per_floor = max(3, num_rooms // num_floors)
        return self.advanced_patterns.generate_tower_floors(
            rooms_per_floor=rooms_per_floor,
            num_floors=num_floors
        )

    def _generate_complex_pattern(self, num_rooms: int, num_floors: int) -> LevelLayout:
        """Generate complex irregular pattern"""
        if num_floors > 1:
            return self.room_generator.generate_complex_layout(
                num_rooms=num_rooms,
                floors=num_floors
            )
        else:
            # Use maze-like pattern for single floor complex layouts
            grid_size = max(3, int(np.sqrt(num_rooms * 1.2)))
            return self.advanced_patterns.generate_maze_like(grid_size=grid_size)

    def _generate_symmetric_pattern(self, num_rooms: int, num_floors: int) -> LevelLayout:
        """Generate symmetric pattern"""
        if num_rooms <= 13:  # Cross pattern works well for symmetric layouts
            arm_length = max(2, min(4, (num_rooms - 1) // 4))
            return self.advanced_patterns.generate_cross_pattern(arm_length=arm_length)
        else:
            # Use tower pattern for larger symmetric layouts
            rooms_per_floor = max(4, min(8, num_rooms // max(1, num_floors)))
            return self.advanced_patterns.generate_tower_floors(
                rooms_per_floor=rooms_per_floor,
                num_floors=max(1, num_floors)
            )

    def _create_generation_summary(self, generated_levels: List[Dict], output_dir: str):
        """Create a summary of generated levels"""
        summary_file = os.path.join(output_dir, "generation_summary.txt")

        # Analyze generation statistics
        pattern_counts = Counter(level['pattern_id'] for level in generated_levels)
        room_counts = [len(level['layout'].rooms) for level in generated_levels]
        node_counts = [len(level['graph']) for level in generated_levels]

        with open(summary_file, 'w') as f:
            f.write("LEVEL GENERATION SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Total levels generated: {len(generated_levels)}\n\n")

            f.write("PATTERN DISTRIBUTION:\n")
            for pattern_id, count in pattern_counts.most_common():
                f.write(f"  Pattern {pattern_id}: {count} levels\n")

            f.write(f"\nSTATISTICS:\n")
            f.write(f"  Rooms per level: avg={np.mean(room_counts):.1f}, range={min(room_counts)}-{max(room_counts)}\n")
            f.write(f"  Nodes per level: avg={np.mean(node_counts):.1f}, range={min(node_counts)}-{max(node_counts)}\n")

            f.write(f"\nGENERATED FILES:\n")
            for level in generated_levels[:10]:  # Show first 10 as examples
                f.write(f"  Level {level['id']}: {os.path.basename(level['files']['level'])}\n")
            if len(generated_levels) > 10:
                f.write(f"  ... and {len(generated_levels) - 10} more\n")

        print(f"📊 Generation summary saved to: {summary_file}")


class CompleteLevelSystem:
    """Complete system that learns and generates"""

    def __init__(self):
        self.generator = PatternBasedGenerator()

    def run_complete_pipeline(self, input_dir: str, output_dir: str, num_levels: int,
                              learn_patterns: bool = True):
        """Run the complete pipeline: learn patterns -> generate levels"""

        print("🚀 Running Complete Level Generation Pipeline")
        print("=" * 60)
        print(f"Input directory: {input_dir}")
        print(f"Output directory: {output_dir}")
        print(f"Levels to generate: {num_levels}")
        print(f"Learn patterns: {learn_patterns}")
        print()

        # Create main output directory
        os.makedirs(output_dir, exist_ok=True)

        pattern_info_file = None

        # Step 1: Learn patterns (if requested)
        if learn_patterns:
            print("STEP 1: LEARNING PATTERNS")
            print("-" * 30)

            if not os.path.exists(input_dir):
                print(f" Input directory not found: {input_dir}")
                return None

            learn_dir = os.path.join(output_dir, "learned_patterns")
            pattern_analysis, pattern_rules = self.generator.learn_patterns(input_dir, learn_dir)

            if pattern_analysis is None:
                print(" Pattern learning failed")
                return None

            pattern_info_file = os.path.join(learn_dir, "pattern_info.json")

            print(f"✅ Pattern learning completed")
            print(f"📁 Pattern analysis saved to: {learn_dir}")

        else:
            # Look for existing pattern info
            pattern_info_file = os.path.join(output_dir, "learned_patterns", "pattern_info.json")
            if not os.path.exists(pattern_info_file):
                print(" No existing pattern info found. Run with --learn-patterns first.")
                return None

        print("\nSTEP 2: GENERATING LEVELS")
        print("-" * 30)

        # Step 2: Generate levels based on patterns
        generation_dir = os.path.join(output_dir, "generated_levels")
        generated_levels = self.generator.generate_levels_from_patterns(
            pattern_info_file, num_levels, generation_dir
        )

        if not generated_levels:
            print(" Level generation failed")
            return None

        print(f"✅ Level generation completed")
        print(f"📁 Generated levels saved to: {generation_dir}")

        # Step 3: Create final summary
        self._create_final_summary(input_dir, output_dir, generated_levels)

        print("\n🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        print(f"📊 Check {output_dir}/ for all results")

        return generated_levels

    def _create_final_summary(self, input_dir: str, output_dir: str, generated_levels: List[Dict]):
        """Create final pipeline summary"""
        summary_file = os.path.join(output_dir, "pipeline_summary.txt")

        with open(summary_file, 'w') as f:
            f.write("COMPLETE LEVEL GENERATION PIPELINE SUMMARY\n")
            f.write("=" * 60 + "\n\n")

            f.write(f"INPUT:\n")
            f.write(f"  Source directory: {input_dir}\n")
            if os.path.exists(input_dir):
                txt_files = [f for f in os.listdir(input_dir) if f.endswith('.txt')]
                f.write(f"  Source levels: {len(txt_files)}\n")

            f.write(f"\nOUTPUT:\n")
            f.write(f"  Generated levels: {len(generated_levels)}\n")
            f.write(f"  Output directory: {output_dir}\n")

            f.write(f"\nDIRECTORY STRUCTURE:\n")
            f.write(f"  {output_dir}/\n")
            f.write(f"  ├── learned_patterns/          # Pattern analysis results\n")
            f.write(f"  │   ├── pattern_info.json      # Pattern parameters\n")
            f.write(f"  │   ├── *.png                  # Pattern visualizations\n")
            f.write(f"  │   └── level_features.csv     # Extracted features\n")
            f.write(f"  ├── generated_levels/          # Generated level files\n")
            f.write(f"  │   ├── *.txt                  # Level data files\n")
            f.write(f"  │   ├── *_layout.png           # Room layout visualizations\n")
            f.write(f"  │   ├── *_graph.png            # Graph visualizations\n")
            f.write(f"  │   └── generation_summary.txt # Generation statistics\n")
            f.write(f"  └── pipeline_summary.txt       # This summary\n")

            # Pattern distribution
            pattern_counts = Counter(level['pattern_id'] for level in generated_levels)
            f.write(f"\nPATTERN DISTRIBUTION:\n")
            for pattern_id, count in pattern_counts.most_common():
                f.write(f"  Pattern {pattern_id}: {count} levels ({count / len(generated_levels) * 100:.1f}%)\n")


def main():
    """Command line interface"""
    parser = argparse.ArgumentParser(description='Complete Pattern Learning + Level Generation System')

    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing existing level files to learn from')
    parser.add_argument('--output_dir', type=str, default='complete_generation_output',
                        help='Directory to save all results (default: complete_generation_output)')
    parser.add_argument('--num_levels', type=int, default=10,
                        help='Number of levels to generate (default: 10)')
    parser.add_argument('--skip_learning', action='store_true',
                        help='Skip pattern learning (use existing pattern_info.json)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducible generation (default: 42)')

    args = parser.parse_args()

    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Run complete pipeline
    system = CompleteLevelSystem()
    generated_levels = system.run_complete_pipeline(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        num_levels=args.num_levels,
        learn_patterns=not args.skip_learning
    )

    if generated_levels:
        print(f"\n✨ SUCCESS: Generated {len(generated_levels)} levels")
        print(f"🎯 Results available in: {args.output_dir}/")

        # Show some examples
        print(f"\n📝 Example generated files:")
        for level in generated_levels[:3]:
            print(f"  • {os.path.basename(level['files']['level'])}")
        if len(generated_levels) > 3:
            print(f"  • ... and {len(generated_levels) - 3} more")
    else:
        print(f"\n❌ FAILED: No levels were generated")
        return 1

    return 0


# Alternative simple usage functions
def quick_generate(input_dir: str, num_levels: int = 10, output_dir: str = "quick_output"):
    """Quick generation function for interactive use"""
    system = CompleteLevelSystem()
    return system.run_complete_pipeline(input_dir, output_dir, num_levels, True)


def generate_from_existing_patterns(pattern_info_file: str, num_levels: int = 10,
                                    output_dir: str = "pattern_based_output"):
    """Generate levels from existing pattern analysis"""
    generator = PatternBasedGenerator()
    return generator.generate_levels_from_patterns(pattern_info_file, num_levels, output_dir)


# Example usage
if __name__ == "__main__":

    system = CompleteLevelSystem()
    levels = system.run_complete_pipeline(
        input_dir="/Users/michaelkolomenkin/Data/playo/files_for_yaron/",
        output_dir="/Users/michaelkolomenkin/Data/playo/levels/",
        num_levels=25,
        learn_patterns=True
    )


    # Command line usage
    # if len(os.sys.argv) > 1:
    #     exit(main())
    #
    # # Interactive usage examples
    # print(" Interactive Usage Examples:")
    # print("-" * 40)
    #
    # # Example 1: Quick generation
    # if os.path.exists("sample_data"):
    #     print("\n Example 1: Quick generation from sample_data")
    #     try:
    #         levels = quick_generate("sample_data", num_levels=5, output_dir="example_output")
    #         print(f" Generated {len(levels) if levels else 0} levels")
    #     except Exception as e:
    #         print(f" Error: {e}")
    #
    # # Example 2: Complete pipeline
    # print(f"\n Example 2: Complete pipeline usage")
    # print(f"Command line:")
    # print(f"  python {__file__} --input_dir your_data --num_levels 20 --output_dir my_output")
    #
    # print(f"\nProgrammatic:")
    # print(f"  system = CompleteLevelSystem()")
    # print(f"  levels = system.run_complete_pipeline('your_data', 'my_output', 20)")
    #
    # # Example 3: Pattern-only generation
    # print(f"\n️ Example 3: Generate from existing patterns")
    # print(f"  levels = generate_from_existing_patterns('learned_patterns/pattern_info.json', 15)")
    #
    # print(f"\n For help: python {__file__} --help")