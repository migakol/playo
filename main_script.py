#!/usr/bin/env python3
"""
Complete script to train graph diffusion model and generate game levels
Usage: python train_and_generate.py --data_dir your_data_directory
"""

import argparse
import os
import sys
from graph_diffusion_levels import GraphLevelGenerator, create_sample_data
from level_generation_usage import generate_new_levels, analyze_level_statistics, LevelPostProcessor, LevelVisualizer


def main():
    parser = argparse.ArgumentParser(description='Train graph diffusion model for game level generation')
    parser.add_argument('--data_dir', type=str, default='/Users/michaelkolomenkin/Data/playo/files_for_yaron',
                        help='Directory containing level .txt files')
    parser.add_argument('--output_dir', type=str, default='/Users/michaelkolomenkin/Data/playo/levels',
                        help='Directory to save generated levels')
    parser.add_argument('--model_dir', type=str, default='/Users/michaelkolomenkin/Data/playo/models',
                        help='Directory to save/load models')
    parser.add_argument('--num_generate', type=int, default=10,
                        help='Number of levels to generate')
    parser.add_argument('--vae_epochs', type=int, default=100,
                        help='Number of VAE training epochs')
    parser.add_argument('--diffusion_epochs', type=int, default=200,
                        help='Number of diffusion training epochs')
    parser.add_argument('--create_sample', action='store_true',
                        help='Create sample data for testing')
    parser.add_argument('--skip_training', action='store_true',
                        help='Skip training and only generate (requires trained models)')

    args = parser.parse_args()

    # Create sample data if requested
    if args.create_sample:
        print("Creating sample data...")
        create_sample_data(num_graphs=100, save_dir=args.data_dir)
        print(f"Sample data created in {args.data_dir}/")

    # Check if data directory exists
    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory {args.data_dir} not found!")
        print("Use --create_sample to generate sample data, or specify existing directory with --data_dir")
        return

    # Analyze existing data
    print(f"Analyzing data in {args.data_dir}...")
    analyze_level_statistics(args.data_dir)

    if not args.skip_training:
        print("\n" + "=" * 50)
        print("TRAINING PHASE")
        print("=" * 50)

        # Initialize generator
        generator = GraphLevelGenerator()

        # Load and process data
        print("Loading graphs...")
        graphs = generator.data_processor.load_dataset(args.data_dir)
        print(f"Loaded {len(graphs)} graphs")

        if len(graphs) == 0:
            print("Error: No valid graphs found in data directory!")
            return

        # Train VAE
        print(f"\nTraining VAE for {args.vae_epochs} epochs...")
        vae_losses = generator.train_vae(graphs, epochs=args.vae_epochs)
        print("VAE training completed!")

        # Extract embeddings
        print("Extracting graph embeddings...")
        embeddings = generator.extract_graph_embeddings(graphs)
        print(f"Extracted embeddings shape: {embeddings.shape}")

        # Train diffusion model
        print(f"\nTraining diffusion model for {args.diffusion_epochs} epochs...")
        diffusion_losses = generator.train_diffusion(embeddings, epochs=args.diffusion_epochs)
        print("Diffusion training completed!")

        # Save models
        model_prefix = os.path.join(args.model_dir, "graph_diffusion")
        generator.save_models(model_prefix)

        # Plot training curves
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(vae_losses)
        plt.title('VAE Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(diffusion_losses)
        plt.title('Diffusion Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)

        plt.tight_layout()
        loss_plot_path = os.path.join(args.output_dir, 'training_losses.png')
        os.makedirs(args.output_dir, exist_ok=True)
        plt.savefig(loss_plot_path)
        print(f"Training loss plots saved to {loss_plot_path}")
        plt.show()

    print("\n" + "=" * 50)
    print("GENERATION PHASE")
    print("=" * 50)

    # Generate new levels
    model_prefix = os.path.join(args.model_dir, "graph_diffusion")
    generate_new_levels(
        model_path_prefix=model_prefix,
        num_levels=args.num_generate,
        output_dir=args.output_dir
    )

    print("\n" + "=" * 50)
    print("COMPLETION")
    print("=" * 50)
    print(f"✓ Generated {args.num_generate} new levels")
    print(f"✓ Output saved to {args.output_dir}/")
    print(f"✓ Models saved to {args.model_dir}/")

    # Show example of generated level
    print("\nExample of generated level format:")
    example_file = os.path.join(args.output_dir, "generated_level_000.txt")
    if os.path.exists(example_file):
        with open(example_file, 'r') as f:
            lines = f.readlines()[:10]  # Show first 10 lines
            for line in lines:
                print(line.strip())
        if len(lines) >= 10:
            print("...")


def generation():
    generator = GraphLevelGenerator()
    generator.load_models('/Users/michaelkolomenkin/Code/playo/models/graph_diffusion')
    print("Generating new graphs...")
    new_graphs = generator.generate_graphs(num_graphs=10)

    post_processor = LevelPostProcessor()
    visualizer = LevelVisualizer()

    # Generate raw graphs
    new_graphs = generator.generate_graphs(num_graphs=10)

    # Convert and save each graph
    import os
    os.makedirs("my_generated_levels", exist_ok=True)

    for i, raw_graph in enumerate(new_graphs):
        print(f"Processing level {i + 1}/{len(new_graphs)}")

        # Convert raw embedding to structured level
        nodes = post_processor.embedding_to_level(raw_graph, num_nodes=25)

        # Save in your text format
        filename = f"my_generated_levels/level_{i:03d}.txt"
        post_processor.save_level_to_file(nodes, filename)

        # Optional: Create visualization
        visualizer.visualize_level(
            nodes,
            title=f"Generated Level {i + 1}",
            save_path=f"my_generated_levels/level_{i:03d}_visual.png"
        )

        print(f"✅ Saved: {filename}")

    print(f"🎉 All {len(new_graphs)} levels saved to my_generated_levels/")


if __name__ == "__main__":
    # main()
    generation()