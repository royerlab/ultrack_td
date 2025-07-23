#!/usr/bin/env python3
"""
Demo script for ultrack_td Rust extension module.

This script demonstrates how to use the hierarchical segmentation functionality
implemented in Rust and exposed to Python via PyO3.
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from typing import List, Tuple
from skimage import data
from skimage.transform import rescale

try:
    from ultrack_td._rustlib import hello_rust, hierarchical_segmentation
    RUST_MODULE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import Rust module: {e}")
    print("Make sure to build the module with: maturin develop")
    RUST_MODULE_AVAILABLE = False


def create_sample_graph() -> Tuple[List[Tuple[int, int, float]], int]:
    """
    Create a sample graph for testing hierarchical segmentation.
    
    Returns:
        tuple: (edges, num_nodes) where edges is a list of (source, target, weight) tuples
    """
    # Create a simple graph with some structure
    # This represents a small image region with pixel similarities
    edges = [
        # Main cluster 1 (nodes 0-3)
        (0, 1, 0.1),  # Strong connection
        (1, 2, 0.15), # Strong connection
        (2, 3, 0.12), # Strong connection
        (0, 3, 0.2),  # Medium connection
        
        # Main cluster 2 (nodes 4-6)
        (4, 5, 0.08), # Very strong connection
        (5, 6, 0.11), # Strong connection
        (4, 6, 0.18), # Medium connection
        
        # Weak connections between clusters
        (3, 4, 0.8),  # Weak bridge
        (2, 7, 0.9),  # Weak connection to outlier
        
        # Outlier nodes
        (7, 8, 0.3),  # Medium connection
        (8, 9, 0.25), # Medium connection
    ]
    
    num_nodes = 10
    return edges, num_nodes


def create_graph_from_image(scale_factor: float = 0.1) -> Tuple[List[Tuple[int, int, float]], int, np.ndarray]:
    """
    Create a grid graph from a real image using a 4-connectivity.
    Edge weights are the Euclidean distance between pixel intensities.

    Args:
        scale_factor: Factor to scale the image down.

    Returns:
        tuple: (edges, num_nodes, image)
    """
    print("Loading and processing image...")
    image = data.camera()
    # Rescale and convert to float for distance calculation
    image = rescale(image, scale_factor, anti_aliasing=True, mode='reflect')

    if image.dtype != np.float64:
        image = image.astype(np.float64)

    height, width = image.shape
    num_nodes = width * height

    # Generate node indices
    nodes = np.arange(num_nodes).reshape(height, width)

    # Use numpy to create edges and weights efficiently
    # Horizontal edges
    edges_h = np.empty((height, width - 1), dtype=object)
    nodes1_h = nodes[:, :-1]
    nodes2_h = nodes[:, 1:]
    weights_h = np.abs(image[:, :-1] - image[:, 1:])
    
    # Vertical edges
    edges_v = np.empty((height - 1, width), dtype=object)
    nodes1_v = nodes[:-1, :]
    nodes2_v = nodes[1:, :]
    weights_v = np.abs(image[:-1, :] - image[1:, :])

    # Combine edges into a single list of tuples
    edges = list(zip(nodes1_h.ravel(), nodes2_h.ravel(), weights_h.ravel()))
    edges.extend(list(zip(nodes1_v.ravel(), nodes2_v.ravel(), weights_v.ravel())))
    
    return edges, num_nodes, image


def visualize_graph_and_segmentation(edges: List[Tuple[int, int, float]], 
                                   num_nodes: int, 
                                   labels: List[int]):
    """
    Visualize the original graph and the segmentation result.
    
    Args:
        edges: List of edges (source, target, weight)
        num_nodes: Number of nodes
        labels: Segmentation labels for each node
    """
    # Create NetworkX graph for visualization
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    
    for source, target, weight in edges:
        G.add_edge(source, target, weight=weight)
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Original graph
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, ax=ax1, with_labels=True, node_color='lightblue', 
            node_size=500, font_size=10, font_weight='bold')
    
    # Draw edge weights
    edge_labels = {(u, v): f'{d["weight"]:.2f}' for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels, ax=ax1, font_size=8)
    ax1.set_title("Original Graph with Edge Weights")
    
    # Segmented graph
    unique_labels = list(set(labels))
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
    label_to_color = {label: colors[i] for i, label in enumerate(unique_labels)}
    node_colors = [label_to_color[labels[node]] for node in range(num_nodes)]
    
    nx.draw(G, pos, ax=ax2, with_labels=True, node_color=node_colors, 
            node_size=500, font_size=10, font_weight='bold')
    ax2.set_title("Hierarchical Segmentation Result")
    
    # Add legend
    for i, label in enumerate(unique_labels):
        ax2.scatter([], [], c=[colors[i]], label=f'Segment {label}', s=100)
    ax2.legend()
    
    plt.tight_layout()
    plt.show()


def demo_basic_functionality():
    """Demonstrate basic functionality of the Rust module."""
    print("=== Basic Functionality Demo ===")
    
    if not RUST_MODULE_AVAILABLE:
        print("Rust module not available. Skipping demo.")
        return
    
    # Test hello_rust function
    greeting = hello_rust()
    print(f"Greeting from Rust: {greeting}")
    print()


def demo_hierarchical_segmentation():
    """Demonstrate hierarchical segmentation with different parameters."""
    print("=== Hierarchical Segmentation Demo ===")
    
    if not RUST_MODULE_AVAILABLE:
        print("Rust module not available. Skipping demo.")
        return
    
    # Create sample graph
    edges, num_nodes = create_sample_graph()
    print(f"Created graph with {num_nodes} nodes and {len(edges)} edges")
    print("Edges (source, target, weight):")
    for edge in edges:
        print(f"  {edge}")
    print()
    
    # Test different parameter combinations
    test_cases = [
        {"min_frontier": 0.3, "min_size": 1, "max_size": 5, "name": "Moderate threshold"},
        {"min_frontier": 0.15, "min_size": 2, "max_size": 4, "name": "Low threshold, min size 2"},
        {"min_frontier": 0.5, "min_size": 1, "max_size": 3, "name": "High threshold"},
    ]
    
    for i, params in enumerate(test_cases):
        print(f"Test case {i+1}: {params['name']}")
        print(f"Parameters: min_frontier={params['min_frontier']}, "
              f"min_size={params['min_size']}, max_size={params['max_size']}")
        
        try:
            labels = hierarchical_segmentation(
                edges, 
                num_nodes, 
                params['min_frontier'], 
                params['min_size'], 
                params['max_size']
            )
            
            print(f"Segmentation result: {labels}")
            
            # Analyze results
            unique_labels = set(labels)
            print(f"Number of segments: {len(unique_labels)}")
            
            segment_sizes = {}
            for label in labels:
                segment_sizes[label] = segment_sizes.get(label, 0) + 1
            
            print("Segment sizes:")
            for label, size in sorted(segment_sizes.items()):
                print(f"  Segment {label}: {size} nodes")
            
        except Exception as e:
            print(f"Error: {e}")
        
        print("-" * 50)


def demo_grid_segmentation():
    """Demonstrate segmentation on a grid graph (simulating image segmentation)."""
    print("=== Grid Graph Segmentation Demo ===")
    
    if not RUST_MODULE_AVAILABLE:
        print("Rust module not available. Skipping demo.")
        return
    
    # Create a small grid graph
    edges, num_nodes, image = create_graph_from_image(scale_factor=1.0)
    height, width = image.shape
    
    print(f"Created {width}x{height} grid graph with {num_nodes} nodes and {len(edges)} edges")
    
    # Perform segmentation
    min_frontier = 0.001
    min_size = 1000
    max_size = 100000
    
    print(f"Segmentation parameters: min_frontier={min_frontier}, "
          f"min_size={min_size}, max_size={max_size}")
    
    try:
        labels = hierarchical_segmentation(edges, num_nodes, min_frontier, min_size, max_size)
        
        # Reshape labels to image shape
        labels_grid = np.array(labels).reshape(height, width)

        # Visualize the results
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        ax = axes.ravel()

        ax[0].imshow(image, cmap=plt.cm.gray)
        ax[0].set_title('Original Image')
        ax[0].set_axis_off()

        ax[1].imshow(labels_grid, cmap=plt.cm.nipy_spectral)
        ax[1].set_title('Segmentation Result')
        ax[1].set_axis_off()

        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Error during segmentation: {e}")


def demo_image_to_graph():
    """Demonstrates creating a graph from an image and visualizing it."""
    print("=== Image to Graph Demo ===")
    
    edges, num_nodes, image = create_graph_from_image(scale_factor=0.1)
    height, width = image.shape
    
    print(f"Created {width}x{height} graph with {num_nodes} nodes and {len(edges)} edges")

    # For demonstration, let's create dummy labels to use the visualization function
    labels = np.arange(num_nodes)
    
    # We can't show the segmentation, but we can show the original graph structure
    # This part is for visualizing the graph itself, which can be slow for large graphs.
    # We will visualize the image instead.
    
    # Visualize the original image
    plt.figure(figsize=(6, 6))
    plt.imshow(image, cmap=plt.cm.gray)
    plt.title('Original Image')
    plt.axis('off')
    plt.show()

    print("Image to graph conversion demonstrated.")


def main():
    """Main demo function."""
    print("UlTrack-TD Image to Graph Demo")
    print("=" * 50)
    print()
    
   # demo_image_to_graph()

    demo_grid_segmentation()
    
    print("\nSkipping Rust-based hierarchical segmentation demos.")
    
    # demo_basic_functionality()
    # print()
    
    # demo_hierarchical_segmentation()
    # print()
    
    # demo_grid_segmentation()
    # print()


if __name__ == "__main__":
    main() 