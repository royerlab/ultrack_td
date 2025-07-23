#!/usr/bin/env python3
"""
Simple demo script for ultrack_td Rust extension module.

This is a minimal demo that shows basic usage without external dependencies.
"""

try:
    from ultrack_td._rustlib import hello_rust, hierarchical_segmentation
    RUST_MODULE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import Rust module: {e}")
    print("Make sure to build the module with: maturin develop")
    RUST_MODULE_AVAILABLE = False


def main():
    """Simple demo of the Rust functionality."""
    print("Simple UlTrack-TD Demo")
    print("=" * 30)
    
    if not RUST_MODULE_AVAILABLE:
        print("Rust module not available. Please build it first:")
        print("  maturin develop")
        return
    
    # Test basic functionality
    print("1. Testing hello_rust():")
    greeting = hello_rust()
    print(f"   {greeting}")
    print()
    
    # Test hierarchical segmentation
    print("2. Testing hierarchical_segmentation():")
    
    # Create a simple graph
    # This represents connections between 6 nodes with different edge weights
    edges = [
        (0, 1, 0.1),  # Strong connection
        (1, 2, 0.15), # Strong connection
        (2, 3, 0.8),  # Weak connection
        (3, 4, 0.12), # Strong connection
        (4, 5, 0.2),  # Medium connection
        (0, 5, 0.9),  # Weak connection (creates cycle)
    ]
    num_nodes = 6
    
    print(f"   Graph: {num_nodes} nodes, {len(edges)} edges")
    print("   Edges (source, target, weight):")
    for edge in edges:
        print(f"     {edge}")
    
    # Test with different parameters
    test_params = [
        (0.3, 1, 4),  # min_frontier=0.3, min_size=1, max_size=4
        (0.15, 2, 3), # min_frontier=0.15, min_size=2, max_size=3
    ]
    
    for i, (min_frontier, min_size, max_size) in enumerate(test_params):
        print(f"\n   Test {i+1}: min_frontier={min_frontier}, min_size={min_size}, max_size={max_size}")
        
        try:
            labels = hierarchical_segmentation(edges, num_nodes, min_frontier, min_size, max_size)
            print(f"   Result: {labels}")
            
            # Count segments
            unique_labels = set(labels)
            print(f"   Number of segments: {len(unique_labels)}")
            
            # Show which nodes belong to which segment
            segments = {}
            for node_id, label in enumerate(labels):
                if label not in segments:
                    segments[label] = []
                segments[label].append(node_id)
            
            print("   Segments:")
            for label, nodes in sorted(segments.items()):
                print(f"     Segment {label}: nodes {nodes}")
                
        except Exception as e:
            print(f"   Error: {e}")
    
    print("\nDemo completed!")


if __name__ == "__main__":
    main() 