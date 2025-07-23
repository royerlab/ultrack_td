# UlTrack-TD Demo Scripts

This directory contains demo scripts that show how to use the UlTrack-TD Rust extension module for hierarchical graph segmentation.

## Prerequisites

1. **Build the Rust extension module** first:
   ```bash
   maturin develop
   ```
   or
   ```bash
   pip install -e .
   ```

2. **Install Python dependencies** (for the full demo):
   ```bash
   pip install numpy matplotlib networkx
   ```

## Demo Scripts

### 1. `simple_demo.py` - Minimal Demo

A simple demonstration with no external dependencies beyond the built Rust module.

**Usage:**
```bash
python simple_demo.py
```

**What it demonstrates:**
- Basic `hello_rust()` function call
- Hierarchical segmentation on a small graph with 6 nodes
- Different parameter combinations
- Analysis of segmentation results

### 2. `demo.py` - Comprehensive Demo

A full-featured demo with visualization capabilities and multiple test cases.

**Usage:**
```bash
python demo.py
```

**What it demonstrates:**
- Basic functionality testing
- Multiple graph types (structured graphs, grid graphs)
- Different parameter combinations for hierarchical segmentation
- Detailed analysis of results
- Optional visualization (requires matplotlib and networkx)

**Additional dependencies for visualization:**
```bash
pip install matplotlib networkx
```

## Understanding the Hierarchical Segmentation

The `hierarchical_segmentation` function performs graph segmentation using a minimum spanning tree approach with the following parameters:

### Input Parameters

- `edges`: List of tuples `(source, target, weight)` representing graph edges
- `num_nodes`: Total number of nodes in the graph
- `min_frontier`: Minimum edge weight threshold for merging segments
- `min_size`: Minimum size of each segment
- `max_size`: Maximum size of each segment

### Output

Returns a list of integers where each integer represents the segment ID for the corresponding node.

### Algorithm Overview

1. **MST Construction**: Builds a minimum spanning tree from the input graph
2. **Initial Merging**: Merges nodes connected by edges with weight ≤ `min_frontier`, respecting `max_size`
3. **Size Enforcement**: Ensures all segments meet the `min_size` requirement by merging small segments with neighbors

## Example Usage

```python
from ultrack_td._rustlib import hierarchical_segmentation

# Define a simple graph
edges = [
    (0, 1, 0.1),  # Strong connection
    (1, 2, 0.15), # Strong connection  
    (2, 3, 0.8),  # Weak connection
    (3, 4, 0.12), # Strong connection
    (4, 5, 0.2),  # Medium connection
]
num_nodes = 6

# Perform segmentation
labels = hierarchical_segmentation(
    edges=edges,
    num_nodes=num_nodes,
    min_frontier=0.3,  # Only merge edges with weight ≤ 0.3
    min_size=2,        # Each segment must have ≥ 2 nodes
    max_size=4         # Each segment must have ≤ 4 nodes
)

print(f"Segmentation result: {labels}")
# Example output: [0, 0, 0, 1, 1, 1]
# This means nodes 0,1,2 are in segment 0 and nodes 3,4,5 are in segment 1
```

## Use Cases

This hierarchical segmentation is particularly useful for:

- **Image segmentation**: Where pixels are nodes and edge weights represent similarity
- **Region growing**: Starting from seed points and growing regions based on similarity
- **Clustering**: Grouping similar data points with size constraints
- **Medical imaging**: Segmenting anatomical structures with size priors

## Troubleshooting

### ImportError: cannot import name '_rustlib'

This means the Rust extension hasn't been built yet. Run:
```bash
maturin develop
```

### Module not found errors for visualization

Install the optional dependencies:
```bash
pip install matplotlib networkx
```

### Permission errors when building

Try using a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install maturin
maturin develop
```

## Performance Notes

The Rust implementation provides significant performance benefits over pure Python implementations, especially for:
- Large graphs (thousands of nodes)
- Dense connectivity
- Multiple segmentation runs with different parameters

The algorithm complexity is O(E log E) where E is the number of edges, dominated by the minimum spanning tree construction. 