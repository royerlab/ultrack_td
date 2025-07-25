import numpy as np
from ultrack_td import hierarchical_segmentation

def test_hierarchical_segmentation_simple_1():
    num_nodes = 6
    edges = [
        (0, 1, 0.1), (1, 2, 0.1), (0, 2, 0.1),  # cluster 1
        (3, 4, 0.1), (4, 5, 0.1), (3, 5, 0.1),  # cluster 2
        (2, 3, 1.0),  # bridge
    ]
    min_frontier = 0.5
    min_size = 0
    max_size = 10000

    components = hierarchical_segmentation(
        edges,
        num_nodes=num_nodes,
        min_frontier=min_frontier,
        min_size=min_size,
        max_size=max_size,
    )

    assert len(components) == 3

    assert set(components[0]) == {0, 1, 2}
    assert set(components[1]) == {3, 4, 5}
    assert set(components[2]) == {0, 1, 2, 3, 4, 5}

def test_hierarchical_segmentation_simple_2():
    num_nodes = 6
    edges = [
        (0, 1, 0.1), (1, 2, 0.1), (0, 2, 0.1),  # cluster 1
        (3, 4, 0.1), (4, 5, 0.1), (3, 5, 0.1),  # cluster 2
        (2, 3, 1.0),  # bridge
    ]
    min_frontier = 0.0
    min_size = 0
    max_size = 10000

    components = hierarchical_segmentation(
        edges,
        num_nodes=num_nodes,
        min_frontier=min_frontier,
        min_size=min_size,
        max_size=max_size,
    )

    assert len(components) == 11

def test_hierarchical_segmentation_simple_3():
    num_nodes = 6
    edges = [
        (0, 1, 0.1), (1, 2, 0.1), (0, 2, 0.1),  # cluster 1
        (3, 4, 0.1), (4, 5, 0.1), (3, 5, 0.1),  # cluster 2
        (2, 3, 1.0),  # bridge
    ]
    min_frontier = 1.0
    min_size = 0
    max_size = 10000

    components = hierarchical_segmentation(
        edges,
        num_nodes=num_nodes,
        min_frontier=min_frontier,
        min_size=min_size,
        max_size=max_size,
    )

    assert len(components) == 1
    assert set(components[0]) == {0, 1, 2, 3, 4, 5}