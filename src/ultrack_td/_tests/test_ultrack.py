from ultrack_td._rustlib import hierarchical_segmentation


def test_hierarchical_segmentation_simple():
    num_nodes = 6
    edges = [
        (0, 1, 0.1),
        (1, 2, 0.1),
        (0, 2, 0.1),  # cluster 1
        (3, 4, 0.1),
        (4, 5, 0.1),
        (3, 5, 0.1),  # cluster 2
        (2, 3, 1.0),  # bridge
    ]
    min_frontier = 0.5
    min_size = 2
    max_size = 4

    labels = hierarchical_segmentation(
        edges,
        num_nodes=num_nodes,
        min_frontier=min_frontier,
        min_size=min_size,
        max_size=max_size,
    )

    assert len(labels) == num_nodes

    label0 = labels[0]
    assert labels[1] == label0
    assert labels[2] == label0

    label3 = labels[3]
    assert labels[4] == label3
    assert labels[5] == label3

    assert label0 != label3
