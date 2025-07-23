import numpy as np

from ultrack_td._rustlib import (
    compute_connected_components,
    compute_connected_components_2d,
    compute_connected_components_3d,
)


def test_connected_components_2d() -> None:
    # Create a simple 2D binary mask with two separate components
    foreground = np.array(
        [
            [1, 1, 0, 0, 0],
            [1, 0, 0, 1, 1],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0],
        ],
        dtype=bool,
    )

    # Create corresponding contour values
    contours = np.array(
        [
            [1.0, 1.5, 0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0, 2.5, 3.0],
            [0.0, 0.0, 0.0, 1.8, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )

    # Test the 2D function
    components = compute_connected_components_2d(
        foreground=foreground,
        contours=contours,
        min_num_pixels=1,
        max_num_pixels=10,
        min_frontier=None,
    )

    # Should find 2 components
    assert len(components) == 2

    # Check first component (top-left)
    comp1 = components[0]
    assert comp1["area"] == 3  # 3 pixels
    assert "y" in comp1
    assert "x" in comp1
    assert "frontier_score" in comp1
    assert "mean_contour_value" in comp1
    assert "pixels" in comp1
    assert "graph" in comp1

    # Check graph structure
    graph = comp1["graph"]
    assert "nodes" in graph
    assert "edges" in graph

    # Check second component (right side)
    comp2 = components[1]
    assert comp2["area"] == 3  # 3 pixels


def test_connected_components_3d() -> None:
    # Create a simple 3D binary mask
    foreground = np.array([[[1, 0], [0, 1]], [[1, 0], [0, 0]]], dtype=bool)

    # Create corresponding contour values
    contours = np.array([[[1.0, 0.0], [0.0, 2.0]], [[1.5, 0.0], [0.0, 0.0]]])

    # Test the 3D function
    components = compute_connected_components_3d(
        foreground=foreground,
        contours=contours,
        min_num_pixels=1,
        max_num_pixels=10,
        min_frontier=None,
    )

    # Should find 2 components
    assert len(components) == 2

    # Check component attributes exist
    for comp in components:
        assert "area" in comp
        assert "z" in comp
        assert "y" in comp
        assert "x" in comp
        assert "frontier_score" in comp
        assert "mean_contour_value" in comp
        assert "pixels" in comp
        assert "graph" in comp

        # Check graph structure
        graph = comp["graph"]
        assert "nodes" in graph
        assert "edges" in graph


def test_connected_components_generic() -> None:
    # Test the generic function with 2D input
    foreground = np.array([[True, False], [False, True]])
    contours = np.array([[1.0, 0.0], [0.0, 2.0]])

    components = compute_connected_components(
        foreground=foreground,
        contours=contours,
        min_num_pixels=1,
        max_num_pixels=10,
        min_frontier=None,
    )

    assert len(components) == 2
