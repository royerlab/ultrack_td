import numpy as np

from ultrack_td._rustlib import (
    compute_connected_components,
    compute_connected_components_2d,
    compute_connected_components_3d,
    create_component_mask_2d,
    create_component_mask_3d,
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


def test_create_component_mask_2d() -> None:
    # Test creating a mask from 2D component pixels
    pixels = [(0, 0), (0, 1), (1, 1), (3, 4)]

    mask_result = create_component_mask_2d(pixels)

    # Check structure
    assert "mask" in mask_result
    assert "bounding_box" in mask_result

    mask = mask_result["mask"]
    bbox = mask_result["bounding_box"]

    # Check bounding box format: [y_start, x_start, y_end, x_end]
    assert len(bbox) == 4
    assert bbox[0] == 0  # y_start
    assert bbox[1] == 0  # x_start
    assert bbox[2] == 4  # y_end (max_y + 1)
    assert bbox[3] == 5  # x_end (max_x + 1)

    # Check mask dimensions (should be (max_y - min_y + 1) x (max_x - min_x + 1))
    assert mask.shape == (4, 5)
    assert mask.dtype == bool

    # Check that the correct pixels are set
    assert mask[0, 0] == True  # (0, 0)
    assert mask[0, 1] == True  # (0, 1)
    assert mask[1, 1] == True  # (1, 1)
    assert mask[3, 4] == True  # (3, 4)

    # Check that other pixels are not set
    assert mask[0, 2] == False
    assert mask[2, 2] == False


def test_create_component_mask_3d() -> None:
    # Test creating a mask from 3D component pixels
    pixels = [(0, 0, 0), (0, 0, 1), (1, 2, 3)]

    mask_result = create_component_mask_3d(pixels)

    # Check structure
    assert "mask" in mask_result
    assert "bounding_box" in mask_result

    mask = mask_result["mask"]
    bbox = mask_result["bounding_box"]

    # Check bounding box format: [z_start, y_start, x_start, z_end, y_end, x_end]
    assert len(bbox) == 6
    assert bbox[0] == 0  # z_start
    assert bbox[1] == 0  # y_start
    assert bbox[2] == 0  # x_start
    assert bbox[3] == 2  # z_end (max_z + 1)
    assert bbox[4] == 3  # y_end (max_y + 1)
    assert bbox[5] == 4  # x_end (max_x + 1)

    # Check mask dimensions
    assert mask.shape == (2, 3, 4)
    assert mask.dtype == bool

    # Check that the correct pixels are set
    assert mask[0, 0, 0] == True  # (0, 0, 0)
    assert mask[0, 0, 1] == True  # (0, 0, 1)
    assert mask[1, 2, 3] == True  # (1, 2, 3)

    # Check that other pixels are not set
    assert mask[0, 1, 0] == False
    assert mask[1, 1, 1] == False


def test_create_component_mask_empty() -> None:
    # Test with empty pixel list
    mask_result_2d = create_component_mask_2d([])
    assert "mask" in mask_result_2d
    assert "bounding_box" in mask_result_2d

    mask_result_3d = create_component_mask_3d([])
    assert "mask" in mask_result_3d
    assert "bounding_box" in mask_result_3d
