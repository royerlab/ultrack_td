use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2, PyReadonlyArray3};
use petgraph::graph::UnGraph;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::collections::{HashMap, VecDeque};

#[pyfunction]
pub fn compute_connected_components_2d<'py>(
    py: Python<'py>,
    foreground: PyReadonlyArray2<bool>,
    contours: PyReadonlyArray2<f64>,
    min_num_pixels: usize,
    max_num_pixels: usize,
    min_frontier: Option<f64>,
) -> PyResult<PyObject> {
    let foreground = foreground.as_array();
    let contours = contours.as_array();

    if foreground.shape() != contours.shape() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Foreground and contours arrays must have the same shape",
        ));
    }

    let components = find_components_2d(
        &foreground,
        &contours,
        min_num_pixels,
        max_num_pixels,
        min_frontier,
    );
    components_to_python_dict_2d(py, components)
}

#[pyfunction]
pub fn compute_connected_components_3d<'py>(
    py: Python<'py>,
    foreground: PyReadonlyArray3<bool>,
    contours: PyReadonlyArray3<f64>,
    min_num_pixels: usize,
    max_num_pixels: usize,
    min_frontier: Option<f64>,
) -> PyResult<PyObject> {
    let foreground = foreground.as_array();
    let contours = contours.as_array();

    if foreground.shape() != contours.shape() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Foreground and contours arrays must have the same shape",
        ));
    }

    let components = find_components_3d(
        &foreground,
        &contours,
        min_num_pixels,
        max_num_pixels,
        min_frontier,
    );
    components_to_python_dict_3d(py, components)
}

#[pyfunction]
pub fn compute_connected_components<'py>(
    py: Python<'py>,
    foreground: &Bound<'py, PyAny>,
    contours: &Bound<'py, PyAny>,
    min_num_pixels: usize,
    max_num_pixels: usize,
    min_frontier: Option<f64>,
) -> PyResult<PyObject> {
    // Try to extract as 2D arrays first
    if let (Ok(fg_2d), Ok(cont_2d)) = (
        foreground.extract::<PyReadonlyArray2<bool>>(),
        contours.extract::<PyReadonlyArray2<f64>>(),
    ) {
        return compute_connected_components_2d(
            py,
            fg_2d,
            cont_2d,
            min_num_pixels,
            max_num_pixels,
            min_frontier,
        );
    }

    // Try to extract as 3D arrays
    if let (Ok(fg_3d), Ok(cont_3d)) = (
        foreground.extract::<PyReadonlyArray3<bool>>(),
        contours.extract::<PyReadonlyArray3<f64>>(),
    ) {
        return compute_connected_components_3d(
            py,
            fg_3d,
            cont_3d,
            min_num_pixels,
            max_num_pixels,
            min_frontier,
        );
    }

    Err(pyo3::exceptions::PyValueError::new_err(
        "Arrays must be 2D or 3D numpy arrays",
    ))
}

#[derive(Debug)]
struct Component2D {
    pixels: Vec<(usize, usize)>,
    centroid: (f64, f64),
    frontier_score: f64,
    mean_contour_value: f64,
    mask: ndarray::Array2<bool>,
    bbox: ndarray::Array1<usize>,
}

impl Component2D {
    fn new(pixels: Vec<(usize, usize)>, frontier_score: f64, mean_contour_value: f64) -> Self {
        if pixels.is_empty() {
            panic!("Cannot create component with empty pixels");
        }

        // Single pass to compute bounds and centroid
        let mut y_min = usize::MAX;
        let mut y_max = usize::MIN;
        let mut x_min = usize::MAX;
        let mut x_max = usize::MIN;
        let mut y_sum = 0;
        let mut x_sum = 0;

        for &(y, x) in &pixels {
            y_min = y_min.min(y);
            y_max = y_max.max(y);
            x_min = x_min.min(x);
            x_max = x_max.max(x);
            y_sum += y;
            x_sum += x;
        }

        let centroid = (
            y_sum as f64 / pixels.len() as f64,
            x_sum as f64 / pixels.len() as f64,
        );

        // Create mask
        let y_size = y_max - y_min + 1;
        let x_size = x_max - x_min + 1;
        let mut mask = ndarray::Array2::from_elem((y_size, x_size), false);

        for &(y, x) in &pixels {
            mask[[y - y_min, x - x_min]] = true;
        }

        let bbox = ndarray::Array1::from_vec(vec![y_min, x_min, y_max, x_max]);

        Component2D {
            pixels,
            centroid,
            frontier_score,
            mean_contour_value,
            mask,
            bbox,
        }
    }
}

#[derive(Debug)]
struct Component3D {
    pixels: Vec<(usize, usize, usize)>,
    centroid: (f64, f64, f64),
    frontier_score: f64,
    mean_contour_value: f64,
    mask: ndarray::Array3<bool>,
    bbox: ndarray::Array1<usize>,
}

impl Component3D {
    fn new(
        pixels: Vec<(usize, usize, usize)>,
        frontier_score: f64,
        mean_contour_value: f64,
    ) -> Self {
        if pixels.is_empty() {
            panic!("Cannot create component with empty pixels");
        }

        // Single pass to compute bounds and centroid
        let mut z_min = usize::MAX;
        let mut z_max = usize::MIN;
        let mut y_min = usize::MAX;
        let mut y_max = usize::MIN;
        let mut x_min = usize::MAX;
        let mut x_max = usize::MIN;
        let mut z_sum = 0;
        let mut y_sum = 0;
        let mut x_sum = 0;

        for &(z, y, x) in &pixels {
            z_min = z_min.min(z);
            z_max = z_max.max(z);
            y_min = y_min.min(y);
            y_max = y_max.max(y);
            x_min = x_min.min(x);
            x_max = x_max.max(x);
            z_sum += z;
            y_sum += y;
            x_sum += x;
        }

        let centroid = (
            z_sum as f64 / pixels.len() as f64,
            y_sum as f64 / pixels.len() as f64,
            x_sum as f64 / pixels.len() as f64,
        );

        // Create mask
        let z_size = z_max - z_min + 1;
        let y_size = y_max - y_min + 1;
        let x_size = x_max - x_min + 1;
        let mut mask = ndarray::Array3::from_elem((z_size, y_size, x_size), false);

        for &(z, y, x) in &pixels {
            mask[[z - z_min, y - y_min, x - x_min]] = true;
        }

        let bbox = ndarray::Array1::from_vec(vec![z_min, y_min, x_min, z_max, y_max, x_max]);

        Component3D {
            pixels,
            centroid,
            frontier_score,
            mean_contour_value,
            mask,
            bbox,
        }
    }
}

fn find_components_2d(
    foreground: &ndarray::ArrayView2<bool>,
    contours: &ndarray::ArrayView2<f64>,
    min_num_pixels: usize,
    max_num_pixels: usize,
    min_frontier: Option<f64>,
) -> Vec<Component2D> {
    let (height, width) = (foreground.shape()[0], foreground.shape()[1]);
    let mut visited = vec![vec![false; width]; height];
    let mut candidate_components = Vec::new();

    for i in 0..height {
        for j in 0..width {
            if foreground[[i, j]] && !visited[i][j] {
                let components = flood_fill_2d(
                    foreground,
                    contours,
                    &mut visited,
                    i,
                    j,
                    height,
                    width,
                    min_num_pixels,
                    max_num_pixels,
                    min_frontier,
                );

                for component in components {
                    candidate_components.push(component);
                }
            }
        }
    }

    candidate_components
}

fn find_components_3d(
    foreground: &ndarray::ArrayView3<bool>,
    contours: &ndarray::ArrayView3<f64>,
    min_num_pixels: usize,
    max_num_pixels: usize,
    min_frontier: Option<f64>,
) -> Vec<Component3D> {
    let (depth, height, width) = (
        foreground.shape()[0],
        foreground.shape()[1],
        foreground.shape()[2],
    );
    let mut visited = vec![vec![vec![false; width]; height]; depth];
    let mut candidate_components = Vec::new();

    for k in 0..depth {
        for i in 0..height {
            for j in 0..width {
                if foreground[[k, i, j]] && !visited[k][i][j] {
                    let components = flood_fill_3d(
                        foreground,
                        contours,
                        &mut visited,
                        k,
                        i,
                        j,
                        depth,
                        height,
                        width,
                        min_num_pixels,
                        max_num_pixels,
                        min_frontier,
                    );

                    for component in components {
                        candidate_components.push(component);
                    }
                }
            }
        }
    }

    candidate_components
}

fn flood_fill_2d(
    foreground: &ndarray::ArrayView2<bool>,
    contours: &ndarray::ArrayView2<f64>,
    visited: &mut Vec<Vec<bool>>,
    start_i: usize,
    start_j: usize,
    height: usize,
    width: usize,
    min_num_pixels: usize,
    max_num_pixels: usize,
    min_frontier: Option<f64>,
) -> Vec<Component2D> {
    let mut queue = VecDeque::new();
    let mut pixels = Vec::new();
    let mut contour_sum = 0.0;
    let mut boundary_pixels = 0;

    queue.push_back((start_i, start_j));
    visited[start_i][start_j] = true;

    // 4-connectivity for 2D
    let directions = [(-1, 0), (0, -1), (0, 1), (1, 0)];

    while let Some((i, j)) = queue.pop_front() {
        pixels.push((i, j));
        contour_sum += contours[[i, j]];

        let mut is_boundary = false;
        for (di, dj) in &directions {
            let ni = i as i32 + di;
            let nj = j as i32 + dj;

            if ni >= 0 && ni < height as i32 && nj >= 0 && nj < width as i32 {
                let ni = ni as usize;
                let nj = nj as usize;

                if foreground[[ni, nj]] {
                    if !visited[ni][nj] {
                        visited[ni][nj] = true;
                        queue.push_back((ni, nj));
                    }
                } else {
                    is_boundary = true;
                }
            } else {
                is_boundary = true;
            }
        }

        if is_boundary {
            boundary_pixels += 1;
        }
    }

    // TODO: you will compute these values per segment
    let mean_contour_value = contour_sum / pixels.len() as f64;
    let frontier_score = boundary_pixels as f64 / pixels.len() as f64;

    // Build graph with flattened node indices
    let (graph, node_to_pixel) = build_flattened_graph_2d(&pixels, contours);

    // TODO: to be replaced by proper segmentation function
    // use min_num_pixels, max_num_pixels and min_frontier to filter candidate components
    let candidate_node_ids = vec![graph.node_indices().collect::<Vec<_>>()];

    let mut candidate_components = Vec::new();

    for node_ids in candidate_node_ids {
        let selected_pixels = node_ids
            .iter()
            .map(|idx| node_to_pixel[&idx.index()])
            .collect();

        let component = Component2D::new(selected_pixels, frontier_score, mean_contour_value);

        candidate_components.push(component);
    }

    candidate_components
}

fn flood_fill_3d(
    foreground: &ndarray::ArrayView3<bool>,
    contours: &ndarray::ArrayView3<f64>,
    visited: &mut Vec<Vec<Vec<bool>>>,
    start_k: usize,
    start_i: usize,
    start_j: usize,
    depth: usize,
    height: usize,
    width: usize,
    min_num_pixels: usize,
    max_num_pixels: usize,
    min_frontier: Option<f64>,
) -> Vec<Component3D> {
    let mut queue = VecDeque::new();
    let mut pixels = Vec::new();
    let mut contour_sum = 0.0;
    let mut boundary_pixels = 0;

    queue.push_back((start_k, start_i, start_j));
    visited[start_k][start_i][start_j] = true;

    // 6-connectivity for 3D
    let directions = [
        (-1, 0, 0),
        (0, -1, 0),
        (0, 0, -1),
        (0, 0, 1),
        (0, 1, 0),
        (1, 0, 0),
    ];

    while let Some((k, i, j)) = queue.pop_front() {
        pixels.push((k, i, j));
        contour_sum += contours[[k, i, j]];

        let mut is_boundary = false;
        for (dk, di, dj) in &directions {
            let nk = k as i32 + dk;
            let ni = i as i32 + di;
            let nj = j as i32 + dj;

            if nk >= 0
                && nk < depth as i32
                && ni >= 0
                && ni < height as i32
                && nj >= 0
                && nj < width as i32
            {
                let nk = nk as usize;
                let ni = ni as usize;
                let nj = nj as usize;

                if foreground[[nk, ni, nj]] {
                    if !visited[nk][ni][nj] {
                        visited[nk][ni][nj] = true;
                        queue.push_back((nk, ni, nj));
                    }
                } else {
                    is_boundary = true;
                }
            } else {
                is_boundary = true;
            }
        }

        if is_boundary {
            boundary_pixels += 1;
        }
    }

    // TODO: you will compute these values per segment
    let mean_contour_value = contour_sum / pixels.len() as f64;
    let frontier_score = boundary_pixels as f64 / pixels.len() as f64;

    // Build graph with flattened node indices
    let (graph, node_to_pixel) = build_flattened_graph_3d(&pixels, contours);

    // TODO: to be replaced by proper segmentation function
    // use min_num_pixels, max_num_pixels and min_frontier to filter candidate components
    let candidate_node_ids = vec![graph.node_indices().collect::<Vec<_>>()];

    let mut candidate_components = Vec::new();

    for node_ids in candidate_node_ids {
        let selected_pixels = node_ids
            .iter()
            .map(|idx| node_to_pixel[&idx.index()])
            .collect();

        let component = Component3D::new(selected_pixels, frontier_score, mean_contour_value);

        candidate_components.push(component);
    }

    candidate_components
}

fn components_to_python_dict_2d<'py>(
    py: Python<'py>,
    components: Vec<Component2D>,
) -> PyResult<PyObject> {
    let mut result: Vec<PyObject> = Vec::new();

    for (idx, component) in components.into_iter().enumerate() {
        let node_attrs = PyDict::new(py);

        node_attrs.set_item("id", idx)?;
        node_attrs.set_item("num_pixels", component.pixels.len())?;
        node_attrs.set_item("y", component.centroid.0)?;
        node_attrs.set_item("x", component.centroid.1)?;
        node_attrs.set_item("frontier_score", component.frontier_score)?;
        node_attrs.set_item("mean_contour_value", component.mean_contour_value)?;
        node_attrs.set_item("mask", component.mask.into_pyarray(py))?;
        node_attrs.set_item("bbox", component.bbox.into_pyarray(py))?;

        result.push(node_attrs.into());
    }

    Ok(PyList::new(py, result)?.into())
}

fn components_to_python_dict_3d<'py>(
    py: Python<'py>,
    components: Vec<Component3D>,
) -> PyResult<PyObject> {
    let mut result: Vec<PyObject> = Vec::new();

    for (idx, component) in components.into_iter().enumerate() {
        let node_attrs = PyDict::new(py);

        node_attrs.set_item("id", idx)?;
        node_attrs.set_item("num_pixels", component.pixels.len())?;
        node_attrs.set_item("z", component.centroid.0)?;
        node_attrs.set_item("y", component.centroid.1)?;
        node_attrs.set_item("x", component.centroid.2)?;
        node_attrs.set_item("frontier_score", component.frontier_score)?;
        node_attrs.set_item("mean_contour_value", component.mean_contour_value)?;
        node_attrs.set_item("mask", component.mask.into_pyarray(py))?;
        node_attrs.set_item("bbox", component.bbox.into_pyarray(py))?;

        result.push(node_attrs.into());
    }

    Ok(PyList::new(py, result)?.into())
}

// TODO: remove this
fn convert_graph_2d_to_python<'py>(
    graph: &UnGraph<usize, f32>,
    py: Python<'py>,
) -> PyResult<PyObject> {
    let graph_dict = PyDict::new(py);

    // Extract nodes - already flattened as usize indices
    let mut nodes = Vec::new();
    for node_idx in graph.node_indices() {
        nodes.push(graph[node_idx] as i32);
    }

    // Extract edges with weights
    let mut edges = Vec::new();
    for edge_idx in graph.edge_indices() {
        let (node_a, node_b) = graph.edge_endpoints(edge_idx).unwrap();
        let weight = graph[edge_idx];

        edges.push(vec![graph[node_a] as f32, graph[node_b] as f32, weight]);
    }

    // Convert to Python arrays
    let nodes_py = PyArray2::from_vec2(py, &[nodes]).unwrap();
    let edges_vec: Vec<Vec<f32>> = edges;
    let edges_py = PyArray2::from_vec2(py, &edges_vec).unwrap();

    graph_dict.set_item("nodes", nodes_py)?;
    graph_dict.set_item("edges", edges_py)?;

    Ok(graph_dict.into())
}

// TODO: remove this
fn convert_graph_3d_to_python<'py>(
    graph: &UnGraph<usize, f32>,
    py: Python<'py>,
) -> PyResult<PyObject> {
    let graph_dict = PyDict::new(py);

    // Extract nodes - already flattened as usize indices
    let mut nodes = Vec::new();
    for node_idx in graph.node_indices() {
        nodes.push(graph[node_idx] as i32);
    }

    // Extract edges with weights
    let mut edges = Vec::new();
    for edge_idx in graph.edge_indices() {
        let (node_a, node_b) = graph.edge_endpoints(edge_idx).unwrap();
        let weight = graph[edge_idx];

        edges.push(vec![graph[node_a] as f32, graph[node_b] as f32, weight]);
    }

    // Convert to Python arrays
    let nodes_py = PyArray2::from_vec2(py, &[nodes]).unwrap();
    let edges_vec: Vec<Vec<f32>> = edges;
    let edges_py = PyArray2::from_vec2(py, &edges_vec).unwrap();

    graph_dict.set_item("nodes", nodes_py)?;
    graph_dict.set_item("edges", edges_py)?;

    Ok(graph_dict.into())
}

fn build_flattened_graph_2d(
    pixels: &[(usize, usize)],
    contours: &ndarray::ArrayView2<f64>,
) -> (UnGraph<usize, f32>, HashMap<usize, (usize, usize)>) {
    let mut graph = UnGraph::new_undirected();
    let mut pixel_to_node = HashMap::new();
    let mut node_to_pixel = HashMap::new();

    // Add nodes with flattened indices
    for (idx, &pixel) in pixels.iter().enumerate() {
        let node_idx = graph.add_node(idx);
        pixel_to_node.insert(pixel, node_idx);
        node_to_pixel.insert(node_idx.index(), pixel);
    }

    // 4-connectivity for 2D
    let directions = [(-1, 0), (0, -1), (0, 1), (1, 0)];

    // Add edges between adjacent pixels
    for &(i, j) in pixels {
        if let Some(&node_a) = pixel_to_node.get(&(i, j)) {
            for (di, dj) in &directions {
                let ni = i as i32 + di;
                let nj = j as i32 + dj;

                if ni >= 0 && nj >= 0 {
                    let neighbor = (ni as usize, nj as usize);
                    if let Some(&node_b) = pixel_to_node.get(&neighbor) {
                        // Calculate edge weight as average of contour values
                        let weight =
                            ((contours[[i, j]] + contours[[neighbor.0, neighbor.1]]) * 0.5) as f32;
                        graph.add_edge(node_a, node_b, weight);
                    }
                }
            }
        }
    }

    (graph, node_to_pixel)
}

fn build_flattened_graph_3d(
    pixels: &[(usize, usize, usize)],
    contours: &ndarray::ArrayView3<f64>,
) -> (UnGraph<usize, f32>, HashMap<usize, (usize, usize, usize)>) {
    let mut graph = UnGraph::new_undirected();
    let mut pixel_to_node = HashMap::new();
    let mut node_to_pixel = HashMap::new();

    // Add nodes with flattened indices
    for (idx, &pixel) in pixels.iter().enumerate() {
        let node_idx = graph.add_node(idx);
        pixel_to_node.insert(pixel, node_idx);
        node_to_pixel.insert(node_idx.index(), pixel);
    }

    // 6-connectivity for 3D
    let directions = [
        (-1, 0, 0),
        (0, -1, 0),
        (0, 0, -1),
        (0, 0, 1),
        (0, 1, 0),
        (1, 0, 0),
    ];

    // Add edges between adjacent pixels
    for &(k, i, j) in pixels {
        if let Some(&node_a) = pixel_to_node.get(&(k, i, j)) {
            for (dk, di, dj) in &directions {
                let nk = k as i32 + dk;
                let ni = i as i32 + di;
                let nj = j as i32 + dj;

                if nk >= 0 && ni >= 0 && nj >= 0 {
                    let neighbor = (nk as usize, ni as usize, nj as usize);
                    if let Some(&node_b) = pixel_to_node.get(&neighbor) {
                        // Calculate edge weight as average of contour values
                        let weight = ((contours[[k, i, j]]
                            + contours[[neighbor.0, neighbor.1, neighbor.2]])
                            * 0.5) as f32;
                        graph.add_edge(node_a, node_b, weight);
                    }
                }
            }
        }
    }

    (graph, node_to_pixel)
}
