use numpy::{PyArray2, PyReadonlyArray2, PyReadonlyArray3};
use petgraph::graph::UnGraph;
use pyo3::prelude::*;
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
    graph: UnGraph<usize, f32>,
}

#[derive(Debug)]
struct Component3D {
    pixels: Vec<(usize, usize, usize)>,
    centroid: (f64, f64, f64),
    frontier_score: f64,
    mean_contour_value: f64,
    graph: UnGraph<usize, f32>,
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
    let mut components = Vec::new();

    for i in 0..height {
        for j in 0..width {
            if foreground[[i, j]] && !visited[i][j] {
                let component =
                    flood_fill_2d(foreground, contours, &mut visited, i, j, height, width);

                if component.pixels.len() >= min_num_pixels
                    && component.pixels.len() <= max_num_pixels
                {
                    if let Some(min_frontier_val) = min_frontier {
                        if component.frontier_score >= min_frontier_val {
                            components.push(component);
                        }
                    } else {
                        components.push(component);
                    }
                }
            }
        }
    }

    components
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
    let mut components = Vec::new();

    for k in 0..depth {
        for i in 0..height {
            for j in 0..width {
                if foreground[[k, i, j]] && !visited[k][i][j] {
                    let component = flood_fill_3d(
                        foreground,
                        contours,
                        &mut visited,
                        k,
                        i,
                        j,
                        depth,
                        height,
                        width,
                    );

                    if component.pixels.len() >= min_num_pixels
                        && component.pixels.len() <= max_num_pixels
                    {
                        if let Some(min_frontier_val) = min_frontier {
                            if component.frontier_score >= min_frontier_val {
                                components.push(component);
                            }
                        } else {
                            components.push(component);
                        }
                    }
                }
            }
        }
    }

    components
}

fn flood_fill_2d(
    foreground: &ndarray::ArrayView2<bool>,
    contours: &ndarray::ArrayView2<f64>,
    visited: &mut Vec<Vec<bool>>,
    start_i: usize,
    start_j: usize,
    height: usize,
    width: usize,
) -> Component2D {
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

    let centroid = calculate_centroid_2d(&pixels);
    let mean_contour_value = contour_sum / pixels.len() as f64;
    let frontier_score = boundary_pixels as f64 / pixels.len() as f64;

    // Build graph with flattened node indices
    let graph = build_flattened_graph_2d(&pixels, contours);

    Component2D {
        pixels,
        centroid,
        frontier_score,
        mean_contour_value,
        graph,
    }
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
) -> Component3D {
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

    let centroid = calculate_centroid_3d(&pixels);
    let mean_contour_value = contour_sum / pixels.len() as f64;
    let frontier_score = boundary_pixels as f64 / pixels.len() as f64;

    // Build graph with flattened node indices
    let graph = build_flattened_graph_3d(&pixels, contours);

    Component3D {
        pixels,
        centroid,
        frontier_score,
        mean_contour_value,
        graph,
    }
}

fn calculate_centroid_2d(pixels: &[(usize, usize)]) -> (f64, f64) {
    let sum_i: usize = pixels.iter().map(|(i, _)| *i).sum();
    let sum_j: usize = pixels.iter().map(|(_, j)| *j).sum();
    let count = pixels.len() as f64;

    (sum_i as f64 / count, sum_j as f64 / count)
}

fn calculate_centroid_3d(pixels: &[(usize, usize, usize)]) -> (f64, f64, f64) {
    let sum_k: usize = pixels.iter().map(|(k, _, _)| *k).sum();
    let sum_i: usize = pixels.iter().map(|(_, i, _)| *i).sum();
    let sum_j: usize = pixels.iter().map(|(_, _, j)| *j).sum();
    let count = pixels.len() as f64;

    (
        sum_k as f64 / count,
        sum_i as f64 / count,
        sum_j as f64 / count,
    )
}

fn components_to_python_dict_2d<'py>(
    py: Python<'py>,
    components: Vec<Component2D>,
) -> PyResult<PyObject> {
    let mut result = Vec::new();

    for (idx, component) in components.into_iter().enumerate() {
        let mut node_attrs = HashMap::new();

        node_attrs.insert("id".to_string(), idx.into_py(py));
        node_attrs.insert("area".to_string(), component.pixels.len().into_py(py));
        node_attrs.insert("y".to_string(), component.centroid.0.into_py(py));
        node_attrs.insert("x".to_string(), component.centroid.1.into_py(py));
        node_attrs.insert(
            "frontier_score".to_string(),
            component.frontier_score.into_py(py),
        );
        node_attrs.insert(
            "mean_contour_value".to_string(),
            component.mean_contour_value.into_py(py),
        );

        // Convert graph to Python-readable format
        let graph_data = convert_graph_2d_to_python(&component.graph, py)?;
        node_attrs.insert("graph".to_string(), graph_data);

        let pixels_vec: Vec<Vec<i32>> = component
            .pixels
            .into_iter()
            .map(|(i, j)| vec![i as i32, j as i32])
            .collect();
        let pixels_py = PyArray2::from_vec2_bound(py, &pixels_vec)
            .unwrap()
            .into_py(py);
        node_attrs.insert("pixels".to_string(), pixels_py);

        result.push(node_attrs.into_py(py));
    }

    Ok(result.into_py(py))
}

fn components_to_python_dict_3d<'py>(
    py: Python<'py>,
    components: Vec<Component3D>,
) -> PyResult<PyObject> {
    let mut result = Vec::new();

    for (idx, component) in components.into_iter().enumerate() {
        let mut node_attrs = HashMap::new();

        node_attrs.insert("id".to_string(), idx.into_py(py));
        node_attrs.insert("area".to_string(), component.pixels.len().into_py(py));
        node_attrs.insert("z".to_string(), component.centroid.0.into_py(py));
        node_attrs.insert("y".to_string(), component.centroid.1.into_py(py));
        node_attrs.insert("x".to_string(), component.centroid.2.into_py(py));
        node_attrs.insert(
            "frontier_score".to_string(),
            component.frontier_score.into_py(py),
        );
        node_attrs.insert(
            "mean_contour_value".to_string(),
            component.mean_contour_value.into_py(py),
        );

        // Convert graph to Python-readable format
        let graph_data = convert_graph_3d_to_python(&component.graph, py)?;
        node_attrs.insert("graph".to_string(), graph_data);

        let pixels_vec: Vec<Vec<i32>> = component
            .pixels
            .into_iter()
            .map(|(k, i, j)| vec![k as i32, i as i32, j as i32])
            .collect();
        let pixels_py = PyArray2::from_vec2_bound(py, &pixels_vec)
            .unwrap()
            .into_py(py);
        node_attrs.insert("pixels".to_string(), pixels_py);

        result.push(node_attrs.into_py(py));
    }

    Ok(result.into_py(py))
}

// TODO: remove this
fn convert_graph_2d_to_python<'py>(
    graph: &UnGraph<usize, f32>,
    py: Python<'py>,
) -> PyResult<PyObject> {
    let mut graph_dict = HashMap::new();

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
    let nodes_py = PyArray2::from_vec2_bound(py, &[nodes]).unwrap();
    let edges_vec: Vec<Vec<f32>> = edges;
    let edges_py = PyArray2::from_vec2_bound(py, &edges_vec).unwrap();

    graph_dict.insert("nodes".to_string(), nodes_py.into_py(py));
    graph_dict.insert("edges".to_string(), edges_py.into_py(py));

    Ok(graph_dict.into_py(py))
}

// TODO: remove this
fn convert_graph_3d_to_python<'py>(
    graph: &UnGraph<usize, f32>,
    py: Python<'py>,
) -> PyResult<PyObject> {
    let mut graph_dict = HashMap::new();

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
    let nodes_py = PyArray2::from_vec2_bound(py, &[nodes]).unwrap();
    let edges_vec: Vec<Vec<f32>> = edges;
    let edges_py = PyArray2::from_vec2_bound(py, &edges_vec).unwrap();

    graph_dict.insert("nodes".to_string(), nodes_py.into_py(py));
    graph_dict.insert("edges".to_string(), edges_py.into_py(py));

    Ok(graph_dict.into_py(py))
}

fn build_flattened_graph_2d(
    pixels: &[(usize, usize)],
    contours: &ndarray::ArrayView2<f64>,
) -> UnGraph<usize, f32> {
    let mut graph = UnGraph::new_undirected();
    let mut pixel_to_node = HashMap::new();

    // Add nodes with flattened indices
    for (idx, &pixel) in pixels.iter().enumerate() {
        let node_idx = graph.add_node(idx);
        pixel_to_node.insert(pixel, node_idx);
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

    graph
}

fn build_flattened_graph_3d(
    pixels: &[(usize, usize, usize)],
    contours: &ndarray::ArrayView3<f64>,
) -> UnGraph<usize, f32> {
    let mut graph = UnGraph::new_undirected();
    let mut pixel_to_node = HashMap::new();

    // Add nodes with flattened indices
    for (idx, &pixel) in pixels.iter().enumerate() {
        let node_idx = graph.add_node(idx);
        pixel_to_node.insert(pixel, node_idx);
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

    graph
}
