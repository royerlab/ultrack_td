use numpy::{PyArray1, PyArray2, PyArray3};
use pyo3::prelude::*;
use std::collections::HashMap;

#[derive(Debug)]
pub struct ComponentMask2D {
    pub mask: Vec<Vec<bool>>,
    pub bounding_box: Vec<usize>, // [y_start, x_start, y_end, x_end]
}

#[derive(Debug)]
pub struct ComponentMask3D {
    pub mask: Vec<Vec<Vec<bool>>>,
    pub bounding_box: Vec<usize>, // [z_start, y_start, x_start, z_end, y_end, x_end]
}

#[pyfunction]
pub fn create_component_mask_2d<'py>(
    py: Python<'py>,
    pixels: Vec<(usize, usize)>,
) -> PyResult<PyObject> {
    let mask = generate_component_mask_2d(&pixels);
    component_mask_2d_to_python(py, mask)
}

#[pyfunction]
pub fn create_component_mask_3d<'py>(
    py: Python<'py>,
    pixels: Vec<(usize, usize, usize)>,
) -> PyResult<PyObject> {
    let mask = generate_component_mask_3d(&pixels);
    component_mask_3d_to_python(py, mask)
}

fn generate_component_mask_2d(pixels: &[(usize, usize)]) -> ComponentMask2D {
    if pixels.is_empty() {
        return ComponentMask2D {
            mask: Vec::new(),
            bounding_box: vec![0, 0, 0, 0], // [y_start, x_start, y_end, x_end]
        };
    }

    // Calculate bounding box
    let min_y = pixels.iter().map(|(y, _)| *y).min().unwrap();
    let max_y = pixels.iter().map(|(y, _)| *y).max().unwrap();
    let min_x = pixels.iter().map(|(_, x)| *x).min().unwrap();
    let max_x = pixels.iter().map(|(_, x)| *x).max().unwrap();

    // Bounding box format: [y_start, x_start, y_end, x_end]
    let bounding_box = vec![min_y, min_x, max_y + 1, max_x + 1];

    // Create mask within bounding box
    let height = max_y - min_y + 1;
    let width = max_x - min_x + 1;
    let mut mask = vec![vec![false; width]; height];

    // Set pixels to true in the mask
    for &(y, x) in pixels {
        let mask_y = y - min_y;
        let mask_x = x - min_x;
        mask[mask_y][mask_x] = true;
    }

    ComponentMask2D { mask, bounding_box }
}

fn generate_component_mask_3d(pixels: &[(usize, usize, usize)]) -> ComponentMask3D {
    if pixels.is_empty() {
        return ComponentMask3D {
            mask: Vec::new(),
            bounding_box: vec![0, 0, 0, 0, 0, 0], // [z_start, y_start, x_start, z_end, y_end, x_end]
        };
    }

    // Calculate bounding box
    let min_z = pixels.iter().map(|(z, _, _)| *z).min().unwrap();
    let max_z = pixels.iter().map(|(z, _, _)| *z).max().unwrap();
    let min_y = pixels.iter().map(|(_, y, _)| *y).min().unwrap();
    let max_y = pixels.iter().map(|(_, y, _)| *y).max().unwrap();
    let min_x = pixels.iter().map(|(_, _, x)| *x).min().unwrap();
    let max_x = pixels.iter().map(|(_, _, x)| *x).max().unwrap();

    // Bounding box format: [z_start, y_start, x_start, z_end, y_end, x_end]
    let bounding_box = vec![min_z, min_y, min_x, max_z + 1, max_y + 1, max_x + 1];

    // Create mask within bounding box
    let depth = max_z - min_z + 1;
    let height = max_y - min_y + 1;
    let width = max_x - min_x + 1;
    let mut mask = vec![vec![vec![false; width]; height]; depth];

    // Set pixels to true in the mask
    for &(z, y, x) in pixels {
        let mask_z = z - min_z;
        let mask_y = y - min_y;
        let mask_x = x - min_x;
        mask[mask_z][mask_y][mask_x] = true;
    }

    ComponentMask3D { mask, bounding_box }
}

fn component_mask_2d_to_python<'py>(
    py: Python<'py>,
    component_mask: ComponentMask2D,
) -> PyResult<PyObject> {
    let mut result: HashMap<String, PyObject> = HashMap::new();

    // Convert mask to boolean numpy array
    let mask_py = PyArray2::from_vec2(py, &component_mask.mask).unwrap();
    result.insert("mask".to_string(), mask_py.into());

    // Convert bounding box to numpy array [y_start, x_start, y_end, x_end]
    let bbox_py = PyArray1::from_vec(py, component_mask.bounding_box);
    result.insert("bounding_box".to_string(), bbox_py.into());

    Ok(result.into_pyobject(py)?.into())
}

fn component_mask_3d_to_python<'py>(
    py: Python<'py>,
    component_mask: ComponentMask3D,
) -> PyResult<PyObject> {
    let mut result: HashMap<String, PyObject> = HashMap::new();

    // Convert mask to boolean numpy array
    let mask_py = PyArray3::from_vec3(py, &component_mask.mask).unwrap();
    result.insert("mask".to_string(), mask_py.into());

    // Convert bounding box to numpy array [z_start, y_start, x_start, z_end, y_end, x_end]
    let bbox_py = PyArray1::from_vec(py, component_mask.bounding_box);
    result.insert("bounding_box".to_string(), bbox_py.into());

    Ok(result.into_pyobject(py)?.into())
}
