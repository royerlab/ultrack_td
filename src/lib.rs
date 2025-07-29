use pyo3::prelude::*;

mod connected_components;
mod hierarchical_segmentation;

#[pymodule]
fn _rustlib<'py>(_py: Python<'py>, m: &Bound<'py, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(
        connected_components::compute_connected_components,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        connected_components::compute_connected_components_2d,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        connected_components::compute_connected_components_3d,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        hierarchical_segmentation::hierarchical_segmentation,
        m
    )?)?;
    Ok(())
}
