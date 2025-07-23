use pyo3::prelude::*;

#[pyfunction]
fn hello_rust() -> PyResult<String> {
    Ok("Hello from Rust!".to_string())
}

#[pymodule]
fn _rustlib(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(hello_rust, m)?)?;
    Ok(())
}