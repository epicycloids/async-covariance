mod batch;
mod engine;
mod factor;
mod interval_index;

use engine::Engine;
use numpy::{ndarray::Array2, IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

fn value_error(message: String) -> PyErr {
    PyValueError::new_err(message)
}

type Statistics<'py> = (
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray2<f64>>,
);

fn arrays(py: Python<'_>, stats: Vec<(usize, usize, [f64; 14])>) -> Statistics<'_> {
    let mut rows = Vec::with_capacity(stats.len());
    let mut cols = Vec::with_capacity(stats.len());
    let mut values = Vec::with_capacity(stats.len() * 14);
    for (a, b, v) in stats {
        rows.push(a as i64);
        cols.push(b as i64);
        values.extend_from_slice(&v);
    }
    let values = Array2::from_shape_vec((rows.len(), 14), values).expect("statistic shape");
    (
        rows.into_pyarray(py),
        cols.into_pyarray(py),
        values.into_pyarray(py),
    )
}

#[pyfunction]
#[pyo3(signature = (n_assets, times, assets, prices, half_life=None, window=None, edges=None, threads=None, reference_drift=None))]
fn batch_statistics<'py>(
    py: Python<'py>,
    n_assets: usize,
    times: PyReadonlyArray1<'py, f64>,
    assets: PyReadonlyArray1<'py, i64>,
    prices: PyReadonlyArray1<'py, f64>,
    half_life: Option<f64>,
    window: Option<f64>,
    edges: Option<Vec<(usize, usize)>>,
    threads: Option<usize>,
    reference_drift: Option<PyReadonlyArray1<'py, f64>>,
) -> PyResult<Statistics<'py>> {
    let times = times.as_slice()?.to_vec();
    let assets = assets.as_slice()?.to_vec();
    let prices = prices.as_slice()?.to_vec();
    let reference = reference_drift
        .map(|array| array.as_slice().map(|x| x.to_vec()))
        .transpose()?;
    let stats = py
        .detach(|| {
            batch::batch(
                &times,
                &assets,
                &prices,
                n_assets,
                half_life,
                window,
                edges,
                threads,
                reference.as_deref(),
            )
        })
        .map_err(value_error)?;
    Ok(arrays(py, stats))
}

#[pyfunction]
fn _factor_optimize<'py>(
    py: Python<'py>,
    rows: PyReadonlyArray1<'py, i64>,
    cols: PyReadonlyArray1<'py, i64>,
    targets: PyReadonlyArray1<'py, f64>,
    weights: PyReadonlyArray1<'py, f64>,
    initial: PyReadonlyArray1<'py, f64>,
    n: usize,
    rank: usize,
    radius: f64,
    max_iter: usize,
    tol: f64,
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    bool,
    &'static str,
    usize,
    f64,
)> {
    let rows = rows.as_slice()?.to_vec();
    let cols = cols.as_slice()?.to_vec();
    let targets = targets.as_slice()?.to_vec();
    let weights = weights.as_slice()?.to_vec();
    let initial = initial.as_slice()?.to_vec();
    let result = py
        .detach(|| {
            factor::optimize(
                &rows, &cols, &targets, &weights, &initial, n, rank, radius, max_iter, tol,
            )
        })
        .map_err(value_error)?;
    Ok((
        result.loadings.into_pyarray(py),
        result.history.into_pyarray(py),
        result.converged,
        result.reason,
        result.rejected,
        result.max_update,
    ))
}

/// Python owns this object; all update algebra runs in Rust without the GIL.
#[pyclass(name = "Core")]
struct Core {
    engine: Engine,
}

#[pymethods]
impl Core {
    #[new]
    #[pyo3(signature = (n_assets, half_life=None, edges=None, window=None, reference_drift=None))]
    fn new(
        n_assets: usize,
        half_life: Option<f64>,
        edges: Option<Vec<(usize, usize)>>,
        window: Option<f64>,
        reference_drift: Option<Vec<f64>>,
    ) -> PyResult<Self> {
        let mut engine = Engine::new(n_assets, half_life, edges).map_err(value_error)?;
        engine.set_window(window).map_err(value_error)?;
        if let Some(reference) = reference_drift {
            engine.set_reference_drift(reference).map_err(value_error)?;
        }
        Ok(Self { engine })
    }

    fn update(&mut self, py: Python<'_>, asset: usize, time: f64, log_price: f64) -> PyResult<()> {
        py.detach(|| self.engine.update(asset, time, log_price))
            .map_err(value_error)
    }

    fn update_many(
        &mut self,
        py: Python<'_>,
        times: PyReadonlyArray1<'_, f64>,
        assets: PyReadonlyArray1<'_, i64>,
        prices: PyReadonlyArray1<'_, f64>,
    ) -> PyResult<()> {
        // Copy before releasing the GIL: callers can otherwise mutate a NumPy
        // buffer while the native engine is reading it.
        let times = times.as_slice()?.to_vec();
        let assets = assets.as_slice()?.to_vec();
        let prices = prices.as_slice()?.to_vec();
        if times.len() != assets.len() || times.len() != prices.len() {
            return Err(PyValueError::new_err(
                "times, assets, and log_prices must have equal lengths",
            ));
        }
        if times.iter().any(|x| !x.is_finite()) || prices.iter().any(|x| !x.is_finite()) {
            return Err(PyValueError::new_err("times and log_prices must be finite"));
        }
        if assets.iter().any(|&x| x < 0 || x > i32::MAX as i64) {
            return Err(PyValueError::new_err(
                "assets must be nonnegative integer indices below 2**31",
            ));
        }
        if let (Some(window), Some(now)) = (self.engine.window(), self.engine.watermark()) {
            if times.iter().any(|&t| t < now - window) {
                return Err(PyValueError::new_err(
                    "observation precedes retained window",
                ));
            }
        }
        py.detach(|| {
            for ((&t, &a), &p) in times.iter().zip(&assets).zip(&prices) {
                self.engine.update(a as usize, t, p)?;
            }
            Ok::<(), String>(())
        })
        .map_err(value_error)
    }

    fn remove(&mut self, py: Python<'_>, asset: usize, time: f64) -> PyResult<()> {
        py.detach(|| self.engine.remove(asset, time))
            .map_err(value_error)
    }

    fn advance(&mut self, py: Python<'_>, time: f64) -> PyResult<()> {
        py.detach(|| self.engine.advance(time)).map_err(value_error)
    }

    fn statistics<'py>(&mut self, py: Python<'py>) -> Statistics<'py> {
        let stats = py.detach(|| self.engine.snapshot());
        arrays(py, stats)
    }

    #[getter]
    fn n_assets(&self) -> usize {
        self.engine.n_assets()
    }
    #[getter]
    fn n_points(&self) -> usize {
        self.engine.n_points()
    }
    #[getter]
    fn n_intervals(&self) -> usize {
        self.engine.n_intervals()
    }
    #[getter]
    fn n_pairs(&self) -> usize {
        self.engine.n_pairs()
    }
    #[getter]
    fn watermark(&self) -> Option<f64> {
        self.engine.watermark()
    }
    #[getter]
    fn reference_drift(&self) -> Vec<f64> {
        self.engine.reference_drift().to_vec()
    }
}

#[pymodule]
fn _native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<Core>()?;
    module.add_function(wrap_pyfunction!(batch_statistics, module)?)?;
    module.add_function(wrap_pyfunction!(_factor_optimize, module)?)?;
    Ok(())
}
