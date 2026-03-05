use crate::deserialize::value::{PyDeserializeValue, PyDeserializedValue};
use crate::errors::DriverDeserializationError;
use pyo3::exceptions::{PyRuntimeError, PyStopIteration};
use pyo3::prelude::{PyDictMethods, PyModule, PyModuleMethods};
use pyo3::types::{PyDict, PyString};
use pyo3::{Bound, Py, PyAny, PyErr, PyRefMut, PyResult, Python, pyclass, pymethods, pymodule};
use scylla::response::query_result::QueryResult;
use scylla_cql::deserialize::FrameSlice;
use scylla_cql::deserialize::result::RawRowIterator;
use scylla_cql::deserialize::row::ColumnIterator;
use stable_deref_trait::StableDeref;
use std::ops::Deref;
use std::sync::Arc;
use yoke::{Yoke, Yokeable};

/// Result of a single request to the database. It represents any kind of Result frame.
#[pyclass(frozen)]
pub(crate) struct RequestResult {
    pub(crate) inner: Arc<QueryResult>,
}

#[pymethods]
impl RequestResult {
    /// Iterate over rows returned by the query.
    ///
    /// This method returns a Python iterator yielding rows from the result set.
    /// Each row is automatically deserialized and materialized using a
    /// `RowFactory`.
    ///
    /// By default, rows are returned as Python dictionaries mapping column
    /// names (`str`) to deserialized values. A custom `RowFactory` may be
    /// provided to control how rows are constructed.
    ///
    /// Parameters
    /// ----------
    /// factory : RowFactory, optional
    ///     Custom factory used to build each row.
    ///
    /// Returns
    /// -------
    /// RowsIterator
    ///     An iterator yielding deserialized rows.
    #[pyo3(signature = (factory=None))]
    fn iter_rows<'py>(
        &self,
        py: Python<'py>,
        factory: Option<Bound<RowFactory>>,
    ) -> PyResult<RowsIterator> {
        let row_col_cursor = Py::new(py, RowColumnCursor::new(Arc::clone(&self.inner))?)?;

        let f: Py<RowFactory> = match factory {
            Some(bound) => bound.unbind(),
            None => Py::new(py, RowFactory::new())?,
        };

        Ok(RowsIterator {
            row_col_cursor,
            factory: f,
        })
    }
}

/// Iterator yielding deserialized rows from a query result.
///
/// `RowsIterator` behaves like a standard Python iterator. Each iteration
/// returns a single row produced by the query and materialized using a
/// `RowFactory`.
///
/// By default, rows are returned as Python dictionaries mapping column
/// names to values.
#[pyclass]
pub struct RowsIterator {
    // This cursor maintains the current row and column iteration state
    // and is shared with the `RowFactory` during row construction.
    row_col_cursor: Py<RowColumnCursor>,

    // Python object responsible for building a single row from the cursor.
    factory: Py<RowFactory>,
}

#[pymethods]
impl RowsIterator {
    pub fn __next__(&mut self) -> PyResult<Py<PyAny>> {
        Python::attach(|py| {
            self.row_col_cursor
                .borrow_mut(py)
                .yoked
                .with_mut_return(|view| view.next_row())?;

            self.factory
                .call_method1(py, "build", (&self.row_col_cursor.bind(py),))
        })
    }

    pub fn __iter__(slf: PyRefMut<'_, Self>) -> PyRefMut<'_, Self> {
        slf
    }
}

/// Yoke-backed wrapper holding row and column iterators.
///
/// `Cursor` is stored inside a `Yoke` so that both the row iterator
///  and the column iterator can borrow from the same data without cloning.
///
/// - `next_row` advances the row iterator and switches the active column
///   iterator to the value received from row iterator.
/// - `next_column` advances the column iterator and deserializes column values
///   into Python objects.
#[derive(Yokeable)]
struct Cursor<'a> {
    row_iterator: RawRowIterator<'a, 'a>,
    column_iterator: ColumnIterator<'a, 'a>,
    row_index: usize,
}

impl<'a> Cursor<'a> {
    fn next_column(&mut self) -> Result<Option<Column>, DriverDeserializationError> {
        Python::attach(|py| {
            let raw_col = match self.column_iterator.next() {
                None => return Ok(None), // End of columns in the current row
                Some(result) => result.map_err(DriverDeserializationError::scylla)?,
            };

            let col_name_str = raw_col.spec.name().to_string();

            let value = PyDeserializedValue::deserialize_py(raw_col.spec.typ(), raw_col.slice, py)
                .map_err(|e| {
                    e.at_row(self.row_index)
                        .at_column_name(col_name_str.clone())
                })?;

            let column_name = PyString::new(py, &col_name_str).unbind();

            Ok(Some(Column { column_name, value }))
        })
    }

    fn next_row(&mut self) -> Result<bool, DriverDeserializationError> {
        let column_iterator = match self.row_iterator.next() {
            None => return Ok(false), // End of rows
            Some(result) => result.map_err(DriverDeserializationError::scylla)?,
        };

        self.column_iterator = column_iterator;

        // row_index is used for error context
        self.row_index = self.row_index.wrapping_add(1);
        Ok(true)
    }
}

/// Stable cart holding deserialized metadata and raw row data.
///
/// This type exists solely to serve as a `StableDeref` cart for `Yoke`.
struct QueryResultCart(Arc<QueryResult>);

impl Deref for QueryResultCart {
    type Target = QueryResult;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

unsafe impl StableDeref for QueryResultCart {}

/// Iterator over columns of the current row.
///
/// This object is passed to `RowFactory.build` and allows iterating over
/// column values of a single row. Each iteration yields a `Column` object
/// containing the column name and its deserialized value.
///
/// This iterator is only intended to be consumed while building a row and
/// should not be stored or reused outside of that context.
#[pyclass(name = "ColumnIterator")]
pub struct RowColumnCursor {
    // Yoke-backed container holding both row and column iterators.
    //
    // The yoke ensures that iterators can borrow directly from the
    // underlying query result frame without cloning buffers or allocating
    // intermediate representations.
    //
    // `Cursor` holds:
    // - a `RawRowIterator` to advance between rows
    // - a `ColumnIterator` for iterating columns of the current row
    yoked: Yoke<Cursor<'static>, QueryResultCart>,
}

impl RowColumnCursor {
    fn new(query_result: Arc<QueryResult>) -> PyResult<Self> {
        let cart = QueryResultCart(query_result);

        let yoked = Yoke::try_attach_to_cart(cart, |cart| -> PyResult<_> {
            let Some(raw_rows_with_metadata) = cart.deserialized_metadata_and_rows() else {
                return Err(PyRuntimeError::new_err("Result does not have rows"));
            };
            let frame_slice = FrameSlice::new(raw_rows_with_metadata.raw_rows());
            let col_specs = raw_rows_with_metadata.metadata().col_specs();
            let row_iterator =
                RawRowIterator::new(raw_rows_with_metadata.rows_count(), col_specs, frame_slice);

            let column_iterator = ColumnIterator::new(col_specs, frame_slice);

            Ok(Cursor {
                row_iterator,
                column_iterator,
                row_index: usize::MAX, // Will be incremented to 0 on first next_row call
            })
        })?;

        Ok(Self { yoked })
    }
}

#[pymethods]
impl RowColumnCursor {
    pub fn __next__(&mut self) -> PyResult<Column> {
        match self.yoked.with_mut_return(|view| view.next_column()) {
            Ok(Some(column)) => Ok(column),
            Ok(None) => Err(PyStopIteration::new_err(())),
            Err(err) => Err(PyErr::from(err)),
        }
    }
    pub fn __iter__(slf: PyRefMut<'_, Self>) -> PyRefMut<'_, Self> {
        slf
    }
}

/// A single column value within a row.
///
/// `Column` represents one column of a row returned by a query. It contains
/// the column name and the corresponding deserialized Python value.
#[pyclass(frozen)]
pub struct Column {
    #[pyo3(get)]
    column_name: Py<PyString>,
    #[pyo3(get)]
    value: PyDeserializedValue,
}

/// Factory responsible for constructing Python row objects.
///
/// `RowFactory` defines how a row is materialized from a column iterator.
/// The default implementation consumes all columns of the current row and
/// returns a Python dictionary mapping column names to values.
///
/// Users may subclass this type to implement custom row mappings.
#[pyclass(subclass)]
pub struct RowFactory {}

#[pymethods]
impl RowFactory {
    /// Create a new `RowFactory`.
    ///
    /// The default row factory builds each row as a Python `dict`
    /// mapping column names to deserialized Python values.
    #[new]
    pub fn new() -> Self {
        RowFactory {}
    }

    /// Build a Python object representing a single row.
    ///
    /// This method consumes all columns from the provided column iterator
    /// and returns a Python `dict` mapping column names to values.
    ///
    /// Parameters
    /// ----------
    /// column_iterator : RowColumnCursor
    ///     Iterator over columns of the current row.
    ///
    /// Returns
    /// -------
    /// dict
    ///     A dictionary mapping column names (`str`) to deserialized
    ///     Python values.
    ///
    /// Raises
    /// ------
    /// DriverDeserializationError
    ///     If deserialization of any column fails.
    pub fn build<'py>(
        &self,
        py: Python<'py>,
        column_iterator: &Bound<'py, RowColumnCursor>,
    ) -> Result<Py<PyDict>, DriverDeserializationError> {
        let mut columns = column_iterator.borrow_mut();

        let dict = PyDict::new(py);

        loop {
            let next = columns.yoked.with_mut_return(|view| view.next_column())?;

            let Some(column) = next else {
                break;
            };

            dict.set_item(column.column_name, column.value)
                .map_err(DriverDeserializationError::python)?;
        }

        Ok(dict.into())
    }
}

impl Default for RowFactory {
    fn default() -> Self {
        RowFactory::new()
    }
}

#[pymodule]
pub(crate) fn results(_py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<RowFactory>()?;
    module.add_class::<Column>()?;
    module.add_class::<RequestResult>()?;
    module.add_class::<RowColumnCursor>()?;
    module.add_class::<RowsIterator>()?;

    Ok(())
}
