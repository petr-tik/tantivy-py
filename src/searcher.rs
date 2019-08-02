use crate::facet::Facet;
use crate::query::Query;
use crate::to_pyerr;
use pyo3::exceptions;
use pyo3::prelude::*;
use pyo3::types::{PyDateTime, PyDict};
use std::collections::BTreeMap;
use tantivy as tv;
use tantivy::schema;
use tantivy::schema::Value;

/// Tantivy's Searcher class
///
/// A Searcher is used to search the index given a prepared Query.
#[pyclass]
pub(crate) struct Searcher {
    pub(crate) inner: tv::LeasedItem<tv::Searcher>,
}

#[pymethods]
impl Searcher {
    /// Search the index with the given query and collect results.
    ///
    /// Args:
    ///     query (Query): The query that will be used for the search.
    ///     collector (Collector): A collector that determines how the search
    ///         results will be collected. Only the TopDocs collector is
    ///         supported for now.
    ///
    /// Returns a list of tuples that contains the scores and DocAddress of the
    /// search results.
    ///
    /// Raises a ValueError if there was an error with the search.
    fn search(
        &self,
        query: &Query,
        collector: &mut TopDocs,
    ) -> PyResult<Vec<(f32, DocAddress)>> {
        let ret = self.inner.search(&query.inner, &collector.inner);
        match ret {
            Ok(r) => {
                let result: Vec<(f32, DocAddress)> = r
                    .iter()
                    .map(|(f, d)| (f.clone(), DocAddress::from(d)))
                    .collect();
                Ok(result)
            }
            Err(e) => Err(exceptions::ValueError::py_err(e.to_string())),
        }
    }

    /// Returns the overall number of documents in the index.
    #[getter]
    fn num_docs(&self) -> u64 {
        self.inner.num_docs()
    }

    /// Fetches a document from Tantivy's store given a DocAddress.
    ///
    /// Args:
    ///     doc_address (DocAddress): The DocAddress that is associated with
    ///         the document that we wish to fetch.
    ///
    /// Returns the Document, raises ValueError if the document can't be found.
    fn doc(&self, doc_address: &DocAddress) -> PyResult<WrappedDoc> {
        let doc = self.inner.doc(doc_address.into()).map_err(to_pyerr)?;
        let named_doc = self.inner.schema().to_named_doc(&doc);
        Ok(WrappedDoc(named_doc.0))
    }
}

struct WrappedDoc(BTreeMap<String, Vec<Value>>);

fn value_to_py(
    py: Python,
    value: tv::schema::Value,
) -> Result<PyObject, PyErr> {
    match value {
        tv::schema::Value::Str(text) => Ok(text.into_object(py)),
        tv::schema::Value::U64(num) => Ok(num.into_object(py)),
        tv::schema::Value::I64(num) => Ok(num.into_object(py)),
        tv::schema::Value::F64(num) => Ok(num.into_object(py)),
        tv::schema::Value::Bytes(b) => Ok(b.to_object(py)),
        tv::schema::Value::Date(d) => {
            Ok(PyDateTime::from_timestamp(py, d.timestamp() as f64, None)?
                .into())
        }
        schema::Value::Facet(f) => {
            Ok(Facet { inner: f.clone() }.into_object(py))
        }
    }
}

impl IntoPyObject for WrappedDoc {
    fn into_object(self, py: Python) -> PyObject {
        let dict = PyDict::new(py);
        for (key, values) in self.0 {
            let values_py: Vec<PyObject> = values
                .into_iter()
                .map(|v| value_to_py(py, v).unwrap())
                .collect();
            dict.set_item(key, values_py).unwrap();
        }
        dict.into()
    }
}
/// DocAddress contains all the necessary information to identify a document
/// given a Searcher object.
///
/// It consists in an id identifying its segment, and its segment-local DocId.
/// The id used for the segment is actually an ordinal in the list of segment
/// hold by a Searcher.
#[pyclass]
pub(crate) struct DocAddress {
    pub(crate) segment_ord: tv::SegmentLocalId,
    pub(crate) doc: tv::DocId,
}

#[pymethods]
impl DocAddress {
    /// The segment ordinal is an id identifying the segment hosting the
    /// document. It is only meaningful, in the context of a searcher.
    #[getter]
    fn segment_ord(&self) -> u32 {
        self.segment_ord
    }

    /// The segment local DocId
    #[getter]
    fn doc(&self) -> u32 {
        self.doc
    }
}

impl From<&tv::DocAddress> for DocAddress {
    fn from(doc_address: &tv::DocAddress) -> Self {
        DocAddress {
            segment_ord: doc_address.segment_ord(),
            doc: doc_address.doc(),
        }
    }
}

impl Into<tv::DocAddress> for &DocAddress {
    fn into(self) -> tv::DocAddress {
        tv::DocAddress(self.segment_ord(), self.doc())
    }
}

/// The Top Score Collector keeps track of the K documents sorted by their
/// score.
///
/// Args:
///     limit (int, optional): The number of documents that the top scorer will
///         retrieve. Must be a positive integer larger than 0. Defaults to 10.
#[pyclass]
pub(crate) struct TopDocs {
    inner: tv::collector::TopDocs,
}

#[pymethods]
impl TopDocs {
    #[new]
    #[args(limit = 10)]
    fn new(obj: &PyRawObject, limit: usize) -> PyResult<()> {
        let top = tv::collector::TopDocs::with_limit(limit);
        obj.init(TopDocs { inner: top });
        Ok(())
    }
}
