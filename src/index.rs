use pyo3::exceptions;
use pyo3::prelude::*;

use crate::schema::Schema;
use crate::searcher::Searcher;
use crate::to_pyerr;
use pyo3::types::{PyAny, PyDict, PyList, PyTuple};
use std::collections::BTreeMap;
use tantivy as tv;
use tantivy::directory::MmapDirectory;
use tantivy::schema::{NamedFieldDocument, Value};

const RELOAD_POLICY: &str = "commit";

/// IndexWriter is the user entry-point to add documents to the index.
///
/// To create an IndexWriter first create an Index and call the writer() method
/// on the index object.
#[pyclass]
pub(crate) struct IndexWriter {
    inner: tv::IndexWriter,
}

fn extract_value(any: &PyAny) -> PyResult<Value> {
    if let Ok(s) = any.extract::<String>() {
        return Ok(Value::Str(s));
    }
    Ok(Value::U64(0u64))
}

fn extract_value_single_or_list(any: &PyAny) -> PyResult<Vec<Value>> {
    if let Ok(values) = any.downcast_ref::<PyList>() {
        values.iter().map(extract_value).collect()
    } else {
        Ok(vec![extract_value(any)?])
    }
}

#[pymethods]
impl IndexWriter {
    /// Add a document to the index.
    ///
    /// If the indexing pipeline is full, this call may block.
    ///
    /// Returns an `opstamp`, which is an increasing integer that can be used
    /// by the client to align commits with its own document queue.
    /// The `opstamp` represents the number of documents that have been added
    /// since the creation of the index.
    pub fn add_document(&mut self, py_dict: &PyDict) -> PyResult<()> {
        let mut fields: BTreeMap<String, Vec<Value>> = BTreeMap::new();
        for key_value_any in py_dict.items() {
            if let Ok(key_value) = key_value_any.downcast_ref::<PyTuple>() {
                if key_value.len() != 2 {
                    continue;
                }
                let key: String = key_value.get_item(0).extract()?;
                let value_list =
                    extract_value_single_or_list(key_value.get_item(1))?;
                fields.insert(key, value_list);
            }
        }
        self.inner
            .add_named_document(NamedFieldDocument(fields))
            .map_err(to_pyerr)?;
        Ok(())
    }

    pub fn add_json(&mut self, json: &str) -> PyResult<()> {
        self.inner.add_json(json).map_err(to_pyerr)?;
        Ok(())
    }

    /// Commits all of the pending changes
    ///
    /// A call to commit blocks. After it returns, all of the document that
    /// were added since the last commit are published and persisted.
    ///
    /// In case of a crash or an hardware failure (as long as the hard disk is
    /// spared), it will be possible to resume indexing from this point.
    ///
    /// Returns the `opstamp` of the last document that made it in the commit.
    fn commit(&mut self) -> PyResult<()> {
        let ret = self.inner.commit();
        match ret {
            Ok(_) => Ok(()),
            Err(e) => Err(exceptions::ValueError::py_err(e.to_string())),
        }
    }

    /// Rollback to the last commit
    ///
    /// This cancels all of the update that happened before after the last
    /// commit. After calling rollback, the index is in the same state as it
    /// was after the last commit.
    fn rollback(&mut self) -> PyResult<()> {
        let ret = self.inner.rollback();

        match ret {
            Ok(_) => Ok(()),
            Err(e) => Err(exceptions::ValueError::py_err(e.to_string())),
        }
    }

    /// Detect and removes the files that are not used by the index anymore.
    fn garbage_collect_files(&mut self) -> PyResult<()> {
        let ret = self.inner.garbage_collect_files();

        match ret {
            Ok(_) => Ok(()),
            Err(e) => Err(exceptions::ValueError::py_err(e.to_string())),
        }
    }

    /// The opstamp of the last successful commit.
    ///
    /// This is the opstamp the index will rollback to if there is a failure
    /// like a power surge.
    ///
    /// This is also the opstamp of the commit that is currently available
    /// for searchers.
    #[getter]
    fn commit_opstamp(&self) -> u64 {
        self.inner.commit_opstamp()
    }
}

/// Create a new index object.
///
/// Args:
///     schema (Schema): The schema of the index.
///     path (str, optional): The path where the index should be stored. If
///         no path is provided, the index will be stored in memory.
///     reuse (bool, optional): Should we open an existing index if one exists
///         or always create a new one.
///
/// If an index already exists it will be opened and reused. Raises OSError
/// if there was a problem during the opening or creation of the index.
#[pyclass]
pub(crate) struct Index {
    pub(crate) index: tv::Index,
    reader: tv::IndexReader,
}

#[pymethods]
impl Index {
    #[staticmethod]
    fn open(path: &str) -> PyResult<Index> {
        let index = tv::Index::open_in_dir(path).map_err(to_pyerr)?;
        let reader = index.reader().map_err(to_pyerr)?;
        Ok(Index { index, reader })
    }

    #[new]
    #[args(reuse = true)]
    fn new(
        obj: &PyRawObject,
        schema: &Schema,
        path: Option<&str>,
        reuse: bool,
    ) -> PyResult<()> {
        let index = match path {
            Some(p) => {
                let directory = MmapDirectory::open(p);

                let dir = match directory {
                    Ok(d) => d,
                    Err(e) => {
                        return Err(exceptions::OSError::py_err(e.to_string()))
                    }
                };

                let i = if reuse {
                    tv::Index::open_or_create(dir, schema.inner.clone())
                } else {
                    tv::Index::create(dir, schema.inner.clone())
                };

                match i {
                    Ok(index) => index,
                    Err(e) => {
                        return Err(exceptions::OSError::py_err(e.to_string()))
                    }
                }
            }
            None => tv::Index::create_in_ram(schema.inner.clone()),
        };

        let reader = index.reader().map_err(to_pyerr)?;
        obj.init(Index { index, reader });
        Ok(())
    }

    /// Create a `IndexWriter` for the index.
    ///
    /// The writer will be multithreaded and the provided heap size will be
    /// split between the given number of threads.
    ///
    /// Args:
    ///     overall_heap_size (int, optional): The total target memory usage of
    ///         the writer, can't be less than 3000000.
    ///     num_threads (int, optional): The number of threads that the writer
    ///         should use. If this value is 0, tantivy will choose
    ///         automatically the number of threads.
    ///
    /// Raises ValueError if there was an error while creating the writer.
    #[args(heap_size = 3000000, num_threads = 0)]
    fn writer(
        &self,
        heap_size: usize,
        num_threads: usize,
    ) -> PyResult<IndexWriter> {
        let writer = match num_threads {
            0 => self.index.writer(heap_size),
            _ => self.index.writer_with_num_threads(num_threads, heap_size),
        };

        match writer {
            Ok(w) => Ok(IndexWriter { inner: w }),
            Err(e) => Err(exceptions::ValueError::py_err(e.to_string())),
        }
    }

    // Configure the way .
    //
    // Args:
    //     reload_policy (str, optional): The reload policy that the
    //         IndexReader should use. Can be manual or OnCommit.
    //     num_searchers (int, optional): The number of searchers that the
    //         reader should create.
    // TODO update doc
    #[args(reload_policy = "RELOAD_POLICY", num_searchers = 0)]
    fn config_reader(
        &mut self,
        reload_policy: &str,
        num_searchers: usize,
    ) -> Result<(), PyErr> {
        let reload_policy = reload_policy.to_lowercase();
        let reload_policy = match reload_policy.as_ref() {
            "commit" => tv::ReloadPolicy::OnCommit,
            "on-commit" => tv::ReloadPolicy::OnCommit,
            "oncommit" => tv::ReloadPolicy::OnCommit,
            "manual" => tv::ReloadPolicy::Manual,
            _ => return Err(exceptions::ValueError::py_err(
                "Invalid reload policy, valid choices are: 'manual' and 'OnCommit'"
            ))
        };
        let builder = self.index.reader_builder();
        let builder = builder.reload_policy(reload_policy);
        let builder = if num_searchers > 0 {
            builder.num_searchers(num_searchers)
        } else {
            builder
        };

        self.reader = builder.try_into().map_err(to_pyerr)?;
        Ok(())
    }

    fn searcher(&self) -> Searcher {
        Searcher {
            inner: self.reader.searcher(),
        }
    }

    /// Check if the given path contains an existing index.
    /// Args:
    ///     path: The path where tantivy will search for an index.
    ///
    /// Returns True if an index exists at the given path, False otherwise.
    ///
    /// Raises OSError if the directory cannot be opened.
    #[staticmethod]
    fn exists(path: &str) -> PyResult<bool> {
        let directory = MmapDirectory::open(path);
        let dir = match directory {
            Ok(d) => d,
            Err(e) => return Err(exceptions::OSError::py_err(e.to_string())),
        };
        Ok(tv::Index::exists(&dir))
    }

    /// The schema of the current index.
    #[getter]
    fn schema(&self) -> Schema {
        let schema = self.index.schema();
        Schema { inner: schema }
    }

    /// Update searchers so that they reflect the state of the last .commit().
    ///
    /// If you set up the the reload policy to be on 'commit' (which is the
    /// default) every commit should be rapidly reflected on your IndexReader
    /// and you should not need to call reload() at all.
    fn reload(&self) -> PyResult<()> {
        self.reader.reload().map_err(to_pyerr)
    }
}
