# Vector Dataset Toolkit - Source Directory

This directory contains the main source code for the Vector Dataset Toolkit application.

## Module Structure

```
src/
├── app.py              # Main PySide6 application entry point
├── cli.py              # Command-line interface for headless operations
├── ui/                 # Main UI components
│   └── main_window.py  # MainWindow class with sidebar, toolbar, status bar
├── views/              # View widgets for each sidebar section
│   ├── inspector_view.py   # File inspection and metadata display
│   ├── converter_view.py   # Format conversion interface
│   ├── merge_view.py       # Shard merging (placeholder)
│   ├── logs_view.py        # Application log viewer
│   └── settings_view.py    # Settings configuration
├── widgets/            # Reusable PySide6 widgets (for future use)
├── workers/            # Background thread infrastructure
│   └── worker.py       # Worker and WorkerManager for async operations
└── utils/
    └── io/             # File I/O utilities
        ├── npy_reader.py   # NumPy .npy file reader
        ├── hdf5_reader.py  # HDF5 file reader
        ├── fbin_reader.py  # FBIN (binary float32) reader
        ├── ibin_reader.py  # IBIN (binary int32) reader
        └── converter.py    # Format conversion core
```

## Running the Application

### GUI Application

```bash
# From the repository root
python -m src.app
```

### Command-Line Interface

```bash
# Display file information
python -m src.cli info dataset.npy

# Convert NPY to HDF5
python -m src.cli convert input.npy output.h5

# Convert HDF5 to NPY
python -m src.cli convert input.h5 output.npy --dataset vectors

# Sample vectors from a file
python -m src.cli sample dataset.fbin --start 0 --count 10
```

## Key Components

### I/O Readers

All readers follow a common pattern:
- `get_metadata()` - Extract file metadata without loading all data
- `sample(start, count)` - Sample vectors from the dataset
- `load()` - Load the full dataset (with optional memory-mapping)

#### NPYReader
Reads NumPy .npy files with memory-mapping support for large files.

#### HDF5Reader  
Reads HDF5 files, supports listing datasets/groups and accessing individual datasets.

#### FBINReader / IBINReader
Reads binary vector formats used in ANN benchmarks:
- FBIN: float32 vectors with 8-byte header (count, dimension)
- IBIN: int32 indices with 8-byte header (count, k)

**Note:** These implementations are based on common FBIN/IBIN format conventions.
See the module docstrings for format specifications and TODOs for any incomplete features.

### Converter

The `Converter` class provides chunked, memory-efficient conversion between formats:
- NPY → HDF5 with optional compression
- HDF5 → NPY

Conversion supports:
- Configurable chunk size for memory efficiency
- Progress callbacks for UI integration
- Cancellation support

### Background Workers

The `Worker` and `WorkerManager` classes provide:
- QRunnable-based background task execution
- Progress, result, and error signals
- Cancellation support
- Thread pool management

## Testing

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_npy_reader.py -v
```

## Future Milestones

- **Merge View**: Merge multiple FBIN shards into a single file
- **Settings Persistence**: Save and restore application settings
- **Advanced Inspector**: Deep inspection of vector statistics
- **Additional Formats**: Support for more vector dataset formats
