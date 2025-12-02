# UI Components Documentation

This document describes the UI components and panels in the Vector Dataset Toolkit.

## Inspector Panel

The Inspector panel allows you to examine vector dataset files and view their metadata.

### Basic Inspector

**Location**: Sidebar → Inspector

**Features**:
- Open files: NPY, HDF5 (.h5/.hdf5), FBIN, IBIN
- View metadata table: vector count, dimension, dtype, file size
- Preview sample vectors (first 10)
- Dataset selection for HDF5 files with multiple datasets

### Advanced Inspector

**Access**: Inspector panel → "Advanced Inspector..." button

**Features**:

#### Metadata Table
- Complete field listing with descriptions
- Export metadata as JSON file
- Copy metadata to clipboard

#### Format Details
- Format-specific information
- Header structure documentation
- File layout description

#### Sampling Controls

| Mode | Description | Options |
|------|-------------|---------|
| First N | Sequential sampling from start | Start index, count |
| Random | Random sampling with optional seed | Count, seed |
| Strided | Evenly spaced sampling | Stride, max count |

---

## Converter Panel

The Converter panel enables format conversion between vector dataset files.

**Location**: Sidebar → Converter

### Supported Conversions

| Input | Output | Notes |
|-------|--------|-------|
| NPY | HDF5 | Supports compression (gzip, lzf) |
| NPY | FBIN | Float32 output |
| HDF5 | NPY | Specify dataset path |
| FBIN | NPY | Memory-efficient chunked conversion |
| FBIN | HDF5 | Supports compression |

### Options

- **Chunk Size**: Number of vectors per processing chunk (100 - 100,000)
- **Compression**: gzip, lzf, or None (HDF5 only)
- **Dataset Name**: Name for the dataset in HDF5 output

### Progress

- Progress bar showing percentage complete
- Vector count progress (e.g., "50,000 / 100,000 vectors")
- Cancel button for long-running conversions

---

## Merge Panel

The Merge panel allows you to combine multiple FBIN shard files into a single contiguous file.

**Location**: Sidebar → Merge

### Workflow

1. **Add Shards**: Click "Add Shards..." to select multiple FBIN files
2. **Validation**: Shards are automatically validated for compatibility
3. **Preview**: Click "Dry Run / Preview" to see expected results
4. **Merge**: Click "Merge Shards" to perform the merge

### Shard Table

| Column | Description |
|--------|-------------|
| File | Shard filename |
| Vectors | Number of vectors in shard |
| Dimension | Vector dimension |
| Status | Validation result (Compatible/Incompatible) |

### Merge Options

- **Output File**: Destination path for merged file
- **Output Format**: FBIN or NPY
- **Chunk Size**: Vectors per processing chunk
- **Compute Checksum**: Generate SHA256 hash of output

### Preview Information

The preview shows:
- Total vectors across all shards
- Expected output file size
- Compatibility status
- Any warnings

### Safety Features

- Dry-run mode to preview without writing
- Atomic writes (temp file + rename)
- Checksum verification
- Cancel button with clean abort

---

## Logs Panel

The Logs panel displays application activity and messages.

**Location**: Sidebar → Logs

### Features

- Timestamped log entries
- Color-coded severity (INFO/WARNING/ERROR)
- Clear logs button
- Copy to clipboard button

### Quick Log

A compact log view is always visible at the bottom of the main window.

---

## Settings Panel

The Settings panel provides configuration options.

**Location**: Sidebar → Settings

### Performance Settings

- **Worker Threads**: Number of concurrent worker threads (1-16)
- **Default Chunk Size**: Default chunk size for operations

### Appearance Settings

- **Theme**: Dark, Light, or System

### Logging Settings

- **Verbose Logging**: Enable detailed log messages
- **Write Logs to File**: Save logs to disk

> Note: Settings persistence is planned for a future milestone.

---

## Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| Ctrl+O | Open file |
| Ctrl+R | Refresh current view |
| Escape | Cancel current operation |

---

## Error Handling

The application provides clear error messages for common issues:

- **File Not Found**: Check file path and permissions
- **Unsupported Format**: Verify file extension and format
- **Dimension Mismatch**: Ensure all shards have same dimension
- **Insufficient Shards**: At least 2 shards required for merge
- **Memory Error**: Reduce chunk size or use memory-mapping
