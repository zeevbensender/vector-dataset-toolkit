# Binary Vector Format Specifications

This document describes the FBIN and IBIN binary file formats used for storing vector datasets and ground truth indices.

## FBIN Format (Binary Float32 Vectors)

FBIN files store dense floating-point vectors in a compact binary format. This format is commonly used by ANN Benchmarks and Big ANN Benchmarks for storing vector embeddings.

### File Structure

```
┌──────────────────────────────────────────────┐
│ Header (8 bytes)                             │
│ ├─ Bytes 0-3: uint32 LE - number of vectors  │
│ └─ Bytes 4-7: uint32 LE - dimension          │
├──────────────────────────────────────────────┤
│ Data (num_vectors × dimension × 4 bytes)     │
│ └─ float32 LE values, row-major order        │
└──────────────────────────────────────────────┘
```

### Header Fields

| Offset | Size | Type | Description |
|--------|------|------|-------------|
| 0 | 4 bytes | uint32 LE | Number of vectors in the file |
| 4 | 4 bytes | uint32 LE | Dimension of each vector |

### Data Section

- **Element Type**: float32 (4 bytes per element)
- **Byte Order**: Little-endian
- **Layout**: Row-major (contiguous vectors)
- **Total Size**: `num_vectors × dimension × 4` bytes

### Example

For a file with 1000 vectors of dimension 128:
- Header: 8 bytes
- Data: 1000 × 128 × 4 = 512,000 bytes
- Total: 512,008 bytes

### Reading FBIN Files

```python
from src.utils.io import FBINReader

reader = FBINReader("vectors.fbin")
metadata = reader.get_metadata()
print(f"Vectors: {metadata['vector_count']}, Dim: {metadata['dimension']}")

# Load all vectors
vectors = reader.load()

# Sample first 10 vectors
sample = reader.sample(start=0, count=10)

# Random sampling
indices, vectors = reader.sample_random(count=10, seed=42)

# Strided sampling
vectors = reader.sample_strided(stride=100, max_count=100)

# Sequential reading in chunks
for chunk in reader.read_sequential(chunk_size=1000):
    process(chunk)
```

### Writing FBIN Files

```python
from src.utils.io import FBINWriter, write_fbin
import numpy as np

# Method 1: Using FBINWriter class
data = np.random.randn(1000, 128).astype(np.float32)
writer = FBINWriter("output.fbin")
result = writer.write(data, compute_checksum=True)

# Method 2: Convenience function
result = write_fbin("output.fbin", data, compute_checksum=True)
print(f"Checksum: {result['checksum']}")
```

---

## IBIN Format (Binary Int32 Indices)

IBIN files store integer index vectors, typically used for ground truth nearest neighbor indices in ANN benchmarks.

### File Structure

```
┌──────────────────────────────────────────────┐
│ Header (8 bytes)                             │
│ ├─ Bytes 0-3: uint32 LE - number of queries  │
│ └─ Bytes 4-7: uint32 LE - k (neighbors/query)│
├──────────────────────────────────────────────┤
│ Data (num_queries × k × 4 bytes)             │
│ └─ int32 LE values, row-major order          │
└──────────────────────────────────────────────┘
```

### Header Fields

| Offset | Size | Type | Description |
|--------|------|------|-------------|
| 0 | 4 bytes | uint32 LE | Number of queries/vectors |
| 4 | 4 bytes | uint32 LE | k (number of neighbors per query) |

### Data Section

- **Element Type**: int32 (4 bytes per element)
- **Byte Order**: Little-endian
- **Layout**: Row-major (contiguous index vectors)
- **Semantics**: Each row contains k neighbor indices for one query

### Reading IBIN Files

```python
from src.utils.io import IBINReader

reader = IBINReader("ground_truth.ibin")
metadata = reader.get_metadata()
print(f"Queries: {metadata['vector_count']}, k: {metadata['k']}")

# Get neighbors for a specific query
neighbors = reader.get_neighbors(query_index=42)
```

---

## Format Assumptions

The following assumptions are made by this implementation:

1. **Byte Order**: Always little-endian
2. **Data Types**: 
   - FBIN: float32 only
   - IBIN: int32 only
3. **No Magic Number**: Files do not contain a format identifier
4. **No Compression**: Data is stored uncompressed
5. **No Checksums**: File integrity is not verified by format

### Known Limitations

- No automatic endianness detection
- No support for float16/float64 variants
- No validation of index ranges in IBIN files
- File size mismatch only produces warning, not error

---

## FBIN Sharding

Large datasets may be split into multiple shard files. Each shard contains a subset of vectors with the same dimension.

### Shard Naming Convention

Common naming patterns:
- `dataset_000.fbin`, `dataset_001.fbin`, ...
- `shard-00000-of-00100.fbin`, ...
- `vectors.part00.fbin`, ...

### Merging Shards

```python
from src.utils.io import ShardMerger, merge_fbin_shards

# Validate compatibility
merger = ShardMerger()
infos = merger.validate_shards([
    "shard_0.fbin",
    "shard_1.fbin", 
    "shard_2.fbin"
])

# Preview merge
preview = merger.preview_merge(shard_paths, output_format="fbin")
print(f"Total vectors: {preview.total_vectors}")

# Perform merge
result = merger.merge(
    shard_paths,
    output_path="merged.fbin",
    output_format="fbin",
    compute_checksum=True
)
print(f"Checksum: {result['checksum']}")
```

### Merge Safety Guarantees

1. **Atomic Writes**: Output is written to a temp file, then atomically renamed
2. **No Source Modification**: Original shard files are never modified
3. **Checksum Verification**: Optional SHA256 checksum of merged output
4. **Validation**: Shards are validated for dimension compatibility before merge
5. **Cancellation Safety**: On cancellation, temp files are cleaned up

---

## Conversion Support

### FBIN ↔ NPY

```python
from src.utils.io import convert_fbin_to_npy, convert_npy_to_fbin

# FBIN to NPY
result = convert_fbin_to_npy(
    "input.fbin", 
    "output.npy",
    chunk_size=10000,
    dry_run=False
)

# NPY to FBIN
result = convert_npy_to_fbin(
    "input.npy",
    "output.fbin", 
    compute_checksum=True
)
```

### FBIN → HDF5

```python
from src.utils.io import convert_fbin_to_hdf5

result = convert_fbin_to_hdf5(
    "input.fbin",
    "output.h5",
    dataset_name="vectors",
    compression="gzip"
)
```

### Conversion Features

- **Chunked Processing**: Memory-efficient for large files
- **Progress Callbacks**: UI integration support
- **Dry-Run Mode**: Preview without writing
- **Atomic Writes**: Safe output via temp files
- **Checksum Generation**: Optional verification

---

## References

- [ANN Benchmarks](https://github.com/erikbern/ann-benchmarks) - Original format usage
- [Big ANN Benchmarks](https://big-ann-benchmarks.com/) - Large-scale benchmarks
