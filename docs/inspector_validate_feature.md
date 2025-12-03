# Feature Request: Add “Validate” Button to Inspector Panel

## Summary
Add a new **Validate** button to the **Inspector** panel.  
When clicked, it should run a full validation suite on the currently loaded file—whether **FBIN**, **IBIN**, or **HDF5**—and present structured validation results using UI/UX best practices.

Validation should run in a **background thread**, update the **progress bar & logs**, and display the results inside Inspector (summary) and in the Log panel (full details).  
Reference logic may come from:  
https://github.com/qdrant/vector-db-benchmark/blob/master/dataset_reader/ann_h5_reader.py

---

# UI Requirements

## Inspector Panel Additions

### Buttons
- **New Button:** `Validate`
  - Located under the “Scan File” button
  - Disabled until a file is selected
  - Auto-runs “Scan File” first if metadata is missing

### Results Display (Inside Inspector)
- A **validation results table** appears **below the Metadata table**
- Uses color-coded rows for severity:
  - **Green** — OK  
  - **Yellow** — Warning  
  - **Red** — Error  
  - **Bright Red** — Fatal  
  - **Gray** — Info  
- Columns:
  - **Check**
  - **Result**
  - **Details**

### Logs Panel
- Displays the **full validation report** (detailed messages, exceptions, stack traces)
- Includes timestamps for each validation step

### Optional (for later milestone)
- “Save Validation Report…” button

---

# UX Requirements

### Behavior
- Validation runs in a **background worker** (no UI freezing)
- Buttons disabled during validation
- Bottom status bar shows:
  - Validation progress (%)
  - Completion status (“Validation complete”, “Validation failed”, etc.)

### Failure Handling
- If validation encounters **FATAL** issues:
  - Show a modal error dialog
  - Validation results still populate in Inspector

### Auto-Scan
- If metadata has not been scanned:
  - `Validate` triggers `Scan File` automatically
- This avoids errors and duplicated logic

---

# Functional Requirements

The validation suite must run format-specific checks.  
If file type is unknown → detect automatically from extension.

---

# Validation Rules

## 1. FBIN Validation

### Structural
- File exists and is readable
- File size divisible by 4 (float32)
- File size > 0

### Dimension-dependent
(Only if dimension is known from metadata or user input)
- `vector_count * dimension * 4 == file_size_bytes`
- Reasonable dimension range (2–32768)
- Reasonable vector count (< 1e9 suggested limit)
- Shape matches expectations

### Sample-based
- Read first vector
- Check:
  - no NaNs
  - no ±inf
  - not all zeros
  - floats decode properly

### Cross-check (if extra files open)
- GT row count matches queries
- Dimension aligns with neighbor distance vector expectation

---

## 2. IBIN Validation

### Structural
- File size divisible by 4 (int32)
- Total integer count > 0

### Shape-related
- Rows have uniform K
- If queries known:  
  `num_queries == gt_rows`

### Content
- No negative indices
- No indices ≥ base_vector_count (if base known)
- No duplicate indices inside each row

### Optional deeper checks
- If distances exist separately → shapes must match
- Spot-check correctness (expensive, optional)

---

## 3. HDF5 Validation

### File integrity
- File can be opened via h5py
- No internal corruption errors

### Dataset presence
Support **multiple possible field names**:

| Purpose | Canonical | Alternatives |
|--------|-----------|--------------|
| Base vectors | train | base, vectors, x |
| Query vectors | test | query, queries, q |
| Neighbors | neighbors | neigh, gt, knn |
| Distances (optional) | distances | dists, metric |

### Shape rules
- Base vectors are 2D
- Query vectors are 2D
- Dimensions match across datasets
- neighbors.shape[0] == queries.shape[0]

### Dtype checks
- Vectors: float32 or float64
- Neighbors: int32 or int64
- Distances: float32/float64

### Content (sampled)
- No NaNs
- No infinities
- No all-zero rows

### Output consistency
If validated before Unwrap:
- HDF5→fbin/ibin mapping is internally consistent

---

# Background Worker Requirements

### The Validate worker must:
- Emit progress updates
- Emit per-check results (OK/WARNING/ERROR/FATAL)
- Produce a structured result object:
  ```json
  {
      "format": "fbin|ibin|hdf5",
      "errors": [],
      "warnings": [],
      "passed": [],
      "fatal": []
  }
  ```
- Send complete report to Logs panel

---

# Acceptance Criteria

- “Validate” button appears and is fully functional
- Validation summary table appears under Metadata section
- Full report appears in Logs
- UI remains responsive during validation
- Severe errors produce modal dialogs
- Validation suite covers FBIN/IBIN/HDF5 as described
- Uses ann_h5_reader conventions where applicable
- Auto-scan runs if metadata missing
- No crashes, freezes, or silent failures

---

# Notes / Future Enhancements
- Add “Save validation report”
- Add auto-run validation on file scan
- Add dimension override input for FBIN
- Extended deep numeric validation (vector norm histograms, etc.)
