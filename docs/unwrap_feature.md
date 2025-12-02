# Feature Request: Add “Unwrap” Panel for Extracting Datasets From HDF5 Files

## Summary
Add a new panel to the left-side navigation (alongside Inspector, Converter, Wrap, Merge, etc.) called **Unwrap**.  
The purpose of this panel is to extract datasets from an `.h5` / `.hdf5` file and export them into three separate files:

- `base.fbin`
- `queries.fbin`
- `gt.ibin`

Reference implementation:  
https://github.com/qdrant/vector-db-benchmark/blob/master/dataset_reader/ann_h5_reader.py

The panel must follow the existing UI conventions of the application (left = actions, right = metadata/results, bottom = logs).

---

## UI Requirements

### Left Panel
- **File Picker Input**
  - Accepts `.h5` and `.hdf5`
  - Displays path in read-only text box
  - Supports drag & drop

- **Output Directory Selector (optional)**
  - If empty → use the directory of the input file

- **Settings**
  - **Max vectors to extract (optional)**
    - Default: extract **all** vectors
    - If provided: extract only the first N vectors from each dataset  
    - Must be consistent with the reference reader behavior

- **Buttons**
  - `Scan HDF5` — loads metadata and detects available datasets  
  - `Extract Datasets` — writes base/queries/gt

### Right Panel
- **Metadata Table**
  - file size  
  - detected dataset names  
  - shapes  
  - dtypes  
  - number of vectors  
  - groups

- **Extraction Summary**
  - Paths of generated files  
  - Vector counts  
  - Dimensionality  
  - Validation results

### Dialogs
- If output files already exist:
  - Show **warning dialog**:
    > “Output files already exist. Overwrite?”

- On extraction error:
  - Show detailed error dialog  
  - Log full traceback

### Bottom Log Panel
- Show progress:
  - scanning  
  - dataset detection  
  - extraction steps  
  - validation results  
  - errors  

---

## Functional Requirements

### 1. HDF5 Extraction Logic
Follow the behavior and conventions from `ann_h5_reader.py`:

- detect available datasets  
- support alternative dataset field names (see next section)  
- read base/train vectors  
- read query vectors  
- read ground-truth/neighbors dataset  
- follow the same GT structure conventions (neighbors, distances, shapes)

### 2. Alternative Dataset Field Names (New Requirement)
The panel must support dataset names that differ between sources.

Supported synonyms:

| Purpose | Canonical Name | Supported Alternatives |
|---------|----------------|-------------------------|
| Base vectors | `train` | `base`, `train`, `vectors`, `x` |
| Query vectors | `test` | `query`, `queries`, `test`, `q` |
| Ground truth (neighbors) | `neighbors` | `neighbors`, `neigh`, `gt`, `knn` |
| GT distances (optional) | `distances` | `distances`, `dists`, `metric` |

Rules:
- Prefer the canonical names `train`, `test`, `neighbors`, `distances`
- If multiple synonyms exist → pick the first valid match
- If the required dataset is missing:
  - Show error message  
  - Do not start extraction  

### 3. Ground-Truth Requirements
Follow conventions used in the reference code:

- **Neighbors dataset is mandatory**
- **Distances dataset is optional**
  - If available → include validation  
  - If missing → continue (same behavior as reference)

### 4. Output File Generation
Generate:

```
base.fbin
queries.fbin
gt.ibin
```

Rules:
- Output directory = user-specified OR same directory as input  
- If files exist → show overwrite warning  
- Use existing FBIN/IBIN write utilities  
- Create consistent binary structure

### 5. Validation
After writing:

- Re-open each file  
- Validate:
  - vector count  
  - dimension  
  - dtype  
  - GT row count  
- If validation fails →  
  - delete partial outputs  
  - show error dialog  
  - log error

### 6. Threading & Performance
- Extraction must run in a **background worker thread**
- UI must stay responsive  
- Display progress bar + ETA  
- Logs must stream in real time  

---

## UX Behavior

- Disable buttons while extraction is running  
- Bottom status bar shows progress percentage  
- After success:
  - Show notification: **“Extraction completed successfully.”**

---

## Acceptance Criteria

- New “Unwrap” panel appears in sidebar  
- Can load HDF5 file and view metadata  
- Supports alternative dataset names  
- Extracts base.fbin, queries.fbin, gt.ibin  
- Respects “max vectors to extract” option  
- Validates output files  
- Shows overwrite warning when needed  
- No UI freezing  
- Logs show full trace of actions  
- Behavior matches conventions of reference reader

---

## Open Questions (Resolved)

1. **Should the panel support alternative dataset field names?**  
   ✔ Yes — added to requirements.

2. **Should ground-truth datasets require both distances and neighbors?**  
   ✔ Follow reference:  
   - neighbors = required  
   - distances = optional

3. **Do we need a setting for max vectors to extract?**  
   ✔ Yes — added.  
   Default = extract full dataset.
