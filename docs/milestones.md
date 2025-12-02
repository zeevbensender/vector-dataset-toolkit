
# Project Milestones for the Vector Dataset Desktop Tool

Each milestone below is phrased so it can be directly used as a GitHub Issue.  
All milestones include UI layout requirements and acceptance criteria.

---

# ✅ Milestone A — “Hello World” Desktop App (Ubuntu X11)

## Goal
Create a minimal PySide6 application that runs on Ubuntu X11 with a basic window and no functional logic.

## UI Requirements
- Standard application window
- Title: “Vector Dataset Tool”
- Central label displaying “Hello, World”
- Window size: 900×600

## Acceptance Criteria
- App launches without errors
- Window renders on Ubuntu X11
- Project structure created (`src/`, `main.py`, etc.)

---

# ✅ Milestone B — Basic App Layout (Menus + Sidebar + Empty Views)

## Goal
Create the structural skeleton of the UI.

## UI Layout
### Components:
1. **Left Sidebar Navigation**
   - Home
   - Inspect
   - Convert
   - Shards & Merge
   - GT Tools
   - Settings
   - Logs

2. **Top Toolbar**
   - Open File
   - Save As
   - Refresh
   - Theme Toggle

3. **Main Content Area**
   - Currently blank / placeholder per section
   - Must switch views when sidebar items clicked

4. **Bottom Dock**
   - Placeholder for Logs panel
   - Placeholder Progress bar area

## Acceptance Criteria
- Sidebar switches central view
- Toolbar clickable (no logic yet)
- Bottom dock visible
- Skeleton ready for functional milestones

---

# ✅ Milestone C — Open FBIN and Display Metadata (Minimal Table)

## Goal
Implement ability to select an `.fbin` file, parse minimal metadata, and display it.

## UI Layout
### Left Panel:
- File picker (button + drag & drop)
- “Scan File” button

### Right Panel:
- Metadata table:
  - File path
  - Vector count
  - Dimension
  - Data type (float32)
  - Total size (bytes/MB)

- “Advanced Inspector…” button *(non-functional placeholder)*

## Acceptance Criteria
- User opens an `.fbin`
- Metadata is displayed in the right panel table
- Errors appear in bottom log dock
- Non-blocking (UI responsive during scan)

---

# ✅ Milestone D — Open H5/HDF5 and Display Metadata

## Goal
Add support for `.h5` or `.hdf5` files, including wrapped datasets (dbpedia format etc.).

## UI Layout
### Left Panel:
- File picker
- “Scan File” button

### Right Panel:
- Metadata table including:
  - Groups
  - Dataset names
  - Shapes
  - Dtypes
  - File size
- “Advanced Inspector…” button (placeholder)

## Acceptance Criteria
- App successfully reads HDF5 structure
- Metadata appears in simple table form
- Proper error messages for malformed files
- No full inspector yet (postponed)

---

# Milestone E — Merge Multiple FBIN Shards Into One

## Goal
Select multiple FBIN shard files of the same dataset and merge them into a single contiguous FBIN file.

## UI Layout
### Left Panel:
- Multi-file selector
- File list with:
  - Shard index
  - Shard size
  - Dimension (auto-detected)
- Output directory selector
- “Merge Shards” button

### Right Panel:
- Summary table of:
  - Number of shards
  - Total vectors
  - Total size
  - Output filename

### Bottom Dock:
- Progress bar with ETA
- Logs showing each shard processed

## Acceptance Criteria
- User selects ≥2 shards
- App auto-detects compatibility (dims, dtype)
- App produces merged FBIN
- Progress/logs show merge process
- Errors shown for mismatched shard metadata

---

# Milestone F — Wrap FBIN/IBIN Into HDF5 Packages

## Goal
Provide a guided workflow to wrap existing FBIN vector files (and optional IBIN ground-truth neighbors) into a single HDF5 container with consistent group/dataset naming.

## UI Layout
### Left Panel:
- Multi-file selector for input FBIN files (supports shards) and an optional IBIN file
- Output file picker (HDF5)
- Text fields / dropdowns:
  - Dataset name for vectors (default: `vectors`; can be overridden to match inspector labels such as `features`)
  - Dataset name for neighbors (default: `neighbors`; can be overridden to match inspector labels such as `distances`)
  - Compression options (None, gzip, lzf)
- "Validate Inputs" button
- "Wrap Into HDF5" button

### Right Panel:
- Preview table showing:
  - Source files detected (FBIN shards, IBIN)
  - Vector count & dimension (aggregated)
  - Neighbor count & k if IBIN provided
  - Estimated output size and compression choice
- Status area for validation messages
- Placeholder link/button: "Open in Inspector" (non-functional for this milestone)

### Bottom Dock:
- Progress bar with stages (validation → copy → finalize)
- Log messages streamed during wrapping

## Acceptance Criteria
- User can select one or more FBIN files (shards allowed) and an optional IBIN file
- App validates compatibility (matching dimensions across shards; IBIN query count matches vector count)
- HDF5 output contains:
  - `/vectors` (default) or chosen dataset name with float32 vectors
  - `/neighbors` (default) or chosen dataset name with int32 indices when IBIN provided
  - Basic metadata attributes: `vector_count`, `dimension`, `k` (if neighbors), `source_files`
- Operation runs in a background thread and streams progress/logs
- Errors for invalid inputs or write failures surface in the log dock

---

# General Implementation Notes
- All long operations must run in background threads.
- Avoid UI freezes.
- Every milestone should be easy to extend for future features.


---

# Current Status

**Completed Milestones:**
- ✅ Milestone A — "Hello World" Desktop App
- ✅ Milestone B — Basic App Layout
- ✅ Milestone C — Open FBIN and Display Metadata
- ✅ Milestone D — Open H5/HDF5 and Display Metadata
- ✅ Milestone E — Merge Multiple FBIN Shards Into One

**Additional Features Implemented:**
- ✅ FBIN Writer with atomic writes and checksum support
- ✅ FBIN ↔ NPY and FBIN → HDF5 conversions
- ✅ Advanced Inspector UI with configurable sampling
- ✅ IBIN enhanced sampling (random, strided)
