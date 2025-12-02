
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
