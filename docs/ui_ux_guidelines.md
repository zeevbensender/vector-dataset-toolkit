
# UI/UX Guidelines for the Vector Dataset Desktop Tool

## Overview
This document defines UI and UX guidelines for the Python desktop application designed to inspect, convert, merge, and manipulate FBIN, IBIN, NPY, and HDF5 datasets.  
The design emphasizes extensibility, modularity, and ease of adding new features.

---

# 1. Core UX Principles

## 1.1 Utility-first
The application focuses on clarity and productivity over aesthetics.  
All actions should be discoverable with minimal clicks.

## 1.2 Extensibility
Layouts and navigation must support frequent future expansion:
- New tools
- New dataset formats
- New tabs or panels

## 1.3 Non-blocking UI
Long-running operations (file scanning, merging) must:
- Run in background threads
- Update progress bar + logs
- Never freeze UI

---

# 2. Layout Structure

## 2.1 Main Window Regions
The application uses the following structure:

### **Left Sidebar (Navigation)**
Section entries such as:
- Home
- Inspect
- Convert
- Shards & Merge
- GT Tools
- Settings
- Logs

Each sidebar item maps to a central “view” in the main area.

### **Top Toolbar**
Contains only global actions:
- Open File
- Save As
- Refresh
- Theme Toggle
- Stop/Quit

### **Main Content Area (Center)**
Split into two panels:
- **Left Panel** — Input, settings, parameters, action buttons
- **Right Panel** — Metadata table or operation results

### **Bottom Dock**
Dedicated for:
- Logs
- Progress bars with ETA
- Error messages
- Operation history entries

---

# 3. Interaction and Component Design

## 3.1 File Selection
Must support:
- Drag & drop
- "Open File…" button
- List of recent files

## 3.2 Metadata Display (Minimal Table)
Metadata table includes:
- File path
- Format type
- Vector count
- Dimension
- Data type
- File size

Include an **"Advanced Inspector…"** button which will open the full inspector in a later milestone.

## 3.3 Operation Buttons
Buttons must follow:
- Primary action highlighted (e.g., **Scan File**)
- Secondary actions aligned below
- Disabled state when action is not applicable

## 3.4 Status & Feedback
Use non-intrusive notifications:
- Short success/failure messages
- Progress bars in bottom dock
- Streaming logs with timestamps

---

# 4. Navigation Structure

## 4.1 Sections and Their Purpose
**Home** – quick entry points and recent files  
**Inspect** – view metadata  
**Convert** – fbin/npy/hdf5 converters  
**Shards & Merge** – combine multiple FBIN shards  
**GT Tools** – generate ground truth (future)  
**Settings** – thread count, log settings, default paths  
**Logs** – full-screen log viewer

---

# 5. Visual Style

## 5.1 Themes
Provide:
- Dark mode (default)
- Light mode (optional)

## 5.2 Typography
- System UI font for all controls
- Monospace font for metadata, paths, logs

## 5.3 Icons
Use monochrome Material or Feather icons.  
Icons should appear in left sidebar + toolbar.

---

# 6. Extensibility Guidelines

## 6.1 Adding New Tools
Every new feature:
- Adds a sidebar item OR appears as a subsection inside a main area
- Uses the same “input-left / output-right” pattern
- Uses centralized logging system
- Uses background worker threads

## 6.2 Adding New File Types
Design all inspectors and converters to:
- Have a standardized metadata interface
- Support plugin-like expansions

---

# 7. Accessibility
- Buttons and controls must have clear labels
- All colors must pass basic readability (WCAG AA)

