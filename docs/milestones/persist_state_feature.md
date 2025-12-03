# Feature Request: Persist Application State (Last Directory, UI Settings, Reset Settings)

## Summary
Add persistent application state storage using **Qt QSettings** so that the tool remembers the last directory used on startup and other useful UI-related settings.  
This feature enhances user experience by restoring UI state across application launches.

---

# Requirements

## 1. Remember Last Used Directory (Primary Requirement)
The application must:

- Store the **last directory** the user interacted with (open file / select folder).
- On application startup, the **Open File dialog** should automatically start in this directory.
- If no stored directory exists → fall back to `$HOME`.

Only the **last directory** is stored, **not** the last file.

Stored key example:
```
[file_dialogs]
last_directory = /path/to/dir
```

---

# Additional Recommended State Persistence

## 2. Window Geometry & Placement
Persist:

- Window size  
- Window position  
- Maximized state  
- Splitter positions (for multi-pane layouts)  

This ensures the UI loads exactly as the user left it.

Suggested keys:
```
[ui]
window_geometry = <Qt byte array>
window_state = <Qt byte array>
splitter_main = <sizes>
splitter_preview = <sizes>
```

---

## 3. Reset Settings Button (Under “Settings” Panel)
Add a button:
### **“Reset All Settings to Default”**

Behavior:

1. Delete all QSettings entries  
2. Confirm with user:
   ```
   Are you sure you want to reset all application settings?
   ```
3. Restart or reload UI to default state

---

# What NOT to Remember

To avoid accidental leaks or confusing persistence, these must not be stored:

- Recent files list  
- Last-used FBIN dimension  
- Paths of previous unwrap/merge operations  
- Validation results  
- Any temporary or sensitive data  

---

# Settings Structure

Recommended grouping:

```
[general]
version = 1

[file_dialogs]
last_directory = ...

[ui]
window_geometry = ...
window_state = ...
splitter_main = ...
splitter_preview = ...

[settings]
reset_available = true
```

---

# UX Behavior

### On Startup
- Load QSettings  
- Restore window geometry  
- Restore splitters  
- Use last_directory for file dialogs  

### On Exit
- Save:
  - window geometry
  - window state
  - last_directory

### Reset Settings
- Removes all stored settings
- Shows confirmation dialog
- Restarts or resets the UI

---

# Technical Requirements

- Use Qt’s built-in `QSettings`  
- Serve all OS targets:
  - Linux (XDG paths)
  - macOS (plist)
  - Windows (Registry)

- Must work with PyInstaller builds

---

# Acceptance Criteria

- Last directory persists across runs  
- Window geometry/position restores correctly  
- Splitter sizes restore correctly  
- Reset Settings button works and clears everything  
- No unnecessary or sensitive data is stored  
- Corrupted settings trigger safe fallback to defaults  
- No regressions in startup time or UI responsiveness  

---

# Future Enhancements (Optional, not part of this task)

- Persist theme (dark/light)  
- Persist sidebar last-selected panel  
- Persist unwrap output directory  
- Persist validation configuration  

