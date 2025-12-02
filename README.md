# Vector Dataset Tool

A desktop application for inspecting, converting, and manipulating vector dataset files such as **FBIN**, **IBIN**, **NPY**, and **HDF5**.  
Built with **PySide6** for a clean, extensible, and non‑blocking UX.

---

## 🚀 Features (Planned)

- Inspect FBIN, IBIN, NPY, and HDF5 metadata  
- Minimal metadata table with optional advanced inspector (future milestone)  
- Convert between formats (FBIN ↔ NPY, HDF5 ↔ NPY, etc.)  
- Merge multiple FBIN shards into a single dataset  
- Threaded operations with progress bar + logs  
- Extensible sidebar-based UI for future tools  

---

## 📦 Tech Stack

- **Python 3.10+**
- **PySide6 (Qt for Python)**
- **h5py** for HDF5 inspection
- **NumPy** for vector parsing
- **Custom utilities** for FBIN/IBIN reading

---

## 📁 Repository Structure (initial suggestion)

```
vector-dataset-tool/
│
├── src/
│   ├── app.py               # Main PySide6 application entrypoint
│   ├── ui/                  # Qt .ui files (if using Qt Designer)
│   ├── views/               # Views for sidebar sections
│   ├── widgets/             # Reusable PySide6 widgets
│   ├── workers/             # Background threads for long operations
│   └── utils/               # File readers: fbin, hdf5, ibin, npy
│
├── docs/
│   ├── ui_ux_guidelines.md
│   └── milestones.md
│
├── README.md                # This file
└── requirements.txt
```

---

## 🧭 Milestones

See:  
- `docs/ui_ux_guidelines.md`  
- `docs/milestones.md`

Both documents are written to be easily used as **GitHub Issues** or **Copilot prompts**.

---

## ▶️ Running the App

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the desktop application

```bash
python -m src.app
```

(Exact command may change once final project structure is set.)

---

## 💡 Contributing

Contributions, ideas, and feature requests are welcome.  
The UI is designed to make it easy to add new dataset formats, tools, and panels.

---

## 📜 License

MIT (or choose another license before publishing publicly)

---

## ✨ Author

Created by **Zeev Ben‑Sender** as part of tooling for vector dataset research and manipulation.

