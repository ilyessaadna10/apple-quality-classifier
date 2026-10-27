# Apple Photo Labeling Tool

A fast, keyboard-driven two-pass labeling web application built with Python Flask and vanilla HTML/JS/CSS (no external build step or node dependencies).

---

## 🚀 Quick Start

### 1. Requirements
Ensure Python 3.10+ is installed along with the required libraries:
```bash
pip install flask Pillow numpy opencv-python-headless scikit-image
```

### 2. Start the Labeler
From this directory:
```bash
python3 app.py
```
Then open your browser at:
👉 **[http://localhost:5000](http://localhost:5000)**

---

## ⌨️ Keyboard Controls Reference

### Navigation & Actions (Global)
| Key | Action |
| :--- | :--- |
| `Space` or `S` or `→` | **Skip** to next image without labeling |
| `←` | **Back** to previous image |
| `Backspace` or `Ctrl+Z` | **Undo** last action (restores previous state) |
| `Delete` or `X` | **Delete** (soft-delete from queues and exports; leaves original files untouched) |
| `M` | **Toggle Pass** (switch between Pass 1 and Pass 2) |
| `F` | **Cycle Queue Filter** (`all` → `unlabeled` → `unsure` → `seg_problem`) |
| `Z` | **Toggle 2x Zoom** (or use mouse wheel to zoom & drag to pan) |

---

### Pass 1: Good / Bad / Quality Flags
| Key | Action |
| :--- | :--- |
| `G` | Mark as **Good** and advance |
| `B` | Mark as **Bad** and advance |
| `U` | Mark as **Unsure** and advance |
| `Q` | Toggle **Segmentation Problem** flag (hand/fingers in cutout, cut off, etc.) |
| `D` | Toggle **Lighting Override** (`normal` ↔ `dim`) |
| `V` | Cycle **Variety Override** (`green` → `yellow` → `red`) |

---

### Pass 2: Defect Categorization (Bad Apples Only)
Only apples labeled **Bad** appear in the Pass 2 queue.
| Key | Action |
| :--- | :--- |
| `1` | Set primary defect: **rot** and advance |
| `2` | Set primary defect: **bruise** and advance |
| `3` | Set primary defect: **scab/scar** and advance |
| `4` | Set primary defect: **hole/worm** and advance |
| `5` | Set primary defect: **other** and advance |
| `Shift + 1..5` | Add **secondary defect** tag to current image |
| `G` | Reclassify image to **Good** (removes from bad queue) |
| `U` | Reclassify image to **Unsure** |

---

## 📊 Live Features & Quality Tracking
- **Live 2x2 Contingency Table**: Good/Bad × Normal/Dim counts displayed in header at all times.
- **Otsu Auto-Lightness**: Mean lightness $L^*$ computed automatically; slider in Settings allows tweaking the dim/normal threshold live.
- **Autosave & Persistence**: Saved atomically to `labels.csv` on every keypress with rolling backup `labels.csv.bak` every 20 actions. Resumes where you left off.

---

## 📦 Exporting the Dataset
Click the **📦 Export** button in the top bar or run:
```bash
python3 export.py
```
This generates:
- `dataset_export/stage1/good/` & `stage1/bad/` (gray 384×384 cutouts)
- `dataset_export/stage1_rgba/` (transparent cutouts)
- `dataset_export/stage2/<defect_primary>/` (bad apples by primary defect)
- `dataset_export/labels.csv`
- Complete quality distribution & imbalance report with bias warnings (<30 per cell).
