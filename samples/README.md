# samples/ — PEER Structural Performance Database

Downloaded public RC column data for training. **Primary set: rectangular.**

## Layout

```
samples/
├── README.md
├── rectangular_properties.txt      # PEER bulk properties (tab-delimited)
├── spiral_properties.txt           # Spiral/circular properties (optional later)
├── properties/
│   ├── rectangular_properties.txt  # copy
│   ├── rectangular_properties.csv  # same table as CSV (254 rows incl. header)
│   ├── rectangular_curve_index.csv # specimen ↔ curve file map
│   └── spiral_properties.txt
├── curves/
│   └── rectangular/                # 286 force–displacement .txt files
└── docs/
    ├── about.html
    ├── datarect.htm                # UW index of rectangular tests
    ├── performance_database_manual.pdf
    └── performance_database_manual_v1-2.pdf
```

## What each file is for

| Asset | Use |
|-------|-----|
| `rectangular_properties.csv` | Features (X): geometry, materials, reinforcement, axial load |
| `curves/rectangular/*.txt` | Targets: cyclic force–displacement → peak + backbone |
| `rectangular_curve_index.csv` | Join specimen name → curve filename |
| Manuals in `docs/` | Column definitions & units |

## Curve file format (UW / PEER)

1. Row 1: specimen name  
2. Row 2: number of points  
3. Row 3+: `displacement_mm`, `lateral_load_kN` (tab-separated; optional axial column)

## Sources

- Properties: https://nisee.berkeley.edu/spd/rectangular_properties.txt  
- Curves: https://depts.washington.edu/columdat/ (via `datarect.htm` → `rectcol/txfiles/`)  
- Hub: https://nisee.berkeley.edu/spd/

## Counts (this download)

- Properties rows: **253 specimens** (+ header)  
- Curve files downloaded: **286 / 298** listed (12 links 404 — mostly Mattock & Wang dummy / Xiao missing files)  
- Spiral properties: included; spiral curve index page was not available (404)  
- Approx. size: **~15 MB**

## Cite

When publishing, cite the **original specimen papers** plus the PEER/UW Structural Performance Database (Berry & Eberhard / PEER SPD).
