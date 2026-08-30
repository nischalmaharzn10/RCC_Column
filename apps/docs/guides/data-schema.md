# Data Schema

Authoritative column contracts for raw and processed tables.  
**Cursor rule:** `.cursor/rules/data-schema.mdc`  
**Code constants:** `src/shared/schema.py`

## Units

| Quantity | Unit |
|----------|------|
| Length / displacement | mm |
| Force / load | kN |
| Stress / strength | MPa |
| Ratios | dimensionless |

## Raw properties — `data/raw/properties_rect.csv`

| Column | Type | Description |
|--------|------|-------------|
| `specimen` | str | Citation / specimen name |
| `fc_MPa` | float | Concrete compressive strength |
| `P_axial_kN` | float | Axial load |
| `b_mm` | float | Section width |
| `h_mm` | float | Section depth |
| `L_mm` | float | Length / shear span |
| `Lsplice_mm` | float | Splice length |
| `test_config` | str | e.g. `DE` |
| `db_long_mm` | float | Longitudinal bar diameter |
| `n_long_bars` | float | Number of long. bars |
| `cover_mm` | float | Cover |
| `rho_long` | float | Longitudinal reinforcement ratio |
| `fyl_MPa` | float | Long. steel yield |
| `steel_grade` | float/int | Grade label |
| `db_trans_mm` | float | Transverse bar diameter |
| `s_hoop_mm` | float | Hoop spacing |
| `rho_trans` | float | Transverse reinforcement ratio |
| `fyt_MPa` | float | Transverse steel yield |
| `failure_mode` | str/int | Failure mode code |
| `axial_load_ratio` | float | Axial load ratio |
| `source` | str | `peer` \| `ansys` (optional; default `peer`) |

## Processed — peak (`dataset_ml.csv`)

Properties columns plus:

| Column | Description |
|--------|-------------|
| `specimen_id` | Stable id for grouping |
| `peak_load_kN` | Target |
| `disp_at_peak_mm` | Displacement at peak |

## Processed — curve (`curves_backbone.csv`)

| Column | Description |
|--------|-------------|
| `specimen_id` | Group key |
| `specimen` | Display name |
| `displacement_mm` | Feature (curve mode) |
| `lateral_load_kN` | Target |
| `source` | `peer` \| `ansys` |
| (+ design features) | Joined from properties for training |

## Adding a column

1. Update `src/shared/schema.py`
2. Update this guide + `.cursor/rules/data-schema.mdc`
3. Changelog row in [platform-guide.md](../core/platform-guide.md)
4. Migrate any builders/trainers that assume the old set
