# WorldDynamicPOP

A two-stage pipeline for generating and spatially downscaling dynamic (time-varying) population grids, combining synthetic mobility trajectory data with building morphology and land-use information.

## Overview

Accurate, high-resolution dynamic population data is essential for urban planning, disaster response, and transportation analysis. **WorldDynamicPOP** addresses this need through a two-stage approach:

1. **Dynamic Population Generation** — Estimates hourly population distribution at 1 km resolution by fitting a power-law model between visit density in midnight and census population, following the methodology of Deville et al. (2014) and Liu et al. (2018) (see [References](#references)).

2. **Spatial Downscaling** — Refines the 1 km estimates to 100 m resolution by distributing population according to Effective Housing Population (EHP) weights derived from building footprints, floor counts, and land-use categories.

## Project Structure

```text
WorldDynamicPOP/scripts
├── Step1-Dynamic_Population/
│   ├── grid_to_tif.py                     # Create GeoTIFF grid reference from grid.json + npy shape
│   ├── visit_time_extract.py              # Extract visit matrix [48 × n_grids] from trajectories
│   └── generate_dynamic_pop.py            # Power-law model fitting & dynamic population generation
│
├── Step2-Population_Downscaling/
│   ├── Land_cover_retrieval_and_processing.py # Land-cover extraction & classification
│   ├── Building_retrieval_and_processing.py   # Building extraction, POI linking & classification
│   ├── Building_add_height.py                 # Fill missing building heights
│   └── Dynamic_population_downscaling.py      # EHP-based downscaling (1 km → 100 m)
│
├── requirements.txt
└── README.md
```

---

## Usage

![Pipeline overview](<https://github.com/SKL-CRCC/WorldDynamicPOP/blob/main/images/dynamic_pop.png>)

The pipeline runs in order:

```bash
pip install -r requirements.txt

# Step 1: Dynamic Population Generation
python Step1-Dynamic_Population/grid_to_tif.py --grid_json_dir <grid_json_dir> --npy_dir <npy_dir> --output_dir <grid_tif_dir>
python Step1-Dynamic_Population/visit_time_extract.py --traj_dir <traj_npz_dir> --grid_dir <grid_json_dir> --output_dir <visit_out_dir>
python Step1-Dynamic_Population/generate_dynamic_pop.py --visit_dir <visit_out_dir> --pop_dir <pop_npy_dir> --output_dir <dynamic_pop_out_dir> --params_dir <params_out_dir>

# Step 2: Population Downscaling
python Step2-Population_Downscaling/Land_cover_retrieval_and_processing.py --grid_dir <grid_tif_dir> --parquet_dir <landuse_parquet_dir> --extracted_dir <landuse_extracted_dir> --classified_dir <landuse_classified_dir>
python Step2-Population_Downscaling/Building_retrieval_and_processing.py --grid_dir <grid_tif_dir> --building_parquet_dir <buildings_parquet_dir> --places_parquet_dir <places_parquet_dir> --landuse_dir <landuse_classified_dir> --extracted_dir <bld_extracted_dir> --poi_linked_dir <bld_poi_dir> --classified_dir <bld_classified_dir>
python Step2-Population_Downscaling/Building_add_height.py --grid_file <world_grid_shp> --height_dir <3dglobfp_zip_dir> --building_dir <bld_classified_dir> --output_dir <bld_final_dir>
python Step2-Population_Downscaling/Dynamic_population_downscaling.py --pop_dir <dynamic_pop_1km_tif_dir> --building_dir <bld_final_dir> --landuse_dir <landuse_classified_dir> --output_dir <dynamic_pop_100m_out_dir> --ehp_weights <ehp_weights_csv_optional>
```

## Data Sources

| Dataset | Purpose | Link |
| ------- | ------- | ---- |
| **WorldMove** | Synthetic mobility trajectories for 1 600+ cities | `https://fi.ee.tsinghua.edu.cn/worldmove/` |
| **Overture Maps** | Building footprints, POIs, and land-use data | `https://overturemaps.org/` |
| **3D-GloBFP** | Global building height data | `https://zenodo.org/records/15459025` |

## Methodology

### Stage 1 — Dynamic Population Estimation

Following Deville et al. (2014), nighttime (3:00–4:00 AM) visit density is used to calibrate a power-law relationship with census population:

$$\rho_c = \alpha \cdot \sigma_c^{\beta}$$

where $\sigma_c$ is the visit density and $\rho_c$ is the population density. The calibrated model is then applied to all 48 half-hour time slots, with each slot's total rescaled to match the census total.

### Stage 2 — Spatial Downscaling

Population in each 1 km cell is distributed to 100 m sub-cells proportional to the Effective Housing Population (EHP):

$$Pop_{100m} = Pop_{1km} \times \frac{EHP_{100m}}{\sum EHP_{100m \in 1km}}$$

$$EHP = \sum \left( W_{time} \cdot W_{indoor} \cdot S_{eff} \right)$$

where $W_{time}$ is the hourly time-use weight by land-use category, $W_{indoor}$ is the indoor/outdoor proportion factor, and $S_{eff}$ is the effective floor area computed from building footprints and floor counts.

---

## References

Papers cited for the dynamic population methodology ([Overview](#overview), [Stage 1](#stage-1--dynamic-population-estimation)):

1. Deville, P., Linard, C., Martin, S., Gilbert, M., Stevens, F. R., Gaughan, A. E., Blondel, V. D., & Tatem, A. J. (2014). Dynamic population mapping using mobile phone data. *Proceedings of the National Academy of Sciences*, *111*(45), 15888–15893. https://doi.org/10.1073/pnas.1408439111

2. Liu, Z., Ma, T., Du, Y., & Pei, T. (2018). Mapping hourly dynamics of urban population using trajectories reconstructed from mobile phone records. *Transactions in GIS*, *22*(2), 494–513. https://doi.org/10.1111/tgis.12323

---

## Citation

If you use this code in your research, please cite:

```bibtex
@software{WorldDynamicPOP,
  title  = {WorldDynamicPOP},
  author = {Zhiyong LONG, Ruiyi YANG, Huanfeng DUAN},
  year   = {2026},
  url    = {https://github.com/SKL-CRCC/WorldDynamicPOP}
}
```

## License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).
