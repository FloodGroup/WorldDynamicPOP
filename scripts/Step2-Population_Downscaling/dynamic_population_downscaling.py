# -*- coding: utf-8 -*-
"""
Dynamic Population Downscaling Module
=====================================

Downscale 1km gridded population data to 100m resolution based on
building morphology and land use function.

Input:
    - Population TIF (1km): WorldPop or similar gridded population data
    - Buildings (Shapefile): {city}_buildings_final.shp
    - Land Use (Shapefile): {city}_landuse_classified.shp

Output:
    - Population TIF (100m): Dynamic population distribution

Method:
    Pop_100m = Pop_1km * (EHP_100m / Sum(EHP_100m_in_1km))
    EHP = Sum(Time_Weight * Indoor_Weight * Effective_Floor_Area)
"""

import gc
import sys
import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple, Any

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from rasterio.transform import Affine
from shapely.geometry import box


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class DownscalingConfig:
    """Configuration parameters for the downscaling algorithm."""
    
    # Downscaling factor: 1km -> 100m
    downscale_factor: int = 10
    
    # Fine rasterization factor: 100m -> 10m
    fine_rasterize_factor: int = 10
    
    # Indoor population proportion factor M
    season_factor_M: float = 0.9
    
    # Floor area coefficients (alpha)
    alpha_residential: float = 0.95
    alpha_ci: float = 0.91
    alpha_other: float = 0.98
    
    @property
    def floor_area_coefficients(self) -> Dict[str, float]:
        return {
            'residential': self.alpha_residential,
            'c&i': self.alpha_ci,
            'education': self.alpha_other,
            'leisure': self.alpha_other,
            'other': self.alpha_other,
            'restricted': 0.0
        }
    
    category_codes: Dict[str, int] = field(default_factory=lambda: {
        'restricted': 0,
        'residential': 1,
        'c&i': 2,
        'education': 3,
        'leisure': 4,
        'other': 5
    })


# Default time-use survey weights (24h x 5 categories)
DEFAULT_EHP_WEIGHTS = {
    'hour': list(range(24)),
    'residential': [93.4, 96.4, 97.6, 97.7, 95.7, 92.5, 82.6, 67.4, 52.9, 43.7, 37.0, 33.6,
                    31.4, 31.7, 31.4, 33.5, 38.2, 40.6, 45.1, 48.8, 57.2, 67.0, 78.5, 87.0],
    'c&i':         [1.9, 1.3, 1.0, 1.1, 1.9, 3.6, 8.4, 16.3, 27.3, 33.3, 37.5, 38.2,
                    32.9, 35.7, 38.3, 36.3, 32.6, 24.6, 16.5, 11.9, 9.5, 7.6, 5.3, 3.3],
    'education':   [0.2, 0.1, 0.1, 0.0, 0.1, 0.0, 0.2, 1.0, 2.5, 3.7, 4.3, 4.6,
                    3.3, 3.6, 3.8, 2.8, 2.2, 1.8, 1.7, 1.5, 1.8, 1.3, 0.7, 0.3],
    'leisure':     [3.7, 1.7, 0.9, 0.7, 1.2, 2.2, 3.2, 5.1, 6.9, 9.1, 10.5, 11.3,
                    11.4, 13.5, 15.2, 16.5, 17.8, 18.5, 17.2, 18.9, 18.8, 17.8, 12.0, 7.6],
    'other':       [0.8, 0.5, 0.4, 0.5, 1.1, 1.7, 5.6, 10.2, 10.4, 10.2, 10.7, 12.3,
                    21.0, 15.5, 11.3, 10.9, 9.2, 14.5, 19.5, 18.9, 12.7, 6.3, 3.5, 1.8]
}


# =============================================================================
# Core Processing Class
# =============================================================================

class PopulationDownscaler:
    """Core class for spatial population downscaling."""
    
    def __init__(self, config: DownscalingConfig = None):
        self.config = config if config else DownscalingConfig()
        self.ehp_weights = pd.DataFrame(DEFAULT_EHP_WEIGHTS)
        
    def load_external_weights(self, csv_path: Path) -> None:
        """Load custom EHP weights from a CSV file."""
        if not csv_path.exists():
            raise FileNotFoundError(f"Weight file not found: {csv_path}")
        
        df = pd.read_csv(csv_path)
        col_map = {}
        for col in df.columns:
            c = col.lower().strip()
            if 'res' in c: col_map[col] = 'residential'
            elif 'c&i' in c or 'c&c' in c or 'com' in c: col_map[col] = 'c&i'
            elif 'edu' in c: col_map[col] = 'education'
            elif 'lei' in c: col_map[col] = 'leisure'
            elif 'oth' in c: col_map[col] = 'other'
        
        df = df.rename(columns=col_map)
        if len(df) < 24:
            raise ValueError("EHP weights must cover 24 hours.")
        self.ehp_weights = df
        
    def _normalize_category(self, category: Any) -> str:
        if pd.isna(category): return 'other'
        cat = str(category).lower().strip()
        if 'residential' in cat: return 'residential'
        if 'c&i' in cat or 'c&c' in cat or 'commercial' in cat: return 'c&i'
        if 'education' in cat: return 'education'
        if 'leisure' in cat: return 'leisure'
        return 'other'

    def _create_10m_affine(self, trans_1km: Affine, total_factor: int) -> Affine:
        cell_w = trans_1km.a / total_factor
        cell_h = trans_1km.e / total_factor
        return Affine(cell_w, 0, trans_1km.c, 0, cell_h, trans_1km.f)

    def _compute_effective_area(
        self, 
        buildings_gdf: gpd.GeoDataFrame, 
        landuse_gdf: gpd.GeoDataFrame, 
        width_1km: int, 
        height_1km: int, 
        transform_1km: Affine,
        restriction_mask: Optional[np.ndarray] = None
    ) -> Tuple[Dict[Tuple[int, int], np.ndarray], int, int]:
        """Compute effective floor area at 100m resolution."""
        cfg = self.config
        factor_100m = cfg.downscale_factor
        factor_10m = cfg.fine_rasterize_factor
        total_factor = factor_100m * factor_10m
        
        w_10m, h_10m = width_1km * total_factor, height_1km * total_factor
        w_100m, h_100m = width_1km * factor_100m, height_1km * factor_100m
        
        trans_10m = self._create_10m_affine(transform_1km, total_factor)

        category_grid = np.zeros((h_10m, w_10m), dtype=np.uint8)
        sput_grid = np.zeros((h_10m, w_10m), dtype=np.uint8)
        floors_grid = np.zeros((h_10m, w_10m), dtype=np.float32)
        coeff_grid = np.zeros((h_10m, w_10m), dtype=np.float32)

        valid_mask = np.ones((h_10m, w_10m), dtype=bool)
        if restriction_mask is not None:
            if restriction_mask.shape != (h_10m, w_10m):
                temp = np.ones((h_10m, w_10m), dtype=bool) 
                r_h, r_w = restriction_mask.shape
                temp[:min(h_10m, r_h), :min(w_10m, r_w)] = restriction_mask[:min(h_10m, r_h), :min(w_10m, r_w)]
                restriction_mask = temp
            valid_mask = ~restriction_mask
        
        category_grid[valid_mask] = cfg.category_codes['c&i']
        sput_grid[valid_mask] = 2
        floors_grid[valid_mask] = 1.0
        
        if not landuse_gdf.empty:
            for cat, code in cfg.category_codes.items():
                if cat == 'restricted': continue
                subset = landuse_gdf[landuse_gdf['category'] == cat]
                if not subset.empty:
                    shapes = [(g, 1) for g in subset.geometry if g and not g.is_empty]
                    if shapes:
                        mask = rasterize(shapes, out_shape=(h_10m, w_10m), transform=trans_10m, dtype=np.uint8)
                        update_locs = (mask == 1) & valid_mask
                        category_grid[update_locs] = code
                        sput_grid[update_locs] = 2
                        floors_grid[update_locs] = 1.0

        if not buildings_gdf.empty:
            for cat, code in cfg.category_codes.items():
                if cat == 'restricted': continue
                subset = buildings_gdf[buildings_gdf['category'] == cat]
                if not subset.empty:
                    coef = cfg.floor_area_coefficients.get(cat, cfg.alpha_other)
                    for fl in subset['num_floors'].unique():
                        fl_subset = subset[subset['num_floors'] == fl]
                        shapes = [(g, 1) for g in fl_subset.geometry if g and not g.is_empty]
                        if shapes:
                            mask = rasterize(shapes, out_shape=(h_10m, w_10m), transform=trans_10m, dtype=np.uint8)
                            update_locs = (mask == 1) & valid_mask
                            category_grid[update_locs] = code
                            sput_grid[update_locs] = 1
                            floors_grid[update_locs] = float(fl)
                            coeff_grid[update_locs] = coef

        fa_10m = np.where(sput_grid == 1, floors_grid * coeff_grid,
                 np.where(sput_grid == 2, 1.0, 0.0))

        combo_10m = category_grid.astype(np.int16) * 10 + sput_grid.astype(np.int16)
        
        fa_100m_results = {}
        for cat_name, cat_code in cfg.category_codes.items():
            if cat_name == 'restricted': continue
            for sp_code in [1, 2]:
                combo_id = cat_code * 10 + sp_code
                mask = (combo_10m == combo_id)
                if not np.any(mask): continue
                
                extracted_fa = np.zeros_like(fa_10m)
                extracted_fa[mask] = fa_10m[mask]
                
                aggregated = extracted_fa.reshape(
                    h_100m, factor_10m, w_100m, factor_10m
                ).sum(axis=(1, 3))
                
                if np.any(aggregated > 0):
                    fa_100m_results[(cat_code, sp_code)] = aggregated

        del category_grid, sput_grid, floors_grid, coeff_grid, fa_10m, combo_10m
        gc.collect()

        return fa_100m_results, w_100m, h_100m

    def _calculate_ehp_grid(
        self, 
        fa_results: Dict[Tuple[int, int], np.ndarray], 
        shape: Tuple[int, int], 
        hour: int
    ) -> np.ndarray:
        """Calculate the EHP grid for a given hour."""
        h_100m, w_100m = shape
        ehp_grid = np.zeros((h_100m, w_100m), dtype=np.float32)
        
        w_row = self.ehp_weights.iloc[hour]
        code_to_name = {v: k for k, v in self.config.category_codes.items()}
        
        for (cat_code, sput_code), fa_grid in fa_results.items():
            cat_name = code_to_name.get(cat_code, 'other')
            h_weight = w_row.get(cat_name, 0.0) / 100.0
            
            if sput_code == 1:
                w_factor = self.config.season_factor_M
            else:
                w_factor = 1.0 - self.config.season_factor_M
            
            ehp_grid += (h_weight * w_factor * fa_grid)
            
        return ehp_grid


# =============================================================================
# Batch Processing
# =============================================================================

def run_batch_downscaling(
    pop_dir: Path,
    building_dir: Path,
    landuse_dir: Path,
    output_dir: Path,
    config: DownscalingConfig,
    ehp_weights_csv: Optional[Path] = None
) -> Dict[str, bool]:
    """Run population downscaling for all cities in batch."""
    
    # Expected file name suffixes
    building_suffix = "_buildings_final.shp"
    landuse_suffix = "_landuse_classified.shp"
    
    processor = PopulationDownscaler(config)
    
    if ehp_weights_csv and ehp_weights_csv.exists():
        processor.load_external_weights(ehp_weights_csv)
    
    pop_dir = Path(pop_dir)
    building_dir = Path(building_dir)
    landuse_dir = Path(landuse_dir)
    output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cities = [d.name for d in pop_dir.iterdir() if d.is_dir()]
    
    if not cities:
        return {}
    
    results = {}
    
    for city in cities:
        bld_path = building_dir / f"{city}{building_suffix}"
        lnd_path = landuse_dir / f"{city}{landuse_suffix}"
        
        if not bld_path.exists() or not lnd_path.exists():
            results[city] = False
            continue
        
        city_pop_dir = pop_dir / city
        if not city_pop_dir.exists():
            results[city] = False
            continue
        
        try:
            bld_gdf = gpd.read_file(bld_path)
            lnd_gdf = gpd.read_file(lnd_path)
            
            bld_gdf['category'] = bld_gdf['category'].apply(processor._normalize_category)
            lnd_gdf['category'] = lnd_gdf['category'].apply(processor._normalize_category)
            
            if 'num_floors' not in bld_gdf.columns:
                bld_gdf['num_floors'] = 2
            bld_gdf['num_floors'] = pd.to_numeric(bld_gdf['num_floors'], errors='coerce').fillna(1).astype(int)
            
            pop_files = sorted(list(city_pop_dir.glob("*.tif")))
            if not pop_files:
                results[city] = False
                continue
            
            ref_tif = pop_files[0]
            with rasterio.open(ref_tif) as src:
                trans_1km = src.transform
                crs_1km = src.crs
                w_1km, h_1km = src.width, src.height
                bounds = src.bounds
            
            if bld_gdf.crs != crs_1km: bld_gdf = bld_gdf.to_crs(crs_1km)
            if lnd_gdf.crs != crs_1km: lnd_gdf = lnd_gdf.to_crs(crs_1km)
            
            bbox_geom = box(bounds.left, bounds.bottom, bounds.right, bounds.top)
            bld_gdf = bld_gdf[bld_gdf.intersects(bbox_geom)]
            lnd_gdf = lnd_gdf[lnd_gdf.intersects(bbox_geom)]
            
            fa_results, w_100m, h_100m = processor._compute_effective_area(
                bld_gdf, lnd_gdf, w_1km, h_1km, trans_1km
            )
            
            city_out_dir = output_dir / city
            city_out_dir.mkdir(parents=True, exist_ok=True)
            
            factor = processor.config.downscale_factor
            new_a = trans_1km.a / factor
            new_e = trans_1km.e / factor
            trans_100m = Affine(new_a, 0, trans_1km.c, 0, new_e, trans_1km.f)
            
            ehp_cache = {}
            
            for pop_file in pop_files:
                try:
                    hour_str = pop_file.stem.split('_')[-1]
                    hour_idx = int(hour_str)
                    ehp_hour = hour_idx // 2 if hour_idx >= 24 else hour_idx
                    ehp_hour = ehp_hour % 24
                except:
                    ehp_hour = 12
                
                if ehp_hour not in ehp_cache:
                    ehp_cache[ehp_hour] = processor._calculate_ehp_grid(fa_results, (h_100m, w_100m), ehp_hour)
                ehp_grid = ehp_cache[ehp_hour]
                
                with rasterio.open(pop_file) as src:
                    pop_1km = src.read(1)
                    nodata = src.nodata
                
                ehp_sum_1km = ehp_grid.reshape(h_1km, factor, w_1km, factor).sum(axis=(1, 3))
                ehp_sum_expanded = np.repeat(np.repeat(ehp_sum_1km, factor, axis=0), factor, axis=1)
                pop_expanded = np.repeat(np.repeat(pop_1km, factor, axis=0), factor, axis=1)
                
                with np.errstate(divide='ignore', invalid='ignore'):
                    pop_100m = pop_expanded * (ehp_grid / ehp_sum_expanded)
                    pop_100m = np.nan_to_num(pop_100m, nan=0.0, posinf=0.0, neginf=0.0)
                
                if nodata is not None:
                    nodata_mask = np.repeat(np.repeat((pop_1km == nodata), factor, axis=0), factor, axis=1)
                    pop_100m[nodata_mask] = -9999
                
                out_name = f"{city}_{hour_idx:02d}_100m.tif"
                out_path = city_out_dir / out_name
                
                with rasterio.open(
                    out_path, 'w', driver='GTiff',
                    height=h_100m, width=w_100m, count=1, dtype=np.float32,
                    crs=crs_1km, transform=trans_100m, nodata=-9999, compress='lzw'
                ) as dst:
                    dst.write(pop_100m, 1)
            
            results[city] = True
            
        except Exception:
            results[city] = False
    
    return results


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Downscale 1km population grids to 100m using building and land use data."
    )
    
    parser.add_argument('--pop_dir', type=str, required=True,
                        help="Directory containing city population TIF subdirectories")
    parser.add_argument('--building_dir', type=str, required=True,
                        help="Directory containing final building shapefiles")
    parser.add_argument('--landuse_dir', type=str, required=True,
                        help="Directory containing classified land use shapefiles")
    parser.add_argument('--output_dir', type=str, required=True,
                        help="Output directory for 100m population TIFs")

    parser.add_argument('--season_factor_M', type=float, default=0.9,
                        help="Indoor population proportion factor M (default: 0.9)")
    parser.add_argument('--alpha_residential', type=float, default=0.95,
                        help="Floor area coefficient for residential (default: 0.95)")
    parser.add_argument('--alpha_ci', type=float, default=0.91,
                        help="Floor area coefficient for c&i (default: 0.91)")
    parser.add_argument('--alpha_other', type=float, default=0.98,
                        help="Floor area coefficient for other categories (default: 0.98)")
    parser.add_argument('--downscale_factor', type=int, default=10,
                        help="Downscaling factor from 1km to target (default: 10)")
    parser.add_argument('--fine_rasterize_factor', type=int, default=10,
                        help="Fine rasterization factor within each target cell (default: 10)")
    parser.add_argument('--ehp_weights', type=str, default=None,
                        help="Path to custom EHP weights CSV file")
    
    args = parser.parse_args()
    
    config = DownscalingConfig(
        downscale_factor=args.downscale_factor,
        fine_rasterize_factor=args.fine_rasterize_factor,
        season_factor_M=args.season_factor_M,
        alpha_residential=args.alpha_residential,
        alpha_ci=args.alpha_ci,
        alpha_other=args.alpha_other
    )
    
    ehp_weights = Path(args.ehp_weights) if args.ehp_weights else None
    
    results = run_batch_downscaling(
        pop_dir=Path(args.pop_dir),
        building_dir=Path(args.building_dir),
        landuse_dir=Path(args.landuse_dir),
        output_dir=Path(args.output_dir),
        config=config,
        ehp_weights_csv=ehp_weights
    )
    
    sys.exit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()