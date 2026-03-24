#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Land Cover Retrieval and Processing Module

This module extracts land cover data from Overture Maps Parquet files and 
classifies features into functional categories for dynamic population downscaling.

Processing Pipeline:
    Step 1: Extract land cover from Parquet based on grid extents
    Step 2: Classify features into categories (residential, education, leisure, c&c, other)

Classification Categories:
    - residential: Housing and accommodation facilities
    - education: Educational institutions
    - leisure: Recreation, entertainment, and religious facilities  
    - c&c (Commercial & Civic): Business, industrial, and public services (also as work place)
    - other: Features not matching above categories

Data Sources:
    - Overture Maps Foundation: https://overturemaps.org/

"""

import argparse
import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from shapely.geometry import shape
from shapely import wkt, wkb

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# Classification Rules
# =============================================================================

CATEGORY_RULES: Dict[str, List[str]] = {
    'residential': [
        'residential', 'apartment', 'cabin', 'duplex', 'triplex', 'military',
        'apartments', 'house', 'detached', 'semidetached_house', 'semi',
        'prison', 'flat', 'condo', 'bungalow', 'hotel', 'hostel', 'dormitory',
        'farm', 'garage', 'manor', 'mansion', 'villa', 'estate', 'terrace',
        'terraced'
    ],
    'education': [
        'education', 'school', 'university', 'college', 'institute', 'academy',
        'research', 'laboratory', 'lab', 'kindergarten', 'preschool', 'nursery',
        'primary', 'elementary', 'secondary', 'vocational', 'technical',
        'campus', 'training', 'tutorial', 'graduate', 'doctoral'
    ],
    'leisure': [
        'library', 'stadium', 'cafe', 'sports_centre', 'swimming_hall',
        'recreation', 'entertainment', 'sauna', 'museum', 'sport',
        'horse_arena', 'event_space', 'pavilion', 'hall', 'religious',
        'horticulture', 'cathedral', 'church', 'chapel', 'social_facility',
        'hut', 'shed', 'workshop', 'golf', 'park', 'cinema', 'play_hut',
        'playhut', 'manege', 'outhouse', 'bird_hide', 'playground'
    ],
    'c&c': [
        'industrial', 'warehouse', 'office', 'public', 'civic',
        'public_building', 'childcare', 'townhouse', 'construction', 'utility',
        'roundhouse', 'commercial', 'hospital', 'logistics', 'storage',
        'hangar', 'farm_auxiliary', 'parking', 'train_station', 'station',
        'transportation', 'underground_entrance', 'cowshed', 'barn', 'stable',
        'silo', 'stables', 'guard_booth', 'guard', 'manufacture', 'retail',
        'shop', 'mall', 'service', 'carwash', 'store', 'supermarket', 'kiosk'
    ]
}

CATEGORY_PRIORITY: List[str] = ['residential', 'education', 'leisure', 'c&c']
LANDUSE_COLUMNS: List[str] = ['subtype', 'class']


# =============================================================================
# Path Configuration
# =============================================================================

@dataclass
class PathConfig:
    """Configuration for input/output paths.
    
    Attributes:
        grid_dir: Directory containing TIF grid files.
        parquet_dir: Directory containing Overture landuse Parquet files.
        extracted_dir: Output directory for extracted landuse data.
        classified_dir: Output directory for classified landuse data.
        target_crs: Target coordinate reference system.
    """
    
    grid_dir: Path
    parquet_dir: Path
    extracted_dir: Path
    classified_dir: Path
    target_crs: str = "EPSG:4326"
    
    def __post_init__(self):
        self.grid_dir = Path(self.grid_dir)
        self.parquet_dir = Path(self.parquet_dir)
        self.extracted_dir = Path(self.extracted_dir)
        self.classified_dir = Path(self.classified_dir)
    
    def validate(self) -> bool:
        """Validate input directories exist."""
        if not self.grid_dir.exists():
            logger.error(f"Grid directory not found: {self.grid_dir}")
            return False
        if not self.parquet_dir.exists():
            logger.error(f"Parquet directory not found: {self.parquet_dir}")
            return False
        return True
    
    def create_output_dirs(self):
        """Create output directories."""
        self.extracted_dir.mkdir(parents=True, exist_ok=True)
        self.classified_dir.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Utility Functions
# =============================================================================

def safe_json_dumps(obj: Any) -> Optional[str]:
    """Safely serialize object to JSON string."""
    if obj is None:
        return None
    try:
        if isinstance(obj, (list, dict)) and len(obj) == 0:
            return None
        if isinstance(obj, np.ndarray):
            obj = obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            obj = obj.item()
        return json.dumps(obj, ensure_ascii=False)
    except:
        return str(obj) if obj is not None else None


def safe_to_string(x: Any, max_len: int = 254) -> Optional[str]:
    """Safely convert value to string with length limit."""
    if x is None:
        return None
    try:
        if isinstance(x, np.ndarray):
            x = x.tolist()
        if isinstance(x, (dict, list)):
            s = json.dumps(x, ensure_ascii=False)
        else:
            s = str(x)
        return s[:max_len] if len(s) > max_len else s
    except:
        return None


def is_complex_type(val: Any) -> bool:
    """Check if value is a complex type requiring serialization."""
    return isinstance(val, (dict, list, np.ndarray))


def get_column_mapping(columns: List[str]) -> Dict[str, str]:
    """Generate column name mapping for Shapefile (10 char limit)."""
    short_names = {
        'subtype': 'subtype', 'class': 'class', 'surface': 'surface',
        'wikidata': 'wikidata', 'names': 'names', 'source_tags': 'src_tags',
        'sources': 'sources', 'center_x': 'center_x', 'center_y': 'center_y',
        'area_m2': 'area_m2', 'num_pts': 'num_pts', 'geometry_wkt': 'geom_wkt',
        'name_primary': 'name_pri', 'name_common': 'name_comm', 'category': 'category',
    }
    
    mapping = {}
    for col in columns:
        if col == 'geometry' or len(col) <= 10:
            continue
        short = short_names.get(col, col[:10])
        counter = 1
        original_short = short
        while short in mapping.values():
            short = original_short[:8] + f"{counter:02d}"
            counter += 1
        mapping[col] = short
    
    return mapping


# =============================================================================
# Spatial Functions
# =============================================================================

def get_tif_bbox(tif_path: Path) -> Dict[str, float]:
    """Get bounding box from TIF file."""
    with rasterio.open(tif_path) as src:
        bounds = src.bounds
        return {
            'minx': bounds.left, 'miny': bounds.bottom,
            'maxx': bounds.right, 'maxy': bounds.top
        }


def get_center_from_bbox(bbox: Dict) -> Tuple[Optional[float], Optional[float]]:
    """Calculate center point from bbox dictionary."""
    if not bbox or not isinstance(bbox, dict):
        return None, None
    try:
        if 'xmin' in bbox:
            return (bbox['xmin'] + bbox['xmax']) / 2, (bbox['ymin'] + bbox['ymax']) / 2
        elif 'minx' in bbox:
            return (bbox['minx'] + bbox['maxx']) / 2, (bbox['miny'] + bbox['maxy']) / 2
        return None, None
    except:
        return None, None


def convert_to_geodataframe(df: pd.DataFrame) -> Optional[gpd.GeoDataFrame]:
    """Convert DataFrame with geometry column to GeoDataFrame."""
    if 'geometry' not in df.columns or len(df) == 0:
        return None
    
    try:
        geometries = []
        sample_geom = df['geometry'].iloc[0]
        
        if isinstance(sample_geom, dict):
            geometries = [shape(g) if isinstance(g, dict) else None for g in df['geometry']]
        elif isinstance(sample_geom, str):
            geometries = [wkt.loads(g) if isinstance(g, str) else None for g in df['geometry']]
        elif isinstance(sample_geom, bytes):
            geometries = [wkb.loads(g) if isinstance(g, bytes) else None for g in df['geometry']]
        else:
            return None
        
        gdf = gpd.GeoDataFrame(
            df.drop(columns=['geometry']).reset_index(drop=True),
            geometry=geometries, crs="EPSG:4326"
        )
        gdf = gdf[gdf.geometry.notna()]
        return gdf if len(gdf) > 0 else None
    except Exception as e:
        logger.error(f"GeoDataFrame conversion failed: {e}")
        return None


def parse_names(names: Any) -> Tuple[str, str]:
    """Parse names field from Overture data."""
    if names is None:
        return '', ''
    if isinstance(names, np.ndarray):
        names = names.tolist()
    
    try:
        if isinstance(names, dict):
            primary = names.get('primary', '')
            common = names.get('common')
            common_str = '; '.join(f"{k}:{v}" for k, v in common.items()) if isinstance(common, dict) else ''
            return primary, common_str
        elif isinstance(names, str):
            return names, ''
        elif isinstance(names, list) and names:
            first = names[0]
            return (first.get('primary', '') if isinstance(first, dict) else str(first)), ''
    except:
        pass
    return '', ''


def parse_sources(sources: Any) -> str:
    """Parse sources field from Overture data."""
    if sources is None:
        return ''
    if isinstance(sources, np.ndarray):
        sources = sources.tolist()
    
    try:
        if isinstance(sources, list):
            parts = []
            for src in sources[:3]:
                if isinstance(src, dict):
                    parts.append(f"{src.get('dataset', '')}:{src.get('record_id', '')}")
                elif isinstance(src, str):
                    parts.append(src)
            return '; '.join(parts)
        return str(sources) if sources else ''
    except:
        return ''


# =============================================================================
# Classification Functions
# =============================================================================

def classify_value(value: Any) -> Optional[str]:
    """Classify a single value based on keyword matching."""
    if pd.isna(value) or value is None:
        return None
    
    value_lower = str(value).lower()
    for category in CATEGORY_PRIORITY:
        for keyword in CATEGORY_RULES[category]:
            if keyword in value_lower:
                return category
    return None


def classify_landuse_row(row: pd.Series, columns: List[str]) -> str:
    """Classify a landuse row using specified columns in priority order."""
    for col in columns:
        if col in row.index:
            result = classify_value(row[col])
            if result is not None:
                return result
    return 'other'


# =============================================================================
# Step 1: Extraction
# =============================================================================

def extract_landuse_for_grid(
    tif_path: Path,
    parquet_dir: Path,
    output_dir: Path,
    target_crs: str
) -> int:
    """
    Extract landuse data for a single grid.
    
    Args:
        tif_path: Path to TIF grid file
        parquet_dir: Directory containing Parquet files
        output_dir: Output directory
        target_crs: Target CRS
        
    Returns:
        Number of features extracted
    """
    tif_name = tif_path.stem
    logger.info(f"Extracting: {tif_name}")
    
    # Get grid bbox
    bbox = get_tif_bbox(tif_path)
    
    # Find matching features from all Parquet files
    parquet_files = list(parquet_dir.glob("*.parquet"))
    matched_features = []
    
    for pf in parquet_files:
        try:
            # Try center columns first, fallback to bbox
            try:
                df = pd.read_parquet(pf, columns=['center_x', 'center_y'])
                has_center = True
            except:
                df = pd.read_parquet(pf, columns=['bbox'])
                has_center = False
            
            for idx in range(len(df)):
                if has_center:
                    cx, cy = df['center_x'].iloc[idx], df['center_y'].iloc[idx]
                    if pd.isna(cx) or pd.isna(cy):
                        continue
                else:
                    cx, cy = get_center_from_bbox(df['bbox'].iloc[idx])
                    if cx is None:
                        continue
                
                if bbox['minx'] <= cx <= bbox['maxx'] and bbox['miny'] <= cy <= bbox['maxy']:
                    matched_features.append((pf, idx))
        except Exception as e:
            logger.warning(f"Error reading {pf.name}: {e}")
    
    if not matched_features:
        logger.info(f"  No features found")
        return 0
    
    # Group by file and read full data
    file_indices = defaultdict(list)
    for pf, idx in matched_features:
        file_indices[pf].append(idx)
    
    all_gdfs = []
    for pf, indices in file_indices.items():
        try:
            full_df = pd.read_parquet(pf)
            matched_df = full_df.iloc[indices].copy()
            gdf = convert_to_geodataframe(matched_df)
            if gdf is not None and len(gdf) > 0:
                all_gdfs.append(gdf)
        except Exception as e:
            logger.warning(f"Error processing {pf.name}: {e}")
    
    if not all_gdfs:
        return 0
    
    gdf = pd.concat(all_gdfs, ignore_index=True)
    
    # Filter to polygons only
    valid_types = ['Polygon', 'MultiPolygon']
    gdf = gdf[gdf.geometry.notna()]
    gdf = gdf[gdf.geometry.apply(lambda g: g.geom_type in valid_types if g else False)]
    
    if len(gdf) == 0:
        return 0
    
    # CRS transformation
    if target_crs != "EPSG:4326":
        try:
            gdf = gdf.to_crs(target_crs)
        except:
            pass
    
    # Add computed fields
    if target_crs == "EPSG:4326":
        gdf['area_m2'] = gdf.geometry.apply(
            lambda g: g.area * 111320 * 111320 * abs(np.cos(np.radians(g.centroid.y))) if g else None
        )
    else:
        gdf['area_m2'] = gdf.geometry.area
    
    if 'center_x' not in gdf.columns:
        gdf['center_x'] = gdf.geometry.apply(lambda g: round(g.centroid.x, 6) if g else None)
        gdf['center_y'] = gdf.geometry.apply(lambda g: round(g.centroid.y, 6) if g else None)
    
    gdf['num_pts'] = gdf.geometry.apply(
        lambda g: len(g.exterior.coords) if g and hasattr(g, 'exterior') else
                  sum(len(p.exterior.coords) for p in g.geoms) if hasattr(g, 'geoms') else None
    )
    
    # Parse complex fields
    if 'names' in gdf.columns:
        parsed = gdf['names'].apply(parse_names)
        gdf['name_pri'] = [n[0] for n in parsed]
        gdf['name_comm'] = [n[1] for n in parsed]
    
    if 'sources' in gdf.columns:
        gdf['sources'] = gdf['sources'].apply(parse_sources)
    
    # Save CSV
    csv_path = output_dir / f"{tif_name}_landuse.csv"
    csv_df = gdf.copy()
    csv_df['geometry_wkt'] = csv_df.geometry.apply(lambda g: g.wkt if g else None)
    csv_df = csv_df.drop(columns=['geometry'])
    
    for col in csv_df.columns:
        if csv_df[col].dtype == 'object':
            non_null = csv_df[col].dropna()
            if len(non_null) > 0 and is_complex_type(non_null.iloc[0]):
                csv_df[col] = csv_df[col].apply(safe_json_dumps)
    
    csv_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    
    # Save Shapefile
    shp_gdf = gdf.copy()
    for col in shp_gdf.columns:
        if col == 'geometry':
            continue
        if shp_gdf[col].dtype == 'object':
            non_null = shp_gdf[col].dropna()
            if len(non_null) > 0 and is_complex_type(non_null.iloc[0]):
                shp_gdf[col] = shp_gdf[col].apply(lambda x: safe_to_string(x, 254))
    
    col_mapping = get_column_mapping(shp_gdf.columns.tolist())
    if col_mapping:
        shp_gdf = shp_gdf.rename(columns=col_mapping)
    
    shp_path = output_dir / f"{tif_name}_landuse.shp"
    shp_gdf.to_file(shp_path, encoding='utf-8')
    
    logger.info(f"  Saved {len(gdf)} features")
    return len(gdf)


# =============================================================================
# Step 2: Classification
# =============================================================================

def classify_landuse_file(input_path: Path, output_path: Path) -> Dict[str, int]:
    """
    Classify a landuse Shapefile.
    
    Args:
        input_path: Input Shapefile path
        output_path: Output Shapefile path
        
    Returns:
        Category counts dictionary
    """
    logger.info(f"Classifying: {input_path.name}")
    
    gdf = gpd.read_file(input_path)
    
    # Find available classification columns
    available_cols = [col for col in LANDUSE_COLUMNS if col in gdf.columns]
    if not available_cols:
        logger.warning(f"  No classification columns found, assigning 'other'")
        gdf['category'] = 'other'
    else:
        gdf['category'] = gdf.apply(lambda row: classify_landuse_row(row, available_cols), axis=1)
    
    # Get statistics
    category_counts = gdf['category'].value_counts().to_dict()
    
    # Save
    gdf.to_file(output_path, encoding='utf-8')
    
    logger.info(f"  Classified {len(gdf)} features")
    for cat, count in sorted(category_counts.items()):
        logger.info(f"    {cat}: {count}")
    
    return category_counts


# =============================================================================
# Main Processing
# =============================================================================

def process_all(config: PathConfig) -> Dict[str, Any]:
    """
    Run complete processing pipeline.
    
    Args:
        config: Path configuration
        
    Returns:
        Processing statistics
    """
    if not config.validate():
        raise ValueError("Invalid configuration")
    
    config.create_output_dirs()
    
    stats = {
        'extraction': {'total': 0, 'success': 0, 'features': 0},
        'classification': {'total': 0, 'success': 0, 'features': 0, 'categories': {}}
    }
    
    # ===================
    # Step 1: Extraction
    # ===================
    logger.info("=" * 60)
    logger.info("Step 1: Extracting land cover from Parquet files")
    logger.info("=" * 60)
    
    tif_files = list(config.grid_dir.glob("*.tif"))
    stats['extraction']['total'] = len(tif_files)
    
    for i, tif_path in enumerate(tif_files, 1):
        logger.info(f"[{i}/{len(tif_files)}] {tif_path.name}")
        count = extract_landuse_for_grid(
            tif_path, config.parquet_dir, config.extracted_dir, config.target_crs
        )
        if count > 0:
            stats['extraction']['success'] += 1
            stats['extraction']['features'] += count
    
    # ===================
    # Step 2: Classification
    # ===================
    logger.info("")
    logger.info("=" * 60)
    logger.info("Step 2: Classifying extracted land cover")
    logger.info("=" * 60)
    
    shp_files = list(config.extracted_dir.glob("*_landuse.shp"))
    stats['classification']['total'] = len(shp_files)
    
    for i, shp_path in enumerate(shp_files, 1):
        city_name = shp_path.stem.replace('_landuse', '')
        output_path = config.classified_dir / f"{city_name}_landuse_classified.shp"
        
        logger.info(f"[{i}/{len(shp_files)}] {shp_path.name}")
        try:
            category_counts = classify_landuse_file(shp_path, output_path)
            stats['classification']['success'] += 1
            stats['classification']['features'] += sum(category_counts.values())
            
            for cat, count in category_counts.items():
                stats['classification']['categories'][cat] = \
                    stats['classification']['categories'].get(cat, 0) + count
        except Exception as e:
            logger.error(f"  Error: {e}")
    
    # Save summary
    summary_path = config.classified_dir / "_processing_summary.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)
    
    return stats


def print_summary(stats: Dict[str, Any]):
    """Print processing summary."""
    print("\n" + "=" * 60)
    print("Processing Summary")
    print("=" * 60)
    
    ext = stats['extraction']
    print(f"\nStep 1 - Extraction:")
    print(f"  Grids processed: {ext['success']}/{ext['total']}")
    print(f"  Features extracted: {ext['features']:,}")
    
    cls = stats['classification']
    print(f"\nStep 2 - Classification:")
    print(f"  Files processed: {cls['success']}/{cls['total']}")
    print(f"  Features classified: {cls['features']:,}")
    
    if cls['categories']:
        print(f"\n  Category Distribution:")
        total = cls['features']
        for cat in CATEGORY_PRIORITY + ['other']:
            if cat in cls['categories']:
                count = cls['categories'][cat]
                pct = count / total * 100 if total > 0 else 0
                print(f"    {cat}: {count:,} ({pct:.1f}%)")


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract and classify land cover data from Overture Maps Parquet files."
    )
    parser.add_argument('--grid_dir', type=str, required=True,
                        help="Directory containing TIF grid files")
    parser.add_argument('--parquet_dir', type=str, required=True,
                        help="Directory containing Overture landuse Parquet files")
    parser.add_argument('--extracted_dir', type=str, required=True,
                        help="Output directory for extracted landuse data")
    parser.add_argument('--classified_dir', type=str, required=True,
                        help="Output directory for classified landuse data")
    parser.add_argument('--target_crs', type=str, default="EPSG:4326",
                        help="Target CRS (default: EPSG:4326)")

    args = parser.parse_args()

    config = PathConfig(
        grid_dir=Path(args.grid_dir),
        parquet_dir=Path(args.parquet_dir),
        extracted_dir=Path(args.extracted_dir),
        classified_dir=Path(args.classified_dir),
        target_crs=args.target_crs
    )

    stats = process_all(config)
    print_summary(stats)