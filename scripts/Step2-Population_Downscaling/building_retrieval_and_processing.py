# -*- coding: utf-8 -*-
"""
Building Data Processor for Dynamic Population Estimation
==========================================================

This module provides a complete pipeline for processing building data from 
Overture Maps, including extraction, POI linking, and category classification.

Processing Pipeline:
    Step 1: Extract buildings from Overture Parquet based on grid extents
    Step 2: Link buildings with POI data for functional attributes
    Step 3: Classify buildings into categories and associate with land use

Classification Categories:
    - residential: Housing and accommodation facilities
    - education: Educational institutions
    - leisure: Recreation, entertainment, and religious facilities
    - c&i (Commercial & Industrial): Business, industrial, and public services
    - other: Features not matching above categories

Data Sources:
    - Overture Maps Buildings (parquet format)
    - Overture Maps Places/POI (parquet format)
    - Land Cover/Land Use data (shapefile format)
"""

import argparse
import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from shapely.geometry import shape, Point
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
    'c&i': [
        'industrial', 'warehouse', 'office', 'public', 'civic',
        'public_building', 'childcare', 'townhouse', 'construction', 'utility',
        'roundhouse', 'commercial', 'hospital', 'logistics', 'storage',
        'hangar', 'farm_auxiliary', 'parking', 'train_station', 'station',
        'transportation', 'underground_entrance', 'cowshed', 'barn', 'stable',
        'silo', 'stables', 'guard_booth', 'guard', 'manufacture', 'retail',
        'shop', 'mall', 'service', 'carwash', 'store', 'supermarket', 'kiosk'
    ]
}

# Category priority for classification (when keyword matches multiple categories)
CATEGORY_PRIORITY: List[str] = ['residential', 'education', 'leisure', 'c&i']

# Column priority for classification: bcat1 > poi1 > poi2 > poi3 > bcat2 > bcat3
CLASSIFICATION_COLUMNS: List[str] = ['bcat1', 'poi1', 'poi2', 'poi3', 'bcat2', 'bcat3']


# =============================================================================
# Path Configuration
# =============================================================================

@dataclass
class PathConfig:
    """Configuration for input/output paths.
    
    Attributes:
        grid_dir: Directory containing TIF grid files.
        building_parquet_dir: Directory containing Overture Buildings Parquet files.
        places_parquet_dir: Directory containing Overture Places/POI Parquet files.
        landuse_dir: Directory containing classified land use shapefiles.
        extracted_dir: Output directory for extracted buildings.
        poi_linked_dir: Output directory for POI-linked buildings.
        classified_dir: Output directory for final classified buildings.
        target_crs: Target coordinate reference system.
    """
    
    grid_dir: Path
    building_parquet_dir: Path
    places_parquet_dir: Path
    landuse_dir: Path
    extracted_dir: Path
    poi_linked_dir: Path
    classified_dir: Path
    target_crs: str = "EPSG:4326"
    
    def __post_init__(self):
        """Convert all paths to Path objects."""
        self.grid_dir = Path(self.grid_dir)
        self.building_parquet_dir = Path(self.building_parquet_dir)
        self.places_parquet_dir = Path(self.places_parquet_dir)
        self.landuse_dir = Path(self.landuse_dir)
        self.extracted_dir = Path(self.extracted_dir)
        self.poi_linked_dir = Path(self.poi_linked_dir)
        self.classified_dir = Path(self.classified_dir)
    
    def validate(self) -> bool:
        """Validate that input directories exist."""
        valid = True
        if not self.grid_dir.exists():
            logger.error(f"Grid directory not found: {self.grid_dir}")
            valid = False
        if not self.building_parquet_dir.exists():
            logger.error(f"Building Parquet directory not found: {self.building_parquet_dir}")
            valid = False
        # places and landuse are optional
        if not self.places_parquet_dir.exists():
            logger.warning(f"Places Parquet directory not found: {self.places_parquet_dir}")
        if not self.landuse_dir.exists():
            logger.warning(f"Land use directory not found: {self.landuse_dir}")
        return valid
    
    def create_output_dirs(self):
        """Create all output directories."""
        self.extracted_dir.mkdir(parents=True, exist_ok=True)
        self.poi_linked_dir.mkdir(parents=True, exist_ok=True)
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
    if x is None or pd.isna(x):
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


def is_empty_value(val: Any) -> bool:
    """Check if value is empty or NA."""
    if val is None or pd.isna(val):
        return True
    if isinstance(val, str) and val.strip() == '':
        return True
    return False


def get_column_mapping(columns: List[str]) -> Dict[str, str]:
    """Generate column name mapping for Shapefile (10 char limit)."""
    short_names = {
        'num_floors': 'floors', 'num_floors_underground': 'floors_und',
        'facade_color': 'fac_color', 'facade_material': 'fac_mat',
        'roof_material': 'roof_mat', 'roof_shape': 'roof_shp',
        'roof_color': 'roof_col', 'roof_height': 'roof_hgt',
        'roof_direction': 'roof_dir', 'roof_orientation': 'roof_ori',
        'is_underground': 'undergrd', 'min_height': 'min_hgt',
        'min_floor': 'min_flr', 'geometry_wkt': 'geom_wkt',
        'name_primary': 'name_pri', 'name_common': 'name_comm',
    }
    
    mapping = {}
    used_names = set()
    for col in columns:
        if col == 'geometry' or len(col) <= 10:
            continue
        short = short_names.get(col, col[:10])
        counter = 1
        original_short = short
        while short in used_names:
            short = original_short[:8] + f"{counter:02d}"
            counter += 1
        mapping[col] = short
        used_names.add(short)
    
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


# =============================================================================
# Classification Functions
# =============================================================================

def classify_value(value: Any) -> Optional[str]:
    """Classify a single value based on keyword matching."""
    if is_empty_value(value):
        return None
    
    value_lower = str(value).lower()
    for category in CATEGORY_PRIORITY:
        for keyword in CATEGORY_RULES[category]:
            if keyword in value_lower:
                return category
    return None


def classify_building_row(row: pd.Series) -> Optional[str]:
    """Classify a building row using POI-linked columns.
    
    Priority: bcat1 > poi1 > poi2 > poi3 > bcat2 > bcat3
    """
    has_any_value = False
    
    for col in CLASSIFICATION_COLUMNS:
        if col in row.index:
            value = row[col]
            if not is_empty_value(value):
                has_any_value = True
                result = classify_value(value)
                if result is not None:
                    return result
    
    # Has values but no keyword match -> 'other'
    if has_any_value:
        return 'other'
    
    return None


# =============================================================================
# Step 1: Extract Buildings from Parquet
# =============================================================================

def extract_buildings_for_grid(
    tif_path: Path,
    parquet_dir: Path,
    output_dir: Path,
    target_crs: str
) -> int:
    """
    Extract building data for a single grid.
    
    Args:
        tif_path: Path to TIF grid file
        parquet_dir: Directory containing Parquet files
        output_dir: Output directory
        target_crs: Target CRS
        
    Returns:
        Number of buildings extracted
    """
    tif_name = tif_path.stem
    bbox = get_tif_bbox(tif_path)
    
    parquet_files = list(parquet_dir.glob("*.parquet"))
    if not parquet_files:
        logger.warning(f"  No parquet files found in {parquet_dir}")
        return 0
    
    matched_dfs = []
    
    for pf in parquet_files:
        try:
            import pyarrow.parquet as pq
            parquet_file = pq.ParquetFile(pf)
            column_names = parquet_file.schema_arrow.names
            has_center = 'center_x' in column_names and 'center_y' in column_names
            
            if has_center:
                df = pd.read_parquet(pf, columns=['center_x', 'center_y'])
                mask = ((df['center_x'] >= bbox['minx']) & 
                        (df['center_x'] <= bbox['maxx']) &
                        (df['center_y'] >= bbox['miny']) & 
                        (df['center_y'] <= bbox['maxy']))
                matched_indices = df[mask].index.tolist()
            else:
                df = pd.read_parquet(pf, columns=['bbox'])
                def point_in_bbox(b):
                    if not b or not isinstance(b, dict):
                        return False
                    try:
                        cx = (b.get('xmin', b.get('minx', 0)) + b.get('xmax', b.get('maxx', 0))) / 2
                        cy = (b.get('ymin', b.get('miny', 0)) + b.get('ymax', b.get('maxy', 0))) / 2
                        return bbox['minx'] <= cx <= bbox['maxx'] and bbox['miny'] <= cy <= bbox['maxy']
                    except:
                        return False
                mask = df['bbox'].apply(point_in_bbox)
                matched_indices = df[mask].index.tolist()
            
            if matched_indices:
                full_df = pd.read_parquet(pf)
                matched_dfs.append(full_df.iloc[matched_indices])
                
        except Exception as e:
            logger.debug(f"  Error reading {pf.name}: {e}")
    
    if not matched_dfs:
        return 0
    
    combined_df = pd.concat(matched_dfs, ignore_index=True)
    gdf = convert_to_geodataframe(combined_df)
    
    if gdf is None or len(gdf) == 0:
        return 0
    
    # Transform CRS if needed
    if target_crs != "EPSG:4326":
        try:
            gdf = gdf.to_crs(target_crs)
        except Exception as e:
            logger.warning(f"  CRS transformation failed: {e}")
    
    # Add computed fields
    if gdf.crs and gdf.crs.is_geographic:
        gdf['area_m2'] = gdf.geometry.apply(
            lambda g: g.area * 111320 * 111320 * np.cos(np.radians(g.centroid.y)) if g else None
        )
    else:
        gdf['area_m2'] = gdf.geometry.area
    
    if 'center_x' not in gdf.columns:
        gdf['center_x'] = gdf.geometry.apply(lambda g: g.centroid.x if g else None)
        gdf['center_y'] = gdf.geometry.apply(lambda g: g.centroid.y if g else None)
    
    # Save CSV
    csv_path = output_dir / f"{tif_name}_buildings.csv"
    csv_df = gdf.copy()
    csv_df['geometry_wkt'] = csv_df.geometry.apply(lambda g: g.wkt if g else None)
    csv_df = csv_df.drop(columns=['geometry'])
    
    for col in csv_df.columns:
        if csv_df[col].dtype == 'object':
            non_null = csv_df[col].dropna()
            if len(non_null) > 0 and isinstance(non_null.iloc[0], (dict, list)):
                csv_df[col] = csv_df[col].apply(safe_json_dumps)
    
    csv_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    
    # Save Shapefile
    shp_gdf = gdf.copy()
    for col in shp_gdf.columns:
        if col == 'geometry':
            continue
        if shp_gdf[col].dtype == 'object':
            non_null = shp_gdf[col].dropna()
            if len(non_null) > 0 and isinstance(non_null.iloc[0], (dict, list)):
                shp_gdf[col] = shp_gdf[col].apply(lambda x: safe_to_string(x, 254))
    
    col_mapping = get_column_mapping(shp_gdf.columns.tolist())
    if col_mapping:
        shp_gdf = shp_gdf.rename(columns=col_mapping)
    
    shp_path = output_dir / f"{tif_name}_buildings.shp"
    shp_gdf.to_file(shp_path, encoding='utf-8')
    
    return len(gdf)


# =============================================================================
# Step 2: Link Buildings with POI
# =============================================================================

def extract_places_for_grid(
    tif_path: Path,
    parquet_dir: Path,
    target_crs: str
) -> Optional[gpd.GeoDataFrame]:
    """Extract POI/places data for a grid extent."""
    if not parquet_dir.exists():
        return None
    
    bbox = get_tif_bbox(tif_path)
    parquet_files = list(parquet_dir.glob("*.parquet"))
    
    if not parquet_files:
        return None
    
    matched_dfs = []
    
    for pf in parquet_files:
        try:
            df = pd.read_parquet(pf, columns=['bbox'])
            
            def point_in_bbox(b):
                if not b or not isinstance(b, dict):
                    return False
                try:
                    cx = (b.get('xmin', b.get('minx', 0)) + b.get('xmax', b.get('maxx', 0))) / 2
                    cy = (b.get('ymin', b.get('miny', 0)) + b.get('ymax', b.get('maxy', 0))) / 2
                    return bbox['minx'] <= cx <= bbox['maxx'] and bbox['miny'] <= cy <= bbox['maxy']
                except:
                    return False
            
            mask = df['bbox'].apply(point_in_bbox)
            matched_indices = df[mask].index.tolist()
            
            if matched_indices:
                full_df = pd.read_parquet(pf)
                matched_dfs.append(full_df.iloc[matched_indices])
                
        except Exception as e:
            logger.debug(f"  Error reading places {pf.name}: {e}")
    
    if not matched_dfs:
        return None
    
    combined_df = pd.concat(matched_dfs, ignore_index=True)
    gdf = convert_to_geodataframe(combined_df)
    
    if gdf is not None and target_crs != "EPSG:4326":
        try:
            gdf = gdf.to_crs(target_crs)
        except:
            pass
    
    return gdf


def link_buildings_with_pois(
    buildings_gdf: gpd.GeoDataFrame,
    places_gdf: Optional[gpd.GeoDataFrame]
) -> gpd.GeoDataFrame:
    """
    Link buildings with POI data.
    
    Logic:
        1. If building has subtype -> bcat1 = subtype
        2. If building has class -> poi1 = class
        3. If has subtype OR class -> skip POI spatial matching
        4. Otherwise -> match POIs by containment, take top 3 by confidence
        5. If no POI match -> leave empty (handled by classification step)
    """
    # Initialize POI columns
    for col in ['poi1', 'poi2', 'poi3', 'bcat1', 'bcat2', 'bcat3']:
        buildings_gdf[col] = ''
    
    # Prepare places spatial index
    places_sindex = None
    if places_gdf is not None and len(places_gdf) > 0:
        if buildings_gdf.crs != places_gdf.crs:
            try:
                places_gdf = places_gdf.to_crs(buildings_gdf.crs)
            except:
                places_gdf = None
        if places_gdf is not None:
            try:
                places_sindex = places_gdf.sindex
            except:
                pass
    
    for idx, row in buildings_gdf.iterrows():
        # Check building's own attributes
        subtype = row.get('subtype', '')
        building_class = row.get('class', '')
        
        subtype = '' if is_empty_value(subtype) else str(subtype).strip()
        building_class = '' if is_empty_value(building_class) else str(building_class).strip()
        
        if subtype or building_class:
            if subtype:
                buildings_gdf.at[idx, 'bcat1'] = subtype
            if building_class:
                buildings_gdf.at[idx, 'poi1'] = building_class
            continue
        
        # No own attributes, try POI matching
        if places_gdf is None or places_sindex is None:
            continue
        
        building_geom = row.geometry
        if building_geom is None or building_geom.is_empty:
            continue
        
        # Find POIs inside building
        try:
            possible_idx = list(places_sindex.intersection(building_geom.bounds))
            candidates = places_gdf.iloc[possible_idx]
        except:
            candidates = places_gdf
        
        pois_inside = []
        for _, poi_row in candidates.iterrows():
            poi_geom = poi_row.geometry
            if poi_geom is None:
                continue
            try:
                if building_geom.contains(poi_geom):
                    confidence = poi_row.get('confidence', 0)
                    confidence = float(confidence) if pd.notna(confidence) else 0
                    category = poi_row.get('category', '')
                    basic_cat = poi_row.get('basic_cat', '')
                    pois_inside.append({
                        'category': str(category)[:30] if pd.notna(category) else '',
                        'basic_cat': str(basic_cat)[:30] if pd.notna(basic_cat) else '',
                        'confidence': confidence
                    })
            except:
                continue
        
        if pois_inside:
            pois_inside.sort(key=lambda x: x['confidence'], reverse=True)
            for i, poi in enumerate(pois_inside[:3]):
                buildings_gdf.at[idx, f'poi{i+1}'] = poi['category']
                buildings_gdf.at[idx, f'bcat{i+1}'] = poi['basic_cat']
    
    return buildings_gdf


def process_poi_linking(
    building_path: Path,
    places_gdf: Optional[gpd.GeoDataFrame],
    output_dir: Path
) -> int:
    """Process POI linking for a single building file."""
    city_name = building_path.stem.replace('_buildings', '')
    
    buildings_gdf = gpd.read_file(building_path)
    buildings_gdf = link_buildings_with_pois(buildings_gdf, places_gdf)
    
    # Save
    output_path = output_dir / f"{city_name}_buildings_poi.shp"
    buildings_gdf.to_file(output_path, encoding='utf-8')
    
    # Also save CSV
    csv_path = output_dir / f"{city_name}_buildings_poi.csv"
    csv_df = buildings_gdf.drop(columns=['geometry'])
    csv_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    
    return len(buildings_gdf)


# =============================================================================
# Step 3: Classify and Associate with Land Use
# =============================================================================

def process_classification(
    building_path: Path,
    landuse_gdf: Optional[gpd.GeoDataFrame],
    output_dir: Path
) -> Dict[str, int]:
    """
    Classify buildings and associate with land use.
    
    Logic:
        1. Classify using POI columns (bcat1 > poi1 > poi2 > poi3 > bcat2 > bcat3)
        2. For buildings with no attribute info (category is None):
           - Query Land Cover to get category
           - If not found: fallback to 'residential'
        3. Fill any remaining empty categories with 'other'
    
    Note:
        Default floor assignment by category is handled downstream
        by Building_add_height.py to avoid redundancy.
    """
    city_name = building_path.stem.replace('_buildings_poi', '')
    
    buildings_gdf = gpd.read_file(building_path)
    
    # Step 3a: Classify based on POI columns
    buildings_gdf['category'] = buildings_gdf.apply(classify_building_row, axis=1)
    
    # Step 3b: For buildings with empty class/subtype and no POI info
    # (category is None from Step 3a), try Land Cover, then fallback to 'residential'
    none_mask = buildings_gdf['category'].isna()
    
    if none_mask.any() and landuse_gdf is not None and len(landuse_gdf) > 0:
        if buildings_gdf.crs != landuse_gdf.crs:
            try:
                landuse_gdf = landuse_gdf.to_crs(buildings_gdf.crs)
            except:
                landuse_gdf = None
        
        if landuse_gdf is not None:
            try:
                landuse_sindex = landuse_gdf.sindex
            except:
                landuse_sindex = None
            
            corrected_count = 0
            for idx in buildings_gdf[none_mask].index:
                row = buildings_gdf.loc[idx]
                
                # Only process buildings with no own attributes AND no POI match
                if not (is_empty_value(row.get('class', '')) and 
                        is_empty_value(row.get('subtype', '')) and
                        all(is_empty_value(row.get(col, ''))
                            for col in CLASSIFICATION_COLUMNS)):
                    continue
                
                if 'center_x' in row.index and 'center_y' in row.index:
                    cx, cy = row['center_x'], row['center_y']
                    if pd.notna(cx) and pd.notna(cy):
                        center = Point(cx, cy)
                    else:
                        center = row.geometry.centroid if row.geometry else None
                else:
                    center = row.geometry.centroid if row.geometry else None
                
                if center is None:
                    continue
                
                if landuse_sindex is not None:
                    try:
                        possible_idx = list(landuse_sindex.intersection(center.bounds))
                        candidates = landuse_gdf.iloc[possible_idx]
                    except:
                        candidates = landuse_gdf
                else:
                    candidates = landuse_gdf
                
                for _, lu_row in candidates.iterrows():
                    if lu_row.geometry is not None and lu_row.geometry.contains(center):
                        lu_cat = lu_row.get('category', '')
                        if not is_empty_value(lu_cat):
                            buildings_gdf.at[idx, 'category'] = str(lu_cat).strip()
                            corrected_count += 1
                        break
            
            if corrected_count > 0:
                logger.info(f"  Land Cover corrected {corrected_count} buildings")
    
    # Step 3c: Fill remaining None categories with 'residential'
    buildings_gdf['category'] = buildings_gdf['category'].fillna('residential')
    
    # Get statistics
    category_counts = buildings_gdf['category'].value_counts().to_dict()
    
    # Save Shapefile
    output_path = output_dir / f"{city_name}_buildings_classified.shp"
    buildings_gdf.to_file(output_path, encoding='utf-8')
    
    # Save CSV
    csv_path = output_dir / f"{city_name}_buildings_classified.csv"
    csv_df = buildings_gdf.drop(columns=['geometry'])
    csv_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    
    return category_counts


# =============================================================================
# Main Processing
# =============================================================================

def process_all(config: PathConfig) -> Dict[str, Any]:
    """
    Run complete building processing pipeline.
    
    Args:
        config: Path configuration
        
    Returns:
        Processing statistics
    """
    if not config.validate():
        raise ValueError("Invalid configuration")
    
    config.create_output_dirs()
    
    stats = {
        'extraction': {'total': 0, 'success': 0, 'buildings': 0},
        'poi_linking': {'total': 0, 'success': 0, 'buildings': 0},
        'classification': {'total': 0, 'success': 0, 'buildings': 0, 'categories': {}}
    }
    
    # ===================
    # Step 1: Extraction
    # ===================
    logger.info("=" * 60)
    logger.info("Step 1: Extracting buildings from Parquet files")
    logger.info("=" * 60)
    
    tif_files = list(config.grid_dir.glob("*.tif"))
    stats['extraction']['total'] = len(tif_files)
    
    for i, tif_path in enumerate(tif_files, 1):
        logger.info(f"[{i}/{len(tif_files)}] {tif_path.name}")
        count = extract_buildings_for_grid(
            tif_path, config.building_parquet_dir, config.extracted_dir, config.target_crs
        )
        if count > 0:
            stats['extraction']['success'] += 1
            stats['extraction']['buildings'] += count
            logger.info(f"  Extracted {count} buildings")
        else:
            logger.info(f"  No buildings found")
    
    # ===================
    # Step 2: POI Linking
    # ===================
    logger.info("")
    logger.info("=" * 60)
    logger.info("Step 2: Linking buildings with POI data")
    logger.info("=" * 60)
    
    building_files = list(config.extracted_dir.glob("*_buildings.shp"))
    stats['poi_linking']['total'] = len(building_files)
    
    for i, building_path in enumerate(building_files, 1):
        city_name = building_path.stem.replace('_buildings', '')
        tif_path = config.grid_dir / f"{city_name}.tif"
        
        logger.info(f"[{i}/{len(building_files)}] {building_path.name}")
        
        # Extract places for this grid
        places_gdf = None
        if config.places_parquet_dir.exists() and tif_path.exists():
            places_gdf = extract_places_for_grid(
                tif_path, config.places_parquet_dir, config.target_crs
            )
            if places_gdf is not None:
                logger.info(f"  Found {len(places_gdf)} POIs")
        
        try:
            count = process_poi_linking(building_path, places_gdf, config.poi_linked_dir)
            stats['poi_linking']['success'] += 1
            stats['poi_linking']['buildings'] += count
            logger.info(f"  Linked {count} buildings")
        except Exception as e:
            logger.error(f"  Error: {e}")
    
    # ===================
    # Step 3: Classification
    # ===================
    logger.info("")
    logger.info("=" * 60)
    logger.info("Step 3: Classifying buildings and associating with land use")
    logger.info("=" * 60)
    
    poi_linked_files = list(config.poi_linked_dir.glob("*_buildings_poi.shp"))
    stats['classification']['total'] = len(poi_linked_files)
    
    for i, poi_path in enumerate(poi_linked_files, 1):
        city_name = poi_path.stem.replace('_buildings_poi', '')
        
        logger.info(f"[{i}/{len(poi_linked_files)}] {poi_path.name}")
        
        # Load land use if available
        landuse_path = config.landuse_dir / f"{city_name}_landuse_classified.shp"
        landuse_gdf = None
        if landuse_path.exists():
            try:
                landuse_gdf = gpd.read_file(landuse_path)
                logger.info(f"  Found {len(landuse_gdf)} land use features")
            except Exception as e:
                logger.warning(f"  Could not load land use: {e}")
        
        try:
            category_counts = process_classification(poi_path, landuse_gdf, config.classified_dir)
            stats['classification']['success'] += 1
            stats['classification']['buildings'] += sum(category_counts.values())
            
            for cat, count in category_counts.items():
                stats['classification']['categories'][cat] = \
                    stats['classification']['categories'].get(cat, 0) + count
            
            logger.info(f"  Classified {sum(category_counts.values())} buildings")
            for cat, count in sorted(category_counts.items()):
                logger.info(f"    {cat}: {count}")
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
    print(f"  Buildings extracted: {ext['buildings']:,}")
    
    poi = stats['poi_linking']
    print(f"\nStep 2 - POI Linking:")
    print(f"  Files processed: {poi['success']}/{poi['total']}")
    print(f"  Buildings linked: {poi['buildings']:,}")
    
    cls = stats['classification']
    print(f"\nStep 3 - Classification:")
    print(f"  Files processed: {cls['success']}/{cls['total']}")
    print(f"  Buildings classified: {cls['buildings']:,}")
    
    if cls['categories']:
        print(f"\n  Category Distribution:")
        total = cls['buildings']
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
        description="Extract, link POI, and classify buildings from Overture Maps data."
    )
    parser.add_argument('--grid_dir', type=str, required=True,
                        help="Directory containing TIF grid files")
    parser.add_argument('--building_parquet_dir', type=str, required=True,
                        help="Directory containing Overture Buildings Parquet files")
    parser.add_argument('--places_parquet_dir', type=str, required=True,
                        help="Directory containing Overture Places/POI Parquet files")
    parser.add_argument('--landuse_dir', type=str, required=True,
                        help="Directory containing classified land use shapefiles")
    parser.add_argument('--extracted_dir', type=str, required=True,
                        help="Output directory for extracted buildings")
    parser.add_argument('--poi_linked_dir', type=str, required=True,
                        help="Output directory for POI-linked buildings")
    parser.add_argument('--classified_dir', type=str, required=True,
                        help="Output directory for classified buildings")
    parser.add_argument('--target_crs', type=str, default="EPSG:4326",
                        help="Target CRS (default: EPSG:4326)")

    args = parser.parse_args()

    config = PathConfig(
        grid_dir=Path(args.grid_dir),
        building_parquet_dir=Path(args.building_parquet_dir),
        places_parquet_dir=Path(args.places_parquet_dir),
        landuse_dir=Path(args.landuse_dir),
        extracted_dir=Path(args.extracted_dir),
        poi_linked_dir=Path(args.poi_linked_dir),
        classified_dir=Path(args.classified_dir),
        target_crs=args.target_crs
    )

    stats = process_all(config)
    print_summary(stats)