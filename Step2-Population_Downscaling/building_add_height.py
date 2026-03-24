# -*- coding: utf-8 -*-
"""
Building Height Filler for Dynamic Population Estimation
=========================================================

This module fills missing height and num_floors data for classified buildings
using external 3D building footprint data (3D-GloBFP).

Input:
    - Classified building data from building_processor.py 
      ({city}_buildings_classified.shp)
    - Must contain 'category' column for default floor assignment

Processing Pipeline:
    Step 1: Load world grid for spatial indexing
    Step 2: For each building file, find corresponding height data by grid ID
    Step 3: Match buildings with height data by center point containment
    Step 4: Fill missing values using external data or category-based defaults

Fill Rules (per documentation):
    1. Buildings with BOTH height AND num_floors: keep original values
    2. Buildings matched in 3D-GloBFP: 
       - height = matched_height
       - num_floors = round(height / 4)
    3. Unmatched buildings use category-based defaults:
       - residential: num_floors = 2
       - c&c: num_floors = 2
       - education: num_floors = 4
       - leisure: num_floors = 1
       - other: num_floors = 1

Data Sources:
    - 3D-GloBFP (3D Global Building Footprints): https://github.com/3D-GloBFP
    - Height data organized in ZIP files by world grid ID
"""

import argparse
import json
import logging
import os
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# Configuration Constants
# =============================================================================

# Default floors by category
DEFAULT_FLOORS: Dict[str, int] = {
    'residential': 2,
    'c&c': 2,
    'education': 4,
    'leisure': 1,
    'other': 1
}

# Height column names to search in 3D-GloBFP data (priority order)
HEIGHT_COLUMNS: List[str] = ['Height', 'height', 'HEIGHT', 'h', 'H', 'bld_height']

# Grid ID column names to search in world grid (priority order)
GRID_ID_COLUMNS: List[str] = ['grid_ID', 'grid_id', 'GRID_ID', 'gridid', 'ID', 'id', 'FID']

# Height to floors conversion factor
FLOOR_HEIGHT_METERS: float = 4.0

# Minimum number of floors
MIN_FLOORS: int = 1


# =============================================================================
# Path Configuration
# =============================================================================

@dataclass
class PathConfig:
    """Configuration for input/output paths.
    
    Attributes:
        grid_file: Path to world grid shapefile.
        height_dir: Directory containing 3D-GloBFP height ZIP files.
        building_dir: Directory containing classified building shapefiles.
        output_dir: Output directory for height-filled buildings.
        cleanup_extracted: Whether to delete extracted ZIP folders after processing.
    """
    
    grid_file: Path
    height_dir: Path
    building_dir: Path
    output_dir: Path
    cleanup_extracted: bool = True
    
    def __post_init__(self):
        """Convert all paths to Path objects."""
        self.grid_file = Path(self.grid_file)
        self.height_dir = Path(self.height_dir)
        self.building_dir = Path(self.building_dir)
        self.output_dir = Path(self.output_dir)
    
    def validate(self) -> bool:
        """Validate that required input paths exist."""
        valid = True
        
        if not self.grid_file.exists():
            logger.error(f"Grid file not found: {self.grid_file}")
            valid = False
        
        if not self.height_dir.exists():
            logger.error(f"Height data directory not found: {self.height_dir}")
            valid = False
        
        if not self.building_dir.exists():
            logger.error(f"Building directory not found: {self.building_dir}")
            valid = False
        else:
            # Check for classified building files
            shp_files = list(self.building_dir.glob("*_buildings_classified.shp"))
            if not shp_files:
                logger.warning(f"No *_buildings_classified.shp files found in: {self.building_dir}")
        
        return valid
    
    def create_output_dirs(self):
        """Create output directory if not exists."""
        self.output_dir.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Utility Functions
# =============================================================================

def is_empty_value(val: Any) -> bool:
    """
    Check if value is empty, NA, or zero.
    
    Args:
        val: Value to check
        
    Returns:
        True if value is considered empty
    """
    if val is None:
        return True
    if pd.isna(val):
        return True
    if isinstance(val, (int, float)) and val == 0:
        return True
    if isinstance(val, str) and val.strip() == '':
        return True
    return False


def get_default_floors(category: Any) -> int:
    """
    Get default number of floors for a building category.
    
    Args:
        category: Building category string
        
    Returns:
        Default number of floors
    """
    if category is None or pd.isna(category):
        return DEFAULT_FLOORS['other']
    
    category_lower = str(category).lower().strip()
    
    # Handle various spellings
    if category_lower in ['residential']:
        return DEFAULT_FLOORS['residential']
    elif category_lower in ['c&c', 'cc', 'c_c', 'commercial', 'civic']:
        return DEFAULT_FLOORS['c&c']
    elif category_lower in ['education', 'educational']:
        return DEFAULT_FLOORS['education']
    elif category_lower in ['leisure', 'recreation']:
        return DEFAULT_FLOORS['leisure']
    else:
        return DEFAULT_FLOORS['other']


# =============================================================================
# Grid Functions
# =============================================================================

def load_world_grid(grid_path: Path) -> Optional[gpd.GeoDataFrame]:
    """
    Load world grid shapefile for spatial indexing.
    
    Args:
        grid_path: Path to world grid shapefile
        
    Returns:
        GeoDataFrame or None if failed
    """
    try:
        gdf = gpd.read_file(grid_path)
        logger.info(f"Loaded world grid: {len(gdf)} grid cells")
        return gdf
    except Exception as e:
        logger.error(f"Failed to load world grid: {e}")
        return None


def get_grid_id_for_bounds(
    grid_gdf: gpd.GeoDataFrame,
    bounds: Tuple[float, float, float, float]
) -> Optional[str]:
    """
    Find grid ID that contains the center of given bounds.
    
    Args:
        grid_gdf: World grid GeoDataFrame
        bounds: Tuple of (minx, miny, maxx, maxy)
        
    Returns:
        Grid ID string or None if not found
    """
    minx, miny, maxx, maxy = bounds
    center_x = (minx + maxx) / 2
    center_y = (miny + maxy) / 2
    center_point = Point(center_x, center_y)
    
    # Use spatial index if available
    try:
        sindex = grid_gdf.sindex
        possible_idx = list(sindex.intersection(center_point.bounds))
        candidates = grid_gdf.iloc[possible_idx]
    except:
        candidates = grid_gdf
    
    for idx, row in candidates.iterrows():
        if row.geometry is not None and row.geometry.contains(center_point):
            # Try to find grid ID column
            for col in GRID_ID_COLUMNS:
                if col in row.index and pd.notna(row[col]):
                    return str(row[col])
            # Fallback to index
            return str(idx)
    
    return None


# =============================================================================
# Height Data Functions
# =============================================================================

def find_height_zip(height_dir: Path, grid_id: str) -> Optional[Path]:
    """
    Find height data ZIP file for a given grid ID.
    
    Searches for files matching pattern: {grid_id}_*.zip
    
    Args:
        height_dir: Directory containing height ZIP files
        grid_id: Grid ID to search for
        
    Returns:
        Path to ZIP file or None if not found
    """
    try:
        for filename in os.listdir(height_dir):
            if filename.endswith('.zip'):
                # Check various naming patterns
                if filename.startswith(f"{grid_id}_") or filename.startswith(f"{grid_id}."):
                    return height_dir / filename
                # Also check if grid_id is contained in filename
                name_parts = filename.replace('.zip', '').split('_')
                if grid_id in name_parts:
                    return height_dir / filename
    except Exception as e:
        logger.debug(f"Error searching for height ZIP: {e}")
    
    return None


def extract_height_zip(zip_path: Path, extract_base_dir: Path) -> Optional[Path]:
    """
    Extract height data ZIP file to a temporary directory.
    
    Args:
        zip_path: Path to ZIP file
        extract_base_dir: Base directory for extraction
        
    Returns:
        Path to extracted directory or None if failed
    """
    extract_dir = extract_base_dir / f"extracted_{zip_path.stem}"
    
    # Check if already extracted
    if extract_dir.exists():
        shp_files = list(extract_dir.rglob("*.shp"))
        if shp_files:
            return extract_dir
    
    try:
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        logger.debug(f"Extracted: {zip_path.name}")
        return extract_dir
    except Exception as e:
        logger.error(f"Failed to extract {zip_path.name}: {e}")
        return None


def load_height_data(extract_dir: Path) -> Optional[gpd.GeoDataFrame]:
    """
    Load height data from extracted directory.
    
    Args:
        extract_dir: Path to extracted directory
        
    Returns:
        GeoDataFrame with height data or None if failed
    """
    # Find shapefile (search recursively)
    shp_files = list(extract_dir.rglob("*.shp"))
    
    if not shp_files:
        logger.warning(f"No shapefile found in: {extract_dir}")
        return None
    
    try:
        gdf = gpd.read_file(shp_files[0])
        return gdf
    except Exception as e:
        logger.error(f"Failed to load height data: {e}")
        return None


def find_height_for_point(
    height_gdf: gpd.GeoDataFrame,
    point_x: float,
    point_y: float,
    height_sindex: Any = None
) -> Optional[float]:
    """
    Find building height for a given point from 3D-GloBFP data.
    
    Args:
        height_gdf: GeoDataFrame with height data
        point_x: X coordinate (longitude)
        point_y: Y coordinate (latitude)
        height_sindex: Spatial index (optional, for performance)
        
    Returns:
        Height value in meters or None if not found
    """
    point = Point(point_x, point_y)
    
    # Use spatial index for faster lookup
    if height_sindex is not None:
        try:
            possible_idx = list(height_sindex.intersection(point.bounds))
            if possible_idx:
                candidates = height_gdf.iloc[possible_idx]
                for _, row in candidates.iterrows():
                    if row.geometry is not None and row.geometry.contains(point):
                        for col in HEIGHT_COLUMNS:
                            if col in row.index and pd.notna(row[col]):
                                height = float(row[col])
                                if height > 0:
                                    return height
                return None
        except:
            pass
    
    # Fallback: iterate through all (slow but reliable)
    for _, row in height_gdf.iterrows():
        if row.geometry is not None and row.geometry.contains(point):
            for col in HEIGHT_COLUMNS:
                if col in row.index and pd.notna(row[col]):
                    height = float(row[col])
                    if height > 0:
                        return height
    
    return None


def cleanup_extracted_folders(height_dir: Path):
    """
    Remove all extracted_* folders in height directory.
    
    Args:
        height_dir: Directory containing extracted folders
    """
    try:
        for item in height_dir.iterdir():
            if item.is_dir() and item.name.startswith('extracted_'):
                shutil.rmtree(item)
                logger.debug(f"Cleaned up: {item.name}")
    except Exception as e:
        logger.warning(f"Cleanup failed: {e}")


# =============================================================================
# Main Processing Functions
# =============================================================================

def process_building_file(
    building_path: Path,
    grid_gdf: gpd.GeoDataFrame,
    height_dir: Path,
    output_dir: Path
) -> Dict[str, int]:
    """
    Fill height data for a single classified building file.
    
    Args:
        building_path: Path to classified building shapefile
        grid_gdf: World grid GeoDataFrame
        height_dir: Directory containing height ZIP files
        output_dir: Output directory
        
    Returns:
        Processing statistics dictionary
    """
    stats = {
        'total': 0,
        'already_complete': 0,
        'filled_from_3d': 0,
        'filled_default': 0,
        'failed': 0
    }
    
    # Extract city name from filename
    city_name = building_path.stem.replace('_buildings_classified', '')
    
    # Load classified building data
    try:
        buildings_gdf = gpd.read_file(building_path)
        stats['total'] = len(buildings_gdf)
    except Exception as e:
        logger.error(f"Failed to load building file: {e}")
        return stats
    
    # Verify 'category' column exists (required for default floors)
    if 'category' not in buildings_gdf.columns:
        logger.warning("  'category' column not found, using 'other' as default")
        buildings_gdf['category'] = 'other'
    
    # Ensure height and num_floors columns exist
    if 'height' not in buildings_gdf.columns:
        buildings_gdf['height'] = None
    if 'num_floors' not in buildings_gdf.columns:
        buildings_gdf['num_floors'] = None
    
    # Find grid ID for this building set
    bounds = buildings_gdf.total_bounds
    grid_id = get_grid_id_for_bounds(grid_gdf, bounds)
    
    # Load height data if grid found
    height_gdf = None
    height_sindex = None
    
    if grid_id:
        logger.info(f"  Grid ID: {grid_id}")
        
        zip_path = find_height_zip(height_dir, grid_id)
        if zip_path:
            logger.info(f"  Height data: {zip_path.name}")
            extract_dir = extract_height_zip(zip_path, height_dir)
            if extract_dir:
                height_gdf = load_height_data(extract_dir)
                if height_gdf is not None:
                    logger.info(f"  Loaded {len(height_gdf)} height polygons")
                    try:
                        height_sindex = height_gdf.sindex
                    except:
                        pass
        else:
            logger.warning(f"  No height ZIP found for grid: {grid_id}")
    else:
        logger.warning("  No matching grid found for building extent")
    
    # Process each building
    for idx, row in buildings_gdf.iterrows():
        # Check if already has both height and num_floors
        has_height = not is_empty_value(row.get('height'))
        has_floors = not is_empty_value(row.get('num_floors'))
        
        if has_height and has_floors:
            stats['already_complete'] += 1
            continue
        
        # Get building center point
        cx, cy = None, None
        
        # Try center_x/center_y columns first
        if 'center_x' in row.index and 'center_y' in row.index:
            cx, cy = row['center_x'], row['center_y']
        
        # Fallback to geometry centroid
        if (cx is None or cy is None or pd.isna(cx) or pd.isna(cy)):
            if row.geometry is not None and not row.geometry.is_empty:
                centroid = row.geometry.centroid
                cx, cy = centroid.x, centroid.y
            else:
                stats['failed'] += 1
                continue
        
        # Try to find height from 3D-GloBFP data
        found_height = None
        if height_gdf is not None:
            found_height = find_height_for_point(height_gdf, cx, cy, height_sindex)
        
        if found_height is not None and found_height > 0:
            # Successfully matched with 3D-GloBFP
            buildings_gdf.at[idx, 'height'] = found_height
            num_floors = max(MIN_FLOORS, round(found_height / FLOOR_HEIGHT_METERS))
            buildings_gdf.at[idx, 'num_floors'] = num_floors
            stats['filled_from_3d'] += 1
        else:
            # Use category-based default
            category = row.get('category', 'other')
            default_floors = get_default_floors(category)
            buildings_gdf.at[idx, 'num_floors'] = default_floors
            # Note: height remains None for default cases
            stats['filled_default'] += 1
    
    # Save output files
    output_shp = output_dir / f"{city_name}_buildings_final.shp"
    output_csv = output_dir / f"{city_name}_buildings_final.csv"
    
    try:
        buildings_gdf.to_file(output_shp, encoding='utf-8')
        
        csv_df = buildings_gdf.drop(columns=['geometry'])
        csv_df.to_csv(output_csv, index=False, encoding='utf-8-sig')
        
        logger.info(f"  Saved: {output_shp.name}")
    except Exception as e:
        logger.error(f"  Failed to save output: {e}")
    
    return stats


def process_all(config: PathConfig) -> Dict[str, Any]:
    """
    Run complete height filling pipeline for all classified building files.
    
    Args:
        config: Path configuration
        
    Returns:
        Overall processing statistics
    """
    if not config.validate():
        raise ValueError("Invalid configuration - check input paths")
    
    config.create_output_dirs()
    
    # Initialize statistics
    total_stats = {
        'files_total': 0,
        'files_success': 0,
        'files_failed': 0,
        'buildings_total': 0,
        'buildings_already_complete': 0,
        'buildings_filled_from_3d': 0,
        'buildings_filled_default': 0,
        'buildings_failed': 0
    }
    
    # ===================
    # Load World Grid
    # ===================
    logger.info("=" * 60)
    logger.info("Step 1: Loading world grid")
    logger.info("=" * 60)
    
    grid_gdf = load_world_grid(config.grid_file)
    if grid_gdf is None:
        raise ValueError("Failed to load world grid file")
    
    # ===================
    # Process Building Files
    # ===================
    logger.info("")
    logger.info("=" * 60)
    logger.info("Step 2: Filling building heights")
    logger.info("=" * 60)
    
    building_files = sorted(config.building_dir.glob("*_buildings_classified.shp"))
    total_stats['files_total'] = len(building_files)
    
    if not building_files:
        logger.warning(f"No *_buildings_classified.shp files found in: {config.building_dir}")
        return total_stats
    
    for i, building_path in enumerate(building_files, 1):
        logger.info(f"\n[{i}/{len(building_files)}] {building_path.name}")
        
        try:
            stats = process_building_file(
                building_path, grid_gdf, config.height_dir, config.output_dir
            )
            
            total_stats['files_success'] += 1
            total_stats['buildings_total'] += stats['total']
            total_stats['buildings_already_complete'] += stats['already_complete']
            total_stats['buildings_filled_from_3d'] += stats['filled_from_3d']
            total_stats['buildings_filled_default'] += stats['filled_default']
            total_stats['buildings_failed'] += stats['failed']
            
            logger.info(f"  Results: {stats['filled_from_3d']} from 3D data, "
                       f"{stats['filled_default']} defaults, "
                       f"{stats['already_complete']} already complete")
            
        except Exception as e:
            logger.error(f"  Processing failed: {e}")
            total_stats['files_failed'] += 1
    
    # ===================
    # Cleanup
    # ===================
    if config.cleanup_extracted:
        logger.info("")
        logger.info("=" * 60)
        logger.info("Step 3: Cleaning up extracted folders")
        logger.info("=" * 60)
        cleanup_extracted_folders(config.height_dir)
        logger.info("  Cleanup complete")
    
    # Save summary
    summary_path = config.output_dir / "_height_filling_summary.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(total_stats, f, indent=2, ensure_ascii=False)
    
    return total_stats


def print_summary(stats: Dict[str, Any]):
    """Print processing summary to console."""
    print("\n" + "=" * 60)
    print("Building Height Filling Summary")
    print("=" * 60)
    
    print(f"\nFiles:")
    print(f"  Total:     {stats['files_total']}")
    print(f"  Success:   {stats['files_success']}")
    print(f"  Failed:    {stats['files_failed']}")
    
    print(f"\nBuildings:")
    print(f"  Total:            {stats['buildings_total']:,}")
    print(f"  Already complete: {stats['buildings_already_complete']:,}")
    print(f"  From 3D-GloBFP:   {stats['buildings_filled_from_3d']:,}")
    print(f"  Default values:   {stats['buildings_filled_default']:,}")
    print(f"  Failed:           {stats['buildings_failed']:,}")
    
    if stats['buildings_total'] > 0:
        total = stats['buildings_total']
        pct_3d = stats['buildings_filled_from_3d'] / total * 100
        pct_default = stats['buildings_filled_default'] / total * 100
        pct_complete = stats['buildings_already_complete'] / total * 100
        
        print(f"\nCoverage:")
        print(f"  Already complete: {pct_complete:.1f}%")
        print(f"  From 3D data:     {pct_3d:.1f}%")
        print(f"  Default values:   {pct_default:.1f}%")


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fill missing building height/floor data using 3D-GloBFP."
    )
    parser.add_argument('--grid_file', type=str, required=True,
                        help="Path to world grid shapefile")
    parser.add_argument('--height_dir', type=str, required=True,
                        help="Directory containing 3D-GloBFP height ZIP files")
    parser.add_argument('--building_dir', type=str, required=True,
                        help="Directory containing classified building shapefiles")
    parser.add_argument('--output_dir', type=str, required=True,
                        help="Output directory for height-filled buildings")
    parser.add_argument('--no_cleanup', action='store_true',
                        help="Keep extracted ZIP folders after processing")

    args = parser.parse_args()

    config = PathConfig(
        grid_file=Path(args.grid_file),
        height_dir=Path(args.height_dir),
        building_dir=Path(args.building_dir),
        output_dir=Path(args.output_dir),
        cleanup_extracted=not args.no_cleanup
    )

    stats = process_all(config)
    print_summary(stats)
