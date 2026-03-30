# -*- coding: utf-8 -*-
"""
Visit Time Extraction Module

Extract visit count matrix [48, n_grids] from WorldMove trajectory data.
Each cell represents the number of visits to a grid at a specific time slot.

Input:
    - Trajectory files (.npz): Contains 'traj' key with shape [n_trajectories, 48]
    - Grid files (.json): Contains grid ID mapping

Output:
    - Visit matrix (.json): Shape [48, n_grids], visit counts per time slot
"""

import json
import numpy as np
from pathlib import Path
import argparse
from typing import Optional, Callable
import gc


def load_trajectories(npz_path: Path) -> Optional[np.ndarray]:
    """
    Load trajectory data from npz file.
    
    Args:
        npz_path: Path to .npz file
        
    Returns:
        Trajectory array with shape [n_trajectories, 48], or None if failed
    """
    try:
        data = np.load(npz_path)
        if 'traj' not in data:
            print(f"Warning: 'traj' key not found in {npz_path.name}")
            return None
        return data['traj']
    except Exception as e:
        print(f"Error loading {npz_path.name}: {e}")
        return None


def load_grid_info(json_path: Path) -> Optional[tuple]:
    """
    Load grid information from json file.
    
    Args:
        json_path: Path to grid .json file
        
    Returns:
        Tuple of (grid_id_set, total_grids), or None if failed
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            grid_data = json.load(f)
        
        grid_ids = [int(k) for k in grid_data.keys()]
        total_grids = max(grid_ids) + 1
        return set(grid_ids), total_grids
    except Exception as e:
        print(f"Error loading {json_path.name}: {e}")
        return None


def compute_visit_matrix(trajectories: np.ndarray, 
                         grid_id_set: set, 
                         total_grids: int) -> np.ndarray:
    """
    Compute visit count matrix from trajectories.
    
    Args:
        trajectories: Array of shape [n_trajectories, 48]
        grid_id_set: Set of valid grid IDs
        total_grids: Total number of grids
        
    Returns:
        Visit matrix of shape [48, total_grids]
    """
    visit_matrix = np.zeros((48, total_grids), dtype=np.int32)
    
    for traj in trajectories:
        if len(traj) != 48:
            continue
        
        for time_idx in range(48):
            grid_id = int(traj[time_idx])
            if 0 <= grid_id < total_grids and grid_id in grid_id_set:
                visit_matrix[time_idx, grid_id] += 1
    
    return visit_matrix


def process_single_city(traj_file: Path, 
                        grid_file: Path, 
                        output_file: Path) -> bool:
    """
    Process a single city's trajectory data.
    
    Args:
        traj_file: Path to trajectory .npz file
        grid_file: Path to grid .json file
        output_file: Path for output .json file
        
    Returns:
        True if successful, False otherwise
    """
    # Load trajectories
    trajectories = load_trajectories(traj_file)
    if trajectories is None:
        return False
    
    # Load grid info
    grid_info = load_grid_info(grid_file)
    if grid_info is None:
        return False
    
    grid_id_set, total_grids = grid_info
    
    # Compute visit matrix
    visit_matrix = compute_visit_matrix(trajectories, grid_id_set, total_grids)
    
    # Save result
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(visit_matrix.tolist(), f)
    
    print(f"Processed {traj_file.stem}: {len(trajectories)} trajectories -> [{48}, {total_grids}]")
    
    # Clean up memory
    del trajectories, visit_matrix
    gc.collect()
    
    return True


def process_all_cities(traj_folder: str,
                       grid_folder: str,
                       output_folder: str,
                       progress_callback: Optional[Callable] = None) -> dict:
    """
    Process all cities in the trajectory folder.
    
    Args:
        traj_folder: Path to folder containing .npz files
        grid_folder: Path to folder containing .json grid files
        output_folder: Path for output files
        progress_callback: Optional callback(current, total, city_name)
        
    Returns:
        Dict with 'success' and 'failed' city lists
    """
    traj_path = Path(traj_folder)
    grid_path = Path(grid_folder)
    output_path = Path(output_folder)
    
    npz_files = list(traj_path.glob("*.npz"))
    
    if not npz_files:
        print("No .npz files found in trajectory folder")
        return {'success': [], 'failed': []}
    
    print(f"Found {len(npz_files)} trajectory files")
    
    success_list = []
    failed_list = []
    
    for i, npz_file in enumerate(npz_files):
        city_name = npz_file.stem
        grid_file = grid_path / f"{city_name}.json"
        output_file = output_path / f"{city_name}.json"
        
        if progress_callback:
            progress_callback(i + 1, len(npz_files), city_name)
        
        if not grid_file.exists():
            print(f"Warning: Grid file not found for {city_name}, skipping")
            failed_list.append(city_name)
            continue
        
        if process_single_city(npz_file, grid_file, output_file):
            success_list.append(city_name)
        else:
            failed_list.append(city_name)
        
        # Periodic garbage collection
        if (i + 1) % 10 == 0:
            gc.collect()
    
    print(f"\nCompleted: {len(success_list)} success, {len(failed_list)} failed")
    
    return {'success': success_list, 'failed': failed_list}


# =============================================================================
if __name__ == "__main__":

    
    parser = argparse.ArgumentParser(
        description="Extract visit count matrix from WorldMove trajectory data"
    )
    parser.add_argument(
        "--traj_dir", "-t",
        required=True,
        help="Directory containing trajectory .npz files"
    )
    parser.add_argument(
        "--grid_dir", "-g",
        required=True,
        help="Directory containing grid .json files"
    )
    parser.add_argument(
        "--output_dir", "-o",
        required=True,
        help="Output directory for per-city visit matrices (.json)"
    )
    
    args = parser.parse_args()
    
    result = process_all_cities(args.traj_dir, args.grid_dir, args.output_dir)
    
    print(f"\nSuccess: {result['success']}")
    if result['failed']:
        print(f"Failed: {result['failed']}")