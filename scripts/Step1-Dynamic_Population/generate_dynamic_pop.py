# -*- coding: utf-8 -*-
"""
Dynamic Population Generation Module

Generate dynamic population estimates from mobility trajectory data using 
power-law model calibration.

Methodology:
    Based on Deville et al. (2014) and Liu et al. (2018):
    1. Fit power-law model: ρc = α × σc^β using nighttime (3-4 AM) data
    2. Apply calibrated model to all 48 time slots
    3. Calibrate total population to match census data

Input (from WorldMove dataset):
    - Visit matrix (.json): Shape [48, n_grids], extracted from WorldMove 
      synthetic trajectory data (each trajectory has 48 time slots per day,
      30-min intervals)
    - Population data (.npy): Static population grid (1km×1km) from WorldMove,
      aggregated from WorldPop 100m resolution data

Output:
    - Dynamic population (.npy): Shape [48, n_grids]
    - Calibration parameters (.json): α, β, r, RMSE statistics

Data Source:
    WorldMove: An open-access worldwide human mobility dataset providing 
    mobility data for 1600+ cities across 179 countries. Uses generative 
    AI to create city-scale mobility trajectories from population distribution,
    POIs, and synthetic commuting OD flows.
    
References:
    - Deville et al. (2014) Dynamic population mapping using mobile phone data. 
      PNAS, 111(45), 15888-15893.
    - Liu et al. (2018) Mapping hourly dynamics of urban population using 
      trajectories reconstructed from mobile phone records. 
      Transactions in GIS, 22(2), 494-513.
"""

import json
import numpy as np
from scipy.stats import pearsonr
from pathlib import Path
from typing import Optional, Callable, Tuple, Dict, Any


# =============================================================================
# Data Loading Functions
# =============================================================================

def load_population_data(pop_file: str) -> Tuple[np.ndarray, tuple]:
    """
    Load static population data from numpy file.
    
    Args:
        pop_file: Path to population .npy file (from WorldMove dataset,
                  aggregated from WorldPop 100m data to 1km grid)
        
    Returns:
        Tuple of (flattened population array, original shape)
        
    Note:
        Data is flattened in row-major (C) order:
        [[0,1,2], [3,4,5]] -> [0,1,2,3,4,5]
        This corresponds to grid numbering convention.
    """
    pop_data = np.load(pop_file)
    return pop_data.flatten(), pop_data.shape


def load_visit_data_nighttime(visit_file: str) -> np.ndarray:
    """
    Load visit data and extract nighttime (3:00-4:00 AM) period.
    
    Used for power-law model calibration, as nighttime population
    distribution most closely matches residential census data.
    
    Args:
        visit_file: Path to visit data file (.json or .npy), derived from
                    WorldMove synthetic trajectory data
        
    Returns:
        1D array of nighttime visit counts per grid
    """
    if visit_file.endswith('.json'):
        with open(visit_file, 'r', encoding='utf-8') as f:
            visit_data = np.array(json.load(f))
        # JSON format: [48, n_grids]
        # Time slots 6 and 7 correspond to 3:00-4:00 AM (30-min intervals)
        return (visit_data[6, :] + visit_data[7, :]) / 2.0
    
    elif visit_file.endswith('.npy'):
        visit_data = np.load(visit_file)
        # NPY format: [rows, cols, 48]
        nighttime = (visit_data[:, :, 6] + visit_data[:, :, 7]) / 2.0
        return nighttime.flatten()
    
    else:
        raise ValueError(f"Unsupported file format: {visit_file}")


def load_visit_data_all_timeslots(visit_file: str) -> np.ndarray:
    """
    Load visit data for all 48 time slots.
    
    Args:
        visit_file: Path to visit data file (.json or .npy), derived from
                    WorldMove synthetic trajectory data (48 slots × 30min = 24h)
        
    Returns:
        2D array of shape [48, n_grids]
    """
    if visit_file.endswith('.json'):
        with open(visit_file, 'r', encoding='utf-8') as f:
            visit_data = np.array(json.load(f))
        return visit_data  # Already [48, n_grids]
    
    elif visit_file.endswith('.npy'):
        visit_data = np.load(visit_file)
        # [rows, cols, 48] -> [48, n_grids]
        rows, cols, timeslots = visit_data.shape
        return visit_data.transpose(2, 0, 1).reshape(48, rows * cols)
    
    else:
        raise ValueError(f"Unsupported file format: {visit_file}")


# =============================================================================
# Power-Law Model Functions
# =============================================================================

def fit_power_law(visit_density: np.ndarray, 
                  pop_density: np.ndarray) -> Dict[str, float]:
    """
    Fit power-law model parameters using Ordinary Least Squares.
    
    Model: ρc = α × σc^β
    Log-transformed: log(ρc) = log(α) + β × log(σc)
    
    Based on Liu et al. (2018) methodology.
    
    Args:
        visit_density: Mobile phone user density (σc)
        pop_density: Census population density (ρc)
        
    Returns:
        Dict containing:
            - alpha: Scale parameter
            - beta: Superlinear effect parameter (typically 0.7-0.9)
            - r: Pearson correlation coefficient
            - rmse: Root mean square error
            - n_points: Number of valid data points used
    """
    # Filter zero/negative values (required for log transform)
    mask = (visit_density > 0) & (pop_density > 0)
    sigma = visit_density[mask]
    rho = pop_density[mask]
    n_points = len(sigma)
    
    if n_points < 2:
        # Insufficient data: use linear approximation
        if n_points == 1:
            alpha = float(rho[0] / sigma[0])
            return {
                'alpha': alpha, 'beta': 1.0, 'r': 1.0, 
                'rmse': 0.0, 'n_points': 1
            }
        raise ValueError("No valid data points for fitting")
    
    # Log-linear regression
    log_sigma = np.log(sigma)
    log_rho = np.log(rho)
    
    # OLS fit: log(ρ) = log(α) + β × log(σ)
    coeffs = np.polyfit(log_sigma, log_rho, deg=1)
    beta = coeffs[0]
    alpha = np.exp(coeffs[1])
    
    # Compute fit statistics
    rho_predicted = alpha * np.power(sigma, beta)
    r, _ = pearsonr(rho, rho_predicted)
    rmse = np.sqrt(np.mean((rho - rho_predicted) ** 2))
    
    return {
        'alpha': float(alpha),
        'beta': float(beta),
        'r': float(r),
        'rmse': float(rmse),
        'n_points': int(n_points)
    }


def apply_power_law(visit_density: np.ndarray, 
                    alpha: float, 
                    beta: float) -> np.ndarray:
    """
    Apply power-law model to estimate population.
    
    Args:
        visit_density: Visit count array
        alpha: Scale parameter
        beta: Superlinear parameter
        
    Returns:
        Estimated population density array
    """
    # Handle zero values to avoid numerical issues
    safe_density = np.maximum(visit_density, 1e-10)
    return alpha * np.power(safe_density, beta)


# =============================================================================
# Dynamic Population Generation
# =============================================================================

def generate_dynamic_population(visit_data: np.ndarray,
                                static_pop: np.ndarray,
                                alpha: float,
                                beta: float) -> np.ndarray:
    """
    Generate dynamic population for all 48 time slots.
    
    Key implementation details:
    1. Grids with static_pop=0 remain 0 (no prediction)
    2. Power-law model applied only to grids with static_pop>0
    3. Each time slot calibrated to match total static population
    
    Args:
        visit_data: Visit matrix [48, n_grids]
        static_pop: Static population array [n_grids]
        alpha: Calibrated scale parameter
        beta: Calibrated superlinear parameter
        
    Returns:
        Dynamic population matrix [48, n_grids]
    """
    n_timeslots, n_grids = visit_data.shape
    
    # Identify valid grids (static_pop > 0)
    valid_mask = static_pop > 0
    total_static_pop = np.sum(static_pop[valid_mask])
    
    # Initialize output matrix
    dynamic_pop = np.zeros((n_timeslots, n_grids))
    
    for t in range(n_timeslots):
        visit_t = visit_data[t, :]
        
        # Apply power-law only to valid grids
        pop_predicted = np.zeros(n_grids)
        
        if np.any(valid_mask):
            # Predict for valid grids
            pop_valid = apply_power_law(visit_t[valid_mask], alpha, beta)
            
            # Calibrate to match total static population
            total_predicted = np.sum(pop_valid)
            if total_predicted > 0:
                calibration_factor = total_static_pop / total_predicted
                pop_valid *= calibration_factor
            
            pop_predicted[valid_mask] = pop_valid
        
        dynamic_pop[t, :] = pop_predicted
    
    return dynamic_pop


# =============================================================================
# Main Processing Functions
# =============================================================================

def process_single_city(visit_file: str,
                        pop_file: str,
                        output_file: str,
                        params_file: Optional[str] = None) -> Dict[str, Any]:
    """
    Process a single city: fit model and generate dynamic population.
    
    Args:
        visit_file: Path to visit data file
        pop_file: Path to static population file
        output_file: Path for output dynamic population
        params_file: Optional path to save calibration parameters
        
    Returns:
        Dict with calibration parameters and processing info
    """
    # Load data
    static_pop, pop_shape = load_population_data(pop_file)
    visit_nighttime = load_visit_data_nighttime(visit_file)
    visit_all = load_visit_data_all_timeslots(visit_file)
    
    # Validate dimensions
    if len(static_pop) != len(visit_nighttime):
        raise ValueError(
            f"Grid count mismatch: pop={len(static_pop)}, visit={len(visit_nighttime)}"
        )
    
    # Fit power-law model using nighttime data
    params = fit_power_law(visit_nighttime, static_pop)
    params['pop_shape'] = list(pop_shape)
    params['n_grids'] = len(static_pop)
    
    # Generate dynamic population
    dynamic_pop = generate_dynamic_population(
        visit_all, static_pop, params['alpha'], params['beta']
    )
    
    # Save outputs
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    np.save(output_file, dynamic_pop)
    
    if params_file:
        Path(params_file).parent.mkdir(parents=True, exist_ok=True)
        with open(params_file, 'w', encoding='utf-8') as f:
            json.dump(params, f, indent=2)
    
    return params


def process_all_cities(visit_folder: str,
                       pop_folder: str,
                       output_folder: str,
                       params_folder: Optional[str] = None,
                       progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
    """
    Process all cities in the input folders.
    
    Args:
        visit_folder: Folder containing visit data files
        pop_folder: Folder containing population data files
        output_folder: Folder for output dynamic population files
        params_folder: Optional folder for calibration parameters
        progress_callback: Optional callback(current, total, city_name)
        
    Returns:
        Dict with 'success' and 'failed' results
    """
    visit_path = Path(visit_folder)
    pop_path = Path(pop_folder)
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if params_folder:
        params_path = Path(params_folder)
        params_path.mkdir(parents=True, exist_ok=True)
    
    # Find matching city files
    visit_files = {f.stem: f for f in visit_path.glob("*.json")}
    visit_files.update({f.stem: f for f in visit_path.glob("*.npy")})
    pop_files = {f.stem: f for f in pop_path.glob("*.npy")}
    
    common_cities = set(visit_files.keys()) & set(pop_files.keys())
    
    if not common_cities:
        raise ValueError("No matching city data found")
    
    print(f"Found {len(common_cities)} cities to process")
    
    results = {'success': {}, 'failed': {}}
    
    for i, city in enumerate(sorted(common_cities), 1):
        if progress_callback:
            progress_callback(i, len(common_cities), city)
        
        try:
            output_file = str(output_path / f"{city}.npy")
            params_file = str(params_path / f"{city}.json") if params_folder else None
            
            params = process_single_city(
                str(visit_files[city]),
                str(pop_files[city]),
                output_file,
                params_file
            )
            
            results['success'][city] = params
            print(f"[{i}/{len(common_cities)}] {city}: α={params['alpha']:.4f}, "
                  f"β={params['beta']:.4f}, r={params['r']:.4f}")
            
        except Exception as e:
            results['failed'][city] = str(e)
            print(f"[{i}/{len(common_cities)}] {city}: FAILED - {e}")
    
    # Save summary
    summary_file = output_path / "_processing_summary.json"
    summary = {
        'total_cities': len(common_cities),
        'success_count': len(results['success']),
        'failed_count': len(results['failed']),
        'failed_cities': results['failed']
    }
    
    if results['success']:
        alphas = [r['alpha'] for r in results['success'].values()]
        betas = [r['beta'] for r in results['success'].values()]
        rs = [r['r'] for r in results['success'].values()]
        
        summary['statistics'] = {
            'alpha_mean': float(np.mean(alphas)),
            'alpha_std': float(np.std(alphas)),
            'beta_mean': float(np.mean(betas)),
            'beta_std': float(np.std(betas)),
            'r_mean': float(np.mean(rs)),
            'r_std': float(np.std(rs))
        }
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nCompleted: {len(results['success'])} success, {len(results['failed'])} failed")
    print(f"Summary saved to: {summary_file}")
    
    return results

if __name__ == "__main__":
    
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate dynamic population estimates using power-law model"
    )
    parser.add_argument(
        "--visit_dir", "-v", required=True,
        help="Directory containing visit files (.json/.npy)"
    )
    parser.add_argument(
        "--pop_dir", "-p", required=True,
        help="Directory containing population files (.npy)"
    )
    parser.add_argument(
        "--output_dir", "-o", required=True,
        help="Output directory for dynamic population files (.npy)"
    )
    parser.add_argument(
        "--params_dir", default=None,
        help="Optional output directory for calibration parameters (.json)"
    )
    
    args = parser.parse_args()
    
    process_all_cities(args.visit_dir, args.pop_dir, args.output_dir, args.params_dir)