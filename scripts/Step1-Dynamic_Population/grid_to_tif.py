"""
Grid JSON -> GeoTIFF reference (batch).

For each city:
- Read the center lon/lat of grid id 0 from `{city}.json`.
- Read `{city}.npy` to get the grid shape (rows, cols).
- Create a 1 km regular grid GeoTIFF in EPSG:4326.

Notes:
- The output GeoTIFF is intended as a spatial reference (bbox/transform) for downstream steps.
- By default this script writes a zero-filled raster to avoid embedding any semantic values.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np
import rasterio
from rasterio.transform import from_origin


@dataclass(frozen=True)
class GridRef:
    rows: int
    cols: int
    center_lon: float
    center_lat: float
    lon_res: float
    lat_res: float


def _load_grid0_center(json_path: Path) -> Tuple[float, float]:
    with open(json_path, "r", encoding="utf-8") as f:
        grid_data = json.load(f)
    origin = grid_data.get("0", grid_data.get(0))
    if origin is None or not isinstance(origin, (list, tuple)) or len(origin) != 2:
        raise ValueError("Grid JSON must contain key '0' with [lon, lat].")
    return float(origin[0]), float(origin[1])


def _load_npy_shape(npy_path: Path) -> Tuple[int, int]:
    arr = np.load(npy_path)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D npy array, got shape={arr.shape}.")
    return int(arr.shape[0]), int(arr.shape[1])


def _km_to_degree_resolution(cell_size_km: float, center_lat: float) -> Tuple[float, float]:
    if cell_size_km <= 0:
        raise ValueError("cell_size_km must be > 0.")
    lat_res = cell_size_km / 111.0
    cos_lat = math.cos(math.radians(abs(center_lat)))
    if cos_lat <= 0:
        raise ValueError("Invalid latitude for resolution computation.")
    lon_res = cell_size_km / (111.0 * cos_lat)
    return lon_res, lat_res


def build_grid_ref(json_path: Path, npy_path: Path, cell_size_km: float) -> GridRef:
    center_lon, center_lat = _load_grid0_center(json_path)
    rows, cols = _load_npy_shape(npy_path)
    lon_res, lat_res = _km_to_degree_resolution(cell_size_km, center_lat)
    return GridRef(
        rows=rows,
        cols=cols,
        center_lon=center_lon,
        center_lat=center_lat,
        lon_res=lon_res,
        lat_res=lat_res,
    )


def write_reference_tif(
    out_path: Path,
    ref: GridRef,
    crs: str,
    *,
    dtype: str = "float32",
    write_data_from_npy: Optional[Path] = None,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    upper_left_lon = ref.center_lon - (ref.lon_res / 2.0)
    upper_left_lat = ref.center_lat + (ref.lat_res / 2.0)
    transform = from_origin(upper_left_lon, upper_left_lat, ref.lon_res, ref.lat_res)

    if write_data_from_npy is not None:
        data = np.load(write_data_from_npy)
        if data.shape != (ref.rows, ref.cols):
            raise ValueError("NPY shape does not match computed grid shape.")
        data_to_write = data
        out_dtype = data.dtype
    else:
        data_to_write = np.zeros((ref.rows, ref.cols), dtype=np.dtype(dtype))
        out_dtype = data_to_write.dtype

    with rasterio.open(
        out_path,
        "w",
        driver="GTiff",
        height=ref.rows,
        width=ref.cols,
        count=1,
        dtype=out_dtype,
        crs=crs,
        transform=transform,
        nodata=None,
        compress="lzw",
    ) as dst:
        dst.write(data_to_write, 1)
        dst.update_tags(
            AREA_OR_POINT="Area",
            cell_size_km=str(float(ref.lat_res) * 111.0),
            origin_lon=str(ref.center_lon),
            origin_lat=str(ref.center_lat),
        )


def iter_city_stems(grid_json_dir: Path) -> Iterable[str]:
    for p in sorted(grid_json_dir.glob("*.json")):
        yield p.stem


def run_batch(grid_json_dir: Path, npy_dir: Path, output_dir: Path, *, cell_size_km: float, crs: str, embed_npy: bool) -> int:
    json_files = sorted(grid_json_dir.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No .json files found in: {grid_json_dir}")

    ok = 0
    fail = 0

    for json_path in json_files:
        city = json_path.stem
        npy_path = npy_dir / f"{city}.npy"
        out_path = output_dir / f"{city}.tif"

        try:
            if not npy_path.exists():
                raise FileNotFoundError(f"Missing npy for city '{city}': {npy_path}")

            ref = build_grid_ref(json_path, npy_path, cell_size_km=cell_size_km)
            write_reference_tif(
                out_path,
                ref,
                crs,
                write_data_from_npy=npy_path if embed_npy else None,
            )
            ok += 1
            print(f"[OK] {city}: wrote {out_path.name} ({ref.rows}x{ref.cols})")
        except Exception as e:
            fail += 1
            print(f"[FAIL] {city}: {e}")

    print(f"Done. success={ok}, failed={fail}, out_dir={output_dir}")
    return 0 if fail == 0 else 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch-generate GeoTIFF grid references from grid JSON + NPY shape.")
    parser.add_argument("--grid_json_dir", type=str, required=True, help="Directory containing {city}.json grid files.")
    parser.add_argument("--npy_dir", type=str, required=True, help="Directory containing {city}.npy files (2D).")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for {city}.tif files.")
    parser.add_argument("--cell_size_km", type=float, default=1.0, help="Cell size in km (default: 1.0).")
    parser.add_argument("--crs", type=str, default="EPSG:4326", help="Output CRS (default: EPSG:4326).")
    parser.add_argument(
        "--embed_npy_values",
        action="store_true",
        help="Write NPY values into the GeoTIFF instead of zeros (off by default).",
    )

    args = parser.parse_args()

    exit_code = run_batch(
        grid_json_dir=Path(args.grid_json_dir),
        npy_dir=Path(args.npy_dir),
        output_dir=Path(args.output_dir),
        cell_size_km=float(args.cell_size_km),
        crs=str(args.crs),
        embed_npy=bool(args.embed_npy_values),
    )
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()