"""
Cleans shapefiles before importing them into Reforger Workbench.

Removes/fixes the kinds of problems that cause "invalid segments" warnings
or "Entity out of world bounds" crashes on import:
  - features with null/empty geometry
  - features whose coordinates fall way outside your expected bounding box
    (e.g. a point that's still in degrees instead of UTM meters)
  - invalid geometries (self-intersections etc.) -> repaired
  - duplicate/near-duplicate consecutive vertices -> removed
  - resulting zero-length segments -> removed

Usage:
    python3 clean_shapefiles.py input1.shp input2.shp ... \
        --xmin 421526 --ymin 5801129 --xmax 431526 --ymax 5811129 \
        --out-dir cleaned/

Requires: geopandas, shapely
    pip install geopandas shapely
"""

import argparse
import glob
import os
import geopandas as gpd
from shapely.geometry import LineString, MultiLineString
from shapely.validation import make_valid


def remove_duplicate_points(coords, tolerance=0.01):
    """Drop consecutive points that are (near-)identical, which create
    zero-length segments the Road Generator flags as invalid."""
    if not coords:
        return coords
    cleaned = [coords[0]]
    for pt in coords[1:]:
        last = cleaned[-1]
        if abs(pt[0] - last[0]) > tolerance or abs(pt[1] - last[1]) > tolerance:
            cleaned.append(pt)
    return cleaned


def clean_linestring(geom, tolerance=0.01):
    if geom is None or geom.is_empty:
        return None

    if isinstance(geom, MultiLineString):
        parts = [clean_linestring(part, tolerance) for part in geom.geoms]
        parts = [p for p in parts if p is not None]
        if not parts:
            return None
        return MultiLineString(parts) if len(parts) > 1 else parts[0]

    if isinstance(geom, LineString):
        coords = remove_duplicate_points(list(geom.coords), tolerance)
        if len(coords) < 2:
            return None
        return LineString(coords)

    return geom


def clean_polygon(geom):
    if geom is None or geom.is_empty:
        return None
    if not geom.is_valid:
        geom = make_valid(geom)
    if geom.is_empty or geom.area == 0:
        return None
    return geom


def clean_shapefile(path, xmin, ymin, xmax, ymax, out_dir, margin=100, vertex_tolerance=0.01,
                     target_epsg=25833):
    print(f"\n=== {os.path.basename(path)} ===")
    gdf = gpd.read_file(path)
    original_count = len(gdf)
    print(f"Loaded {original_count} feature(s)")
    print(f"  CRS: {gdf.crs}")
    print(f"  Bounds: {gdf.total_bounds}")

    if original_count == 0:
        print("  File is already empty, skipping.")
        return

    # Reproject to the target CRS (UTM meters) if it isn't already --
    # this is the fix for shapefiles that got exported still in
    # EPSG:4326 (lat/lon degrees), which would otherwise fail every
    # bounds check below and produce an empty output.
    target_crs = f"EPSG:{target_epsg}"
    if gdf.crs is None:
        print(f"  WARNING: no CRS set on this file -- assuming it's already {target_crs}. "
              f"If that's wrong, the bounds check below will drop everything.")
    elif str(gdf.crs).upper() not in (target_crs, f"EPSG:{target_epsg}"):
        print(f"  Reprojecting from {gdf.crs} to {target_crs}...")
        gdf = gdf.to_crs(target_crs)
        print(f"  New bounds: {gdf.total_bounds}")

    # 1. Drop null/empty geometries
    gdf = gdf[~gdf.geometry.isna() & ~gdf.geometry.is_empty]
    dropped_empty = original_count - len(gdf)
    if dropped_empty:
        print(f"  Dropped {dropped_empty} feature(s) with null/empty geometry")

    # 2. Drop features whose bounding box falls (mostly) outside the
    #    expected extent, with a margin -- these are the "wrong CRS" /
    #    degenerate-coordinate culprits (e.g. still in lat/lon degrees).
    exp_xmin, exp_xmax = xmin - margin, xmax + margin
    exp_ymin, exp_ymax = ymin - margin, ymax + margin

    def in_bounds(geom):
        gx_min, gy_min, gx_max, gy_max = geom.bounds
        return not (gx_max < exp_xmin or gx_min > exp_xmax or
                    gy_max < exp_ymin or gy_min > exp_ymax)

    before = len(gdf)
    gdf = gdf[gdf.geometry.apply(in_bounds)]
    dropped_oob = before - len(gdf)
    if dropped_oob:
        print(f"  Dropped {dropped_oob} feature(s) far outside expected bounds "
              f"({dropped_oob} likely bad-coordinate feature(s))")

    # 3. Fix invalid geometry / remove duplicate vertices
    geom_type = gdf.geometry.geom_type.iloc[0] if len(gdf) else None
    if geom_type in ("LineString", "MultiLineString"):
        gdf["geometry"] = gdf.geometry.apply(lambda g: clean_linestring(g, vertex_tolerance))
    elif geom_type in ("Polygon", "MultiPolygon"):
        gdf["geometry"] = gdf.geometry.apply(clean_polygon)

    before = len(gdf)
    gdf = gdf[~gdf.geometry.isna() & ~gdf.geometry.is_empty]
    dropped_degenerate = before - len(gdf)
    if dropped_degenerate:
        print(f"  Dropped {dropped_degenerate} feature(s) that became degenerate after cleaning")

    print(f"  Final: {len(gdf)} feature(s) kept "
          f"(removed {original_count - len(gdf)} total)")

    if len(gdf) == 0:
        print("  WARNING: all features were dropped! Check the CRS/bounds printed above --  "
              "this almost always means the source file's coordinates don't match the "
              "--xmin/--ymin/--xmax/--ymax you passed in.")
        return

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, os.path.basename(path))
    gdf.to_file(out_path, driver="ESRI Shapefile")
    print(f"  Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Clean shapefiles before Enfusion import")
    parser.add_argument("inputs", nargs="+", help="Input .shp file(s)")
    parser.add_argument("--xmin", type=float, required=True)
    parser.add_argument("--ymin", type=float, required=True)
    parser.add_argument("--xmax", type=float, required=True)
    parser.add_argument("--ymax", type=float, required=True)
    parser.add_argument("--out-dir", default="cleaned", help="Output directory")
    parser.add_argument("--margin", type=float, default=100,
                         help="Allowed margin (meters) outside the bbox before a feature is dropped")
    parser.add_argument("--vertex-tolerance", type=float, default=0.01,
                         help="Minimum distance (meters) between consecutive vertices")
    parser.add_argument("--target-epsg", type=int, default=25833,
                         help="EPSG code your bbox coordinates are in (default: 25833, "
                              "ETRS89/UTM 33N for Brandenburg). Files in a different CRS "
                              "get reprojected to this automatically.")
    args = parser.parse_args()

    # Expand wildcards ourselves -- Windows shells (cmd/PowerShell) don't
    # do this automatically like bash/zsh do.
    input_paths = []
    for pattern in args.inputs:
        matches = glob.glob(pattern)
        if matches:
            input_paths.extend(matches)
        else:
            input_paths.append(pattern)  # let it fail later with a clear error if truly missing

    if not input_paths:
        print("No input files matched.")
        return

    for path in input_paths:
        clean_shapefile(path, args.xmin, args.ymin, args.xmax, args.ymax,
                         args.out_dir, args.margin, args.vertex_tolerance, args.target_epsg)

    print(f"\nDone. Cleaned files are in '{args.out_dir}/'")


if __name__ == "__main__":
    main()
