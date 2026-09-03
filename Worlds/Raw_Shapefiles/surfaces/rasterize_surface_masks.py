"""
Rasterizes ATKIS surface shapefiles into Enfusion terrain surface masks.

Reads the polygon shapefiles in this folder, clips them to the world bounding
box, and burns them into the 8-bit greyscale PNGs in Worlds/surface_masks_export/
that the Workbench terrain uses as per-surface weight layers.

How the masks work
------------------
Each PNG in surface_masks_export/ corresponds 1:1 to a material listed in
Terrain.terr (Grass_01.png <-> Terrains/Common/Surfaces/Grass_01.emat).
A pixel value of 255 means "this surface at full weight here", 0 means absent.
default.png is the base layer: it starts fully white, and every pixel we paint
into another mask gets carved out of default so the weights stay consistent.

IMPORTANT -- the 5-surface budget
---------------------------------
Enfusion allows at most 5 surfaces blending per terrain tile. default.png
counts as one, so RULES below is deliberately kept to 4 painted masks. If you
add a rule, either drop another or make sure the new surface never shares a
tile with the ones it would push over the limit.

Later rules in RULES win where polygons overlap, so order matters: broad
background classes first, specific ones last.

Usage:
    python rasterize_surface_masks.py                # write the masks
    python rasterize_surface_masks.py --dry-run      # report coverage only
    python rasterize_surface_masks.py --only forest  # just one group

Requires: geopandas, shapely, rasterio, pillow, numpy
"""

import argparse
import os
import sys

import numpy as np
import geopandas as gpd
from shapely.geometry import box
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from PIL import Image

Image.MAX_IMAGE_PIXELS = None  # the masks are 16129^2, well past PIL's bomb guard

HERE = os.path.dirname(os.path.abspath(__file__))
MASK_DIR = os.path.normpath(os.path.join(HERE, "..", "..", "surface_masks_export"))

# World bounding box in EPSG:25833 (ETRS89 / UTM 33N), matching the extent the
# terrain was built from -- same numbers used for the road import.
XMIN, YMIN, XMAX, YMAX = 421526.0, 5801129.0, 431526.0, 5811129.0
EPSG = 25833

MASK_SIZE = 16129  # px per side, from the existing exported masks
DEFAULT_MASK = "default.png"

# Which shapefile (plus optional attribute filter) paints which mask.
# Applied in order; later entries overwrite earlier ones on overlap.
# Kept to 4 masks + default to stay inside the 5-surfaces-per-tile limit.
RULES = [
    # (group, shapefile, mask, filter)
    ("grass",  "shp_landwirtschaft_gruenland.shp",  "Grass_01.png",       None),
    ("dirt",   "shp_unland.shp",                    "Dirt_01.png",        None),
    ("crops",  "shp_landwirtschaft_ackerland.shp",  "Crop_Field_01.png",  None),
    # All woodland onto one pine mask -- Brandenburg is ~85% coniferous here,
    # and a single forest surface keeps us inside the per-tile budget. The
    # ATKIS vegetation codes (1100 deciduous, 1200 coniferous, 1300 mixed) are
    # available if you later want to split this across separate masks.
    ("forest", "shp_wald.shp",                      "ForestPine_01_Base.png", None),
]


def load_clipped(shp_path, attr_filter, clip_geom):
    """Load a shapefile, reproject if needed, apply the attribute filter and
    clip to the world box. Returns a list of geometries (possibly empty)."""
    gdf = gpd.read_file(shp_path)
    if len(gdf) == 0:
        return []

    if gdf.crs is None:
        print("      WARNING: no CRS on %s, assuming EPSG:%d"
              % (os.path.basename(shp_path), EPSG))
    elif gdf.crs.to_epsg() != EPSG:
        gdf = gdf.to_crs(epsg=EPSG)

    if attr_filter:
        col, val = attr_filter
        if col not in gdf.columns:
            print("      WARNING: column '%s' missing, skipping this rule" % col)
            return []
        gdf = gdf[gdf[col] == val]

    gdf = gdf[~gdf.geometry.isna() & ~gdf.geometry.is_empty]
    if len(gdf) == 0:
        return []

    # buffer(0) repairs self-intersections, which would otherwise rasterize
    # with holes or spill outside the ring.
    invalid = ~gdf.geometry.is_valid
    if invalid.any():
        gdf.loc[invalid, "geometry"] = gdf.loc[invalid, "geometry"].buffer(0)

    clipped = gdf.geometry.intersection(clip_geom)
    clipped = clipped[~clipped.is_empty & ~clipped.isna()]
    return [g for g in clipped if g.area > 0]


def main():
    ap = argparse.ArgumentParser(description="Rasterize shapefiles into Enfusion surface masks")
    ap.add_argument("--dry-run", action="store_true",
                    help="Report what would be painted without writing any PNG")
    ap.add_argument("--only", action="append",
                    help="Limit to one or more groups (grass, dirt, crops, forest)")
    ap.add_argument("--mask-dir", default=MASK_DIR, help="Directory holding the mask PNGs")
    args = ap.parse_args()

    rules = RULES
    if args.only:
        wanted = {g.lower() for g in args.only}
        rules = [r for r in RULES if r[0] in wanted]
        if not rules:
            print("No rules match --only %s. Groups: %s"
                  % (args.only, sorted({r[0] for r in RULES})))
            return 1

    transform = from_bounds(XMIN, YMIN, XMAX, YMAX, MASK_SIZE, MASK_SIZE)
    clip_geom = box(XMIN, YMIN, XMAX, YMAX)
    px_area = ((XMAX - XMIN) / MASK_SIZE) ** 2

    print("World box : %.0f, %.0f -> %.0f, %.0f (EPSG:%d)" % (XMIN, YMIN, XMAX, YMAX, EPSG))
    print("Mask size : %d x %d  (%.4f m/px)" % (MASK_SIZE, MASK_SIZE, (XMAX - XMIN) / MASK_SIZE))
    print("Mask dir  : %s\n" % args.mask_dir)

    # claimed tracks which pixels any surface has taken, so we can carve them
    # out of default.png at the end.
    claimed = np.zeros((MASK_SIZE, MASK_SIZE), dtype=bool)
    painted = {}  # mask filename -> bool array

    for group, shp_name, mask_name, attr_filter in rules:
        shp_path = os.path.join(HERE, shp_name)
        label = "[%s] %s" % (group, shp_name)
        if attr_filter:
            label += " (%s=%s)" % attr_filter
        print("%s\n    -> %s" % (label, mask_name))

        if not os.path.exists(shp_path):
            print("    MISSING shapefile, skipped\n")
            continue

        geoms = load_clipped(shp_path, attr_filter, clip_geom)
        if not geoms:
            print("    no features inside the world box, skipped\n")
            continue

        area = sum(g.area for g in geoms)
        burn = rasterize(
            [(g, 1) for g in geoms],
            out_shape=(MASK_SIZE, MASK_SIZE),
            transform=transform,
            fill=0,
            all_touched=False,
            dtype=np.uint8,
        ).astype(bool)

        # Later rules win: clear these pixels from anything painted earlier.
        for other, arr in painted.items():
            if other != mask_name:
                arr &= ~burn

        if mask_name in painted:
            painted[mask_name] |= burn
        else:
            painted[mask_name] = burn

        claimed |= burn
        print("    %d polygons, %.2f km2 clipped, %d px (%.2f%% of map)\n"
              % (len(geoms), area / 1e6, burn.sum(), burn.mean() * 100))

    if not painted:
        print("Nothing to paint.")
        return 0

    print("Total painted coverage: %.2f%% of the map (%.2f km2)"
          % (claimed.mean() * 100, claimed.sum() * px_area / 1e6))
    print("Surfaces in play: %d painted + default = %d (limit is 5 per tile)"
          % (len(painted), len(painted) + 1))

    if args.dry_run:
        print("\nDry run -- no files written.")
        return 0

    print()
    for mask_name, arr in painted.items():
        path = os.path.join(args.mask_dir, mask_name)
        if not os.path.exists(path):
            print("  SKIP %s: not in surface_masks_export "
                  "(is the material added to the terrain?)" % mask_name)
            continue
        out = np.where(arr, np.uint8(255), np.uint8(0))
        Image.fromarray(out, mode="L").save(path, optimize=True)
        print("  wrote %-34s %6.2f%%" % (mask_name, arr.mean() * 100))

    # Carve every painted pixel out of the base layer.
    default_path = os.path.join(args.mask_dir, DEFAULT_MASK)
    if os.path.exists(default_path):
        base = np.full((MASK_SIZE, MASK_SIZE), 255, dtype=np.uint8)
        base[claimed] = 0
        Image.fromarray(base, mode="L").save(default_path, optimize=True)
        print("  wrote %-34s %6.2f%% (base layer)" % (DEFAULT_MASK, (~claimed).mean() * 100))
    else:
        print("  WARNING: %s not found, base layer left untouched" % DEFAULT_MASK)

    print("\nDone. Reimport the masks in the Workbench terrain tool to see them.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
