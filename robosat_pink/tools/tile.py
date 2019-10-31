import os
import sys
from tqdm import tqdm
import concurrent.futures as futures

import numpy as np

import shutil
import mercantile

from rasterio import open as rasterio_open
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from rasterio.warp import transform_bounds, calculate_default_transform
from rasterio.transform import from_bounds

from robosat_pink.core import load_config, check_classes, make_palette, web_ui
from robosat_pink.tiles import (
    tiles_from_csv,
    tile_image_to_file,
    tile_label_to_file,
    tile_from_xyz,
    tile_image_from_file,
    tile_label_from_file,
)


def add_parser(subparser, formatter_class):
    parser = subparser.add_parser("tile", help="Tile a raster, or a rasters coverage", formatter_class=formatter_class)

    inp = parser.add_argument_group("Inputs")
    inp.add_argument("rasters", type=str, nargs="+", help="path to raster files to tile [required]")
    inp.add_argument("--cover", type=str, help="path to csv tiles cover file, to filter tiles to tile [optional]")

    out = parser.add_argument_group("Output")
    out.add_argument("--zoom", type=int, required=True, help="zoom level of tiles [required]")
    out.add_argument("--ts", type=str, default="512,512", help="tile size in pixels [default: 512,512]")
    help = "nodata pixel value, used by default to remove coverage border's tile [default: 0]"
    out.add_argument("--nodata", type=int, default=0, choices=range(0, 256), metavar="[0-255]", help=help)
    help = "Skip tile if nodata pixel ratio > threshold. [default: 100]"
    out.add_argument("--nodata_threshold", type=int, default=100, choices=range(0, 101), metavar="[0-100]", help=help)
    out.add_argument("--keep_borders", action="store_true", help="keep tiles even if borders are empty (nodata)")
    out.add_argument("out", type=str, help="output directory path [required]")

    lab = parser.add_argument_group("Labels")
    lab.add_argument("--label", action="store_true", help="if set, generate label tiles")
    lab.add_argument("--config", type=str, help="path to config file [required in label mode]")

    perf = parser.add_argument_group("Performances")
    perf.add_argument("--workers", type=int, help="number of workers [default: raster files]")

    ui = parser.add_argument_group("Web UI")
    ui.add_argument("--web_ui_base_url", type=str, help="alternate Web UI base URL")
    ui.add_argument("--web_ui_template", type=str, help="alternate Web UI template path")
    ui.add_argument("--no_web_ui", action="store_true", help="desactivate Web UI output")

    parser.set_defaults(func=main)


def is_nodata(image, nodata, threshold, keep_borders = False):
    if (not keep_borders):
        if (
            np.all(image[0, :, :] == nodata)
            or np.all(image[-1, :, :] == nodata)
            or np.all(image[:, 0, :] == nodata)
            or np.all(image[:, -1, :] == nodata)
        ):
            return True  # pixel border is nodata, on all bands

    C, W, H = image.shape
    return np.sum(image[:, :, :] == nodata) >= W * H * (threshold / 100)


def main(args):

    if not args.workers:
        args.workers = min(os.cpu_count(), len(args.rasters))

    if args.label:
        config = load_config(args.config)
        check_classes(config)
        colors = [classe["color"] for classe in config["classes"]]
        palette = make_palette(*colors)

    assert len(args.ts.split(",")) == 2, "--ts expect width,height value (e.g 512,512)"
    width, height = list(map(int, args.ts.split(",")))

    cover = [tile for tile in tiles_from_csv(os.path.expanduser(args.cover))] if args.cover else None

    splits_path = os.path.join(os.path.expanduser(args.out), ".splits")
    tiles_map = {}

    print("RoboSat.pink - tile on CPU, with {} workers".format(args.workers), file=sys.stderr, flush=True)

    bands = -1
    for path in args.rasters:
        raster = rasterio_open(path)
        w, s, e, n = transform_bounds(raster.crs, "EPSG:4326", *raster.bounds)

        if bands != -1:
            assert bands == len(raster.indexes), "Coverage must be bands consistent"
        bands = len(raster.indexes)

        tiles = [mercantile.Tile(x=x, y=y, z=z) for x, y, z in mercantile.tiles(w, s, e, n, args.zoom)]
        tiles = list(set(tiles) & set(cover)) if cover else tiles

        for tile in tiles:
            tile_key = (str(tile.x), str(tile.y), str(tile.z))
            if tile_key not in tiles_map.keys():
                tiles_map[tile_key] = []
            tiles_map[tile_key].append(path)

    if args.label:
        ext = "png"
        bands = 1
    if not args.label:
        if bands == 1:
            ext = "png"
        if bands == 3:
            ext = "webp"
        if bands > 3:
            ext = "tiff"

    tiles = []
    progress = tqdm(total=len(tiles_map), ascii=True, unit="tile")
    # Begin to tile plain tiles
    with futures.ThreadPoolExecutor(args.workers) as executor:

        def worker(path):

            raster = rasterio_open(path)
            w, s, e, n = transform_bounds(raster.crs, "EPSG:4326", *raster.bounds)
            transform, _, _ = calculate_default_transform(raster.crs, "EPSG:3857", raster.width, raster.height, w, s, e, n)
            tiles = [mercantile.Tile(x=x, y=y, z=z) for x, y, z in mercantile.tiles(w, s, e, n, args.zoom)]
            tiled = []

            for tile in tiles:

                if cover and tile not in cover:
                    continue

                w, s, e, n = mercantile.xy_bounds(tile)

                warp_vrt = WarpedVRT(
                    raster,
                    crs="epsg:3857",
                    resampling=Resampling.bilinear,
                    add_alpha=False,
                    transform=from_bounds(w, s, e, n, width, height),
                    width=width,
                    height=height,
                )
                data = warp_vrt.read(out_shape=(len(raster.indexes), width, height), window=warp_vrt.window(w, s, e, n))
                image = np.moveaxis(data, 0, 2)  # C,H,W -> H,W,C

                tile_key = (str(tile.x), str(tile.y), str(tile.z))
                if not args.label and len(tiles_map[tile_key]) == 1 and is_nodata(image, args.nodata, args.nodata_threshold, args.keep_borders):
                    progress.update()
                    continue

                if len(tiles_map[tile_key]) > 1:
                    out = os.path.join(splits_path, str(tiles_map[tile_key].index(path)))
                else:
                    out = args.out

                x, y, z = map(int, tile)

                if not args.label:
                    ret = tile_image_to_file(out, mercantile.Tile(x=x, y=y, z=z), image)
                if args.label:
                    ret = tile_label_to_file(out, mercantile.Tile(x=x, y=y, z=z), palette, image)

                assert ret, "Unable to write tile {} from raster {}.".format(str(tile), raster)

                if len(tiles_map[tile_key]) == 1:
                    progress.update()
                    tiled.append(mercantile.Tile(x=x, y=y, z=z))

            return tiled

        for tiled in executor.map(worker, args.rasters):
            if tiled is not None:
                tiles.extend(tiled)

    # Aggregate remaining tiles splits
    with futures.ThreadPoolExecutor(args.workers) as executor:

        def worker(tile_key):

            if len(tiles_map[tile_key]) == 1:
                return

            image = np.zeros((width, height, bands), np.uint8)

            x, y, z = map(int, tile_key)
            for i in range(len(tiles_map[tile_key])):
                root = os.path.join(splits_path, str(i))
                _, path = tile_from_xyz(root, x, y, z)

                if not args.label:
                    split = tile_image_from_file(path)
                if args.label:
                    split = tile_label_from_file(path)
                    split = split.reshape((width, height, 1))  # H,W -> H,W,C

                assert image.shape == split.shape
                image[np.where(image == 0)] += split[np.where(image == 0)]

            if not args.label and is_nodata(image, args.nodata, args.nodata_threshold, args.keep_borders):
                progress.update()
                return

            tile = mercantile.Tile(x=x, y=y, z=z)

            if not args.label:
                ret = tile_image_to_file(args.out, tile, image)

            if args.label:
                ret = tile_label_to_file(args.out, tile, palette, image)

            assert ret, "Unable to write tile {} from raster {}.".format(str(tile_key))

            progress.update()
            return tile

        for tiled in executor.map(worker, tiles_map.keys()):
            if tiled is not None:
                tiles.append(tiled)

        if splits_path and os.path.isdir(splits_path):
            shutil.rmtree(splits_path)  # Delete suffixes dir if any

    if tiles and not args.no_web_ui:
        template = "leaflet.html" if not args.web_ui_template else args.web_ui_template
        base_url = args.web_ui_base_url if args.web_ui_base_url else "."
        web_ui(args.out, base_url, tiles, tiles, ext, template)
