"""Slippy Map Tiles.
   See: https://wiki.openstreetmap.org/wiki/Slippy_map_tilenames
"""

import io
import os
import re
import glob
import warnings

import numpy as np
from PIL import Image
from rasterio import open as rasterio_open
import cv2

import csv
import json
import mercantile
import supermercado

warnings.simplefilter("ignore", UserWarning)  # To prevent rasterio NotGeoreferencedWarning


def tile_pixel_to_location(tile, dx, dy):
    """Converts a pixel in a tile to lon/lat coordinates."""

    assert 0 <= dx <= 1 and 0 <= dy <= 1, "x and y offsets must be in [0, 1]"

    w, s, e, n = mercantile.bounds(tile)

    def lerp(a, b, c):
        return a + c * (b - a)

    return lerp(w, e, dx), lerp(s, n, dy)  # lon, lat


def tiles_from_csv(path):
    """Retrieve tiles from a line-delimited csv file."""

    with open(os.path.expanduser(path)) as fp:
        reader = csv.reader(fp)

        for row in reader:
            if not row:
                continue

            try:
                assert len(row) == 3
                yield mercantile.Tile(*map(int, row))
            except:
                yield row


def tiles_from_dir(root, xyz=True, xyz_path=False):
    """Loads files from an on-disk dir."""
    root = os.path.expanduser(root)

    if xyz is True:
        paths = glob.glob(os.path.join(root, "[0-9]*/[0-9]*/[0-9]*.*"))

        for path in paths:
            tile = re.match(os.path.join(root, "(?P<z>[0-9]+)/(?P<x>[0-9]+)/(?P<y>[0-9]+).+"), path)
            if not tile:
                continue

            if xyz_path is True:
                yield mercantile.Tile(int(tile["x"]), int(tile["y"]), int(tile["z"])), path
            else:
                yield mercantile.Tile(int(tile["x"]), int(tile["y"]), int(tile["z"]))

    else:
        paths = glob.glob(root, "**/*.*", recursive=True)

        for path in paths:
            return path


def tile_from_xyz(root, x, y, z):
    """Retrieve a single tile from a slippy map dir."""

    path = glob.glob(os.path.join(os.path.expanduser(root), str(z), str(x), str(y) + ".*"))
    if not path:
        return None

    assert len(path) == 1, "ambiguous tile path"

    return mercantile.Tile(x, y, z), path[0]


def tile_bbox(tile, mercator=False):

    if isinstance(tile, mercantile.Tile):
        if mercator:
            return mercantile.xy_bounds(tile)  # EPSG:3857
        else:
            return mercantile.bounds(tile)  # EPSG:4326

    else:
        with open(rasterio_open(tile)) as r:

            if mercator:
                w, s, e, n = r.bounds
                w, s = mercantile.xy(w, s)
                e, n = mercantile.xy(e, n)
                return w, s, e, n  # EPSG:3857
            else:
                return r.bounds  # EPSG:4326

        assert False, "Unable to open tile"


def tiles_to_geojson(tiles, union=True):
    """Convert tiles to their footprint GeoJSON."""

    first = True
    geojson = '{"type":"FeatureCollection","features":['

    if union:  # smaller tiles union geometries (but losing properties)
        tiles = [str(tile.z) + "-" + str(tile.x) + "-" + str(tile.y) + "\n" for tile in tiles]
        for feature in supermercado.uniontiles.union(tiles, True):
            geojson += json.dumps(feature) if first else "," + json.dumps(feature)
            first = False
    else:  # keep each tile geometry and properties (but fat)
        for tile in tiles:
            prop = '"properties":{{"x":{},"y":{},"z":{}}}'.format(tile.x, tile.y, tile.z)
            geom = '"geometry":{}'.format(json.dumps(mercantile.feature(tile, precision=6)["geometry"]))
            geojson += '{}{{"type":"Feature",{},{}}}'.format("," if not first else "", geom, prop)
            first = False

    geojson += "]}"
    return geojson


def tile_image_from_file(path, bands=None):
    """Return a multiband image numpy array, from an image file path, or None."""
    path_expanded = os.path.expanduser(path)
    _, file_extension = os.path.splitext(path_expanded)
    try:
        if file_extension in set(".png", ".webp", ".jpeg", "jpg"):
            return np.array(Image.open(path_expanded).convert("RGB"))

        raster = rasterio_open(path_expanded)
    except:
        return None

    image = None
    for i in raster.indexes if bands is None else bands:
        data_band = raster.read(i)
        data_band = data_band.reshape(data_band.shape[0], data_band.shape[1], 1)  # H,W -> H,W,C
        image = np.concatenate((image, data_band), axis=2) if image is not None else data_band

    return image


def tile_image_to_file(root, tile, image):
    """ Write an image tile on disk. """

    root = os.path.expanduser(root)
    out_path = os.path.join(root, str(tile.z), str(tile.x)) if isinstance(tile, mercantile.Tile) else root
    os.makedirs(out_path, exist_ok=True)

    if image.shape[2] > 3:
        ext = "tiff"
    else:
        ext = "webp"

    filename = "{}.{}".format(str(tile.y), ext) if isinstance(tile, mercantile.Tile) else "{}.{}".format(tile, ext)
    return cv2.imwrite(os.path.join(out_path, filename), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def tile_label_from_file(path):
    """Return a numpy array, from a label file path, or None."""

    try:
        return np.array(Image.open(os.path.expanduser(path))).astype(int)
    except:
        return None


def tile_label_to_file(root, tile, palette, label, append=False):
    """ Write a label tile on disk. """

    root = os.path.expanduser(root)
    dir_path = os.path.join(root, str(tile.z), str(tile.x)) if isinstance(tile, mercantile.Tile) else root
    path = os.path.join(dir_path, "{}.png".format(str(tile.y)))

    if len(label.shape) == 3:  # H,W,C -> H,W
        assert label.shape[2] == 1
        label = label.reshape((label.shape[0], label.shape[1]))

    if append and os.path.isfile(path):
        previous = tile_label_from_file(path)
        assert previous is not None, "Unable to open existing label: {}".format(path)
        label = np.uint8(previous + label)
    else:
        os.makedirs(dir_path, exist_ok=True)

    try:
        out = Image.fromarray(label, mode="P")
        out.putpalette(palette)
        out.save(path, optimize=True, transparency=0)
        return True
    except:
        return False


def tile_image_from_url(requests_session, url, timeout=10):
    """Fetch a tile image using HTTP, and return it or None """

    try:
        resp = requests_session.get(url, timeout=timeout)
        resp.raise_for_status()
        image = np.fromstring(io.BytesIO(resp.content).read(), np.uint8)
        return cv2.cvtColor(cv2.imdecode(image, cv2.IMREAD_ANYCOLOR), cv2.COLOR_BGR2RGB)

    except Exception:
        return None
