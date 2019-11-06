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


def tiles_from_dir(root, cover=None, xyz=True, xyz_path=False, xyz_translate=False):
    """Loads files from an on-disk dir."""
    root = os.path.expanduser(root)

    if xyz is True:
        paths = glob.glob(os.path.join(root, "[0-9]*/[0-9]*/[0-9]*.*"))

        for path in paths:
            tile_xyz = re.match(os.path.join(root, "(?P<z>[0-9]+)/(?P<x>[0-9]+)/(?P<y>[0-9]+).+"), path)
            if not tile_xyz:
                continue
            tile = mercantile.Tile(int(tile_xyz["x"]), int(tile_xyz["y"]), int(tile_xyz["z"]))

            if cover is not None and tile not in cover:
                continue

            if xyz_translate is True and tile_translate(root, tile, cover) is None:
                continue

            if xyz_path is True:
                yield tile, path
            else:
                yield tile

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
    """Return a multiband image numpy array, from an image file path"""
    path_expanded = os.path.expanduser(path)
    _, file_extension = os.path.splitext(path_expanded)
    try:
        if file_extension in set([".png", ".webp", ".jpeg", ".jpg"]):
            return np.array(Image.open(path_expanded).convert("RGB"))

        raster = rasterio_open(path_expanded)
        assert raster, "Unable to open {}".format(path)

    except:
        return None

    image = None
    for i in raster.indexes if bands is None else bands:
        data_band = raster.read(i)
        data_band = data_band.reshape(data_band.shape[0], data_band.shape[1], 1)  # H,W -> H,W,C
        image = np.concatenate((image, data_band), axis=2) if image is not None else data_band

    assert image is not None, "Unable to open {}".format(path)
    return image


def tile_translate(root, tile, cover):

    try:
        ul_tile, ul_path = tile_from_xyz(root, tile.x - 1, tile.y - 1, tile.z)
        ur_tile, ur_path = tile_from_xyz(root, tile.x - 0, tile.y - 1, tile.z)
        ll_tile, ll_path = tile_from_xyz(root, tile.x - 1, tile.y - 0, tile.z)
        lr_tile, lr_path = tile_from_xyz(root, tile.x - 0, tile.y - 0, tile.z)
    except:
        return None

    if cover is not None:
        if ul_tile not in cover or ur_tile not in cover or lr_tile not in cover or ll_tile not in cover:
            return None

    return {"ul": (ul_tile, ul_path), "ur": (ur_tile, ur_path), "ll": (ll_tile, ll_path), "lr": (lr_tile, lr_path)}


def tile_translate_from_file(root, tile, cover, bands=None):

    tiles = tile_translate(root, tile, cover)
    assert tiles, "Translate tiles mismatch"

    ul = tile_image_from_file(tiles["ul"][1], bands)
    ur = tile_image_from_file(tiles["ur"][1], bands)
    lr = tile_image_from_file(tiles["lr"][1], bands)
    ll = tile_image_from_file(tiles["ll"][1], bands)

    assert ul.shape == ur.shape == ll.shape == lr.shape, "Inconsistency in tiles shape while translating {}".format(tile)

    H, W, C = ul.shape
    assert H % 2 == W % 2 == 0, "tile width and height must be an even number"

    image = np.zeros((H, W, C), dtype="float32")
    image[0 : int(H / 2), 0 : int(W / 2), :] = ul[int(H / 2) : H, int(W / 2) : W, :]
    image[0 : int(H / 2), int(W / 2) : W, :] = ur[int(H / 2) : H, 0 : int(W / 2), :]
    image[int(H / 2) : H, int(W / 2) : W, :] = lr[0 : int(H / 2), 0 : int(W / 2), :]
    image[int(H / 2) : H, 0 : int(W / 2), :] = ll[0 : int(H / 2), int(W / 2) : W, :]

    return image


def tile_translate_to_file(root, tile, palette, label, margin=16):

    assert label.shape[0] % 2 == label.shape[1] % 2 == 0, "tile width and height must be an even number"

    if len(label.shape) == 3:  # H,W,C -> H,W
        assert label.shape[2] == 1
        label = label.reshape((label.shape[0], label.shape[1]))

    H, W = label.shape

    tiles = tile_translate(root, tile, None)
    assert tiles, "Translate tiles mismatch"

    translate = np.zeros((H, W), dtype="int8")
    translate[int(H / 2) + margin : H, int(W / 2) + margin : W] = label[margin : int(H / 2), margin : int(W / 2)]
    tile_label_to_file(root, tiles["ul"][0], palette, translate, translate="ul", margin=margin)

    translate = np.zeros((H, W), dtype="int8")
    translate[int(H / 2) + margin : H, 0 : int(W / 2) - margin] = label[margin : int(H / 2), int(W / 2) : W - margin]
    tile_label_to_file(root, tiles["ur"][0], palette, translate, translate="ur", margin=margin)

    translate = np.zeros((H, W), dtype="int8")
    translate[0 : int(H / 2) - margin, 0 : int(W / 2) - margin] = label[int(H / 2) : H - margin, int(W / 2) : W - margin]
    tile_label_to_file(root, tiles["lr"][0], palette, translate, translate="lr", margin=margin)

    translate = np.zeros((H, W), dtype="int8")
    translate[0 : int(H / 2) - margin, int(W / 2) + margin : W] = label[int(H / 2) : H - margin, margin : int(W / 2)]
    tile_label_to_file(root, tiles["ll"][0], palette, translate, translate="ll", margin=margin)


def tile_image_to_file(root, tile, image):
    """ Write an image tile on disk. """

    root = os.path.expanduser(root)
    path = os.path.join(root, str(tile.z), str(tile.x)) if isinstance(tile, mercantile.Tile) else root
    os.makedirs(path, exist_ok=True)

    ext = "tiff" if image.shape[2] > 3 else "webp"
    filename = "{}.{}".format(str(tile.y), ext) if isinstance(tile, mercantile.Tile) else "{}.{}".format(tile, ext)
    path = os.path.join(path, filename)

    try:
        cv2.imwrite(path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    except:
        assert False, "Unable to write {}".format(path)


def tile_label_from_file(path, silent=True):
    """Return a numpy array, from a label file path, or None."""

    try:
        return np.array(Image.open(os.path.expanduser(path))).astype(int)
    except:
        assert silent, "Unable to open existing label: {}".format(path)


def tile_label_to_file(root, tile, palette, label, append=False, translate=False, margin=16):
    """ Write a label (or a mask) tile on disk. """

    root = os.path.expanduser(root)
    dir_path = os.path.join(root, str(tile.z), str(tile.x)) if isinstance(tile, mercantile.Tile) else root
    path = os.path.join(dir_path, "{}.png".format(str(tile.y)))
    assert not (append and translate), "tile_label_to_file can be either in append OR translate mode"

    if len(label.shape) == 3:  # H,W,C -> H,W
        assert label.shape[2] == 1
        label = label.reshape((label.shape[0], label.shape[1]))

    if append and os.path.isfile(path):
        previous = tile_label_from_file(path, silent=False)
        label = np.uint8(previous + label)
    elif translate:
        previous = tile_label_from_file(path, silent=False)

        H, W = label.shape

        assert translate in ["lr", "ll", "ur", "ul"], "Unknown translation"

        if translate == "lr":
            previous[:margin, : int(W / 2) - margin] = 0
            previous[: int(H / 2) - margin, :margin] = 0
        if translate == "ur":
            previous[-margin:, : int(W / 2) - margin] = 0
            previous[int(H / 2) + margin :, :margin] = 0
        if translate == "ll":
            previous[:margin, int(W / 2) + margin :] = 0
            previous[: int(H / 2) - margin, -margin:] = 0
        if translate == "ul":
            previous[-margin:, int(W / 2) + margin :] = 0
            previous[int(H / 2) + margin :, -margin:] = 0

        label = np.uint8(np.logical_or(previous, label))
    else:
        os.makedirs(dir_path, exist_ok=True)

    try:
        out = Image.fromarray(label, mode="P")
        out.putpalette(palette)
        out.save(path, optimize=True, transparency=0)
    except:
        assert False, "Unable to write {}".format(path)


def tile_image_from_url(requests_session, url, timeout=10):
    """Fetch a tile image using HTTP, and return it or None """

    try:
        resp = requests_session.get(url, timeout=timeout)
        resp.raise_for_status()
        image = np.fromstring(io.BytesIO(resp.content).read(), np.uint8)
        return cv2.cvtColor(cv2.imdecode(image, cv2.IMREAD_ANYCOLOR), cv2.COLOR_BGR2RGB)

    except Exception:
        return None
