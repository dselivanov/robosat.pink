# RoboSat.pink tools documentation
## rsp compare
```
usage: rsp compare [-h] [--mode {side,stack,list}] [--labels LABELS]
                   [--masks MASKS] [--images IMAGES [IMAGES ...]]
                   [--workers WORKERS] [--minimum_fg MINIMUM_FG]
                   [--maximum_fg MAXIMUM_FG] [--minimum_qod MINIMUM_QOD]
                   [--maximum_qod MAXIMUM_QOD] [--vertical] [--geojson]
                   [--format FORMAT] [--web_ui_base_url WEB_UI_BASE_URL]
                   [--web_ui_template WEB_UI_TEMPLATE] [--no_web_ui]
                   out

optional arguments:
 -h, --help                         show this help message and exit

Inputs:
 --mode {side,stack,list}           compare mode [default: side]
 --labels LABELS                    path to tiles labels directory [required for QoD filtering]
 --masks MASKS                      path to tiles masks directory [required for QoD filtering)
 --images IMAGES [IMAGES ...]       path to images directories [required for stack or side modes]
 --workers WORKERS                  number of workers [default: CPU]

QoD Filtering:
 --minimum_fg MINIMUM_FG            skip tile if label foreground below. [default: 0]
 --maximum_fg MAXIMUM_FG            skip tile if label foreground above [default: 100]
 --minimum_qod MINIMUM_QOD          skip tile if QoD metric below [default: 0]
 --maximum_qod MAXIMUM_QOD          skip tile if QoD metric above [default: 100]

Outputs:
 --vertical                         output vertical image aggregate [optionnal for side mode]
 --geojson                          output results as GeoJSON [optionnal for list mode]
 --format FORMAT                    output images file format [default: webp]
 out                                output path

Web UI:
 --web_ui_base_url WEB_UI_BASE_URL  alternate Web UI base URL
 --web_ui_template WEB_UI_TEMPLATE  alternate Web UI template path
 --no_web_ui                        desactivate Web UI output
```
## rsp cover
```
usage: rsp cover [-h] [--dir DIR] [--bbox BBOX] [--geojson GEOJSON]
                 [--cover COVER] [--raster RASTER] [--sql SQL] [--pg PG]
                 [--no_xyz] [--zoom ZOOM] [--extent] [--splits SPLITS]
                 [out [out ...]]

optional arguments:
 -h, --help         show this help message and exit

Input [one among the following is required]:
 --dir DIR          plain tiles dir path
 --bbox BBOX        a lat/lon bbox: xmin,ymin,xmax,ymax or a bbox: xmin,xmin,xmax,xmax,EPSG:xxxx
 --geojson GEOJSON  a geojson file path
 --cover COVER      a cover file path
 --raster RASTER    a raster file path
 --sql SQL          SQL to retrieve geometry features [e.g SELECT geom FROM a_table]

Spatial DataBase [required with --sql input]:
 --pg PG            PostgreSQL dsn using psycopg2 syntax (e.g 'dbname=db user=postgres')

Tiles:
 --no_xyz           if set, tiles are not expected to be XYZ based.

Outputs:
 --zoom ZOOM        zoom level of tiles [required with --geojson or --bbox]
 --extent           if set, rather than a cover, output a bbox extent
 --splits SPLITS    if set, shuffle and split in several cover subpieces. [e.g 50/15/35]
 out                cover csv output paths [required except for extent]
```
## rsp download
```
usage: rsp download [-h] [--type {XYZ,WMS}] [--rate RATE] [--timeout TIMEOUT]
                    [--workers WORKERS] [--format FORMAT]
                    [--web_ui_base_url WEB_UI_BASE_URL]
                    [--web_ui_template WEB_UI_TEMPLATE] [--no_web_ui]
                    url cover out

optional arguments:
 -h, --help                         show this help message and exit

Web Server:
 url                                URL server endpoint, with: {z}/{x}/{y} or {xmin},{ymin},{xmax},{ymax} [required]
 --type {XYZ,WMS}                   service type [default: XYZ]
 --rate RATE                        download rate limit in max requests/seconds [default: 10]
 --timeout TIMEOUT                  download request timeout (in seconds) [default: 10]
 --workers WORKERS                  number of workers [default: CPU / 2]

Coverage to download:
 cover                              path to .csv tiles list [required]

Output:
 --format FORMAT                    file format to save images in [default: webp]
 out                                output directory path [required]

Web UI:
 --web_ui_base_url WEB_UI_BASE_URL  alternate Web UI base URL
 --web_ui_template WEB_UI_TEMPLATE  alternate Web UI template path
 --no_web_ui                        desactivate Web UI output
```
## rsp export
```
usage: rsp export [-h] --checkpoint CHECKPOINT [--type {onnx,jit,pth}]
                  [--nn NN] [--loader LOADER] [--doc_string DOC_STRING]
                  [--shape_in SHAPE_IN] [--shape_out SHAPE_OUT]
                  out

optional arguments:
 -h, --help               show this help message and exit

Inputs:
 --checkpoint CHECKPOINT  model checkpoint to load [required]
 --type {onnx,jit,pth}    output type [default: onnx]

To set or override metadata pth parameters::
 --nn NN                  nn name
 --loader LOADER          nn loader
 --doc_string DOC_STRING  nn documentation abstract
 --shape_in SHAPE_IN      nn shape in (e.g 3,512,512)
 --shape_out SHAPE_OUT    nn shape_out  (e.g 2,512,512)

Output:
 out                      path to save export model to [required]
```
## rsp extract
```
usage: rsp extract [-h] --type TYPE pbf out

optional arguments:
 -h, --help   show this help message and exit

Inputs:
 --type TYPE  type of feature to extract (e.g Building, Road) [required]
 pbf          path to .osm.pbf file [required]

Output:
 out          GeoJSON output file path [required]
```
## rsp info
```
usage: rsp info [-h] [--processes]

optional arguments:
 -h, --help   show this help message and exit
 --processes  if set, output GPU processes list

Usages:
To kill GPU processes: rsp info --processes | xargs sudo kill -9
```
## rsp predict
```
usage: rsp predict [-h] --checkpoint CHECKPOINT [--config CONFIG]
                   [--cover COVER] [--translate] [--workers WORKERS] [--bs BS]
                   [--web_ui_base_url WEB_UI_BASE_URL]
                   [--web_ui_template WEB_UI_TEMPLATE] [--no_web_ui]
                   dataset out

optional arguments:
 -h, --help                         show this help message and exit

Inputs:
 dataset                            predict dataset directory path [required]
 --checkpoint CHECKPOINT            path to the trained model to use [required]
 --config CONFIG                    path to config file [required]
 --cover COVER                      path to csv tiles cover file, to filter tiles to predict [optional]

Outputs:
 out                                output directory path [required]

Data Loaders:
 --translate                        translate tiles coverage to avoid borders effect
 --workers WORKERS                  number of workers to load images [default: GPU x 2]
 --bs BS                            batch size value for data loader [default: 4]

Web UI:
 --web_ui_base_url WEB_UI_BASE_URL  alternate Web UI base URL
 --web_ui_template WEB_UI_TEMPLATE  alternate Web UI template path
 --no_web_ui                        desactivate Web UI output
```
## rsp rasterize
```
usage: rsp rasterize [-h] [--cover COVER] [--config CONFIG] --type TYPE
                     [--pg PG] [--sql SQL] [--geojson GEOJSON [GEOJSON ...]]
                     [--append] [--ts TS] [--web_ui_base_url WEB_UI_BASE_URL]
                     [--web_ui_template WEB_UI_TEMPLATE] [--no_web_ui]
                     out

optional arguments:
 -h, --help                         show this help message and exit

Inputs [either --postgis or --geojson is required]:
 --cover COVER                      path to csv tiles cover file [required]
 --config CONFIG                    path to config file [required]
 --type TYPE                        type of feature to rasterize (e.g Building, Road) [required]
 --pg PG                            PostgreSQL dsn using psycopg2 syntax (e.g 'dbname=db user=postgres')
 --sql SQL                          SQL to retrieve geometry features [e.g SELECT geom FROM a_table WHERE ST_Intersects(TILE_GEOM, geom)]
 --geojson GEOJSON [GEOJSON ...]    path to GeoJSON features files

Outputs:
 out                                output directory path [required]
 --append                           Append to existing tile if any, useful to multiclass labels
 --ts TS                            output tile size [default: 512,512]

Web UI:
 --web_ui_base_url WEB_UI_BASE_URL  alternate Web UI base URL
 --web_ui_template WEB_UI_TEMPLATE  alternate Web UI template path
 --no_web_ui                        desactivate Web UI output
```
## rsp subset
```
usage: rsp subset [-h] --dir DIR --cover COVER [--copy] [--delete]
                  [--web_ui_base_url WEB_UI_BASE_URL]
                  [--web_ui_template WEB_UI_TEMPLATE] [--no_web_ui]
                  [out]

optional arguments:
 -h, --help                         show this help message and exit

Inputs:
 --dir DIR                          to XYZ tiles input dir path [required]
 --cover COVER                      path to csv cover file to filter dir by [required]

Alternate modes, as default is to create relative symlinks.:
 --copy                             copy tiles from input to output
 --delete                           delete tiles listed in cover

Output:
 out                                output dir path [required for copy or move]

Web UI:
 --web_ui_base_url WEB_UI_BASE_URL  alternate Web UI base URL
 --web_ui_template WEB_UI_TEMPLATE  alternate Web UI template path
 --no_web_ui                        desactivate Web UI output
```
## rsp tile
```
usage: rsp tile [-h] [--cover COVER] --zoom ZOOM [--ts TS] [--nodata [0-255]]
                [--nodata_threshold [0-100]] [--keep_borders] [--label]
                [--config CONFIG] [--workers WORKERS]
                [--web_ui_base_url WEB_UI_BASE_URL]
                [--web_ui_template WEB_UI_TEMPLATE] [--no_web_ui]
                rasters [rasters ...] out

optional arguments:
 -h, --help                         show this help message and exit

Inputs:
 rasters                            path to raster files to tile [required]
 --cover COVER                      path to csv tiles cover file, to filter tiles to tile [optional]

Output:
 --zoom ZOOM                        zoom level of tiles [required]
 --ts TS                            tile size in pixels [default: 512,512]
 --nodata [0-255]                   nodata pixel value, used by default to remove coverage border's tile [default: 0]
 --nodata_threshold [0-100]         Skip tile if nodata pixel ratio > threshold. [default: 100]
 --keep_borders                     keep tiles even if borders are empty (nodata)
 out                                output directory path [required]

Labels:
 --label                            if set, generate label tiles
 --config CONFIG                    path to config file [required in label mode]

Performances:
 --workers WORKERS                  number of workers [default: raster files]

Web UI:
 --web_ui_base_url WEB_UI_BASE_URL  alternate Web UI base URL
 --web_ui_template WEB_UI_TEMPLATE  alternate Web UI template path
 --no_web_ui                        desactivate Web UI output
```
## rsp train
```
usage: rsp train [-h] [--config CONFIG] [--loader LOADER] [--workers WORKERS]
                 [--bs BS] [--lr LR] [--ts TS] [--nn NN] [--loss LOSS]
                 [--da DA] [--dap DAP] [--epochs EPOCHS] [--resume]
                 [--checkpoint CHECKPOINT] [--no_validation] [--no_training]
                 [--saving SAVING]
                 dataset out

optional arguments:
 -h, --help               show this help message and exit
 --config CONFIG          path to config file [required]

Dataset:
 dataset                  training dataset path
 --loader LOADER          dataset loader name [if set override config file value]
 --workers WORKERS        number of pre-processing images workers [default: batch size]

Hyper Parameters [if set override config file value]:
 --bs BS                  batch size
 --lr LR                  learning rate
 --ts TS                  tile size
 --nn NN                  neurals network name
 --loss LOSS              model loss
 --da DA                  kind of data augmentation
 --dap DAP                data augmentation probability [default: 1.0]

Model Training:
 --epochs EPOCHS          number of epochs to train [default: 10]
 --resume                 resume model training, if set imply to provide a checkpoint
 --checkpoint CHECKPOINT  path to a model checkpoint. To fine tune or resume a training
 --no_validation          No validation, training only
 --no_training            No training, validation only

Output:
 --saving SAVING          number of epochs beetwen checkpoint .pth saving [default: 1]
 out                      output directory path to save checkpoint .pth files and logs [required]
```
## rsp vectorize
```
usage: rsp vectorize [-h] --type TYPE [--config CONFIG] masks out

optional arguments:
 -h, --help       show this help message and exit

Inputs:
 masks            input masks directory path [required]
 --type TYPE      type of features to extract (i.e class title) [required]
 --config CONFIG  path to config file [required]

Outputs:
 out              path to output file to store features in [required]
```
