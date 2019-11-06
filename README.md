<a href="https://twitter.com/RobosatPink"><img src="https://img.shields.io/badge/Follow-%40RoboSatPink-ff69b4.svg" /></a>  <a href="https://gitter.im/RoboSatPink/community"><img src="https://img.shields.io/gitter/room/robosatpink/community.svg?color=ff69b4&style=popout" /></a> <a href="https://pepy.tech/project/robosat.pink"><img src="https://pepy.tech/badge/robosat-pink/month" align="right" /></a>

<h1 align='center'>RoboSat.pink</h1>
<h2 align='center'>Computer Vision framework for GeoSpatial imagery</h2>

<p align=center>
  <a href="http://www.datapink.tools/rsp/opendata_to_opendataset/compare_side_clean/"><img src="https://pbs.twimg.com/media/DpjonykWwAANpPr.jpg" alt="RoboSat.pink buildings segmentation from Imagery" /></a>
</p>



Purposes:
---------
- DataSet Quality Analysis
- Change Detection highlighter
- Features extraction and completion


Main Features:
--------------
- Provides several command line tools, you can combine together to build your own workflow
- Follows geospatial standards to ease interoperability and data preparation 
- Build-in cutting edge Computer Vision model, Data Augmentation and Loss implementations (and allows to replace by your owns)
- Support either RGB and multibands imagery, and allows Data Fusion 
- Web-UI tools to easily display, hilight or select results (and allow to use your own templates)
- High performances
- Eeasily extensible by design




<img alt="Draw me RoboSat.pink" src="https://raw.githubusercontent.com/datapink/robosat.pink/master/docs/img/readme/draw_me_robosat_pink.png" />


 
Documentation:
--------------

### Tutorials:
- <a href="https://github.com/datapink/robosat.pink/tree/master/docs/101.md">RoboSat.pink 101</a>
- <a href="https://github.com/datapink/robosat.pink/tree/master/docs/from_opendata_to_opendataset.md">How to use plain OpenData to create a decent training OpenDataSet</a>
- <a href="https://github.com/datapink/robosat.pink/tree/master/docs/extensibility_by_design.md">How to extend RoboSat.pink features, and use it as a Framework</a>

### Config file:
- <a href="https://github.com/datapink/robosat.pink/tree/master/docs/config.md">RoboSat.pink configuration file</a>

### Tools:

- <a href="https://github.com/datapink/robosat.pink/tree/master/docs/tools.md#rsp-cover">`rsp cover`</a> Generate a tiles covering, in csv format: X,Y,Z
- <a href="https://github.com/datapink/robosat.pink/tree/master/docs/tools.md#rsp-download">`rsp download`</a> Downloads tiles from a remote server (XYZ, WMS, or TMS)
- <a href="https://github.com/datapink/robosat.pink/tree/master/docs/tools.md#rsp-extract">`rsp extract`</a> Extracts GeoJSON features from OpenStreetMap .pbf
- <a href="https://github.com/datapink/robosat.pink/tree/master/docs/tools.md#rsp-rasterize">`rsp rasterize`</a> Rasterize vector features (GeoJSON or PostGIS), to raster tiles
- <a href="https://github.com/datapink/robosat.pink/tree/master/docs/tools.md#rsp-subset">`rsp subset`</a> Filter images in a slippy map dir using a csv tiles cover
- <a href="https://github.com/datapink/robosat.pink/tree/master/docs/tools.md#rsp-tile">`rsp tile`</a> Tile raster coverage
- <a href="https://github.com/datapink/robosat.pink/tree/master/docs/tools.md#rsp-train">`rsp train`</a> Trains a model on a dataset
- <a href="https://github.com/datapink/robosat.pink/tree/master/docs/tools.md#rsp-export">`rsp export`</a> Export a model to ONNX or Torch JIT
- <a href="https://github.com/datapink/robosat.pink/tree/master/docs/tools.md#rsp-predict">`rsp predict`</a> Predict masks, from given inputs and an already trained model
- <a href="https://github.com/datapink/robosat.pink/tree/master/docs/tools.md#rsp-compare">`rsp compare`</a> Compute composite images and/or metrics to compare several XYZ dirs
- <a href="https://github.com/datapink/robosat.pink/tree/master/docs/tools.md#rsp-vectorize">`rsp vectorize`</a> Extract simplified GeoJSON features from segmentation masks
- <a href="https://github.com/datapink/robosat.pink/tree/master/docs/tools.md#rsp-info">`rsp info`</a> Print RoboSat.pink version informations

### Presentations slides:
  - <a href="http://www.datapink.com/presentations/2019-sotm.pdf">@SOTM 2019</a>
  - <a href="http://www.datapink.com/presentations/2019-foss4g-cv.pdf">@FOSS4G 2019</a>






Installs:
--------

### With PIP:
```
pip3 install RoboSat.pink                                     # For latest stable version
```

or

```
pip3 install git+https://github.com/datapink/robosat.pink     # For current dev version
```

### With Ubuntu 19.04, from scratch:

```
sudo sh -c "apt update && apt install -y build-essential python3-pip"
pip3 install RoboSat.pink && export PATH=$PATH:~/.local/bin
wget http://us.download.nvidia.com/XFree86/Linux-x86_64/430.40/NVIDIA-Linux-x86_64-430.40.run
sudo sh NVIDIA-Linux-x86_64-430.40.run -a -q --ui=none
```

### With CentOS 7, from scratch:
```
sudo sh -c "yum -y update && yum install -y python36 wget && python3.6 -m ensurepip"
pip3 install --user RoboSat.pink
sudo sh -c "yum groupinstall -y 'Development Tools' && yum install -y kernel-devel epel-release"
wget http://us.download.nvidia.com/XFree86/Linux-x86_64/430.40/NVIDIA-Linux-x86_64-430.40.run
sudo sh NVIDIA-Linux-x86_64-430.40.run -a -q --ui=none
```


### NOTAS: 
- Requires: Python 3.6 or 3.7
- GPU is not strictly mandatory, but `rsp train` and `rsp predict` would be -that- slower without.
- To test RoboSat.pink install, launch in a terminal: `rsp -h` or `rsp info`
- Upon your ```pip``` PATH setting, you may have to update it: ```export PATH=$PATH:.local/bin```
- If needed, to remove pre-existing Nouveau driver: ```sudo sh -c "echo blacklist nouveau > /etc/modprobe.d/blacklist-nvidia-nouveau.conf && update-initramfs -u && reboot"```




Architecture:
------------

RoboSat.pink use cherry-picked Open Source libs among Deep Learning, Computer Vision and GIS stacks.

<img alt="Stacks" src="https://raw.githubusercontent.com/datapink/robosat.pink/master/docs/img/readme/stacks.png" />



GeoSpatial OpenDataSets:
------------------------
- <a href="https://github.com/chrieke/awesome-satellite-imagery-datasets">Christoph Rieke's Awesome Satellite Imagery Datasets</a>
- <a href="https://zhangbin0917.github.io/2018/06/12/%E9%81%A5%E6%84%9F%E6%95%B0%E6%8D%AE%E9%9B%86/">Zhang Bin, Earth Observation OpenDataset blog</a> 
- <a href="https://github.com/wenhwu/awesome-remote-sensing-change-detection">Wehwu's Awesome Remote Sensing Change Detection</a>

Bibliography:
-------------

- <a href="https://arxiv.org/abs/1505.04597">U-Net: Convolutional Networks for Biomedical Image Segmentation</a>
- <a href="https://arxiv.org/abs/1512.03385">Deep Residual Learning for Image Recognition</a>
- <a href="https://arxiv.org/pdf/1804.08024.pdf">Angiodysplasia Detection and Localization Using Deep
Convolutional Neural Networks</a>
- <a href="https://arxiv.org/abs/1806.00844">TernausNetV2: Fully Convolutional Network for Instance Segmentation</a>
- <a href="https://arxiv.org/abs/1705.08790">The Lovász-Softmax loss: A tractable surrogate for the optimization of the IoU measure in neural networks</a>
- <a href="https://arxiv.org/abs/1809.06839">Albumentations: fast and flexible image augmentations</a>










Contributions and Services:
---------------------------

- Pull Requests are welcome ! Feel free to send code...
  Don't hesitate either to initiate a prior discussion via <a href="https://gitter.im/RoboSatPink/community">gitter</a> or ticket on any implementation question.
  And give also a look at <a href="https://github.com/datapink/robosat.pink/blob/master/docs/makefile.md">Makefile rules</a>.

- If you want to collaborate through code production and maintenance on a long term basis, please get in touch, co-edition with an ad hoc governance can be considered.

- If you want a new feature, but don't want to implement it, <a href="http://datapink.com">DataPink</a> provide core-dev services.

- Expertise and training on RoboSat.pink are also provided by <a href="http://datapink.com">DataPink</a>.

- And if you want to support the whole project, because it means for your own business, funding is also welcome.


### Requests for funding:

We've already identified several new features and research papers able to improve again RoboSat.pink,
your funding would make a difference to implement them on a coming release:

- Increase (again) prediction accuracy :
  - on low resolution imagery
  - even with few labels
  - feature extraction when they are (really) close
  - with multibands and Data Fusion

- Add support for :
  - MultiClass classification
  - Linear features extraction
  - PointCloud data support
  - Time Series Imagery
  - StreetView Imagery
  
- Improve (again) performances




Authors:
--------
- Olivier Courtin <https://github.com/ocourtin>
- Daniel J. Hofmann <https://github.com/daniel-j-h>



Citing:
-------
```
  @Manual{,
    title = {{RoboSat.pink} Computer Vision framework for GeoSpatial Imagery},
    author = {Olivier Courtin, Daniel J. Hofmann},
    organization = {DataPink},
    year = {2019},
    url = {http://RoboSat.pink},
  }
```
