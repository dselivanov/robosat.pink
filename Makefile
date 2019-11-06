help:
	@echo "This Makefile rules are designed for RoboSat.pink devs and power-users."
	@echo "For plain user installation follow README.md instructions, instead."
	@echo ""
	@echo ""
	@echo " make install     To install, few Python dev tools and RoboSat.pink in editable mode."
	@echo "                  So any further RoboSat.pink Python code modification will be usable at once,"
	@echo "                  throught either rsp tools commands or robosat_pink.* modules."
	@echo ""
	@echo " make check       Launchs code tests, and tools doc updating."
	@echo "                  Do it, at least, before sending a Pull Request."
	@echo ""
	@echo " make check_tuto  Launchs rsp commands embeded in tutorials, to be sure everything still up to date."
	@echo "                  Do it, at least, on each CLI modifications, and before a release."
	@echo "                  NOTA: It takes a while."
	@echo ""
	@echo " make pink        Python code beautifier,"
	@echo "                  as Pink is the new Black ^^"



# Dev install
install:
	pip3 install pytest black flake8 twine
	pip3 install -e .


# Lauch all tests
check: ut it doc
	@echo "==================================================================================="
	@echo "All tests passed !"
	@echo "==================================================================================="


# Python code beautifier
pink:
	black -l 125 *.py robosat_pink/*.py robosat_pink/*/*.py tests/*py tests/*/*.py


# Perform units tests, and linter checks
ut:
	@echo "==================================================================================="
	black -l 125 --check *.py robosat_pink/*.py robosat_pink/*/*.py
	@echo "==================================================================================="
	flake8 --max-line-length 125 --ignore=E203,E241,E226,E272,E261,E221,W503,E722
	@echo "==================================================================================="
	pytest tests -W ignore::UserWarning


# Launch Integration Tests
it: it_pre it_train it_post


# Integration Tests: Data Preparation
it_pre:
	@echo "==================================================================================="
	rm -rf it
	rsp cover --zoom 18 --bbox 4.8,45.7,4.82,45.72  it/cover
	rsp download --rate 20 --type WMS 'https://download.data.grandlyon.com/wms/grandlyon?SERVICE=WMS&REQUEST=GetMap&VERSION=1.3.0&LAYERS=Ortho2015_vue_ensemble_16cm_CC46&WIDTH=512&HEIGHT=512&CRS=EPSG:3857&BBOX={xmin},{ymin},{xmax},{ymax}&FORMAT=image/jpeg' it/cover it/images
	echo "Download GeoJSON" && wget --show-progress -q -nc -O it/lyon_roofprint.json 'https://download.data.grandlyon.com/wfs/grandlyon?SERVICE=WFS&REQUEST=GetFeature&TYPENAME=ms:fpc_fond_plan_communaut.fpctoit&VERSION=1.1.0&srsName=EPSG:4326&BBOX=4.8,45.7,4.82,45.72&outputFormat=application/json; subtype=geojson' | true
	rsp rasterize --type Building --geojson it/lyon_roofprint.json --config config.toml --cover it/cover it/labels
	echo "Download PBF" && wget --show-progress -q -O it/lyon.pbf http://datapink.tools/rsp/it/lyon.pbf
	rsp extract --type Building it/lyon.pbf it/osm_lyon_footprint.json
	rsp rasterize --type Building --geojson it/lyon_roofprint.json --config config.toml --cover it/cover it/labels_osm
	rsp cover --dir it/images --splits 80/20 it/training/cover it/validation/cover
	rsp subset --dir it/images --cover it/training/cover it/training/images
	rsp subset --dir it/labels --cover it/training/cover it/training/labels
	rsp subset --dir it/images --cover it/validation/cover it/validation/images
	rsp subset --dir it/labels --cover it/validation/cover it/validation/labels
	wget -nc -O it/tanzania.tif http://datapink.tools/rsp/it/tanzania.tif
	rsp tile --zoom 19 it/tanzania.tif it/prediction/images
	rsp cover --zoom 19 --dir it/prediction/images it/prediction/cover
	wget -nc -O it/tanzania.geojson http://datapink.tools/rsp/it/tanzania.geojson
	rsp rasterize --type Building --geojson it/tanzania.geojson --config config.toml --cover it/prediction/cover it/prediction/labels



# Integration Tests: Training
it_train:
	@echo "==================================================================================="
	rsp train --config config.toml --workers 2 --bs 2 --lr 0.00025 --epochs 2 it it/pth
	rsp train --config config.toml --workers 2 --bs 2 --lr 0.00025 --epochs 3 --resume --checkpoint it/pth/checkpoint-00002.pth it it/pth


# Integration Tests: Post Training
it_post:
	@echo "==================================================================================="
	rsp export --checkpoint it/pth/checkpoint-00003.pth --type jit it/pth/export.jit
	rsp export --checkpoint it/pth/checkpoint-00003.pth --type onnx it/pth/export.onnx
	rsp predict --config config.toml --bs 4 --checkpoint it/pth/checkpoint-00003.pth it/prediction it/prediction/masks
	rsp predict --config config.toml --bs 4 --translate --checkpoint it/pth/checkpoint-00003.pth it/prediction it/prediction/masks
	rsp compare --images it/prediction/images it/prediction/labels it/prediction/masks --mode stack --labels it/prediction/labels --masks it/prediction/masks it/prediction/compare
	rsp compare --images it/prediction/images it/prediction/compare --mode side it/prediction/compare_side
	rsp compare --mode list --labels it/prediction/labels --maximum_qod 75 --minimum_fg 5 --masks it/prediction/masks --geojson it/prediction/compare/tiles.json
	cp it/prediction/compare/tiles.json it/prediction/compare_side/tiles.json
	rsp vectorize --type Building --config config.toml it/prediction/masks it/prediction/vector.json


# Documentation generation (tools and config file)
doc:
	@echo "==================================================================================="
	@echo "# RoboSat.pink tools documentation" > docs/tools.md
	@for tool in `ls robosat_pink/tools/[^_]*py | sed -e 's#.*/##g' -e 's#.py##'`; do \
		echo "Doc generation: $$tool"; 						  \
		echo "## rsp $$tool" >> docs/tools.md; 				  	  \
		echo '```'           >> docs/tools.md; 				  	  \
		rsp $$tool -h        >> docs/tools.md; 				  	  \
		echo '```'           >> docs/tools.md; 				  	  \
	done
	@echo "Doc generation: config.toml"
	@echo "## config.toml"        > docs/config.md; 			  	  \
	echo '```'                    >> docs/config.md; 			  	  \
	cat config.toml               >> docs/config.md; 			  	  \
	echo '```'                    >> docs/config.md;
	@echo "Doc generation: Makefile"
	@echo "## Makefile"           > docs/makefile.md; 			  	  \
	echo '```'                    >> docs/makefile.md; 			  	  \
	make --no-print-directory     >> docs/makefile.md; 			  	  \
	echo '```'                    >> docs/makefile.md;


# Check rsp commands embeded in Documentation
check_doc:
	@echo "==================================================================================="
	@echo "Checking README:"
	@echo "==================================================================================="
	@rm -rf ds && sed -n -e '/```bash/,/```/ p' README.md | sed -e '/```/d' > .CHECK && sh .CHECK
	@echo "==================================================================================="


# Check rsp commands embeded in Tutorials
check_tuto:
	@mkdir tuto 
	@echo "==================================================================================="
	@echo "Checking 101"
	@sudo su postgres -c 'dropdb tanzania' || :
	@cd tuto && mkdir 101 && sed -n -e '/```bash/,/```/ p' ../docs/101.md | sed -e '/```/d' > 101/.CHECK && cd 101 && sh .CHECK && cd ..
	@echo "==================================================================================="
	@echo "Checking Tutorial OpenData to OpenDataset:"
	@cd tuto && mkdir gl && sed -n -e '/```bash/,/```/ p' ../docs/from_opendata_to_opendataset.md | sed -e '/```/d' > gl/.CHECK && cd gl && sh .CHECK && cd ..
	@echo "==================================================================================="


# Send a release on PyPI
pypi:
	rm -rf dist RoboSat.pink.egg-info
	python3 setup.py sdist
	twine upload dist/* -r pypi
