# Logo Segmentation

## Prerequisites
- Python3
- inkscape pdf2svg texlive-font-utils
```
sudo apt install inkscape pdf2svg texlive-font-utils
```
- Node
- svgo
```
sudo npm install -g svgo
```
- multiprocessing-logging numpy opencv-python svgpathtools svgwrite tqdm
```
python3 -m pip install -r requirements.txt
```

## Format Conversion
Convert .eps file to .svg and .png.

### Usage
```
convert.py dirs [dirs ...] [-h] [--num-workers NUM_WORKERS] [--log LOG_FILE]

Arguments:
dirs    Directories that stores .eps files.

Optional Arguments
-h, --help  Show help message and exit.
--num-workers NUM_WORKERS   Number of processes. 0 for all available cpu cores. Default to 0.
--log LOG_FILE  Path to log file. Default to "convert.log".
```

## Segmentation
Perform segmentation on .svg and .png files.

### Usage
```
segmentation.py [-h] [--num-workers NUM_WORKERS] [--log LOG_FILE]
                       [--conf CONFIDENCE_FILE] [--no-optimize]
                       [--export-contour] [--export-mask]
                       dirs [dirs ...]

Arguments:
dirs    Directories that stores .svg&.png files.

Optional Arguments
-h, --help  show help message and exit
--num-workers NUM_WORKERS     Number of processes. 0 for all available cpu cores. Default to 0.
--log LOG_FILE    Path to log file. Default to "segmentation.log".
--conf CONFIDENCE_FILE    Path to segmentation confidence file. Default to "seg_conf.json".
--no-optimize     Dont't use svgo optimization. This will produce larger svg files but cost much less time. Default to use optimization.
--export-contour  Export contour segmentation results. Default to not export.
--export-mask     Export morphed mask for debug use. Default to not export.
```

## Examples
```
python3 convert.py dir1 dir2 ---num-workers=8 --log test_convert.log
python3 segmentation.py dir1 dir2 --num-workers=8 --log test_seg.log --conf test_conf.json
```
