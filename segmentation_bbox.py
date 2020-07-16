import os
import os.path as osp
import re
import cv2
import sys
import numpy as np
import subprocess
import shutil
from shutil import rmtree
from svgpathtools import wsvg, parse_path
from xml.dom.minidom import parse, parseString
import glob
import math
import argparse
from utils import *

def find_paths(doc,
               convert_circles_to_paths=True,
               convert_ellipses_to_paths=True,
               convert_lines_to_paths=True,
               convert_polylines_to_paths=True,
               convert_polygons_to_paths=True,
               convert_rectangles_to_paths=True):

    # Use minidom to extract path strings from input SVG
    pathnodes = doc.getElementsByTagName('path')
    paths = [dom2dict(el) for el in pathnodes]
    d_strings = [el['d'] for el in paths]

    # Use minidom to extract polyline strings from input SVG, convert to
    # path strings, add to list
    if convert_polylines_to_paths:
        plinnodes = doc.getElementsByTagName('polyline')
        plins = [dom2dict(el) for el in plinnodes]
        d_strings += [polyline2pathd(pl['points']) for pl in plins]
        pathnodes.extend(plinnodes)

    # Use minidom to extract polygon strings from input SVG, convert to
    # path strings, add to list
    if convert_polygons_to_paths:
        pgonnodes = doc.getElementsByTagName('polygon')
        pgons = [dom2dict(el) for el in pgonnodes]
        d_strings += [polygon2pathd(pg['points']) for pg in pgons]
        pathnodes.extend(pgonnodes)

    if convert_lines_to_paths:
        linenodes = doc.getElementsByTagName('line')
        lines = [dom2dict(el) for el in linenodes]
        d_strings += [('M' + l['x1'] + ' ' + l['y1'] +
                       'L' + l['x2'] + ' ' + l['y2']) for l in lines]
        pathnodes.extend(linenodes)

    if convert_ellipses_to_paths:
        ellipsenodes = doc.getElementsByTagName('ellipse')
        ellipses = [dom2dict(el) for el in ellipsenodes]
        d_strings += [ellipse2pathd(e) for e in ellipses]
        pathnodes.extend(ellipsenodes)

    if convert_circles_to_paths:
        circlenodes = doc.getElementsByTagName('circle')
        circles = [dom2dict(el) for el in circlenodes]
        d_strings += [ellipse2pathd(c) for c in circles]
        pathnodes.extend(circlenodes)

    if convert_rectangles_to_paths:
        rectanglenodes = doc.getElementsByTagName('rect')
        rectangles = [dom2dict(el) for el in rectanglenodes]
        d_strings += [rect2pathd(r) for r in rectangles]
        pathnodes.extend(rectanglenodes)
    path_list = []
    for d in d_strings:
        try:
            path_list.append(parse_path(d))
        except:
            print('Parse d string {}... fail!'.format(d[:100]))
            continue
    return pathnodes, path_list, d_strings


def find_parent(element):
    while element.parentNode and element.parentNode.getAttribute('id') != 'surface1':
        element = element.parentNode
    return element


def bb_intersection_over_union(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
 
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
 
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
 
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)
 
	# return the intersection over union value
	return interArea, iou


def inter_area_with_bbox(bbox0, bbox):
    x, y, w, h = bbox0
    x0, y0, x1, y1 = x, y, x + w, y + h
    inter_area, iou = bb_intersection_over_union([x0, y0, x1, y1], bbox)
    return inter_area


def affine_point(pt, m):
    _x, _y = pt
    x = m[0][0] * _x + m[0][1] * _y + m[0][2]
    y = m[1][0] * _x + m[1][1] * _y + m[1][2]
    return x, y


def _bbox_of(node, path):
    x0, x1, y0, y1 = path.bbox()
    if 'transform' in node.attributes:
        m = list(map(float, re.search(r'\(([\d\-\.,]+)\)', node.attributes['transform'].value).group(1).split(',')))
        m = [[m[0], m[2], m[4]], [m[1], m[3], m[5]]]
        x0, y0 = affine_point((x0, y0), m)
        x1, y1 = affine_point((x1, y1), m)
    x0, x1, y0, y1 = min(x0, x1), max(x0, x1), min(y0, y1), max(y0, y1)
    return x0, x1, y0, y1


def bbox_of(node, paths):
    x0, x1, y0, y1 = math.inf, -math.inf, math.inf, -math.inf
    if type(paths) is not list:
        paths = [paths]
    for p in paths:
        _x0, _x1, _y0, _y1 = _bbox_of(node, p)
        x0, x1, y0, y1 = min(x0, _x0), max(
            x1, _x1), min(y0, _y0), max(y1, _y1)
    return x0, x1, y0, y1


def segmentation_bbox(png_file, svg_file, bbox, output_file, use_optimize=True):
    """
    Params:
    - png_file(str): path to input png file
    - svg_file(str): path to input svg file
    - bbox(list): segmentation bounding box, [x, y, w, h]
    - output_file(str): path to output segment svg file
    """
    img = load_img(png_file)
    _height, _width, _ = img.shape
    aspect_ratio = _height / _width
    zoom_ratio = min(_height, _width) / IMG_SIZE
    height, width = int(_height / zoom_ratio), int(_width / zoom_ratio)

    svg = load_svg(svg_file)
    defs, g, svg_attributes = svg
    viewbox = re.findall(r'\d+\.?\d*', svg_attributes['viewBox'])
    t = width / (float(viewbox[2]) - float(viewbox[0]))
    for attr in svg_attributes:
        if attr not in SVGAttribute:
            svg_attributes.pop(attr)
    g = parseString(g)
    path_nodes, paths, d_strings = find_paths(g)
    path_bboxes = [bbox_of(n, p) for n, p in zip(path_nodes, paths)]                

    bbox = tuple(bbox)
    x, y, w, h = bbox
    x, y, w, h = x / zoom_ratio, y / zoom_ratio, w / zoom_ratio, h / zoom_ratio

    seg_path = []
    for pi, (node, path, path_bbox) in enumerate(zip(path_nodes, paths, path_bboxes)):
        _xmin, _xmax, _ymin, _ymax = path_bbox
        xmin, ymin, xmax, ymax = _xmin * t, _ymin * t, _xmax * t, _ymax * t
        xmin, xmax, ymin, ymax = int(min(xmin, xmax)), int(max(xmin, xmax)), int(min(ymin, ymax)), int(max(ymin, ymax))
        inter_area = inter_area_with_bbox((x, y, w, h), [xmin, ymin, xmax, ymax])
        path_bbox_area = (ymax-ymin) * (xmax-xmin)
        valid = path_bbox_area > 0 and inter_area / path_bbox_area > 0.5
        if valid:
            seg_path.append(path_nodes[pi] if defs == '' else find_parent(path_nodes[pi]))

    svg_attributes['viewBox'] = '{} {} {} {}'.format(x / t, y / t, w / t, h / t)
    if len(seg_path) > 0:
        print('Extract svg crop to {}.'.format(output_file))
        write_svg(output_file, defs, seg_path, svg_attributes)

        if use_optimize:
            subprocess.call('NODE_OPTIONS=--max_old_space_size=8192 svgo --pretty --quiet --config=svgo.yml {}'.format(output_file), shell=True)


if __name__ == "__main__":
    segmentation_bbox(
        'check_20200706/506285959.png',
        'check_20200706/506285959.svg',
        [0, 0, 300, 230],
        'debug/test_segmentation_bbox.svg'
    )