# -*- coding: utf-8 -*-
import os
import os.path as osp
import re
import cv2
import sys
import errno
import numpy as np
import multiprocessing
import shutil
from shutil import rmtree
from svgpathtools import wsvg, parse_path
from xml.dom.minidom import parse, parseString
import traceback
import subprocess
import multiprocessing
from multiprocessing import Pool
from multiprocessing_logging import install_mp_handler
import glob
import math
import json
import argparse
from tqdm import tqdm
from utils import *

import logging
logger = None

USE_OPTIMIZE = True
EXPORT_CONTOUR_RESULTS = True
EXPORT_MASK = True

# def path2pathd(path):
#     return path.get('d', '')

# def ellipse2pathd(ellipse):
#     """converts the parameters from an ellipse or a circle to a string for a
#     Path object d-attribute"""

#     cx = ellipse.get('cx', 0)
#     cy = ellipse.get('cy', 0)
#     rx = ellipse.get('rx', None)
#     ry = ellipse.get('ry', None)
#     r = ellipse.get('r', None)

#     if r is not None:
#         rx = ry = float(r)
#     else:
#         rx = float(rx)
#         ry = float(ry)

#     cx = float(cx)
#     cy = float(cy)

#     d = ''
#     d += 'M' + str(cx - rx) + ',' + str(cy)
#     d += 'a' + str(rx) + ',' + str(ry) + ' 0 1,0 ' + str(2 * rx) + ',0'
#     d += 'a' + str(rx) + ',' + str(ry) + ' 0 1,0 ' + str(-2 * rx) + ',0'

#     return d

# def polyline2pathd(polyline_d, is_polygon=False):
#     """converts the string from a polyline points-attribute to a string for a
#     Path object d-attribute"""
#     points = COORD_PAIR_TMPLT.findall(polyline_d)
#     closed = (float(points[0][0]) == float(points[-1][0]) and
#               float(points[0][1]) == float(points[-1][1]))

#     # The `parse_path` call ignores redundant 'z' (closure) commands
#     # e.g. `parse_path('M0 0L100 100Z') == parse_path('M0 0L100 100L0 0Z')`
#     # This check ensures that an n-point polygon is converted to an n-Line path.
#     if is_polygon and closed:
#         points.append(points[0])

#     d = 'M' + 'L'.join('{0} {1}'.format(x, y) for x, y in points)
#     if is_polygon or closed:
#         d += 'z'
#     return d

# def polygon2pathd(polyline_d):
#     """converts the string from a polygon points-attribute to a string
#     for a Path object d-attribute.
#     Note:  For a polygon made from n points, the resulting path will be
#     composed of n lines (even if some of these lines have length zero).
#     """
#     return polyline2pathd(polyline_d, True)

# def rect2pathd(rect):
#     """Converts an SVG-rect element to a Path d-string.

#     The rectangle will start at the (x,y) coordinate specified by the
#     rectangle object and proceed counter-clockwise."""
#     x0, y0 = float(rect.get('x', 0)), float(rect.get('y', 0))
#     w, h = float(rect.get('width', 0)), float(rect.get('height', 0))
#     x1, y1 = x0 + w, y0
#     x2, y2 = x0 + w, y0 + h
#     x3, y3 = x0, y0 + h

#     d = ("M{} {} L {} {} L {} {} L {} {} z"
#          "".format(x0, y0, x1, y1, x2, y2, x3, y3))
#     return d

# def line2pathd(l):
#     return 'M' + l['x1'] + ' ' + l['y1'] + 'L' + l['x2'] + ' ' + l['y2']

# def dom2dict(element):
#     """Converts DOM elements to dictionaries of attributes."""
#     keys = list(element.attributes.keys())
#     values = [val.value for val in list(element.attributes.values())]
#     return dict(list(zip(keys, values)))

# def load_svg(file_path):
#     """Load svg file as defs, g and svg_attributes."""
#     assert os.path.exists(file_path)
#     doc = parse(file_path)

#     svg = doc.getElementsByTagName('svg')[0]
#     svg_attributes = dom2dict(svg)

#     defs = g = ''
#     for i, tag in enumerate(svg.childNodes):
#         if tag.localName == 'defs':
#             defs = tag.toxml()
#         if tag.localName == 'g':
#             g = tag.toxml()

#     doc.unlink()

#     return defs, g, svg_attributes

# def write_svg(svgpath, defs, paths, svg_attributes):
#     # Create an SVG file
#     assert svg_attributes is not None
#     dwg = Drawing(filename=svgpath, **svg_attributes)
#     doc = parseString(dwg.tostring())

#     svg = doc.firstChild
#     if defs != '':
#         defsnode = parseString(defs).firstChild
#         svg.replaceChild(defsnode, svg.firstChild)
#     for i, path in enumerate(paths):
#         svg.appendChild(path)

#     xmlstring = doc.toprettyxml()
#     doc.unlink()
#     with open(svgpath, 'w') as f:
#         f.write(xmlstring)

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
            logger.debug('Parse d string {}... fail!'.format(d[:100]))
            continue
    # path_list = [parse_path(d) for d in d_strings]
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


def inter_area_with_contour(contour, bbox):
    x, y, w, h = cv2.boundingRect(contour)
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


def calc_gaps(img, axis):
    in_gap = False
    start = 0
    gaps = []
    for r, row_sum in enumerate(np.sum(img, axis=axis)):
        if row_sum != 0 and in_gap:
            in_gap = False
            if start > 0:
                gaps.append((start, r))
        if row_sum == 0 and not in_gap:
            in_gap = True
            start = r
        if row_sum > 0:
            start = True
    return gaps


def filter_gaps(gaps):
    _gap_sizes = list(map(lambda g: g[1] - g[0], gaps))
    g = zip(gaps, _gap_sizes)
    def get_gap_sizes(g):
        return [gg[1] for gg in g]
    g = list(filter(lambda gg: gg[1] > GAP_SIZE_THRESH, g))
    gap_sizes = get_gap_sizes(g)
    if len(g) > 1:
        mean, std = np.mean(gap_sizes), np.std(gap_sizes) + 1e-5
        g = list(filter(lambda gg: gg[1] > mean - GAP_FILTER_N_STD * std and gg[1] < mean + GAP_FILTER_N_STD * std, g))
    return [gg[0] for gg in g]


def get_adaptive_threshold(gaps):
    if len(gaps) == 0:
        return 0
    gap_sizes = np.array(list(map(lambda g: g[1] - g[0], gaps)))
    if len(gaps) < 5:
        return np.min(gap_sizes) - 1
    mean = np.mean(gap_sizes)
    if np.mean(gap_sizes[gap_sizes <= mean]) < GAP_DISTINGUISH_THRESH * np.mean(gap_sizes[gap_sizes >= mean]):
        return int(mean)
    return np.min(gap_sizes) - 1

def morph_mask(img):
    gaps0, gaps1 = calc_gaps(img, axis=1), calc_gaps(img, axis=0)
    gaps0, gaps1 = filter_gaps(gaps0), filter_gaps(gaps1)
    gap_sizes0, gap_sizes1 = list(map(lambda g: g[1] - g[0], gaps0)), list(map(lambda g: g[1] - g[0], gaps1))
    logger.debug('Gaps: {} {}'.format(gap_sizes0, gap_sizes1))
    th0, th1 = get_adaptive_threshold(gaps0), get_adaptive_threshold(gaps1)
    if not th0 and not th1:
        th0, th1 = 8, 8
    k0, k1 = max(th0, 1), max(th1, 1)
    logger.debug('Morphing kernel: {} {}'.format(k0, k1))
    morph_kernel = np.ones((k0, k1), np.uint8)
    mask_morph = cv2.morphologyEx(img, cv2.MORPH_CLOSE, morph_kernel)
    return mask_morph       


def save_segs(img, svg, t, zoom_ratio, contours, _id, output_dir):
    defs, g, svg_attributes = svg
    g = parseString(g)
    path_nodes, paths, d_strings = find_paths(g)
    path_bboxes = [bbox_of(n, p) for n, p in zip(path_nodes, paths)]
    for cidx, cnt in enumerate(contours):
        # Get the position of each contour
        x, y, w, h = cv2.boundingRect(cnt)
        x, y, w, h = math.ceil(x * zoom_ratio), math.ceil(y * zoom_ratio), math.ceil(w * zoom_ratio), math.ceil(h * zoom_ratio)

        seg_path = []
        for pi, (node, path, path_bbox) in enumerate(zip(path_nodes, paths, path_bboxes)):
            _xmin, _xmax, _ymin, _ymax = path_bbox
            xmin, ymin, xmax, ymax = _xmin * t / zoom_ratio, _ymin * t / zoom_ratio, _xmax * t / zoom_ratio, _ymax * t / zoom_ratio
            xmin, xmax, ymin, ymax = int(min(xmin, xmax)), int(max(xmin, xmax)), int(min(ymin, ymax)), int(max(ymin, ymax))
            x0, x1, y0, y1 = x, y, x + w, y + h
            inter_area = inter_area_with_contour(cnt, [xmin, ymin, xmax, ymax])
            path_bbox_area = (ymax-ymin) * (xmax-xmin)
            # valid = path_bbox_area > 0 and inter_area / path_bbox_area > BBOX_INTER_RATIO_THRESH
            # if valid:
            #     all_inter_areas = list(map(lambda c: inter_area_with_contour(c, [xmin, ymin, xmax, ymax]), contours))
            #     if inter_area < max(all_inter_areas):
            #         valid = False
            valid = inter_area > 0
            if valid:
                seg_path.append(path_nodes[pi] if defs == '' else find_parent(path_nodes[pi]))

        svg_attributes['viewBox'] = '{} {} {} {}'.format(x / t, y / t, w / t, h / t)
        if len(seg_path) > 0:
            try:
                cv2.imwrite(osp.join(output_dir, '{}-{}.png'.format(_id, cidx)), img[y:y+h, x:x+w])
            except:
                logger.warning('Save png seg {}-{} failed.'.format(_id, cidx))
                logger.debug('Save segment {}-{} failed.'.format(_id, cidx), exc_info=True)
            try:
                write_svg(osp.join(output_dir, '{}-{}.svg'.format(_id, cidx)), defs, seg_path, svg_attributes)
            except:
                logger.warning('Save svg seg {}-{} failed.'.format(_id, cidx))
                logger.debug('Save segment {}-{} failed.'.format(_id, cidx), exc_info=True)

            if USE_OPTIMIZE:
                try:
                    subprocess.call('NODE_OPTIONS=--max_old_space_size=8192 svgo --pretty --quiet --config=svgo.yml -f {}'.format(output_dir), shell=True)
                except:
                    logger.warning('Exception occurred when optimizing svg segs of {}.'.format(_id))
                    logger.debug('svgo fails when optimizing svg segs of {}.'.format(_id), exc_info=True)

def mark_outliars(arr, lower_thresh, upper_thresh):
    try:
        arr = np.array(arr)
        _arr = arr[(arr > lower_thresh) & (arr < upper_thresh)]
        if len(_arr) == 0:
            return np.ones(arr.shape, dtype=np.bool)
        mean, std = _arr.mean(), _arr.std()
        _arr = _arr[(_arr <= mean + std) & (_arr >= mean - std)]
        mean = _arr.mean()
        return (arr > 2 * mean) | (arr < 0.5 * mean)
    except:
        logging.error('Exception occurred when marking outliars in {} with ({},{}).'.format(arr, lower_thresh, upper_thresh), exc_info=True)
        exit(1)

def get_confidence_score(size, orig_contours, contours):
    width, height = size
    orig_contours_bbox = [cv2.boundingRect(cnt) for cnt in orig_contours]
    contours_bbox = [cv2.boundingRect(cnt) for cnt in contours]
    n_orig_contours, n_contours = len(orig_contours), len(contours)
    
    conf_score = 1

    if n_contours == 0 or (n_contours == 1 and contours_bbox[0][2] * contours_bbox[0][3] > 0.8 * width * height):
        return 0


    if int(math.sqrt(n_contours))**2 == n_contours:
        conf_score *= 1
    else:
        conf_score *= 0.9

    outliar_mark = mark_outliars([b[2] * b[3] for b in contours_bbox], 9, 0.8 * width * height) | mark_outliars([b[2] for b in contours_bbox], 3, 0.8 * width) | mark_outliars([b[3] for b in contours_bbox], 3, 0.8 * height)
    n_size_invalid = np.count_nonzero(outliar_mark)
    conf_score *= 0.9**n_size_invalid

    return conf_score


def do_seg(inp):
    tgt, tgt_dir = inp

    _id = tgt['id']
    svg_file, png_file = tgt['svg_file'], tgt['png_file']
    logger.debug('Processing ({}, {})'.format(svg_file, png_file))

    img = load_img(png_file)
    mask = get_img_mask(img)

    _height, _width, _ = img.shape
    aspect_ratio = _height / _width
    zoom_ratio = min(_height, _width) / IMG_SIZE
    height, width = int(_height / zoom_ratio), int(_width / zoom_ratio)
    
    img_resized = cv2.resize(img, (width, height), interpolation=cv2.INTER_NEAREST)
    mask_resized = get_img_mask(img_resized)

    svg = load_svg(svg_file)
    defs, g, svg_attributes = svg
    

    viewbox = re.findall(r'\d+\.?\d*', svg_attributes['viewBox'])
    t = _width / (float(viewbox[2]) - float(viewbox[0]))
    for attr in svg_attributes:
        if attr not in SVGAttribute:
            svg_attributes.pop(attr)

    mask_morph = morph_mask(mask_resized)
    if cv2.__version__ >= '4.0.0':
        contours, _ = cv2.findContours(mask_morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        orig_contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    else:
        _, contours, _ = cv2.findContours(mask_morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        _, orig_contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    conf_score = get_confidence_score((width, height), orig_contours, contours)

    canvas = cv2.cvtColor(mask_morph, cv2.COLOR_GRAY2RGB)
    cv2.drawContours(canvas, contours, -1, (255, 255, 0), 3)

    if EXPORT_MASK:
        cv2.imwrite(osp.join(tgt_dir, '{}-morph-mask.jpg'.format(_id)), mask_morph)
        cv2.imwrite(osp.join(tgt_dir, '{}-morph-contour.jpg'.format(_id)), canvas)

    save_dir = osp.join(tgt_dir, _id + '-morph-segs')
    os.makedirs(save_dir, exist_ok=True)
    if EXPORT_CONTOUR_RESULTS:
        contour_save_dir = osp.join(tgt_dir, _id + '-contour-segs')
        os.makedirs(contour_save_dir, exist_ok=True)

    ok = True
    if not len(os.listdir(save_dir)) == len(contours) * 2:
        ok = False
    if EXPORT_CONTOUR_RESULTS and not len(os.listdir(contour_save_dir)) == len(orig_contours) * 2:
        ok = False
    
    if ok:
        return conf_score

    try:
        save_segs(img, svg, t, zoom_ratio, contours, _id, save_dir)
    except:
        logger.error('{} save morph segs failed. (probably due to image elements)'.format(_id))
        logger.debug('{} segmentation failed.'.format(_id), exc_info=True)
        return 0
    
    if EXPORT_CONTOUR_RESULTS:
        try:
            save_segs(img, svg, t, zoom_ratio, orig_contours, _id, contour_save_dir)    
        except:
            logger.error('{} save contour segs failed. (probably due to image elements)'.format(_id))
            logger.debug('{} segmentation failed.'.format(_id), exc_info=True)
            return 0

    return conf_score


def seg(tgts, tgt_dir, num_workers=1):
    if num_workers > 1:
        with Pool(num_workers) as p:
            conf = list(tqdm(p.imap(do_seg, [(tgt, tgt_dir) for tgt in tgts]), total=len(tgts)))
    else:
        conf = [do_seg((tgt, tgt_dir)) for tgt in tgts]
    return conf


def main(argv):
    parser = argparse.ArgumentParser(description='Perform segmentation on .svg and .png files.')
    parser.add_argument('dirs', nargs='+', help='Directories that stores .svg&.png files.')
    parser.add_argument('--num-workers', default=0, type=int, dest='num_workers', help='Number of processes. 0 for all available cpu cores.')
    parser.add_argument('--log', default='segmentation.log', type=str, dest='log_file', help='Path to log file.')
    parser.add_argument('--conf', default='seg_conf.json', type=str, dest='confidence_file', help='Path to segmentation confidence file.')
    parser.add_argument('--no-optimize', default=True, action='store_false', dest='optimize', help="Dont't use svgo optimization. This will produce larger svg files but cost much less time.")
    parser.add_argument('--export-contour', default=False, action='store_true', dest='export_contour', help='Export contour segmentation results.')
    parser.add_argument('--export-mask', default=False, action='store_true', dest='export_mask', help='Export morphed mask for debug use.')
    args = parser.parse_args(argv[1:])

    global logger
    logger = get_logger('segmentation', args.log_file, echo=False, multiprocessing=True)
    install_mp_handler(logger)

    global USE_OPTIMIZE, EXPORT_CONTOUR_RESULTS, EXPORT_MASK
    USE_OPTIMIZE = args.optimize
    EXPORT_CONTOUR_RESULTS = args.export_contour
    EXPORT_MASK = args.export_mask

    num_workers = args.num_workers
    if num_workers == 0:
        num_workers = multiprocessing.cpu_count()
    logger.info('Using {} processes.'.format(num_workers))    

    src_dirs = args.dirs
    for src_dir in src_dirs:
        if not osp.isdir(src_dir):
            continue
        logger.info('Processing {} ...'.format(src_dir))
        tgt_dir = src_dir
        tgts = []
        for f in glob.glob(osp.join(src_dir, '*.eps')):
            _id, _ = osp.splitext(osp.basename(f))
            svg_file, png_file = osp.join(src_dir, _id + '.svg'), osp.join(src_dir, _id + '.png')
            if osp.exists(svg_file) and osp.exists(png_file):
                tgts.append({
                    'id': _id,
                    'svg_file': svg_file,
                    'png_file': png_file
                })
        conf = seg(tgts, tgt_dir, num_workers=num_workers)
        with open(args.confidence_file, 'w') as f:
            f.write(json.dumps([
                {'id': tgt['id'], 'score': s} for tgt, s in zip(tgts, conf)
            ]))


def debug(files, debug_dir):
    global logger
    logger = get_logger('segmentation', None, echo=True, multiprocessing=False)

    global USE_OPTIMIZE, EXPORT_CONTOUR_RESULTS, EXPORT_MASK
    USE_OPTIMIZE = True
    EXPORT_CONTOUR_RESULTS = True
    EXPORT_MASK = True

    if osp.exists(debug_dir):
        shutil.rmtree(debug_dir)
    os.makedirs(debug_dir)
    tgts = []
    for f in files:
        src_dir = osp.dirname(f)
        _id, _ = osp.splitext(osp.basename(f))
        svg_file, png_file = osp.join(src_dir, _id + '.svg'), osp.join(src_dir, _id + '.png')
        if osp.exists(svg_file) and osp.exists(png_file):
            tgts.append({
                'id': _id,
                'svg_file': svg_file,
                'png_file': png_file
            })              
            shutil.copy(png_file, debug_dir)
    conf = seg(tgts, debug_dir, num_workers=1)
    with open(osp.join(debug_dir, 'debug_confidence.json'), 'w') as f:
        f.write(json.dumps([
            {'id': tgt['id'], 'score': s} for tgt, s in zip(tgts, conf)
        ]))    
    logger.debug('Confidence: {}'.format(conf))


def evaluate(files, gt_dir, vis_dir):
    pass


if __name__ == "__main__":
    main(sys.argv)
    # main(['', 'ICON-621-fails'])
    # debug(
    #     ['check_20200818/VCG41N694733006.eps'],
    #     'debug'
    # )