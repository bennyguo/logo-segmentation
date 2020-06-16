# -*- coding: utf-8 -*-
from xml.dom.minidom import parse, parseString
from svgpathtools import wsvg, parse_path
from svgwrite import Drawing
from pathlib import Path
import os
import os.path as osp
import re
import sys
import glob
import subprocess
from functools import reduce
from itertools import starmap
import shutil
import math
import multiprocessing
from multiprocessing import Pool
from multiprocessing_logging import install_mp_handler
from collections import defaultdict
import argparse
from tqdm import tqdm
from utils import *

import logging
logger = None

USE_OPTIMIZE = False

tasks = {
    'win32': {
        'svg2png': {
            'bin': 'inkscape.exe',
            'command': '{0} -z {1} -e {2}',
            'args': ['svg_file', 'png_file']
        },
        'pdf2svg': {
            'bin': 'pdf2svg.exe',
            'command': '{0} {1} {2}',
            'args': ['pdf_file', 'svg_file']
        },
        'eps2pdf': {
            'bin': 'epstopdf.exe',
            'command': '{0} {1} {2}',
            'args': ['eps_file', 'pdf_file']
        },
        'svgo': {
            'bin': 'svgo.exe',
            'command': '{0} --pretty --quiet --config=svgo.yml {1}',
            'args': ['svg_file']
        }
    },

    'others': {
        'svg2png': {
            'bin': 'inkscape',
            'command': '{0} -z {1} -e {2}',
            'args': ['svg_file', 'png_file']
        },
        'pdf2svg': {
            'bin': 'pdf2svg',
            'command': '{0} {1} {2}',
            'args': ['pdf_file', 'svg_file']
        },
        'eps2pdf': {
            'bin': 'epstopdf',
            'command': '{0} {1} {2}',
            'args': ['eps_file', 'pdf_file']
        },
        'svgo': {
            'bin': 'svgo',
            'command': 'NODE_OPTIONS=--max_old_space_size=8192 {0} --pretty --quiet --config=svgo.yml {1}',
            'args': ['svg_file']
        }
    }
}

platform = 'win32' if sys.platform == 'win32' else 'others'
        
def safe_parse_path(d):
    p = None
    try:
        p = parse_path(d)
    except:
        logger.debug('Parse d string {}... failed.'.format(d[:100]))
    return p 

def to_path(node):
    if node.localName == 'path':
        return safe_parse_path(dom2dict(node)['d'])
    elif node.localName == 'polyline':
        pl = dom2dict(node)
        return safe_parse_path(polyline2pathd(pl['points']))
    elif node.localName == 'polygon':
        pg = dom2dict(node)
        return safe_parse_path(polygon2pathd(pg['points']))
    elif node.localName == 'line':
        l = dom2dict(node)
        return safe_parse_path('M' + l['x1'] + ' ' + l['y1'] + 'L' + l['x2'] + ' ' + l['y2'])
    elif node.localName == 'ellipse':
        e = dom2dict(node)
        return safe_parse_path(ellipse2pathd(e))
    elif node.localName == 'circle':
        c = dom2dict(node)
        return safe_parse_path(ellipse2pathd(c))
    elif node.localName == 'rect':
        r = dom2dict(node)
        return safe_parse_path(rect2pathd(r))


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

    path_list = [safe_parse_path(d) for d in d_strings]
    return path_list


def check_env():
    requirements = [t['bin'] for t in tasks[platform].values()]
    path_files = set(reduce(lambda a, b: a+b, [os.listdir(p) for p in os.environ['PATH'].split(
        ';' if sys.platform == 'win32' else ':') if os.path.isdir(p)]))
    ok = True
    for r in requirements:
        if not r in path_files:
            logger.error('{} not exists in path!'.format(r))
            ok = False
    if not ok:
        raise EnvironmentError


def run_task(task_name, **kwargs):
    logger.debug('Running task {} ...'.format(task_name))
    task = tasks[platform][task_name]
    p = subprocess.check_output(
        task['command'].format(
            task['bin'],
            *(kwargs[arg_name] for arg_name in task['args'])
        ),
        stderr=subprocess.STDOUT,
        shell=True
    ).decode('utf-8').strip()
    logger.debug('{}: {}'.format(task_name, p))


def node2paths(node):
    paths = []
    if node.localName in PATH:
        paths.append(to_path(node))
    if node.localName == 'g':
        paths += find_paths(node)
    return list(filter(None, paths))


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
    for p in paths:
        _x0, _x1, _y0, _y1 = _bbox_of(node, p)
        x0, x1, y0, y1 = min(x0, _x0), max(
            x1, _x1), min(y0, _y0), max(y1, _y1)
    return x0, x1, y0, y1


def at_center_of(height, width, els):
    if len(els) > 0:
        center_x2 = sum(map(lambda el: el[2][
                        0] + el[2][1], els)) / len(els)
        center_y2 = sum(map(lambda el: el[2][
                        2] + el[2][3], els)) / len(els)
        logger.debug('{} [{} {} {}] [{} {} {}]'.format(
            len(els), 0.9 * width, center_x2, 1.1 * width, 0.9 * height, center_y2, 1.1 * height))
        if 0.9 * width < center_x2 < 1.1 * width and 0.9 * height < center_y2 < 1.1 * height:
            return True
    return False


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
    # logger.debug('Gaps: {} {}'.format(gap_sizes0, gap_sizes1))
    th0, th1 = get_adaptive_threshold(gaps0), get_adaptive_threshold(gaps1)
    if not th0 and not th1:
        th0, th1 = 8, 8
    k0, k1 = max(th0, 1), max(th1, 1)
    # logger.debug('Morphing kernel: {} {}'.format(k0, k1))
    morph_kernel = np.ones((k0, k1), np.uint8)
    mask_morph = cv2.morphologyEx(img, cv2.MORPH_CLOSE, morph_kernel)
    return mask_morph  


def number_of_morph_contours(png_file):
    img = load_img(png_file)
    mask = get_img_mask(img)
    _height, _width, _ = img.shape
    aspect_ratio = _height / _width
    zoom_ratio = min(_height, _width) / IMG_SIZE
    height, width = int(_height / zoom_ratio), int(_width / zoom_ratio)
    img_resized = cv2.resize(img, (width, height), interpolation=cv2.INTER_NEAREST)
    mask_resized = get_img_mask(img_resized)
    mask_morph = morph_mask(mask_resized)
    if cv2.__version__ >= '4.0.0':
        contours, _ = cv2.findContours(mask_morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        orig_contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    else:
        _, contours, _ = cv2.findContours(mask_morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        _, orig_contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return len(contours)

def remove_background(input_svg_file, output_svg_file):
    logger.debug('Removing background ...')

    doc = parse(input_svg_file)
    # if len(doc.getElementsByTagName('image')) != 0:
    #     doc.unlink()
    #     return

    svg = doc.getElementsByTagName('svg')[0]
    svg_attributes = dom2dict(svg)

    viewbox = re.findall(r'\d+\.?\d*', svg_attributes['viewBox'])
    xmin, ymin = float(viewbox[0]), float(viewbox[1])
    xmax, ymax = float(viewbox[0]) + float(viewbox[2]), \
        float(viewbox[1]) + float(viewbox[3])
    width, height = float(viewbox[2]), float(viewbox[3])
    _xmin, _xmax, _ymin, _ymax = xmin - SIZE_OUTER_THRESH * width, xmax + SIZE_OUTER_THRESH * width, \
        ymin - SIZE_OUTER_THRESH * height, ymax + SIZE_OUTER_THRESH * height

    g = next(iter(filter(lambda n: n.localName == 'g', svg.childNodes)), None)

    def is_background_node_in_g_strict(bbox):
        x0, x1, y0, y1 = bbox
        if (x1 - x0) > G_SIZE_FILTER_THRESH * width and (y1 - y0) > G_SIZE_FILTER_THRESH * height:
            return True
        return False

    def is_background_node_in_g_loose(bbox):
        x0, x1, y0, y1 = bbox
        if ((x1 - x0) > G_SIZE_FILTER_THRESH * width) ^ ((y1 - y0) > G_SIZE_FILTER_THRESH * height):
            return True
        return False

    paths = list(map(node2paths, g.childNodes))
    bboxes = list(starmap(bbox_of, zip(g.childNodes, paths)))
    els = list(filter(lambda e: e[1], zip(g.childNodes, paths, bboxes)))

    bg_nodes_in_g_strict, bg_nodes_in_g_loose = [], []
    similar_els = defaultdict(list)
    path_pt_count = {}
    # last_k, last_pt, last_style = None, None, None
    for el in els:
        node, path, bbox = el
        if is_background_node_in_g_strict(bbox):
            bg_nodes_in_g_strict.append(el)
        if is_background_node_in_g_loose(bbox):
            bg_nodes_in_g_loose.append(el)
        
        if node.localName == 'path' and 'style' in node.attributes:
            style = node.attributes['style'].value.strip()
            area = '{:.2g}'.format(abs((bbox[1] - bbox[0]) * (bbox[3] - bbox[2])))
            k = '{}-{}'.format(area, style)
            similar_els[k].append(el)
            path_pt_count[k] = path

    if not at_center_of(height, width, bg_nodes_in_g_loose):
        bg_nodes_in_g_loose = []

    bg_nodes_in_g = bg_nodes_in_g_strict + bg_nodes_in_g_loose

    logger.debug('{} loose nodes, {} strict nodes'.format(
        len(bg_nodes_in_g_loose), len(bg_nodes_in_g_strict)))


    def remove_el_list(src, els):
        for node, _, _ in els:
            try:
                g.removeChild(node)
            except:
                continue
    
    remove_el_list(g, bg_nodes_in_g)
    tmp_svg_file, tmp_png_file = output_svg_file + '.tmp.svg', output_svg_file + 'tmp.svg'
    with open(tmp_svg_file, 'w') as f:
        doc.writexml(f)
    run_task('svg2png', svg_file=tmp_svg_file, png_file=tmp_png_file)
    n_contour = number_of_morph_contours(tmp_png_file)
    os.remove(tmp_svg_file)
    os.remove(tmp_png_file)
    if n_contour <= 1 or n_contour > 64:
        for k, group in similar_els.items():
            if len(group) > SIMILAR_NODE_NUM_THRESH and at_center_of(height, width, group):
                logger.debug('{} similar nodes: {}'.format(len(group), k))
                remove_el_list(g, group)

    with open(output_svg_file, 'w') as f:
        doc.writexml(f)
    doc.unlink()

    if USE_OPTIMIZE:
        try:
            run_task('svgo', svg_file=output_svg_file)
        except:
            logger.warning('Exception occurred when optimizing svg file {}.'.format(output_svg_file))
            logger.debug('svgo fails when optimizing {}.'.format(output_svg_file), exc_info=True)


def do_convert(tgt):
    f, tgt_dir = tgt['f'], tgt['tgt_dir']
    logger.debug('Converting {}'.format(f))
    filename = osp.basename(f)
    name, ext = osp.splitext(filename)
    outputs = {
        t: osp.join(tgt_dir, name + '.' + t) for t in ['png', 'pdf', 'svg']
    }
    run_task('eps2pdf', eps_file=f, pdf_file=outputs['pdf'])
    run_task('pdf2svg', pdf_file=outputs['pdf'], svg_file=outputs['svg'])
    remove_background(outputs['svg'], outputs['svg'])
    run_task('svg2png', svg_file=outputs['svg'], png_file=outputs['png'])
    os.remove(outputs['pdf'])


def convert(files, tgt_dir, num_workers=1):
    try:
        check_env()
    except EnvironmentError:
        logger.error('Environment error.')
        exit(1)

    if num_workers > 1:
        with Pool(num_workers) as p:
            r = list(tqdm(p.imap(do_convert, [{'f': f, 'tgt_dir': tgt_dir} for f in files]), total=len(files)))
    else:
        for f in files:
            do_convert({'f': f, 'tgt_dir': tgt_dir})


def main(argv):
    parser = argparse.ArgumentParser(description='Convert .eps file to .svg and .png.')
    parser.add_argument('dirs', nargs='+', help='Directories that stores .eps files.')
    parser.add_argument('--num-workers', default=0, type=int, dest='num_workers', help='Number of processes. 0 for all available cpu cores.')
    parser.add_argument('--log', default='convert.log', type=str, dest='log_file', help='Path to log file.')
    args = parser.parse_args(argv[1:])

    global logger
    logger = get_logger('convert', args.log_file, echo=False, multiprocessing=True)
    install_mp_handler(logger)

    num_workers = args.num_workers
    if num_workers == 0:
        num_workers = multiprocessing.cpu_count()
    logger.info('Using {} processes.'.format(num_workers))

    for src_dir in args.dirs:
        logger.info('Processing {} ...'.format(src_dir))
        tgt_dir = src_dir
        convert(glob.glob(osp.join(src_dir, '*.eps')), tgt_dir, num_workers=num_workers)


def debug(files, debug_dir):
    global logger
    logger = get_logger('convert', None, echo=True, multiprocessing=False)
    if osp.exists(debug_dir):
        shutil.rmtree(debug_dir)
    os.makedirs(debug_dir)
    convert(files, debug_dir, num_workers=1)


def evaluate(files, gt_dir, vis_dir):
    pass


if __name__ == "__main__":
    # main(sys.argv)
    # main(['', 'ICON-621-fails'])
    debug([
        osp.join('test', f) for f in os.listdir('test')
    ], 'debug')
