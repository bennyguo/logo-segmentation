import cv2
import numpy as np
import re
import logging
import os
import os.path as osp
from xml.dom.minidom import parse, parseString
from svgwrite import Drawing

# segmentation config
SVGAttribute = ['about', 'baseProfile', 'class', 'content', 'contentScriptType', 'datatype',
                'externalResourcesRequired', 'focusHighlight', 'focusable', 'height', 'id',
                'nav-down', 'nav-down-left', 'nav-down-right', 'nav-left', 'nav-next', 'nav-prev',
                'nav-right', 'nav-up', 'nav-up-left', 'nav-up-right', 'playbackOrder', 'preserveAspectRatio',
                'property', 'rel', 'resource', 'rev', 'role', 'snapshotTime', 'syncBehaviorDefault',
                'syncToleranceDefault', 'timelineBegin', 'typeof', 'version', 'viewBox', 'width', 'xml:base',
                'xml:id', 'xml:lang', 'xml:space', 'xmlns', 'xmlns:xlink', 'xmlns:ev', 'zoomAndPan']
COORD_PAIR_TMPLT = re.compile(
    r'([\+-]?\d*[\.\d]\d*[eE][\+-]?\d+|[\+-]?\d*[\.\d]\d*)' +
    r'(?:\s*,\s*|\s+|(?=-))' +
    r'([\+-]?\d*[\.\d]\d*[eE][\+-]?\d+|[\+-]?\d*[\.\d]\d*)'
)
IMG_SIZE = 1000
POINT_NUM = 100
DISTANCE = -10
BBOX_INTER_RATIO_THRESH = 0.2
GAP_SIZE_THRESH = 1
GAP_FILTER_N_STD = 3
GAP_DISTINGUISH_THRESH = 0.6


# convert config
COORD_PAIR_TMPLT = re.compile(
    r'([\+-]?\d*[\.\d]\d*[eE][\+-]?\d+|[\+-]?\d*[\.\d]\d*)' +
    r'(?:\s*,\s*|\s+|(?=-))' +
    r'([\+-]?\d*[\.\d]\d*[eE][\+-]?\d+|[\+-]?\d*[\.\d]\d*)'
)
PATH = ['rect', 'circle', 'ellipse', 'line', 'polygon', 'polyline', 'path']
SIZE_OUTER_THRESH = 0.1
G_SIZE_FILTER_THRESH = 0.8
SIMILAR_NODE_NUM_THRESH = 32


def path2pathd(path):
    return path.get('d', '')


def ellipse2pathd(ellipse):
    """converts the parameters from an ellipse or a circle to a string for a
    Path object d-attribute"""

    cx = ellipse.get('cx', 0)
    cy = ellipse.get('cy', 0)
    rx = ellipse.get('rx', None)
    ry = ellipse.get('ry', None)
    r = ellipse.get('r', None)

    if r is not None:
        rx = ry = float(r)
    else:
        rx = float(rx)
        ry = float(ry)

    cx = float(cx)
    cy = float(cy)

    d = ''
    d += 'M' + str(cx - rx) + ',' + str(cy)
    d += 'a' + str(rx) + ',' + str(ry) + ' 0 1,0 ' + str(2 * rx) + ',0'
    d += 'a' + str(rx) + ',' + str(ry) + ' 0 1,0 ' + str(-2 * rx) + ',0'

    return d


def polyline2pathd(polyline_d, is_polygon=False):
    """converts the string from a polyline points-attribute to a string for a
    Path object d-attribute"""
    points = COORD_PAIR_TMPLT.findall(polyline_d)
    closed = (float(points[0][0]) == float(points[-1][0]) and
              float(points[0][1]) == float(points[-1][1]))

    # The `parse_path` call ignores redundant 'z' (closure) commands
    # e.g. `parse_path('M0 0L100 100Z') == parse_path('M0 0L100 100L0 0Z')`
    # This check ensures that an n-point polygon is converted to an n-Line path.
    if is_polygon and closed:
        points.append(points[0])

    d = 'M' + 'L'.join('{0} {1}'.format(x, y) for x, y in points)
    if is_polygon or closed:
        d += 'z'
    return d


def polygon2pathd(polyline_d):
    """converts the string from a polygon points-attribute to a string
    for a Path object d-attribute.
    Note:  For a polygon made from n points, the resulting path will be
    composed of n lines (even if some of these lines have length zero).
    """
    return polyline2pathd(polyline_d, True)


def rect2pathd(rect):
    """Converts an SVG-rect element to a Path d-string.

    The rectangle will start at the (x,y) coordinate specified by the
    rectangle object and proceed counter-clockwise."""
    x0, y0 = float(rect.get('x', 0)), float(rect.get('y', 0))
    w, h = float(rect.get('width', 0)), float(rect.get('height', 0))
    x1, y1 = x0 + w, y0
    x2, y2 = x0 + w, y0 + h
    x3, y3 = x0, y0 + h

    d = ("M{} {} L {} {} L {} {} L {} {} z"
         "".format(x0, y0, x1, y1, x2, y2, x3, y3))
    return d


def line2pathd(l):
    return 'M' + l['x1'] + ' ' + l['y1'] + 'L' + l['x2'] + ' ' + l['y2']


def dom2dict(element):
    """Converts DOM elements to dictionaries of attributes."""
    keys = list(element.attributes.keys())
    values = [val.value for val in list(element.attributes.values())]
    return dict(list(zip(keys, values)))


def load_svg(file_path):
    """Load svg file as defs, g and svg_attributes."""
    assert os.path.exists(file_path)
    doc = parse(file_path)

    svg = doc.getElementsByTagName('svg')[0]
    svg_attributes = dom2dict(svg)

    defs = g = ''
    for i, tag in enumerate(svg.childNodes):
        if tag.localName == 'defs':
            defs = tag.toxml()
        if tag.localName == 'g':
            g = tag.toxml()

    doc.unlink()

    return defs, g, svg_attributes


def write_svg(svgpath, defs, paths, svg_attributes):
    # Create an SVG file
    assert svg_attributes is not None
    dwg = Drawing(filename=svgpath, **svg_attributes)
    doc = parseString(dwg.tostring())

    svg = doc.firstChild
    if defs != '':
        defsnode = parseString(defs).firstChild
        svg.replaceChild(defsnode, svg.firstChild)
    for i, path in enumerate(paths):
        svg.appendChild(path.cloneNode(deep=True))

    xmlstring = doc.toprettyxml()
    doc.unlink()
    with open(svgpath, 'w') as f:
        f.write(xmlstring)       


def load_img(img_path):
    return cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)

def get_img_mask(img):
    alpha = img[:,:,-1]
    mask = cv2.inRange(alpha, 1, 255)
    return mask

def get_logger(name, filename=None, echo=True, multiprocessing=False):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    if filename:
        fh = logging.FileHandler(filename)
        fh.setLevel(logging.DEBUG)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO if not echo else logging.DEBUG)

    if not multiprocessing:
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    else:
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(processName)-10s - %(levelname)s - %(message)s')

    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if filename:
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    
    return logger