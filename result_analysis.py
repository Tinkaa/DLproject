"""
Iterate through results in a txt file
For each result, find a corresponding annotation xml file
Find a matching object
Compare if IoU is more than 0.75 (this parameter is tunable)
"""
from os import path, walk as walk_dir
from itertools import groupby
import untangle
from frcnn.data_generators import iou


def parse_int(num):
    return int(float(num))


def load_xml_tree(xml_file_path):
    with open(xml_file_path, 'r') as f:
        return untangle.parse(f.read())

def build_result_tree(_file_path):
    """
    Build a result tree structure from txt
    """
    tree = {}
    with open(_file_path, 'r') as rs:
        objs = map(lambda l: tuple(l.split()), rs)
        objs = map(lambda o: { 'name': o[0], 'score': float(o[1]), 'x1': parse_int(o[2]), 'y1': parse_int(o[3]), 'x2': parse_int(o[4]), 'y2': parse_int(o[5]) }, objs)
        for k, g in groupby(objs, lambda k: k['name']):
            tree[k] = list(g)
    return tree

# result_tree = build_result_tree(file_path)

def find_confusion_matrix(clazz, result_tree, annotation_root, iou_thres, confidence_thres):
    """
    Find TP - match clazz with confidence at least 0.8
    Find FP - with confidence >= 0.8, no match clazz
    Find FN - match clazz with low confidence (< 0.8), or no bounding box at all
    """
    _tp, _fp, _fn = 0, 0, 0

    for img_name, stats in result_tree.items():
        annotation_path = path.join(annotation_root, img_name + '.xml')
        xml = load_xml_tree(annotation_path) # read annotation xml
        bbs = xml.annotation.object
        bbs = filter(lambda b: b.name.cdata == clazz, bbs)

        for stat in stats: # compare a detection with all bounding boxes
            score = stat['score']
            x1 = stat['x1']
            y1 = stat['y1']
            x2 = stat['x2']
            y2 = stat['y2']

            overlap_class_match_count = 0

            for _bb in bbs:
                _x1 = parse_int(_bb.bndbox.xmin.cdata)
                _y1 = parse_int(_bb.bndbox.ymin.cdata)
                _x2 = parse_int(_bb.bndbox.xmax.cdata)
                _y2 = parse_int(_bb.bndbox.ymax.cdata)

                a = (x1, y1, x2, y2)
                b = (_x1, _y1, _x2, _y2)

                iou_score = iou(a, b)
                if iou_score > iou_thres: # If bounding box overlap is enough
                    overlap_class_match_count += 1
                    if score > confidence_thres:
                        _tp += 1
                    else: # This almost never happen
                        _fn += 1

            if overlap_class_match_count == 0:
                _fp += 1

    # Find remaining FN
    for subdirs, dirs, files in walk_dir(annotation_root):
        for annotation_file in files:
            annotation_path = path.join(annotation_root, annotation_file)
            xml = load_xml_tree(annotation_path)
            img_name = xml.filename.cdata
            img_name = img_name[:-4]
            bbs = xml.annotation.object
            bbs = list(filter(lambda b: b.name.cdata == clazz, bbs))

            if not bbs:
                continue

            if img_name in result_tree:
                # if no overlap can be found, this is a FN
                overlap_found = 0
                for _bb in bbs:
                    _x1 = parse_int(_bb.bndbox.xmin.cdata)
                    _y1 = parse_int(_bb.bndbox.ymin.cdata)
                    _x2 = parse_int(_bb.bndbox.xmax.cdata)
                    _y2 = parse_int(_bb.bndbox.ymax.cdata)
                    for det in result_tree[img_name]:
                        x1 = det['x1']
                        y1 = det['y1']
                        x2 = det['x2']
                        y2 = det['y2']

                        a = (x1, y1, x2, y2)
                        b = (_x1, _y1, _x2, _y2)
                        if iou(a, b) > iou_thres:
                            overlap_found += 1
                if overlap_found == 0:
                    _fn += 1
            elif bbs: # if there is class that we want, this is a FN
                _fn += 1
  
    return _tp, _fp, _fn


def mAP(_result_file_path, clazz, annotation_root, iou_thres=0.75, confidence_thres=0.8):
    """
    Return precision and recall
    sample usage: `mAP('results/VOC2012/Main/comp3_det_test_dog.txt', 'dog', '/Users/kha/Downloads/VOCdevkit/VOC2007/Annotations')`
    """
    result_tree = build_result_tree(_result_file_path)
    tp, fp, fn = find_confusion_matrix(clazz, result_tree, annotation_root, iou_thres, confidence_thres)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return precision, recall

