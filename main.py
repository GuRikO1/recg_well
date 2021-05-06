import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
from typing import Tuple, List, Dict
import time
from src.label_decoder import LabelDecoder
from src.coordinate_detection import CoordinateDetection


def main(args):
    start = time.time()
    label_decoder = LabelDecoder()
    labelload_elapsed_time = time.time() - start

    _start = time.time()
    img = cv2.imread(args.path_img, cv2.IMREAD_GRAYSCALE)
    cd = CoordinateDetection(label_decoder, img, args.debug)
    imgload_elapsed_time = time.time() - _start

    _start = time.time()
    cd.rotate()
    rotate_elapsed_time = time.time() - _start

    _start = time.time()
    cd.get_edge_location()
    edge_elapsed_time = time.time() - _start

    # return (cd.edge0, cd.edge1)

    _start = time.time()
    cd.get_labels()
    coord_elapsed_time = time.time() - _start

    elapsed_time = time.time() - start

    if args.time_log:
        print(f"label load: {labelload_elapsed_time}[sec]")
        print(f"img load: {imgload_elapsed_time}[sec]")
        print(f"rotate: {rotate_elapsed_time}[sec]")
        print(f"edge detection: {edge_elapsed_time}[sec]")
        print(f"convert coordinate: {coord_elapsed_time}[sec]")
        print(f"all process: {elapsed_time}[sec]")

    if args.debug:
        cv2.namedWindow("result", cv2.WINDOW_NORMAL)
        cv2.imshow("result", cd.test_img)
        cv2.waitKey(0)

    return cd.markers


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="detection the coordinate or location in microwellarray image")
    parser.add_argument("-p", "--path_img", help="a path name of an image file")
    parser.add_argument("-t", "--time_log", help="whether time_log output or not", default=False)
    parser.add_argument("-d", "--debug", help="whether debug output or not", default=False)

    args = parser.parse_args()

    main(args)
