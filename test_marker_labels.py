import cv2
import numpy as np
import csv
import matplotlib.pyplot as plt
import argparse
from typing import Tuple, List, Dict
import time
from src.label_decoder import LabelDecoder
from src.coordinate_detection import CoordinateDetection

MARK_BITS: int = 3
bit_pow: List[int] = [2 ** i for i in range(MARK_BITS)]

class TestCoordinateDetection(CoordinateDetection):
    def __init__(self, label_decoder, img, img_name, debug, smooth_window=15, min_interval=70):
        super(TestCoordinateDetection, self).__init__(label_decoder, img, debug, smooth_window, min_interval)
        img_name = img_name.split('/')[2].split('.')[0]
        with open(f'marker_labels/{img_name}.csv', 'r') as f:
            true_label = f.read().rstrip('\n').split(',')
            if len(true_label) == 1:
                self.true_label = []
            else:
                self.true_label = list(map(int, true_label))


    def get_true_labels(self) -> np.ndarray:
        self.test_img = cv2.cvtColor(self.rotate_img, cv2.COLOR_BGR2RGB)

        if self.well_px is None:
            self.markers = []
            return

        if self.debug:
            fig, axs = plt.subplots(len(self.edge0), len(self.edge1))
            ar = axs.ravel()

        ind = 0
        for i, ex in enumerate(self.edge0):
            for j, ey in enumerate(self.edge1):
                ix0 = max(0, ex - self.well_px + self.edge_px)
                ix1 = min(self.img_width, ex)
                iy0 = max(0, ey - self.well_px + self.edge_px)
                iy1 = min(self.img_height, ey)
                cv2.rectangle(self.test_img, (iy0, ix0), (iy1, ix1), (0, 255, 0), 8)

                if ex - self.well_px < 0 or ex + self.edge_px > self.img_width \
                    or ey - self.well_px < 0 or ey + self.edge_px > self.img_height:
                    continue

                x0 = ex - self.well_px
                x1 = ex + self.edge_px
                y0 = ey - self.well_px
                y1 = ey + self.edge_px

                not_label = 0
                cnt_lst = []
                for k in range(4):
                    cnt = 0
                    for bit in range(MARK_BITS):
                        b = self.true_label[ind]
                        ind += 1
                        self.markers.append(b)

                        if b == 1:
                            cnt += bit_pow[2 - bit] if k < 2 else bit_pow[bit]
                        elif b == -1:
                            not_label = 1

                    cnt_lst.append(cnt)

                if not_label:
                    cv2.putText(self.test_img, f"(x, x)", ((y0 + y1) // 2 , (x0 + x1) // 2),
                                cv2.FONT_ITALIC, 1.0, (255, 0, 255), 3)
                else:
                    label = np.array([cnt_lst[0],cnt_lst[2],cnt_lst[1],cnt_lst[3]])
                    coord = self.label_decoder.decode_label(label)
                    if coord:
                        cv2.putText(self.test_img, f"({coord[0]}, {coord[1]})", ((y0 + y1) // 2 , (x0 + x1) // 2),
                                cv2.FONT_ITALIC, 1.0, (255, 0, 255), 3)
        if self.debug:
            plt.show()

        return


def main(args):
    start = time.time()
    label_decoder = LabelDecoder()
    labelload_elapsed_time = time.time() - start

    _start = time.time()
    img = cv2.imread(args.path_img, cv2.IMREAD_GRAYSCALE)
    tcd = TestCoordinateDetection(label_decoder, img, args.path_img, args.debug)
    imgload_elapsed_time = time.time() - _start

    _start = time.time()
    tcd.rotate()
    rotate_elapsed_time = time.time() - _start

    _start = time.time()
    tcd.get_edge_location()
    edge_elapsed_time = time.time() - _start

    # return (cd.edge0, cd.edge1)

    _start = time.time()
    tcd.get_true_labels()
    coord_elapsed_time = time.time() - _start

    elapsed_time = time.time() - start

    if args.time_log:
        print(f"label load: {labelload_elapsed_time}[sec]")
        print(f"img load: {imgload_elapsed_time}[sec]")
        print(f"rotate: {rotate_elapsed_time}[sec]")
        print(f"edge detection: {edge_elapsed_time}[sec]")
        print(f"convert coordinate: {coord_elapsed_time}[sec]")
        print(f"all process: {elapsed_time}[sec]")

    cv2.namedWindow("result", cv2.WINDOW_NORMAL)
    cv2.imshow("result", tcd.test_img)
    cv2.waitKey(0)

    return tcd.markers


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_img')
    parser.add_argument('--time_log', default=False)
    parser.add_argument('--debug', default=False)
    args = parser.parse_args()

    for i in range(1,81):
        with open('./marker_labels/Image_{:05d}_CH4.csv'.format(i)) as f:
            actual = f.read().rstrip('\n').split(',')
            if len(actual) == 1:
                continue
            actual = list(map(int, actual))

        args.path_img = './pictures/Image_{:05d}_CH4.jpg'.format(i)
        print(f"\ntest {args.path_img}")
        main(args)
