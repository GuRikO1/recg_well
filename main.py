import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
from typing import Tuple, List, Dict
import time

MARK_BITS: int = 3
bit_pow: List[int] = [2 ** i for i in range(MARK_BITS)]
THRESHOLD_MARKER_CHECKER: float = 1.8
label_decoder = None
WELL_PX_MIN: int = 500
WELL_PX_MAX: int = 550

class LabelDecoder():
    def __init__(self):
        with open('labels.csv', 'r') as f:
            xlabels: List[int] = list(map(int, f.readline().split(',')))
            ylabels: List[int] = list(map(int, f.readline().split(',')))
        shift_bit: int = 2 ** MARK_BITS
        self.pb: np.ndarray = [shift_bit ** i for i in range(4)]
        self.label_map: Dict[int, (int, int)] = {}
        for i in range(32):
            for j in range(32):
                label = np.array([xlabels[32*(j+1)+i], ylabels[32*i+j], xlabels[32*j+i], ylabels[32*(i+1)+j]])
                for _ in range(4):
                    self.label_map[self._hash_label(label)] = (i, j)
                    label = self.rotate(label)


    def _hash_label(self, label: np.ndarray) -> int:
        return np.sum(self.pb * label)

    def _inv(self, n: int) -> int:
        res = 0
        for i in range(MARK_BITS):
            res += ((n >> i) & 1) << (MARK_BITS - i - 1)
        return res

    def rotate(self, label: List[int]) -> List[int]:
        label[0], label[1], label[2], label[3] = self._inv(label[3]), label[0], self._inv(label[1]), label[2]
        return label

    def decode_label(self, label):
        return self.label_map.get(self._hash_label(label))


class CoordinateDetection():
    def __init__(self, img, debug, smooth_window=20, min_interval=70):
        img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11,2)
        self.img = img
        self.debug = debug
        self.img_width, self.img_height = self.img.shape[:2]
        self.blur_img = cv2.GaussianBlur(img, (7, 7), 2)
        self.smooth_window = smooth_window
        self.min_interval = min_interval

    def _remove_outlier(self, lst: List[float], sigma: float =1.0) -> List[float]:
        m = np.mean(lst)
        s = np.std(lst)
        return [t for t in lst if abs(t - m) < sigma * s]

    def _calc_grad(self, img: np.ndarray) -> (float, float):
        sobel_x_img = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y_img = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        return sobel_x_img, sobel_y_img

    def _calc_norm(self, img: np.ndarray) -> (float, float):
        grad_x, grad_y = self._calc_grad(img)
        g_abs2 = grad_x ** 2 + grad_y ** 2
        # valid = bigger(g_abs2)
        valid = g_abs2 > np.mean(g_abs2)
        grad_x = grad_x[valid]
        grad_y = grad_y[valid]
        grad_abs = np.sqrt(grad_x ** 2 + grad_y ** 2)
        grad_abs_inv = 1 / grad_abs
        cos_x = grad_x * grad_abs_inv
        sin_x = grad_y * grad_abs_inv
        return cos_x, sin_x

    def _to_two_times_angle(self, trigonometrics: Tuple[float, float]) -> (float, float):
        cos, sin = trigonometrics
        return 2 * cos ** 2 - 1, 2 * cos * sin

    def rotate(self) -> None:
        cos4, sin4 = self._to_two_times_angle(self._to_two_times_angle(self._calc_norm(self.blur_img)))
        n_x = np.mean(cos4)
        n_y = np.mean(sin4)
        four_theta = np.arctan2(n_y, n_x)
        theta = np.rad2deg(four_theta * 0.25)
        # print(f'theta = {theta}')

        centroid = np.array(self.img.shape) // 2
        m = cv2.getRotationMatrix2D(tuple(centroid), theta, 1)

        self.rotate_img = cv2.warpAffine(self.img, m, (self.img_height, self.img_width),
                                         borderValue=(255, 255, 255))
        return

    def _get_smooth_edge(self, smooth):
        diff = 5
        # threshold_high, threshold_low = 120, 80
        edge = []
        flag = 0

        for ind in range(len(smooth) - 2):
            if flag == 0 and smooth[ind + 2] - smooth[ind + 1] < -diff and smooth[ind + 1] - smooth[ind] < -diff:
                edge.append(ind+1)
                flag = 1
            elif flag == 1 and smooth[ind + 2] - smooth[ind + 1] > diff and smooth[ind + 1] - smooth[ind] > diff:
                flag = 0

        try:
            interval_group = [[edge[0]]]
            for e in edge[1:]:
                new_group = 1
                for i, ig in enumerate(interval_group):
                    if WELL_PX_MIN <= e - ig[-1] <= WELL_PX_MAX:
                        interval_group[i].append(e)
                        new_group = 0
                if new_group:
                    interval_group.append([e])

            interval_group.sort(key=lambda x: -len(x))

            idx = np.abs(np.asarray(interval_group[0]) - interval_group[1][0]).argmin()

            if interval_group[1][0] - interval_group[0][idx] > 0:
                return interval_group[0]
            else:
                return interval_group[1]
        except:
            return []


    def _remove_non_edge(self, edge, well_px):
        if len(edge) <= 1:
            return edge
        return edge[:-1] if edge[-1] - edge[-2] < 0.9 * well_px else edge

    def get_edge_location(self) -> (int, List[int], List[int]):
        self.blur_rotate_img = cv2.GaussianBlur(self.rotate_img, (5, 5), 2)
        _, self.thresh = cv2.threshold(self.blur_rotate_img, 0, 255, cv2.THRESH_OTSU)
        w = np.ones(self.smooth_window) / self.smooth_window
        smooth_ax0: np.ndarray = np.convolve(np.mean(self.blur_rotate_img, axis=1), w, mode='same')
        smooth_ax1: np.ndarray = np.convolve(np.mean(self.blur_rotate_img, axis=0), w, mode='same')

        if self.debug:
            fig = plt.figure(figsize=(14, 4))
            ax1 = fig.add_subplot(1, 2, 1)
            ax2 = fig.add_subplot(1, 2, 2)
            ax1.plot(smooth_ax0)
            ax2.plot(smooth_ax1)
            plt.show()

        self.edge0 = self._get_smooth_edge(smooth_ax0)
        self.edge1 = self._get_smooth_edge(smooth_ax1)

        print(self.edge0, self.edge1)

        diff_edge0 = [self.edge0[i + 1] - self.edge0[i] for i in range(len(self.edge0) - 1)]
        diff_edge1 = [self.edge1[i + 1] - self.edge1[i] for i in range(len(self.edge1) - 1)]
        print(diff_edge0,diff_edge1)
        try:
            self.well_px = int(np.mean(diff_edge0 + diff_edge1))
            self.edge_px = self.well_px // 4

            if self.well_px < WELL_PX_MIN or self.well_px > WELL_PX_MAX:
                raise Exception
        except:
            self.well_px = None

        return

    def _is_mark(self, img) -> bool:
        img_x = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        img_y = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        std_x = np.sum(np.std(img_x, axis=0))
        std_y = np.sum(np.std(img_y, axis=1))
        if self.debug:
            print(std_y, std_x)
        return std_y < THRESHOLD_MARKER_CHECKER * std_x

    def get_labels(self) -> np.ndarray:
        show_img = cv2.cvtColor(self.rotate_img, cv2.COLOR_BGR2RGB)

        if self.debug:
            fig, axs = plt.subplots(len(self.edge0), len(self.edge1))
            ar = axs.ravel()

        print(self.well_px, self.edge_px)
        for i, ex in enumerate(self.edge0):
            for j, ey in enumerate(self.edge1):
                ix0 = max(0, ex - self.well_px + self.edge_px)
                ix1 = min(self.img_width, ex)
                iy0 = max(0, ey - self.well_px + self.edge_px)
                iy1 = min(self.img_height, ey)
                cv2.rectangle(show_img, (iy0, ix0), (iy1, ix1), (0, 255, 0), 8)

                if ex - self.well_px < 0 or ex + self.edge_px > self.img_width \
                    or ey - self.well_px < 0 or ey + self.edge_px > self.img_height:
                    continue

                x0 = ex - self.well_px
                x1 = ex + self.edge_px
                y0 = ey - self.well_px
                y1 = ey + self.edge_px
                crop_img = self.blur_rotate_img[x0:x1, y0:y1]
                delta = self.well_px / MARK_BITS

                if self.debug:
                    ar[len(self.edge1) * i + j].axis('off')
                    ar[len(self.edge1) * i + j].imshow(crop_img)

                    _fig, _axs = plt.subplots(4, MARK_BITS, figsize=(3,5))
                    _ar = _axs.ravel()

                edges = [
                    crop_img[:self.edge_px].transpose(),
                    crop_img[-self.edge_px:].transpose(),
                    crop_img[:, :self.edge_px],
                    crop_img[:, -self.edge_px:]
                ]

                cnt_lst = []
                for k, edge in enumerate(edges):
                    cnt = 0
                    for bit in range(MARK_BITS):
                        left = int(delta * (bit + 0.5))
                        mark = edge[left:][:self.edge_px]
                        print(mark.shape)
                        b = self._is_mark(mark)

                        if b:
                            cnt += bit_pow[2-bit] if k<2 else bit_pow[bit]

                        if self.debug:
                            _ar[MARK_BITS * k + bit].axis('off')
                            _ar[MARK_BITS * k + bit].set_title(b, fontsize=10, pad=-10)
                            _ar[MARK_BITS * k + bit].imshow(mark)

                    cnt_lst.append(cnt)

                # print(cnt_lst[0],cnt_lst[2],cnt_lst[1],cnt_lst[3])
                print(cnt_lst)
                label = np.array([cnt_lst[0],cnt_lst[2],cnt_lst[1],cnt_lst[3]])
                coord = label_decoder.decode_label(label)
                if coord:
                    print(coord)
                    cv2.putText(show_img, f"({coord[0]}, {coord[1]})", ((y0 + y1) // 2 , (x0 + x1) // 2),
                                cv2.FONT_ITALIC, 1.0, (255, 0, 255), 3)

        plt.show()
        return show_img


def main(args):
    start = time.time()
    global label_decoder
    label_decoder = LabelDecoder()
    labelload_elapsed_time = time.time() - start

    _start = time.time()
    img = cv2.imread(args.path_img, cv2.IMREAD_GRAYSCALE)
    cd = CoordinateDetection(img, args.debug)
    imgload_elapsed_time = time.time() - _start

    _start = time.time()
    cd.rotate()
    rotate_elapsed_time = time.time() - _start

    _start = time.time()
    cd.get_edge_location()
    edge_elapsed_time = time.time() - _start

    _start = time.time()
    show_img = cd.get_labels()
    coord_elapsed_time = time.time() - _start

    elapsed_time = time.time() - start

    print(f"label load: {labelload_elapsed_time}[sec]")
    print(f"img load: {imgload_elapsed_time}[sec]")
    print(f"rotate: {rotate_elapsed_time}[sec]")
    print(f"edge detection: {edge_elapsed_time}[sec]")
    print(f"convert coordinate: {coord_elapsed_time}[sec]")
    print(f"all process: {elapsed_time}[sec]")
    cv2.imshow("result", show_img)
    cv2.waitKey(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="detection the coordinate or location in microwellarray image")
    parser.add_argument("-p", "--path_img", help="a path name of an image file")
    parser.add_argument("-d", "--debug", help="whether debug output or not", default=False)

    args = parser.parse_args()

    main(args)
