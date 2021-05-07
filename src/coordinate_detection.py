import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple


MARK_BITS: int = 3
WELL_PX_MIN: int = 500
WELL_PX_MAX: int = 540
THRESHOLD_MARKER_CHECKER: float = 1.8
bit_pow: List[int] = [2 ** i for i in range(MARK_BITS)]


class CoordinateDetection():
    def __init__(self, label_decoder, img, debug, smooth_window=20, min_interval=50):
        self.label_decoder = label_decoder
        # self.thresh_img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11,2)
        _, self.thresh_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        self.blur_img = cv2.GaussianBlur(img, (7, 7), 2)
        self.img_width, self.img_height = self.thresh_img.shape[:2]
        self.debug = debug
        self.smooth_window = smooth_window
        self.min_interval = min_interval
        self.markers =[]

    def _remove_outlier(self, lst: List[float], sigma: float =1.0) -> List[float]:
        m = np.median(lst)
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
        n_x = np.mean(self._remove_outlier(cos4))
        n_y = np.mean(self._remove_outlier(sin4))
        four_theta = np.arctan2(n_y, n_x)
        theta = np.rad2deg(four_theta * 0.25)

        centroid = np.array(self.thresh_img.shape) // 2
        m = cv2.getRotationMatrix2D(tuple(centroid), theta, 1)

        self.rotate_img = cv2.warpAffine(self.thresh_img, m, (self.img_height, self.img_width),
                                         borderValue=(255, 255, 255))
        return

    def _get_smooth_edge(self, smooth):
        diff = 6
        # threshold_high, threshold_low = 120, 80
        edge = []
        flag = 0
        min_smooth = np.min(smooth)
        smooth = (smooth - min_smooth) / (np.max(smooth) - min_smooth) * 255

        for ind in range(len(smooth) - 4):
            if flag == 0 and smooth[ind + 4] - smooth[ind + 2] < -diff and smooth[ind + 2] - smooth[ind] < -diff:
                edge.append(ind + 2)
                flag = 1
            elif flag == 1 and smooth[ind + 4] - smooth[ind + 1] > diff and smooth[ind + 2] - smooth[ind] > diff:
                flag = 0

        try:
            interval_group = [[edge[0]]]
            for e in edge[1:]:
                new_group = 1
                for i, ig in enumerate(interval_group):
                    if WELL_PX_MIN <= e - ig[-1] <= WELL_PX_MAX:
                        interval_group[i].append(e)
                        new_group = 0
                        break
                if new_group:
                    interval_group.append([e])
            if len(interval_group) == 1:
                return interval_group[0]

            interval_group.sort(key=lambda x: -len(x))

            min_idx = np.abs(np.asarray(interval_group[0]) - interval_group[1][0]).argmin()

            if interval_group[1][0] - interval_group[0][min_idx] > 0:
                return interval_group[0]
            else:
                return interval_group[1]
        except:
            return []


    # def _remove_non_edge(self, edge, well_px):
    #     if len(edge) <= 1:
    #         return edge
    #     return edge[:-1] if edge[-1] - edge[-2] < 0.9 * well_px else edge

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

        diff_edge0 = [self.edge0[i + 1] - self.edge0[i] for i in range(len(self.edge0) - 1)]
        diff_edge1 = [self.edge1[i + 1] - self.edge1[i] for i in range(len(self.edge1) - 1)]

        try:
            self.well_px = int(np.mean(diff_edge0 + diff_edge1))
            self.edge_px = self.well_px // 4

            if self.well_px < WELL_PX_MIN or self.well_px > WELL_PX_MAX:
                raise Exception
        except:
            self.well_px = None
            self.edge_px = None

        return

    def _is_mark(self, img) -> bool:
        img_x = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        img_y = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        std_x = np.sum(np.std(img_x, axis=0))
        std_y = np.sum(np.std(img_y, axis=1))
        return std_y < THRESHOLD_MARKER_CHECKER * std_x

    def get_labels(self) -> np.ndarray:
        self.test_img = cv2.cvtColor(self.rotate_img, cv2.COLOR_BGR2RGB)

        if self.well_px is None:
            self.markers = []
            return

        if self.debug:
            fig, axs = plt.subplots(len(self.edge0), len(self.edge1))
            ar = axs.ravel()

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
                        b = self._is_mark(mark)
                        self.markers.append(b)

                        if b:
                            cnt += bit_pow[2 - bit] if k < 2 else bit_pow[bit]

                        if self.debug:
                            _ar[MARK_BITS * k + bit].axis('off')
                            _ar[MARK_BITS * k + bit].set_title(b, fontsize=10, pad=-10)
                            _ar[MARK_BITS * k + bit].imshow(mark)

                    cnt_lst.append(cnt)

                label = np.array([cnt_lst[0],cnt_lst[2],cnt_lst[1],cnt_lst[3]])
                coord = self.label_decoder.decode_label(label)
                if coord:
                    cv2.putText(self.test_img, f"({coord[0]}, {coord[1]})", ((y0 + y1) // 2 , (x0 + x1) // 2),
                                cv2.FONT_ITALIC, 1.0, (255, 0, 255), 3)
        if self.debug:
            plt.show()

        return
