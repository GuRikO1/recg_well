import cv2
import numpy as np
import argparse
from typing import Tuple
import matplotlib.pyplot as plt

EDGE_PX = 46
WELL_PX = EDGE_PX * 4
MARK_BITS = 2


def bigger(arr):
    return arr > np.mean(arr) + np.std(arr)


def calc_grad(img: np.ndarray) -> (float, float):
    sobel_x_img = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y_img = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    return sobel_x_img, sobel_y_img


def calc_norm(img: np.ndarray) -> (float, float):
    grad_x, grad_y = calc_grad(img)
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


def compress_edge(edge):
    if edge.shape[0] & 1 == 0:
        edge = np.logical_and(edge[::2], edge[1::2])
    else:
        edge = np.logical_and(edge[:-1:2], edge[1::2])
    return edge


def to_two_times_angle(trigonometrics: Tuple[float, float]) -> (float, float):
    cos, sin = trigonometrics
    return 2 * cos ** 2 - 1, 2 * cos * sin


def rotate(img: np.ndarray) -> np.ndarray:
    cos4, sin4 = to_two_times_angle(to_two_times_angle(calc_norm(img)))
    n_x = np.mean(cos4)
    n_y = np.mean(sin4)
    four_theta = np.arctan2(n_y, n_x)
    theta = np.rad2deg(four_theta * 0.25)
    print('theta = {}'.format(theta))

    centroid = (np.array(img.shape) / 2).astype(int)
    m = cv2.getRotationMatrix2D((centroid[0], centroid[1]), theta, 1)

    rotated_img = cv2.warpAffine(img, m, img.shape)
    return rotated_img


def is_edge(grad_x, grad_y):
    return np.logical_and(bigger(grad_x), grad_x > grad_y * 4)


def get_edge_location(img, EDGE_PX, WELL_PX):
    edge_mask = np.ones((EDGE_PX + WELL_PX,), dtype=int)
    edge_mask[WELL_PX:-WELL_PX] = -1
    valid_area = img > 0
    valid_area = np.logical_and(valid_area[2:], valid_area[:-2])
    valid_area = np.logical_and(valid_area[:, 2:], valid_area[:, :-2])

    def inner(edge):
        for _ in range(2):
            edge = compress_edge(edge)
        # edge = np.max(edge, axis=0)
        # edge = edge.astype(int)
        edge = np.sum(edge.astype(int), axis=0)
        edge[edge == 0] = -1
        print(np.convolve(edge, edge_mask, 'valid'))
        fig = plt.figure(figsize=(14, 4))
        ax1 = fig.add_subplot(1, 1, 1)
        ax1.plot(np.convolve(edge, edge_mask, 'valid'))
        plt.show()
        return np.argmax(np.convolve(edge, edge_mask, 'valid'))

    grad_x, grad_y = calc_grad(img)
    edge_v = is_edge(grad_x, grad_y)
    edge_h = is_edge(grad_y, grad_x)
    edge_v = np.logical_and(edge_v[1:-1, 1:-1], valid_area)
    edge_h = np.logical_and(edge_h[1:-1, 1:-1], valid_area)

    return (inner(edge_v), inner(edge_h.transpose()))


def is_mark(img):
    std_x = np.sum(np.std(img, axis=0))
    std_y = np.sum(np.std(img, axis=1))
    return std_x * 2 > std_y


def main(args):
    gray_img = cv2.imread(args.path_img, cv2.IMREAD_GRAYSCALE)
    # cv2.imshow('original', gray_img)

    blur_img = cv2.GaussianBlur(gray_img, (5, 5), 2)
    img = rotate(blur_img)
    # cv2.imshow('rotated', img)
    px = get_edge_location(img, EDGE_PX, WELL_PX)
    print(px)

    img = img[px[1]:, px[0]:]
    img = img[:EDGE_PX + WELL_PX, :EDGE_PX + WELL_PX]
    cv2.imshow('cropped', img)

    edges = [
        img[:EDGE_PX].transpose(),
        img[-EDGE_PX:].transpose(),
        img[:, :EDGE_PX],
        img[:, -EDGE_PX:]]

    print('top, bottom, left, right')

    delta = WELL_PX / MARK_BITS
    for i, edge in enumerate(edges):
        for j in range(MARK_BITS):
            left = int(delta * (j + 0.5))
            mark = edge[left:][:EDGE_PX]
            grad_x, grad_y = calc_grad(mark)
            b = is_mark(mark)
            print(b)
            cv2.imshow('{}_{}_{}'.format(i, j, b), mark)

    cv2.waitKey(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="detection the coordinate or location in microwellarray image")
    parser.add_argument("-p", "--path_img", help="a path name of an image file")

    args = parser.parse_args()

    main(args)
