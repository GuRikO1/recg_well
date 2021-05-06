import os
import numpy as np
from typing import List


MARK_BITS: int = 3
bit_pow: List[int] = [2 ** i for i in range(MARK_BITS)]

class LabelDecoder():
    def __init__(self):
        cwd = os.getcwd()
        with open(f'{cwd}/src/labels.csv', 'r') as f:
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
