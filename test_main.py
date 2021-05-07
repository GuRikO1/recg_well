import csv
from main import main
import argparse


def test_main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_img')
    parser.add_argument('--time_log', default=False)
    parser.add_argument('--debug', default=False)
    args = parser.parse_args()

    count_all = 0
    count_ok = 0

    for i in range(1,81):
        with open('./marker_labels/Image_{:05d}_CH4.csv'.format(i)) as f:
            actual = f.read().rstrip('\n').split(',')
            if len(actual) == 1:
                continue
            actual = list(map(int, actual))

        args.path_img = './pictures/Image_{:05d}_CH4.jpg'.format(i)
        print(f"\ntest {args.path_img}")
        expected = main(args)
        expected = [int(e) for e in expected]

        _count_ok, _count_non = 0, 0
        if len(expected) != len(actual):
            print("The data length of marker_label does not match.")
        for e,a in zip(expected, actual):
            if a == -1:
                _count_non += 1
            elif e is a:
                _count_ok += 1

        _count_all = len(expected) - _count_non
        count_all += _count_all
        count_ok += _count_ok

        print(f"accracy: {_count_ok / _count_all} ({_count_ok} / {_count_all})")

    print(f"\n*** Test Results for All Dataset ***")

    print(f"accracy: {count_ok / count_all} ({count_ok} / {count_all})")


if __name__ == "__main__":
    test_main()
