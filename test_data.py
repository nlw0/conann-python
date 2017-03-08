from pylab import *

import numpy.random


def plot_stuff(x_positive, x_negative, lines_homog):
    xp = vstack((x_positive, ones((1, x_positive.shape[1]))))
    xn = vstack((x_negative, ones((1, x_negative.shape[1]))))
    acTP = zeros(xp.shape[1])
    acFP = zeros(xp.shape[1])
    acFN = zeros(xp.shape[1]) == 0
    acTN = zeros(xp.shape[1]) == 0

    for lllt in lines_homog:
        lll = lllt.T
        acTP = acTP + (sum(dot(lll, xp) > 0, axis=0) == lll.shape[0])
        acFP = acFP + (sum(dot(lll, xn) > 0, axis=0) == lll.shape[0])
        acTN = acTN * (sum(dot(lll, xn) > 0, axis=0) < lll.shape[0])
        acFN = acFN * (sum(dot(lll, xp) > 0, axis=0) < lll.shape[0])

    TP = sum(acTP > 0)
    FP = sum(acFP > 0)
    TN = sum(acTN)
    FN = sum(acFN)

    print("True Positives: {}".format(TP))
    print("False Positives: {}".format(FP))
    print("True Negatives: {}".format(TN))
    print("False Negatives: {}".format(FN))
    print()
    print("Precision: {:.2f}%".format(100 * TP / (TP + FP)))
    print("Recall: {:.2f}%".format(100 * TP / (TP + FN)))


if __name__ == '__main__':

    data = load(sys.argv[1])
    solution_filename = sys.argv[2]

    x_positive = data['x_positive']
    x_negative = data['x_negative']
    lines_homog = data['lines_homog']
    test_positive = data['test_positive']
    test_negative = data['test_negative']

    whoinll = open(solution_filename).readlines()

    possible = [x for x in whoinll if x != "0\n"]
    impossible = [x for x in whoinll if x == "0\n"]
    print("{} positive training samples were left behind".format(len(impossible)))

    nsets = len(possible)

    lines_sel = []
    for nn, row in enumerate(possible):
        whoin = array([int(x) for x in row.split(" ")], dtype=int)
        sel = whoin[whoin > 0] - 1
        lines_sel.append(lines_homog[:, sel])

    print("\n=== TRAIN ===")
    plot_stuff(x_positive, x_negative, lines_sel)

    print("\n=== TEST ===")
    plot_stuff(test_positive, test_negative, lines_sel)
