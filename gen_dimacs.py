from pylab import *


def calc_line_projections(lines, sample):
    xx = vstack((sample, ones((1, sample.shape[1]))))
    return dot(lines.T, xx).T


if __name__ == '__main__':

    data_filename = sys.argv[1]
    data = load(data_filename)

    x_positive = data['x_positive']
    x_negative = data['x_negative']
    lines_homog = data['lines_homog']

    fn_pos = data_filename + "-soft.dimacs"
    fp = open(fn_pos, 'w')

    # Positive samples
    for row in calc_line_projections(lines_homog, x_positive):
        fp.write(" ".join("%d" % int(1 + x) for x in nonzero(row < 0)[0]) + "\n")

    fn_neg = data_filename + "-hard.dimacs"
    fp = open(fn_neg, 'w')

    # Negative samples
    for row in calc_line_projections(lines_homog, x_negative):
        fp.write(" ".join("%d" % int(1 + x) for x in nonzero(row < 0)[0]) + "\n")
