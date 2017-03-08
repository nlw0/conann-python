from pylab import *

import numpy.random


def plot_line(ll):
    a, b, c = ll

    ax = array([-3, 3])
    ay = - c / b - a / b * ax
    by = array([-3, 3])
    bx = - c / a - b / a * by

    x = array([0.0, 0.0])
    y = array([0.0, 0.0])

    x = []
    y = []

    if abs(ax[0]) <= 3 and abs(ay[0]) <= 3:
        x.append(ax[0])
        y.append(ay[0])
    if abs(bx[0]) <= 3 and abs(by[0]) <= 3:
        x.append(bx[0])
        y.append(by[0])
    if abs(ax[1]) <= 3 and abs(ay[1]) <= 3:
        x.append(ax[1])
        y.append(ay[1])
    if abs(bx[1]) <= 3 and abs(by[1]) <= 3:
        x.append(bx[1])
        y.append(by[1])

    plot(x, y, 'k--', lw=1)


if __name__ == '__main__':

    Nin = 500
    Nout = 500

    Nl = 1000

    lim = 2.0
    pla = 0.9
    # margin = 0.8
    ring_size = 0.9

    asc = pla * numpy.random.uniform(size=Nin)**0.5

    ring_dist = lim - ring_size / 2
    ring_width = ring_size / 2 * sqrt(numpy.random.uniform(size=Nout))

    tt1 = numpy.random.uniform(size=Nout)
    tt2 = numpy.random.uniform(size=Nout)
    tt3 = numpy.random.uniform(size=Nout)

    x_inliers = asc * vstack((sin(2 * pi * tt1), cos(2 * pi * tt1)))
    x_outliers = ring_dist * vstack((sin(2 * pi * tt2), cos(2 * pi * tt2))) + \
        ring_width * vstack((sin(2 * pi * tt3), cos(2 * pi * tt3)))

    # x_inliers, x_outliers = x_outliers, x_inliers

    cc = hstack((x_inliers, x_outliers))
    print(mean(cc, axis=1))
    print(std(cc, axis=1))

    ttl = numpy.random.uniform(size=Nl)
    line_directions = vstack((sin(2 * pi * ttl), cos(2 * pi * ttl)))
    lines_homog = vstack((line_directions, 3 * (numpy.random.uniform(size=Nl) * 2 - 1.0)))

    savez('saturn_data', x_inliers=x_inliers, x_outliers=x_outliers, lines_homog=lines_homog)

    ion()
    plot(x_inliers[0], x_inliers[1], '.')
    plot(x_outliers[0], x_outliers[1], '.')

    for ll in lines_homog.T:
        plot_line(ll)

    grid()
    axis('equal')
    xlim(-3, 3)
    ylim(-3, 3)
