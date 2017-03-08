from pylab import *

import numpy.random


def accept_moon(p, width=0.5):
    x, y = p
    if y > 0:
        return abs(norm(p) - 1.0) < width
    else:
        xp = x - 1 if x > 0 else x + 1
        return norm(array([xp, y])) < width


def accept_snail(p, width=0.5):
    x, y = p
    c1 = y > 0 and abs(norm(p) - 1.0) < width
    c2 = y < 0 and norm(array([x + 1, y])) < width
    c3 = x < -1 and norm(array([x + 1, y + 2])) < width
    c4 = y < 0 and x > -1 and abs(norm(array([x + 1, y])) - 2.0) < width
    return c1 or c2 or c3 or c4


def gen_disc(Npts):
    asc = numpy.random.uniform(size=Npts)**0.5
    tt = numpy.random.uniform(size=Npts)
    return asc * vstack((sin(2 * pi * tt), cos(2 * pi * tt)))


def gen_moon(Npts, width=0.5):
    out = []
    while len(out) < Npts:
        xx = (1.0 + width) * gen_disc(2 * (Npts - len(out))).T
        yy = [x for x in xx if accept_moon(x, width)]
        out.extend(yy)
    return array(out[0:Npts]).T


def gen_snail(Npts, width=0.5):
    out = []
    while len(out) < Npts:
        xx = (3.0 + width) * gen_disc(2 * (Npts - len(out))).T
        yy = [x for x in xx if accept_snail(x, width)]
        out.extend(yy)
    return array(out[0:Npts]).T


if __name__ == '__main__':

    Nin = 250
    Nout = 250

    Nl = 100

    ww = 0.4

    x_positive = gen_snail(Nin, ww) + array([[0.5, 0.0]]).T

    MM = array([
        [-1.0, 0.0],
        [0.0, -1.0]
    ])

    x_negative = dot(MM, gen_snail(Nout, ww)) + array([[-0.5, 0.0]]).T

    cc = hstack((x_positive, x_negative))

    print("sample mean:{}".format(mean(cc, axis=1)))
    print("sample std: {}".format(std(cc, axis=1)))

    ttl = numpy.random.uniform(size=Nl)
    line_directions = vstack((sin(2 * pi * ttl), cos(2 * pi * ttl)))
    lines_homog = vstack((line_directions, 3 * (numpy.random.uniform(size=Nl) * 2 - 1.0)))

    savez('moons_data', x_positive=x_positive, x_negative=x_negative, lines_homog=lines_homog)
