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

    plot(x, y, 'k--', lw=1, color='LightGray')


# def inliers(lines_homog, x_inliers):
#     xp = vstack((x_inliers, ones((1, x_inliers.shape[1]))))
#     inp = sum(dot(lll, xp) > 0, axis=0) == lll.shape[0]


def plot_stuff(x_inliers, x_outliers, lines_homog):
    xp = vstack((x_inliers, ones((1, x_inliers.shape[1]))))
    xn = vstack((x_outliers, ones((1, x_outliers.shape[1]))))
    onp = zeros(xp.shape[1]) == 0
    for lllt in lines_homog:
        lll = lllt.T
        for ll in lll:
            plot_line(ll)

        inp = sum(dot(lll, xp) > 0, axis=0) == lll.shape[0]
        onp = onp * (sum(dot(lll, xp) > 0, axis=0) < lll.shape[0])

        inn = sum(dot(lll, xn) > 0, axis=0) == lll.shape[0]
        onn = sum(dot(lll, xn) > 0, axis=0) < lll.shape[0]

        plot(x_inliers[0, inp], x_inliers[1, inp], 'b.')

    plot(x_outliers[0, inn], x_outliers[1, inn], '.', color='Orange')
    plot(x_outliers[0, onn], x_outliers[1, onn], '.', color='Green')

    plot(x_inliers[0, onp], x_inliers[1, onp], 'r.')


def get_hull(inlines):
    sq = array([
        [1, -1, 0.001, 0.001],
        [0.001, 0.001, 1, -1],
        [3, 3, 3, 3]
    ])
    # sq = array([
        # [1, -1, 0.0, 0.0],
        # [0.0, 0.0, 1, -1],
        # [3, 3, 3, 3]
    # ])
    lines = hstack((inlines, sq))
    Nlines = lines.shape[1]
    aa = []
    for i in range(0, Nlines - 1):
        for j in range(i + 1, Nlines):
            a1, b1, c1 = lines[:, i]
            a2, b2, c2 = lines[:, j]

            # x = (c1 / b1 - c2 / b2) / (a2 / b2 - a1 / b1)
            # y = (c1 / a1 - c2 / a2) / (b2 / a2 - b1 / a1)
            x = (c1 * b2 - c2 * b1)
            y = -(c1 * a2 - c2 * a1)
            pt = array([x, y,  (a2 * b1 - a1 * b2)])
            if sum(dot(pt, lines) / pt[2] < 1e-9) <= 2:
                aa.append([pt[0] / pt[2], pt[1] / pt[2]])

    aa = array(aa)

    return array(aa)


if __name__ == '__main__':

    data = load(sys.argv[1])

    x_inliers = data['x_inliers']
    x_outliers = data['x_outliers']
    lines_homog = data['lines_homog']

    print(x_inliers[:, 0])

    ion()
    
    solution_filename = sys.argv[2]

    whoinll = open(solution_filename).readlines()

    possible = [x for x in whoinll if x != "0\n"]
    impossible = [x for x in whoinll if x == "0\n"]
    print("{} impossible points were found".format(len(impossible)))

    nsets = len(possible)
    figure(2, figsize=(8,8))

    ax = None

    
    for nn, row in enumerate(possible):
        if ax is None:
            ax = subplot(3, (nsets + 1) / 3, nn + 1)
        else:
            subplot(3, (nsets + 1) / 3, nn + 1, sharex=ax, sharey=ax)

        whoin = array([int(x) for x in row.split(" ")], dtype=int)
        print(sorted(whoin))
        sel = whoin[whoin > 0] - 1
        lines_sel = lines_homog[:, sel]

        plot_stuff(x_inliers, x_outliers, [lines_sel])
        grid()
        axis('equal')
        xlim(-2.5, 2.5)
        ylim(-2.5, 2.5)

        if len(sel) > 0:
            ll = get_hull(lines_sel)
            lm = mean(ll, axis=0)
            ll = array(sorted(ll, key=lambda x: arctan2(x[1] - lm[1], x[0] - lm[0])))
            ll = vstack((ll, ll[0]))

            plot(ll[:, 0], ll[:, 1], '-', lw=2)

    tight_layout()

    #suptitle("Individual convex regions")

    figure(3, figsize=(6,12))



    ax=subplot(2,1,1)
    title('Sample and base partitions')
    plot_stuff(x_inliers, x_outliers, [lines_homog])
    grid()
    axis('equal')
    xlim(-2.5, 2.5)
    ylim(-2.5, 2.5)

    
    subplot(2,1,2, sharex=ax, sharey=ax)
    title('Union of convex regions')
    xx = []
    for nn, row in enumerate(possible):
        whoin = array([int(x) for x in row.split(" ")], dtype=int)
        print(sorted(whoin))
        sel = whoin[whoin > 0] - 1
        xx.append(lines_homog[:, sel])

        
    plot_stuff(x_inliers, x_outliers, xx)

    for lines_sel in xx:
        ll = get_hull(lines_sel)
        lm = mean(ll, axis=0)
        ll = array(sorted(ll, key=lambda x: arctan2(x[1] - lm[1], x[0] - lm[0])))
        ll = vstack((ll, ll[0]))
        plot(ll[:, 0], ll[:, 1], '-', lw=2, color='Blue')

    grid()
    axis('equal')
    xlim(-2.5, 2.5)
    ylim(-2.5, 2.5)

    tight_layout()
