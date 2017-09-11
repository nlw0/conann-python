# coding: utf-8
import re
from pylab import *


def get_times(sel_bits):
    myre = re.compile(r'MultilayerXorTestJob\((\d+),(\d+),(\d+),(\d+),(\d+)\) (\d+) (\d+)\n')
    for row in open("xorlog.txt"):
        yo = myre.match(row)

        if yo is None:
            continue

        bits, leaves, layers, prob, time, = \
            [int(x) for x in [yo.group(1), yo.group(2), yo.group(3), yo.group(6), yo.group(7)]]

        if bits != sel_bits:
            continue

        yield (leaves, layers, prob, time)

mydata = get_times(8)
#mydata = get_times(4)

print(mydata)






stuph = array([(le, le * 2**la, la, suc, tm) for (le, la, suc, tm) in mydata])

order = argsort(stuph[:, 2])

leaves, tot_leaves, layers, x, y = stuph[order].T

x  = x/128.0

ion()

limi = {1:100, 2:100, 4:100, 8:100, 16:4, 32:2, 64:100, 128:100, 256:100}

for l in set(leaves):
    #sel = (leaves == l) * (layers < limi[l])
    sel = (leaves == l)

    sx = x[sel]
    #sy = log2(y[sel] / 1000.0)
    sy = log2(y[sel])
    slay = layers[sel]
    slea = leaves[sel]
    plot(sx, sy, '--o', lw=2, ms=7)
    print(l)
    print(leaves[sel])
    print(layers[sel])
    print(sx)
    print(sy)
    text(sx[-1] + 0.01, sy[-1], u'{}×{}L={}'.format(
        slea[-1], 1 + slay[-1], slea[-1] * 2**slay[-1]))


# title('4-bit XOR')
#title('ConANN performance on the 4-bit XOR')
#ylabel('Training time [ms]')

title('ConANN performance on the 8-bit XOR')
ylabel('Training time [ks]')

xlabel(u'Training success rate')
xticks([0, 0.25,0.5,0.75,1.0])

#yticks(mgrid[-1:9], ['1/2'] + [x/1000. for x in 2**mgrid[0:9]])
yticks(mgrid[-1:29], 2**mgrid[-1:29]/1000)

xlim(-0.05, 1.25)
#ylim(7, 18)
ylim(8.3, 15.6)
grid()
tight_layout()
