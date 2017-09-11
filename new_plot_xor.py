# coding: utf-8
import re
from pylab import *

from matplotlib.font_manager import FontProperties

ff = FontProperties()
ff.set_family('sans-serif')
ff.set_weight('bold')
#ff.set_size('')
    

def get_times(sel_bits, sel_leaves):
    myre = re.compile(r'MultilayerXorTestJob\((\d+),(\d+),(\d+),(\d+),(\d+)\) (\d+) (\d+)\n')
    for row in open("xorlog.txt"):
        yo = myre.match(row)

        if yo is None:
            continue

        bits, leaves, layers, prob, time, = \
            [int(x) for x in [yo.group(1), yo.group(2), yo.group(3), yo.group(6), yo.group(7)]]

        if bits == sel_bits and leaves == sel_leaves:
            yield (1+layers, prob/128.0, time+0.0)

def docurvacc(b, l):
    mydata = array([x for x in get_times(b, l)]).T
    plot(mydata[0], mydata[1], 'b--o', lw=3)
    d = mydata[0,-1]

    xx = {
        (4,128):0.1,
        (4,64):0.04,
        (4,32):0.11,
        (4,16):0.05,
        (4,8):-0.1,
        (4,2):0.1,
        (8,128):0.1,
        (8,8):0.1,
        (8,2):-0.1,
        (8,4):0.05,
        (8,64):0.02,
        (8,32):-0.1,
        (8,16):0.05,
    }.get((b,l), 0)
    
    text(mydata[0,-1], mydata[1,-1]+xx, u'{}×{}L={}'.format(l, int(d), int(l * 2**(d-1))), fontproperties=ff)

def docurve(b, l):
    mydata = array([x for x in get_times(b, l)]).T
    #plot(mydata[0], mydata[1], '--o', lw=3)
    #twinx()
    plot(mydata[0], mydata[2], 'r--o', lw=3)
    d = mydata[0,-1]

    xx = {
        (4,4):10,
        (4,32):100,
        (4,16):-50,
        (8,4):35000,
        (8,8):-40000,
        (8,2):-4000,
        (8,32):-5000,
        (8,64):8000,
    }.get((b,l), 0)
    
    text(mydata[0,-1], mydata[2,-1]+xx, u'{}×{}L={}'.format(l, int(d), int(l * 2**(d-1))), fontproperties=ff)


ion()
figure(figsize=(8,8))
subplot(2,2,1)
b=4
title('Training efficacy for {}-bit'.format(b))
for l in [2,4,8,16,32,64,128]:
    docurvacc(b,l)
grid()
ylim(-0.1, 1.19)
xlim(0.7, 12)
ylabel("Training success rate")

subplot(2,2,2)
b=8
title('Training efficacy for {}-bit'.format(b))
for l in [2,4,8,16,32,64,128]:
    docurvacc(b,l)
grid()
ylim(-0.1, 1.19)
xlim(0.7, 12)

subplot(2,2,3)
semilogy()
b=4
title('Training times for {}-bit'.format(b))
for l in [2,4,8,16,32,64,128]:
    docurve(b,l)
grid()
xlim(0.7, 12)
xlabel("Number of layers (L)")
ylabel("Training time [s]")

subplot(2,2,4)
semilogy()
b=8
title('Training times for {}-bit'.format(b))
for l in [2,4,8,16,32,64,128]:
    docurve(b,l)
grid()
xlim(0.7, 12)
xlabel("Number of layers (L)")

suptitle("ConANN performance on the XOR function")


tight_layout(rect=[0.03, 0.03, 0.95, 0.97])
