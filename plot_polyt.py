from pylab import *


def relu(x):
        return clip(x, 0, Inf)


if __name__ == '__main__':

	# qq = loadtxt('breast-cancer-wisconsin.data', delimiter=',')
	qq = loadtxt('xor.data', delimiter=',')

	aa = qq[qq[:,-1]==0]
	bb = qq[qq[:,-1]==1]

	aa += random(aa.shape)*0.2
	bb += random(bb.shape)*0.2

        w1 = array([
                [-0.707, -0.707],
                [0, 1],
                [1, 0],
                [0.707, 0.707],
                [0, -1],
                [-1, 0]
        ])
        b1 = array([[3.0, -3, -2, 1, -2, -2]]).T

        w2 = array([[-1, -1, -1, 0,0,0],
                    [0, 0, 0, -1,-1,-1]])
        b2 = array([[0.5, 0.5]]).T

        w3 = array([[1, 1]])
        b3 = array([[-0.0]]).T
        
        Nrow = 100
        
        xx = mgrid[-Nrow:Nrow+1.0, -Nrow:Nrow+1.0].T.reshape(-1,2).T * 4 / Nrow

        wx1 = dot(w1, xx) + b1
        wx2 = dot(w2, relu(wx1)) + b2
        wx3 = dot(w3, relu(wx2)) + b3

        ffimg = wx3.reshape(Nrow * 2 + 1, -1)
        
        ion()
        figure(0, figsize=(6,6))
        imshow(ffimg, extent=[-4,4,-4,4], cmap=cm.RdBu, vmin=-3, vmax=3)
        axis('equal')
	grid()
