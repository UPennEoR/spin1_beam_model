import numpy as np, matplotlib.pyplot as plt, healpy as hp
import os


import matplotlib
from mpl_toolkits.axes_grid1 import AxesGrid
import cmocean

def StokesMatrix(n):
    if n not in [0,1,2,3]: raise Exception('Input must be an integer in [0,1,2,3]')

    if n == 0:
        p = np.array([[1.,0],[0.,1.]])
    elif n == 1:
        p = np.array([[1.,0],[0,-1.]])
    elif n == 2:
        p = np.array([[0,1.],[1.,0]])
    elif n == 3:
        p = np.array([[0., -1j],[1j,0]])

    return p

def MuellerMatrixElement(J,i,j):

    Pi = StokesMatrix(i)
    Pj = StokesMatrix(j)

    M_ij = np.einsum('...ab,...bc,...cd,...ad',Pi,J,Pj,J.conj()) / 2.

    M_ij = np.real(M_ij)

    return M_ij

def PlotMueller(jones, rot_angle=90):
    npix = jones.shape[0]
    nside = hp.npix2nside(npix)
    xsize = 1600
    reso = 120*180*np.sqrt(2.)/np.pi /(xsize-1)
    LambProj = hp.projector.AzimuthalProj(xsize=xsize,reso=reso, lamb=True, half_sky=True, rot=[0,rot_angle])

    # Mind the negative sign on the z-coordinate. This came from trial and error for
    # making a particular plot. Its possible it should be removed.
    p2v = lambda x,y,z: hp.vec2pix(nside,x,y,-z)

    logthresh = 4
    linscale = 2
    fig = plt.figure(figsize=(12,12))
    grid = AxesGrid(fig,(1,1,1),
                    nrows_ncols=(4,4),
                    axes_pad=(1.0,0.5),
                    label_mode='all',
                    share_all=False,
                    cbar_location='right',
                    cbar_mode='each',
                    cbar_size='5%',
                    cbar_pad='1%',
                   )
    for i in range(4):
        for j in range(4):
            M_ij = MuellerMatrixElement(jones, i, j)
            img_d = LambProj.projmap(M_ij, p2v)

            if i == j == 0:
                cmap = 'viridis'
                vmin = 0
                vmax = 1

                tick_locs = list(np.linspace(0,1,7, endpoint=True))
                tick_labels = [r'$ < 10^{-6}$',
                               r'$10^{-5}$',
                               r'$10^{-4}$',
                               r'$10^{-3}$',
                               r'$10^{-2}$',
                               r'$10^{-1}$',
                               r'$10^{0}$']

            elif i != j:
                cmap='RdBu_r'
                vmin=-0.05
                vmax=0.05

                d = np.log10(5) * np.diff(np.linspace(vmax*1e-6,vmax,7))[0]
                q = np.linspace(vmax*1e-6,vmax,7)[0::2] - d
                tick_locs = list(np.r_[-np.flipud(q)[:-1],[0], q[1:]])
                tick_labels = [r'$-10^{-2}$',
                               r'$-10^{-4}$',
                               r'$-10^{-6}$',
                               r'$< 5 \times 10^{-8}$',
                               r'$10^{-6}$',
                               r'$10^{-4}$',
                               r'$10^{-2}$']

            else:
                # cmap='RdBu_r'
                cmap=cmocean.cm.delta
                vmin=-1.
                vmax=1

                q = np.linspace(vmax*1e-6, vmax,7)[0::2]
                tick_locs = list(np.r_[-np.flipud(q)[:-1],[0], q[1:]])
                tick_labels = [r'$-10^{0}$',
                               r'$-10^{-2}$',
                               r'$-10^{-4}$',
                               r'$< 10^{-6}$',
                               r'$10^{-4}$',
                               r'$10^{-2}$',
                               r'$10^{0}$']
            n = 4 * i + j
            im = grid[n].imshow(img_d, interpolation='none',
                            cmap=cmap,
                            aspect='equal',
                            vmin=vmin,
                            vmax=vmax,)

            grid[n].set_xticks([])
            grid[n].set_yticks([])

            cbar = grid.cbar_axes[n].colorbar(im, ticks=tick_locs)
            grid.cbar_axes[n].set_yticklabels(tick_labels)

            im.set_norm(matplotlib.colors.SymLogNorm(10**-logthresh,linscale, vmin=vmin,vmax=vmax))
    plt.tight_layout(w_pad=0.5, h_pad=1.0)
    plt.show()
