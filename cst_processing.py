import os
import h5py

import numpy as np
import numba as nb
import pyssht

import ssht_numba as sshtn

@nb.njit
def Elm_fourier_series(nu_axis, Elm_ft, nu_0, bandwidth):
    """
    Evaluates the fourier series which defines the frequency spectrum
    of each spatial harmonic mode of the model.
    """
    Nfreq = nu_axis.size

    Nft, Nmodes = Elm_ft.shape
    T = 2*bandwidth
    Elm = np.zeros((Nfreq, Nmodes), dtype=nb.complex128)

    k = np.arange(Nft)

    for ii in range(Nfreq):
        nu_i = (nu_axis[ii] - nu_0)
        arg = 2*np.pi * k * nu_i/T
        kernel = np.cos(arg) + 1j*np.sin(arg)
        for jj in range(Nmodes):
            for kk in range(Nft):
                Elm[ii,jj] += Elm_ft[kk,jj] * kernel[kk]

    Elm /= Nft

    return Elm

def periodic_extension(Elm_in):
    Nfreq, Nmode = Elm_in.shape[0]

    Elm_r = np.zeros((2*Nfreq-2, Nmode), dtype=np.complex128)

    Elm_r[:Nfreq,:] = np.copy(Elm_in)
    Elm_r[Nfreq:,:] = np.flip(np.copy(Elm_in[1:-1,:]), axis=0)
    return Elm_r

@nb.njit
def ssht_power_spectrum(f_lm):
    L = int(np.sqrt(f_lm.size))
    C_l = np.zeros(L)
    for el in range(L):
        for m in range(-el, el+1):
            ind = sshtn.elm2ind(el,m)
            C_l[el] += np.abs(f_lm[ind])**2.
        C_l[el] /= (2*el +1)
    return C_l

class CSTDataProcessor(object):
    """
    Processes a set of CST E-Field text files to derive the spin- +-1 harmonic
    coefficients that define a global beam model on the frequency domain
    supported by the input data.

    Inputs:
    data_files: a list of strings, sorted by frequency, where each element is
        the full path to the CST text file with the E-Field data at that
        frequency.
    data_freqs: a numpy array of floats, with elements corresponding to the
        frequency in MHz of the file in the data_files list.
    spatial_sampling: the equiangular resolution of the data files, in degrees.

    Assumes that all input files have the same spatial sampling resolution.
    """
    def __init__(self, data_files, data_freqs, spatial_sampling=1.,
                zetnith_theta=0., zenith_phi=0.,
                data_source_notes_string='left blank by user'):

        # check data_files paths for validity
        for data_file in data_files:
            if os.path.exists(data_file) is not True:
                error_message = 'The path {} does not exist.'.format(data_file)
                raise IOError(error_message)

        self.data_source_notes_string = data_source_notes_string

        self.data_files = data_files
        self.data_freqs = data_freqs

        self.Nfreq = data_freqs.size

        data_file = np.loadtxt(data_files[0], skiprows=2)

        self.Ntheta = int(181./spatial_sampling)
        self.Nphi = int(360./spatial_sampling)

        self.theta_data = np.reshape(np.radians(data_file[:,0]), (self.Nphi, self.Ntheta)).T
        self.phi_data = np.reshape(np.radians(data_file[:,1]), (self.Nphi, self.Ntheta)).T

        self.L = self.Ntheta - 1 # spatial harmonic bandlimit of the data

        self.zenith_theta = zetnith_theta
        self.zenith_phi = zenith_phi

        self.overwrite = False

    def compute_spin1_harmonics(self):

        self.pos1_Elm_nodes = np.zeros((self.Nfreq, self.L**2), dtype=np.complex128)
        self.neg1_Elm_nodes = np.zeros((self.Nfreq, self.L**2), dtype=np.complex128)
        self.zenith_Eabs = np.zeros(self.Nfreq, dtype=np.float64)
        delta = pyssht.generate_dl(np.pi/2., self.L)

        for ii in range(self.Nfreq):
            data = np.loadtxt(self.data_files[ii], skiprows=2)

            # phase is conjugated because the data is for an outward propagating wave
            Et = np.reshape(data[:,3] * np.exp(-1j*np.radians(data[:,4])),
                            (self.Nphi, self.Ntheta)).T
            Ep = np.reshape(data[:,5] * np.exp(-1j*np.radians(data[:,6])),
                            (self.Nphi, self.Ntheta)).T

            Et = np.ascontiguousarray(Et)
            Ep = np.ascontiguousarray(Ep)

            pos1_E = (Et + 1j*Ep)/np.sqrt(2.)
            neg1_E = (Et - 1j*Ep)/np.sqrt(2.)

            pos1_Elm = pyssht.forward(pos1_E, self.L, Spin=1, Method='MWSS', Reality=False)
            neg1_Elm = pyssht.forward(neg1_E, self.L, Spin=-1, Method='MWSS', Reality=False)

            # normalize to make the directivity 1 in the zenith direction
            # note, this is not necessarily the exact peak of the function. But it
            # should be very close, to within the precision of the CST data
            zen_pos1_E = sshtn.ssht_numba_series_eval(pos1_Elm, 1, self.L, delta, np.array([self.zenith_theta]), np.array([self.zenith_phi]))
            zen_neg1_E = sshtn.ssht_numba_series_eval(neg1_Elm, -1, self.L, delta, np.array([self.zenith_theta]), np.array([self.zenith_phi]))

            zen_Et = (zen_pos1_E + zen_neg1_E)/np.sqrt(2.)
            zen_Ep = (zen_pos1_E - zen_neg1_E)*(-1j)/np.sqrt(2.)
            zen_Eabs = np.sqrt(np.abs(zen_Et)**2. + np.abs(zen_Ep)**2.)

            pos1_Elm /= zen_Eabs
            neg1_Elm /= zen_Eabs

            self.pos1_Elm_nodes[ii] = pos1_Elm
            self.neg1_Elm_nodes[ii] = neg1_Elm
            self.zenith_Eabs[ii] = zen_Eabs

    def write_model_data(self, save_dir, save_name):

        full_file_name = os.path.join(save_dir, save_name + '.h5')

        if self.overwrite == False and os.path.exists(full_file_name):
            raise ValueError('A save file with that name already exists.')

        with h5py.File(full_file_name, 'w') as h5f:
            h5f.create_dataset('pos1_Elm_nodes', data=self.pos1_Elm_nodes)
            h5f.create_dataset('neg1_Elm_nodes', data=self.neg1_Elm_nodes)
            h5f.create_dataset('zenith_Eabs', data=self.zenith_Eabs)
            h5f.create_dataset('frequencies', data=self.data_freqs)
            h5f.create_dataset('zenith_theta', data=self.zenith_theta)
            h5f.create_dataset('zenith_phi', data=self.zenith_phi)
            h5f.create_dataset('spatial_bandlimit', data=self.L)

            data_source_notes_string = np.string_(self.data_source_notes_string)
            h5f.create_dataset('data_source_notes', data=data_source_notes_string)
