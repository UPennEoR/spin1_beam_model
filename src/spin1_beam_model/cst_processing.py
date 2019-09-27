import os

import h5py
import numba as nb
import numpy as np
import ssht_numba as sshtn


@nb.njit
def ssht_power_spectrum(f_lm):
    L = int(np.sqrt(f_lm.size))
    C_l = np.zeros(L)
    for el in range(L):
        for m in range(-el, el + 1):
            ind = sshtn.elm2ind(el, m)
            C_l[el] += np.abs(f_lm[ind]) ** 2.0
        C_l[el] /= 2 * el + 1
    return C_l


def elm2ind(el, m):
    """Same as pyssht.elm2ind, but will broadcast numpy arrays"""
    return el * el + el + m


def reflected_extension(Elm_in):
    Nfreq, Nmode = Elm_in.shape

    Elm_r = np.zeros((2 * Nfreq - 2, Nmode), dtype=np.complex128)

    Elm_r[:Nfreq, :] = np.copy(Elm_in)
    Elm_r[Nfreq:, :] = np.flip(np.copy(Elm_in[1:-1, :]), axis=0)
    return Elm_r


class CSTDataProcessor(object):
    """
    Processes a set of CST E-Field text files to derive the spin- +-1 harmonic
    coefficients that define a global beam model on the frequency domain
    supported by the input data.

    The fields are not normalized here, they are to be normalized once a global
    model is fit to these data.

    Inputs:
    data_files: a list of strings, sorted by frequency, where each element is
        the full path to the CST text file with the E-Field data at that
        frequency.
    data_freqs: a numpy array of floats, with elements corresponding to the
        frequency in MHz of the file in the data_files list.
    spatial_sampling: the equiangular resolution of the data files, in degrees.
    data_uncertainty: the fractional accuracy of the input CST data.
        Example:
            If the CST solver was run with an accuracy setting of 40dB, then
            data_uncertainty should be 1e-4.

    Assumes that all input files have the same spatial sampling resolution.
    """

    def __init__(
        self,
        data_files,
        data_freqs,
        spatial_sampling,
        data_uncertainty,
        zetnith_theta=0.0,
        zenith_phi=0.0,
        data_source_notes_string="left blank by user",
    ):

        # check data_files paths for validity
        for data_file in data_files:
            if os.path.exists(data_file) is not True:
                error_message = "The path {} does not exist.".format(data_file)
                raise IOError(error_message)

        self.data_source_notes_string = data_source_notes_string

        self.data_files = data_files
        self.data_freqs = data_freqs

        self.data_uncertainty = data_uncertainty

        self.Nfreq = data_freqs.size

        data_file = np.loadtxt(data_files[0], skiprows=2)

        self.Ntheta = int(181.0 / spatial_sampling)
        self.Nphi = int(360.0 / spatial_sampling)

        self.theta_data = np.reshape(
            np.radians(data_file[:, 0]), (self.Nphi, self.Ntheta)
        ).T
        self.phi_data = np.reshape(
            np.radians(data_file[:, 1]), (self.Nphi, self.Ntheta)
        ).T

        self.L_data = self.Ntheta - 1  # spatial harmonic bandlimit of the data

        self.zenith_theta = zetnith_theta
        self.zenith_phi = zenith_phi

        self.overwrite = False

    def compute_spin1_harmonics(self, filter_cut_factor=1.0):

        self.pos1_Elm_nodes = np.zeros(
            (self.Nfreq, self.L_data ** 2), dtype=np.complex128
        )
        self.neg1_Elm_nodes = np.zeros(
            (self.Nfreq, self.L_data ** 2), dtype=np.complex128
        )
        self.zenith_Eabs = np.zeros(self.Nfreq, dtype=np.float64)
        sshtn.generate_dl(np.pi / 2.0, self.L_data)

        for ii in range(self.Nfreq):
            data = np.loadtxt(self.data_files[ii], skiprows=2)

            # phase is conjugated because the data is for an outward propagating wave
            Et = np.reshape(
                data[:, 3] * np.exp(-1j * np.radians(data[:, 4])),
                (self.Nphi, self.Ntheta),
            ).T
            Ep = np.reshape(
                data[:, 5] * np.exp(-1j * np.radians(data[:, 6])),
                (self.Nphi, self.Ntheta),
            ).T

            Et = np.ascontiguousarray(Et)
            Ep = np.ascontiguousarray(Ep)

            pos1_E = (Et + 1j * Ep) / np.sqrt(2.0)
            neg1_E = (Et - 1j * Ep) / np.sqrt(2.0)

            pos1_Elm = np.empty(self.L_data * self.L_data, dtype=np.complex128)
            sshtn.mw_forward_sov_conv_sym_ss(pos1_E, self.L_data, 1, pos1_Elm)

            neg1_elm = np.empty(self.L_data * self.L_data, dtype=np.complex128)
            sshtn.mw_forward_sov_conv_sym_ss(neg1_e, self.L_data, -1, neg1_Elm)

            # pos1_Elm = pyssht.forward(
            #     pos1_E, self.L_data, Spin=1, Method="MWSS", Reality=False
            # )
            # neg1_Elm = pyssht.forward(
            #     neg1_E, self.L_data, Spin=-1, Method="MWSS", Reality=False
            # )

            self.pos1_Elm_nodes[ii] = pos1_Elm
            self.neg1_Elm_nodes[ii] = neg1_Elm

        pos1_Elm_ex = reflected_extension(self.pos1_Elm_nodes)
        neg1_Elm_ex = reflected_extension(self.neg1_Elm_nodes)

        pos1_Elm_ft = np.fft.fft(pos1_Elm_ex, axis=0)
        neg1_Elm_ft = np.fft.fft(neg1_Elm_ex, axis=0)

        Nft_mode = 2 * self.Nfreq - 2

        pCl_tau = np.zeros((Nft_mode, self.L_data))
        for ii in range(Nft_mode):
            pCl_tau[ii] = ssht_power_spectrum(pos1_Elm_ft[ii])

        nCl_tau = np.zeros((Nft_mode, self.L_data))
        for ii in range(Nft_mode):
            nCl_tau[ii] = ssht_power_spectrum(neg1_Elm_ft[ii])

        Cl_tau = pCl_tau + nCl_tau
        max_val = np.amax(Cl_tau)

        eps = self.data_uncertainty

        mode_filter = Cl_tau / (Cl_tau + (eps ** 2.0 * max_val))
        cut_inds = np.where(mode_filter < (filter_cut_factor * eps))
        mode_filter[cut_inds] = 0.0

        L_model = self.L_data
        for ii in range(self.L_data):
            # for each possible bandlimit (decending from L_data),
            # check if all modes are zero
            if np.count_nonzero(mode_filter[:, -(ii + 1)]) == 0:
                L_model = self.L_data - ii
            else:
                # the bandlimit of the filtered modes has been found, so
                break

        pos1_Elm_ft_s = np.copy(pos1_Elm_ft)
        neg1_Elm_ft_s = np.copy(neg1_Elm_ft)

        for ii in range(Nft_mode):
            for ell in range(self.L_data):
                m = np.arange(-ell, ell + 1)
                indices = elm2ind(ell, m)
                pos1_Elm_ft_s[ii, indices] *= mode_filter[ii, ell]
                neg1_Elm_ft_s[ii, indices] *= mode_filter[ii, ell]

        self.pos1_Elm_s = np.fft.ifft(pos1_Elm_ft_s[:, : L_model ** 2], axis=0)[
            : self.Nfreq
        ]
        self.neg1_Elm_s = np.fft.ifft(neg1_Elm_ft_s[:, : L_model ** 2], axis=0)[
            : self.Nfreq
        ]

        self.L_model = L_model
        self.filter_cut_factor = filter_cut_factor

    def write_model_data(self, save_dir, save_name):

        full_file_name = os.path.join(save_dir, save_name + ".h5")

        if not self.overwrite and os.path.exists(full_file_name):
            raise ValueError("A save file with that name already exists.")

        with h5py.File(full_file_name, "w") as h5f:

            h5f.create_dataset("zenith_theta", data=self.zenith_theta)
            h5f.create_dataset("zenith_phi", data=self.zenith_phi)
            h5f.create_dataset("spatial_bandlimit", data=self.L_model)
            h5f.create_dataset("filter_cut_factor", data=self.filter_cut_factor)
            h5f.create_dataset("data_uncertainty", data=self.data_uncertainty)

            data_source_notes_string = np.string_(self.data_source_notes_string)
            h5f.create_dataset("data_source_notes", data=data_source_notes_string)

            h5f.create_dataset("pos1_Elm", data=self.pos1_Elm_s)
            h5f.create_dataset("neg1_Elm", data=self.neg1_Elm_s)
            h5f.create_dataset("frequencies", data=self.data_freqs)
