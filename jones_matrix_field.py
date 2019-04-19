import h5py

import numpy as np
import numba as nb

# @nb.njit
# def compute_rotation_operator()

class InstrumentalJonesMatrixField(object):
    """
    A model
    """
    def __init__(self, spin1_data_file_path):
        with h5py.File(spin1_data_file_path, 'r') as h5f:
            self.L = h5f['spatial_bandlimit'].value
            self.pos1_Elm_nodes = h5f['pos1_Elm_nodes'].value
            self.neg1_Elm_nodes = h5f['neg1_Elm_nodes'].value
            self.data_freqs = h5f['frequencies'].value

    def derive_symmetric_rotated_feed(self, rotation_angle_sign='negative')):
        if rotation_angle_sign == 'negative':
            rot_angle = -np.pi/2.
        elif rotation_angle_sign == 'positive':
            rot_angle = np.pi/2.
        else:
            raise ValueError("The parameter rotation_orientation must be 'negative' or 'positive'.")

        rotation_operator = np.zeros_like()
        for ii in range(rotation_operator):
            el, m = pyssht.ind2elm(ii)
            arg = m * rot_angle
            rotation_operator[ii] = np.cos(arg) + 1j*np.sin(arg)



    def compute_spline_approximation(self, sampling_factor=1.):
        """
        Compute a cubic spline approximation of the function defined
        by this data.

        Parameters:
        sampling_factor: defines the spatial resolution of the data used to
            derive the spline function. The sampling of the function is defined
            by a band limit, the sampling factor is a multiplicative factor
            for the spatial bandlimit of the data.


        """

        L_use = int(sampling_factor * self.L)
