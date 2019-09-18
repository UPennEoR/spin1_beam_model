import h5py
import numba as nb
import numpy as np
import ssht_numba as sshtn
from scipy.interpolate import Rbf, RectBivariateSpline, interp1d


class AntennaFarFieldResponse:
    """
    A model of the farfield response of an antenna specified by the data product
    of a cst_processing.CSTDataProcessor object.

    Frequencies specified in units of MHz.
    """

    def __init__(self, spin1_data_file_path):
        with h5py.File(spin1_data_file_path, "r") as h5f:
            self.pos1_Elm_samples = h5f["pos1_Elm"].value
            self.neg1_Elm_samples = h5f["neg1_Elm"].value
            self.freq_nodes = h5f["frequencies"].value
            self.L_model = h5f["spatial_bandlimit"].value
            self.zenith_theta = h5f["zenith_theta"].value
            self.zenith_phi = h5f["zenith_phi"].value

        self.dual_feed = False

    def derive_symmetric_rotated_feed(self, rotation_angle_sign="positive"):
        """
        Compute the spatial harmonics for an antenna feed identical to the one
        specified by the input data, but rotated by 90 degrees about the z-axis.

        The sign conventions are those of McEwen and Wiaux 2011, Equation 2.

        For the cannonical CST output coordinates, in which:
            -the dipole feed is oriented along 'x'
            -the boresight is along 'z'
            -the 'theta' angle corrdinate is measured away from the 'z' axis,
                i.e. z(theta=0) = 1, and 0 <= theta <= pi
            -the 'phi' azimuthal angle coordinate is measured from 'x' to 'y',
                and 0 <= phi < 2*pi

        the rotated harmonics obtained with rotation_angle_sign='positive'
        correspond to an East-West oriented antenna feed, while the input
        harmonics correspond to a North-South oriented feed.
        """
        if rotation_angle_sign == "negative":
            rot_angle = -np.pi / 2.0
        elif rotation_angle_sign == "positive":
            rot_angle = np.pi / 2.0
        else:
            raise ValueError(
                "The parameter rotation_angle_sign must be 'negative' or 'positive'."
            )

        rotation_operator = np.zeros(self.L_model ** 2, dtype=np.complex128)
        for ii in range(rotation_operator.size):
            el, m = sshtn.ind2elm(ii)
            arg = m * rot_angle
            rotation_operator[ii] = np.cos(arg) - 1j * np.sin(arg)

        self.pos1_rElm_samples = rotation_operator[None, :] * self.pos1_Elm_samples
        self.neg1_rElm_samples = rotation_operator[None, :] * self.neg1_Elm_samples

        self.dual_feed = True

    def interpolate_spatial_harmonics_spectra(self, nu_axis, interp_method="sinc_rbf"):
        """
        Evaluates the spatial harmonics at the frequency points specified
        (in MHz) by nu_axis.

        interp_method:
            -'sinc_rbf' is a radial basis function interpolation using a
             sinc kernel with length scale set by the sampling rate of the input data.
             This effectively assumes that the input data is Nyquist sampled, and aims
             to avoid both 1) additional smoothing beyond what was done in the initial
             processing to derive the model data, or 2) extrapolating beyond what
             the input data supports.

            -'cubic_spline' uses a cubic spline.

        The fits are always over the full input frequency band, even if `nu_axis`
        is only covers a small subset of `freq_nodes`.
        """
        self.nu_axis = nu_axis

        if interp_method == "sinc_rbf":

            # harmonic mean of input sample spacing
            # delta_nu_in = 1./np.mean(1./np.diff(self.freq_nodes))
            delta_nu_in = np.diff(self.freq_nodes)[0]
            print("delta_nu_in is", delta_nu_in)

            def sinc_kernel(self, r):
                tau_c = 1.0 / (2 * self.epsilon)

                r = np.where(r == 0, 1e-20, r)
                y = 2 * np.pi * tau_c * r
                kernel = np.sin(y) / y
                return kernel

            def rbf_obj(data):
                rbf = Rbf(
                    self.freq_nodes,
                    data,
                    function=sinc_kernel,
                    epsilon=delta_nu_in,
                    smooth=0.0,
                )
                return rbf

            self.pos1_Elm = np.zeros(
                (nu_axis.size, self.L_model ** 2), dtype=np.complex
            )
            self.neg1_Elm = np.zeros(
                (nu_axis.size, self.L_model ** 2), dtype=np.complex
            )

            for ii in range(self.L_model ** 2):
                re_pos1_Elm_rbf = rbf_obj(self.pos1_Elm_samples[:, ii].real)
                im_pos1_Elm_rbf = rbf_obj(self.pos1_Elm_samples[:, ii].imag)

                self.pos1_Elm[:, ii] = re_pos1_Elm_rbf(nu_axis) + 1j * im_pos1_Elm_rbf(
                    nu_axis
                )

                re_neg1_Elm_rbf = rbf_obj(self.neg1_Elm_samples[:, ii].real)
                im_neg1_Elm_rbf = rbf_obj(self.neg1_Elm_samples[:, ii].imag)

                self.neg1_Elm[:, ii] = re_neg1_Elm_rbf(nu_axis) + 1j * im_neg1_Elm_rbf(
                    nu_axis
                )

        elif interp_method == "cubic_spline":

            re_pos1_Elm_spl = interp1d(
                self.freq_nodes, self.pos1_Elm_samples.real, kind="cubic", axis=0
            )
            im_pos1_Elm_spl = interp1d(
                self.freq_nodes, self.pos1_Elm_samples.imag, kind="cubic", axis=0
            )

            self.pos1_Elm = re_pos1_Elm_spl(nu_axis) + 1j * im_pos1_Elm_spl(nu_axis)

            ###
            re_neg1_Elm_spl = interp1d(
                self.freq_nodes, self.neg1_Elm_samples.real, kind="cubic", axis=0
            )
            im_neg1_Elm_spl = interp1d(
                self.freq_nodes, self.neg1_Elm_samples.imag, kind="cubic", axis=0
            )

            self.neg1_Elm = re_neg1_Elm_spl(nu_axis) + 1j * im_neg1_Elm_spl(nu_axis)
        else:
            raise ValueError("interp_method must be 'sinc_rbf' or 'cubic_spline'")

        zth = self.zenith_theta
        zph = self.zenith_phi
        delta = sshtn.generate_dl(np.pi / 2.0, self.L_model)

        zen_Eabs = np.zeros(nu_axis.size)
        for ii in range(nu_axis.size):
            zen_pos1_E = ssht_numba_series_eval(
                self.pos1_Elm[ii],
                1,
                self.L_model,
                delta,
                np.array([zth]),
                np.array([zph]),
            )
            zen_neg1_E = ssht_numba_series_eval(
                self.neg1_Elm[ii],
                -1,
                self.L_model,
                delta,
                np.array([zth]),
                np.array([zph]),
            )

            zen_Et = (zen_pos1_E + zen_neg1_E) / np.sqrt(2.0)
            zen_Ep = (zen_pos1_E - zen_neg1_E) * (-1j) / np.sqrt(2.0)
            zen_Eabs[ii] = np.sqrt(np.abs(zen_Et) ** 2.0 + np.abs(zen_Ep) ** 2.0)

        self.pos1_Elm /= zen_Eabs[:, None]
        self.neg1_Elm /= zen_Eabs[:, None]

        if self.dual_feed:
            if interp_method == "sinc_rbf":

                self.pos1_rElm = np.zeros(
                    (nu_axis.size, self.L_model ** 2), dtype=np.complex
                )
                self.neg1_rElm = np.zeros(
                    (nu_axis.size, self.L_model ** 2), dtype=np.complex
                )

                for ii in range(self.L_model ** 2):
                    re_pos1_rElm_rbf = rbf_obj(self.pos1_rElm_samples[:, ii].real)
                    im_pos1_rElm_rbf = rbf_obj(self.pos1_rElm_samples[:, ii].imag)

                    self.pos1_rElm[:, ii] = re_pos1_rElm_rbf(
                        nu_axis
                    ) + 1j * im_pos1_rElm_rbf(nu_axis)

                    re_neg1_rElm_rbf = rbf_obj(self.neg1_rElm_samples[:, ii].real)
                    im_neg1_rElm_rbf = rbf_obj(self.neg1_rElm_samples[:, ii].imag)

                    self.neg1_rElm[:, ii] = re_neg1_rElm_rbf(
                        nu_axis
                    ) + 1j * im_neg1_rElm_rbf(nu_axis)

            elif interp_method == "cubic_spline":

                re_pos1_rElm_spl = interp1d(
                    self.freq_nodes, self.pos1_rElm_samples.real, kind="cubic", axis=0
                )
                im_pos1_rElm_spl = interp1d(
                    self.freq_nodes, self.pos1_rElm_samples.imag, kind="cubic", axis=0
                )

                self.pos1_rElm = re_pos1_rElm_spl(nu_axis) + 1j * im_pos1_rElm_spl(
                    nu_axis
                )

                ###
                re_neg1_rElm_spl = interp1d(
                    self.freq_nodes, self.neg1_rElm_samples.real, kind="cubic", axis=0
                )
                im_neg1_rElm_spl = interp1d(
                    self.freq_nodes, self.neg1_rElm_samples.imag, kind="cubic", axis=0
                )

                self.neg1_rElm = re_neg1_rElm_spl(nu_axis) + 1j * im_neg1_rElm_spl(
                    nu_axis
                )

            else:
                raise ValueError("interp_method must be 'sinc_rbf' or 'cubic_spline'")

            zen_rEabs = np.zeros(nu_axis.size)
            for ii in range(nu_axis.size):
                zen_pos1_rE = ssht_numba_series_eval(
                    self.pos1_rElm[ii],
                    1,
                    self.L_model,
                    delta,
                    np.array([zth]),
                    np.array([zph]),
                )
                zen_neg1_rE = ssht_numba_series_eval(
                    self.neg1_rElm[ii],
                    -1,
                    self.L_model,
                    delta,
                    np.array([zth]),
                    np.array([zph]),
                )

                zen_rEt = (zen_pos1_rE + zen_neg1_rE) / np.sqrt(2.0)
                zen_rEp = (zen_pos1_rE - zen_neg1_rE) * (-1j) / np.sqrt(2.0)
                zen_rEabs[ii] = np.sqrt(np.abs(zen_rEt) ** 2.0 + np.abs(zen_rEp) ** 2.0)

            self.pos1_rElm /= zen_rEabs[:, None]
            self.neg1_rElm /= zen_rEabs[:, None]

    def compute_spatial_spline_approximations(
        self, nu_axis, L_synth="model", interp_method="sinc_rbf"
    ):
        """
        Compute a 2D cubic spline approximation of the elements of the components
        of the Jones matrix at each specified frequency.

        Parameters:
        nu_axis: the frequencies at which to compute the spatial 2D spline
            approximations.
        L_synth: an angular bandlimit that defines the spatial resolution of
            the data used to derive the spline function.

        """
        if not self.dual_feed:
            raise ValueError(
                "Only data for a single feed is set, "
                "this method requires data for dual feeds."
            )

        self.interpolate_spatial_harmonics_spectra(nu_axis, interp_method=interp_method)

        if L_synth == "model":
            L_synth = self.L_model
        if L_synth < self.L_model:
            # there nothing wrong in principle with L_synth < L_model, but
            # the code doesn't support it right now (see pad_Elm function def).
            raise ValueError(
                "The synthesized bandlimit (L_synth) must be higher than the bandlimit "
                "of the model (L_synth > L_model)."
            )

        theta_axis, phi_axis = sshtn.mwss_sample_positions(L_synth)
        mu_axis = np.cos(theta_axis)

        mu_axis_flip = np.flipud(mu_axis)
        phi_axis_pad = np.r_[phi_axis, np.array([2 * np.pi])]

        Nfreq = self.nu_axis.size

        def pad_Elm(Elm, L_padded):
            """
            Pad an array of spatial harmonic modes with zeros, for use in fast
            synthesis of arbitrarily high resolution regularly gridded maps.
            """
            L_in = int(np.sqrt(Elm.size))
            Elm_padded = np.zeros(L_padded ** 2, dtype=np.complex128)
            Elm_padded[: L_in ** 2] = Elm
            return Elm_padded

        def flipped_and_periodic_padded(E):
            E_flip = np.flip(E, axis=0)
            periodic_padding = E_flip[:, 0].reshape(-1, 1)
            E_fliped_and_padded = np.append(E_flip, periodic_padding, 1)

            return E_fliped_and_padded

        def spline_from_data(grid_data):
            grid_data_use = flipped_and_periodic_padded(grid_data)
            spl = RectBivariateSpline(
                mu_axis_flip,
                phi_axis_pad,
                grid_data_use,
                bbox=[-1.0, 1.0, 0.0, 2 * np.pi],
                kx=3,
                ky=3,
                s=0,
            )
            return spl

        def splines_from_harmonics(pos1_Elm, neg1_Elm, L_synth):
            pos1_Elm_pad = pad_Elm(pos1_Elm, L_synth)
            neg1_Elm_pad = pad_Elm(neg1_Elm, L_synth)

            pos1_E = np.empty([L_synth + 1, 2 * L_synth], dtype=np.complex)
            sshtn.mw_inverse_sov_sym_ss(pos1_Elm_pad, L_synth, s=1, f=pos1_E)

            neg1_E = np.empty([L_synth + 1, 2 * L_synth], dtype=np.complex)
            sshtn.mw_inverse_sov_sym_ss(neg1_Elm_pad, L_synth, s=1, f=neg1_E)

            Et = (pos1_E + neg1_E) / np.sqrt(2.0)
            Ep = (pos1_E - neg1_E) * (-1j) / np.sqrt(2.0)

            re_Et_spl, im_Et_spl = [spline_from_data(f(Et)) for f in [np.real, np.imag]]

            re_Ep_spl, im_Ep_spl = [spline_from_data(f(Ep)) for f in [np.real, np.imag]]

            return ((re_Et_spl, im_Et_spl), (re_Ep_spl, im_Ep_spl))

        self.E_spls = [
            splines_from_harmonics(p, n, L_synth)
            for (p, n) in zip(self.pos1_Elm, self.neg1_Elm)
        ]
        self.rE_spls = [
            splines_from_harmonics(p, n, L_synth)
            for (p, n) in zip(self.pos1_rElm, self.neg1_rElm)
        ]

        # knots for all the splines are the same
        self.xknots, self.yknots = self.E_spls[0][0][0].get_knots()

        # orders of the constructed splines
        self.kx = 3
        self.ky = 3

        # number of coefficients for each spline
        N_c = (self.E_spls[0][0][0].get_coeffs()).shape[0]

        self.E_spl_coeffs = np.zeros((Nfreq, 2, 2, N_c), dtype=np.float64)
        self.rE_spl_coeffs = np.zeros((Nfreq, 2, 2, N_c), dtype=np.float64)

        for ii in range(Nfreq):
            for aa in range(2):
                for bb in range(2):
                    self.E_spl_coeffs[ii, aa, bb] = self.E_spls[ii][aa][bb].get_coeffs()
                    self.rE_spl_coeffs[ii, aa, bb] = self.rE_spls[ii][aa][
                        bb
                    ].get_coeffs()

        def construct_component_functions(E_spl):
            def Et_func(theta, phi, grid=False):
                mu = np.cos(theta)
                return E_spl[0][0](mu, phi, grid=grid) + 1j * E_spl[0][1](
                    mu, phi, grid=grid
                )

            def Ep_func(theta, phi, grid=False):
                mu = np.cos(theta)
                return E_spl[1][0](mu, phi, grid=grid) + 1j * E_spl[1][1](
                    mu, phi, grid=grid
                )

            return Et_func, Ep_func

        def construct_directivity_function(Et_spl, Ep_spl):
            def D_func(theta, phi, grid=False):
                return (
                    np.abs(Et_spl(theta, phi, grid=grid)) ** 2.0
                    + np.abs(Ep_spl(theta, phi, grid=grid)) ** 2.0
                )

            return D_func

        self.Et_funcs = []
        self.Ep_funcs = []
        self.rEt_funcs = []
        self.rEp_funcs = []

        for ii in range(Nfreq):

            Et_func, Ep_func = construct_component_functions(self.E_spls[ii])
            self.Et_funcs.append(Et_func)
            self.Ep_funcs.append(Ep_func)

            rEt_func, rEp_func = construct_component_functions(self.rE_spls[ii])
            self.rEt_funcs.append(rEt_func)
            self.rEp_funcs.append(rEp_func)

        self.D_funcs = [
            construct_directivity_function(t, p)
            for (t, p) in zip(self.Et_funcs, self.Ep_funcs)
        ]
        self.rD_funcs = [
            construct_directivity_function(t, p)
            for (t, p) in zip(self.rEt_funcs, self.rEp_funcs)
        ]

    def construct_jones_matrix_functions(self, imap="default"):
        if imap == "default":
            imap = {"Et": (0, 0), "Ep": (0, 1), "rEt": (1, 0), "rEp": (1, 1)}

        def construct_jones_matrix_func(Et, Ep, rEt, rEp, imap):
            def J_func(theta, phi):
                theta = np.array(theta)
                phi = np.array(phi)

                J_out = np.zeros((theta.size, 2, 2), dtype=np.complex128)

                J_out[:, imap["Et"][0], imap["Et"][1]] = Et(theta, phi)
                J_out[:, imap["Ep"][0], imap["Ep"][1]] = Ep(theta, phi)
                J_out[:, imap["rEt"][0], imap["rEt"][1]] = rEt(theta, phi)
                J_out[:, imap["rEp"][0], imap["rEp"][1]] = rEp(theta, phi)

                return J_out

            return J_func

        Nfreq = self.nu_axis.size
        self.J_funcs = []
        for ii in range(Nfreq):
            Et_i = self.Et_funcs[ii]
            Ep_i = self.Ep_funcs[ii]
            rEt_i = self.rEt_funcs[ii]
            rEp_i = self.rEp_funcs[ii]

            J_func = construct_jones_matrix_func(Et_i, Ep_i, rEt_i, rEp_i, imap)
            self.J_funcs.append(J_func)


@nb.njit
def dl_m(el, s, beta, delta):
    L = (delta.shape[2] + 1) / 2
    mp = np.arange(-el, el + 1)

    #     k = np.exp(1j*mp*beta)
    arg = mp * beta
    k = np.cos(arg) + 1j * np.sin(arg)

    ms = -el + L - 1
    mf = (el + 1) + (L - 1)
    s_i = -s + L - 1

    delta_1 = delta[el, ms:mf, ms:mf]
    delta_2 = delta[el, ms:mf, s_i]

    dl_m_out = np.zeros(2 * el + 1, dtype=nb.complex128)

    for i_m in range(len(mp)):
        dl_m_out[i_m] = 1j ** (-s - mp[i_m]) * np.sum(
            k * delta_1[:, i_m] * delta_2, axis=0
        )

    return dl_m_out


@nb.njit(parallel=True)
def ssht_numba_series_eval(f_lm, s, L, delta, theta, phi):
    f = np.zeros(len(theta), dtype=nb.complex128)

    spin_sign = (-1.0) ** s
    for i in nb.prange(len(theta)):
        for el in range(L):
            m_axis = np.arange(-el, el + 1)

            phases = m_axis * phi[i]
            sY_elm = (
                spin_sign
                * np.sqrt((2.0 * el + 1.0) / 4.0 / np.pi)
                * (np.cos(phases) + 1j * np.sin(phases))
            )
            sY_elm *= dl_m(el, s, theta[i], delta)

            j0 = el * (el + 1) - el
            j1 = el * (el + 1) + el

            f[i] += np.sum(sY_elm * f_lm[j0 : j1 + 1])

    return f
