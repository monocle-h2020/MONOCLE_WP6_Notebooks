#!/usr/bin/python3
""" 
Python implementation of an analytic remote sensing reflectance model as described in Gege and Groetsch (2016) and Groetsch et al. (2017)

References:
Gege, P., & Groetsch, P. (2016). A spectral model for correcting sun glint and sky glint. In Proceedings of Ocean Optics XXIII.

Groetsch, P. M. M., Gege, P., Simis, S. G. H., Eleveld, M. A., and Peters, S. W. M.: Validation of a spectral correction procedure for sun and sky reflections in above-water reflectance measurements, submitted to Optics Express.
"""

import numpy as np
import pandas as pd

import theano as th
import theano.tensor as T
from theano.ifelse import ifelse

import lmfit as lm
import time

"""
__author__ = "Philipp Groetsch"
__copyright__ = "Copyright 2017, Philipp Groetsch"
__license__ = "LGPL"
__version__ = "0.1"
__maintainer__ = "Philipp Groetsch"
__email__ = "philipp.g@gmx.de"
__status__ = "Development"

This program is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""


class rrs_model_3C(object):

    def __init__(self, downsampling=False, wl_range = False, spectra_path = 'spectra/'):
        """ 
        spectra_path -- path to input spectra 
        downsampling -- reduce spectral resolution (False or window_length for rolling_mean)
        wl_range = limit spectral range (False or [float, float])
        """

        a_ph = pd.read_csv(spectra_path + 'phyto.A', skiprows=11, sep='\t', index_col=0).iloc[:,0]
        
        a_w = pd.read_csv(spectra_path + 'water.A', skiprows=10, sep='\t', index_col=0).iloc[:,0]
        daw_dT = pd.read_csv(spectra_path + 'daWdT.txt', skiprows=10, sep='\t', index_col=0).iloc[:,0]
        astar_y = pd.read_csv(spectra_path + 'Y.A', skiprows=11, delimiter=' ', index_col=0).loc[350:].iloc[:,0]

        d = pd.DataFrame(index = a_ph.index)
        d.index.name = 'Wavelength, [nm]'
        d['astar_ph'] = a_ph
        d['astar_y'] = astar_y    
        d['a_w'] = a_w            
        d['daw_dT'] = daw_dT

        if downsampling == False:
            self.spectra = d
        else: 
            self.spectra = d.apply(lambda x: pd.rolling_mean(x, downsampling, center=True))

        if wl_range != False:
            self.spectra = self.spectra.loc[wl_range[0]:wl_range[1]].dropna()

        self.wl = np.array(self.spectra.index)
                
        self.model = self._compile_model()


    def _compile_model(self):
        """ theano implementation of 3C model """

        ### GC90 atmospheric model implementation
        
        theta_sun, beta, alpha, am, rh, pressure = T.scalars('theta_sun', 'beta', 'alpha', 'am', 'rh', 'pressure')

        wl = T.vector('wl')
        
        wl_a = 550

        theta_sun_ = theta_sun * np.pi / 180.

        z3 = -0.1417 * alpha + 0.82
        z2 = ifelse(T.gt(alpha, 1.2), 0.65, z3)
        z1 = ifelse(T.lt(alpha, 0), 0.82, z2)
        
        theta_sun_mean = z1

        B3 = T.log(1 - theta_sun_mean)
        B2 = B3 * (0.0783 + B3 * (-0.3824 - 0.5874 * B3))
        B1 = B3 * (1.459 + B3 * (0.1595 + 0.4129 * B3))
        Fa = 1 - 0.5 * T.exp((B1 + B2 * T.cos(theta_sun_)) * T.cos(theta_sun_))

        omega_a = (-0.0032 * am + 0.972) * T.exp(3.06 * 1e-4 * rh)
        tau_a = beta*(wl/wl_a)**(-alpha)

        # fixed a bug in M, thanks Jaime! [brackets added]
        M  = 1 / (T.cos(theta_sun_) + 0.50572 * (90 + 6.07995 - theta_sun)**(-1.6364)) 
        M_ = M * pressure / 1013.25

        Tr = T.exp(- M_ / (115.6406 * (wl / 1000)**4 - 1.335 * (wl / 1000)**2)) 

        Tas = T.exp(- omega_a * tau_a * M)

        Edd = Tr * Tas    
        Edsr = 0.5 * (1 - Tr**0.95)
        Edsa = Tr**1.5 * (1 - Tas) * Fa
        
        Ed = Edd + Edsr + Edsa
        Edd_Ed = Edd / Ed
        Edsr_Ed = Edsr / Ed
        Edsa_Ed = Edsa / Ed
        Eds_Ed = Edsr_Ed + Edsa_Ed

        ### Albert and Mobley bio-optical model implementation

        a_w, daw_dT, astar_ph, astar_y, Ls_Ed = T.vectors('a_w', 'daw_dT', 'astar_ph', 'astar_y', 'Ls_Ed')             
        
        C_chl, C_sm, C_mie, n_mie, C_y, S_y, T_w, theta_view, n_w, rho_s, rho_dd, rho_ds, delta= T.scalars('C_chl', 'C_sm', 'C_mie', 'n_mie', 'C_y', 'S_y', 'T_w', 'theta_view', 'n_w', 'rho_s', 'rho_dd', 'rho_ds', 'delta')

        # calc_a_ph
        a_ph = C_chl * astar_ph

        # calc_a_y
        wl_ref_y = 440
        a_y = ifelse(T.eq(S_y, -1), C_y * astar_y, C_y * T.exp(- S_y * (wl - wl_ref_y)))

        # calc_a
        T_w_ref = 20.
        a_w_corr = a_w + (T_w - T_w_ref) * daw_dT
        
        a = a_w_corr + a_ph + a_y

        # calc_bb_sm
        bbstar_sm = 0.0086
        bbstar_mie = 0.0042
        wl_ref_mie = 500
        
        bb_sm = C_sm * bbstar_sm + C_mie * bbstar_mie * (wl / wl_ref_mie)**n_mie

        # calc_bb
        b1 = ifelse(T.eq(n_w, 1.34), 0.00144, 0.00111)
        
        wl_ref_water = 500
        S_water = -4.32

        bb_water = b1 * (wl / wl_ref_water)**S_water
        bb = bb_water + bb_sm

        # calc omega_b
        omega_b = bb / (bb + a)

        # calc sun and viewing zenith angles under water
        theta_sun_ = theta_sun * np.pi / 180.
        theta_sun_ss = T.arcsin(T.sin(theta_sun_) / n_w)
        theta_view_ = theta_view * np.pi / 180.
        theta_view_ss = T.arcsin(T.sin(theta_view_) / n_w)

        p_f = [0.1034, 1, 3.3586, -6.5358, 4.6638, 2.4121]
        p_frs = [0.0512, 1, 4.6659, -7.8387, 5.4571, 0.1098, 0.4021]

        # calc subsurface reflectance            
        f = p_f[0] * (p_f[1] + p_f[2] * omega_b + p_f[3] * omega_b**2 + p_f[4] * omega_b**3) * (1 + p_f[5] / T.cos(theta_sun_ss)) 

        R0minus = f * omega_b

        # calc subsurface remote sensing reflectance       
        frs = p_frs[0] * (p_frs[1] + p_frs[2] * omega_b + p_frs[3] * omega_b**2 + p_frs[4] * omega_b**3) * (1 + p_frs[5] / T.cos(theta_sun_ss)) * (1 + p_frs[6] / T.cos(theta_view_ss))

        Rrs0minus = frs * omega_b

        # calc water surface reflected reflectance        
        Rrs_refl = rho_s * Ls_Ed + rho_dd * Edd_Ed / np.pi + rho_ds * Eds_Ed / np.pi + delta

        # calc_Rrs0plus (Lee1998, eq22), R=Q*Rrs
        gamma = 0.48
        zeta = 0.518

        Rrs = zeta * Rrs0minus / ( 1 - gamma * R0minus )
        
        Lu_Ed = Rrs + Rrs_refl
        
        f = th.function([beta, alpha, am, rh, pressure, C_chl, C_sm, C_mie, n_mie, C_y, S_y, T_w, theta_sun, theta_view, n_w, rho_s, rho_dd, rho_ds, delta, wl, a_w, daw_dT, astar_ph, astar_y, Ls_Ed], [Rrs, Rrs_refl, Lu_Ed, Ed], on_unused_input='warn')

        return f

    def fit_LuEd(self, wl, Ls, Lu, Ed, params, weights, verbose=True):
        """ Fit of 3C to a set of radiometric measurements.

        Input:
            wl: wavelength array (numpy array)
            Ls: sky radiance (numpy array)
            Lu: upwelling radiance (numpy array)
            Ed: downwelling irradiance (numpy array)
            params: lmfit parameter array
            weights: spectral weighting function (numpy array)
            verbose: flag to enable extensive optimization routine feedback

        Output:
            reg: lmfit minimizer result set
            Rrs_modelled: modelled remote sensing reflectance (numpy array)
            Rrs_refl: surface-reflected reflectance (numpy array)
            Lu_Ed_modelled: modelled Lu/Ed spectrum (numpy array)
            Rrs_obs: final derived remote sensing reflectance (numpy array)

        """

        def min_funct(params):
            p = params.valuesdict()        
            
            Rrs_modelled, Rrs_refl, Lu_Ed_modelled, Ed_cg = self.model(beta = p['beta'], alpha = p['alpha'], am = p['am'], rh = p['rh'], pressure = p['pressure'], C_chl = p['C_chl'], C_sm = p['C_sm'], C_mie = p['C_mie'], n_mie = p['n_mie'], C_y = p['C_y'], S_y = p['S_y'], T_w = p['T_w'], theta_sun = p['theta_sun'], theta_view = p['theta_view'], n_w = p['n_w'], rho_s = p['rho_s'], rho_dd = p['rho_dd'], rho_ds = p['rho_ds'], delta = p['delta'], wl = wl, a_w = self.spectra['a_w'].values, daw_dT = self.spectra['daw_dT'].values, astar_ph = self.spectra['astar_ph'].values, astar_y = self.spectra['astar_y'].values, Ls_Ed = Ls/Ed)

            Rrs_obs = Lu/Ed - Rrs_refl

            # Least squares
            resid = np.sum((Lu_Ed_modelled - Lu/Ed)**2 * weights)

            return resid, Rrs_modelled, Rrs_refl, Lu_Ed_modelled, Rrs_obs, Ed_cg

        start_time = time.time()

        reg = lm.minimize(lambda x: min_funct(x)[0], params=params, method='lbfgsb', options={'disp': verbose, 'gtol': 1e-16, 'eps': 1e-07, 'maxiter': 15000, 'ftol': 1e-16, 'maxls': 20, 'maxcor': 20})  

        print("Fitting 3C model took %s seconds" % (time.time() - start_time))

        resid, Rrs_modelled, Rrs_refl, Lu_Ed_modelled, Rrs_obs, Ed_cg = min_funct(reg.params)
        reg.params.add('resid', resid, False, 0.0, 100, None)

        return reg, Rrs_modelled, Rrs_refl, Lu_Ed_modelled, Rrs_obs, Ed_cg

