#!/usr/bin/python3
"""
Do 3C model fit
Calculate Greg&Carder Ed

3C model fit from 
https://gitlab.com/pgroetsch/rrs_model_3C/-/blob/master/rrs_model_3C.py
referencing
Gege, P., & Groetsch, P. (2016). A spectral model for correcting sun glint and sky glint. In Proceedings of Ocean Optics XXIII.
Groetsch, P. M. M., Gege, P., Simis, S. G. H., Eleveld, M. A., and Peters, S. W. M.: Validation of a spectral correction procedure for sun and sky reflections in above-water reflectance measurements, submitted to Optics Express.
"""

import os
import pandas as pd
import numpy as np
import lmfit as lm
import datetime
import sys
sys.path.insert(0,'.')
from . import gc90
from . import rrs_model_3C
from scipy import stats

sd = os.path.dirname(os.path.realpath(__file__))

wlin = np.arange(350., 900., 1.)  # common wavelength grid used for observations and models

# initialize model
print("Initializing 3C model...")
model = rrs_model_3C.rrs_model_3C(wl_range = (wlin[0], wlin[-1]), spectra_path=os.path.join(sd, 'spectra/'))
print("..done")

# weights
weights = pd.Series(1, index=wlin)
weights.loc[:500] = 5
weights.loc[675:750] = 0.1
weights.loc[760:770] = 0.1    

params = lm.Parameters()
# (Name,  Value,  Vary,   Min,  Max,  Expr)
params.add_many(
    ('C_chl', 5, True, 0.01, 100, None),             
    ('C_mie', 0, False, 0, 100, None), 
    ('n_mie', -1, False, -2, 2, None), 
    ('C_sm', 1, True, 0.01, 100, None), 
    ('C_y', 0.5, True, 0.01, 5, None),         
    ('S_y', 0.018, False, 0.01, 0.03, None),         
    #('S_y', -1, False, -1, 0.03, None),
    ('T_w', 12, False, 0, 35, None),         
    ('theta_sun', 0, False, 0, 90, None),         
    ('theta_view', 40, False, 0, 180, None),                 
    ('n_w', 1.34, False, 1.33, 1.34, None),
    ('rho_s', 0.0256, False, 0.0, 0.1, None),
    ('am', 1, False, 1, 10, None), 
    ('rh', 60, False, 30, 99.9, None), 
    ('pressure', 1013.25, False, 950, 1100, None), 
    ('delta', 0.00, False, 0, 1, None),
    ('rho_dd', 0.0, True, 0, 0.1, None),
    ('rho_ds', 0.01, True, 0.0, 0.1, None),
    ('alpha', 1.0, True, 0, 3, None),
    ('beta', 0.05, True, 0.0, 10, None), 
    )


## some functions below, scroll down for the main work

def process_single(r):
    """
    Apply 3C model (Groetsch et al. 2016) to (ir)radiance spectra to model Rrs
    Addditionally derive Greg & Carder 1990 ideal sky Ed using some of the 3C fit parameters
    r = data from pandas dataframe created from Rflex data (Baltic Sea)
    output:
        : resultdict
    """
    timestamp = datetime.datetime.strptime(r.StartTimeUTC_s, '%Y-%m-%d %H:%M:%S.%f')
    lat = float(r.GpsLatitude.replace(',','.'))
    lon = float(r.GpsLongitude.replace(',','.'))

    wl = np.arange(float(r[4]), float(r[5]), float(r[6]))

    Ltstart = r.tolist().index('Lt')
    Lsstart = r.tolist().index('Ls')
    Edstart = r.tolist().index('Ed')
    
    Lt = r.tolist()[Ltstart+1: Lsstart]
    Ls = r.tolist()[Lsstart+1: Edstart]
    Ed = r.tolist()[Edstart+1: Edstart+len(wl)+1]

    sunzenith = 90.0-r.RflexSunElevation
    params['theta_sun'].value = sunzenith
    Ltin = np.interp(wlin, wl, Lt)
    Lsin = np.interp(wlin, wl, Ls)
    Edin = np.interp(wlin, wl, Ed)

    wl400 = np.argmin(abs(wlin-400))
    Ls_Ed_400 = Ltin[wl400]/Edin[wl400]
        
    # fit Groetsch 3C model
    reg, Rrs_modelled, Rrs_refl, Lu_Ed_modelled, Rrs_obs, Ed_cg = model.fit_LuEd(wlin, Lsin, Ltin, Edin, params, weights.values)
    
    pressure = reg.params['pressure'].value
    rh = reg.params['rh'].value
    T_w = reg.params['T_w']
    # independently guess ideal sky G&C, using optimized rh, pressure, temperature
    Edgc = gc90.model(timestamp, lat, lon, windspeed=5, sea_p=pressure, air_t=T_w, rh=rh)
    Edgcin = np.interp(wlin/1000., gc90.wave, Edgc)

    # fit metrics
    Ed3C_slope, Ed3C_int, Ed3C_r_value, Ed3C_p, Ed3C_SE = stats.linregress(Lu_Ed_modelled, Ltin/Edin)
    EdGC_slope, EdGC_int, EdGC_r_value, EdGC_p, EdGC_SE = stats.linregress(Edgcin, Edin)

    resultdict = {'id': r.name,
                  'lat': lat,
                  'lon': lon,
                  'timestamp': timestamp,
                  'sunzenith': sunzenith,
                  'Lt': Ltin,
                  'Ed': Edin,
                  'Lu_Ed_mod': Lu_Ed_modelled,
                  'Ed400': Edin[wl400],
                  'Ls_Ed_400': Ls_Ed_400,
                  'Rrs_mod': Rrs_modelled,
                  'Rrs_obs': Rrs_obs,
                  'Ed_GC90': Edgcin,
                  'rho_dd': reg.params['rho_dd'].value,
                  'rho_ds': reg.params['rho_ds'].value,
                  'Ed3C_slope': Ed3C_slope, 
                  'Ed3C_int': Ed3C_int, 
                  'Ed3C_r_value': Ed3C_r_value, 
                  'Ed3C_p': Ed3C_p,
                  'Ed3C_SE': Ed3C_SE,
                  'EdGC_slope': EdGC_slope, 
                  'EdGC_int': EdGC_int, 
                  'EdGC_r_value': EdGC_r_value, 
                  'EdGC_p': EdGC_p,
                  'EdGC_SE': EdGC_SE,
                  }

    return resultdict


