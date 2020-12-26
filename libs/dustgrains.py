#!/usr/bin/env python
from __future__ import print_function
import os
import glob
import re
import math
import csv

import theano.tensor as tt

import numpy as np
import pandas as pd

from scipy.interpolate import interp1d

from libs.dustdata  import DustData


__all__ = ['DustGrains']

# Some params
c        = 2.998e10   # cm/s
ma       = 1.660e-24  # grams
p        = 4.         # grain-size power law
sel_step = 5


# min/max wavelengths for storage
min_wave = 0.
max_wave = 1.e6
min_wave_emission = 0.
max_wave_emission = 1.e6

_allowed_components = [
            'silicates',
            'carbonaceous',
            'graphite',
            'PAH-neutral',
            'PAH-ionized'
        ]


# Object for the proprerties of dust grain with a specific composition
class DustGrains(object):
    '''
    | Read data from precomputed source or observed source
    |
    |
    |
    | --Params:
    |           
    |           
    |           
    |           
    |           
    |           
    |
    | --Return:
    |            
    '''

    def __init__(self, grain_type, path='./', from_obs=False, gal='MW', sel_step=sel_step):
        '''
        | Read precomputed dust grain information from files
        |
        |
        |
        | --Params:
        |           
        |           
        |           
        |           
        |           
        |           
        |
        | --Return:
        |            
        '''

        self.origin = 'precomputed'
        self.gal    = gal

        
        # check the allowed components
        if grain_type not in _allowed_components:
            print(grain_type + ' is not allowed. Please select the following components')
            print(_allowed_components)
            exit()


        self.name = grain_type


        # The params of grains from WD2001
        # (See the paper for details or see github/nv-hiep/dust_scattering)
        if grain_type == 'silicates':  
            self.density            = 3.5                                                   # g/cm^3
            self.atomic_composition = 'MgFeSiO4'
            self.atomic_comp_names  = ['Mg', 'Fe', 'Si', 'O']
            self.atomic_comp_number = np.array([1, 1, 1, 4])
            self.atomic_comp_masses = ( np.array([24.305, 55.845, 28.0855, 15.994]) * ma )  # grams
        
        elif grain_type == 'carbonaceous':  # from WD01
            self.density            = 2.24                                                  # g/cm^3
            self.atomic_composition = 'C'
            self.atomic_comp_names  = ['C']
            self.atomic_comp_number = np.array([1])
            self.atomic_comp_masses = np.array([12.0107]) * ma                              # grams

        elif grain_type == 'graphite':
            self.density            = 2.24  # g/cm^3
            self.atomic_composition = 'C'
            self.atomic_comp_names  = ['C']
            self.atomic_comp_number = np.array([1])
            self.atomic_comp_masses = np.array([12.0107]) * ma                              # grams

        
        # useful quantities
        self.mass_per_mol_comp = np.sum(self.atomic_comp_masses * self.atomic_comp_number)
        self.col_den_constant  = ( (4./3.) * math.pi * self.density * (self.atomic_comp_number / self.mass_per_mol_comp) )

        
        # Read in precomputed dust grain information from files
        self.__read_data(grain_type, path)

        # Read observed data
        if from_obs:
            self.__read_obsdata()

    



    def __read_data(self, grain_type, path):
        '''
        | Read precomputed dust grain information from files
        |
        |
        |
        | --Params:
        |           
        |           
        |           
        |           
        |           
        |           
        |
        | --Return:
        |            
        '''

        # Read data
        try:
            df = pd.read_csv(os.path.join(path, 'data', 'grain_components', grain_type+'.csv'))
        except Exception as e:
            print ('No grain data found!')
            exit()
            raise e
        
        # dust grain data
        (composition,)    = df['composition'].unique()
        (composition_id,) = df['compostion_id'].unique() # 0 or 1 or 2 or 3 or 4
        assert composition == grain_type

        grain_size = df['grain_size'].unique()          # grain-sizes
        size_index = df['size_index'].unique()          # grain-size indexes
        assert len(grain_size) == len(size_index)

        # Number of grain-sizes
        n_sizes = len(size_index)

        # Wavelengths for grain-sizes are the same
        lambda_ = df['wavelength'].to_numpy().reshape(n_sizes, -1)
        lambda_ = lambda_[0]

        # Number of wavelength bins
        nbins   = len(lambda_)

        # Limit the wavelength to the desired range
        (idxs,)  = np.where( (lambda_ >= min_wave) & (lambda_ <= max_wave) )
        (eidxs,) = np.where( (lambda_ >= min_wave_emission) & (lambda_ <= max_wave_emission) )

        # Infor of each grain-size
        stochastic_heating = df['stochastic_heating'].to_numpy()
        stoch_heating      = stochastic_heating.astype(int)             # 0 and 1
        stoch_heating_neg  = np.logical_not(stoch_heating).astype(int)  # 0 and 1

        stochastic_heating = stochastic_heating.reshape(n_sizes, -1)
        heating_timescale  = df['heating_timescale'].to_numpy().reshape(n_sizes, -1)
        cooling_timescale  = df['cooling_timescale'].to_numpy().reshape(n_sizes, -1)
        heat2cool_ratio    = df['heat2cool_ratio'].to_numpy().reshape(n_sizes, -1)
        stype              = df['type'].to_numpy().reshape(n_sizes, -1)
        scale              = df['scale'].to_numpy().reshape(n_sizes, -1)
        temperature        = df['temperature'].to_numpy().reshape(n_sizes, -1)



        stochastic_heating = np.unique(stochastic_heating, axis=1).flatten().tolist()
        heating_timescale  = np.unique(heating_timescale, axis=1).flatten().tolist()
        cooling_timescale  = np.unique(cooling_timescale, axis=1).flatten().tolist()
        heat2cool_ratio    = np.unique(heat2cool_ratio, axis=1).flatten().tolist()
        scale              = np.unique(scale, axis=1).flatten().tolist()
        temperature        = np.unique(temperature, axis=1).flatten().tolist()
        
        # Data for grain-sizes
        cext               = df['Cext'].to_numpy().reshape(n_sizes, -1)
        cabs               = df['Cabs'].to_numpy().reshape(n_sizes, -1)
        csca               = df['Csca'].to_numpy().reshape(n_sizes, -1)
        albedo             = df['albedo'].to_numpy().reshape(n_sizes, -1)
        g                  = df['g'].to_numpy().reshape(n_sizes, -1)
        
        eq_emis            = df['eq_emis'].to_numpy()
        st_emis            = df['st_emis'].to_numpy()
        
        # if stochastic_heating, emission = st_emis, else emission = eq_emis
        emission           = stoch_heating*st_emis + stoch_heating_neg*eq_emis
        
        eq_emis            = eq_emis.reshape(n_sizes, -1)
        st_emis            = st_emis.reshape(n_sizes, -1)
        emission           = emission.reshape(n_sizes, -1)

        assert cext.shape[1] == cabs.shape[1] == csca.shape[1] == albedo.shape[1] ==\
               g.shape[1] == eq_emis.shape[1] == st_emis.shape[1]


        # Store the properties
        self.lambda_       = lambda_[idxs]           # Wavelengths
        self.lambda_emis   = lambda_[eidxs]
        
        self.n_lambda      = len(self.lambda_)
        self.n_lambda_emis = len(self.lambda_emis)
        
        self.grain_type    = grain_type              # carbonaceous, graphite, silicates, PAH-ionized, PAH-neutral

        
        if sel_step is not None:
            #  Pick every nth grain size, make the fitting faster, but the size distributions coarser
            size_sel = np.arange(0, n_sizes, sel_step)
            n_sizes  = len(size_sel)

            # Grain infor
            self.n_sizes            = n_sizes                        # number of grain-sizes selected
            self.sizes              = grain_size[size_sel]           # grain-sizes selected
            self.size_index         = size_index[size_sel]
            self.stochastic_heating = [stochastic_heating[j] for j in size_sel]  # True/False
            
            self.cext               = cext[:, idxs][size_sel, :]
            self.cabs               = cabs[:, idxs][size_sel, :]
            self.csca               = csca[:, idxs][size_sel, :]
            self.albedo             = albedo[:, idxs][size_sel, :]
            self.scat_g             = g[:, idxs][size_sel, :]
            
            self.emission           = emission[:, eidxs][size_sel, :]
        
        else:

            # Grain infor
            self.n_sizes            = n_sizes                        # number of grain-sizes
            self.sizes              = grain_size                     # grain-sizes
            self.size_index         = size_index
            self.stochastic_heating = stochastic_heating
            
            self.cext               = cext[:, idxs]                  # Ext cross-section
            self.cabs               = cabs[:, idxs]                  # Absorption cross-section
            self.csca               = csca[:, idxs]                  # Scattering cross-section
            self.albedo             = albedo[:, idxs]                # Albedo = scat/ext
            self.scat_g             = g[:, idxs]                     # g=cos(theta) in scattering
            
            self.emission           = emission[:, eidxs]             # MJy/sr
        # End - if



        '''
        convert emission from ergs/(s cm sr) to Jy/sr;   erg = 100nJ
        1Jy = 10^{-23} erg s^{-1} cm^{-2} Hz^{-1}
        
        wavelengths in microns
        convert from cm^-1 to Hz^-1
        c = 2.998e10 cm/s
        '''
        self.emission *= (self.lambda_emis)**2 / c        
        self.emission /= 1.e-19                              # ergs/(s Hz) -> Jy
        self.emission *= 1.e-6                               # Jy/sr ->  MJy/sr
        
        # convert from m^-2 to cm^-2
        self.emission *= 1.e-4

        

        # default size distributions, can change latter (p = 3.5 -- 4)
        self.size_dist = self.sizes ** (-4.)


        # Set some variables for latter use
        self.albedo_lambda    = self.lambda_       # wavelengths when albedo_err > 0.
        self.n_albedo_lambda  = self.n_lambda      # number of wavelengths when albedo_err > 0.
        self.scat_albedo_cext = self.cext          # Extinction cross-section when albedo_err > 0.
        self.scat_albedo_csca = self.csca          # Scattering cross-section when albedo_err > 0.
        
        self.lambda_scat_g    = self.lambda_       # wavelengths when g_err > 0.
        self.n_g_lambda       = self.n_lambda      # number of wavelengths when g_err > 0.
        self.scat_g_csca      = self.csca          # Scattering cross-section when g_err > 0.





    def __read_obsdata(self):
        '''
        | Read dust grain information from observed data
        |
        |
        |
        | --Params:
        |           
        |           
        |           
        |           
        |           
        |           
        |
        | --Return:
        |            
        '''
        self.origin = 'observed'

        # Path to the data
        path = os.getcwd()
        path = os.path.join(path, 'data', 'MW') if ( self.gal == 'MW') else os.path.join(path, 'data', 'SMC')


        # Obseved data
        obsdata = DustData(path,
                           abundance  = True,
                           extinction = True,
                           emission   = True,
                           scattering = True,
                           ISRF       = True)

        # Observed scattering wavelengths
        scat_lambda_obs   = obsdata.ext_lambda
        n_scat_lambda_obs = len(scat_lambda_obs)

        # Scattering parameters
        scat_cext = np.empty((self.n_sizes, n_scat_lambda_obs))
        scat_cabs = np.empty((self.n_sizes, n_scat_lambda_obs))
        scat_csca = np.empty((self.n_sizes, n_scat_lambda_obs))


        # Emission
        emis_lambda_obs     = obsdata.emis_lambda
        n_emis_lambda_obs   = len(emis_lambda_obs)
        emis_spec           = np.empty((self.n_sizes, n_emis_lambda_obs))
        
        
        # Albedo
        albedo_lambda_obs   = obsdata.scat_albedo_lambda
        n_albedo_lambda_obs = len(albedo_lambda_obs)
        albedo_lambda_csca  = np.empty((self.n_sizes, n_albedo_lambda_obs))
        albedo_lambda_cext  = np.empty((self.n_sizes, n_albedo_lambda_obs))
        
        
        # g=<cos(theta)>
        g_lambda_obs        = obsdata.scat_g_lambda
        n_g_lambda_obs      = len(g_lambda_obs)
        scat_g_obs          = np.empty((self.n_sizes, n_g_lambda_obs))
        g_lambda_csca       = np.empty((self.n_sizes, n_g_lambda_obs))


        # loop over the sizes and generate grain info on the observed data grid
        for i in range(self.n_sizes):
            cext_interp = interp1d(self.lambda_, self.cext[i, :])
            cabs_interp = interp1d(self.lambda_, self.cabs[i, :])
            csca_interp = interp1d(self.lambda_, self.csca[i, :])
            
            scat_cext[i, :] = cext_interp(scat_lambda_obs)
            scat_cabs[i, :] = cabs_interp(scat_lambda_obs)
            scat_csca[i, :] = csca_interp(scat_lambda_obs)

            albedo_lambda_cext[i, :] = cext_interp(albedo_lambda_obs)
            albedo_lambda_csca[i, :] = csca_interp(albedo_lambda_obs)

            g_interp            = interp1d(self.lambda_, self.scat_g[i, :])
            scat_g_obs[i,    :] = g_interp(g_lambda_obs)
            g_lambda_csca[i, :] = csca_interp(g_lambda_obs)

            emission_interp = interp1d(self.lambda_emis, self.emission[i, :])
            emis_spec[i, :] = emission_interp(emis_lambda_obs)
        # End - for


        # assign\
        # Lambda
        self.lambda_       = scat_lambda_obs
        self.lambda_emis   = emis_lambda_obs
        
        self.n_lambda      = len(self.lambda_)
        self.n_lambda_emis = len(self.lambda_emis)
        
        # Scattering
        self.cabs          = scat_cabs
        self.csca          = scat_csca
        self.cext          = scat_cext

        # albedo
        self.albedo_lambda    = albedo_lambda_obs
        self.n_albedo_lambda  = len(self.albedo_lambda)
        self.scat_albedo_csca = albedo_lambda_csca
        self.scat_albedo_cext = albedo_lambda_cext
        self.albedo           = self.scat_albedo_csca / self.scat_albedo_cext

        # g=cos(theta)
        self.lambda_scat_g    = g_lambda_obs
        self.n_g_lambda       = len(self.lambda_scat_g)
        self.scat_g           = scat_g_obs

        # Emission
        self.emission         = emis_spec

        # g_scat
        self.scat_g_csca      = g_lambda_csca


    



    # function to integrate this component
    # returns the effective/total cabs, csca, etc.
    # these are normalized to NHI (assumption)
    def eff_grain_props(self, ObsData, predict_all=False):
        '''
        | Calculate the grain properties integrated over the size distribution for a single grain composition.
        |
        |
        |
        | --Params:
        |           
        |           
        |           
        |           
        |           
        |           
        |
        | --Return: Dict of
        |           C(abs) : 'np.ndarray' -  Absorption cross section
        |           C(sca) : 'np.ndarray'  -  Scattering cross section
        |           Abundances : ('list', 'numpy.ndarray') named 'natoms' - Tuple with (atomic elements, # per/10^6 H atoms
        |           Emission : 'np.ndarray' named 'emission' - IR emission
        |           albedo : 'numpy.ndarray' named 'albedo' - Dust scattering albedo [Albedo C(sca)/Albedo C(ext)]
        |           g : 'numpy.ndarray' named 'g' - Dust scattering phase function assymetry [g = <cos theta>]
        |           Albedo C(ext) : 'np.ndarray' named 'scat_a_cext'
        |                             Extinction cross section on the albedo wavelength grid
        |                             (needed for combining with other dust grain compositions)
        |           Albedo C(sca) : 'np.ndarray' named 'scat_a_csca'
        |                           Scattering cross section on the albedo wavelength grid
        |                           (needed for combining with other dust grain compositions)
        |           G C(sca) : 'numpy.ndarray' named 'scat_g_csca'
        |                       Scattering cross section on the g wavelength grid
        |                       (needed for combining with other dust grain compositions)
        |            
        '''

        # output
        results = {}

        # initialize the variables
        _effcabs = np.empty(self.n_lambda)
        _effcsca = np.empty(self.n_lambda)

        # do a very simple integration (later this could be made more complex)
        deltas    = 0.5*(self.sizes[1 : self.n_sizes] - self.sizes[0 : self.n_sizes - 1])
        sizedist1 = self.size_dist[0 : self.n_sizes - 1]
        sizedist2 = self.size_dist[1 : self.n_sizes]

        
        for i in range(self.n_lambda):

            _effcabs[i] = np.sum(deltas*
                                        ((self.cabs[0:self.n_sizes-1, i]*sizedist1) + (self.cabs[1:self.n_sizes, i]*sizedist2))
                                )

            _effcsca[i] = np.sum(deltas*
                                       ((self.csca[0:self.n_sizes-1, i]*sizedist1) + (self.csca[1:self.n_sizes, i]*sizedist2))
                                )

        results["cabs"] = _effcabs
        results["csca"] = _effcsca

        # compute the number of atoms/NHI
        _natoms = np.empty(len(self.atomic_comp_names))
        for i in range(len(self.atomic_comp_names)):
            _natoms[i] = np.sum(deltas*((
                                          self.sizes[0 : self.n_sizes-1]**3
                                          *self.size_dist[0 : self.n_sizes-1]
                                          *self.col_den_constant[i]
                                        )
                                       + ((self.sizes[1 : self.n_sizes] ** 3)
                                        *self.size_dist[1 : self.n_sizes]
                                        *self.col_den_constant[i])
                                       ))

        # convert to N(N) per 1e6 N(HI)
        _natoms *= 1e6

        results["natoms"] = dict(zip(self.atomic_comp_names, _natoms))

        # compute the integrated emission spectrum
        if ObsData.fit_ir_emission or predict_all:
            _emission = np.empty(self.n_lambda_emis)
            for i in range(self.n_lambda_emis):
                _emission[i] = np.sum( deltas*(
                                               (self.emission[0 : self.n_sizes - 1, i] * sizedist1) +
                                               (self.emission[1 : self.n_sizes, i] * sizedist2)
                                               )
                                     )
            results["emission"] = _emission

        # scattering parameters a & g
        if ObsData.fit_scat_a or predict_all:
            n_waves_scat_a = self.n_albedo_lambda
            scat_a_cext = self.scat_albedo_cext
            scat_a_csca = self.scat_albedo_csca

            _effscat_a_cext = np.empty(n_waves_scat_a)
            _effscat_a_csca = np.empty(n_waves_scat_a)

            for i in range(n_waves_scat_a):
                _effscat_a_cext[i] = np.sum(
                    deltas
                    * (
                        (scat_a_cext[0 : self.n_sizes - 1, i] * sizedist1)
                        + (scat_a_cext[1 : self.n_sizes, i] * sizedist2)
                    )
                )
                _effscat_a_csca[i] = np.sum(
                    deltas
                    * (
                        (scat_a_csca[0 : self.n_sizes - 1, i] * sizedist1)
                        + (scat_a_csca[1 : self.n_sizes, i] * sizedist2)
                    )
                )

            results["albedo"] = _effscat_a_csca / _effscat_a_cext
            results["scat_a_cext"] = _effscat_a_cext
            results["scat_a_csca"] = _effscat_a_csca

        if ObsData.fit_scat_g or predict_all:
            n_waves_scat_g = self.n_g_lambda
            scat_g_csca = self.scat_g_csca

            _effg = np.empty(n_waves_scat_g)
            _effscat_g_csca = np.empty(n_waves_scat_g)

            for i in range(n_waves_scat_g):
                _effg[i] = np.sum(
                    deltas
                    * (
                        (
                            self.scat_g[0 : self.n_sizes - 1, i]
                            * scat_g_csca[0 : self.n_sizes - 1, i]
                            * sizedist1
                        )
                        + (
                            self.scat_g[1 : self.n_sizes, i]
                            * scat_g_csca[1 : self.n_sizes, i]
                            * sizedist2
                        )
                    )
                )
                _effscat_g_csca[i] = np.sum(
                    deltas
                    * (
                        (scat_g_csca[0 : self.n_sizes - 1, i] * sizedist1)
                        + (scat_g_csca[1 : self.n_sizes, i] * sizedist2)
                    )
                )
            results["g"] = _effg / _effscat_g_csca
            results["scat_g_csca"] = _effscat_g_csca


        return results