#!/usr/bin/env python
'''
Observed data
'''
from __future__ import print_function

import numpy as np
import pandas as pd

import sys, os, glob
import csv

__all__ = ['DustData']


# Object for the observed dust data
class DustData(object):
    '''
    | Read data
    |
    |
    |
    | --Params:
    |           datdir     : string - path to data directory
    |           abundance  : boolen - Read abundance data
    |           extinction : boolen - Read extinction data
    |           emission   : boolen - Read emission data
    |           scattering : boolen - Read scattering data
    |           ISRF       : boolen - Read ISRF (interstellar radiation field) data
    |
    | --Return:
    |            
    '''

    # read data from files
    def __init__(self, datdir,
                 abundance  = False,
                 extinction = False,
                 emission   = False,
                 scattering = False,
                 ISRF       = False ):

        if abundance:
            self.__abundance(datdir)

        if extinction:
            self.__ext(datdir)

        if emission:
            self.__emission(datdir)

        if scattering:
            self.__scat(datdir)

        if ISRF:
            self.__ISRF(datdir)


    def __abundance(self, path):
        '''
        | Read abundance data
        |
        |
        |
        | --Params:
        |           path     : string - path to data directory
        |
        | --Return:
        |            
        '''
        files = glob.glob(os.path.join(path, 'abundance', '*.csv'))

        if len(files) == 0:
            print('')
            sys.exit('No abundance data found!')
            return

        # Abundance data, header = ['material', 'abundance', 'abund_err', 'total_abund', 'total_err']
        # abundance unit: atoms per 10^6 H atoms
        data        = pd.read_csv(files[0])
        atoms       = data['material'].to_numpy()           # O, C, Mg, Si, Fe
        abundance   = data['abundance'].to_numpy()          # Dust, atoms per 10^6 H atoms
        abund_err   = 0.1*abundance                         # 10% of the values
        total_abund = data['total_abund'].to_numpy()        # dust+gas, atoms per 10^6 H atoms
        total_err   = 0.1*total_abund                       # 10% of the values

        
        self.atoms  = atoms

        self.abundance   = {}
        self.total_abund = {}

        for i,atom in enumerate(atoms):
            self.abundance[atom]   = ( abundance[i], abund_err[i] )
            self.total_abund[atom] = ( total_abund[i], total_err[i]  )

        self.fit_abundance = True




    # read in the data from files
    def __ext(self, path):
        '''
        | Read data
        |
        |
        |
        | --Params:
        |           path     : string - path to data directory
        |
        | --Return:
        |            
        '''

        # Dust-to-gas conversion
        d2g_file  = glob.glob(os.path.join(path, 'dust2gas', '*.csv'))
        if len(d2g_file) == 0:
            print('')
            sys.exit('No dust-to-gas data found!')
            return


        # Extinction
        ext_files = glob.glob(os.path.join(path, 'ext', '*.csv'))
        if len(ext_files) == 0:
            print('')
            sys.exit('No extinction data found!')
            return



        # Dust-to-gas conversion, A(V) to N(HI)
        # header = ['gal', 'Rv', 'Av2NHI', 'err', 'ref']
        data            = pd.read_csv(d2g_file[0])
        self.Av2NHI     = data['Av2NHI'].to_numpy()[0]
        self.Av2NHI_err = data['err'].to_numpy()[0]


        # extinction curve
        # Header: gal, wavelength_number, wavelength, AlAv, AlAv_err, class, ref
        data              = pd.read_csv(ext_files[0])
        self.ext_wavenum  = data['wavelength_number'].to_numpy()
        self.ext_lambda   = data['wavelength'].to_numpy()            # wavenum = 1. / wavelength
        self.ext_AlAv     = data['AlAv'].to_numpy()
        self.ext_AlAv_err = data['AlAv_err'].to_numpy()

        # sort
        sindxs            = np.argsort(self.ext_lambda)
        self.ext_wavenum  = self.ext_wavenum[sindxs]
        self.ext_lambda   = self.ext_lambda[sindxs]
        self.ext_AlAv     = self.ext_AlAv[sindxs]
        self.ext_AlAv_err = self.ext_AlAv_err[sindxs]

        # A(lambda)/A(V) -> A(lambda)/N(HI)
        self.ext_AlNHI     = self.ext_AlAv * self.Av2NHI
        self.ext_AlNHI_err = (self.ext_AlAv_err/self.ext_AlAv)**2 + (self.Av2NHI_err / self.Av2NHI)**2
        self.ext_AlNHI_err = self.ext_AlNHI * np.sqrt(self.ext_AlNHI_err)

        self.fit_extinction = True






    # read in the data from files
    def __scat(self, path):
        '''
        | Read data
        |
        |
        |
        | --Params:
        |           path     : string - path to data directory
        |
        | --Return:
        |            
        '''

        scat_files = glob.glob(os.path.join(path, 'scat', '*.csv'))

        if len(scat_files) == 0:
            print('')
            sys.exit('No scattering data found!')
            return

        # scattering data, header = ['reference', 'wavelength' [in Angstrom], 'albedo', 'albedo_err', 'g', 'g_err']
        data = pd.read_csv(scat_files[0])
        data = data[ (data['albedo_err'] > 0.) & (data['g_err'] > 0.) ]

        ref        = data['reference'].to_numpy()
        lambda_    = 1.e-4*data['wavelength'].to_numpy()      # lambda (in microns)
        albedo     = data['albedo'].to_numpy()
        albedo_err = data['albedo_err'].to_numpy()
        g          = data['g'].to_numpy()
        g_err      = data['g_err'].to_numpy()


        # remove all the measurements with zero uncertainty
        (gindxs,)               = np.where(albedo_err > 0.)
        self.scat_albedo_lambda = lambda_[gindxs]
        self.scat_albedo        = albedo[gindxs]
        self.scat_albedo_err    = albedo_err[gindxs]
        self.scat_albedo_ref    = ref[gindxs]
        
        # sort
        sindxs                  = np.argsort(self.scat_albedo_lambda)
        self.scat_albedo_lambda = self.scat_albedo_lambda[sindxs]
        self.scat_albedo        = self.scat_albedo[sindxs]
        self.scat_albedo_err    = self.scat_albedo_err[sindxs]
        self.scat_albedo_ref    = self.scat_albedo_ref[sindxs]

        # remove all the measurements with zero uncertainty
        (gindxs,)          = np.where(g_err > 0.)
        self.scat_g_lambda = lambda_[gindxs]
        self.scat_g        = g[gindxs]
        self.scat_g_err    = g_err[gindxs]
        self.scat_g_ref    = ref[gindxs]
        
        # sort
        sindxs             = np.argsort(self.scat_g_lambda)
        self.scat_g_lambda = self.scat_g_lambda[sindxs]
        self.scat_g        = self.scat_g[sindxs]
        self.scat_g_err    = self.scat_g_err[sindxs]
        self.scat_g_ref    = self.scat_g_ref[sindxs]


        self.fit_scat_a = True
        self.fit_scat_g = True





    # read in the data from files
    def __emission(self, path):
        '''
        | Read data
        |
        |
        |
        | --Params:
        |           path     : string - path to data directory
        |
        | --Return:
        |            
        '''

        emis_file = glob.glob(os.path.join(path, 'emission', '*.csv'))

        if len(emis_file) == 0:
            print('')
            sys.exit('No emission data found!')
            return


        # Diffuse IR emission spectrum
        # header = ['instrument', 'filter', 'lambda', 'spectrum', 'error']
        # lambda (microns)
        # \nu * I_{\nu}     [MJy/sr]        [Unit : [10^-20 W m-2 sr-1 H-1]]
        data = pd.read_csv(emis_file[0])

        self.emis_lambda = data['lambda'].to_numpy()   # lambda (microns)
        self.spec_       = 1.e-20*data['spectrum'].to_numpy()   # \nu * I_{\nu}     [MJy/sr]
        self.spec_err    = 1.e-20*data['error'].to_numpy()
        self.instru      = data['instrument'].to_numpy()
        self.filter      = data['filter'].to_numpy()
        
        # If error = 0, set error = 10% of the value
        (idx,) = np.where(self.spec_err == 0.)                                  # 'where' gives (array([]), )

        if len(idx) > 0:
            self.spec_err[idx] = 0.1 * self.spec_[idx]

        # sort
        sids             = np.argsort(self.emis_lambda)
        self.emis_lambda = self.emis_lambda[sids]
        self.spec_       = self.spec_[sids]
        self.spec_err    = self.spec_err[sids]
        self.instru      = self.instru[sids]
        self.filter      = self.filter[sids]

        self.fit_ir_emission = True




    # read in the data from files
    def __ISRF(self, path):
        '''
        | Read data
        |
        |
        |
        | --Params:
        |           path     : string - path to data directory
        |
        | --Return:
        |            
        '''

        ISRF_files = glob.glob(os.path.join(path, 'ISRF', '*.csv'))

        if len(ISRF_files) == 0:
            print('')
            sys.exit('No ISRF data found!')
            return


        # ISRF data, header = ['wavelength', 'ISRF']
        # wavelength [um], ISRF [erg/cm^3/s/st]
        data = pd.read_csv(ISRF_files[0])
        # data = data[ (data['albedo_err'] > 0.) & (data['g_err'] > 0.) ]

        self.ISRF_lambda = data['wavelength'].to_numpy()    # lambda (in microns)
        self.ISRF        = data['ISRF'].to_numpy()          # ISRF [erg/cm^3/s/st]

        # sort
        sids             = np.argsort(self.ISRF_lambda)
        self.ISRF_lambda = self.ISRF_lambda[sids]
        self.ISRF        = self.ISRF[sids]