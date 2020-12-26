import math
import pickle
import numpy as np

from math       import erf
from astropy.io import fits

from .dustgrains import DustGrains

__all__ = ['DustModel', 'MRN77', 'WD01']


class DustModel(object):
    '''
    | Dust model (MRN1977 or WD2001)
    | Full dust model including arbirary size and composition distributions.
    |Includes the physical properties of the individual dust grains.
    |
    |
    | --Params:
    |           componentnames : list of strings - names of dust grain materials
    |           path :  string - names of dust grain materials
    |           from_obs : Boolean - From observed data or precomputed data
    |           sel_step : int - Only use every nth size, faster fitting
    |           parameters   : Dictonary of parameters with an entry for each composition
    |                          each entry is then a dictonary giving the value by parameter name.
    | 
    |
    | --Return:
    |            
    '''

    def __init__(self,
                 componentnames=None,
                 path="./",
                 from_obs=False,
                 sel_step=5):

        self.origin       = None
        self.n_components = 0
        self.components   = []
        self.sizedisttype = ''
        self.n_params     = None
        self.parameters   = {}

        # Read grain infor
        if componentnames is not None:
            self.n_components = len(componentnames)
            self.read_dustgrain(componentnames, path=path, from_obs=from_obs, sel_step=sel_step)



        # set the number of size distribution parametres
        if self.n_components > 0:
            self.n_params = []
            for component in self.components:
                self.n_params.append(component.n_sizes)

    

    def read_dustgrain(self, componentnames, path="./", from_obs=False, sel_step=5):
        '''
        | Read in the precomputed/observed dust grain physical properties from files
        | for each grain component.
        |
        |
        | --Params:
        |           componentnames : list of strings - names of dust grain materials
        |           path :  string - names of dust grain materials
        |           from_obs : Boolean - From observed data or precomputed data
        |           sel_step : int - Only use every nth size, faster fitting
        |
        | --Return:
        |          updated DustGrains properties
        |            
        '''

        self.origin = 'observed data' if from_obs else 'precomputed data'
        self.n_components = len(componentnames)
        
        # get the basic grain data  
        for componentname in componentnames:
            dg = DustGrains(componentname, path=path, from_obs=from_obs, sel_step=sel_step)
            self.components.append(dg)


    

    def compute_size_dist(self, x, params):
        '''
        | Like an abstract functions:
        | Compute the size distribution for the input sizes.
        |
        | --Params:
        |           x      : grain-size
        |           params : List/array of size-distribution parameters
        |
        | --Return:
        |          Size distribution as a function of x
        |            
        '''
        return params

    def set_size_dist_parameters(self, params):
        '''
        | Like an abstract functions:
        | Set the size distributions for each component based on
        |
        | --Params:
        |           params : List/array of size-distribution parameters
        |
        | --Return:
        |          Update the size_dist
        |            
        '''
        pass

    def set_size_dist(self, params):
        '''
        | Set the size distributions for each component based on
        | the parameters of the functional form of the distributions.
        |
        | --Params:
        |           params : List/array of size-distribution parameters
        |
        | --Return:
        |          Update the size_dist
        |            
        '''
        k1 = 0
        for k, component in enumerate(self.components):
            delta_val = self.n_params[k]
            k2        = k1 + delta_val

            # Update the values/properties of self.components (which is a list of DustGrain objects)
            # sizedist = params[0] * np.power(x, -1. * params[1])
            component.size_dist = self.compute_size_dist(component.sizes[:], params[k1:k2])
            k1 += delta_val

    


    def eff_grain_props(self, OD, predict_all=False):
        '''
        | Compute the effective grain properties of the ensemble of grain sizes and compositions.
        |
        |
        | --Params:
        |           OD          : ObsData object - All the observed data 
        |           predict_all : Regardless of the ObsData, compute all possible observations
        |
        | --Return:
        |          Dict : Dictonary of predicted observations
        |                 E.g., keys of cext, natoms, emission, albedo, g
        |            
        '''

        # storage for results
        _cabs = np.zeros(self.components[0].n_lambda)
        _csca = np.zeros(self.components[0].n_lambda)
        _natoms = {}

        if OD.fit_ir_emission or predict_all:
            _emission = np.zeros(self.components[0].n_lambda_emis)

        if OD.fit_scat_a or predict_all:
            _scat_a_cext = np.zeros(self.components[0].n_albedo_lambda)
            _scat_a_csca = np.zeros(self.components[0].n_albedo_lambda)

        if OD.fit_scat_g or predict_all:
            _g = np.zeros(self.components[0].n_g_lambda)
            _scat_g_csca = np.zeros(self.components[0].n_g_lambda)

        # self.components.append(DustGrain_Object)
        for component in self.components:
            results = component.eff_grain_props(OD, predict_all=predict_all)

            _tcabs = results["cabs"]
            _tcsca = results["csca"]
            _cabs += _tcabs
            _csca += _tcsca

            # for the depletions (# of atoms), a bit more careful work needed
            _tnatoms = results["natoms"]
            for aname in _tnatoms.keys():
                if aname in _natoms.keys():
                    _natoms[aname] += _tnatoms[aname]
                else:
                    _natoms[aname] = _tnatoms[aname]

            if OD.fit_ir_emission or predict_all:
                _temission = results["emission"]
                _emission += _temission

            if OD.fit_scat_a or predict_all:
                _tscat_a_cext = results["scat_a_cext"]
                _tscat_a_csca = results["scat_a_csca"]
                _scat_a_cext += _tscat_a_cext
                _scat_a_csca += _tscat_a_csca

            if OD.fit_scat_g or predict_all:
                _tg = results["g"]
                _tscat_g_csca = results["scat_g_csca"]
                _g += _tscat_g_csca * _tg
                _scat_g_csca += _tscat_g_csca

        results = {}
        results["cabs"] = _cabs
        results["csca"] = _csca
        results["natoms"] = _natoms

        if OD.fit_ir_emission or predict_all:
            results["emission"] = _emission

        if OD.fit_scat_a or predict_all:
            results["albedo"] = _scat_a_csca / _scat_a_cext

        if OD.fit_scat_g or predict_all:
            results["g"] = _g / _scat_g_csca

        return results

    def save_results(self, filename, OD):
        '''
        | Save fitting results to a pickle file.
        | Results include the size distribution and all predicted observations.
        |
        |
        | --Params:
        |           filename : string - Name of the file to save the results
        |           OD       : ObsData object - All the observed data 
        |
        | --Return:
        |            
        '''
                
        res = {}
        n_components           = len(self.components)
        res['n_components']    = n_components
        res['component_names'] = []
        

        for k, component in enumerate(self.components):
            res['component_names'].append(component.name)


        res['size_dist_model'] = self.sizedisttype
        res['best_params']     = self.parameters
        res['sizes']           = []
        res['size_dist']       = []
        

        # output the dust grain size distribution
        for component in self.components:
            res['sizes'].append( component.sizes )
            res['size_dist'].append(component.size_dist)
        

        # output the resulting observable parameters
        results = self.eff_grain_props(OD, predict_all=True)
        cabs    = results['cabs']
        csca    = results['csca']
        natoms  = results['natoms']

        # Abundance
        res['abundance'] = natoms                                      # abundances in units of # atoms/1e6 H atoms

        # extinction
        res['cabs'] = cabs
        res['csca'] = csca
        res['ext']  = 1.086 * (cabs + csca)                            # extinction in A(lambda)/N(HI) units
        res['wavelengths'] = self.components[0].lambda_

        
        # emission
        res['emission_wavelengths'] = self.components[0].lambda_emis
        res['emission']             = results['emission']              # dust scattering phase function asymmetry


        # albedo
        res['albedo']             = results['albedo']
        res['albedo_wavelengths'] = self.components[0].albedo_lambda


        # g
        res['g']             = results['g']
        res['g_wavelengths'] = self.components[0].lambda_scat_g


        res['ext_components']  = []       # extinction in A(lambda)/N(HI) units
        res['cabs_components'] = []
        res['csca_components'] = []

        res['emission_components'] = []   # emission MJy/sr/H atom units
        res['albedo_components']   = []
        res['g_components']        = []
        

        for k, component in enumerate(self.components):
            results = component.eff_grain_props(OD, predict_all=True)
            tcabs = results['cabs']
            tcsca = results['csca']

            res['cabs_components'].append(tcabs)
            res['csca_components'].append(tcsca)
            res['ext_components'].append( 1.086 * (tcabs + tcsca) )  

            res['emission_components'].append(results['emission'])

            res['albedo_components'].append(results['albedo'])

            res['g_components'].append( results['g'] )


        with open(filename, 'wb') as f:
            pickle.dump(res, f)




    def save_best_results(self, tofile, fit_params_best, obsdata):

        self.set_size_dist_parameters(fit_params_best)

        self.set_size_dist(fit_params_best)

        # save the best fit size distributions
        self.save_results(tofile, obsdata)


# ================================================================


class MRN77(DustModel):
    '''
    | Dust model that uses powerlaw size distributions with min/max sizes (MRN).
    | Same keywords and attributes as the parent DustModel class.
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

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.sizedisttype = 'MRN'
        self.n_params = [4] * self.n_components

        # min and max grain radii for MRN distribution
        # AMIN     = 0.005   # micron
        # AMAX     = 0.3     # micron

        for component in self.components:
            self.parameters[component.name] = {
                'C': 1.e-25,
                'p': 3.5,
                'amin': 1.e-7,
                'amax': 1.e-3,
            }

    def compute_size_dist(self, x, params):
        '''
        | Compute the size distribution for the input sizes.
        | Powerlaw size distribution (aka MRN size distribution)
        |
        | sizedist = A*a^(-p)
        |
        | where:
        |    a = grain size,
        |    A = amplitude,
        |    p = exponent of power law,
        |    amin = min grain size,
        |    amax = max grain size,
        |
        |
        | --Params:
        |           x :  array of floats - grains sizes [micron]
        |           params : array of floats - Size distribution parameters
        |
        | --Return:
        |           Size distribution as a function of x (grain-size)
        |            
        '''
        sizedist = params[0] * np.power(x, -1. * params[1])
        (indxs,) = np.where(np.logical_or(x < params[2], x > params[3]))

        if len(indxs) > 0:
            sizedist[indxs] = 0.

        return sizedist

    def set_size_dist_parameters(self, params):
        '''
        | Set the size distribution parameters in the object dictonary.
        |
        |
        |
        | --Params:           
        |          params : array of floats - Size distribution parameters 
        |           
        |           
        |           
        |
        | --Return:
        |            
        '''
        k1 = 0
        for k, component in enumerate(self.components):
            k2      = k1 + self.n_params[k]
            cparams = params[k1:k2]
            k1      += self.n_params[k]
            self.parameters[component.name] = {
                'C': cparams[0],
                'alpha': cparams[1],
                'a_min': cparams[2],
                'a_max': cparams[3],
            }





class WD01(DustModel):
    '''
    | Dust model that uses the Weingartner & Draine (2001) size distributions.
    |
    |
    |
    | --Params:           
    |          Same kewyords and attributes as the parent DustModel class.
    |           
    |           
    |           
    |
    | --Return:
    |            
    '''

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.sizedisttype = 'WD'

        # set the number of size distribution parametres
        if self.n_components > 0:
            self.n_params = []
            for component in self.components:
                if component.name == 'silicates':
                    self.n_params.append(4)
                    self.parameters['silicates'] = {
                        'C_s': 1.33e-12,
                        'a_ts': 0.171e4,
                        'alpha_s': -1.41,
                        'beta_s': -11.5,
                    }
                elif component.name == 'carbonaceous':
                    self.n_params.append(6)
                    self.parameters['carbonaceous'] = {
                        'C_g': 4.15e-11,
                        'a_tg': 0.00837e4,
                        'alpha_g': -1.91,
                        'beta_g': -0.125,
                        'a_cg': 0.499e4,
                        'b_C': 3.0e-5,
                    }
                else:
                    raise ValueError(
                        component.name + ' is invalid! Please select silicates/carbonaceous.'
                    )

    def compute_size_dist(self, x, params):
        '''
        | Compute the size distribution for the input sizes.
        |
        |
        | --Params:
        |           x :  array of floats - grains sizes in cm
        |           params : array of floats - Size distribution parameters
        |
        | --Return:
        |           Size distribution as a function of x (grain-size)
        |            
        '''

        # input grain sizes are in cm, but needed in Angstroms
        a = x * 1.e8


        if len(params) == 6:
            # carbonaceous, see Equations 2 and 4 in the paper WD01
            # carbonaceous, see Table 1 in WD2001 for details
            # a_t and a_c here are in Angstrom (but in the Table 1 are in micron)
            C, a_t, alpha, beta, a_c, input_bC = params

            sigma = 0.4
            rho   = 2.24                              # in g/cm^3 for graphite
            mC    = 12.0107 * 1.660e-24               # g, Mass of carbon atom in grams (12 m_p)
            
            a01    = 3.5                              # in A
            a01_cm = a01 * 1.e-8                      # in cm
            bC1    = 0.75 * input_bC

            
            B_1    = (3./(2.*np.pi)**1.5) * (np.exp(-4.5*sigma**2) / (rho * a01_cm**3 * sigma)) * bC1 * mC /\
                      (1. + erf( 3.*sigma/np.sqrt(2) + np.log(a01/3.5)/(sigma*np.sqrt(2)) ) )
            
            
            a02    = 30.                              # in A
            a02_cm = a02 * 1.e-8                      # in cm
            bC2    = 0.25 * input_bC

            B_2    = (3./(2.*np.pi)**1.5) * (np.exp(-4.5*sigma**2) / (rho * a02_cm**3 * sigma)) * bC2 * mC /\
                      (1. + erf( 3.*sigma/np.sqrt(2) + np.log(a02/3.5)/(sigma*np.sqrt(2)) ) )


            # Note, x in [cm]
            D      = (B_1/x) * np.exp( -0.5*( np.log(a/a01)/sigma )**2 ) + \
                     (B_2/x) * np.exp( -0.5*( np.log(a/a02)/sigma )**2 )
            
            if beta >= 0.:
                F_g = 1. + beta * a / a_t
            else:
                F_g = 1. / (1. - beta * a / a_t)


            
            # Very Small Grain
            (id_vsg,)  = np.where( a < 3.5)
            if np.size(id_vsg) != 0:
                D[id_vsg] = 0.

            
            # Equation 4 in the paper, here for Carbonaceous
            fn_graphite = np.full((len(a)), 1.)

            (indxs,) = np.where( a >= a_t )
            if np.size(indxs) != 0:
                fn_graphite[indxs] = np.exp( -( (a[indxs]-a_t) / a_c )**3 )

            # Note: x in [cm]
            cb_sizedist = (C/x) * (a/a_t)**alpha * F_g * fn_graphite           # cm^-4 per n_H

            # Return the total: VSG + Carbonaceous
            return D + cb_sizedist    




        else:

            # silicates, see dust_scattering ( sizedist/WD01.py -> get_params() )
            C, a_t, alpha, beta = params
            a_c                 = 0.1e4   # a_c,s ~0.1 micron (0.1e4 Angstrom) for Milky Way dust, see Section 3.1.1 of WD01
            input_bC            = None

            fn_silicate  = np.full((len(a)), 1.)
            (indxs,)     = np.where(a > a_t)

            if np.size(indxs) != 0:
                fn_silicate[indxs] = np.exp( -( (a[indxs]-a_t)/a_c )**3 )


            if beta >= 0.:
                F_s = 1. + beta * a / a_t
            else:
                F_s = 1. / (1. - beta * a / a_t)

            return (C/x) * (a/a_t)**alpha * F_s * fn_silicate              # cm^-4 per n_H

    

    def set_size_dist_parameters(self, params):
        '''
        | Set the size distribution parameters in the object dictonary.
        |
        |
        |
        | --Params:           
        |          params : array of floats - Size distribution parameters 
        |           
        |           
        |           
        |
        | --Return:
        |            
        '''
        k1 = 0
        for k, component in enumerate(self.components):
            k2      = k1 + self.n_params[k]
            cparams = params[k1:k2]
            k1      += self.n_params[k]
            if component.name == 'silicates':
                self.parameters['silicates'] = {
                    'C_s': cparams[0],
                    'a_ts': cparams[1],
                    'alpha_s': cparams[2],
                    'beta_s': cparams[3],
                }
            else:
                self.parameters['carbonaceous'] = {
                    'C_g': cparams[0],
                    'a_tg': cparams[1],
                    'alpha_g': cparams[2],
                    'beta_g': cparams[3],
                    'a_cg': cparams[4],
                    'b_C': cparams[5],
                }