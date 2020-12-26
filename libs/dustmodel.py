import math
import pickle
import numpy as np
from scipy.special import erf

from astropy.io import fits

from .dustgrains import DustGrains

__all__ = ['DustModel', 'MRN77', 'WD01']


class DustModel(object):
    """
    Full dust model including arbirary size and composition distributions.
    Includes the physical properties of the individual dust grains.

    Dust model that has each bin as an independent variable in the
    grain size distribution providing a truly arbitrary specification.

    Parameters
    ----------
    componentnames : str list, optional
        if set, then read in the grain information from files
    path : str, optional
        path to grain files
    dustmodel : DustModel object, optional
        if set, create the grain info on the obsdata wavelengths using
        the input dustmodel grain information
    obsdata : ObsData object, optional
        observed data information

    Attributes
    ----------
    origin : string
        origin of the dust grain physical properties
        allowed values are 'files' and 'onobsdata'
    n_components : int
        number of dust grain components
    components : array of DustGrain objects
        one DustGrain object per component
    sizedisttype : string
        functional form of component size distributions
    n_params : ints
        number of size distribution parameters per grain component
    parameters : dict
        Dictonary of parameters with an entry for each composition
        each entry is then a dictonary giving the value by parameter name.
        For the bins case, the dictonary is empty as the parameters is
        the size distribution.
    """

    def __init__(self,
                 componentnames=None,
                 path="./",
                 from_obs=False,
                 dustmodel=None,
                 obsdata=None,
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



        # if componentnames is not None:
        #     self.read_grain_files(componentnames, path=path, every_nth=every_nth)
        # elif dustmodel is not None:
        #     self.grains_on_obs(dustmodel, obsdata)

        # set the number of size distribution parametres
        if self.n_components > 0:
            self.n_params = []
            for component in self.components:
                self.n_params.append(component.n_sizes)

    

    def read_dustgrain(self, componentnames, path="./", from_obs=False, sel_step=5):
        """
        Read in the precomputed dust grain physical properties from files
        for each grain component.

        Parameters
        ----------
        componentnames : list of strings
            names of dust grain materials
        path : type
            path to files
        sel_step : int
            Only use every nth size, faster fitting

        Returns
        -------
        updated class variables
        """
        self.origin = 'observed data' if from_obs else 'precomputed data'
        self.n_components = len(componentnames)
        
        # get the basic grain data  
        for componentname in componentnames:
            dg = DustGrains(componentname, path=path, from_obs=from_obs, sel_step=sel_step)
            self.components.append(dg)


    

    def compute_size_dist(self, x, params):
        """
        Compute the size distribution for the input sizes.
        For the bins case, just passes the parameters back.  Allows for
        other functional forms of the size distribution with minimal new code.

        Parameters
        ----------
        x : floats
            grains sizes
        params : floats
            Size distribution parameters
            For the arbitrary bins case, the parameters are the number
            of grains per size distribution

        Returns
        -------
        floats
            Size distribution as a function of x
        """
        return params

    def set_size_dist_parameters(self, params):
        """
        Set the size distribution parameters in the object dictonary.
        For the bins case, this does nothing.  Allows for
        other functional forms of the size distribution with minimal new code.

        Parameters
        ----------
        params : floats
            Size distribution parameters
            For the arbitrary bins case, the parameters are the number
            of grains per size distribution
        """
        pass

    def set_size_dist(self, params):
        """
        Set the size distributions for each component based on the
        parameters of the functional form of the distributions.

        Parameters
        ----------
        new_size_dists : type
            Description of parameter `new_size_dists`.

        Returns
        -------
        type
            Description of returned object.

        """
        k1 = 0
        for k, component in enumerate(self.components):
            delta_val = self.n_params[k]
            k2        = k1 + delta_val

            # Update the values/properties of self.components (which is a list of DustGrain objects)
            # sizedist = params[0] * np.power(x, -1. * params[1])
            component.size_dist = self.compute_size_dist(component.sizes[:], params[k1:k2])
            k1 += delta_val

    def eff_grain_props(self, OD, predict_all=False):
        """
        Compute the effective grain properties of the ensemble of grain
        sizes and compositions.

        Parameters
        ----------
        OD : ObsData object
            Observed data object specifically used to determine which
            observations to compute (only those needed for speed)
        predict_all : type
            Regardless of the ObsData, compute all possible observations

        Returns
        -------
        dict
            Dictonary of predicted observations
            E.g., keys of cext, natoms, emission, albedo, g
        """
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
        """
        Save fitting results to a file.  Results include the
        size distribution and all predicted observations.

        Creates a FITS file with the results

        Parameters
        ----------
        filename : str
            Name of the file to save the results
        OD : ObsData object
            All the observed data (may not be needed)
        """
        
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
                if component.name == "silicates":
                    self.n_params.append(4)
                    self.parameters["silicates"] = {
                        "C_s": 1.33e-12,
                        "a_ts": 0.171e4,
                        "alpha_s": -1.41,
                        "beta_s": -11.5,
                    }
                elif component.name == "carbonaceous":
                    self.n_params.append(6)
                    self.parameters["carbonaceous"] = {
                        "C_g": 4.15e-11,
                        "a_tg": 0.00837e4,
                        "alpha_g": -1.91,
                        "beta_g": -0.125,
                        "a_cg": 0.499e4,
                        "b_C": 3.0e-5,
                    }
                else:
                    raise ValueError(
                        "%s grain material note supported" % component.name
                    )

    def compute_size_dist(self, x, params):
        '''
        | Compute the size distribution for the input sizes.
        |
        |
        | --Params:
        |           x :  array of floats - grains sizes
        |           params : array of floats - Size distribution parameters
        |
        | --Return:
        |           Size distribution as a function of x (grain-size)
        |            
        '''

        # input grain sizes are in cm, needed in Angstroms
        a = x * 1.e8

        if len(params) == 6:
            # carbonaceous
            C, a_t, alpha, beta, a_c, input_bC = params
        else:
            # silicates
            C, a_t, alpha, beta = params
            a_c = 0.1e4
            input_bC = None

        # larger grain size distribution
        # same for silicates and carbonaceous grains
        if beta >= 0.:
            Fa = 1. + beta * a / a_t
        else:
            Fa = 1. / (1. - beta * a / a_t)

        Ga = np.full((len(a)), 1.)

        (indxs,)  = np.where(a > a_t)
        Ga[indxs] = np.exp(-1. * np.power((a[indxs] - a_t) / a_c, 3.))

        sizedist  = (C / (1.e-8 * a)) * np.power(a / a_t, alpha) * Fa * Ga

        # very small gain size distribution
        # only for carbonaceous grains
        if input_bC is not None:
            a0 = np.array([3.5, 30.])                 # in A
            bC = np.array([0.75, 0.25]) * input_bC
            sigma = 0.4
            rho = 2.24  # in g/cm^3 for graphite
            mC = 12.0107 * 1.660e-24

            Da = 0.
            for i in range(2):
                Bi = (
                    (3.0 / (np.power(2.0 * np.pi, 1.5)))
                    * (
                        np.exp(-4.5 * np.power(sigma, 2.0))
                        / (rho * np.power(1e-8 * a0[i], 3.0) * sigma)
                    )
                    * (
                        bC[i]
                        * mC
                        / (
                            1.0
                            + erf(
                                (3.0 * sigma / np.sqrt(2.0))
                                + np.log(a0[i] / 3.5) / (sigma * np.sqrt(2.0))
                            )
                        )
                    )
                )

                Da += (Bi / (1e-8 * a)) * np.exp(
                    -0.5 * np.power(np.log(a / a0[i]) / sigma, 2.0)
                )

            sizedist += Da

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