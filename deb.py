# -*- coding: utf-8 -*-
'''
Standard Dynamic Energy Budget model
'''
import numpy as np
import pandas as pd
import scipy.integrate as sid
import lmfit
import matplotlib.pyplot as plt
import seaborn as sns
import corner
from tqdm import tqdm
from collections import namedtuple

from lossfunc import symmetric_loss_function
from deb_aux import physical_volume

EPS = np.finfo(np.double).eps


def get_deb_params_pandas():
    '''Set up primary parameters with standard animal value.
    
    Also include temperature params here for now.

    Use standard animal values from AMP:

    Code	Value	Unit	Description
    v	0.02	cm/d	energy conductance
    kap	0.8	/	allocation fraction to soma
    kap_R	0.95	/	reproduction efficiency
    p_M	18	J/d.cm^3	volume-specific somatic maintenance costs
    k_J	0.002	1/d	maturity maintenance rate coefficient
    kap_G	0.8	/	growth efficiency
    
    '''
    df = pd.DataFrame(columns=['Min', 'Max', 'Value', 'Dimension', 'Units', 'Description'])

    df.loc['Fm'] = (EPS, np.nan, 6.5, 'l L**-2 t**-1', '', 'Specific searching rate')
    df.loc['kappaX'] = (EPS, 1.0, 0.8, '-', '', 'Assimilation efficiency')
    df.loc['pAm'] = (EPS, np.nan, 530.0, 'e L**-2 t**-1', '', 'max specific assimilation rate')
    df.loc['v'] = (EPS, np.nan, 0.02, 'L t**-1', 'cm/d', 'Energy conductance')
    df.loc['kappa'] = (EPS, 1.0, 0.8, '-', '', 'Allocation fraction to soma')
    df.loc['kappaR'] = (EPS, 1.0, 0.95, '-', '', 'Reproduction efficiency')
    df.loc['pM'] = (EPS, np.nan, 18.0, 'e L**-3 t**-1', 'J/d/cm**3', 'Volume-specific somatic maintenance cost')
    df.loc['pT'] = (0.0, np.nan, 0.0, 'e L**-1 t**-1', '', 'Surface-specific somatic maintenance cost')
    df.loc['kJ'] = (EPS, np.nan, 0.002, 't**-1', '', 'Maturity maintenance rate coefficient')
    df.loc['EG'] = (EPS, np.nan, 4184., 'e L**-3', '', 'Specific cost for structure')
    df.loc['EbH'] = (EPS, np.nan, 1e-6, 'e', '', 'Maturity at birth')
    df.loc['EpH'] = (EPS, np.nan, 843.6, 'e', '', 'Maturity at puberty')

    df.loc['TA'] = (EPS, np.nan, 6000.0, 'T', 'K', 'Arrhenius temperature')
    df.loc['Ts'] = (EPS, np.nan, 273.1+20.0, 'T', 'K', 'Reference temperature')

    return df


def get_deb_params(df=None):
    if df is None:
        df = get_deb_params_pandas()
    v = lmfit.Parameters()
    for name, (mi, ma, va, dim, unit, desc) in df.iterrows():
        v.add(name, value=va, min=mi, max=ma, vary=False)

    return v

def params_to_dataframe(params):
    '''Convert lmfit Parameters to Pandas DataFrame'''
    df = get_deb_params_pandas()
    for name, p in params.items():
        df.loc[name, 'Min'] = p.min
        df.loc[name, 'Max'] = p.max
        df.loc[name, 'Value'] = p.value

    return df


def arrhenius_scaling(T, TA, Ts):
    '''Arrhenius temperature scaling relationship'''
    return np.exp(TA/Ts - TA/T)


class Fluxes:
    '''Fluxes in the standard DEB model'''

    def __init__(self):
        # Rates scale with temperature (per time times)
        self.temp_scale_params = ['pAm', 'v', 'pM', 'pT', 'kJ']

    @staticmethod
    def pA(f, V, pAm):
        return f * pAm * np.cbrt(V**2)

    @staticmethod
    def pC(E, V, EG, v, kappa, pS):
        return E * (EG * v * np.cbrt(V**2)  + pS) / (kappa * E + EG * V)

    @staticmethod
    def pS(V, pM, pT):
        return pM*V + pT*V**(2/3.)

    @staticmethod
    def pG(kappa, pC, pS):
        return kappa*pC - pS

    @staticmethod
    def pJ(EH, kJ):
        return kJ * EH

    @staticmethod
    def pR(kappa, pC, pJ):
        return (1 - kappa) * pC - pJ

    def __call__(self, t, forcings, state, params):
        '''Evaluate fluxes for given time, state and environment'''
        #Unpack state
        E, V, EH, ER = state

        #Food level
        f = forcings['f']

        #Temperature
        if 'T' in forcings.keys():
            T = forcings['T'](t)
        else:
            T = params['Ts']

        #Temperature scaling of rates
        TA = params['TA']
        Ts = params['Ts']
        ats = arrhenius_scaling(T, TA, Ts)
        pAm = ats * params['pAm']
        v = ats * params['v']
        pM = ats * params['pM']
        pT = ats * params['pT']
        kJ = ats * params['kJ']

        #s = pd.Series(name='Fluxes')
        s = dict()
        s['pA'] = Fluxes.pA(f(t), V, pAm)
        s['pS'] = Fluxes.pS(V, pM, pT)
        s['pC'] = Fluxes.pC(E, V, params['EG'], v, params['kappa'], s['pS'])
        s['pG'] = Fluxes.pG(params['kappa'], s['pC'], s['pS'])
        s['pJ'] = Fluxes.pJ(EH, kJ)
        s['pR'] = Fluxes.pR(params['kappa'], s['pC'], s['pJ'])
         
        return s


class DEBStandard(Fluxes):
    '''Standard DEB model for reserve energy, structural volume, maturity
    energy and reproduction buffer (E, V, EH, ER)
    '''

    def __init__(self, forcings, pripars=None):
        self.params = get_deb_params(pripars)
        self.fluxes = Fluxes()
        self.forcings = forcings

        if 'T' in forcings.keys():
            if callable(forcings['T']):
                print('Applying dynamic temperature adjustment of rates')
            else:
                print('Applying static temperature adjustment of rates.')

        # Parameters for ODE solver and data fit
        self._ode_nsteps = 6000

        self.dy = np.zeros(4)

    def _rhs(self, t, y, params, forcings):  
        '''Standard DEB model equation set

            dE/dt = pA - pC       Reserve energy
            dV/dt = pG / [EG]     Structural volume
            dEH/dt = pR(EH < EHp) Maturity energy
            dER/dt = pR(EH = EHp) Reproduction buffer energy

        State vector indexing:
        y[0] = E, y[1] = V, y[2] = EH, y[3] = ER
        '''

        v = params
        dy = self.dy

        # Current state vector values
        E = y[0]
        V = y[1]
        EH = y[2]
        ER = y[3]

        #Calculate fluxes
        flux = self.fluxes(t, forcings, [E, V, EH, ER], params)

        # Reserve energy equation
        dE = flux['pA'] - flux['pC']

        # Structural volume equation
        dV = flux['pG']  /  v['EG']

        #Maturity and reproductive buffer energy equations 
        if(EH < v['EpH']):
            dEH = flux['pR']
            dER = 0.0
        else:
            dEH = 0.0
            dER = flux['pR']

        dy[:] = [dE, dV, dEH, dER]

        return dy

    def _solve(self, params, y0, times):
        '''Solver for model ODE.
        
        Returns solution to model ODE at times <times>, given model parameters
        <params> and the time-dependent exposure profile function <cd>, for
        given initial conditions <y0>.
        '''

        # Trying explicit bdf for stiff equations, since lsoda complains
        #r = sid.ode(cls.currhs).set_integrator('vode', nsteps=1000, method='bdf')
        r = sid.ode(self._rhs).set_integrator('lsoda', nsteps=self._ode_nsteps,
                                              rtol=1e-6)
        r.set_initial_value(y0, times[0])
        r.set_f_params(params.valuesdict(), self.forcings)
        sols = [y0]
        for t in times[1:]:
            r.integrate(t)
            sols.append(r.y)

        y = np.array(sols)
        return y, r

    def predict(self, y0, times):
        y, r = self._solve(self.params, y0, times)
        self.solver_result = r
        self.times = times
        self.E = y[:, 0]
        self.V = y[:, 1]
        self.EH = y[:, 2]
        self.ER = y[:, 3]


    def plot_state(self, fig=None):
        '''Plot state variables, run predict() first'''

        #Maximum structural length
        Lm = self.params['kappa']*self.params['pAm']/self.params['pM']

        if not fig:
            fig, ax = plt.subplots(2, 2, figsize=(10, 10))
            ax = ax.flatten()
        else:
            ax = fig.axes

        t = self.times
        f = self.forcings['f']

        ax[0].plot(t, self.E)
        ax[0].set_title('Reserve (E)')
        ax[0].set_ylabel('E [J]')

        ax[1].plot(t, self.V)
        ax[1].set_title('Structure (V)')
        ax[1].set_ylabel('V [cm**3]')

        ax[2].plot(t, self.EH)
        ax[2].set_title('Maturity (EH)')
        ax[2].set_xlabel('Time [days]')
        ax[2].set_ylabel('EH [J]')

        ax[3].plot(t, self.ER)
        ax[3].set_title('Reproduction buffer (ER)')
        ax[3].set_xlabel('Time [days]')
        ax[3].set_ylabel('ER [J]')

        ax[1].axhline(Lm**3, ls='--')

        return fig


class DEBFit:
    def __init__(self, debmodel, initial_state, auxpars):
        self.debmodel = debmodel
        self.initial_state = initial_state
        self.auxpars = auxpars

        #This maps the DEB state variables to observables (data variables)
        #Hard-coded default is physical volume auxillary variable
        self.auxmap = self.get_physical_volume

        #Max iterations for Nelder-Mead optimization
        self._fit_maxiter = 3000

    def get_physical_volume(self, S, p):
        E = S[:, 0]
        V = S[:, 1]
        ER = S[:, 3]
        ap  = self.auxpars
        Vw, V, VE, VER = physical_volume(V, E, ER, ap.wE, ap.dE, ap.muE)
        return Vw

    def objective(self, params, datasets):
        '''Objective function to minimize for parameter estimation'''

        modpreds = []
        weights = []
        for data in datasets:
            times = data.index.values
            y0 = self.initial_state

            #Solve DEB model equations
            y, _ = self.debmodel._solve(params, y0, times)

            #Here we need to map the DEB state onto data variables via
            #observable auxillary function
            o = self.auxmap(y, params)
            modpreds.append(o)

            #Use weights=1 for now
            weights.append(np.ones(times.size))
            
        #Calculate loss function
        loss = symmetric_loss_function(datasets, modpreds, weights)

        return loss


    def fit(self, data, progressbar=True):
        if progressbar:
            pbar = tqdm(total=self._fit_maxiter)
            def objective_wrapper(*args, **kwargs):
                pbar.update(1)
                return self.objective(*args, **kwargs)
        else:
            objective_wrapper = self.objective

        # Run the minimizer with the simplex method, simultanous fit to all data
        result = lmfit.minimize(objective_wrapper, self.debmodel.params, 
                                args=(data,), method='nelder', tol=1e-9,
                                options=dict(maxfev=self._fit_maxiter))
        self.params = result.params

        if progressbar:
            pbar.close()

        return result


if __name__ == '__main__':
    forcings = dict(f=lambda t: t)
    state = [1, 1, 1, 1]
    pars = get_deb_params()

    flux = Fluxes()

    print(flux(0.0, forcings, state, pars))

    debmod = DEBStandard({'f': lambda t: 0.1*t})
    y0 = [1, 1, 0, 0]
    print(debmod._rhs(0.5, y0, debmod.params, debmod.forcings))

    y, res = debmod._solve(debmod.params, y0, [0, 1, 2])
    print('Integration successful: ', res.successful())
    print('Stiff ODE: ', res.stiff)
    print(y)
