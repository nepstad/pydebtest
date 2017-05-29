# -*- coding: utf-8 -*-
'''
Auxillary equations for DEB, connection with observables.
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class AuxPars:
    def __init__(self, debmod):
        self.auxpars = get_aux_pars_pandas()
        self.debmod = debmod

        self.vars_avail = ['Lw', 'Vw', 'Ww']

    def __getattr__(self, name):
        return self.auxpars.loc[name, 'Value']

    def __call__(self, name):
        if name == 'Lw':
            V = self.debmod.V
            deltaM = self.auxpars.loc['deltaM', 'Value']
            return physical_length(self.debmod.V, deltaM)
        elif name == 'Ww':
            E = self.debmod.E
            V = self.debmod.V
            return dry_weight(E, V, self.muE, self.dV, self.wV, self.wE)
        elif name == 'Vw':
            E = self.debmod.E
            V = self.debmod.V
            ER = self.debmod.ER
            return physical_volume(V, E, ER, self.wE, self.dE, self.muE)
        else:
            raise RuntimeError('Auxillary parameter {0} not available'.format(name))

    def plot_observables(self, fig=None):
        if fig is None:
            fig, ax = plt.subplots(1, 3, figsize=(18, 6))
        else:
            ax = fig.axes

        t = self.debmod.times

        ax[0].plot(t, self('Lw'), label='Physical length')
        ax[0].set_title('Organism length')
        ax[0].set_xlabel('Time [days]')
        ax[0].set_ylabel('Length [cm]')

        Vw, V, VE, VER = self('Vw')
        #ax[1].plot(t, Vw, label='Physical volume of organism')
        ax[1].stackplot(t, V, VE, VER, alpha=0.75,
                        labels=('Structure', 'Reserve', 'Reproduction'))
        ax[1].legend(loc='best')
        ax[1].plot(t, Vw, color='k', ls='-', lw=1.5)
        ax[1].set_title('Physical volume of organism')
        ax[1].set_xlabel('Time [days]')
        ax[1].set_ylabel('Volume [cm**3]')

        ax[2].plot(t, self('Ww'), label='Total organism dry weight')
        ax[2].set_title('Total organism dry weight')
        ax[2].set_xlabel('Time [days]')
        ax[2].set_ylabel('Weight [g]')

        return fig


def get_aux_pars_pandas(pripars=None):
    '''Set up auxillary parameters with standard animal value.'''

    df = pd.DataFrame(columns=['Min', 'Max', 'Value', 'Dimension', 'Units', 'Description'])
    
    #Value from from addchem.m
    df.loc['muE'] = (0, np.nan,  550000, '', 'J/mol', 'Specific chemical potential of reserve')
    df.loc['muX'] = (0, np.nan,  525000, '', 'J/mol', 'Specific chemical potential of food')
    df.loc['dV'] = (0, np.nan,  0.1, '', 'g/cm**3', 'Specific density of dry structure')
    df.loc['dE'] = (0, np.nan,  0.1, '', 'g/cm**3', 'Specific density of reserve')
    df.loc['wV'] = (0, np.nan, 24.6, '', 'g/mol', 'C-molar weight of dry structure')
    df.loc['wE'] = (0, np.nan, 23.9, '', 'g/mol', 'C-molar weight of dry reserve')
    df.loc['deltaM'] = (0, np.nan, 0.1, '', '-', 'Shape parameter')

    if pripars is not None:
        pp = pripars['Value']
        pAm = pp['pAm'] 
        kappaX = pp['kappaX']
        Fm = pp['Fm']
        muX = df.loc['muX', 'Value']
        K = pAm/(kappaX * Fm * muX)
        df.loc['K'] = (0, np.nan, K, '# L**-3', '', 'Half saturation constant (food)')

    return df

    
def physical_length(V, deltaM):
    '''Calculate physical length from structural volume'''
    return np.cbrt(V) / deltaM


def dry_weight(E, V, muE, dV, wV, wE):
    '''Calculate dry length from reserve, structure and maturity'''
    ME = 1/muE * E
    MV = dV/wV * V
    Ww = wV*MV + wE*ME
    return Ww


def physical_volume(V, E, ER, wE, dE, muE):
    '''Calculate physical volume from reserve, structure and reproduction buffer'''
    VE = E * wE/(dE*muE)
    VER = ER * wE/(dE*muE)
    Vw = V + VE + VER
    return Vw, V, VE, VER

