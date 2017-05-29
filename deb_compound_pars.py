# -*- coding: utf-8 -*-
'''
Compound parameter relationships in DEB
'''
from types import SimpleNamespace
import pandas as pd
import numpy as np


def calculate_compound_pars(primarypars):
    '''Calculate DEB compound parameters from primary parameters'''

    df = pd.DataFrame(columns=['Min', 'Max', 'Value', 'Dimension', 'Unit',
                               'Description'])

    v = SimpleNamespace(**primarypars)
    Em = fEm(v.pAm, v.v)
    g = fg(v.EG, v.kappa, Em)
    Lm = fLm(v.kappa, v.pAm, v.pM)
    kM = fkM(v.pM, v.EG)
    kappaG = fkappaG()

    df.loc['Em'] = (0, np.nan, Em, 'e L**-3', 'J/m', 'Max reserve density')
    df.loc['g'] = (0, np.nan, g, '-', '-', 'Energy investment ration')
    df.loc['Lm'] = (0, np.nan, Lm, 'l', 'm', 'Maximum structural length')
    df.loc['kM'] = (0, np.nan, kM, 't**-1', '1/d', 'Somatic maintenance rate')
    df.loc['kappaG'] = (0, np.nan, kappaG, '-', '-', 'Fraction of growth energy fixed in structure')

    return df


def fEm(pAm, v):
    '''Max reserve density [J m**-3]'''
    return pAm / v


def fLm(kappa, pAm, pM):
    '''Max structural length {L}'''
    return kappa * pAm / pM


def fg(EG, kappa, Em):
    '''Energy investment ratio'''
    return EG / (kappa * Em)


def fkm(pM, EG):
    '''Somatic maintenance rate'''
    return pM / EG

def fkappaG():
    '''Fraction of growth energy fixed in structure [-]'''
    #AMP pseudodata value
    return 0.8

def fkM(pM, EG):
    '''Somatic maintenance rate'''
    return pM/EG
