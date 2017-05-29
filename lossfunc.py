# -*- coding: utf-8 -*-
'''
Loss functions for optimization and parameter estimation with data.
'''
import numpy as np

def symmetric_loss_function(datasets, modpreds, weights):
    '''Symmetric loss functions'''

    #Loop over datasets, weights and model predictions
    sselist = []
    for d, m, w in zip(datasets, modpreds, weights):

        #Calculate SSE for this dataset
        sse = symmetric_squared_error(d, m, w)
        sselist.append(sse)


    #Calculate symmetric loss (mean of SSEs)
    smse = np.mean(sselist)

    return smse


def symmetric_squared_error(data, model, weights):
    '''Symmetric squared error (SSE)'''

    #Shorthands
    p = model
    d = data
    w = weights

    #Sum of weights
    ws = weights.sum() 

    #Mean of model predictions 
    pm = np.mean(p)

    #Mean of data  
    dm = np.mean(d)

    #Compute SSE
    sse = np.sum(weights/w * (p - d)**2 / (pm**2 + dm**2))

    return sse
