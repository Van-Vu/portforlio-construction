import pandas as pd

def readcsv(filename):
    data = pd.read_csv(filename, header = 0, index_col = 0, parse_dates = True)
    data = data/100
    data.index = pd.to_datetime(data.index, format='%Y%m').to_period('M')
    return data

def skewness(data):
    demeaned_return = (data - data.mean())**3
    price = demeaned_return.mean()
    vol = data.std(ddof=0)**3
    skewness = (price / vol).sort_values()
    
    return skewness

def kurtosis(data):
    kdemeaned_return = (data - data.mean())**4
    kprice = kdemeaned_return.mean()
    kvol = data.std(ddof=0)**4
    kurtosis = (kprice / kvol).sort_values()
    
    return kurtosis

import scipy.stats
def is_normal(data):
    statistic, p_value = scipy.stats.jarque_bera(data)
    return [statistic, p_value]

def semidiviation(data):
    return data[data<0].std(ddof=0)

import numpy as np
def var_historic(data, level = 5):
    if isinstance(data, pd.DataFrame):
        return data.aggregate(var_historic, level = level)
    elif isinstance(data, pd.Series):
        return -np.percentile(data, level)
    else:
        raise TypeError('expected DataFrame or Series')

from scipy.stats import norm
def var_gaussian(data, level = 5):
    z_score = norm.ppf(level/100)
    return -(data.mean() + z_score*data.std(ddof=0))

def var_cornishfisher(data, level = 5):
    z = norm.ppf(level/100)
    s = skewness(data)
    k = kurtosis(data)
    z_score = (z 
        + (z**2 - 1) * s / 6
        + (z**3 - 3*z) * (k - 3) / 24
        - (2*z**3 - 5*z) * s**2 / 36)
    return -(data.mean() + z_score*data.std(ddof=0))

def cvar_historic(data, level = 5):
    if isinstance(data, pd.DataFrame):
        return data.aggregate(cvar_historic, level = level)
    elif isinstance(data, pd.Series):
        values = data < -var_historic(data, level)
        return - data[values].mean()

def cvar_gaussian(data, level = 5):
    if isinstance(data, pd.DataFrame):
        return data.aggregate(cvar_gaussian, level = level)
    elif isinstance(data, data.Series):
        values = data < - var_gaussian(data)
        return data[values].mean()