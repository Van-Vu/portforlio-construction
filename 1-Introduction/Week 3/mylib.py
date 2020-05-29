import pandas as pd
import numpy as np

def readcsv(filename):
    data = pd.read_csv(filename, header = 0, index_col = 0, parse_dates = True)
    data = data/100
    data.index = pd.to_datetime(data.index, format='%Y%m').to_period('M')
    return data

def annualized_return(data):
    monthly_returns = (data + 1).prod()**(1/data.shape[0]) - 1
    return (monthly_returns + 1)**12 - 1

def annualized_vol(data):
    monthly_vol = data.std()
    return monthly_vol * np.sqrt(12)

def sharpe_ratio(data, risk_free_rate):
    return (annualized_return(data) - risk_free_rate) / annualized_vol(data)

def drawdown(returns, starting_wealth = 1000):
    wealth_index = starting_wealth * (1 + returns).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdowns = (wealth_index - previous_peaks) / previous_peaks
    return pd.DataFrame({
        "Wealth": wealth_index,
        "Peaks" : previous_peaks,
        "Drawdown": drawdowns
    })

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

def portfolio_ret(weights, ret):
    return weights.T @ ret

def portfolio_vol(weights, vol):
    return (weights.T @ vol @ weights)**0.5

def potfolio_frontier_2(num_points, history_return, cov):
    portfolio_weights = [np.array([w, 1-w]) for w in np.linspace(0, 1, num_points)]
    final_ret = [portfolio_ret(w, history_return) for w in portfolio_weights]
    final_vol = [portfolio_vol(w, cov) for w in portfolio_weights]
    frontier = pd.DataFrame({"Returns": final_ret, "Volatility": final_vol})
    frontier.plot.line(y="Returns", x="Volatility", style=".-")

from scipy.optimize import minimize
def minimize_vol(target_return, history_return, covariance):
    asset_number = history_return.shape[0]
    init_guess = np.repeat(1/asset_number, asset_number)
    bounds = ((0.0, 1.0),)*asset_number
    return_constraint = {"type": "eq", 
                        "args": (history_return,),
                        "fun": lambda weight, history_return: portfolio_ret(weight, history_return) - target_return}
    weightsum_constraint = {"type": "eq",
                        "fun": lambda weights: np.sum(weights) - 1}

    result = minimize(portfolio_vol, init_guess, method="SLSQP",
        args=(covariance,),
        constraints=(return_constraint, weightsum_constraint),
        bounds = bounds)
    return result.x

def optimize_weights(num_points, history_return, cov):
    target_list = np.linspace(history_return.min(), history_return.max(), num_points)
    weights = [minimize_vol(target_return, history_return,cov) for target_return in target_list]
    return weights

def potfolio_frontier_n(num_points, history_return, cov):
    portfolio_weights = optimize_weights(num_points, history_return, cov)
    final_ret = [portfolio_ret(w, history_return) for w in portfolio_weights]
    final_vol = [portfolio_vol(w, cov) for w in portfolio_weights]
    frontier = pd.DataFrame({"Returns": final_ret, "Volatility": final_vol})
    return frontier.plot.line(y="Returns", x="Volatility", style=".-")

def max_sharpe_ratio(riskfree_rate, history_return, covariance):
    asset_number = history_return.shape[0]
    init_guess = np.repeat(1/asset_number, asset_number)
    bounds = ((0.0, 1.0),)*asset_number
    weightsum_constraint = {"type": "eq",
                        "fun": lambda weights: np.sum(weights) - 1}

    def neg_sharpe_ratio(weight, riskfree_rate, history_return, covariance):
        portfolio_rets = portfolio_ret(weight, history_return)
        portfolio_vols = portfolio_vol(weight, covariance)
        neg_sharpe = - (portfolio_rets - riskfree_rate) / portfolio_vols
        #print(neg_sharpe)
        return neg_sharpe

    result = minimize(neg_sharpe_ratio, init_guess, method="SLSQP",
        args=(riskfree_rate, history_return, covariance,),
        constraints=(weightsum_constraint),
        bounds = bounds)
    print("Optimizer:")
    print(result)
    return result.x

def gmv(cov):
    asset_number = cov.shape[0]
    gmv_weight = max_sharpe_ratio(0, np.repeat(1,asset_number), cov)
    return gmv_weight

def portfolio_frontier_n_with_msr(riskfree_rate, num_points, history_return, cov, show_ew = False, show_gmv = False):
    graph = potfolio_frontier_n(num_points, history_return, cov)
    msr = max_sharpe_ratio(riskfree_rate, history_return, cov)
    msr_ret = portfolio_ret(msr, history_return)
    msr_vol = portfolio_vol(msr, cov)    
    cml_x = [0.0, msr_vol]
    cml_y = [riskfree_rate, msr_ret]
    print("Capital Market Line:")
    print("Volatility:", msr_vol)
    print("Return:", msr_ret)
    graph.plot(cml_x, cml_y, marker="o", linestyle="dashed")
    if show_ew:
        asset_number = history_return.shape[0]
        ew_weights = np.repeat(1/asset_number, asset_number)
        ew_ret = portfolio_ret(ew_weights, history_return)
        ew_vol = portfolio_vol(ew_weights, cov)
        graph.plot(ew_vol, ew_ret, marker="o", markersize=10)
    if show_gmv:
        gmv_weights = gmv(cov)
        gmv_ret = portfolio_ret(gmv_weights, history_return)
        gmv_vol = portfolio_vol(gmv_weights, cov)
        graph.plot(gmv_vol, gmv_ret, marker="o", markersize=8, color="blue")
