import pandas as pd
import numpy as np
import math

def readcsv(filename, percentage = True):
    data = pd.read_csv(filename, header = 0, index_col = 0, parse_dates = True)
    if percentage:
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
    skewness = (price / vol)
    
    return skewness

def kurtosis(data):
    kdemeaned_return = (data - data.mean())**4
    kprice = kdemeaned_return.mean()
    kvol = data.std(ddof=0)**4
    kurtosis = (kprice / kvol)
    
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

def run_cppi(risky_assets_return, initial_wealth=1000, cash_rate=0.03, floor=0.8, multiplier=3, drawdown=None):
    account_value = initial_wealth
    peak = initial_wealth
    months = risky_assets_return.shape[0]
    safe_return = np.full_like(risky_assets_return, cash_rate/12)

    account_history = pd.DataFrame().reindex_like(risky_assets_return)
    cushion_history = pd.DataFrame().reindex_like(risky_assets_return)
    risky_weight_history = pd.DataFrame().reindex_like(risky_assets_return)
    floor_value = account_value * floor

    for month in range(months):
        #floor_value = account_value * floor
        if drawdown is not None:
            peak = np.maximum(peak, account_value)
            floor_value = peak * (1 - drawdown)
        cushion = (account_value - floor_value) / account_value
        risky_weight = cushion * multiplier
        risky_weight = np.minimum(risky_weight, 1)
        risky_weight = np.maximum(risky_weight, 0)
        risky_values = risky_weight * account_value * risky_assets_return.iloc[month]
        safe_values = (1 - risky_weight) * account_value * safe_return[month]
        account_value = account_value + risky_values + safe_values

        account_history.iloc[month] = account_value
        cushion_history.iloc[month] = cushion
        risky_weight_history.iloc[month] = risky_weight

    risky_wealth_return = initial_wealth*(1+risky_assets_return).cumprod()

    return {
        "AccountHistory": account_history,
        "CushionHistory": cushion_history,
        "RiskyWeightHistory": risky_weight_history,
        "RiskOnlyReturn": risky_wealth_return
    }

def summary_stats(returns, cash_rate=0.03):
    annualized_r = returns.aggregate(annualized_return)
    annualized_v = returns.aggregate(annualized_vol)
    sharpe = returns.aggregate(sharpe_ratio, risk_free_rate = cash_rate)
    dd = returns.aggregate(lambda r: drawdown(r).Drawdown.min())
    skew = returns.aggregate(skewness)
    kurt = returns.aggregate(kurtosis)
    var_hist = returns.aggregate(var_historic)
    cvar_hist = returns.aggregate(cvar_historic)
    var_cf = returns.aggregate(var_cornishfisher)

    return pd.DataFrame({
        "Annualized Returns": annualized_r,
        "Annualized Volatility": annualized_v,
        "Skewness": skew,
        "Kurtosis": kurt,
        "Max Drawdown": dd,
        "VAR Historic 5%": var_hist,
        "VAR Cornis-Fisher 5%": var_cf,
        "CVAR Historic 5%": cvar_hist,
        "Sharpe Ratio": sharpe
    })

def gbm(n_years=10, n_scenarios=1000, mu=0.07, sigma=0.15, initial_price=100.0, return_price=True, steps_per_year=12):
    """
    Geometric Brownian Motion model
    """
    dt = 1 / steps_per_year
    total_steps = int(n_years * steps_per_year)
    rets_plus_1 = np.random.normal(loc=((1+mu)**dt), scale=(sigma*np.sqrt(dt)), size=(total_steps, n_scenarios))
    rets_plus_1[0] = 1
    if return_price:
        return initial_price * pd.DataFrame(rets_plus_1).cumprod()
    else:
        return pd.DataFrame(rets_plus_1-1)

def discount(time_period, interest_rate):
    return 1 / (1+interest_rate)**time_period

def npv(liabilities, interest_rate):
    dates= liabilities.index
    discounts = discount(dates, interest_rate)
    return (discounts*liabilities).sum()

def funding_ratio(asset, liabilities, interest_rate):
    return asset / npv(liabilities, interest_rate)            

def inst_to_ann(r):
    """
    Convert instant rate to annual rate
    """
    return np.expm1(r)

def ann_to_inst(r):
    return np.log1p(r)

def cir(n_years = 10, n_scenarios=1, a=0.05, b=0.03, sigma=0.05, steps_per_year=12, r_0=None):
    """
    Generate random interest rate evolution over time using the CIR model
    b and r_0 are assumed to be the annualized rates, not the short rate
    and the returned values are the annualized rates as well
    """
    if r_0 is None: r_0 = b 
    r_0 = ann_to_inst(r_0)
    dt = 1/steps_per_year
    num_steps = int(n_years*steps_per_year) + 1 # because n_years might be a float
    
    shock = np.random.normal(0, scale=np.sqrt(dt), size=(num_steps, n_scenarios))
    rates = np.empty_like(shock)
    rates[0] = r_0

    ## For Price Generation
    h = math.sqrt(a**2 + 2*sigma**2)
    prices = np.empty_like(shock)
    ####

    def price(ttm, r):
        _A = ((2*h*math.exp((h+a)*ttm/2))/(2*h+(h+a)*(math.exp(h*ttm)-1)))**(2*a*b/sigma**2)
        _B = (2*(math.exp(h*ttm)-1))/(2*h + (h+a)*(math.exp(h*ttm)-1))
        _P = _A*np.exp(-_B*r)
        return _P
    prices[0] = price(n_years, r_0)
    ####
    
    for step in range(1, num_steps):
        r_t = rates[step-1]
        d_r_t = a*(b-r_t)*dt + sigma*np.sqrt(r_t)*shock[step]
        rates[step] = abs(r_t + d_r_t)
        # generate prices at time t as well ...
        prices[step] = price(n_years-step*dt, rates[step])

    rates = pd.DataFrame(data=inst_to_ann(rates), index=range(num_steps))
    ### for prices
    prices = pd.DataFrame(data=prices, index=range(num_steps))
    ###
    return rates, prices    