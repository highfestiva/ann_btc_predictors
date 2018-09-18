import numpy as np
import pandas as pd


def cumcnt(v):
    cumsum = v.cumsum().fillna(method='pad')
    reset = -cumsum[v.isnull()].diff().fillna(cumsum)
    return v.where(v.notnull(), reset).cumsum().fillna(0.0)


def cumcnt_indices(v):
    v[~v] = np.nan
    r = cumcnt(v)
    return r.astype(int)


def calc_rsi(ser, n):
    diff = ser.diff().values
    gains = diff
    losses = -diff
    with np.errstate(invalid='ignore'):
        gains[gains<0] = 0.0
        losses[losses<=0] = 1e-10 # we don't want divide by zero/NaN
    m = n-1
    ni = 1/n
    g = gains[n] = np.nanmean(gains[:n])
    l = losses[n] = np.nanmean(losses[:n])
    gains[:n] = losses[:n] = np.nan
    for i,v in enumerate(gains[n:],n):
        g = gains[i] = (v+m*g)*ni
    for i,v in enumerate(losses[n:],n):
        l = losses[i] = (v+m*l)*ni
    rs = gains / losses
    rsi = pd.Series(100 - (100/(1+rs)))
    return rsi


def calc_stoch(p, n, m):
    l = p.rolling(n).min()
    h = p.rolling(n).max()
    k = (p-l) / (h-l)
    d = 100 * k.rolling(m).mean()
    return d


def calc_historical_volatility(ser, n=10):
    ln = np.log(ser / ser.shift())
    hv = ln.rolling(n).std(ddof=0)
    f = np.sqrt(365) * 100
    return hv * f


def calc_up_down_length(ser):
    v = ser.diff()
    x = cumcnt_indices(v>0)
    y = cumcnt_indices(v<0)
    x[y>0] = -y
    return x

   
def calc_rate_of_change(ser, n):
    change = ser / ser.shift()
    return change.rolling(n).apply(lambda sub: (sub<sub[-1]).sum(), raw=True)


def calc_stoch_rsi(ser, n=14, m=3):
    rsi = calc_rsi(ser, n)
    s_rsi = calc_stoch(rsi, n, m)
    return s_rsi


def calc_connors_rsi(ser, n=100, m=3, k=2):
    rsi_price = calc_rsi(ser, m)
    updown = calc_up_down_length(ser)
    rsi_updown = calc_rsi(updown, k)
    roc = calc_rate_of_change(ser, n)
    return (rsi_price + rsi_updown + roc) / 3
