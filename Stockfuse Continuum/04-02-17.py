# -*- coding: utf-8 -*-
"""
Created on Sat Feb  4 11:41:56 2017

@author: patel_saurabh_j
"""
#%%
# import lots of stuff
from IPython import get_ipython
get_ipython().magic('reset -sf')


import numpy as np
import pandas as pd
import datetime as dt
import matplotlib as mpl
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

import pandas_datareader.data as web
from pandas_datareader.famafrench import get_available_datasets
import quandl

# NEW PACKAGES! 
import statsmodels.formula.api as smf
import statsmodels.api as sm
from scipy.optimize import minimize

path = 'C:\\Users\\patel_saurabh_j\\OneDrive\\001_Personal_Porfolio_Python_code\\Stockfuse Continuum\\'

store_data = path + "MyPorfolio_Stock_Data_MSR_MIN.h5"

#%%
#
tickers = ('ABBV', 'ACHC', 'BBBY', 'BNFT', 'BURL', 'CVS', 'DLPH', 'DLTR', 'FB', 'JNJ', 'ONB', 'PSX', 'PSA', 'LUV', 'TWTR', 'ULTA', 'UA', 'DIS', 'XRM', '^GSPC')

#%%
start_dt = '2008-1-1'
end_dt = dt.date.today()
#
stk_data = web.DataReader(tickers, 'yahoo', start = start_dt, end = end_dt)
stk_data.rename(items={'Adj Close': 'AdjClose'},inplace=True)
stk_data.rename(minor_axis={'^GSPC': 'S&P500'},inplace=True)
stk_adj = stk_data.AdjClose
stk_adjcl = stk_adj.fillna(method = 'ffill')
stk_adjcl_prc = stk_adjcl.fillna(method = 'bfill')
bchmk_adjcl_prc = stk_adjcl_prc['S&P500']
del stk_adjcl_prc['S&P500']
stk_ret = stk_adjcl_prc.pct_change()
#
store = pd.HDFStore(store_data)
store['stk_data_1']      = stk_data
store.close()

#stk_data = pd.read_hdf(path + "MyPorfolio_Stock_Data_MSR_MIN.h5", 'stk_data')
#stk_adjcl_prc = stk_data.AdjClose
#stk_adjcl_prc = stk_data.AdjClose
#bchmk_adjcl_prc = stk_adjcl_prc['NIFTY50']
#del stk_adjcl_prc['NIFTY50']
#stk_ret = stk_adjcl_prc.pct_change()

#%%
axis_dates          = stk_data.major_axis
alldates            = pd.DataFrame(axis_dates,index=axis_dates)
alleom              = alldates.groupby([alldates.index.year,alldates.index.month]).last()
alleom.index        = alleom.Date
axis_eom            = alleom.index
axis_id             = stk_adjcl_prc.columns
#%%
def Ptf_Sharpe_Ratio(w, ret):
    vcv = ret.cov()
    mu = ret.mean()
    num = w.dot(mu.T)
    den = (w.dot(vcv).dot(w.T))**(0.5)
    sharpe_ratio = num/den
    return sharpe_ratio*-1

def Ptf_Variance(w, ret):
    vcv = ret.cov()
    var = w.dot(vcv).dot(w.T)
    return var

#n = len(stk_adjcl_prc.columns)    
#ret = stk_ret
#w = np.ones((1,n))/n   
#vcv = ret.cov()
#mu = ret.mean()
#num = w.dot(mu.T)
#den = (w.dot(vcv).dot(w.T))**(0.5)
#sharpe_ratio = num/den
#%%

cons = ({'type':'eq', 'fun': lambda x: x.sum()-1}, {'type':'ineq', 'fun': lambda x: x - 0}, {'type':'ineq', 'fun': lambda x: 0.2 - x})

cons2 = ({'type':'eq', 'fun': lambda y: y.sum()-1})

#res1 = minimize(Ptf_Sharpe_Ratio, np.ones((1,n))/n, args=(ret,), 
#                constraints = cons, options = {'disp': True,'ftol':1e-12},method='SLSQP')
#res2 = minimize(Ptf_Variance, np.ones((1,n))/n, args=(ret,), 
#                constraints = cons, options = {'disp': True,'ftol':1e-12},method='SLSQP')

#%%
weights_eom_msr = pd.DataFrame([], index = axis_eom, columns = stk_adjcl_prc.columns)
weights_eom_minv = pd.DataFrame([], index = axis_eom, columns = stk_adjcl_prc.columns)

n = len(stk_adjcl_prc.columns)
#%%
for t in axis_eom[61::]:
    ret = stk_ret.loc[t - pd.DateOffset(months = 60):t,:]
    res1 = minimize(Ptf_Sharpe_Ratio, np.ones((1,n))/n, args=(ret,), 
                    constraints = cons, options = {'disp': True,'ftol':1e-12},method='SLSQP')
    res2 = minimize(Ptf_Variance, np.ones((1,n))/n, args=(ret,), 
                    constraints = cons, options = {'disp': True,'ftol':1e-12},method='SLSQP')
    weights_eom_msr.loc[t] = res1.x
    weights_eom_minv.loc[t] = res2.x
#%%old
#weights_daily_msr = pd.DataFrame(weights_eom_msr, index = axis_dates, columns = axis_id)
#weights_daily_msr = weights_daily_msr.fillna(method = 'ffill')
#weights_daily_minv = pd.DataFrame(weights_eom_minv, index = axis_dates, columns = axis_id)
#weights_daily_minv = weights_daily_minv.fillna(method = 'ffill')
##%%
#ptf_value_msr = pd.DataFrame([], index = axis_dates, columns = ['Ptf_Value_MSR'])
#for t in axis_dates:
#    ptf_value_msr.loc[t] = weights_daily_msr.loc[t].dot(stk_adjcl_prc.loc[t].T)
#
#ptf_value_minv = pd.DataFrame([], index = axis_dates, columns = ['Ptf_Value_MINV'])
#for t in axis_dates:
#    ptf_value_minv.loc[t] = weights_daily_minv.loc[t].dot(stk_adjcl_prc.loc[t].T)
##%%
#ptf_log_ret_msr = np.log(ptf_value_msr.astype('float64') / ptf_value_msr.shift(1).astype('float64'))
#ptf_log_ret_minv = np.log(ptf_value_minv.astype('float64') / ptf_value_minv.shift(1).astype('float64'))
#bchmk_log_ret = np.log(bchmk.AdjClose.astype('float64') / bchmk.AdjClose.shift(1).astype('float64'))
#
#hundred_base_msr = 100 + 100*ptf_log_ret_msr.cumsum()
#hundred_base_minv = 100 + 100*ptf_log_ret_minv.cumsum()
#hundred_base_bchmk = 100 + 100*bchmk_log_ret.cumsum() 
#
#all_val = pd.concat([hundred_base_msr, hundred_base_minv, hundred_base_bchmk], axis=1, join='outer')
#all_val['2013-03-01':].plot()

#%%new
weights_daily_msr = pd.DataFrame(weights_eom_msr, index = axis_dates, columns = axis_id)
weights_daily_msr = weights_daily_msr.fillna(method = 'ffill')
weights_daily_minv = pd.DataFrame(weights_eom_minv, index = axis_dates, columns = axis_id)
weights_daily_minv = weights_daily_minv.fillna(method = 'ffill')
eq_w = 1/n
weights_daily_eq = pd.DataFrame(eq_w, index = axis_dates, columns = axis_id)
weights_eom_eqw = pd.DataFrame(weights_daily_eq, index = axis_eom, columns = axis_id)
#%%
ptf_value_msr = pd.DataFrame([], index = axis_dates, columns = ['Ptf_Value_MSR'])
ptf_value_minv = pd.DataFrame([], index = axis_dates, columns = ['Ptf_Value_MINV'])
ptf_value_eqw = pd.DataFrame([], index = axis_dates, columns = ['Ptf_Value_EQW'])
for t in axis_dates:
    ptf_value_msr.loc[t] = weights_daily_msr.loc[t].dot(stk_adjcl_prc.loc[t].T)
    ptf_value_minv.loc[t] = weights_daily_minv.loc[t].dot(stk_adjcl_prc.loc[t].T)
    ptf_value_eqw.loc[t] = weights_daily_eq.loc[t].dot(stk_adjcl_prc.loc[t].T)
#%%
tt = '2017-02-01'

hundred_base_msr = pd.DataFrame([], index = axis_dates, columns = ['MSR'])
hundred_base_msr = hundred_base_msr[tt:]

hundred_base_minv = pd.DataFrame([], index = axis_dates, columns = ['MINV'])
hundred_base_minv = hundred_base_minv[tt:]

hundred_base_eqw = pd.DataFrame([], index = axis_dates, columns = ['EQW'])
hundred_base_eqw = hundred_base_eqw[tt:]

hundred_base_bchmk = pd.DataFrame([], index = axis_dates, columns = ['BCHMK'])
hundred_base_bchmk = hundred_base_bchmk[tt:]

for t in hundred_base_msr.index:
    hundred_base_msr.loc[t] = np.array(ptf_value_msr.loc[t]*100)/np.array(ptf_value_msr.loc[tt])
    hundred_base_minv.loc[t] = np.array(ptf_value_minv.loc[t]*100)/np.array(ptf_value_minv.loc[tt])
    hundred_base_eqw.loc[t] = np.array(ptf_value_eqw.loc[t]*100)/np.array(ptf_value_eqw.loc[tt])
    hundred_base_bchmk.loc[t] = np.array(bchmk_adjcl_prc.loc[t]*100)/np.array(bchmk_adjcl_prc.loc[tt])
        
all_val = pd.concat([hundred_base_msr, hundred_base_minv, hundred_base_eqw, hundred_base_bchmk], axis=1, join='inner')
#all_val.plot()
#%%

oos = tt
ptf_ret_msr = ptf_value_msr.loc[oos::].pct_change()
ptf_ret_minv = ptf_value_minv.loc[oos::].pct_change()
ptf_ret_eqw = ptf_value_eqw.loc[oos::].pct_change()
ptf_ret_bchmk = hundred_base_bchmk.loc[oos::].pct_change()

sharpe_ratio_msr = (ptf_ret_msr.mean()*252) / (ptf_ret_msr.std()*((252)**0.5))
sharpe_ratio_minv = (ptf_ret_minv.mean()*252) / (ptf_ret_minv.std()*((252)**0.5))
sharpe_ratio_eqw = (ptf_ret_eqw.mean()*252) / (ptf_ret_eqw.std()*((252)**0.5))
sharpe_ratio_bchmk = (ptf_ret_bchmk.mean()*252) / (ptf_ret_bchmk.std()*((252)**0.5))
#%%

all_val.plot()
print ('Minimum Variance Optimization OOS Sharpe Ratio')
print (sharpe_ratio_minv)
print ('Maximum Sharpe Ratio Optimization OOS Sharpe Ratio')
print (sharpe_ratio_msr)
print ('Equal Weighted Optimization OOS Sharpe Ratio')
print (sharpe_ratio_eqw)

#%%Last Day weight and Units to be purchased
notional = 93000

today = end_dt

todays_weight_units_minv = (weights_daily_minv.iloc[-1] * notional) / (stk_adjcl_prc.iloc[-1])
todays_weight_units_msr = (weights_daily_msr.iloc[-1] * notional) / (stk_adjcl_prc.iloc[-1])
todays_weight_units_eqw = (weights_daily_eq.iloc[-1] * notional) / (stk_adjcl_prc.iloc[-1])

print ('Minimum Variance Optimization Weights')
print (todays_weight_units_minv.astype(dtype = float).round(decimals = 0))
#print (todays_weight_units_minv.astype(dtype = float))
#print ('Maximum Sharpe Ratio Optimization Weights')
#print (todays_weight_units_msr.astype(dtype = float).round(decimals = 0))
#print ('Equal Weighted Optimization Weights')
#print (todays_weight_units_eqw.astype(dtype = float).round(decimals = 0))

print ((todays_weight_units_minv.astype(dtype = float).round(decimals = 0) * stk_adjcl_prc.iloc[-1]).sum(0))
print ((todays_weight_units_minv * stk_adjcl_prc.iloc[-1]).sum(0))
#print ((todays_weight_units_msr.astype(dtype = float).round(decimals = 0) * stk_adjcl_prc.iloc[-1]).sum(0))
#print ((todays_weight_units_msr * stk_adjcl_prc.iloc[-1]).sum(0))

#print (weights_daily_msr.iloc[-1])
#print (weights_daily_minv.iloc[-1])
#%%
##==============================================================================
## Basede on units of shares
##==============================================================================
#stk_adjcls_prc_eom = pd.DataFrame(stk_adjcl_prc, index = axis_eom, columns = axis_id)
#
#weight_units_eom_minv = (weights_eom_minv * notional) / (stk_adjcls_prc_eom)
#weight_units_eom_msr = (weights_eom_msr * notional) / (stk_adjcls_prc_eom)
#weight_units_eom_eqw = (weights_eom_eqw * notional) / (stk_adjcls_prc_eom)
#
#weight_units_eom_minv = weight_units_eom_minv.astype(dtype = float).round(decimals = 0)
#weight_units_eom_msr = weight_units_eom_msr.astype(dtype = float).round(decimals = 0)
#weight_units_eom_eqw = weight_units_eom_eqw.astype(dtype = float).round(decimals = 0)
#
#weights_units_daily_minv = pd.DataFrame(weight_units_eom_minv, index = axis_dates, columns = axis_id)
#weights_units_daily_msr = pd.DataFrame(weight_units_eom_msr, index = axis_dates, columns = axis_id)
#weights_units_daily_eqw = pd.DataFrame(weight_units_eom_eqw, index = axis_dates, columns = axis_id)
#
#weights_units_daily_minv = weights_units_daily_minv.fillna(method = 'ffill')
#weights_units_daily_msr = weights_units_daily_msr.fillna(method = 'ffill')
#weights_units_daily_eqw = weights_units_daily_eqw.fillna(method = 'ffill')
#
##%%
#ptf_unit_minv = weights_units_daily_minv * stk_adjcl_prc
#ptf_unit_msr = weights_units_daily_msr * stk_adjcl_prc
#ptf_unit_eqw = weights_units_daily_eqw * stk_adjcl_prc
#
#ptf_unit_minv_value = ptf_unit_minv.sum(1)
#ptf_unit_msr_value = ptf_unit_msr.sum(1)
#ptf_unit_eqw_value = ptf_unit_eqw.sum(1)
#
##%%
#hundred_base_msr = pd.DataFrame([], index = axis_dates, columns = ['MSR'])
#hundred_base_msr = hundred_base_msr['2013-03-01':]
#
#hundred_base_minv = pd.DataFrame([], index = axis_dates, columns = ['MINV'])
#hundred_base_minv = hundred_base_minv['2013-03-01':]
#
#hundred_base_eqw = pd.DataFrame([], index = axis_dates, columns = ['EQW'])
#hundred_base_eqw = hundred_base_eqw['2013-03-01':]
#
#hundred_base_bchmk = pd.DataFrame([], index = axis_dates, columns = ['bchmk'])
#hundred_base_bchmk = hundred_base_bchmk['2013-03-01':]
#
#tt = '2013-03-01'
#
#for t in hundred_base_msr.index:
#    hundred_base_minv.loc[t] = np.array(ptf_unit_minv_value.loc[t]*100)/np.array(ptf_unit_minv_value.loc[tt])
#    hundred_base_msr.loc[t] = np.array(ptf_unit_msr_value.loc[t]*100)/np.array(ptf_unit_msr_value.loc[tt])
#    hundred_base_eqw.loc[t] = np.array(ptf_unit_eqw_value.loc[t]*100)/np.array(ptf_unit_eqw_value.loc[tt])
#    hundred_base_bchmk.loc[t] = np.array(bchmk_adjcl_prc.loc[t]*100)/np.array(bchmk_adjcl_prc.loc[tt])
#        
#all_val = pd.concat([ptf_unit_minv_value.loc[tt::], ptf_unit_msr_value.loc[tt::], ptf_unit_eqw_value.loc[tt::]], axis=1, join='inner')
#all_val.plot()
##%%
#
#oos = '2015-01-01'
#
#ptf_ret_msr = ptf_value_msr.loc[oos::].pct_change()
#ptf_ret_minv = ptf_value_minv.loc[oos::].pct_change()
#ptf_ret_eqw = ptf_value_eqw.loc[oos::].pct_change()
#ptf_ret_bchmk = hundred_base_bchmk.loc[oos::].pct_change()
#
#sharpe_ratio_msr = (ptf_ret_msr.mean()*252) / (ptf_ret_msr.std()*((252)**0.5))
#sharpe_ratio_minv = (ptf_ret_minv.mean()*252) / (ptf_ret_minv.std()*((252)**0.5))
#sharpe_ratio_eqw = (ptf_ret_eqw.mean()*252) / (ptf_ret_eqw.std()*((252)**0.5))
#sharpe_ratio_bchmk = (ptf_ret_bchmk.mean()*252) / (ptf_ret_bchmk.std()*((252)**0.5))