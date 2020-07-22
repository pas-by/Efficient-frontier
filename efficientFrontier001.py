#!/usr/bin/python3
# -*- coding: UTF-8 -*-
#
# efficientFrontier001.py
# compute Efficient Frontier
#
import numpy as np
import pandas as pd
import scipy.optimize as sco

np.set_printoptions(precision=4)
np.set_printoptions(suppress=True)
pd.set_option('precision',4)

# Excel file from fig.8.6
log_returns = pd.read_excel('ch8-6a.xls')

# remove date column
log_returns.pop('date')

print('(Fig 8.28) monthly return')
print(log_returns)

# annual return
year_ret = log_returns.sum()

# however, the example use monthly mean return
mon_mean_ret = log_returns.mean()

print('\nmean monthly return :')
print(mon_mean_ret)
max_ret = np.amax(mon_mean_ret)
print("\nmax. = %6.4f" % (max_ret))
min_ret = np.amin(mon_mean_ret)
print("min. = %6.4f" % (min_ret))

# covariance matrix
cov_mar = log_returns.cov()

print('\n(Fig 8.34) covariance matrix :')
print(cov_mar.copy(deep=True).round(4))
# print(cov_mar)

number_of_assets = log_returns.shape[1]
# print(number_of_assets)

def portfolio_risk(weights):
   weights = np.array(weights)
   pvol = np.sqrt(np.dot(weights.T, np.dot(cov_mar, weights)))
   
   # return stand deviation
   return pvol

def portfolio_return(weights):
   weights = np.array(weights)
   pret = np.dot(weights, mon_mean_ret)
   return pret

bnds = tuple((0, 1) for x in range(number_of_assets))

# target returns to be solved
target_return = [min_ret, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 
                 0.045, 0.05, 0.055, 0.06, 0.065, 0.07, 0.075, 0.08, max_ret]

# compute & print results
print("\nresults (fig 8.40)")
for tret in target_return :
   cons = ({'type': 'eq', 'fun': lambda x:  portfolio_return(x) - tret}, 
           {'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
   res = sco.minimize(portfolio_risk, number_of_assets * [1. / number_of_assets,], 
                      method='SLSQP', bounds=bnds, constraints=cons)
   print("%6.4f" % tret, end=' ')
   print(res['x'], end=' ')
   print("%6.4f" % res['fun'])

