'''
Created on Dec 16, 2018

@author: NOOK
'''
import csv
from scipy.integrate import solve_ivp
from numpy import sqrt, transpose, empty, concatenate, array, zeros
from scipy.optimize._minimize import minimize, minimize_scalar
from thermo.mixture import Mixture
import seawater as sw
from seawater.library import T90conv

import warnings
from scipy.optimize.minpack import fsolve
from scipy.optimize.zeros import newton, brenth
from scipy.optimize.optimize import bracket
from _csv import reader
from scipy import sign, linspace
from math import isnan, log, exp

# [0] - fit coefficients
# [1] - mole in kg
# [2] - atmosphere pressure fraction
gasCoefficients = {
            #  1           100/T       LN T/100       T/100       S*1            S*T/100      S*(T/100)^2
    'N2':  ([-173.2221,    254.6078,    146.3611,    -22.0913,    -0.054052,     0.027266,    -0.00383],
             0.028014, 0.01*78.03),
    'O2':  ([-173.9894,    255.5907,    146.4813,    -22.2040,    -0.037362,     0.016504,    -0.0020564],
            0.031998,  0.01*20.99),
    'Ar':  ([-174.3732,    251.8139,    145.2337,    -22.2046,    -0.038729,     0.017171,    -0.0021281],
            0.039792,  0.01*0.94),
    'CO2': ([-60.2409,      93.4517,     23.3585,           0,     0.023517,    -0.023656,     0.0047036],
            0.04401,   0.01*0.03),
    'H2':  ([7.10E-04,            0,           0,           0,            0,            0,             0],
            0.002,     0.01*0.01)
    }

def dissolvedAir( P, T, S, rho, m ): 
    X = array([1.0, 100.0/T, log(T/100.0), T/100.0, S*1.0, S*T/100.0, S*(T/100.0)**2])
    mdissolved = empty((0))
    for gas in gasCoefficients :
        C = gasCoefficients[gas]
        if (gas == 'CO2') :
            r = 1.0e-3  # CO2 fit is mol/kg/atm
        else :
            r = 1e-3/rho # other are mmol/m3/atm
        Y = array(C[0])
        K = X * Y
        K = exp(K.sum()) 
        kgkgPa = 1.0/101325.0*K*C[1]*r # kg/kg/Pa
        pP = C[2] * P # partial pressure Pa
        m_gas = pP * kgkgPa # kg/kg
#         print(gas, K, K*1e-3*C[1],K*1e-3*C[1]*r, kgkgPa, pP, m_gas)
        mdissolved = concatenate((mdissolved, array([m_gas])))
    Pr = P / 101325
    T0 = 300
    T1 = T0 * Pr ** (0.4/1.4)
    R = 287.058
    w = R * T0 * log(Pr)
    W = m * mdissolved.sum() * w 
    return (W, w, T1, m * mdissolved.sum(), mdissolved)

if __name__ == '__main__':
#     air1 = Mixture('air', T=295.55, P=100198)
#     air2 = Mixture('air', T=295.55, P=95981)
#     print(air1.H - air2.H)
#     
#     print( sw.eos80.dpth(10100800/1e4, 28) )
#     print( sw.eos80.g(0, 28) )
#     print( sw.eos80.g(-1000, 28) )
#     298.05, 36.386
    print ( dissolvedAir( 101325, 298.05, 36.386, 1024.11791654046, 1 )[4] )
#     print ( dissolvedAir( 101325, 30+273.15, 35, 1024.11791654046, 1 )[4] )
