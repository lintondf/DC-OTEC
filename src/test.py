'''
Created on Dec 16, 2018

@author: NOOK
'''

import os
import logging
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
from simIsentropic import air, logger, gasCoefficients, dissolvedAir

#TODO: for deep water O2 *= 0.1; CO2 *= 1.1 input conditions
surfaceGases = 0 # kg/kg
gamma = 1.4
gammam1 = 0.4

if __name__ == '__main__':
    Vs = []
    for gas in gasCoefficients :
        v = gasCoefficients[gas][2]
        Vs.append(v)
    print(list(gasCoefficients.keys()))
    print(Vs)
    air = Mixture(['H2'], Vfgs=[1], T=272.13, P=101325)
    print(air)
    print(air.Cpg, air.Cvg, air.Cpg/air.Cvg) 
    print(air.phase,air.rho)
#     print(air)
#     air2 = Mixture('air', T=295.55, P=95981)
#     print(air1.H - air2.H)
#     
#     print( sw.eos80.dpth(10100800/1e4, 28) )
#     print( sw.eos80.g(0, 28) )
#     print( sw.eos80.g(-1000, 28) )
#     298.05, 36.386
    print ( dissolvedAir( 101325, 25+273.15, 0, 1000, 1 ) )
#     print ( dissolvedAir(988132, 295, 36.386, 1024.11791654046, 1 ) )
