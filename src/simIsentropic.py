'''
Created on Dec 7, 2018

@author: NOOK


'''

import os
import logging
import logging.handlers
import csv
from scipy.integrate import solve_ivp
from numpy import sqrt, transpose, empty, concatenate, array
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

from runstats import Statistics
from sim import Pdepth

logger = ()

# SERI values
Tsurface = 298.05
Tdeep = 279.65
rhoSurface = 1024.1    
rhoDeep = 1028.8
dTfinal = 2.5

air = Mixture('air', T=300, P=101325)

depth = 200
ioHeight = 2.5 # [m]
Pdepth = 1024*9.806*depth # 735450.0; # [Pa]
dPIO = 9.806*1024*ioHeight
Pstart= Pdepth+0.5*dPIO
volume = 1400; # [m3] 
Thot = Tsurface - dTfinal
Tcold = Tdeep + dTfinal
rhoCold = 8.6935 # [kg/m3]
rhoHot = 8.2516 # [kg/m3]
gamma = 1.4
gammam1 = gamma - 1.0
nozzleArea = 4.758 # 0.0215 # 0.025   # [m2]
fanPressure = 500.0
airTurbineEfficency = 0.51 # https://www.windpowermonthly.com/article/1083653/three-blades-really-better-two # 0.37 # wind turbine efficiency
hydroPumpEfficiency = 0.90
hydroTurbineEfficency = 0.84
compressorEfficiency = 0.80 # includes sub-isothermal losses
maxTime = 3600

nY = 9
nZ = 6
nYZ = nY+nZ



'''
  y[0] - T left      [K]
  y[1] - P left      [Pa]
  y[2] - mass left   [kg]
  y[3] - T right     [K]
  y[4] - P right     [Pa]
  y[5] - mass right  [kg]
  y[6] - energy     [J]
  y[7] - isothermal dH [J]
  y[8] - isothermal dH [J]
  
  
  z[0] - rho left   [kg/m3]
  z[1] - rho right  [kg/m3]
  z[2] - dP        [Pa]
  z[3] - v         [m/s]
  z[4] - mdot      [kg/s]
  z[5] - power     [W]
'''

#TODO: for deep water O2 *= 0.1; CO2 *= 1.1 input conditions
surfaceGases = 0 # kg/kg

# [0] - fit coefficients
# [1] - mole in kg
# [2] - atmosphere pressure fraction
gasCoefficients = {
            #  1           100/T       LN T/100       T/100       S*1            S*T/100      S*(T/100)^2
    'N2':  ([-173.2221,    254.6078,    146.3611,    -22.0913,    -0.054052,     0.027266,    -0.00383],
             0.028014, 0.01*78.080),
    'O2':  ([-173.9894,    255.5907,    146.4813,    -22.2040,    -0.037362,     0.016504,    -0.0020564],
            0.031998,  0.01*20.946),
    'Ar':  ([-174.3732,    251.8139,    145.2337,    -22.2046,    -0.038729,     0.017171,    -0.0021281],
            0.039792,  0.01*0.934),
    'CO2': ([-60.2409,      93.4517,     23.3585,           0,     0.023517,    -0.023656,     0.0047036],
            0.04401,   0.01*0.04, 1.977e-3),
    # thermo misinterprets CO as methanol
    'C1O1':([-47.6148,      69.5068,     18.7397,           0,     0.045657,    -0.040721,     0.0079700], # mL/mL
            0.028,     0, 1.145e-3),
    'H2':  ([-47.8948,      65.0368,     20.1709,           0,     -0.082225,    0.049564,    -0.0078689], # mL/mL
            0.002,     0,  0.08988e-3)
    }

def dissolvedAir( P, T, S, rho, m ): 
    X = array([1.0, 100.0/T, log(T/100.0), T/100.0, S*1.0, S*T/100.0, S*(T/100.0)**2])
    mdissolved = empty((0))
    for gas in gasCoefficients :
        C = gasCoefficients[gas]
        if (C[0][3] == 0) :
            r = C[3]  # CO2 fit is mol/kg/atm
        else :
            r = 1e-3/rho # other are mmol/m3/atm
        Y = array(C[0])
        K = X * Y
        K = exp(K.sum()) 
#         print(gas,K,K*r, 1000*K*r)
        kgkgPa = 1.0/101325.0*K*C[1]*r # kg/kg/Pa
        pP = C[2] * P # partial pressure Pa
        m_gas = pP * kgkgPa # kg/kg
#         print(gas, K, K*1e-3*C[1],K*1e-3*C[1]*r, kgkgPa, pP, m_gas)
        mdissolved = concatenate((mdissolved, array([m_gas])))
    mdissolved -= surfaceGases
    Pr = P / 101325
    T0 = 300
    T1 = T0 * Pr ** (gammam1/gamma)
    R = 287.058
    w = R * T0 * log(Pr)
    W = m * mdissolved.sum() * w 
    return (W, w, T1, m * mdissolved.sum(), mdissolved)

# def dissolvedAir( P, m ): 
#     pctByVol = 0.01*array([78.03,20.99,0.94,0.03,0.01]) # N2, O2, Ar, CO2, H2
#     kgkgPa = array([1.6865E-10,1.3579E-10,1.5709E-10,6.9495E-10,1.4014E-11])
#     partialPs = pctByVol * P 
#     mdissolved = m * (partialPs * kgkgPa)
#     Pr = P / 101325
#     T0 = 300
#     T1 = T0 * Pr ** (gammam1/gamma)
#     R = 287.058
#     w = R * T0 * log(Pr)
#     W = mdissolved.sum() * w 
#     return (W, w, T1, mdissolved.sum())

def printState(t, y, z):
    printAugmented((t, y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], z[0], z[1], z[2], z[3], z[4], z[5]))
    
def printAugmented( tyz ):
    # time, T left, P left, m left, T right, P right, m right, E [J]  rho left, rho right, dP, v, mdot, power
    print("%10.3g, %6.2f, %10.0f, %8.1f, %6.2f, %10.0f, %8.1f, %10.0f, %10.3e, %10.3e, %10.4f, %10.4f, %10.1f, %6.2f, %8.2f, %10.0f" % tyz )
    
def eventStartFanMode(t,y):
#     if (y[1] >= y[4]) :
#         event = (y[1] - y[4]) - fanPressure
#     else :
#         event = (y[1] - y[4]) + fanPressure
#     print("eSFM %10.3f %10.0f %10.0f %10.1f" % (t, y[1], y[4], event))
    event = abs(y[1] - y[4]) - fanPressure
#     print( t, y, fanPressure, event)
    return event

def eventEqualized(t,y):
    return (y[1] - y[4])

def blowdownAugmented(t,y):
    dP = y[1] - y[4]
    if (dP == 0) :
        v = 0.0
        mdot = 0.0
        turbinePower = 0.0
    else :
        if (dP > 0):
            air.set_TP(y[0],y[1])
            v = sqrt( 2*air.Cp*y[0] * (1-(y[4]/y[1])**(gammam1/gamma)) ) 
            mdot = nozzleArea*v*y[2]/volume
            turbinePower = airTurbineEfficency * 0.5 * nozzleArea * y[2]/volume * abs(v)**3.0
        else :
            air.set_TP(y[3],y[4])
            v = -sqrt( 2*air.Cp*y[3] * (1-(y[1]/y[4])**(gammam1/gamma)) ) 
            mdot = nozzleArea*v*y[5]/volume
            turbinePower = airTurbineEfficency * 0.5 * nozzleArea * y[5]/volume * abs(v)**3.0
    return [y[2]/volume, y[5]/volume, dP, v, mdot, turbinePower]

def equalizeAugmented(t,y):
    dP = y[1] - y[4]
    if (dP == 0.0) :
        v = 0
        mdot = 0
        fanPower = 9
    elif (dP > 0.0) :
        dP = + fanPressure
        v = sqrt(abs(2*dP/(y[5]/volume)))
        mdot = nozzleArea*v*y[2]/volume
        fanPower = -0.5/airTurbineEfficency * nozzleArea * y[2]/volume * abs(v)**3.0
    else :
        dP = - fanPressure
        v = -sqrt(abs(2*dP/(y[5]/volume)))
        mdot = nozzleArea*v*y[5]/volume
        fanPower = -0.5/airTurbineEfficency * nozzleArea * y[5]/volume * abs(v)**3.0
    return [y[2]/volume, y[5]/volume, dP, v, mdot, fanPower]
    
def blowdownDeriv(t, y): # adiabatic, isentropic
    heatAddition = 0*1e8
    z = blowdownAugmented(t,y)
    drhodt = z[4]/volume
    dPdrho_left = gamma * (y[1]/z[0]**gamma) * z[0]**(gammam1)
    dPdt_left = -dPdrho_left * drhodt
    dTdrho_left = gammam1 * (y[0]/z[0]**gammam1) / z[0]**(1-gammam1)
    dTdt_left = -dTdrho_left * drhodt
    dPdrho_right = gamma * (y[4]/z[1]**gamma) * z[1]**(gammam1)
    dPdt_right = dPdrho_right * drhodt
    dTdrho_right = gammam1 * (y[3]/z[1]**gammam1) / z[1]**(1-gammam1)
    dTdt_right = dTdrho_right * drhodt
    air.set_TP(y[0], y[1])
    dTdt_left += -sign(dTdt_left) * heatAddition / (air.Cvg * y[2])
    air.set_TP(y[3], y[4])
    dTdt_right += -sign(dTdt_right) * heatAddition / (air.Cvg * y[5])
    return [dTdt_left, dPdt_left, z[4], dTdt_right, dPdt_right, -z[4], z[5], 0, 0]

def isothermalDeriv(t, y): # adiabatic, isentropic
    z = blowdownAugmented(t,y)
#     printState(t,y,z,label="iso")
    drhodt = z[4]/volume
    if (drhodt == 0) :
        dHdt_left = 0
        dHdt_right = 0
        dPdt_left = 0
        dPdt_right = 0
    else :
    # heat exchange required to maintain isothermal
        # dW = P dV; rho = m/V -> drho = -m/V^2 dV; dW = P/m V^2 drho 
        dHdt_left = -(y[1]*volume**2/y[2]) * drhodt 
        dHdt_right = +(y[4]*volume**2/y[5]) * drhodt
        dPdt_left = -y[1]/z[0] * drhodt
        dPdt_right = y[4]/z[1] * drhodt
    dTdt_left = 0
    dTdt_right = 0
    d = [dTdt_left, dPdt_left, -z[4], dTdt_right, dPdt_right, +z[4], z[5], dHdt_left, dHdt_right]
#     print(d)
#     if (t > 1.0) :
#         exit()
    return d



def equalizeDeriv(t, y): # adiabatic, isentropic
    z = equalizeAugmented(t,y)
    drhodt = z[4]/volume
    if (z[0] > 0) :
        dPdrho_left = gamma * (y[1]/z[0]**gamma) * z[0]**(gammam1)
        dTdrho_left = gammam1 * (y[0]/z[0]**gammam1) / z[0]**(1-gammam1)
    else :
        dPdrho_left = y[1]/z[0]
        dTdrho_left = y[0]/z[0]
        
    dPdt_left = -dPdrho_left * drhodt
    dTdt_left = -dTdrho_left * drhodt

    if (z[1] > 0) :
        dPdrho_right = gamma * (y[4]/z[1]**gamma) * z[1]**(gammam1)
        dTdrho_right = gammam1 * (y[3]/z[1]**gammam1) / z[1]**(1-gammam1)
    else :
        dPdrho_right = y[4]/z[1]
        dTdrho_right = y[3]/z[1]
    dPdt_right = dPdrho_right * drhodt
    dTdt_right = dTdrho_right * drhodt
    out = [dTdt_left, dPdt_left, z[4], dTdt_right, dPdt_right, -z[4], z[5], 0, 0]
    return out

'''
    Input:
        t - Temperature of the air [K]
        y[0] - total water mass [kg]
        y[1] - total pump work [J]
        y[2] - total turbine work [J]
        Twater - exchanger input water temperature [K]
        mair - total air mass [kg]
        TPair - air T0 time P0 [K.Pa]
        Pdepth - external water pressure at the inlets [Pa]
        dPIO - Inlet/Outlet pressure differential [Pa]
    Output:
        [0] - dm/dT [kg/K]
        [1] - pump [J/K]
        [2] - turbine [J/K]
        [3] - T * dm/dT
'''
def regenerateDeriv(t, y, Twater, mair, TPair, Pdepth, dPIO, extend=False) : # t - Temperature of air; y - mass of water, T water, mass air
    # dm/dT = 1/(dT/dm)
    # dT/dm = Cp_w * (T_w - T_a) dm/dt / (Cp_a * m_a)
    P = TPair / t # cuurent constant volume pressure
    sal = 36.39
    Cp_water = sw.eos80.cp(sal, T90conv(Twater-273.15), P/10000)
    rho_water = sw.eos80.dens(sal, T90conv(Twater-273.15), P/10000)
    air.set_TP(t, P)
    Cv_air = air.Cvg
#     print(t, P, Cp_water,rho_water,Cv_air)
    dTdm = (Twater - t) * Cp_water / (Cv_air * mair)
    if (dTdm == 0) :
        dmdT = 0
    else :
        dmdT = 1.0/dTdm
    dpumpdT = 0
    dturbinedT = 0
    dPin = P - Pdepth
    dPout = P - (Pdepth + dPIO)
    if (dPin >= 0) : # interior pressure higher; pumped input required
        dpumpdT += -dPin/rho_water * dmdT
    else :           # exterior pressure higher; input thru turbine
        dturbinedT += -dPin/rho_water * dmdT
    if (dPout >= 0) : # interior pressure higher; exhaust thru turbine
        dturbinedT += dPout/rho_water * dmdT
    else :            # exterior pressure higher; pumped output required
        dpumpdT += dPout/rho_water * dmdT
    if (extend) :
        return (dmdT, dpumpdT, dturbinedT, t*dmdT, P, dPin, dPout)
    else :
        return (dmdT, dpumpdT, dturbinedT, t*dmdT)

'''
    dPIO - Inlet/Outlet pressure differential [Pa]
    Output:
        (water-mass [kg], pump energy [J], turbine output [J], water-mass * mean-temperature)
'''
def regenerate( T0air, T1air, mair, P0air, Twater, Pdepth, dPIO, show=False ):
#     print(T0air, T1air, mair, Twater)
    if (show) :
        print(T0air, T1air, mair, P0air, Twater, Pdepth, dPIO)
    def wrapper(t, y):
        return regenerateDeriv(t, y, Twater, mair, T0air*P0air, Pdepth, dPIO)
    sol = solve_ivp(wrapper, [T0air, T1air], [0, 0, 0, 0], method="BDF" )
    Y =  transpose(sol.y)
    T = sol.t
    if (show) :
        for i in range(0, T.shape[0]) :
            z = regenerateDeriv(T[i], Y[i,:], Twater, mair, T0air*P0air, Pdepth, dPIO, extend=True)
            print("%6.3f K %10.2f kg %10.0f J %10.0f J %10.4e %10.4f %10.4f %10.4f %10.1f %10.4f %10.4f" % 
                  (T[i], Y[i,0], Y[i,1], Y[i,2], Y[i,3], z[0], z[1], z[2], z[3], z[4], z[5]))
    return Y[-1,:]

def flowPhaseIsentropic(t,y):
    eventStartFanMode.terminal = True
    eventStartFanMode.direction = 0
    sol1 = solve_ivp(blowdownDeriv, [t+0, t+maxTime], y, events=eventStartFanMode, max_step=1, method="BDF", dense_output=True ) # t_eval=range(t+0,t+100),
    T = sol1.t
    t = T[-1]
    Y =  transpose(sol1.y) 
    y = Y[-1,:]
    Z = empty( (0, 6) )
    for i in range(0, Y.shape[0]) :
        z = array(blowdownAugmented(sol1.t[i], Y[i,:])).reshape(1,6)
        Z = concatenate([Z, z] )
    sol1.y = transpose(concatenate([Y, Z], axis=1))
#     print(sol1)

    eventEqualized.terminal = True
    eventEqualized.direction = 0
    sol2 = solve_ivp(equalizeDeriv, [t+0, t+maxTime], y, events=eventEqualized, max_step=1, method="BDF")
    Y =  transpose(sol2.y) 
    Z = empty( (0, 6) )
    for i in range(0, Y.shape[0]) :
        z = array(equalizeAugmented(sol2.t[i], Y[i,:])).reshape(1,6)
        Z = concatenate([Z, z] )
    sol2.y = transpose(concatenate([Y, Z], axis=1))
    return (sol1, sol2)
  
def flowPhaseIsothermal(t,y, equalize=True):
    eventStartFanMode.terminal = True
    eventStartFanMode.direction = 0
    sol1 = solve_ivp(isothermalDeriv, [t+0, t+maxTime], y, events=eventStartFanMode, max_step=1, method="BDF", dense_output=True ) # t_eval=range(t+0,t+100),
    T = sol1.t
    t = T[-1]
    Y =  transpose(sol1.y) 
    y = Y[-1,:]
    Z = empty( (0, nZ) )
    for i in range(0, Y.shape[0]) :
        z = array(blowdownAugmented(sol1.t[i], Y[i,:])).reshape(1,nZ)
        Z = concatenate([Z, z] )
    sol1.y = transpose(concatenate([Y, Z], axis=1))

    if (equalize) :
        eventEqualized.terminal = True
        eventEqualized.direction = 0
        sol2 = solve_ivp(equalizeDeriv, [t+0, t+maxTime], y, events=eventEqualized, max_step=1, method="BDF")
        Y =  transpose(sol2.y) 
        Z = empty( (0, nZ) )
        for i in range(0, Y.shape[0]) :
            z = array(equalizeAugmented(sol2.t[i], Y[i,:])).reshape(1,nZ)
            Z = concatenate([Z, z] )
        sol2.y = transpose(concatenate([Y, Z], axis=1))
        return (sol1, sol2)
    else :
        return (sol1, sol1)
  
def printSolution(sol):
    T = sol.t
    Y =  transpose(sol.y)
#     for i in range(0, T.shape[0]) :
#         printState(T[i], Y[i,0:7], Y[i,7:13])
    printState(T[0], Y[0,0:nY], Y[0,nY:nYZ])
    printState(T[-1], Y[-1,0:nY], Y[-1,nY:nYZ])
    return Y[-1,0:nY]

def printSolutionDetailed(sol):
    T = sol.t
    Y =  transpose(sol.y)
    for i in range(0, T.shape[0]) :
        printState(T[i], Y[i,0:nY], Y[i,nY:nYZ])
    return Y[-1,0:nY]

def getFinalState(sol):
    Y =  transpose(sol.y)
    return (sol.t[-1], Y[-1,0:nY])

def getInitialState(sol):
    Y =  transpose(sol.y)
    return (sol.t[0], Y[0,0:nY])
    
def hotLeft(y, show=False):
    # constant Volume  T0air, T1air, mair, P0air, Twater, Pdepth, dPIO
    m_w_left = regenerate(y[0], Thot, y[2], y[1], Tsurface, Pdepth, dPIO, show=show)
    y[1] = y[1] * Thot / y[0]
    y[0] = Thot
    m_w_right = regenerate(y[3], Tcold, y[5], y[4], Tdeep, Pdepth, dPIO, show=show)
    y[4] = y[4] * Tcold / y[3]
    y[3] = Tcold;
    y[6] = 0;
    y[7] = 0;
    y[8] = 0;
#     print( m_w_left, m_w_right)
    return (y, m_w_left, m_w_right);

def hotRight(y, show=False):
    # constant Volume 
    m_w_left = regenerate(y[0], Tcold, y[2], y[1], Tdeep, Pdepth, dPIO, show=show)
    y[1] = y[1] * Tcold / y[0]
    y[0] = Tcold;
    m_w_right = regenerate(y[3], Thot, y[5], y[4], Tsurface, Pdepth, dPIO, show=show)
    y[4] = y[4] * Thot / y[3]
    y[3] = Thot;
    y[6] = 0;
    y[7] = 0;
    y[8] = 0;
#     print( m_w_left, m_w_right)
    return (y, m_w_right, m_w_left);

def objective_maximumOutputByArea(nArea):
    global nozzleArea
    nozzleArea = nArea  
    (__,y,__, __, __) = runAlternating(2, False)
    return -y[6]

def objective_maximumOutputByVolume(nVolume):
    global volume
    volume = nVolume  
    O = minimize_scalar(objective_maximumOutputByArea, (1e-6, 0.01, 1) )
    print(O)
    
def objective_1MWByArea(nArea):
    global nozzleArea
    nozzleArea = max(nArea, 1e-6)  
    (t,y,*_) = runAlternating(2, False)
    power = y[6] / t
#     print(t,y,nArea,power)
    return 1e6-power 
    
def initialize(dP = 0, dThot=dTfinal,dTcold=dTfinal):
    global Pdepth, Pstart, Thot, Tcold, rhoHot, rhoCold
    Pdepth = sw.eos80.pres(depth, 28)*1e4 # dPa to Pa
    dPIO = sw.eos80.pres(depth+ioHeight, 28)*1e4 - Pdepth
    Pstart = Pdepth + dP
    Thot = Tsurface-dThot   
    Tcold = Tdeep+dTcold
    air.set_TP(Thot, Pstart)
    rhoHot = air.rho
    air.set_TP(Tcold, Pstart)
    rhoCold = air.rho
    return ("Initialzed: %6.0f m %10.0f Pa %10.0f Pa %6.2f K %6.2f K %6.2f K %8.4f %8.4f" % 
          (depth, Pdepth, Pstart, Tsurface, Tdeep, dTfinal, rhoHot, rhoCold) )
    
def runAlternating(count, show=True, detailed=0, startT=0):
    if (not show) :
        detailed = -1
    if (startT == 0) :
        startT = Thot
    t = 0
    y = [startT,    Pstart,   rhoHot*volume,    startT,    Pstart,    rhoHot*volume, 0, 0, 0] # equalized
    z = blowdownAugmented(t, y)
    if (show) :
        printState(t, y, z)
    (y, mhot, mcold) = hotLeft(y)
    solutions = flowPhaseIsentropic(t,y)
    if (show) :
        print("Initial (hot left)")
        printSolution( solutions[0])
        y1 = printSolution( solutions[1])
    else :
        y1 = getFinalState( solutions[1])[1]
    for i in range(0,count) :
        (y2, mhot, mcold) = hotRight(y1, i==detailed);
        solutions = flowPhaseIsentropic(t,y2)
        if (show) :
            print("%d Hot right" % i)
            if (i != detailed) :
                printSolution( solutions[0])
            else :
                print(mhot, mcold)
                printSolutionDetailed( solutions[0])
            y1 = printSolution( solutions[1])
        else :
            y1 = getFinalState( solutions[1])[1]
        (y2, mhot, mcold) = hotLeft(y1, i==detailed)    
        solutions = flowPhaseIsentropic(t,y2)
        if (show) :
            print("%d Hot left" % i)
            if (i != detailed) :
                printSolution( solutions[0])
            else :
                print(mhot, mcold)
                printSolutionDetailed( solutions[0])
            y1 = printSolution( solutions[1])
        else :
            y1 = getFinalState( solutions[1])[1]
        (tfinal,yfinal) = getFinalState( solutions[1])
        if (show) :
            (t0,y0) = getInitialState( solutions[0])
#             print( Pdepth, y0[1]-Pdepth, y0[4]-Pdepth, y[1]-Pdepth, y[4]-Pdepth)
    return (tfinal, yfinal, solutions, mhot, mcold)

def isothermalWaterUsed( TP, H, Twater, Pintake, rhoWater ):
    sal = 36.39
    Cp_water = sw.eos80.cp(sal, T90conv(Twater-273.15), TP[1]/10000)
    dT = TP[0] - Twater
    m_water = abs(H / (Cp_water * dT))
#     print(H, Cp_water, dT, m_water)
    dPin = TP[1] - Pintake
    pumpH = 0
    turbineH = 0
    if (dPin >= 0) : # interior pressure higher; pumped input required
        pumpH += -dPin/rhoWater * m_water
    else :           # exterior pressure higher; input thru turbine
        turbineH += -dPin/rhoWater * m_water
    return [m_water, pumpH, turbineH, m_water * TP[0]] # (water-mass [kg], pump-energy [J], turbine-energy [J], water-mass * mean-temperature)
    
def vadd( v1, v2 ):
    for i in range(0, len(v1)) :
        v1[i] += v2[i]
    return v1

def logPressures( y0, yfinal):
    overUnder = ('O', 'U')
    flags = overUnder[((y0[1] - Pdepth) < 0)]
    flags += overUnder[((yfinal[1] - Pdepth) < 0)]
    flags += overUnder[((y0[4] - Pdepth) < 0)]
    flags += overUnder[((yfinal[4] - Pdepth) < 0)]
    flags += overUnder[((y0[1] - (Pdepth+dPIO)) < 0)]
    flags += overUnder[((yfinal[1] - (Pdepth+dPIO)) < 0)]
    flags += overUnder[((y0[4] - (Pdepth+dPIO)) < 0)]
    flags += overUnder[((yfinal[4] - (Pdepth+dPIO)) < 0)]
    logger.info("pressures: D: %10.0f (%10.0f); hot: %10.0f, %10.0f; cold: %10.0f, %10.0f %10s" %
                (Pdepth, dPIO, y0[1], yfinal[1], y0[4], yfinal[4], flags) )
    
def runAlternatingIsothermal(count, show=True, detailed=0, startX=0.5):
    if (not show) :
        detailed = -1
    startY = 1-startX
    startT = startX*Tcold+startY*Thot
    startRho = startX*rhoCold + startY*rhoHot
    t = 0
    y = [startT,    Pstart,   startRho*volume,    startT,    Pstart,    startRho*volume, 0, 0, 0] # equalized
    z = blowdownAugmented(t, y)
    if (show) :
        printState(t, y, z)
    (y, mhot, mcold) = hotLeft(y)
#     print(y)
#     yd = isothermalDeriv(t, y)
#     z = blowdownAugmented(t,y)
#     print(z)
#     print(yd)
    solutions = flowPhaseIsothermal(t,y)
    if (show) :
        print("Initial (hot left)")
        printSolution( solutions[0])
        y1 = printSolution( solutions[1])
    else :
        y1 = getFinalState( solutions[1])[1]
    imhot = isothermalWaterUsed( y1[0:2], y1[7], Tsurface, Pdepth, rhoSurface )
    imcold = isothermalWaterUsed( y1[3:5], y1[8], Tdeep, Pdepth, rhoDeep )
    for i in range(0,count) :
        (y2, mhot, mcold) = hotRight(y1, i==detailed);
        
        solutions = flowPhaseIsothermal(t,y2)
        if (show) :
            print("%d Hot right" % i)
            if (i != detailed) :
                printSolution( solutions[0])
            else :
                print(mhot, mcold)
                printSolutionDetailed( solutions[0])
            y1 = printSolution( solutions[1])
        else :
            y1 = getFinalState( solutions[1])[1]
        imhot = isothermalWaterUsed( y1[0:2], y1[7], Tsurface, Pdepth, rhoSurface )
        imcold = isothermalWaterUsed( y1[3:5], y1[8], Tdeep, Pdepth, rhoDeep )
        logger.info( "isoT water: hot %6.2f kg, %6.2f K, P %10.4g J, T %10.4g J; cold %6.2f kg, %6.2f K, P %10.4g J, T %10.4g J" % 
                     (imhot[0], imhot[3]/imhot[0], imhot[1], imhot[2], imcold[0], imcold[3]/imcold[0], imcold[1], imcold[2]) )
        
        (y2, mhot, mcold) = hotLeft(y1, i==detailed)    
        logger.info("regn water: hot %6.2f kg, %6.2f K, P %10.4g J, T %10.4g J; cold %6.2f kg, %6.2f K, P %10.4g J, T %10.4g J;" %
                    (mhot[0], mhot[3]/mhot[0], mhot[1], mhot[2], mcold[0], mcold[3]/mcold[0], mcold[1], mcold[2]))
        solutions = flowPhaseIsothermal(t,y2)
        if (show) :
            print("%d Hot left" % i)
            if (i != detailed) :
                printSolution( solutions[0])
            else :
                printSolutionDetailed( solutions[0])
            y1 = printSolution( solutions[1])
        else :
            y1 = getFinalState( solutions[1])[1]
        (t0,y0) = getInitialState( solutions[0])
        (tfinal,yfinal) = getFinalState( solutions[1])
        logPressures( y0, yfinal)
        thot = vadd(imhot, mhot)
        thot[1] /= hydroPumpEfficiency
        thot[2] *= hydroTurbineEfficency
        tcold = vadd(imcold, mcold)
        tcold[1] /= hydroPumpEfficiency
        tcold[2] *= hydroTurbineEfficency
    return (tfinal, yfinal, solutions, thot, tcold )

    
# slightly more water and slightly less energy; power output slightly higher due to shorter cycle times    
def runAlternatingIsothermalUnequalized(count, show=True, detailed=0, startX=0.5):
    convergence = Statistics()
    
    if (not show) :
        detailed = -1
    startY = 1-startX
    startT = startX*Tcold+startY*Thot
    startRho = startX*rhoCold + startY*rhoHot
    t = 0
    y = [startT,    Pstart,   startRho*volume,    startT,    Pstart,    startRho*volume, 0, 0, 0] # equalized
    z = blowdownAugmented(t, y)
    if (show) :
        printState(t, y, z)
    (y, mhot, mcold) = hotLeft(y)
#     print(y)
#     yd = isothermalDeriv(t, y)
#     z = blowdownAugmented(t,y)
#     print(z)
#     print(yd)
    solutions = flowPhaseIsothermal(t,y, equalize=False)
    if (show) :
        print("Initial (hot left)")
        printSolution( solutions[0])
        y1 = getFinalState( solutions[0])[1]
    else :
        y1 = getFinalState( solutions[0])[1]
    imhot = isothermalWaterUsed( y1[0:2], y1[7], Tsurface, Pdepth, rhoSurface )
    imcold = isothermalWaterUsed( y1[3:5], y1[8], Tdeep, Pdepth, rhoDeep )
#     print(mhot, mcold)
    for i in range(0,count) :
        (y2, mhot, mcold) = hotRight(y1, i==detailed);
        convergence.push(y2[2])
        convergence.push(y2[5])
        
        solutions = flowPhaseIsothermal(t,y2, equalize=False)
        if (show) :
            print("%d Hot right" % i)
            if (i != detailed) :
                printSolution( solutions[0])
            else :
                printSolutionDetailed( solutions[0])
            y1 = getFinalState( solutions[0])[1]
        else :
            y1 = getFinalState( solutions[0])[1]
        imhot = isothermalWaterUsed( y1[0:2], y1[7], Tsurface, Pdepth, rhoSurface )
        imcold = isothermalWaterUsed( y1[3:5], y1[8], Tdeep, Pdepth, rhoDeep )
        logger.info( "isoT water: hot %6.2f kg, %6.2f K, P %10.4g J, T %10.4g J; cold %6.2f kg, %6.2f K, P %10.4g J, T %10.4g J" % 
                     (imhot[0], imhot[3]/imhot[0], imhot[1], imhot[2], imcold[0], imcold[3]/imcold[0], imcold[1], imcold[2]) )
        
        (y2, mhot, mcold) = hotLeft(y1, i==detailed)    
        convergence.push(y2[2])
        convergence.push(y2[5])
        logger.info("regn water: hot %6.2f kg, %6.2f K, P %10.4g J, T %10.4g J; cold %6.2f kg, %6.2f K, P %10.4g J, T %10.4g J;" %
                    (mhot[0], mhot[3]/mhot[0], mhot[1], mhot[2], mcold[0], mcold[3]/mcold[0], mcold[1], mcold[2]))
        solutions = flowPhaseIsothermal(t,y2, equalize=False)
        if (show) :
            print("%d Hot left" % i)
            if (i != detailed) :
                printSolution( solutions[0])
            else :
                print(mhot, mcold)
                printSolutionDetailed( solutions[0])
            y1 = getFinalState( solutions[0])[1]
        else :
            y1 = getFinalState( solutions[0])[1]
        (tfinal,yfinal) = getFinalState( solutions[0])
        if (show) :
            (t0,y0) = getInitialState( solutions[0])
#             print( Pdepth, y0[1]-Pdepth, y0[4]-Pdepth, y[1]-Pdepth, y[4]-Pdepth)
#         logger.info("convergence: %10.2f kg (%10.2f kg)" % (convergence.mean(), convergence.stddev()))
        thot = vadd(imhot, mhot)
        thot[1] /= hydroPumpEfficiency
        thot[2] *= hydroTurbineEfficency
        tcold = vadd(imcold, mcold)
        tcold[1] /= hydroPumpEfficiency
        tcold[2] *= hydroTurbineEfficency
    return (tfinal, yfinal, solutions, thot, tcold )
    
    
# less water much less power
# def runRecuperating(count, show=True):
#     t = 0
#     y = [Thot,    Pdepth,    rhoHot*volume,    Thot,    Pdepth,    rhoHot*volume, 0] # equalized
#     y = hotLeft(y)
#     solutions = flowPhaseIsentropic(t,y)
#     if (show) :
#         print("Initial (hot left)")
#         printSolution( solutions[0])
#         y1 = printSolution( solutions[1])
#     else :
#         y1 = getFinalState( solutions[1])[1]
#     
#     for i in range(0,count) :
#         y1 = recuperate(y1)
#         z = blowdownAugmented(t, y1)
#         printState(-1, y1, z)
#         y2 = hotLeft(y1)    
#         solutions = flowPhaseIsentropic(t,y2)
#         if (show) :
#             print("%d Hot left" % i)
#             printSolution( solutions[0])
#             y1 = printSolution( solutions[1])
#         else :
#             y1 = getFinalState( solutions[1])[1]
#         ty = getFinalState( solutions[1])
#         if (show) :
#             print("%10.3f W" % (ty[1][6] / ty[0]) )
#     return (ty[1][6] / ty[0])

def objective_maximumOutputByOverpressure( dP ):
    global pStart
    initialize( dP+Pdepth)
    b = bracket(objective_maximumOutputByArea, 1e-2, 1e0, grow_limit=1e-3 )
    O = minimize_scalar(objective_maximumOutputByArea, b[0:2], tol=1e-4 )
    nozzleArea = O.x
    (tfinal, yfinal, solutions, mhot, mcold) = runAlternating(4, show=False, detailed=False)
    energy = 1e-6*yfinal[6]
    pump = 1e-6*(mhot[1]+mcold[1])
    turbine = 1e-6*(mhot[2]+mcold[2])
    y = 100.0 * (pump + turbine) / energy
    net = energy + (pump + turbine)
    return -net

def printSummary(tfinal, yfinal, solutions, mhot, mcold, makeUpAir) :
    energy = 1e-6*yfinal[6]
    pump = 1e-6*(mhot[1]+mcold[1])
    turbine = 1e-6*(mhot[2]+mcold[2])
    compressor = -1e-6*makeUpAir / compressorEfficiency
    parasiticLoss = 100.0 * (pump + turbine) / energy
    net = energy + (pump + turbine) + compressor
    netPower = net / tfinal
    totalWater = mhot[0] + mcold[0]
    
    fom = totalWater/netPower**2
    
    summary = ("%10.1f Pa  %5.0f m %6.4f m2 %10.4f s %10.3f MW; %10.3f MJ; %6.0f,%6.2f; %6.0f,%6.2f W/C-kg,K; %6.3f MJ %6.3f MJ %6.3f MJ %5.2f  %10.3f MJ %10.3f kg/MJ %10.2f" %
              (Pstart-Pdepth, depth, nozzleArea, tfinal, netPower, energy, 
               mhot[0], mhot[3]/mhot[0], mcold[0], mcold[3]/mcold[0], pump, turbine, compressor, parasiticLoss, net, totalWater/net, fom) )
    logger.info(summary)
    print(summary)
    

bestAreaByDepth = dict([
     (75, 1),
    ])
    
if __name__ == '__main__':
    logger = logging.getLogger('root')
    FORMAT = "%(asctime)s [%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
    filename = "dc-otec.log"
    should_roll_over = os.path.isfile(filename)
    handler = logging.handlers.RotatingFileHandler(filename, mode='w', backupCount=1)
    if should_roll_over:  # log already exists, roll over!
        handler.doRollover()    
    logging.basicConfig(filename=filename, format=FORMAT)
    logger.setLevel(logging.DEBUG)
    
    Vs = []
    for gas in gasCoefficients :
        v = gasCoefficients[gas][2]
        Vs.append(v)
    print(list(gasCoefficients.keys()))
    print(Vs)
    air = Mixture(list(gasCoefficients.keys()), Vfgs=Vs, T=295.55, P=100198)
    gamma = air.Cpg / air.Cvg
    gammam1 = gamma - 1
    
    nozzleArea = 1
    nConverge = 4
    initialize()
    surfaceGases = dissolvedAir( 101325, 298.05, 36.386, 1024.11791654046, 1 )[4]
    if (True) :
        for d in [95] : # range(80,96,1) : #  
            depth = d
#         depth = 90
            for dp in [-13e3] : # range(-20000, 0, 1000) : #  
#                 (tfinal, yfinal, solutions, mhot, mcold) = runAlternatingIsothermal(nConverge, show=False, detailed=nConverge+1)
#                 W = dissolvedAir(yfinal[1], yfinal[0], 35, 1024, mhot[0] + mcold[0])
#                 printSummary(tfinal, yfinal, solutions, mhot, mcold, W )
                for h in [2.5] : #linspace(2, 5, 20) : #  
                    ioHeight = h
                    logger.info( initialize(dp) )
                    (tfinal, yfinal, solutions, mhot, mcold) = runAlternatingIsothermal(nConverge, show=True, detailed=nConverge+1)
                    (W, w, T1, mass, mdissolved) = dissolvedAir(yfinal[1], yfinal[0], 35, 1024, mhot[0] + mcold[0])
                    logger.info((W, w, T1, mass, mdissolved))

                    print("%6.1f " % (h), end='')
                    printSummary(tfinal, yfinal, solutions, mhot, mcold, W )
    if (False) : 
        for d in [90] : # range(80,111,10) : #  
            depth = d
            for dp in [-15e3] : #linspace(-35000, -5000, 10) :# [12e3] : # linspace(-2*dPIO, 2*dPIO, 20) : #   
                logger.info( initialize(dp) )
                (tfinal, yfinal, solutions, mhot, mcold) = runAlternatingIsothermal(nConverge, show=True, detailed=nConverge+1)
                W = dissolvedAir(yfinal[1], yfinal[0], 35, 1024, mhot[0] + mcold[0])
                printSummary(tfinal, yfinal, solutions, mhot, mcold, W )
    exit()
        
#     if (depth in bestAreaByDepth) :
#         nozzleArea = bestAreaByDepth[depth]
#     else :
#         b = bracket(objective_maximumOutputByArea, 1e-2, 1e0, grow_limit=1e-3 )
#         print(b)
#         O = minimize_scalar(objective_maximumOutputByArea, b[0:2], tol=1e-4 )
#         print(O)
#         nozzleArea = O.x
    for n in [1] : #linspace(0.1, 5.0, 10) : # [0.1, 5] : #  
        nozzleArea = n
        for dp in linspace(-dPIO, 2*dPIO, 20) : #[35780] : # linspace(30e3, 40e3, 20 ) : #  [50000] : # 
            for dth in [2.3] : #linspace(2.3, 2.5, 5) : 
                for dtc in [2.45] : # linspace(2.3, 2.5, 5) : 
                    initialize(dp, dth, dtc)
                    
                    (tfinal, yfinal, solutions, mhot, mcold) = runAlternating(4, show=True, detailed=3)
                    print("%6.2f %6.2f" % (Thot, Tcold), end='')
                    printSummary(tfinal, yfinal, solutions, mhot, mcold)
    
#     for d in range(20, 41) :
#         nozzleArea = 1
#         depth = d*10
#         initialize()
#         b = bracket(objective_maximumOutputByOverpressure, 0, 1000, grow_limit=1e-1 )
#         print(b)
#         O = minimize_scalar(objective_maximumOutputByOverpressure, b[0:2], tol=1e-4 )
#         initialize()
#         Pstart = O.x+Pdepth
#         initialize(Pstart)
# #         b = bracket(objective_maximumOutputByArea, 1e-2, 1, grow_limit=1e-3 )
# #         O = minimize_scalar(objective_maximumOutputByArea, b[0:2], tol=1e-4 )
# #         nozzleArea = O.x
#         
#         (tfinal, yfinal, solutions, mhot, mcold) = runAlternating(4, show=False, detailed=False)
#         energy = 1e-6*yfinal[6]
#         pump = 1e-6*(mhot[1]+mcold[1])
#         turbine = 1e-6*(mhot[2]+mcold[2])
#         y = 100.0 * (pump + turbine) / energy
#         net = energy + (pump + turbine)
#         print("%10.1f Pa  %5.0f m %6.4f m2 %10.4f s %10.3f MW; %10.3f MJ; %10.0f, %10.0f W/C-kg; %10.3f MJ %10.3f MJ %5.2f%%  %10.3f MJ %10.3f MW" %
#                   (Pstart-Pdepth, depth, nozzleArea, tfinal, 1e-6*yfinal[6] / tfinal, energy, 
#                    mhot[0], mcold[0], pump, turbine, y, net, net/tfinal) )
        
        
    if (False) :
        nozzleArea = 1
#         if (depth in bestAreaByDepth) :
#             nozzleArea = bestAreaByDepth[depth]
#         else :
#             b = bracket(objective_maximumOutputByArea, 1e-2, 1e0, grow_limit=1e-3 )
#             print(b)
#             O = minimize_scalar(objective_maximumOutputByArea, b[0:2], tol=1e-4 )
#             print(O)
#             nozzleArea = O.x
    
        for d in [50] : # range(50, 500+1, 50) :
            depth = d;
            initialize()
            for o in range(10000, 50000, 1000) :
                Pstart = o+Pdepth
                initialize(o)
                (tfinal, yfinal, solutions, mhot, mcold) = runAlternating(4, show=False, detailed=False)
                energy = 1e-6*yfinal[6]
                pump = 1e-6*(mhot[1]+mcold[1])
                turbine = 1e-6*(mhot[2]+mcold[2])
                y = 100.0 * (pump + turbine) / energy
                net = energy + (pump + turbine)
                if (net > 0) :
                    totalWater = mhot[0] + mcold[0]
                    print("%10.1f Pa  %5.0f m %6.4f m2 %10.4f s %10.3f MW; %10.3f MJ; %10.0f, %10.0f W/C-kg; %10.3f MJ %10.3f MJ %5.2f%%  %10.3f MJ %10.3f kg/MJ" %
                              (Pstart-Pdepth, depth, nozzleArea, tfinal, 1e-6*yfinal[6] / tfinal, energy, 
                               mhot[0], mcold[0], pump, turbine, y, net, totalWater/net) )
#     (tfinal, yfinal, solutions, mhot, mcold) = runAlternating(4, show=True)
#     print("%5.0f m %6.4f m2 %10.4f s %10.3f MW; %10.3f MJ" % (depth, nozzleArea, tfinal, 1e-6*yfinal[6] / tfinal, 1e-6*yfinal[6]) )

    if (False) :        
        Thot = 293.15 
        Tcold = 283.35
        print(Thot, Tcold)
        b = bracket(objective_maximumOutputByArea, 1e-2, 1e0, grow_limit=1e-3 )
        print(b)
        O = minimize_scalar(objective_maximumOutputByArea, b[0:2], tol=1e-4 )
        print(O)
        nozzleArea = O.x
        (tfinal, yfinal, solutions, mhot, mcold) = runAlternating(4, show=False)
        print("%6.2f K %6.2f K  %5.0f m %6.4f m2 %10.4f s %10.3f MW; %10.3f MJ; %10.0f, %10.0f" %
              (Thot, Tcold, depth, nozzleArea, tfinal, 1e-6*yfinal[6] / tfinal, 1e-6*yfinal[6], mhot, mcold) )

#     for h in range(-15, 1) :
#         for c in range(-15, 1) :
#             Thot = 295.55 + 0.2*h 
#             Tcold = 282.15 - 0.2*c 
# #             print(Thot, Tcold)
#             b = bracket(objective_maximumOutputByArea, 1e-2, 1e0, grow_limit=1e-3 )
# #             print(b)
#             O = minimize_scalar(objective_maximumOutputByArea, b[0:2], tol=1e-4 )
# #             print(O)
#             if (O.x <= 0 | isnan(O.x)) :
#                 nozzleArea = 1
#             else :
#                 nozzleArea = O.x
#             (tfinal, yfinal, solutions, mhot, mcold) = runAlternating(4, show=False)
#             print("%6.2f K %6.2f K  %5.0f m %6.4f m2 %10.4f s %10.3f MW; %10.3f MJ; %10.0f, %10.0f" %
#                   (Thot, Tcold, depth, nozzleArea, tfinal, 1e-6*yfinal[6] / tfinal, 1e-6*yfinal[6], mhot, mcold) )
            
    if (False) :
        for d in [50, 100, 150, 200, 250, 500, 1000] :
            depth = d
            initialize()
            sol = brenth( objective_1MWByArea, 0.1, 10, xtol=1e-3 )
    #         print(sol)
            nArea = sol
            (tfinal, yfinal, solutions) = runAlternating(2, show=False, detailed=False)
            print("%5.0f m %6.4f m2 %10.4f s %10.3f MW; %10.3f MJ" % (depth, nozzleArea, tfinal, 1e-6*yfinal[6] / tfinal, 1e-6*yfinal[6]) )

