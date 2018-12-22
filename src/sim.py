'''
Created on Dec 7, 2018

@author: NOOK
'''



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

# SERI values
Tsurface = 298.05
Tdeep = 279.65
dTfinal = 2.5

depth = 100
Pdepth = 1000*9.806*depth # 735450.0; # [Pa]
volume = 150.0e2/2 # 1250.0; # [m3]
Thot = 300.65
Tcold = 285.65;
rhoCold = 8.6935 # [kg/m3]
rhoHot = 8.2516 # [kg/m3]
gamma = 1.4
gammam1 = gamma - 1.0
n = 0.51 # https://www.windpowermonthly.com/article/1083653/three-blades-really-better-two # 0.37 # wind turbine efficiency
nozzleArea = 1 # 0.0215 # 0.025   # [m2]
fanPressure = 500.0
maxTime = 3600

'''
  y[0] - T left      [K]
  y[1] - P left      [Pa]
  y[2] - mass left   [kg]
  y[3] - T right     [K]
  y[4] - P right     [Pa]
  y[5] - mass right  [kg]
  y[6] - energy        [J]
  y[7] - heat exchange (left)   [J]
  y[8] - heat exchange (right)  [J]
  
  z[0] - rho left   [kg/m3]
  z[1] - rho right  [kg/m3]
  z[2] - dP        [Pa]
  z[3] - v         [m/s]
  z[4] - mdot      [kg/s]
  z[5] - power     [W]
'''

def generateArgoData() :
    with open("C:\\Users\\NOOK\\Documents\\argo.csv", 'rt') as f:
        reader = csv.reader(f)
        fields = list(reader)
        print(len(fields))
        argo = empty( (0, 4) )
        for i in range(1,len(fields)) :
            P_dPa = float(fields[i][5])
            T_C = float(fields[i][6])
            psal_psu = float(fields[i][7])
            row = array([10000*P_dPa, T_C+273.15, psal_psu, sw.dens(psal_psu, T90conv(T_C), P_dPa)]).reshape(1,4)
            argo = concatenate((argo, row))
        print(argo[1])
        print(argo[1] / (argo[1][3]*9.806))
        with open("C:\\Users\\NOOK\\Documents\\argoRho.csv", 'wt') as o:
            writer = csv.writer(o, delimiter=',', quotechar='"', quoting=csv.QUOTE_NONE)
            for i in range(0, argo.shape[0]) :
                writer.writerow([str(argo[i][0]), str(argo[i][1]), str(argo[i][2]), str(argo[i][3]) ])    
def printState(t, y, z, label=''):
    printAugmented((t, y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], z[0], z[1], z[2], z[3], z[4], z[5]), label=label)
    
def printAugmented( tyz, label='' ):
    if (label != '') :
        print(label, end='')
    # time, T left, P left, m left, T right, P right, m right, E [J] Hl[H] Hr[J] rho left, rho right, dP, v, mdot, power
    print(" %10.3f, %6.2f, %10.0f, %8.1f, %6.2f, %10.0f, %8.1f, %10.0f, %10.0f, %10.0f,  %10.4f, %10.4f, %10.1f, %6.2f, %8.2f, %10.0f" % tyz )
    
def eventStartFanMode(t,y):
#     if (y[1] >= y[4]) :
#         event = (y[1] - y[4]) - fanPressure
#     else :
#         event = (y[1] - y[4]) + fanPressure
#     print("eSFM %10.3f %10.0f %10.0f %10.1f" % (t, y[1], y[4], event))
    event = abs(y[1] - y[4]) - fanPressure
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
            v = sqrt(2*abs(dP)/(y[2]/volume))
            mdot = nozzleArea*v*y[2]/volume
            turbinePower = n * 0.5 * nozzleArea * y[2]/volume * abs(v)**3.0
        else :
            v = -sqrt(2*abs(dP)/(y[5]/volume))
            mdot = nozzleArea*v*y[5]/volume
            turbinePower = n * 0.5 * nozzleArea * y[5]/volume * abs(v)**3.0
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
        fanPower = -0.5/n * nozzleArea * y[2]/volume * abs(v)**3.0
    else :
        dP = - fanPressure
        v = -sqrt(abs(2*dP/(y[5]/volume)))
        mdot = nozzleArea*v*y[5]/volume
        fanPower = -0.5/n * nozzleArea * y[5]/volume * abs(v)**3.0
    return [y[2]/volume, y[5]/volume, dP, v, mdot, fanPower]
    
def blowdownDeriv(t, y): # adiabatic, isentropic
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
    return [dTdt_left, dPdt_left, z[4], dTdt_right, dPdt_right, -z[4], z[5], 0, 0]

def isothermalDeriv(t, y): # adiabatic, isentropic
    z = blowdownAugmented(t,y)
#     printState(t,y,z,label="iso")
    drhodt = z[4]/volume
#     dTdrho_left = gammam1 * (y[0]/z[0]**gammam1) / z[0]**(1-gammam1)
#     dTdt_left = -dTdrho_left * drhodt
#     dTdrho_right = gammam1 * (y[3]/z[1]**gammam1) / z[1]**(1-gammam1)
#     dTdt_right = dTdrho_right * drhodt
#     dHdt_left = -y[2] * 0.74476e3 * dTdt_left # kg * J/kg.K * dK/dt
#     dHdt_right = -y[5] * 0.74476e3 * dTdt_right # kg * J/kg.K * dK/dt
    # heat exchange required to maintain isothermal
    dHdt_left = -y[1] / drhodt # kg * J/kg.K * dK/dt
    dHdt_right = +y[4] / drhodt # kg * J/kg.K * dK/dt
    dPdt_left = -y[1]/z[0] * drhodt
    dTdt_left = 0
    dPdt_right = y[4]/z[1] * drhodt
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

def flowPhaseIsentropic(t,y, evalRange=()):
    eventStartFanMode.terminal = True
    eventStartFanMode.direction = 0
    if (len(evalRange) == 0) :
        sol1 = solve_ivp(isothermalDeriv, [t+0, t+maxTime], y, events=eventStartFanMode, max_step=1, method="LSODA" ) # t_eval=range(t+0,t+100),
    else :
        sol1 = solve_ivp(isothermalDeriv, [t+0, t+maxTime], y, events=eventStartFanMode, max_step=1, method="LSODA", dense_output=True ) # t_eval=evalRange) # t_eval=range(t+0,t+100),
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
    sol2 = solve_ivp(equalizeDeriv, [t+0, t+maxTime], y, events=eventEqualized, max_step=1, method="LSODA")
    Y =  transpose(sol2.y) 
    Z = empty( (0, 6) )
    for i in range(0, Y.shape[0]) :
        z = array(equalizeAugmented(sol2.t[i], Y[i,:])).reshape(1,6)
        Z = concatenate([Z, z] )
    sol2.y = transpose(concatenate([Y, Z], axis=1))
    return (sol1, sol2)
  
def printSolution(sol):
    T = sol.t
    Y =  transpose(sol.y)
#     for i in range(0, T.shape[0]) :
#         printState(T[i], Y[i,0:7], Y[i,7:13])
    printState(T[0], Y[0,0:9], Y[0,9:15])
    printState(T[-1], Y[-1,0:9], Y[-1,9:15])
    return Y[-1,0:9]

def printSolutionDetailed(sol):
    T = sol.t
    Y =  transpose(sol.y)
    for i in range(0, T.shape[0]) :
        printState(T[i], Y[i,0:9], Y[i,9:15])
    return Y[-1,0:9]

def getFinalState(sol):
    Y =  transpose(sol.y)
    return (sol.t[-1], Y[-1,0:9])

def getInitialState(sol):
    Y =  transpose(sol.y)
    return (sol.t[0], Y[0,0:9])
    
def hotLeft(y):
    # constant Volume 
    y[1] = y[1] * Thot / y[0]
    y[0] = Thot
    y[4] = y[4] * Tcold / y[3]
    y[3] = Tcold;
    y[6] = 0;
    y[7] = 0;
    y[8] = 0;
    return y;

def hotRight(y):
    # constant Volume 
    y[1] = y[1] * Tcold / y[0]
    y[0] = Tcold;
    y[4] = y[4] * Thot / y[3]
    y[3] = Thot;
    y[6] = 0;
    y[7] = 0;
    y[8] = 0;
    return y;

def recuperate(y):
    y[0] = 0.5*(y[0] + y[3])
    y[3] = y[0]
    y[2] = 0.5*(y[2] + y[5])
    y[5] = y[2]
    return y

def objective_maximumOutputByArea(nArea):
    global nozzleArea
    nozzleArea = nArea  
    (t,y,sols) = runAlternating(2, False)
    return -y[6]

def objective_maximumOutputByVolume(nVolume):
    global volume
    volume = nVolume  
    O = minimize_scalar(objective_maximumOutputByArea, (1e-6, 0.01, 1) )
    print(O)
    
def objective_1MWByArea(nArea):
    global nozzleArea
    nozzleArea = max(nArea, 1e-6)  
    (t,y,sols) = runAlternating(2, False)
    power = y[6] / t
#     print(t,y,nArea,power)
    return 1e6-power 
    
def initialize():
    global Pdepth, Thot, Tcold, rhoHot, rhoCold
    Pdepth = 1000*9.806*depth
    Thot = Tsurface-dTfinal   
    Tcold = Tdeep+dTfinal
    air = Mixture('air', T=Thot, P=Pdepth)
    rhoHot = air.rho
    air = Mixture('air', T=Tcold, P=Pdepth)
    rhoCold = air.rho
    
def runAlternating(count, show=True, startT=0):
    if (startT == 0) :
        startT = Thot
    t = 0
    air = Mixture('air', T=startT, P=Pdepth)
    y = [startT,    Pdepth,    air.rho*volume,    startT,    Pdepth,    air.rho*volume, 0, 0, 0] # equalized
    z = blowdownAugmented(t, y)
    if (show) :
        printState(t, y, z)
    y = hotLeft(y)
    solutions = flowPhaseIsentropic(t,y)
    if (show) :
        print("Initial (hot left)")
        printSolution( solutions[0])
        y1 = printSolution( solutions[1])
    else :
        y1 = getFinalState( solutions[1])[1]
    for i in range(0,count) :
        y2 = hotRight(y1);
        solutions = flowPhaseIsentropic(t,y2)
        if (show) :
            print("%d Hot right" % i)
            printSolution( solutions[0])
            y1 = printSolution( solutions[1])
        else :
            y1 = getFinalState( solutions[1])[1]
        y2 = hotLeft(y1)    
        solutions = flowPhaseIsentropic(t,y2)
        if (show) :
            print("%d Hot left" % i)
            printSolution( solutions[0])
            y1 = printSolution( solutions[1])
        else :
            y1 = getFinalState( solutions[1])[1]
        (tfinal,yfinal) = getFinalState( solutions[1])
        if (show) :
            (t0,y0) = getInitialState( solutions[0])
#             print( Pdepth, y0[1]-Pdepth, y0[4]-Pdepth, y[1]-Pdepth, y[4]-Pdepth)
    return (tfinal, yfinal, solutions)
    
def runAlternatingDetailed(count, show=True, startT=0):
    if (startT == 0) :
        startT = Thot
    t = 0
    air = Mixture('air', T=startT, P=Pdepth)
    y = [startT,    Pdepth,    air.rho*volume,    startT,    Pdepth,    air.rho*volume, 0, 0, 0] # equalized
    z = blowdownAugmented(t, y)
    if (show) :
        printState(t, y, z)
    y = hotLeft(y)
#     printState(t, y, blowdownAugmented(t, y), label=' initial')
#     print(isothermalDeriv(t, y))
    solutions = flowPhaseIsentropic(t,y,range(0,100))
    if (show) :
        print("Initial (hot left)")
        printSolutionDetailed( solutions[0])
        y1 = printSolution( solutions[1])
    else :
        y1 = getFinalState( solutions[1])[1]
#     printState(t, y1, blowdownAugmented(t, y1), label=' final  ')
    for i in range(0,count) :
        y2 = hotRight(y1);
#         printState(t, y2, blowdownAugmented(t, y2), label=' initial')
#         print(isothermalDeriv(t, y2))
        solutions = flowPhaseIsentropic(t,y2,range(0,100))
        if (show) :
            print("%d Hot right" % i)
            printSolution( solutions[0])
            y1 = printSolutionDetailed( solutions[1])
        else :
            y1 = getFinalState( solutions[1])[1]
        y2 = hotLeft(y1)    
        solutions = flowPhaseIsentropic(t,y2)
        if (show) :
            print("%d Hot left" % i)
            printSolutionDetailed( solutions[0])
            y1 = printSolution( solutions[1])
        else :
            y1 = getFinalState( solutions[1])[1]
        (tfinal,yfinal) = getFinalState( solutions[1])
        if (show) :
            (t0,y0) = getInitialState( solutions[0])
#             print( Pdepth, y0[1]-Pdepth, y0[4]-Pdepth, y[1]-Pdepth, y[4]-Pdepth)
#     print("%5.0f m %6.4f m2 %10.4f s %10.0f W; %10.0f J" % (depth, nozzleArea, tfinal, yfinal[6] / tfinal, yfinal[6]) )
    return (tfinal, yfinal, solutions)
    
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

bestAreaByDepth = dict([
#     ( 50,1.5562),
#     ( 75, 1.143),
    ])
    
def test_prints() :
    startT = Thot
    t = 0
    air = Mixture('air', T=startT, P=Pdepth)
    y = [startT,    Pdepth,    air.rho*volume,    startT,    Pdepth,    air.rho*volume, 0, 0, 0] # equalized
    z = blowdownAugmented(t, y)
    printState(t, y, z)
    
def test_derivs() :
    startT = Thot
    t = 0
    air = Mixture('air', T=startT, P=Pdepth)
    y = [startT,    Pdepth,    air.rho*volume,    startT,    Pdepth,    air.rho*volume, 0, 0, 0] # equalized
    z = blowdownAugmented(t, y)
    printState(t, y, z)
    y = hotLeft(y)
    printState(t, y, z)
    print( isothermalDeriv(t, y))    

    
if __name__ == '__main__':
#    test_prints()
    initialize()
    (tfinal, yfinal, solutions) = runAlternating(4, show=True)
    print("%5.0f m %6.4f m2 %10.4f s %10.3f MW; %10.3f MJ; %10.3f, %10.3f MJ" % 
          (depth, nozzleArea, tfinal, 1e-6*yfinal[6] / tfinal, 1e-6*yfinal[6], 1e-6*yfinal[7], 1e-6*yfinal[8]) )
    
#     if (depth in bestAreaByDepth) :
#         nozzleArea = bestAreaByDepth[depth]
#     else :
#         b = bracket(objective_maximumOutputByArea, 1e-2, 1e0, grow_limit=1e-3 )
#         print(b)
#         O = minimize_scalar(objective_maximumOutputByArea, b[0:2], tol=1e-4 )
#         print(O)
#         nozzleArea = O.x
#     for a in [0.25, 0.5, 0.75, 1.0, 1.25, 1.50, 1.75, 2.0, 2.25, 2.5] :
#         nozzleArea = a
#         (t,y,sols) = runAlternating(2, False)
#         power = y[6] / t
#         print( "*** %10.6f %10.3f  %10.0f %10.0f" % (nozzleArea, t, power, y[6] ) )

    for d in [50, 75, 100, 150, 200, 250] :
        depth = d
        initialize()
        sol = brenth( objective_1MWByArea, 0.1, 10, xtol=1e-3 )
#         print(sol)
        nArea = sol
        (tfinal, yfinal, solutions) = runAlternatingDetailed(2, show=False)
        print("%5.0f m %6.4f m2 %10.4f s %10.3f MW; %10.3f MJ; %10.3f, %10.3f MJ" % 
              (depth, nozzleArea, tfinal, 1e-6*yfinal[6] / tfinal, 1e-6*yfinal[6], 1e-6*yfinal[7], 1e-6*yfinal[8]) )

#     nozzleArea *= 2
#     runAlternatingDetailed(2, show=False)
#     nozzleArea *= 2
#     runAlternatingDetailed(2, show=False)
    
    
#     print( objective_maximumOutputByArea(1e-4))
#     print( objective_maximumOutputByArea(1e-2))
#     print( objective_maximumOutputByArea(1e-1))
#     print( objective_maximumOutputByArea(1e0))

#     print(objective_maximumOutputByVolume(1250))
#     print(objective_maximumOutputByVolume(1000))
#     print(objective_maximumOutputByVolume(500))
#     print(objective_maximumOutputByVolume(100))

#     O = minimize_scalar(objective_maximumOutputByArea, (1e-5, 1e-2, 1) )
#     print(O)
#     
# 
#     with warnings.catch_warnings():
#         warnings.simplefilter('error')
#         solutions = flowPhaseIsentropic(t,y)
#     y1 = recuperate(y1)
#     t = solutions[1].t[-1]
#     printState(t, y1, blowdownAugmented(t, y1))
#     y2 = hotLeft(y1)    
#     solutions = flowPhaseIsentropic(t,y2)
#     print("Hot left")
#     printSolution( solutions[0])
#     y1 = printSolution( solutions[1])
     
#     air = Mixture('air', T=y[0], P=y[1])
#     print(0.001*air.Cp/gamma, air.rho )
 
#     for depth in [0, 5, 70, 75, 80, 1000] :   
#         PdPa = sw.pres(depth,28)
#         s = [36]
#         t = T90conv([0, 0, 30, 30, 0, 0, 30, 30])
# p = [0, 10000, 0, 10000, 0, 10000, 0, 10000]
# print(t)
