'''
Created on Dec 14, 2018

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


def generateArgoData() :
    with open("C:\\Users\\NOOK\\Documents\\argo.csv", 'rt') as f:
        reader = csv.reader(f)
        fields = list(reader)
        argo = empty( (0, 9) )
        P0 = 0
        depth = 0
        for i in range(1,len(fields)) :
            P_dPa = float(fields[i][5])
            T_C = float(fields[i][6])
            psal_psu = float(fields[i][7])
            rho = sw.dens(psal_psu, T90conv(T_C), P_dPa)
            g = sw.eos80.g(-depth, 28)
            d = sw.eos80.dpth(P_dPa, 28)
            dh = d - depth
            if (dh > 0) :
                depth = d
                row = array([10000*P_dPa, T_C+273.15, psal_psu, rho, dh, depth, 0, 0, g]).reshape(1,9)
                argo = concatenate((argo, row))
                P0 = 10000*P_dPa
            
            if (depth > 1000) :
                break
    return (argo)

def writeArgoData(argo) :
        with open("C:\\Users\\NOOK\\Documents\\argoRho.csv", 'wt') as o:
            writer = csv.writer(o, delimiter=',', quotechar='"', quoting=csv.QUOTE_NONE, lineterminator='\n')
            for i in range(0, argo.shape[0]) :
                writer.writerow([str(argo[i][0]), str(argo[i][1]), str(argo[i][2]), str(argo[i][3]),
                                 str(argo[i][4]), str(argo[i][5]), str(argo[i][6]), str(argo[i][7]), str(argo[i][8]) ])    
                

if __name__ == '__main__':
    argo = generateArgoData()
    print(argo[0,:])
    print(argo[-1,:])
    for i in range(0, argo.shape[0]) :
        argo[i,6] = sw.dens(argo[i,2], T90conv(argo[0,1]-273.15), argo[i,0]/10000)
        argo[i,7] = sw.dens(argo[i,2], T90conv(argo[-1,1]-273.15), argo[i,0]/10000)
    writeArgoData(argo)
