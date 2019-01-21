# -*- coding: utf-8 -*-
import numpy as np                                     # Matlab like syntax for linear algebra and functions
import matplotlib.pyplot as plt                        # Plots and figures like you know them from Matlab
from iminuit import Minuit                             # The actual fitting tool, better than scipy's
from probfit import BinnedLH, Chi2Regression, Extended, UnbinnedLH # Helper tool for fitting
import sys
from scipy import stats
from scipy.special import erfc
from scipy.stats import binom, poisson, norm
from sympy import *
import re
import os
import glob
import bisect
import time
from ExternalFunctions import *

### VAL BOIS ####
verbose = True	 			#Whether to run the entire script or just the new stuff
printFits = True
N_Files = 29
sensorCount = 5
R = 0.6 					#rail distance (cm)
R_ERR = 0.01 				#error on R (cm)
D = 1.3						#diameter of metal ball (cm)
D_ERR = 0.01 				#error on D (cm)
THETA = np.radians(np.mean([14.15,12.25,12.75])) 		#Angle on incline (rad)
THETA_ERR = 0.5*np.pi/180 	#error on angle (rad)
timeErr = 1/50000 			#error on time measurements (sec)
locErr = 0.1 				#error on distance measurements (cm)

defaultPath = os.path.abspath('/Users/Nicolai/Documents/Python/Applied_statistics_2018') # Figures out the absolute path for you in case your working directory moves around.
dataFolder = os.path.abspath('/GeneratedData/')

#Functions
#def linear_fit(x, p0, p1): #obsolete
#    return p0 + p1*x

#def func_exp(x, N0, tau) : #obsolete
#    return N0 * binwidth/tau * np.exp(-x/tau)

def exp_func(t, a, v0, s0): #Fitting for exponential acceleration
	return ( (0.5*a)*(t**2) ) + (v0*t) + s0

def grav_calc(a, theta, R, D): #calculating gravity
	return (a/np.sin(theta))*(1.0+(2.0/5.0)*( (R**2.0) / ( (R**2.0) - ((D/2.0)**2)) ) )

#def gErr_calc(a, aER, b, bER, c, cER, d, dER):
#	np.sqrt(f)

def func_gaussian(x, N, mu, sigma) :
    return N * norm.pdf(x, mu, sigma)


#Loading values
timeStamps = np.loadtxt(defaultPath+'/timeStamps.csv', delimiter=',')
location = np.loadtxt(defaultPath+'/location.csv', delimiter=',')
timeStampDiffs = np.loadtxt(defaultPath+'/timeStampDiffs.csv', delimiter=',')
normTimes = np.loadtxt(defaultPath+'/normTimes.csv', delimiter=',')
#frameCount = np.loadtxt(defaultPath+'/frameCount.csv', delimiter=',', s0kiprows=1)
g_vals = np.zeros(len(normTimes))
g_err = np.zeros(len(normTimes))


x = normTimes
y = location

if verbose:
	for i in range(len(normTimes)): #looping over all the files
		yExp, xExp_edges = y[i], x[i]
	
		x_error = np.full_like(xExp_edges, timeErr) #rasmus siger det baber krans en gang imellem
		y_error = np.full_like(yExp, locErr)


		# Prepare figure
		fig_fit, ax_fit = plt.subplots(figsize=(8, 6))
		ax_fit.set_title("Distance vs time, fittet with minuit")
		ax_fit.set_xlabel("Distance travelled [cm]")
		ax_fit.set_ylabel("Time [sec]")

		# Set values 
		indexes = yExp>-1 # only bins with values!
		xExp = xExp_edges #(xExp_edges[1:] + xExp_edges[:-1])/2 # move from bins edges to bin centers
		syExp = y_error # uncertainties
		sxExp = x_error
		ax_fit.errorbar(xExp[indexes], yExp[indexes], xerr=sxExp[indexes], yerr=syExp[indexes], fmt='k_', ecolor='k', elinewidth=1, capsize=2, capthick=1)

		# Chisquare-fit tau values with our function:
		chi2_object_fit = Chi2Regression(exp_func, xExp[indexes], yExp[indexes], error=syExp[indexes]) #add uncertainties here
		minuit_fit_chi2 = Minuit(chi2_object_fit, pedantic=False, a=1, v0=1, s0=1, print_level=0)
		minuit_fit_chi2.migrad() #perform the fit

		# Plot fit
		a, v0, s0 = minuit_fit_chi2.args
		x_fit = np.linspace(xExp_edges.min()-0.05, xExp_edges.max()*1.1, 1000)
		x_fit = np.linspace(xExp_edges.min()-0.05, xExp_edges.max()*1.1, 1000)
		y_fit_simple = exp_func(x_fit, a, v0, s0)
		ax_fit.plot(x_fit, y_fit_simple, 'b-', label="ChiSquare fit")

		NdofFit = 5-3
		Chi2Fit = minuit_fit_chi2.fval
		ProbFit = stats.chi2.sf(Chi2Fit, NdofFit)
		d = {'Acceleration (a, cm/sec^2)':[minuit_fit_chi2.values['a'], minuit_fit_chi2.errors['a']],
			'Initial velocoty (V0)':    [minuit_fit_chi2.values['v0'], minuit_fit_chi2.errors['v0']],
			'Initial position (S0)':    [minuit_fit_chi2.values['s0'], minuit_fit_chi2.errors['s0']],
			'Chi2':     Chi2Fit,
			'ndf':      NdofFit,
			'Prob':     ProbFit,
			}
		text = nice_string_output(d, extra_spacing=3, decimals=3)
		add_text_to_ax(0.02, 0.95, text, ax_fit, fontsize=14)
		if printFits:			
			fig_fit.savefig(defaultPath + f'/chi2fits/Chi2fit_{i}.png')
		plt.close(fig_fit) #limit the amount of memmory kept occupied
		g = grav_calc(a, THETA, R, D)
		g_vals[i] = g
		#g_error = gErr_calc()
		#g_err[i] = g_error

if verbose:
	np.savetxt(defaultPath+dataFolder+'gVals.csv', g_vals, delimiter=',')
	g_vals = np.loadtxt('gVals.csv', delimiter=',')
	print('gravitational values in kg/m*s^2')
	print(g_vals/100*(-1))



#frameDiff = np.zeros(len(frameCount)-1)












