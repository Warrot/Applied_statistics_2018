# -*- coding: utf-8 -*-
import numpy as np                                     # Matlab like syntax for linear algebra and functions
import matplotlib.pyplot as plt                        # Plots and figures like you know them from Matlab
from iminuit import Minuit                             # The actual fitting tool, better than scipy's
from probfit import BinnedLH, Chi2Regression, Extended, UnbinnedLH # Helper tool for fitting
import sys
from scipy import stats
from scipy.special import erfc
import re
import glob
import bisect
import time
from ExternalFunctions import *

### VAL BOIS ####
accleration = 0
velocity = 0
N_Files = 29
sensorCount = 5

### FUNC BOIS ####
def immaPrint(counter, max_item):
	print('loading: '+str(counter)+' / '+str(max_item))
	counter += 1
	return counter


#### RUN THROUGH ALL THE FILES AND GENERATE CSV FILES #####

start = time.time()
defaultRolling = []
invRolling = []
timeStamps = np.zeros([N_Files,sensorCount]) #Raw timeStamps of when the ball passes a sensor
normTimes = np.zeros([N_Files,sensorCount]) #Raw time stamp for each sensor normalized to 0 at first sensor
timeStampDiffs = np.zeros([N_Files,sensorCount-1]) #Time difference between each sensor passing
location = np.zeros([N_Files,sensorCount]) #location in cm between each sensor


#Get all the peak values
for i in range(N_Files):
	defaultRolling.append('defaultboi'+str(i)+'.csv')
	def_time, def_volt = np.loadtxt(defaultRolling[i], skiprows=10, unpack=True, delimiter=',')
	mark_times = np.clip(def_volt, 2, 4)
	#mark_times = mark_times[100:-100]
	mark_rolled = np.roll(mark_times,1)
	pure_list = []
	#Culling to the first values of the peaks
	culled_times = []
	for j in range(len(mark_times)-1):
		if mark_times[j]==4:
			if len(culled_times)==0:
				timeStamps[i][0] = j
				arrCnt = 1
				culled_times.append(j)
			elif (j-culled_times[len(culled_times)-1]) > 2500:
				timeStamps[i][arrCnt] = j
				arrCnt += 1
				culled_times.append(j)

#Get the differences
for i in range(N_Files*(sensorCount-1)):
	timeStampDiffs[int(i/(sensorCount-1))][i%(sensorCount-1)] = timeStamps[int(i/(sensorCount-1))][i%(sensorCount-1)+1] - timeStamps[int(i/(sensorCount-1))][i%(sensorCount-1)]
	#print(timeStamps[int(i/(sensorCount-1))][i%(sensorCount-1)+1], "-", timeStamps[int(i/(sensorCount-1))][i%(sensorCount-1)], "=" , timeStamps[int(i/(sensorCount-1))][i%(sensorCount-1)+1] - timeStamps[int(i/(sensorCount-1))][i%(sensorCount-1)], " placed in position: ",[int(i/(sensorCount-1))],",",[i%(sensorCount-1)])

for i in range(len(normTimes)):
	normTimes[i] = (timeStamps[i]-timeStamps[i][0])/50000.0

for i in range(len(location)):
	location[i] = (0, 17.5, 17.5+17.2, 17.5+(2*17.2), 17.5+(3*17.2))
np.savetxt('timeStamps.csv', timeStamps, delimiter=',')   # X is an array
np.savetxt('timeStampDiffs.csv', timeStampDiffs, delimiter=',')   # X is an array
np.savetxt('normTimes.csv', normTimes, delimiter=',')   # X is an array
np.savetxt('location.csv', location, delimiter=',')   # X is an array
print("this took. "+str(time.time()-start)+" seconds")


"""timeStamps = np.loadtxt('timeStamps.csv', delimiter=',')
location = np.loadtxt('location.csv', delimiter=',')
timeStampDiffs = np.loadtxt('timeStampDiffs.csv', delimiter=',')
normTimes = np.loadtxt('normTimes.csv', delimiter=',')


#print("location:\n", location)
#print("Raw time stamps:\n", timeStamps)
#print("Time differences:\n", timeStampDiffs)
#print("Normalized times:\n", normTimes)
#
#plt.plot(normTimes, location)
#plt.show()

# Define an exponential PDF (i.e. normalised):
def fit_function_Exp(x, tau):
	return 1.0 / tau * np.exp(- x / tau)

# Define an exponential fit function, which includes a normalisation:
def fit_function_Exp_Ext(x, tau, N):
	return N * fit_function_Exp(x, tau)

def linear_fit(x, p0, p1):
    return p0 + p1*x

def exp_func(x, a, b, c):
	return 0.5*a*x^2 + b*x + c

"dataExp = 1D array of y-values"
# Define data samples
x = np.array([ [ 10.0,  8.0, 13.0,  9.0, 11.0, 14.0,  6.0,  4.0, 12.0,  7.0,  5.0 ] ,
               [ 10.0,  8.0, 13.0,  9.0, 11.0, 14.0,  6.0,  4.0, 12.0,  7.0,  5.0 ] ,
               [ 10.0,  8.0, 13.0,  9.0, 11.0, 14.0,  6.0,  4.0, 12.0,  7.0,  5.0 ] ,
               [  8.0,  8.0,  8.0,  8.0,  8.0,  8.0,  8.0, 19.0,  8.0,  8.0,  8.0 ] ])

y = np.array([ [ 8.04,  6.95,  7.58,  8.81,  8.33,  9.96,  7.24,  4.26, 10.84,  4.82,  5.68 ]  ,
               [ 9.14,  8.14,  8.74,  8.77,  9.26,  8.10,  6.13,  3.10,  9.13,  7.26,  4.74 ]  ,
               [ 7.46,  6.77, 12.74,  7.11,  7.81,  8.84,  6.08,  5.39,  8.15,  6.42,  5.73 ]  ,
               [ 6.58,  5.76,  7.71,  8.84,  8.47,  7.04,  5.25, 12.50,  5.56,  7.91,  6.89 ] ])

#x = normTimes
#y = location

#fig, ax = plt.subplots(nrows=10, ncols=3, figsize=(21,50))
fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(14,10))
ax = ax.flatten() # go from 2d list to 1d list

for x_i, y_i, ax_i in zip(x, y, ax):
        
    ax_i.scatter(x_i, y_i, marker='.', color='blue', s=100)  # make a scatter plot of the i'th data set as blue dots 
    ax_i.set_title('Graph') 

    chi2_object = Chi2Regression(fit_function_Exp_Ext, x_i, y_i) # chi2-regression object
    minuit_exp = Minuit(chi2_object, errordef=1, pedantic=False, tau=1., N=1, print_level=0) # sets the initial parameters of the fit
    minuit_exp.migrad(); # fit the function to the data

    x_fit = np.linspace(0.9*x_i.min(), 1.1*x_i.max()) # Create the x-axis for the plot of the fitted function
    y_fit = fit_function_Exp_Ext(x_fit, *minuit_exp.args) # the fitted function, where we have unpacked the fitted values
    ax_i.plot(x_fit, y_fit, '-r') # plot the fit with a red ("r") line ("-")
    
    d = {'tau': [minuit_exp.values['tau'], minuit_exp.errors['tau']],
         'N': [minuit_exp.values['N'], minuit_exp.errors['N']],
        }
    
    text = nice_string_output(d, extra_spacing=2, decimals=3)
    add_text_to_ax(0.02, 0.97, text, ax_i, fontsize=14)



a_vals = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])


dataExp = r.exponential(tau, NpointsExp)

yExp, xExp_edges = np.histogram(dataExp, bins=NbinsExp, range=(0, NbinsExp))
xExp = (xExp_edges[1:] + xExp_edges[:-1])/2
syExp = np.sqrt(yExp)

Chi2_object = Chi2Regression(fit_function_Exp_Ext, xExp[yExp>0], yExp[yExp>0], syExp[yExp>0])
minuitExp = Minuit(Chi2_object, pedantic=False, tau = tau, N=NpointsExp,  print_level=0)  
minuitExp.migrad();  # perform the actual fit

Chi2Exp = minuitExp.fval

NvarExp = 2                    # Number of variables (alpha0 and alpha1)
NdofExp = NbinsExp - NvarExp   # Number of degrees of freedom

ProbExp =  stats.chi2.sf(Chi2Exp, NdofExp) # The chi2 probability given N_DOF degrees of freedom


fig.tight_layout() 
fig.show()
fig.savefig("æøåmahniggah.pdf")










"""