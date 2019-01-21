#---------------------------------------------------------
#
# Basically there is a figure that shows nothing. On top of this figure
#    are three sets of axes, one for the linear fit, and one for residual
#	 and one for the gaussian. The 'art' is tryint to fit it all on top of 
#	 each other so noe one sees all the spaghetti that it really is.
#
#
# The data needs to be stored in an np.array, and that's all you need to provide.
#---------------------------------------------------------


# -*- coding: utf-8 -*-
import numpy as np                                     # Matlab like syntax for linear algebra and functions
import matplotlib.pyplot as plt                        # Plots and figures like you know them from Matlab
import sys
import os
import glob
from ExternalFunctions import *
from scipy.special import erfc
from scipy.stats import binom, poisson, norm
from iminuit import Minuit                             # The actual fitting tool, better than scipy's
from probfit import BinnedLH, Chi2Regression, Extended, UnbinnedLH # Helper tool for fitting
from scipy import stats


defaultPath = os.path.abspath('/Users/Nicolai/Documents/Python/Applied_statistics_2018/Pendulum') # Figures out the absolute path for you in case your working directory moves around.
dataFolder = os.path.abspath('/GeneratedData/')

framesN = np.loadtxt(defaultPath+dataFolder+'/frameCount_N.csv', delimiter=',', skiprows=1)
timesN = np.loadtxt(defaultPath+dataFolder+'/nicolai.dat', delimiter=',', usecols=1)
#framesL = np.loadtxt(defaultPath+dataFolder+'frameCount_L.csv', delimiter=',')
timesL = np.loadtxt(defaultPath+dataFolder+'/louise.dat', delimiter=',', usecols=1)
#framesR = np.loadtxt(defaultPath+dataFolder+'frameCount_R.csv', delimiter=',')
timesR = np.loadtxt(defaultPath+dataFolder+'/rasmus.dat', delimiter=',', usecols=1)
timesE = np.loadtxt(defaultPath+dataFolder+'/extraswingeren.dat', delimiter=',', usecols=1)

period = 0
figGauss = False

def linFit(x, a, b):
	#moar space
	return a*x + b
def calculate_chi2(function, x_values, y_values, sy_values, *fitparameters):
    # traditional loop-version
    chi2_val = 0
    entries = 0
    for x, y, sy in zip(x_values, y_values, sy_values):
        if y > 0:
            f = function(x, *fitparameters) # calc the model value
            residual  = ( y-f ) / sy  # find the uncertainty-weighted residual
            chi2_val += residual**2  # the chi2-value is the squared residual
            entries += 1 # count the bin as non-empty since sy>0 (and thus y>0)
    
    # numpy version
    mask = (y_values>0)
    yhat = function(x_values, *fitparameters)
    chi2_val = np.sum( (y_values[mask]-yhat[mask])**2/sy_values[mask]**2)
    entries = sum(mask)
            
    return chi2_val, entries
def get_bincenter_and_counts_in_range(hist, xmin=None, xmax=None):
    
    if xmin is None:
        xmin = np.min(hist)
    if xmax is None:
        xmax = np.max(hist)
    
    #counts, bin_edges, _ = hist
    counts, bin_edges = hist
    bin_centers = 0.5*(bin_edges[1:] + bin_edges[:-1])
    mask1 = (xmin < bin_centers) & (bin_centers <= xmax) 
    mask2 = counts > 0
    mask_final = mask1 & mask2
    return bin_centers[mask_final], counts[mask_final], np.sqrt(counts[mask_final])
def gauss_pdf(x, mu, sigma):
    """Normalized Gaussian"""
    return 1 / np.sqrt(2 * np.pi) / sigma * np.exp(-(x - mu) ** 2 / 2. / sigma ** 2)
def gauss_extended(x, N, mu, sigma) :
    """Non-normalized Gaussian"""
    return N * gauss_pdf(x, mu, sigma)

print(type(timesL))

length = len(timesL) #num of manual counts	
background, bkg_ax = plt.subplots(figsize=(14, 8))
bkg_ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False) #Remove stuff from the "background"

#PRIMARY FIT (Linear fit)
#Initialization
L_x = np.linspace(1,length,length) 
L_y = timesL-timesL[0]+5 #normalise to start at 0, and then move up 5 to reduce collision with residual axis
yerror = np.full(length, 0.03) #The error we determined reasonable

#Calculations
L_chi2_object = Chi2Regression(linFit, L_x, L_y, error=yerror)
L_minuit = Minuit(L_chi2_object, pedantic=False, a=10, b=10, print_level=0)
L_minuit.migrad() #perform the fit
Lf_a, Lf_b = L_minuit.args

#Plot
Lf_x = np.linspace(L_x.min()-0.05, L_x.max()+0.05, 1000)
Lf_y = linFit(Lf_x, Lf_a, Lf_b)
L_ax = background.add_axes([0.1, 0.1, 0.8, 0.8]) #Placement of "layer 1/linear fit" on the background
L_ax.plot(Lf_x, Lf_y, 'r-', lw=3)
L_ax.plot(L_x, L_y, 'ko', ms=5)
L_ax.set_ylim(bottom=-15, top=120)
L_ax.set_xlabel('Measurement count')
L_ax.set_ylabel('Time elapsed (s)')
	

#SECONDARY FIT (Time residuals)
#Initialization
residual = np.zeros(length)
for i in range(length-1):
	residual[i] = L_y[i] - linFit(i+1, Lf_a, Lf_b)

#Calculations
R_chi2_object = Chi2Regression(linFit, L_x, residual) #CHANGE res_x TO JUST X THOSE ARE THE SAME
R_minuit = Minuit(R_chi2_object, pedantic=False, a=1, b=1, print_level=0)
R_minuit.migrad()
Rf_a, Rf_b = R_minuit.args 
Rf_x = np.linspace(L_x.min(), L_x.max(), 1000)
Rf_y = linFit(Rf_x, Rf_a, Rf_b)

#Plot
left, bottom, width, height = [0.1, 0.1, 0.8, 0.2] #Placement of "layer 2/residual fit" on the background
R_ax = background.add_axes([left, bottom, width, height])
R_ax.plot(Rf_x, Rf_y, 'k-', lw=1)
R_ax.plot(L_x, residual, 'ko', ms=5)
R_ax.plot(np.linspace(1,length,100), np.full(100,np.std(residual)), 'k--', lw=1)
R_ax.plot(np.linspace(1,length,100), np.full(100,-np.std(residual)), 'k--', lw=1)
R_ax.errorbar(L_x, residual, yerr=0.03, fmt='b_', ecolor='b', elinewidth=2, capsize=2, capthick=1)
R_ax.spines['top'].set_visible(False)
R_ax.spines['right'].set_visible(False)
R_ax.spines['bottom'].set_visible(False)
R_ax.spines['left'].set_visible(False)
R_ax.get_xaxis().set_ticks([])
R_ax.patch.set_alpha(0)
minor_ticks = np.linspace(-0.1,0.1,21)
major_ticks = np.linspace(-0.1,0.1,5)
R_ax.set_yticks(minor_ticks, minor = True)
R_ax.set_yticks(major_ticks)
R_ax.tick_params(axis='y', which='major', direction='in', labelsize=15, length=15, width=3, left=False, right=True, labelright=True, labelleft = False, colors='b')
R_ax.tick_params(axis='y', which='minor', direction='in', labelsize=0, length=10, width=2, left=False, right=True, labelright=True, labelleft = False, colors='b')
plt.text(0.5, 0.70, f"Result of the fit",color='k', style='oblique', fontsize=12, fontdict=None, withdash=False)
plt.text(0.5, 0.65, f"Offset = {R_minuit.values['a']:.4f} +/- {R_minuit.errors['a']:.4f} (s)",color='k', style='italic', fontsize=18, fontdict=None, withdash=False,)
plt.text(0.5, 0.60, f"Period = {L_minuit.values['a']:.4f} +/- {L_minuit.errors['a']:.4f} (s)",color='r', style='italic', fontsize=18, fontdict=None, withdash=False,)
plt.text(0.5, 0.50, f"Uncertainty on time measurements\nobtained from RMS of residuals", color='k', style='oblique', fontsize=12, fontdict=None, withdash=False,)

plt.text(25, 0.10, f"Time residual (s)",color='b', style='italic', fontsize=18)
plt.text(25, 0.08, r"Dashed line show $\pm\ 1\sigma$", color='k', style='oblique',fontsize=12)
	
#TERTIARY FIT (Gaussian on time residuals)
minL = -0.15
maxL = 0.15

#Calculating the fit
hist_res = np.histogram(residual, bins=15, range=(minL, maxL))
G_x, G_y, G_sy = get_bincenter_and_counts_in_range(hist_res, minL, maxL)
G_chi2 = Chi2Regression(gauss_extended, G_x, G_y, G_sy) 
G_minuit = Minuit(G_chi2, pedantic=False, N=length, mu=1, sigma=1, print_level=0) 
G_minuit.migrad();
Gf_x = np.linspace(minL, maxL, 10000)
Gf_y = gauss_extended(Gf_x, *G_minuit.args)

#Plotting the fit
left, bottom, width, height = [0.75, 0.4, 0.2, 0.35] #Placement of "layer 3/gaussian fit" on the background
G_ax = background.add_axes([left, bottom, width, height])
G_ax.plot(Gf_x, Gf_y, 'r-', label='Fit', lw=2)
G_ax.errorbar(G_x, G_y, xerr=0.01, yerr=G_sy, fmt='b_', ecolor='b', elinewidth=1, capsize=1, capthick=0)
G_ax.set_xlim(minL,maxL)
G_ax.set_xticks(np.linspace(minL,maxL,31), minor = True)
G_ax.set_xticks(np.linspace(-0.10,0.10,3))
G_ax.set_yticks(np.linspace(0,13,14), minor = True)
G_ax.set_yticks(np.linspace(0,12,7))
G_ax.tick_params(axis='x', which='both', direction='in', labelsize=10)
G_ax.tick_params(axis='y', which='both', direction='in', labelsize=10)
G_ax.set_xlabel('Time residual(s)')
G_ax.set_ylabel('Frequency')

plt.show()


if print_probs:
	#Extra prints for report
	chi2_time, chi2_times_entries = calculate_chi2(linFit, L_x, L_y, yerror, *L_minuit.args)
	time_NDOF = chi2_times_entries - len(L_minuit.args)
	time_chi2_prob =  stats.chi2.sf(chi2_time, time_NDOF) 
	print(f"chi2 for times linear fit: {chi2_time:.4f}, ndof: {time_NDOF:4d}, and prob: {time_chi2_prob:.4f}")
	
	chi2_resi, chi2_resi_entries = calculate_chi2(linFit, L_x, residual, np.full(len(L_x),0.03), *R_minuit.args)
	#resi_NDOF = chi2_resi_entries - len(minuit_resi.args)
	resi_NDOF = len(residual) - len(R_minuit.args)
	resi_chi2_prob =  stats.chi2.sf(chi2_resi, resi_NDOF) 	
	print(f"chi2 for residual times linear fit: {chi2_resi:.4f}, ndof: {resi_NDOF:4d}, and prob: {resi_chi2_prob:.4f}")
	

if figGauss: #If you want to check your gaussian first
	#The code is the same as above, but the naming is garbarge
	minL = -0.15
	maxL = 0.15

	#big gaussian fit
	fig_raw, ax = plt.subplots(nrows=1, figsize=(14,8))
	hist_res = np.histogram(residual, bins=15, range=(minL, maxL))
	ax.set(xlabel='Time residuals(s)', ylabel='Frequency', title='Distribution of time residuals')
	ax.set_ylim(0,13)
	
	
	L2m_x, L2m_y, L2m_sy = get_bincenter_and_counts_in_range(hist_res, minL, maxL)
	#print(L2m_sy)
	ax.errorbar(L2m_x, L2m_y, xerr=0.01, yerr=np.sqrt(L2m_y), fmt='b_', ecolor='b', elinewidth=2, capsize=2, capthick=1)
	
	L2m_binwidth = L2m_x[1] - L2m_x[0]
	gut_range_2 = []
	for i in range(len(residual)):
		if residual[i]>=-0.15 and residual[i]<=0.15:
			gut_range_2.append(i)
	new_L2m, new_eL2m = np.zeros(len(gut_range_2)), np.zeros(len(gut_range_2))
	
	chi2_L2m = Chi2Regression(gauss_extended, L2m_x, L2m_y, L2m_sy) 
	minuit_L2m = Minuit(chi2_L2m, pedantic=False, N=length, mu=1, sigma=1, print_level=0) 
	minuit_L2m.migrad();
	
	xaxis = np.linspace(minL, maxL, 10000)
	y2maxis = gauss_extended(xaxis, *minuit_L2m.args) 
	ax.plot(xaxis, y2maxis, 'r-', label='Fit')

	L2m_chi2, L2m_entries = calculate_chi2(gauss_extended, L2m_x, L2m_y, L2m_sy, *minuit_L2m.args)
	L2m_NDOF = L2m_entries - len(minuit_L2m.args)
	L2m_chi2_prob =  stats.chi2.sf(L2m_chi2, L2m_NDOF) 

	d = {'Entries': len(new_L2m),
		'Chi2': L2m_chi2, 
		'ndf': L2m_NDOF, 
		'Prob': L2m_chi2_prob, 
		}
	for name in minuit_L2m.parameters:
		d[name] = [minuit_L2m.values[name], minuit_L2m.errors[name]]
	
	# add these results to the plot 
	text = nice_string_output(d, extra_spacing=2, decimals=4)
	add_text_to_ax(0.02, 0.95, text, ax, fontsize=12)
	ax.legend()

	left, bottom, width, height = [0.75, 0.4, 0.2, 0.35]
	ax3 = fig_fit.add_axes([left, bottom, width, height])
	ax3.plot(xaxis, y2maxis, 'r-', label='Fit', lw=2)
	ax3.errorbar(L2m_x, L2m_y, xerr=0.01, yerr=np.sqrt(L2m_y), fmt='b_', ecolor='b', elinewidth=1, capsize=1, capthick=0)
	ax3.set_xlim(-0.15,0.15)
	ax3xminor_ticks = np.linspace(-0.15,0.15,31)
	ax3xmajor_ticks = np.linspace(-0.10,0.10,3)
	ax3ymajor_ticks = np.linspace(0,12,7)
	ax3yminor_ticks = np.linspace(0,13,14)
	ax3.set_xticks(ax3xminor_ticks, minor = True)
	ax3.set_xticks(ax3xmajor_ticks)
	ax3.set_yticks(ax3yminor_ticks, minor = True)
	ax3.set_yticks(ax3ymajor_ticks)
	ax3.tick_params(axis='x', which='both', direction='in', labelsize=10)
	ax3.tick_params(axis='y', which='both', direction='in', labelsize=10)
	ax3.set_xlabel('Time residual(s)')
	ax3.set_ylabel('Frequency')





