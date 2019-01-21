# First, import the modules you want to use:
import numpy as np                                     # Matlab like syntax for linear algebra and functions
import matplotlib.pyplot as plt                        # Plots and figures like you know them from Matlab
from iminuit import Minuit                             # The actual fitting tool, better than scipy's
from probfit import BinnedLH, Chi2Regression, Extended, UnbinnedLH # Helper tool for fitting
import sys
import scipy
from scipy import stats
from scipy.special import erfc
import re
import math
from ExternalFunctions import nice_string_output, add_text_to_ax # useful functions to print fit results on figure

energy = np.loadtxt('data_GammeSpectrum.txt', unpack=True)
energy = energy*2.14
minx = int(min(energy))
maxx = math.ceil(max(energy))
bbin = maxx - minx


plt.show()

fig, ax = plt.subplots(figsize=(12,10))

ax.hist(energy, bins= bbin)

def get_bincenter_and_counts_in_range(hist, xmin=None, xmax=None):
    
    if xmin is None:
        xmin = np.min(hist)
    if xmax is None:
        xmax = np.max(hist)
    
    counts, bin_edges, _ = hist
    bin_centers = 0.5*(bin_edges[1:] + bin_edges[:-1])
    mask1 = (xmin < bin_centers) & (bin_centers <= xmax) 
    mask2 = counts > 0
    mask_final = mask1 & mask2
    return bin_centers[mask_final], counts[mask_final], np.sqrt(counts[mask_final])


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
def gauss_pdf(x, mu, sigma):
    """Normalized Gaussian"""
    return 1 / np.sqrt(2 * np.pi) / sigma * np.exp(-(x - mu) ** 2 / 2. / sigma ** 2)
def gauss_extended(x, N, mu, sigma) :
    """Non-normalized Gaussian"""
    return N * gauss_pdf(x, mu, sigma)
def QND(f_ax, f_x, f_y, f_min=0, f_max=1, f_title='the title', f_ylabel='Frequency', f_xlabel='keV', west=0.02, north=0.95, kolor='k'):
    t_hist = f_ax.hist(f_y, bins=f_max-f_min, range=(f_min, f_max), histtype='step', alpha=0)#, weights=1/new_eL30cm)
    f_ax.set(xlabel=f_xlabel, ylabel=f_ylabel)
    t_x, t_y, t_sy = get_bincenter_and_counts_in_range(t_hist, f_min, f_max)
    t_binwidth = t_x[1] - t_x[0]
    t_chi2 = Chi2Regression(gauss_extended, t_x, t_y, t_sy) 
    t_minuit = Minuit(t_chi2, pedantic=False, N=f_y.sum()*t_binwidth, mu=f_y.mean(), sigma=f_y.std(ddof=1), print_level=0) 
    t_minuit.migrad();
    t_fit_N, t_fit_mu, t_fit_sigma = t_minuit.args 
    t_xaxis = np.linspace(min(f_y)-20, max(f_y)+20, 10000)
    t_yaxis = gauss_extended(t_xaxis, *t_minuit.args)
    t_chi2, t_entries = calculate_chi2(gauss_extended, t_x, t_y, t_sy, *t_minuit.args)
    t_NDOF = t_entries - len(t_minuit.args)
    t_chi2_prob =  stats.chi2.sf(t_chi2, t_NDOF) 
    f_ax.plot(t_xaxis, t_yaxis, '-',color=kolor,  label=f_title)
    f_ax.set_xlim([min(f_y)-0.5,max(f_y)+0.5])
    d = {'Entries': len(f_y),'Mean': f_y.mean(),'Std': f_y.std(ddof=1),'Chi2': t_chi2,'ndf': t_NDOF, 'Prob': t_chi2_prob,}
    for name in t_minuit.parameters:
        d[name] = [t_minuit.values[name], t_minuit.errors[name]]
    text = nice_string_output(d, extra_spacing=2, decimals=4)
    add_text_to_ax(west, north, text, f_ax, fontsize=12, color=kolor)
    f_ax.legend()

#low_miv = 110
#low_mav = 116
low_miv = 236 #d
low_mav = 250 #d

low_y = energy[energy < low_mav]
low_y = low_y[low_y > low_miv]
low_x = np.linspace(low_miv, low_mav, low_mav-low_miv)


#mid_miv = 135
#mid_mav = 141
mid_miv = 289 #u
mid_mav = 301 #u
mid_y = energy[energy < mid_mav]
mid_y = mid_y[mid_y > mid_miv]
mid_x = np.linspace(mid_miv, mid_mav, mid_mav-mid_miv)


#max_miv = 162
#max_mav = 166
max_miv = 347 #u
max_mav = 354 #d
max_y = energy[energy < max_mav]
max_y = max_y[max_y > max_miv]
max_x = np.linspace(max_miv, max_mav, max_mav-max_miv)




#ist_ovj = plt.hist(energy, bins=bbin)
#bin_cen, y, sy = get_bincenter_and_counts_in_range(hist_ovj)

QND(ax, low_x, low_y, f_min=low_miv, f_max=low_mav, f_title='gaussian fit (low Pb)', kolor='xkcd:hot pink')
QND(ax, mid_x, mid_y, f_min=mid_miv, f_max=mid_mav, f_title='gaussian fit (med Pb)', west=0.38, kolor='g')
QND(ax, max_x, max_y, f_min=max_miv, f_max=max_mav, f_title='gaussian fit (high Pb)', north=0.65, kolor='xkcd:diarrhea')

l_mu = 243.1108
m_mu = 294.7936
h_mu = 350.6385

r = (h_mu - m_mu) / (m_mu - l_mu)
# r = 1.080531627543399
#The relative distance is sort of close, but its still not quite perfect
print(r)

bl_miv = 597
bl_mav = 608
bl_y = energy[energy < bl_mav]
bl_y = bl_y[bl_y > bl_miv]
bl_x = np.linspace(bl_miv, bl_mav, bl_mav-bl_miv)

bh_miv = 1097
bh_mav = 1105
bh_y = energy[energy < bh_mav]
bh_y = bh_y[bh_y > bh_miv]
bh_x = np.linspace(bh_miv, bh_mav, bh_mav-bh_miv)


QND(ax, bl_x, bl_y, f_min=bl_miv, f_max=bl_mav, f_title='gaussian fit (low Bi)', west=0.38, north=0.65, kolor='xkcd:piss yellow')
QND(ax, bh_x, bh_y, f_min=bh_miv, f_max=bh_mav, f_title='gaussian fit (high Bi)', west=0.74, north=0.65, kolor='xkcd:perrywinkle')


t_miv = 731
t_mav = 737
t_y = energy[energy < t_mav]
t_y = t_y[t_y > t_miv]
t_x = np.linspace(t_miv, t_mav, t_mav-t_miv)

QND(ax, t_x, t_y, f_min=t_miv, f_max=t_mav, f_title='gaussian fit (theoretical)', west=0.74, kolor='c')

a_miv = 906
a_mav = 915
a_y = energy[energy < a_mav]
a_y = a_y[a_y > a_miv]
a_x = np.linspace(a_miv, a_mav, a_mav-a_miv)

QND(ax, a_x, a_y, f_min=a_miv, f_max=a_mav, f_title='gaussian fit (unknown data)', west=0.56, north=0.35, kolor='xkcd:rust')

ax.legend(loc=3)
ax.set_xlim(left=0, right=1200)
fig.savefig('5_1_6.pdf')
plt.show()







