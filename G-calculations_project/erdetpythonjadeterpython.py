import numpy as np                                     # Matlab like syntax for linear algebra and functions
import matplotlib.pyplot as plt                        # Plots and figures like you know them from Matlab
import matplotlib as mpl
import seaborn as sns                                  # Make the plots nicer to look at
from iminuit import Minuit                             # The actual fitting tool, better than scipy's
from probfit import BinnedLH, Chi2Regression, Extended # Helper tool for fitting
import sys                                             # Modules to see files and folders in directories
from scipy import stats
sys.path.append('../../External_Functions')
from ExternalFunctions import nice_string_output, add_text_to_ax # useful functions to print fit results on figure

def bin_data(x, N_bins=100, xmin=0.0, xmax=1.0):
    """ Function that bins the input data given bins and returns the binned data.
    
    Parameters
    ----------
    x : array-like, shape=[N_numbers]
        Input points.
    bins : int, default 100
        Number of bins to use in the binning.
    xmin : float, default 0.0
        The minimum value of the range of the binning.
    xmax : float, default 1.0
        The maximum value of the range of the binning.

    Returns
    -------
    hist_x : array-like, shape=[bins]
        The centers of the bins.
    hist_y : array-like, shape=[bins]
        The entries of the bins.
    hist_sy : array-like, shape=[bins]
        The standard deviation of the bins.
    hist_mask : array-like, shape=[bins]
        Boolean mask with True where there are non-empty bins.
    """
    
    hist_y, hist_edges = np.histogram(x, bins=N_bins, range=(xmin, xmax))
    hist_x = 0.5*(hist_edges[1:] + hist_edges[:-1])
    hist_sy = np.sqrt(hist_y)
    hist_mask = hist_y > 0
    
    return hist_x, hist_y, hist_sy, hist_mask

# Random numbers
r = np.random
r.seed(1) # try also another value - how much does this change?

save_plots = False

# Set plotting parameters:
mpl.rcParams['font.size'] = 18      # Set the general plotting font size

# Set parameters:
N_numbers = 100000       # Number of random numbers produced.

# Histogram settings (note the choice of 120 bins):
N_bins = 120
xmin = -0.1
xmax = 1.1

x_trans   = np.zeros(N_numbers)
x_hitmiss = np.zeros(N_numbers)
myint = 0
for i in range(N_numbers) :
    # Transformation method:
    # ----------------------
    # Integration gives the function F(x) = x^2, which inverted gives F^-1(x) = sqrt(x):
    x_trans[i] = np.sqrt(r.uniform())      # ...so we let x_trans equal sqrt() of the uniform number!

    # Hit & Miss method:
    # ------------------
    # Generate two random numbers uniformly distributed in [0,1] x [0,2], until they
    # fulfill the "Hit requirement":
    x_hitmiss[i] = 0.0
    y = 1.0
    while (2.0*x_hitmiss[i] < y):      # ...so keep making new numbers, until this is fulfilled!
        x_hitmiss[i] = 1.0 * r.uniform()
        y = 2.0 * r.uniform()
        myint +=1
print(myint)

hist_trans_x_all, hist_trans_y_all, hist_trans_sy_all, hist_trans_mask = bin_data(x_trans, N_bins, xmin, xmax)

hist_trans_x = hist_trans_x_all[hist_trans_mask]
hist_trans_y = hist_trans_y_all[hist_trans_mask]
hist_trans_sy = hist_trans_sy_all[hist_trans_mask]

hist_hitmiss_x_all, hist_hitmiss_y_all, hist_hitmiss_sy_all, hist_hitmiss_mask = bin_data(x_hitmiss, N_bins, xmin, xmax)

hist_hitmiss_x = hist_hitmiss_x_all[hist_hitmiss_mask]
hist_hitmiss_y = hist_hitmiss_y_all[hist_hitmiss_mask]
hist_hitmiss_sy = hist_hitmiss_sy_all[hist_hitmiss_mask]

fig, ax = plt.subplots(figsize=(14, 8))
ax.errorbar(hist_trans_x, hist_trans_y, hist_trans_sy,       fmt='b.', capsize=2, capthick=2, label="Transformation")
ax.errorbar(hist_hitmiss_x, hist_hitmiss_y, hist_hitmiss_sy, fmt='r.', capsize=2, capthick=2, label="Hit & Miss")
ax.set(ylim=(0, ax.get_ylim()[1]*1.2), xlabel="Random number", ylabel="Frequency");

def fit_func(x, m, c):
    return m*x + c


chi2_object_trans = Chi2Regression(fit_func, hist_trans_x, hist_trans_y, hist_trans_sy)
minuit_trans = Minuit(chi2_object_trans, pedantic=False)

minuit_trans.migrad();
chi2_trans = minuit_trans.fval
ndof_trans = len(hist_trans_x) - len(minuit_trans.args)
prob_trans = stats.chi2.sf(chi2_trans, ndof_trans)


x_fit = np.linspace(xmin, xmax, 1000)

y_fit_trans = fit_func(x_fit, *minuit_trans.args)

ax.plot(x_fit, y_fit_trans, 'b-')
fig


chi2_object_hitmiss = Chi2Regression(fit_func, hist_hitmiss_x, hist_hitmiss_y, hist_hitmiss_sy)
minuit_hitmiss = Minuit(chi2_object_hitmiss, pedantic=False)

minuit_hitmiss.migrad();
chi2_hitmiss = minuit_hitmiss.fval
ndof_hitmiss = len(hist_hitmiss_x) - len(minuit_hitmiss.args)
prob_hitmiss = stats.chi2.sf(chi2_hitmiss, ndof_hitmiss)


y_fit_hitmiss = fit_func(x_fit, *minuit_hitmiss.args)
ax.plot(x_fit, y_fit_hitmiss, 'r-');

d_trans = {'Entries': len(x_trans),
           'Mean': x_trans.mean(),
           'Std': x_trans.std(ddof=1),
           'Chi2': chi2_trans,
           'ndof': ndof_trans,
           'Prob': prob_trans, 
           'm': [minuit_trans.values['m'], minuit_trans.errors['m']],
           'c': [minuit_trans.values['c'], minuit_trans.errors['c']],
           }

d_hitmiss = {'Entries': len(x_hitmiss),
           'Mean': x_hitmiss.mean(),
           'Std': x_hitmiss.std(ddof=1),
           'Chi2': chi2_hitmiss,
           'ndof': ndof_hitmiss,
           'Prob': prob_hitmiss, 
           'm': [minuit_hitmiss.values['m'], minuit_hitmiss.errors['m']],
           'c': [minuit_hitmiss.values['c'], minuit_hitmiss.errors['c']],
           }

# fit results text
text_trans = nice_string_output(d_trans, extra_spacing=2, decimals=3)
text_hitmiss = nice_string_output(d_hitmiss, extra_spacing=2, decimals=3)
text = f"Transformation Method: \n{text_trans}  \n\nHit 'n' Miss method: \n{text_hitmiss}"

# add fit result text on the plot
add_text_to_ax(0.02, 0.97, text, ax, fontsize=14)

# Legend:
ax.legend(loc='lower right')
fig.tight_layout()

if save_plots: 
    fig.savefig("Hist_TransVsHitMiss_solution.pdf", dpi=600)
    
fig


val_diff = hist_trans_y-hist_hitmiss_y+0.0001
error_diff = hist_trans_sy-hist_hitmiss_sy+0.0001
new_fit = val_diff / error_diff

print(val_diff)
print(error_diff)
print (new_fit)


a = np.linspace(0,1,100)

plt.plot(a, new_fit)
plt.show()

print(np.sum(new_fit))
print(np.sum(new_fit)/len(new_fit))

# insert probability calculation here



