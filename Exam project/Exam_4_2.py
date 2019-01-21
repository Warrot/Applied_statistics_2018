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
#sys.path.append('../External_Functions')
#from ExternalFunctions import nice_string_output, add_text_to_ax # useful functions to print fit results on figure



#r = np.random()
#r.seed(42)

rolls = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
freq = np.array([185, 1149, 3265, 5475, 6114, 5194, 3067, 1331, 403, 105, 14, 4, 0])
trials = np.sum(freq)


print("It follows a binomial PDF, since it basically is similar to asking the chance of getting a six through 12 trials, and then repeating 20k times, but here the odds are just changed because instead of just one six, then its either a five or a six, which comes out to be 1/3 intead of 1/6. Still binomial.")

def binomial(n, k, p):
	return (scipy.special.factorial(n) / (scipy.special.factorial(k)*scipy.special.factorial(n-k)) * p**k*(1-p)**(n-k))

exp = binomial(12, rolls, 1.0/3)*trials
err = np.sqrt(exp)
#print(exp)


residual = exp-freq
ndof = len(rolls)



#Ks_2sampResult(statistic=0.07692307692307698, pvalue=0.9999999999940523)


chi2 = np.sum( residual**2 / err**2 )

prob = scipy.stats.chi2.sf(chi2, ndof)

print(scipy.stats.ks_2samp(freq, exp))
print(f"chi2: {chi2:.4f} and prob({chi2:.4f}, {ndof:2d}) = {prob:.4f}")

fig, ax = plt.subplots(figsize=(10,8))

#plt.plot(rolls, freq, 'r-', label='actual outcome')
#plt.errorbar(rolls, freq, err)
ax.plot(rolls, freq, 'r-', label='plot of data')
ax.plot(rolls, exp, 'b-', label='expected, based on binomial')
ax.legend()
ax.set_xlabel("amount of 5 or 6 in 12 rolls", fontsize=18)
ax.set_ylabel('Frequency', fontsize=18)

fig.savefig('fig_4_2_2.png', dpi=300)

plt.show()


## 4.2.3

















