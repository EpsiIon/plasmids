#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
from pandas import *
import scipy.optimize

from statsmodels.formula.api import ols

c, y1, y2, y3, my, dy = np.loadtxt("saturation.csv", skiprows=3, unpack=True)

fig = plt.figure()
plt.rc('font', size=14)
ax1 = plt.subplot(221)
n=2.5
n1=6

#********************************CopyNumber Titration*******************************
ax1.set_ylabel('RFU, [a.u.]')
ax1.set_xlabel('Concentration, [nM]')
ax1.set_title('End-point')
#plot all points as holes
ax1.plot(c,y1,'o',mfc='white',ms=n1)
ax1.plot(c,y2,'o',mfc='white',ms=n1)
ax1.plot(c,y3,'o',mfc='white',ms=n1)

ax1.errorbar(c, my, yerr=dy, fmt='-or', ecolor='black',lw=n)

#********************************Michaelis Kinetic*******************************
ax2 = plt.subplot(222)
ax2.set_ylabel('1/RFU, [a.u.]')
ax2.set_xlabel('1/Concentration, [nM$^{-1}$]')
ax2.set_title('Michaelis kinetic')
ax2.yaxis.set_label_position("right")

ax2.set_xlim(-0.25,1.25)
ax2.set_ylim(-0.00005,.0004)

ax2.plot(1/c,1/y1,'o',mfc='white',ms=n1)
ax2.plot(1/c,1/y2,'o',mfc='white',ms=n1)
ax2.plot(1/c,1/y3,'o',mfc='white',ms=n1)

#**********************linear regression 1/cn --- 1/F via xi-squared
def fun(t,k,b):
    return (t*k+b)
  
c0, y10, y20, y30, my0, dy0 = np.loadtxt("michaelis.csv", skiprows=3, unpack=True)

# initial guesses for fitting parameters
k0 = 0.0001
b0 = 0.0001
# fit data using SciPy's Levenberg-Marquart method
nlfit, nlpcov = scipy.optimize.curve_fit(fun,
                c0, my0, p0=[k0,b0], sigma=dy0)
# unpack fitting parameters
k, b = nlfit
print nlpcov
print '*********************'
print nlfit
# unpack uncertainties in fitting parameters from diagonal of covariance matrix
dk, db = \
          [np.sqrt(nlpcov[j,j]) for j in range(nlfit.size)]

# create fitting function from fitted parameters
x_fit = np.linspace(-2.0,1.8, 100)
y_fit = fun(x_fit, k, b)
# Calculate residuals and reduced chi squared
resids = my0 - fun(c0, k, b)
redchisqr = ((resids/dy0)**2).sum()/float(c0.size-2)

#************text out***************************************
ax2.plot(x_fit,y_fit,'r-',mfc='white',lw=n)
#********************simple linear regression***************
data = DataFrame({'x': c0, 'y': my0})
model = ols("y ~ x", data).fit()

print model.summary()
print model.bse[1]


ax2.text(0.02, 0.85, '$R^2$ = {0:0.2f}'
         .format(model.rsquared),transform = ax2.transAxes,fontsize=12)

ax2.axhline(0, color='black',lw=n)
ax2.axvline(0, color='black',lw=n)

#********************PRINT OUT************************************
print 'Parameters estimation:'
print 'Chi^2................. {0:0.2f}'.format(redchisqr)
print 'KM.................... {0:0.2f}'.format(k/b)#error propagation
print 'KM-error.............. {0:0.2f}'.format(np.sqrt(b**-2*dk**2+k**2*b**-4*db**2))
print 'Wmax.................. {0:0.2f}'.format(1/b)
print 'Wmax-error............ {0:0.2f}'.format(db/b)
print '======================'
print 'R^2................... {0:0.2f}'.format(model.rsquared)
print 'KM.................... {0:0.2f}'.format(model.params[1]/model.params[0])
print 'KM-error.............. {0:0.2f}'.format(np.sqrt(model.params[0]**-2*model.bse[1]**2+
					       model.params[1]**2*model.params[0]**-4*model.bse[0]**2))
print 'Wmax.................. {0:0.2f}'.format(1/model.params[0])
print 'Wmax-error............ {0:0.2f}'.format(model.bse[0]/model.params[0])

t, a1, a2, a3, a4, a5 = np.loadtxt("kin.csv", skiprows=1, unpack=True)

num_plots = 5
colormap = plt.cm.jet
plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.9, num_plots)])

ax3 = plt.subplot(212)
ax3.set_ylabel('RFU,[a.u.]')
ax3.set_xlabel('Time, [min]')
ax3.set_title('Kinetic curves')

ax3.plot(t/60, a1, label='1.12 nM', lw=n)
ax3.plot(t/60, a2, label='2.81 nM', lw=n)
ax3.plot(t/60, a3, label='11.64 nM', lw=n)
ax3.plot(t/60, a4, label='31.36 nM', lw=n)
ax3.plot(t/60, a5, label='44.40 nM', lw=n)

plt.legend(loc='upper left', frameon=True, fontsize=12)
plt.grid(True)

plt.show()