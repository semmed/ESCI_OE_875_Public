import numpy as np
from numpy import pi, cos, sin, fft
from scipy import signal
import matplotlib.pyplot as plt

def specter(x,m,dt,ave):
    # This is a Python implementation of Jim Irish's MAtlab specter function
    # Semme J. Dijkstra 3/31/2020
    
    #         function smo = specter(x,m,dt,ave)
    #     SPECTER performs FFT analysis of the series X using the IRISH method
    #       of power spectrum estimation.  The X sequence of N points is divided
    #       into K sections of m points each where m must be a power of two, and
    #       the transforms are log smoothed over sections 'ave' wide in log space.
    #       Successive sections are demeaned, detrended, FFT'd and accumulated.
    #       SPECTRUM returns a tripple array of freq, Pxx, navg.
    #            freq is the frequency of the estimate in the same units as dt,
    #                 i.e. if dt is in sec, then freq is in cycles/sec
    #            Pxx is the power density spectrum of x
    #            navg is the number of blocks times the number of points
    #                 averaged together by the smoothing and is 1/2 the
    #                 independent degrees of freedom.
    #       The units on the power density spectrum Pxx is such that the SUM(Pxx)
    #       times delta f is the variance, or amplitude squared per frequency.
    #            ave  FREQUENCY BAND IN LOG UNITS USED IN BOXCAR AVERAGING OVER
    #            LOG-FREQUENCY INTERVALS - SUGGESTED VALUE TO START = 0.05
    #
    #       J.D. Irish, constructed from matlab - 11 Feb 91
    #                   revised with addition of smoothing - 9 Nov 92
    #
    # *********************************************************************
    #
    

    n = len(x)                          # Number of data points
    k = np.fix((n)/(m), None)           # Number of blocks
    print('     Number of blocks in transform = '+str(k))
    print('     *** transforming ***')
    idx=np.arange(0, m, 1)              # Set up index for pieces
    freq = idx/m/dt                     # Set up array of frequencies
    Pxx = np.zeros((m,1))               # Zero out array to accumulate spectra

    for i in range(int(k)):
        xw = signal.detrend(x[idx])     # Detrend each block
        idx = idx + m
        Xx = abs(fft.fft(xw))**2        # Calculate R^2 + I^2 Xform
        Pxx = (Pxx.T + Xx).T            # Accumulate Spectra


    # Select first half and drop 1st point (mean value)
    # Normalize spectra to spectral density
    #               fft returns values of m^2 A^2 /4
    #               so dividing by 2/(m*m) gets to 1/2 A^2
    #               multiply by m dt (really dividing by df)
    #               gets into units of spectral density
    norm = 2*dt/k/m;                  # Normalization
    # Drop mean and go only to the Nyquist frequency
    PXX = np.matrix(Pxx[1:int(m/2+1)]*norm)
    freq = np.matrix(freq[1:int(m/2+1)]).T
    sp=np.concatenate((freq.copy(),PXX.copy()),axis=1)
 

    
    print('     *** Smoothing ***')
    #  INITIALIZE COUNTS
    [m,n]=PXX.shape
    spec=PXX.copy()

    #  LOG-FREQUENCY BAND SMOOTHING
    if ave <= 0:
        error('Averaging interval less than zero ')
        return


    #  SMOOTHING LOOP
    prev=np.log10(freq[0])
    ssum=spec[0]
    fsum=freq[0]
    nppl=0
    j=1
    navg = np.zeros(freq.shape)
    #  DO SMOOTHING LOOP
    for i in range(1,m):
        c=np.log10(freq[i])
        if (c-prev) > ave:    # time to output
            navg[nppl]=j
            spec[nppl]=ssum/navg[nppl]
            freq[nppl]=fsum/navg[nppl]
            ssum=0.
            fsum=0.
            j=0.
            if not i==m-1:
                nppl=nppl+1
                prev=np.log10(freq[i])
            
        
        if not i==m-1:
            ssum=ssum+spec[i]
            fsum=fsum+freq[i]
            j=j+1;
        
        if i==m-1:
            navg[nppl]=j
            spec[nppl]=ssum/navg[nppl]
            freq[nppl]=fsum/navg[nppl]
#             smo=[freq[1:nppl] spec[1:nppl] k*navg[1:nppl]']

    
    navg = navg[:nppl]
    spec = spec[:nppl]
    freq = freq[:nppl]
    
    smo=np.concatenate((freq[:nppl],spec[:nppl], k*navg[:nppl]),axis=1)


    
    print(['     NUMBER OF SMOOTHED POINTS  = '+str(nppl+1)])

    # plot output
    plt.figure(figsize=(10, 10))
    plt.loglog(sp[:,0],sp[:,1],'b', label=u'Raw')
    plt.loglog(smo[:,0],smo[:,1],'r', label=u'Smooth')
    plt.title('Raw and Smoothed Spectral Densities')
    plt.xlabel('Frequency')
    plt.ylabel('Spectral Density')
    plt.legend(ncol=2, loc='upper center')
    plt.show()

#     text(0.2,0.15,['ave = ',num2str(ave)],'sc')
    
    return smo


