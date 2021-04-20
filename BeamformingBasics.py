#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Collection of simplified tools for delay-and-sum beamforming prediction and
analysis with Uniform Linear Arrays (ULAs).


Author:
    Fabio Casagrande Hirono - fchirono@gmail.com
    April 2021
"""


# package for numeric computing
import numpy as np

# instantiate a random number generator
rng = np.random.default_rng()


class SensorArray:
    """
    Class to store microphone/hydrophone array geometry.
    
    L: array length [m]
    M: number of sensors (always odd)
    d: inter sensor spacing [m]
    XY: (2,M) array with sensor coordinates in 2D (x,y) space
    m: sensor indices, from -(M-1)/2 to +(M-1)/2
    """
    
    def __init__(self, L, M):
        self.M = M
        self.L = L
        
        self.XY, self.d, self.m = self.create_unif_lin_array(L, M)
        
    def create_unif_lin_array(self, L, M):
        """
        Creates a uniform linear array with length 'L' and 'M' elements distributed
        over the 'x' axis. M should be odd
        
        Returns a (2, M)-shaped array with (x,y) coordinates of array elements,
        the array inter-element spacing 'd', and the indices 'm' of the array
        elements (from -M/2 to +M/2, with index 0 being at the center)
        """
    
        M_even_error = "Number of sensors must be odd"
        assert M%2, M_even_error
    
        # inter sensor spacing
        d = L/(M-1)
    
        print("Array inter sensor spacing is {} m".format(d))
    
        # array sensor indices
        m_indices = np.linspace(-(M//2), M//2, M, dtype=int)
    
        # array elements spatial coordinates
        XY_array = np.zeros((2, M))
        XY_array[0, :] = np.linspace(-L/2, L/2, M)
    
        return XY_array, d, m_indices
        

def delay_signal(x, t0, fs):
    """
    Delay a time-domain signal 'x', sampled at 'fs' Hz, by 't0' seconds.
    """
    
    X_f = np.fft.rfft(x)
    
    N_dft = x.shape[0]
    df = fs/N_dft
    f = np.linspace(0, fs-df, N_dft)[:N_dft//2+1]
    
    X_f_delayed = X_f*np.exp(-1j*2*np.pi*f*t0)
    
    return np.fft.irfft(X_f_delayed)


def create_narrowband_pulse(A, T, f0, fs):
    """
    Creates a narrowband pulse signal with amplitude 'A', duration 'T' seconds,
    center frequency 'f0' Hz, and sampling frequency 'fs' Hz.
    
    The signal is a sine wave of amplitude 'A', frequency 'f0', and random
    initial phase, modulated by a Hann window of duration 'T'.
    """
    
    # pulse duration [samples]
    N_pulse = int(T*fs)
    dt = 1./fs

    t_pulse = np.linspace(0, T-dt, N_pulse)
    p_pulse = A*np.sin(2*np.pi*f0*t_pulse + rng.uniform(0, 2*np.pi, 1)[0])*np.hanning(N_pulse)
    
    return p_pulse


def create_array_signals(SensorArrayObj, p_signal, t_initial, T, theta0_deg,
                         fs, c0=1500, SNR_dB=None):
    """
    Create a Numpy array of time-domain signals as recorded from a Uniform Linear Array
    
    SensorArrayObj: instance of SensorArray containing array geometry info
    p_signal: source signal to be inserted into array data (say, a pulse)
    t_initial: onset time (in seconds) for p_signal within sensor signal
    T: sensor signal duration (in seconds)
    theta0_deg: direction of arrival of source signal
    fs: sampling frequency [Hz]
    c0: sound speed (by default, 1500 m/s)
    SNR_dB: signal-to-noise ratio, calculated over duration of p_signal
    """
    
    # instantiate a random number generator
    rng = np.random.default_rng()
    
    # direction of arrival of signals (plane wave propagation)
    theta0 = theta0_deg*np.pi/180
    
    # vector of times-of-arrival for each sensor
    times_of_arrival = -SensorArrayObj.m*SensorArrayObj.d*np.cos(theta0)/c0
    
    N_initial= int(t_initial*fs)
    N_final = N_initial + p_signal.shape[0]
    
    # No. of samples in array data (total duration)
    N = int(T*fs)
    
    # create array of sensor signals    
    if SNR_dB is None:
        # if SNR_dB is not given, initialize signals as array of zeros (no
        # noise)
        p_array = np.zeros((SensorArrayObj.M, N))
    
    else:
        # if SNR_dB is given, add random noise to array signals at desired SNR
        signal_var = np.var(p_signal)
        noise_var = signal_var/(10**(SNR_dB/10))
        p_array = rng.normal(0., np.sqrt(noise_var), (SensorArrayObj.M, N))
    
    # for each sensor in array...
    for m in range(SensorArrayObj.M):
        # ...adds signal at time 't_initial'...
        p_array[m, N_initial:N_final] += p_signal
        
        # ...and time-shifts signal for given time-of-arrival
        p_array[m, :] = delay_signal(p_array[m, :], times_of_arrival[m], fs)
    
    return p_array



def delayandsum_beamformer(SensorArrayObj, p_array, theta, weights, fs, c0=1500):
    """
    Calculates simplified delay-and-sum beamformer for a given array geometry 
    and sensor signals, over a set of pre-determined directions.
    
    SensorArrayObj: 
    p_array:
    theta:
    weights:
    c0: speed of sound
    """
    
    N_theta = theta.shape[0]
    
    M, N_time = p_array.shape
    
    # initialize array of beamformer data (angle, time)
    y_beamformer = np.zeros((N_theta, N_time))

    # for each direction...
    for theta_i in range(N_theta):
        
        # calculate candidate time delays
        time_delays = -SensorArrayObj.m*SensorArrayObj.d*np.cos(theta[theta_i])/c0
    
        # and for each sensor...
        for m in range(M):
            # delay and sum signals
            y_beamformer[theta_i, :] += weights[m]*delay_signal(p_array[m, :], -time_delays[m], fs)
    
    # compensate for No. of sensors in array
    y_beamformer *= 1./M
    
    return y_beamformer
