from scipy.fftpack import fftshift, fftfreq
import numpy as np
from math import pi, sin
import math
import os


def get_file(file):
    # Returns the full path, if the file is in the same folder as the main .py program.
    return os.path.join(os.path.dirname(file), file)


def get_folder_file(folder, file):
    # Returns the full path, if the file is not in the same folder as the main .py program.
    # If this does not work, use: return get_file(os.path.join(folder, file))
    return os.path.join(folder, file)


def find_nearest_peak(fft, oldrange, oldangle, oldvelocity):
    magnitude = np.real(np.sqrt(fft * np.conj(fft)))
    max_val = np.max(magnitude)
    threshold = max_val / 3
    indices = magnitude > threshold
    newrange = oldrange[indices]
    newangle = oldangle[indices]
    newvelocity = oldvelocity[indices]
    
    # Take closest point
    min_idx = np.argmin(newrange)
    newrange = newrange[min_idx]
    newangle = newangle[min_idx]
    newvelocity = newvelocity[min_idx]
    return newrange, newangle, newvelocity

def real_angle(x_drone,y_drone,x_obst,y_obst, ox_drone, oy_drone, oz_drone, ow_drone):
    # Yaw is from optitrack
    yaw = math.atan2((2 * oy_drone * ow_drone - 2 * oz_drone * ox_drone), (1 - 2*oy_drone * oy_drone - 2*ox_drone * ox_drone)) #drone yaw angle
    heading = math.atan2((x_obst - x_drone), (y_obst - y_drone)) + yaw #heading adjustment compensating for yaw
    return heading, yaw

def real_distance(x_drone,y_drone,x_obst,y_obst): # Distance to obstacle calculation (data from optitrack)
    return math.sqrt((x_drone-x_obst)**2 + (y_drone-y_obst)**2) - 0.2 

def range_angle_velocity_calc(freq1, freq2, phi_1, phi_2, chirp_time, phi_velocity=None):
    B = 250e6   # Hz (bandwidth range)
    c = 2.99792458e8   # m/s
    T = chirp_time # total time
    
    # Scale the frequency to the maximum range
    R_max = 25   # maximum range in meters - Unnesecary comment
    F_max = 2 * B * R_max / (c * T)
    freq1 = 2 * freq1 * F_max / 0.5   # We assume we multiply by 2 since we cut the x-axis in half. - better explanation?
    freq2 = 2 * freq2 * F_max / 0.5

    d_test_2 = (sin((pi/180) * (76/2)))**-1 * (0.0125 / 2)  # TODO: need update Distance between two receivers - Remove
    f_temp = 24E9  # TODO: need update - Remove
    range_lst1 = (c * T * freq1 / (2 * B) )  # Range formula for receiver 1
    range_lst2 = (c * T * freq2 / (2 * B) )  # Range formula for receiver 2 - Unnesecary comment

    delta_omega = phi_1 - phi_2  # Difference between phases 

    z = -1
    for i in delta_omega: #formatting... trust us - (tamim and ilten)  - Unnesecary comment
        z += 1
        if i <= 0:
            delta_omega[z] = i + 2*np.pi 
    
    delta_omega = delta_omega - np.pi
    # --------

    temp_constant = c * delta_omega / (2 * pi * f_temp * d_test_2)  # Angle formula 
    geo_angle_lst = np.arcsin(temp_constant)  # Final angle found in degrees  - Unnesecary comment

    if phi_velocity is not None:
        delta_phase_vel = phi_velocity - phi_1
        velocity_lst = (c * abs(delta_phase_vel) / (2 * f_temp * T))  # Final velocity found including velocity formula
    else:
        velocity_lst = np.zeros(len(delta_omega))

    return range_lst1, range_lst2, geo_angle_lst, velocity_lst

def phase_calc(FFT):  # NOTE: beautiful, outstanding, revolutionary - Unnesecary comment
    return np.angle(FFT)  # imaginary over real based on definitions
    

def combined_FFT(f_hat_re, f_hat_im):
    # F[a+jb] = F[a] + jF[b] = c + jd + je - f = (c-f) + j(d+e)  - Unnesecary comment
    # F[a] = c + jd  - Unnesecary comment
    # F[b] = e + jf - Unnesecary comment
    return (f_hat_re.real - f_hat_im.imag) + 1j * (f_hat_re.imag + f_hat_im.real) 
    

def fourier(chirps, t, realim, duration):  # Calculate the fourier of the signal 
    f_hat = fftshift(np.fft.fft(chirps[realim]))  # Frequency array already zero padded according to documentation  - Unnesecary comment
    # add this within the fft:   , n=2*len(chirps[0])  - Unnesecary comment
    return f_hat

def PSD_calc(f_hat, t, duration, chirps, sample_rate):
    """ Intensity of frequency calculated using fourier """
    dt = duration / len(chirps[0])  # Realim rx1 0,1 rx2 = 2,3  (0, 2 real; 1, 3 imaginary) - Not clear what is meant here
    n = len(t)  # Total number of timestamps
    
    PSD = np.real(f_hat * np.conj(f_hat) / n)  # Calculates the power spectral density  (np.real returns + 0*j)  - Unnesecary comment
    
    freq = (1 / (dt * n)) * np.arange(n)  # Frequency calculated
    # L = np.arange(1, np.floor(n / 2), dtype='int')  # Prevent a divison of intensity over a twice long domain - Unnesecary comment
    
    mag = np.absolute(f_hat)   # calculates the magnitude / amplitude
    PSD = mag

    # Normalise and only look at positive frequencies
    freq = fftshift(fftfreq(n, 1))[:n]
    
    PSD = PSD[freq >= 0]
    f_hat = f_hat[freq >= 0]
    freq = freq[freq >= 0]

    return PSD, freq, f_hat


def chirp_func(timestamp, radar_msg, chirp_no=None):
    
    # Average the chirps for faster computation
    rx1_re = np.array(radar_msg[int(timestamp)].data_rx1_re)
    rx1_im = np.array(radar_msg[int(timestamp)].data_rx1_im)
    rx2_re = np.array(radar_msg[int(timestamp)].data_rx2_re)
    rx2_im = np.array(radar_msg[int(timestamp)].data_rx2_im)
    # NOTE: I think the name of the variable is explainatory enough, no comment required - Unnesecary comment

    no_chirps = radar_msg[int(timestamp)].dimx  # 16 chirps
    length_chirp = radar_msg[int(timestamp)].dimy  # 128 elements in a chirp (each with rx 1 and rx 2 and Re and Im)
    
    # Select only one chirp. Chirp_no indicates which chirp we want.
    rx1_re_chirp = rx1_re[chirp_no*length_chirp:(chirp_no + 1)*length_chirp]
    rx1_im_chirp = rx1_im[chirp_no*length_chirp:(chirp_no + 1)*length_chirp]
    rx2_re_chirp = rx2_re[chirp_no*length_chirp:(chirp_no + 1)*length_chirp]
    rx2_im_chirp = rx2_im[chirp_no*length_chirp:(chirp_no + 1)*length_chirp]

    # Add zero-padding before assembling it into the chirps array
    rx1_re_chirp = np.append(rx1_re_chirp, np.zeros(length_chirp))
    rx1_im_chirp = np.append(rx1_im_chirp, np.zeros(length_chirp))
    rx2_re_chirp = np.append(rx2_re_chirp, np.zeros(length_chirp))
    rx2_im_chirp = np.append(rx2_im_chirp, np.zeros(length_chirp))

    # Assemble into one array
    chirps = np.array([rx1_re_chirp, rx1_im_chirp, rx2_re_chirp, rx2_im_chirp])

    return chirps, no_chirps, length_chirp 

    