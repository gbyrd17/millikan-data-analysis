#########################################
#       Graham Byrd - PHYS307L          #
#          Millikan Oil Drop            #
#         Data Analysis Script          #
#             Dr. Becerra               #
#              3/31/2023                #
#########################################
import numpy as np
import matplotlib.pyplot as plt
from sympy import *
from uncertainties import unumpy as unp
import sys

charge_th       = 4.8e-10       # e.s.u.
stokes_const    = 8.20e-3       # pascal-meters
air_pressure    = 75.996799     # cmHg
plate_sep       = 0.878         # cm 
plate_sep_th    = 0.76          # cm 
air_viscosity   = 1.844e-4      # poise - determined by: 2.791e-7 * T^(0.7355), T in Kelvin, letting T = 298.15 K (STP), then convert to poise
oil_density     = 0.886         # kilogram-cm(-3)
air_density     = 1.1839e-3     # kilogram-cm(-3)
local_grav      = 9.792e2       # cm-second(-2)

mean_vel_f      = 3.5e-3
sig_vel_f       = 5e-4
data_raw        = np.random.normal(mean_vel_f, sig_vel_f, 1000)

sig_arr         = [sig_vel_f] * len(data_raw)
data            = unp.uarray(data_raw, sig_arr)

# find radius -> mass -> shape constant -> rise velocities for each trial of fall velocity

drop_radius     = [None] * len(data_raw)
rad_term2       = (stokes_const / (2 * air_pressure))
for ii in range(len(data_raw)):
    rad_term1           = unp.sqrt((stokes_const / (2*air_pressure)) ** 2 + ((9 * air_viscosity * data[ii]) / (2 * local_grav * oil_density)))
    drop_radius[ii]     = rad_term1 - rad_term2

mass = [None] * len(data_raw)
for ii in range(len(data_raw)):
    mass[ii] = (4 * np.pi * (data[ii] ** 3) * oil_density) / 3
    
shape_const = [None] * len(data_raw)
for ii in range(len(data_raw)):
    shape_const[ii] = (mass[ii] * local_grav) / data[ii]
    
vel_r, vel_r_th = [None] * len(data_raw), [None] * len(data_raw)
for ii in range(len(data_raw)):
    vel_r[ii]       = (( 500 / (300 * plate_sep)) * charge_th - mass[ii] * local_grav) / shape_const[ii]     
    vel_r_th[ii]    = (( 500 / (300 * plate_sep_th)) * charge_th - mass[ii] * local_grav) / shape_const[ii]  

# the manual notes that 1 excess electron on the oil will yield equal falling and rising velocities,
# find charge of electron
charge_exp, charge_exp_th = [None] * len(data_raw), [None] * len(data_raw)
term2, term3, term3_th    = [None] * len(data_raw), [None] * len(data_raw), [None] * len(data_raw)
term1           = (400 * np.pi * plate_sep) * np.sqrt((1 / (local_grav * oil_density)) * ((9 * air_viscosity) / 2) ** 3)
term1_th        = (400 * np.pi * plate_sep_th) * np.sqrt((1 / (local_grav * oil_density)) * ((9 * air_viscosity) / 2) ** 3)
for ii in range(len(data_raw)):
    term2[ii]      = unp.sqrt((1 / (1 + (stokes_const / (air_pressure * drop_radius[ii]))))**3)
    term3[ii]      = (data[ii] + vel_r[ii] * unp.sqrt(data[ii])) / 500
    charge_exp[ii] = term1 * term2[ii] * term3[ii]
    term3_th[ii]      = (data[ii] + vel_r_th[ii] * unp.sqrt(data[ii])) / 500
    charge_exp_th[ii] = term1_th * term2[ii] * term3_th[ii]

# calc error in falling vel

stdoutOrigin=sys.stdout 
sys.stdout = open("analysis_out.txt", "w")

print('The estimated value for the charge of the electron in e.s.u is %.5e(\u00B1%.5e) with %.5f%% error from theory.\n' %(np.mean(unp.nominal_values(charge_exp)), np.mean(unp.std_devs(charge_exp)), np.abs((unp.nominal_values(charge_exp)[0]-charge_th)/(charge_th))*100))
print('If we adjust the plate separation to the theoretical expected value, we get the estimation %.5e(\u00B1%.5e) with %.5f%% error from theory.\n' %(np.mean(unp.nominal_values(charge_exp_th)), np.mean(unp.std_devs(charge_exp_th)), np.abs((unp.nominal_values(charge_exp_th)[0]-charge_th)/(charge_th))*100))

sys.stdout.close()
sys.stdout=stdoutOrigin

fig, ax = plt.subplots(figsize=(13,8))
count, bins, ignored = ax.hist(data_raw, 30, density=True)
ax.plot(bins, (1/(sig_vel_f*np.sqrt(2*np.pi)) * np.exp( - (bins - mean_vel_f)**2 / (2 * sig_vel_f**2))), 'r--')
ax.set_xlabel('Fall velocity (cm/s)')
ax.set_ylabel('Counts')
ax.set_title('Artificial generation of fall velocity (cm/s)')
ax.legend(['Probability Density', 'Gaussian normal dist. - μ = %.1e, σ = %.1e' %(mean_vel_f,sig_vel_f)], loc='best')
plt.savefig('fall_velocity.png', format='png')
