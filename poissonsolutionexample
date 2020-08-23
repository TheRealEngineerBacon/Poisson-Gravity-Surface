import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

#Setup Phase
dimensions = 128
particle_count = 10
G = 6.67430e-11
fourier_sampling = 512

orbit_radii = [0, 57.9e9, 108.2e9, 149.6e9, 227.9e9, 778.6e9, 1433.5e9, 2872.5e9, 4495.1e9, 5906.4e9]
mass_array = [1.989e30, 0.330e24, 4.87e24, 5.97e24, 0.642e24, 1898e24, 568e24, 86.8e24, 102e24, 0.0146e24]
#mass_array = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
cell_dimensions = (2 * max(orbit_radii)) / dimensions
delta_x = 1/cell_dimensions


"""
#Bodies,        Mass(kg),   orbit_radius(km, e6)
    Sun,        1.989e30,   0
    Mercury,    0.330e24,   57.9
    Venus,      4.87e24,    108.2
    Earth,      5.97e24,    149.6
    Mars,       0.642e24,   227.9
    Jupiter,    1898e24,    778.6
    Saturn,     568e24,     1433.5
    Uranus,     86.8e24,    2872.5
    Neptune,    102e24,     4495.1
    Pluto,      0.0146e24,  5906.4
                
"""

den = np.ndarray((dimensions, dimensions))
den.fill(0)


for bodies in range(0, 10):
    theta = np.random.uniform(0, 2 * np.pi)
    x_ind = int((np.cos(theta) * orbit_radii[bodies]) / cell_dimensions)
    y_ind = int((np.sin(theta) * orbit_radii[bodies]) / cell_dimensions)
    
    den[y_ind + 64, x_ind + 64] = mass_array[bodies]

# for point in range(0, particle_count):
#     ranx = np.random.randint(0, dimensions)
#     rany = np.random.randint(0, dimensions)
#     # ranx = np.random.randint((dimensions/2)-50, (dimensions/2)+50)
#     # rany = np.random.randint((dimensions/2)-50, (dimensions/2)+50)
#     den[rany, ranx] = np.random.randint(500, 5000)
    
    
    

#CALCULATION PHASE    
#Initial 2D fourier transform
fourier = np.fft.fft2(den, s=(fourier_sampling, fourier_sampling))

#Center fourier for conventional visualization
centered = np.fft.fftshift(fourier)

#Reconstruct to spatial distribution to check correctness of fourier analysis.
reconstruction = np.fft.ifft2(fourier)



#Set negative frequencies to zero and convert to gravitational potential.
potential = np.zeros_like(fourier)
# potential[0, 0] = 0

n_values = np.fft.fftfreq(fourier_sampling)
# print(n_values)
# print(len(n_values))

for j in range(0, fourier_sampling):
    for i in range(0, fourier_sampling):
        if (j + i) > 0:
            potential[j, i] = -((4 * np.pi * G * fourier[j, i])/((2 * np.pi * n_values[j] * (1/delta_x))**2 + (2 * np.pi * n_values[i] * (1/delta_x))**2))
        elif (j + i) == 0:
            potential[j, i] = 0

#Inverse fourier transform the new potential transform back to spatial dimensions.
grav = np.fft.ifft2(potential)
#grav = np.fft.fftshift(np.fft.ifft2(potential))


#Plotting Phase
fig1 = plt.figure()

ax1 = plt.subplot(3, 2, 1)
ax1.set_title('Particle Distribution')
ax1.imshow(den)

ax2 = plt.subplot(3, 2, 2)
ax2.set_title('Forward Fourier')
ax2.imshow(np.real(fourier))

ax3 = plt.subplot(3, 2, 3)
ax3.set_title('Centered Fourier')
ax3.imshow(np.real(centered))

ax4 = plt.subplot(3, 2, 4)
ax4.set_title('Reconstruction')
ax4.imshow(np.real(reconstruction[0:dimensions, 0:dimensions]))

ax5 = plt.subplot(3, 2, 5)
ax5.set_title('Potential Fourier')
ax5.imshow(np.real(potential))

ax6 = plt.subplot(3, 2, 6)
ax6.set_title('Potential Field')
ax6.imshow(np.real(grav), cmap=cm.OrRd, interpolation='none')


fig2 = plt.figure()

ax7 = plt.subplot(1, 1, 1, projection='3d')
x = np.arange(0, len(grav))
y = np.arange(0, len(grav))
X, Y = np.meshgrid(x, y)
Z = grav[Y, X].real
surf = ax7.plot_surface(X, Y, Z, cmap=cm.OrRd)
