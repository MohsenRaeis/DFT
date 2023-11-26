# Trying to solve the Shrodinger Equation in Hydrogen atom
# Expanding the orbital variation through the Ylm(\theta, \phi)
# Expanding the radial term in ters of Fourier componenets
from scipy.special import sph_harm as Ylm
from numpy import cos, sin, exp, pi, zeros, linspace, fft, sqrt, abs
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')

habr = 1.05457e-34
e = 1.60e-19
mu0 = 4*pi*1e-7
eps0 = 8.85e-12

def plt_mag_ph(fig, ax, val_mat, label=''):
    divider0 = make_axes_locatable(ax[0])
    cax0 = divider0.append_axes('right', size='5%', pad=0.05)
    divider1 = make_axes_locatable(ax[1])
    cax1 = divider1.append_axes('right', size='5%', pad=0.05)
    im0 = ax[0].imshow(20*log10(abs(val_mat)), cmap='bone')
    fig.colorbar(im0, cax=cax0, orientation='vertical')
    ax[0].set_title(r'$abs(%s)$'%label)
    im1 = ax[1].imshow(angle(val_mat))
    fig.colorbar(im1, cax=cax1, orientation='vertical')
    ax[1].set_title(r'$\angle %s$'%label)

def cart2rad(x, y, z):
    phi = angle(Xm+1j*Ym)
	r = abs(z + 1j*abs(Xm+1j*Ym))
	theta = arccos(z/r)
	return r, theta, phi

def rad2cart(r, th, ph):
    x = r*sin(th)*cos(ph)
    y = r*sin(th)*sin(ph)
    z = r*cos(th)
    return x, y, z

def rad2cartV(vr, vt, vp, th, ph):
    vx = vr*sin(th)*cos(ph) + vt*cos(th)*cos(ph) - vp*sin(ph)
    vy = vr*sin(th)*sin(ph) + vt*cos(th)*sin(ph) + vp*cos(ph)
    vz = vr*cos(th) - vt*sin(th)
    return vr, vt, vp

def rad2cartV(vx, vy, vz, th, ph):
    vx = vx*sin(th)*cos(ph) + vy*sin(th)*sin(ph) + vz*cos(ph)
    vt = vx*cos(th)*cos(ph) + vy*cos(th)*sin(ph) - vz*sin(ph)
    vz = -vx*sin(th) + vy*cos(th)
    return vr, vt, vp

th = linspace(-pi/2, pi/2, 500)
ph = np.linspace(0.01, pi, 10)
PHm, THm = np.meshgrid(ph, th)



plt.imshow(Ylm)

