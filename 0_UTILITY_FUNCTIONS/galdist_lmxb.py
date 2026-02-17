"""
Using galactic mass density models as a probability distribution for distance:

Using the model developed by Grimm et al. 2002 and implemented by Atri et al. 2019, we estimate a probability density function for distance along the line of sight towards a particular direction (e.g., towards a source of interest).
"""

import numpy as np
from scipy.integrate import quad
#import dynesty
from astropy.coordinates import SkyCoord
import astropy.units as u
import matplotlib.pyplot as plt
#plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rcParams['text.usetex'] = False



#########################################################
## MODEL COMPONENTS

# Models the Galactic bulge.
# r is radial cylindrical distance from Galactic center, z is height above the plane.
# Shape controlled by parameters: q (flattening), gamma (inner slope), rt (cutoff radius).
# Returns bulge density at (r, z).
def bulge(r, z):
    rho_0b = 1.0719
    gamma = 1.8
    q = 0.6
    rt = 1.9
    k = (r**2 + ((z**2)/(q**2)))
    return rho_0b*((np.sqrt(k))**(-gamma))*np.exp(-k/(rt**2))

# Models the thin disk.
# Exponentially falls off with radius and vertical height.
# rm shifts the density peak outward from the centre.
def disk(r, z):
    rho_0d = 2.6387
    rm = 6.5
    rd = 3.5
    rz = 0.41
    return rho_0d*(np.exp(-(rm/rd)-(r/rd)-(np.abs(z)/rz)))

# Models the stellar halo or spheroid.
# R is spherical Galactocentric distance.
# Has a de Vaucouleurs-like profile for outer stars (common in spheroids/halos).
def sphere(R):
    rho_0s = 13.0976
    bs = 7.669
    Re = 2.8
    return rho_0s*(np.exp(-bs*((R/Re)**(1.0/4)))/((R/Re)**(7.0/8.0)))

# Computes total probability density of finding an object at distance x along a line of sight defined by Galactic coordinates.
# Converts x to Galactocentric coordinates:
# z: vertical height from the plane.
# r: cylindrical Galactocentric distance.
# R: spherical Galactocentric distance.
# Adds the three component densities.
# Multiplies by x² to account for volume element in spherical coordinates — since probability grows with volume (4πx²dx).
def lmxb_gal_distribution(x, gal_lon, gal_lat):
    z = x * np.sin(np.radians(gal_lat))
    R0 = 8
    r = np.sqrt(R0**2 + ((x*np.cos(np.radians(gal_lat)))**2) - 2 * x * R0 * np.cos(np.radians(gal_lat)) * np.cos(np.radians(gal_lon)))
    R = np.sqrt(R0**2 + (x**2) - 2 * x * R0 * np.cos(np.radians(gal_lat)) * np.cos(np.radians(gal_lon)))
    rho_b = bulge(r, z)
    rho_d = disk(r, z)
    rho_s = sphere(R)
    return (rho_b+rho_d+rho_s)*((x*1e3)**2)


#########################################################
## TARGET

def get_deg(ra, dec):
    coord = SkyCoord(ra, dec, frame='icrs')

    RA = coord.ra.deg
    DEC = coord.dec.deg

    print('RA in degrees:', RA)
    print('Dec in degrees:', DEC)

    return RA, DEC


def plot_PDF(SOURCE_RA = 265.15878, SOURCE_DEC = -27.62004, DISTANCE_LOLIM = 1, DISTANCE_HARDLIMIT= 27, sample=False , d_eval_min = 6, d_eval_max = 17, zoom=False):
    
    # Define the target position in equatorial coordinates
    TARGET = SkyCoord(SOURCE_RA*u.deg, SOURCE_DEC*u.deg)

    # Convert the position to Galactic longitude and latitude
    # Integrate the distribution over the distance range [LOLIM, HARDLIMIT], which is needed to convert density to a true PDF
    # lmxb_gal_distribution is the function to integrate
    # DISTANCE_LOLIM and DISTANCE_HARDLIMIT are respectively the lower and upper limits for integration
    PDF_NORM = quad(lmxb_gal_distribution, DISTANCE_LOLIM, DISTANCE_HARDLIMIT, args=(TARGET.galactic.l.deg, TARGET.galactic.b.deg))[0]

    # Define distances where we will evaluate the PDF 
    EVAL_DIST = np.linspace(DISTANCE_LOLIM, DISTANCE_HARDLIMIT, 10000)
    # Normalise the distribution to get a PDF 
    EVAL_PDF = lmxb_gal_distribution(EVAL_DIST, TARGET.galactic.l.deg, TARGET.galactic.b.deg) / PDF_NORM

    # Find index of maximum PDF
    max_index = np.argmax(EVAL_PDF)
    # Get corresponding EVAL_DIST
    corresponding_dist = EVAL_DIST[max_index]
    print("EVAL_DIST at max PDF:", corresponding_dist)
    

    # Plot the results
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(EVAL_DIST, EVAL_PDF, color='indigo', lw=4)
    if zoom:ax.set_xlim(d_eval_min, d_eval_max)
    else: ax.set_xlim(DISTANCE_LOLIM, DISTANCE_HARDLIMIT)
    ax.set_ylim(0)
    ax.set_xlabel('Distance (kpc)', fontsize=20)
    ax.set_ylabel('Probability Density (kpc$^{-1}$)', fontsize=20)
    ax.minorticks_on()
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.tick_params(axis='both', which='major', length=5)
    ax.tick_params(axis='both', which='minor', length=2.5)
    ax.tick_params(axis='both', which='both', direction='in', right=True, top=True)
    plt.show()


    print(np.trapezoid(EVAL_PDF, EVAL_DIST))  # should be ~1
    print()
    mask = EVAL_DIST >= d_eval_min
    prob_above= np.trapezoid(EVAL_PDF[mask], EVAL_DIST[mask])
    print(f"Probability of distance > {d_eval_min} kpc: {prob_above:.4f}")
    mask = EVAL_DIST <= d_eval_max
    prob_below= np.trapezoid(EVAL_PDF[mask], EVAL_DIST[mask])
    print(f"Probability of distance < {d_eval_max} kpc: {prob_below:.4f}")


    """
    if sample:

        # Function translating a unit cube to the parameter space according to the prior.
        # The following gives flat priors in the range [0, UPLIM_DISTANCE)
        def prior_transform(u):
            x = np.array(u)
            x[0] = DISTANCE_LOLIM + (DISTANCE_HARDLIMIT - DISTANCE_LOLIM) * u[0] # UPLIM_DISTANCE * u[0] 
            return x

        # Returns log of normalised PDF at distance x (i.e. the log likelihood)
        def loglike(x):
            return np.log(lmxb_gal_distribution(x[0], TARGET.galactic.l.deg, TARGET.galactic.b.deg) / PDF_NORM)

        # Bayesian sampling (via dynesty) to get a posterior over distances.
        # Runs the nested sampling algorithm to produce samples from the PDF.
        # Samples represent probable distances to the target object based on the Galactic model.
        # ndim: The number of parameters returned by prior_transform and accepted by loglikelihood
        # nlive: Number of “live” points. Larger numbers result in a more finely sampled posterior (more accurate evidence), but also a larger number of iterations required to converge.
        sampler = dynesty.DynamicNestedSampler(loglike, prior_transform, ndim=1,nlive=1000)
        sampler.run_nested()
        res = sampler.results
        new_samples = res.samples_equal()

        # Overlay the analytic PDF with a histogram of the sampled distances.
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(EVAL_DIST, EVAL_PDF, color='indigo', lw=4, alpha=0.3,label='PDF')
        ax.hist(new_samples,density=True,bins=100,histtype='step',color='k',label='Random sample')
        ax.legend(fontsize=20)
        ax.set_xlim(0, 20)
        ax.set_ylim(0)
        ax.set_xlabel('Distance (kpc)', fontsize=20)
        ax.set_ylabel('Probability Density (kpc$^{-1}$)', fontsize=20)
        ax.minorticks_on()
        ax.tick_params(axis='both', which='major', labelsize=18)
        ax.tick_params(axis='both', which='major', length=5)
        ax.tick_params(axis='both', which='minor', length=2.5)
        ax.tick_params(axis='both', which='both', direction='in', right=True, top=True)
        plt.show()
    """