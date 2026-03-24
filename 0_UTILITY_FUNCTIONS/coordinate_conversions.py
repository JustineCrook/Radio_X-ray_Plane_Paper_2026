"""
Functions to convert between coordinate systems. 
"""

##############################################################################################################

from astropy.coordinates import SkyCoord, FK4, ICRS, Angle
import astropy.units as u
import math
import numpy as np

##############################################################################################################

def galactic_to_icrs(l, b):
    
    # Create a SkyCoord object
    galactic_coord = SkyCoord(l=l* u.deg, b=b* u.deg, frame='galactic')
    
    # Convert to RA/Dec
    equatorial_coord = galactic_coord.icrs

    print("ICRS coordinates:")
    
    print(f"RA: {equatorial_coord.ra.deg} deg")
    print(f"Dec: {equatorial_coord.dec.deg} deg")
    
    print()
    print(f"RA:  {equatorial_coord .ra.to_string(unit=u.hour, sep=':', precision=5)}")
    print(f"Dec: {equatorial_coord .dec.to_string(unit=u.deg, sep=':', precision=5, alwayssign=True)}")


def icrs_to_galactic(ra_deg, dec_deg):

    c = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")
    
    print(f"Galactic l: {c.galactic.l.deg}")
    print(f"Galactic b: {c.galactic.b.deg}")



# ra_str and dec_str in the format 05h35m42.89326s
def fk4_epb1950_to_icrs_epj200(ra_str, dec_str):

    fk4_coord = SkyCoord(f"{ra_str} {dec_str}", frame='fk4', equinox='B1950')
    icrs_coord = fk4_coord.transform_to('icrs')
    
    ra_hms  = Angle(icrs_coord.ra.deg,  unit=u.deg).to_string(unit=u.hour,   sep=':', precision=3)
    dec_dms = Angle(icrs_coord.dec.deg, unit=u.deg).to_string(unit=u.degree, sep=':', precision=2, alwayssign=True)

    
    print("ICRS coordinates:")
    
    print("RA: ", ra_hms)
    print("Dec: ", dec_dms)


# ra_str and dec_str in the format 05h35m42.89326s
def icrs_epj200_to_fk4_epb1950(ra_str, dec_str):

    # Create SkyCoord in ICRS
    coord_icrs = SkyCoord(f"{ra_str} {dec_str}", frame=ICRS, unit=(u.hourangle, u.deg), equinox='J2000')
    
    # Transform to FK4 (B1950)
    coord_fk4 = coord_icrs.transform_to(FK4(equinox='B1950'))
    
    # Print results with high precision
    print("Input ICRS (J2000):", coord_icrs.to_string('hmsdms', sep=' ', precision=12))
    print("Transformed FK4 (B1950):", coord_fk4.to_string('hmsdms', sep=' ', precision=12))




def sexagesimal_to_degrees(ra_str, dec_str):
    """
    Convert RA/Dec in sexagesimal strings to degrees.

    Parameters:
    - ra_str (str): Right Ascension (e.g. '5:35:39.6945491370')
    - dec_str (str): Declination (e.g. '-66:50:41.2996631541')

    Returns:
    - ra_deg (float): RA in decimal degrees
    - dec_deg (float): Dec in decimal degrees
    """
    coord = SkyCoord(ra=ra_str, dec=dec_str, unit=(u.hourangle, u.deg), frame="icrs")
    return coord.ra.deg, coord.dec.deg


def degrees_to_sexagesimal(ra_deg, dec_deg):
    """
    Convert RA/Dec in decimal degrees to sexagesimal strings.

    Parameters:
    - ra_deg (float): Right Ascension in degrees
    - dec_deg (float): Declination in degrees

    Returns:
    - ra_str (str): RA as sexagesimal string (e.g., '05h35m39.6945s')
    - dec_str (str): Dec as sexagesimal string (e.g., '-66d50m41.2997s')
    """
    coord = SkyCoord(ra=ra_deg*u.deg, dec=dec_deg*u.deg, frame="icrs")
    ra_str = coord.ra.to_string(unit=u.hourangle, sep=':', precision=6, pad=True)
    dec_str = coord.dec.to_string(unit=u.deg, sep=':', precision=6, alwayssign=True, pad=True)
    return ra_str, dec_str



def rad_to_hms(ra_rad):
    """
    Convert Right Ascension from radians to hours:minutes:seconds.
    """
    if ra_rad<0: ra_rad= ra_rad+2*np.pi
    ra_deg = math.degrees(ra_rad)  # Convert radians to degrees
    ra_hours = ra_deg / 15         # Convert degrees to hours (1 hour = 15 degrees)
    hours = int(ra_hours)
    minutes = int((ra_hours - hours) * 60)
    seconds = (ra_hours - hours - minutes / 60) * 3600
    return f"{hours:02d}:{minutes:02d}:{seconds:06.8f}"


def rad_to_dms(dec_rad):
    """
    Convert Declination from radians to degrees:minutes:seconds.
    """
    dec_deg = math.degrees(dec_rad)  # Convert radians to degrees
    sign = "+" if dec_deg >= 0 else "-"  # Check the sign of the declination
    dec_deg = abs(dec_deg)              # Work with the absolute value
    degrees = int(dec_deg)
    minutes = int((dec_deg - degrees) * 60)
    seconds = (dec_deg - degrees - minutes / 60) * 3600
    return f"{sign}{degrees:02d}:{minutes:02d}:{seconds:06.8f}"




##############################################################################################################


def angular_distance(ra1_deg, dec1_deg, ra2_deg, dec2_deg):
    """
    Calculate the angular distance between two points on the celestial sphere.
    Coordinates are given in degrees.

    Parameters:
    - ra1_deg, dec1_deg: Right Ascension and Declination of the first point (degrees)
    - ra2_deg, dec2_deg: Right Ascension and Declination of the second point (degrees)

    Returns:
    - Angular distance in arcseconds
    """
    # Convert degrees to radians
    ra1_rad = math.radians(ra1_deg)
    dec1_rad = math.radians(dec1_deg)
    ra2_rad = math.radians(ra2_deg)
    dec2_rad = math.radians(dec2_deg)

    # Apply haversine formula
    delta_ra = ra2_rad - ra1_rad
    delta_dec = dec2_rad - dec1_rad

    a = math.sin(delta_dec / 2)**2 + math.cos(dec1_rad) * math.cos(dec2_rad) * math.sin(delta_ra / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    # Convert radians to arcseconds
    distance_arcsec = math.degrees(c) * 3600
    print(f"Angular distance: {distance_arcsec:.6f} arcseconds")
    print(f"Angular distance: {distance_arcsec/60:.6f} arcminutes")




##############################################################################################################