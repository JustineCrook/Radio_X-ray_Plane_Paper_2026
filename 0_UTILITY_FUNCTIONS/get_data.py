
from io import StringIO
from typing import Optional, Tuple
import pandas as pd
import numpy as np
import glob
from pathlib import Path
from astropy.time import Time



##############################################################################################################
## UNIT CONVERSIONS:

# 1 Jy = 1e-23 erg s^-1 cm^-2 Hz^-1
# 1 kpc = 1e3 * 3.0857e18 cm : https://lweb.cfa.harvard.edu/~dfabricant/huchra/ay145/constants.html
jansky_to_erg_s_cm2_Hz = 1e-23
kpc_to_cm = 1e3 * 3.0857e18

# The 90% confidence interval is 1.645σ for a Gaussian: https://kalmanfilter.net/confInterval.html? ; https://www.andreaminini.net/math/critical-values-in-the-normal-distribution
gaussian_90_to_68 = 1/1.645




##############################################################################################################
## HELPERS:

def get_utc(mjd):
    """
    Convert MJD to UTC datetime.
    """
    t = Time(mjd, format='mjd')
    utc_datetime = t.to_datetime()
    print(f"MJD {mjd}  -->  UTC {utc_datetime}")


def get_mjd(utc):
    """
    Convert UTC datetime to MJD.
    """
    t = Time(utc, format='isot', scale='utc')
    mjd = t.mjd
    print(f"UTC {utc}  -->  MJD {mjd}")
    return mjd


##############################################################################################################
## HELPERS: FLUX/LUMINOSITY CONVERSION FUNCTIONS


def convert_Fr(Fr_mJy, Fr_mJy_unc, d_kpc, d_kpc_unc, nu_GHz=1.28):
    """
    Convert radio flux density in units of mJy to radio luminosity in units of erg/s.
    Also returns error -- assuming nu has no uncertainty. 
    """
    
    S = Fr_mJy * 1e-3 * jansky_to_erg_s_cm2_Hz  # convert mJy to Jy, then Jy to erg s^-1 cm^-2 Hz^-1
    d = d_kpc * kpc_to_cm  # convert kpc to cm
    nu = nu_GHz * 1e9  # convert GHz to Hz

    # Calculate luminosity
    L = 4 * np.pi * d**2 * S * nu    # in erg/s
    L_unc = L*np.sqrt( (Fr_mJy_unc/Fr_mJy)**2 + (2* d_kpc_unc/d_kpc)**2)

    return L, L_unc


def convert_Fr_mono(Fr_mJy, Fr_mJy_unc, d_kpc, d_kpc_unc):
    """
    Convert radio flux density in units of mJy to radio monochromatic luminosity in units of erg/s/Hz.
    Also returns error -- assuming nu has no uncertainty.
    Factor is 1.1967e18.
    """
    S = Fr_mJy * 1e-3 * jansky_to_erg_s_cm2_Hz        # convert mJy to Jy, then Jy to erg s^-1 cm^-2 Hz^-1
    d = d_kpc * kpc_to_cm       # convert kpc to cm

    # Calculate luminosity
    L = 4 * np.pi * d**2 * S     # in erg/s/Hz

    # Calculate luminosity uncertainty
    L_unc = L*np.sqrt( (Fr_mJy_unc/Fr_mJy)**2 + (2* d_kpc_unc/d_kpc)**2)

    return L, L_unc



def convert_Fx(Fx_erg_s_cm2, Fx_erg_s_cm2_unc_l, Fx_erg_s_cm2_unc_u, d_kpc, d_kpc_unc):
    """
    Convert X-ray flux in units of erg/s/cm^2 to luminosity in units of erg/s.
    Factor in front of flux is about 1.1967e44.
    """

    S = Fx_erg_s_cm2 # units of erg/s/cm^2
    d = d_kpc * kpc_to_cm  # convert kpc to cm

    # Calculate luminosity
    L = 4 * np.pi * d**2 * S     # in erg/s
    
    # Calculate luminosity uncertainty
    L_unc_l = L*np.sqrt( (Fx_erg_s_cm2_unc_l/Fx_erg_s_cm2)**2 + (2* d_kpc_unc/d_kpc)**2)
    L_unc_u = L*np.sqrt( (Fx_erg_s_cm2_unc_u/Fx_erg_s_cm2)**2 + (2* d_kpc_unc/d_kpc)**2)

    return L, L_unc_l, L_unc_u



def Fr(nu_GHz, S0_mJy, alpha, nu0_GHz = 1.28):
    """
    Convert flux density to a different frequency, assuming a particular spectral index.
    """

    S_mJy = S0_mJy * (nu_GHz/nu0_GHz)**alpha

    print(f"{S_mJy} mJy")
    


##############################################################################################################
## READ DATA FOR A PARTICULAR SOURCE

def read_data(file_path, add_sys_er = True, verbose=True):
    """
    Function to read data from a text file and return the source metadata, observation metadata, radio data, and X-ray data as separate Pandas DataFrames.
    """

    # Get the file path
    p = Path(file_path)
    source_name = p.stem
    if verbose: print(source_name)

    # Dictionary to store DataFrames
    sections = {}

    with open(file_path, 'r') as f:
        f.readline() # Ignore first line, as this just states who I got the data from
        content = f.read()

    # Split content by lines that start with '##', as these are the section headers
    raw_sections = content.split('## ')
    for section in raw_sections:

        if not section.strip():  # Skip empty sections
            continue

        # Get section name (first line) and data (the rest)
        lines = section.strip().split('\n')
        section_name = lines[0].strip()  # First line is the section name

        columns = [c.strip() for c in lines[1].lstrip('#').split(',')]
        #print(columns)

        data = '\n'.join(line for line in lines[2:] if not line.strip().startswith('#')) # Remaining lines, without those starting with '#'


        # Convert the data to a Pandas DataFrame
        df = pd.read_csv(StringIO(data), header=None, sep=',', encoding_errors='ignore', engine="python", skipinitialspace=True, quotechar='"') 

        # Assign column headers 
        try:
            df.columns = columns
        except:
            raise ValueError(f"Incorrect or no headers for section: {section_name}")
        

        # Store in dictionary
        sections[section_name] = df


    # Assign each section to a dataframe
    source_df, obs_df, radio_df, xray_df= sections["SOURCE_METADATA"], sections["OBS_METADATA"], sections["RADIO_DATA"], sections["XRAY_DATA"]

    # Get the X-ray confidence interval (e.g. 68% or 90%) for the statistical uncertainty.
    xray_CI = obs_df["xray_CI"].values[0] 

    # Add a column for the source name to each dataframe.
    radio_df.insert(0, "name", source_name)
    xray_df.insert(0, "name", source_name)
    source_df.insert(0, "name", source_name)
    obs_df.insert(0, "name", source_name)
 


    ############ PROCESS RADIO AND X-RAY DATAFRAMES ############


    ## Sort the dataframes 
    radio_df.sort_values(by='t_radio', ascending=True, inplace=True)
    xray_df.sort_values(by='t_xray', ascending=True, inplace=True)


    #### RADIO DATA

    # Replace NaN values in Rstate with "Unclear"
    radio_df["Rstate"] = radio_df["Rstate"].fillna("Unclear")

    # Detections:
    mask_detection = radio_df['Fr_uplim'].isna() # mask = True where there is a detection, i.e. no data in the uplim column
    # Check whether any rows for the detections have Fr or Fr_unc missing
    missing_mask = mask_detection & (radio_df[['Fr', 'Fr_unc']].isna().any(axis=1)) # True if either Fr or Fr_unc or both are NaN
    if missing_mask.any():
        if verbose: print(f"Warning for row(s): {list(radio_df.index[missing_mask] + 1)} -- Fr or Fr_unc is empty for this detection.")
    # Add systematic uncertainty (5%) for the detections -- use vectorisation
    if add_sys_er:
        radio_df.loc[mask_detection, 'Fr_unc'] = np.sqrt(radio_df.loc[mask_detection, 'Fr_unc']**2 + (0.05 * radio_df.loc[mask_detection, 'Fr'])**2)
        if verbose: print("Added 5% systematic uncertainty to the radio data.")
    
    # Upper limits:
    mask_uplim = ~mask_detection
    # Check if there are conficts
    conflict_mask = mask_uplim & ~(radio_df[['Fr', 'Fr_unc']].isna().all(axis=1)) # True if at least one of Fr or Fr_unc NOT NaN
    if conflict_mask.any():
        if verbose: print(f"Warning for row(s): {list(radio_df.index[conflict_mask] + 1)} -- Fr or Fr_unc is not empty when Fr_uplim is present.")
    # Apply transformations for uplim rows
    radio_df.loc[mask_uplim, 'Fr'] = radio_df.loc[mask_uplim, 'Fr_uplim'] # Assign Fr_uplim to Fr
    radio_df.loc[mask_uplim, 'Fr_unc'] = radio_df.loc[mask_uplim, 'Fr_uplim'] / 3 # Set Fr_unc = Fr_uplim / 3
    # Convert NaN values in Fr_uplim to False, and True where values are present
    radio_df['Fr_uplim_bool'] = radio_df['Fr_uplim'].notna()
    # Drop the Fr_uplim column
    radio_df.drop(columns=['Fr_uplim'], inplace=True)


    ######################
    ### X-RAY DATA

    # Replace NaN values in Xstate with "Unclear"
    xray_df["Xstate"] = xray_df["Xstate"].fillna("Unclear")
    
    # Detections:
    mask_detection = xray_df['Fx_uplim'].isna() # mask = True where there is a detection, i.e. no data in the uplim column
    # Check whether any rows for the detections have Fx or Fx_unc_l or Fx_unc_u missing
    missing_mask = mask_detection & (xray_df[['Fx', 'Fx_unc_l', 'Fx_unc_u']].isna().any(axis=1)) # True if Fx, Fx_unc_l, Fx_unc_u, or all are NaN
    if missing_mask.any():
        if verbose: print(f"Warning for row(s): {list(xray_df.index[missing_mask] + 1)} -- Fx, Fx_unc_l, or Fx_unc_u is empty for this detection.")
    # Replace NaN values in Fx_unc_u with values from Fx_unc_l -- i.e. in cases where people just had a symmetric uncertainty and didn't fill it in for the other value
    xray_df.loc[mask_detection,'Fx_unc_u'] = xray_df.loc[mask_detection,'Fx_unc_u'].fillna(xray_df.loc[mask_detection,'Fx_unc_l'])
    # Add systematic uncertainty (10%) for the detections -- use vectorisation
    if add_sys_er: # Systematic uncertainty was not included in the quoted uncertainty
        # If 90% uncertainty was used for the statistical uncertainty, convert this to 68%, assuming Gaussian errors
        # Note that this is not quite correct for the cstat points, but suffices for our purposes
        if verbose: print("X-ray uncertainty percentage: ", xray_CI)
        if xray_CI==68: pass
        elif xray_CI==90:
            if verbose: print("Converting uncertainties to 68% (assuming Gaussian errors).")
            xray_df.loc[mask_detection, 'Fx_unc_l'] = xray_df.loc[mask_detection, 'Fx_unc_l']*gaussian_90_to_68
            xray_df.loc[mask_detection, 'Fx_unc_u'] = xray_df.loc[mask_detection, 'Fx_unc_u']*gaussian_90_to_68
        else: 
            if verbose: print("Warning: The X-ray statistical uncertainty confidence interval should be 68 percent or 90 percent... Assuming it is 68 percent.")
        
        # Add systematic uncertainty (10%) for the detections ; except for XTE J1701-462, which we give 20% uncertainty (see the paper)
        if source_name=="XTE J1701-462":
            perc = 0.2
        else: 
            perc = 0.1
        xray_df.loc[mask_detection, 'Fx_unc_l'] = np.sqrt(xray_df.loc[mask_detection, 'Fx_unc_l']**2 + (perc * xray_df.loc[mask_detection, 'Fx'])**2)
        xray_df.loc[mask_detection, 'Fx_unc_u'] = np.sqrt(xray_df.loc[mask_detection, 'Fx_unc_u']**2 + (perc * xray_df.loc[mask_detection, 'Fx'])**2)
        if verbose: print(f"Added {perc*100}% systematic uncertainty to the X-ray data.")
    
    
    # Upper limits:
    mask_uplim = ~mask_detection
    # Check if there are conficts
    conflict_mask = mask_uplim & ~(xray_df[['Fx', 'Fx_unc_l', 'Fx_unc_u']].isna().all(axis=1))  # True if at least one of Fx, Fx_unc_l, or Fx_unc_u is NOT NaN
    if conflict_mask.any():
        if verbose: print(f"Warning for row(s): {list(xray_df.index[conflict_mask] + 1)} -- Fx, Fx_unc_l, or Fx_unc_u is not empty when Fx_uplim is present.")
    # Apply transformations for uplim rows
    xray_df.loc[mask_uplim, 'Fx'] = xray_df.loc[mask_uplim, 'Fx_uplim'] # Assign Fx_uplim to Fx
    xray_df.loc[mask_uplim, 'Fx_unc_l'] = xray_df.loc[mask_uplim, 'Fx_uplim'] / 3 # Set Fx_unc_l = Fx_uplim / 3
    xray_df.loc[mask_uplim, 'Fx_unc_u'] = xray_df.loc[mask_uplim, 'Fx_uplim'] / 3 # Set Fx_unc_u = Fx_uplim / 3
    # Convert NaN values in Fx_uplim to False, and True where values are present
    xray_df['Fx_uplim_bool'] = xray_df['Fx_uplim'].notna()
    # Drop the Fx_uplim column
    xray_df.drop(columns=['Fx_uplim'], inplace=True)
    


    return source_df, obs_df, radio_df, xray_df






##############################################################################################################
## GET BAHRAMIAN DATA -- FOR COMPARISON PURPOSES
# Note that the radio luminosity is at 5 GHz

def get_bahramian_data(path_to_data = "../bahramian_DATA/", include_oddsources=False, convert_Fr=True, include_uplims=False):
    """
    Function to read data from various CSV files and construct a single Pandas dataframe.

    Parameters
    ----------
    - path_to_data: directory containing all csv data files 
    - include_oddsources: boolean, indicating whether Cyg X-1 and GRS1915 should be included

    Returns
    -------
    - data: a single Pandas dataframe containing all data provided for LrLx

    """

    # Read the data from the CSV files and concatenate them into a single dataframe
    DATA_BHs = pd.read_csv(path_to_data+'lrlx_data_BHs.csv')
    DATA_candBHs = pd.read_csv(path_to_data+'lrlx_data_candidateBHs.csv')
    DATA_NSs = pd.read_csv(path_to_data+'lrlx_data_NSs.csv')
    DATA_canNSs = pd.read_csv(path_to_data+'lrlx_data_candidateNSs.csv')
    DATA_AMXPs = pd.read_csv(path_to_data+'lrlx_data_AMXPs.csv')
    DATA_tMSPs = pd.read_csv(path_to_data+'lrlx_data_tMSPs.csv')
    DATA_WDs = pd.read_csv(path_to_data+'lrlx_data_WDs.csv')
    if include_oddsources:
        DATA_oddsrcs = pd.read_csv(path_to_data+'lrlx_data_oddsrcs.csv')
        DATA_LIST = [DATA_BHs, DATA_candBHs, DATA_NSs, 
                     DATA_canNSs, DATA_AMXPs, DATA_tMSPs, 
                     DATA_tMSPs, DATA_WDs, DATA_oddsrcs]
    else:
        DATA_LIST = [DATA_BHs, DATA_candBHs, DATA_NSs, 
                     DATA_canNSs, DATA_AMXPs, DATA_tMSPs, 
                     DATA_tMSPs, DATA_WDs]
    DATA = pd.concat(DATA_LIST,ignore_index=True)


    # Only include detections, if specified
    if include_uplims==False:
        print("Only including detections from Bahramian et al. data.")
        DATA = DATA[DATA["uplim"].isna()]


    # Remove rows where "Class" is "WD" or "tMSP" -- since we don't have any of these sources in our sample.
    DATA = DATA[~DATA["Class"].isin(["WD", "tMSP"])]


    # Remove rows where "Class" is "candidateBH"
    # These were not identified as candidates with the same techniques (during outburst) used to identify what the current paper defines as BH candidates as observed with MeerKAT
    DATA = DATA[~DATA["Class"].isin(["candidateBH"])]


    # Replace "AMXP" with "NS" in the "Class" column
    DATA["Class"] = DATA["Class"].replace("AMXP", "NS")


    # Extract the relevant columns as numpy arrays
    lr_all = DATA["Lr"].to_numpy()
    lx_all= DATA["Lx"].to_numpy()
    source_class = DATA["Class"].to_numpy()
    uplims = DATA["uplim"].to_numpy()
    uplims_lr = (uplims == "Lr")
    uplims_lx = (uplims == "Lx")

    # Bahramian flux densities are at 5 GHz, but ours is at 1.28 GHz.
    # Assuming flat spectral index for Bahramian data, then the flux at the two frequencies are the same, i.e. F_1.2 = F_5. 
    # But Lr_1.2 = L_5 * (1.28/5) 
    if convert_Fr: 
        print("Converting Bahramian Lr values from 5 GHz to 1.28 GHz, assuming flat spectral index.")
        lr_all=lr_all*(1.28/5)

    if include_uplims:
        return lr_all, lx_all, source_class, uplims_lr, uplims_lx
    return lr_all, lx_all, source_class


##############################################################################################################
## RUNNER FUNCTIONS TO GET THE DATA FOR ALL THE SOURCES



def get_all_data():
    """
    Runner function to get a dataframe containing the data for all the sources.
    """
    
    ## Get the file paths for all the data files in the DATA folder
    folder_path = "../DATA"
    txt_files = glob.glob(f"{folder_path}/*.txt")
    print(f"Found {len(txt_files)} data files.")

    ## Initialise empty dataframes to hold all the data
    all_xray_df = pd.DataFrame()
    all_radio_df = pd.DataFrame()

    for i, path in enumerate(txt_files):

        p = Path(path)
        name = p.stem

        ## Get the Fr and Fx data, using the function defined above.
        # The X-ray data for MAXI J1631-479 already has the X-ray systematic error included, so we don't want to add this again. 
        # For the other sources, we add the systematic error as described in the paper.
        if name == "MAXI J1631-479":  source_metadata, obs_metadata, radio_data, xray_data  = read_data(path, add_sys_er=False, verbose=False) 
        else: source_metadata, obs_metadata, radio_data, xray_data  = read_data(path, add_sys_er=True, verbose=False)


        ## Calculate the luminosities and add them to the dataframes.
        # For the plots, we don't show the distance uncertainties (treat this as zero).
        d_kpc = source_metadata["D"][0]
        d_dist = source_metadata["D_prob"][0]
        print(f"{name}: Distance [kpc] = {d_kpc}, distribution [kpc] = {d_dist}")
        d_kpc_unc = 0 # kpc
        Lr, Lr_unc = convert_Fr(radio_data['Fr'].to_numpy(), radio_data['Fr_unc'].to_numpy(),d_kpc,d_kpc_unc, nu_GHz=1.28)
        radio_data['Lr'] = Lr
        radio_data['Lr_unc'] = Lr_unc
        Lx, Lx_unc_l, Lx_unc_u = convert_Fx(xray_data['Fx'].to_numpy(), xray_data['Fx_unc_l'].to_numpy(), xray_data['Fx_unc_u'].to_numpy(),d_kpc,d_kpc_unc)
        xray_data['Lx'] = Lx
        xray_data['Lx_unc_l'] = Lx_unc_l
        xray_data['Lx_unc_u'] = Lx_unc_u


        ## Add a column for the type of source
        type_source = source_metadata["class"][0]
        radio_data["class"] = type_source
        xray_data["class"] = type_source

        ## Add a column for the source distance
        d_kpc = source_metadata["D"][0]
        radio_data["D"] = d_kpc
        xray_data["D"] = d_kpc


        ## Add data for this source to the final dataframes (all sources combined)
        all_xray_df = pd.concat([all_xray_df, xray_data], ignore_index=True)
        all_radio_df = pd.concat([all_radio_df, radio_data], ignore_index=True)


    ## Remove rows where t_xray = Nan or t_radio = Nan
    all_xray_df = all_xray_df[~all_xray_df["t_xray"].isna()]
    all_radio_df = all_radio_df[~all_radio_df["t_radio"].isna()]


    ## Also calculate the median Lr and Lx for the HS/QS detections
    filtered_radio_df = all_radio_df[(all_radio_df["Fr_uplim_bool"] == False) & all_radio_df["Rstate"].isin(["HS", "QS"]) ]
    filtered_xray_df = all_xray_df[(all_xray_df["Fx_uplim_bool"] == False) & all_xray_df["Xstate"].isin(["HS", "QS"]) ]
    Lr_med = np.median(filtered_radio_df["Lr"])
    Lx_med = np.median(filtered_xray_df["Lx"])
    print(f"Median Lr (HS/QS detections): {Lr_med:.2e} erg/s")
    print(f"Median Lx (HS/QS detections): {Lx_med:.2e} erg/s")

    return all_xray_df, all_radio_df, Lr_med, Lx_med
    

##############################################################################################################