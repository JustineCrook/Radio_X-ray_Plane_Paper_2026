
## FUNCTIONS TO GENERATE RADIO:X-RAY PLANE

##############################################################################################################
## IMPORTS

from glob import glob
import pandas as pd
pd.set_option('display.max_rows', 1000)
pd.set_option('display.width', 200) 
import numpy as np
import os
import glob
from pathlib import Path


from get_data import convert_Fr, convert_Fx, read_data
from interpolation import interp_data_scipy_MC, manual_linear_interpolation

##############################################################################################################
### PAIRING ALGORITHMS


def pair_obs_alg(radio_df, xray_df, dt_mjd=1, weighted_ave = True, verbose=True):
    """
    For every radio observation, find the X-ray observations within dt_mjd of it. 
    Excludes upper limits if detections exist; otherwise, averages upper limits.
    If there are multiple detections, it averages them.
    Outputs a dataframe with paired data and a list of unpaired radio observation times. 
    """

    source_name = radio_df["name"].to_numpy()[0]

    xray_MJDs = xray_df["t_xray"].to_numpy() # this is ordered from smallest to largest
    Fx = xray_df["Fx"].to_numpy()
    Fx_unc_l = xray_df["Fx_unc_l"].to_numpy()
    Fx_unc_u = xray_df["Fx_unc_u"].to_numpy()
    uplims_xray = xray_df["Fx_uplim_bool"].to_numpy()
    xray_states = xray_df["Xstate"].to_numpy()
    
    radio_MJDs = radio_df["t_radio"].to_numpy()  # this is ordered from smallest to largest
    Fr = radio_df["Fr"].to_numpy()
    Fr_unc = radio_df["Fr_unc"].to_numpy()
    uplims_radio = radio_df["Fr_uplim_bool"].to_numpy()
    radio_states = radio_df["Rstate"].to_numpy()
    
    paired_data = []
    unpaired_radio_dates = []

    if verbose:
        print('{:<20s}{:<20s}{:<20s}{:<10s}{:<30s}{:<30s}{:<30s}{:<15s}{:<15s}{:<15s}'.format(
            "t_radio", "Fr [mJy]", "Fr_unc [mJy]", "#xray", "Mean Fx [erg/cm^2/s]", "Fx_unc_l[erg/cm^2/s]", "Fx_unc_u[erg/cm^2/s]", "Fr_uplim_bool", "Fx_uplim_bool", "state"))
    
    for i, t in enumerate(radio_MJDs):
        
        # Create masks for data within the current bin
        xray_mask = (xray_MJDs >= (t - dt_mjd)) & (xray_MJDs < (t + dt_mjd))
        
        # Extract relevant data
        xray_MJDs_all = xray_MJDs[xray_mask]
        xray_fluxes_all = Fx[xray_mask]
        xray_fluxes_unc_l_all = Fx_unc_l[xray_mask]
        xray_fluxes_unc_u_all = Fx_unc_u[xray_mask]
        xray_uplims_all = uplims_xray[xray_mask]
        xray_states_all = xray_states[xray_mask]

        if any(state != radio_states[i] for state in xray_states_all) and verbose:
            print("Warning: Some values in xray_states_all do not match radio_state. xray_states = ", xray_states_all, "; radio state: ", radio_states[i])
    
        # Filter fluxes based on upper limits
        if np.any(~xray_uplims_all): # if there are any xray detections
            # Remove the upper limits
            xray_fluxes, xray_fluxes_unc_l, xray_fluxes_unc_u = xray_fluxes_all[~xray_uplims_all], xray_fluxes_unc_l_all[~xray_uplims_all] , xray_fluxes_unc_u_all[~xray_uplims_all] 
            xray_uplim = False
        else: # only upper limits
            xray_fluxes, xray_fluxes_unc_l, xray_fluxes_unc_u = xray_fluxes_all, xray_fluxes_unc_l_all , xray_fluxes_unc_u_all
            xray_uplim = True
        nx = len(xray_fluxes)
        
        if weighted_ave==False: # normal average
            # Apply averaging logic
            Fx_av = np.mean(xray_fluxes) if xray_fluxes.size > 0 else np.nan
            # For the averaging below, I only use propagation of uncertainties. 
            # I don't include the uncertainty due to the spread in values, as this is assumed to be much smaller.
            Fx_unc_u_av = np.sqrt(np.sum(xray_fluxes_unc_u**2)) / len(xray_fluxes) if xray_fluxes.size > 0 else np.nan
            Fx_unc_l_av = np.sqrt(np.sum(xray_fluxes_unc_l**2)) / len(xray_fluxes) if xray_fluxes.size > 0 else np.nan

        else: # if I instead want the weighted average; note I just weight based on the uncertainty
            if xray_fluxes.size > 0:
                avg_unc = (xray_fluxes_unc_u + xray_fluxes_unc_l) / 2
                weights = 1 / (avg_unc**2)
                Fx_av = np.sum(weights * xray_fluxes) / np.sum(weights)
                weights_u = 1 / (xray_fluxes_unc_u**2)
                weights_l = 1 / (xray_fluxes_unc_l**2)
                Fx_unc_u_av = 1 / np.sqrt(np.sum(weights_u))
                Fx_unc_l_av = 1 / np.sqrt(np.sum(weights_l))
            else: Fx_av, Fx_unc_u_av , Fx_unc_l_av = np.nan, np.nan, np.nan
    
        
        if not np.isnan(Fx_av): # the radio observation was paired

            ## For t_diff, use the maximum difference between the radio and X-ray dates in the bin
            # Note that the code below works since the arrays are ordered
            max_dt_diff = max(
            abs(xray_MJDs_all[0] - t),
            abs(xray_MJDs_all[-1] - t)
            )

            paired_data.append({
                "name": source_name,
                "t": t,
                "t_diff": max_dt_diff,
                "Fr": Fr[i],
                "Fr_unc": Fr_unc[i],
                "Fr_uplim_bool": uplims_radio[i],
                "Fx": Fx_av,
                "Fx_unc_l": Fx_unc_l_av, 
                "Fx_unc_u": Fx_unc_u_av, 
                "Fx_uplim_bool": xray_uplim,
                "state": radio_states[i]})
            

            if verbose:
                print('{:<20.9f}{:<20.5f}{:<20.5f}{:<10d}{:<30.5e}{:<30.5e}{:<30.5e}{:<15s}{:<15s}{:<15s}'.format(
                    t, Fr[i], Fr_unc[i], nx, Fx_av, Fx_unc_l_av, Fx_unc_u_av, str(uplims_radio[i]), str(xray_uplim), str(radio_states[i])))
            
        else:
            unpaired_radio_dates.append(t)
    
    
    paired_data = pd.DataFrame(paired_data)
    unpaired_radio_dates = np.array(unpaired_radio_dates).reshape(-1,)

    return paired_data, unpaired_radio_dates





##############################################################################################################
## HELPERS TO PAIR RADIO AND XRAY DATA
# Get the Fr-Fx pairs, convert fluxes to luminosities, add source metadata, and output to file
# There are two options: 
# (1) Either pair using one of the pairing algorithms, using a window dt, adding some additional error for non-simultaneity
# (2) Interpolate the data to get simultaneous radio and X-ray data points 
# In the final paired dataframe, we store the paired Fr-Fx, and also the corresponding Lr-Lx using the best distance estimate
# I always interpolate the X-ray data onto the radio dates because:
# (1) We are a radio-led programme
# (2) There are in general more X-ray observations than radio ones (so easier to interpolate)

# The input to the functions are the checked (and sorted) dataframes
# Note the nu_GHz parameter is just for testing... If we input 1.28GHz data but put nu_GHz =5, then we are assuming a flat spectral index.



def make_paired_Lr_Lx_df(radio_data, xray_data, source_metadata, dt_days = 1,  d_kpc=None, nu_GHz=1.28, save=True, verbose=True):

    # Note at this stage, the dataframes are processed and sorted

    ## Run pairing algorithm 
    paired_data, unpaired_radio_MJDs = pair_obs_alg(radio_data, xray_data, dt_days, verbose=verbose)

    if len(paired_data)==0: return pd.DataFrame([])


    ## Add columns from source_metadata that will be relevant for filtering the plotting
    names = ["class",  "D", "D_prob"]
    for name in names:
        paired_data[name] = source_metadata[name].iloc[0] 

    ## Calculate the luminosities using the best distance estimates
    # We set the distance uncertainties to zero because this would blow up the uncertainties when visualising the plane. We deal with these uncertainties using a different approach.
    if d_kpc==None: d_kpc = source_metadata["D"][0]
    d_kpc_unc = 0 # kpc
    if verbose: print(f"Converting to luminosity using d_kpc = {d_kpc}")
    Lr, Lr_unc = convert_Fr(paired_data['Fr'].to_numpy(),paired_data['Fr_unc'].to_numpy(),d_kpc,d_kpc_unc, nu_GHz)
    paired_data['Lr'] = Lr
    paired_data['Lr_unc'] = Lr_unc
    Lx, Lx_unc_l, Lx_unc_u = convert_Fx(paired_data['Fx'].to_numpy(),paired_data['Fx_unc_l'].to_numpy(),paired_data['Fx_unc_u'].to_numpy(),d_kpc,d_kpc_unc)
    paired_data['Lx'] = Lx
    paired_data['Lx_unc_l'] = Lx_unc_l
    paired_data['Lx_unc_u'] = Lx_unc_u


    ## Add the distance distribution
    d_distribution = source_metadata["D_prob"][0]
    paired_data['D_prob'] = d_distribution 

    ## Output to a csv file
    if save:
        dir = "../DATA/PAIRED/"
        if not os.path.exists(dir):
            os.makedirs(dir)
        source_name = source_metadata["name"][0]
        paired_data.to_csv(dir+source_name+"_paired_Lr_Lx.csv", index=False)

    return paired_data



# If we opt to do the interpolation, then this is done instead of pairing -- i.e. the interpolation is done for all the data points, even if they could be paired
def make_interpolated_Lr_Lx_df(radio_df, xray_df, source_metadata, d_kpc=None, nu_GHz=1.28, interp_method = "akima", save=True, plotly=False, dt1=3.0, dt2=10.0 , verbose=True):

    # Note at this stage, the dataframes are processed and sorted
    # Get the data
    radio_dates = radio_df["t_radio"].to_numpy()
    xray_dates = xray_df ["t_xray"].to_numpy()
    xray_flux = xray_df ["Fx"].to_numpy()
    xray_flux_unc_l = xray_df ["Fx_unc_l"].to_numpy()
    xray_flux_unc_u = xray_df ["Fx_unc_u"].to_numpy()
    xray_uplims = xray_df ["Fx_uplim_bool"].to_numpy()
    source_name = source_metadata["name"][0]

    ## Interpolate the X-ray data onto the radio dates
    # Returns flux and uncertainty for every radio date, and NaN if invalid -- in linear space
    if interp_method=="akima": flux, flux_unc_l, flux_unc_u, flux_uplim_bool = interp_data_scipy_MC(xray_dates, xray_flux, xray_flux_unc_l, xray_flux_unc_u, xray_uplims, radio_dates, plotly=plotly, dt1=dt1, dt2=dt2, source_name=source_name, verbose=verbose, plot=verbose)
    elif interp_method=="linear": flux, flux_unc_l, flux_unc_u, flux_uplim_bool = manual_linear_interpolation(xray_dates, xray_flux, xray_flux_unc_l, xray_flux_unc_u, xray_uplims, radio_dates, verbose=verbose, plot=verbose)
    
    ## The algorithm above excluded "bad" interpolated points by putting NaN in this position
    mask = ~np.isnan(flux)
    n = sum(mask)
    if verbose: print("Number of used interpolated data points: ", n)
    MJDs = radio_dates[mask]
    interp_xray_data = pd.DataFrame({
        "t": MJDs, 
        "t_diff": np.zeros(n),    
        "Fx": flux[mask],                      
        "Fx_unc_l": flux_unc_l[mask],            
        "Fx_unc_u": flux_unc_u[mask],          
        "Fx_uplim_bool": flux_uplim_bool[mask]

    })

    ## Merge the interpolated X-ray data with the corresponding radio data
    matched_radio_data = radio_df[radio_df['t_radio'].isin(MJDs)]
    paired_data = interp_xray_data.merge(matched_radio_data, left_on="t", right_on="t_radio", how="left")
    paired_data.drop(columns=["t_radio"], inplace=True)
    paired_data["state"] = paired_data["Rstate"]
    paired_data.drop(columns=["Rstate"], inplace=True)

    ## Add columns from source_metadata that will be relevant for filtering the plotting
    names = ["class","D", "D_prob"]
    for name in names:
        paired_data[name] = source_metadata[name].iloc[0] 


    ## Calculate the luminosities using the best distance estimates
    # We set the distance uncertainties to zero because this would blow up the uncertainties when visualising the plane. We deal with these uncertainties using a different approach.
    if d_kpc==None: d_kpc = source_metadata["D"][0]
    d_kpc_unc = 0 # kpc
    if verbose: print(f"Converting to luminosity using d_kpc = {d_kpc}")
    Lr, Lr_unc = convert_Fr(paired_data['Fr'].to_numpy(),paired_data['Fr_unc'].to_numpy(),d_kpc,d_kpc_unc, nu_GHz)
    paired_data['Lr'] = Lr
    paired_data['Lr_unc'] = Lr_unc
    Lx, Lx_unc_l, Lx_unc_u = convert_Fx(paired_data['Fx'].to_numpy(),paired_data['Fx_unc_l'].to_numpy(),paired_data['Fx_unc_u'].to_numpy(),d_kpc,d_kpc_unc)
    paired_data['Lx'] = Lx
    paired_data['Lx_unc_l'] = Lx_unc_l
    paired_data['Lx_unc_u'] = Lx_unc_u


    ## Add the distance distribution
    d_distribution = source_metadata["D_prob"][0]
    paired_data['D_prob'] = d_distribution 

    ## Output to a csv file
    if save:
        dir="../DATA/INTERPOLATED/"
        if not os.path.exists(dir):
            os.makedirs(dir)
        source_name = source_metadata["name"][0]
        paired_data.to_csv(dir+source_name+"_Lr_Lx_interp.csv", index=False)

    return paired_data


##############################################################################################################
## RUNNER TO GET THE LRLX DATA FOR ALL SOURCES

# Use names if we do not want to include allt he sources
def get_all_LrLx_data(names = None, interp=False, rerun = True, save=False):

    if names is None:
        folder_path = "../DATA"
        txt_files = glob.glob(f"{folder_path}/*.txt")
        names = []
        for path in txt_files:
            p = Path(path)
            name = p.stem
            names.append(name)

    print("Source names: ", names)

    # Initialise to hold all data
    all_data = []

    for i, name in enumerate(names):
        

        if rerun: # rerun the pairing/interpolation
            source_df, obs_df, radio_df, xray_df = read_data(f"../DATA/{name}.txt", verbose=False)
            if interp==False: data = make_paired_Lr_Lx_df(radio_df, xray_df, source_df, save=True, verbose=False)
            else: data = make_interpolated_Lr_Lx_df(radio_df, xray_df, source_df, save = True, verbose=False)
   
        else: # use the saved pairing/interpolation results
            try:
                if interp: data = pd.read_csv(f'../DATA/INTERPOLATED/{name}_Lr_Lx_interp.csv')
                else: data = pd.read_csv(f'../DATA/PAIRED/{name}_paired_Lr_Lx.csv')
            except: 
                print(f"{name}: No saved LrLx data")
                data = pd.DataFrame([])

        all_data.append(data)
        all_data_df = pd.concat([df for df in all_data if not df.empty],ignore_index=True)


        if save: # Save the results for later use
            df_save = all_data_df.copy()

            # Rename the columns 
            df_save = df_save.rename(columns={
                'Fr_uplim_bool': 'Lr_uplim_bool',
                'Fx_uplim_bool': 'Lx_uplim_bool'
            })

            # Convert bool columns to integers (0 or 1)
            df_save ['Lr_uplim_bool'] = df_save ['Lr_uplim_bool'].astype(int)
            df_save ['Lx_uplim_bool'] = df_save ['Lx_uplim_bool'].astype(int)

            # Define the desired column order
            columns_to_write = [
                'name', 'class', 't', 'state',
                'Lr', 'Lr_unc', 'Lr_uplim_bool',
                'Lx', 'Lx_unc_l', 'Lx_unc_u', 'Lx_uplim_bool'
            ]

            # Write to a text file
            if interp: name = f'../DATA/INTERPOLATED/interpolated_lrlx.txt'
            else: name = f'../DATA/PAIRED/paired_lrlx.txt'
            with open(name, 'w') as f:
                f.write(','.join(columns_to_write) + '\n')
                for _, row in df_save[columns_to_write].iterrows():
                    row_str = ','.join(str(row[col]) for col in columns_to_write)
                    f.write(row_str + '\n')


    return all_data_df
