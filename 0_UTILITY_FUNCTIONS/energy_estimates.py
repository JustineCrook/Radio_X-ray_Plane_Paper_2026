import sys
import pandas as pd
import os
from scipy.interpolate import Akima1DInterpolator
from scipy.integrate import quad
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.append(os.path.abspath("../0_UTILITY_FUNCTIONS/"))
from get_data import *


"""
Code to compute rough estimates of the HS/QS compact jet energy. I employ different methods, which give similar results: 

METHOD 1:
- Run a trapezoidal integration of the HS/QS data on linear scale. 


METHOD 2:
- Run an akima interpolation of the HS/QS data on linear scale, and integrate using quad. 


METHOD 3:
- Run a trapezoidal integration of the HS/QS data on log scale. 


METHOD 4:
- Run an akima interpolation of the HS/QS data on log scale, and integrate using quad. 


METHOD 5:
- Run an akima interpolation of ALL the data on log scale, and integrate over JUST the HS/QS data using quad. 
"""


def augment_with_gap_zeros(t, L, t_gap, t_gap_first):
    """
    - Insert 0-flux points after/before large time gaps.
    - Forward pass: 
        - For each point, append the point
        - And if the next point is > t_gap away (or it's the last point) append (t + t_gap, 0.0).
    - Backward pass: 
        - Then, for the first point prepend (t_first - t_gap, 0.0)
        - And whenever a gap > t_gap is found between consecutive points, insert (t_current - t_gap, 0.0) before it.
    """

    t = np.asarray(t)
    L = np.asarray(L)
    if len(t) == 0: return np.array([]), np.array([])

    ## Sort data 
    order = np.argsort(t)
    t_sorted = t[order]
    L_sorted = L[order]

    ## Forward pass: append points and insert (t + t_gap, 0) after large gaps / last point
    t_fwd = []
    L_fwd = []
    n = len(t_sorted)
    for j in range(n):
        t_fwd.append(t_sorted[j])
        L_fwd.append(L_sorted[j])
        if j == n - 1 or (t_sorted[j + 1] - t_sorted[j] > t_gap): # last point or next point is > t_gap away
            t_fwd.append(t_sorted[j] + t_gap)
            L_fwd.append(0.0)

    ## Backward pass: sort the forward result (although it is sorted already) and insert zero before large gaps and at start
    t_fwd = np.asarray(t_fwd)
    L_fwd = np.asarray(L_fwd)
    order2 = np.argsort(t_fwd)
    t_sorted2 = t_fwd[order2]
    L_sorted2 = L_fwd[order2]

    t_final = []
    L_final = []
    for j in range(len(t_sorted2)):
        if j == 0: # First point
            t_final.append(t_sorted2[j] - t_gap_first) # insert right before this point
            L_final.append(0.0)
        elif (t_sorted2[j] - t_sorted2[j - 1]) > t_gap: # whenever gap > t_gap is found between consecutive points
            t_final.append(t_sorted2[j] - t_gap) # insert right before this point
            L_final.append(0.0)
        # Add the actual point
        t_final.append(t_sorted2[j])
        L_final.append(L_sorted2[j])

    return np.array(t_final), np.array(L_final)






def get_energy_estimates(all_radio_df, source_names=None, t_gap = 10, t_gap_first=10):
    """
    Function to compute energy estimates for all sources in the radio data frame.

    Parameters:
    -----------
    - all_radio_df is the radio data frame containing all sources
    - source_names is a list of source names to compute energy estimates for. If None, will compute for all sources in the data frame.
    - t_gap is the gap threshold in days
    - t_gap_first is the gap to insert before the first data point in days
    """

    ## Get the source names
    if source_names is None:
        folder_path = "../DATA"
        txt_files = glob.glob(f"{folder_path}/*.txt")
        source_names = []
        for txt_file in txt_files: 
            p = Path(txt_file)
            source_names.append(p.stem)


    ## Initialise totals
    total_E_linear_trapz = 0 # method 1
    total_E_linear_akima = 0 # method 2
    total_E_log_trapz = 0 # method 3
    total_E_log_akima = 0 # method 4
    total_E_log_akima_alt = 0 # method 5


    ## Loop through all the sources 
    for i, name in enumerate(source_names):
        print("\n", name)


        # ---------- Get the data for this source ----------

        radio_data = all_radio_df[all_radio_df["name"] == name]
        
        d_kpc = radio_data["D"].to_numpy()[0]
        source_class = radio_data["class"].to_numpy()[0]
        
        t = radio_data['t_radio'].to_numpy()
        state = radio_data['Rstate'].to_numpy()
        uplim_bool = radio_data['Fr_uplim_bool'].to_numpy()
        Fr = radio_data['Fr'].to_numpy()
        Fr_unc = radio_data['Fr_unc'].to_numpy()
        Lr_orig = radio_data['Lr'].to_numpy()
        Lr_all = Lr_orig.copy()
        Lr_unc_orig = radio_data['Lr_unc'].to_numpy()

        ## Get a lower limit luminosity value
        # This is what I use for "zero" when integrating on a log scale (since I can't take log of zero). I set it to the luminosity corresponding to 5 uJy, which is a typical upper limit value for non-detections in the radio. 
        # This is a conservative choice, as it means that any non-detection will be treated as having a very low luminosity rather than zero, which prevents the integrated energy from being artificially inflated by taking the log of zero.
        minL , _ = convert_Fr(0.005, 0, d_kpc, d_kpc_unc=0, nu_GHz=1.28)  # 5 uJy


        # Sort in case
        sort_idx = np.argsort(t)
        t_sorted = t[sort_idx]
        state_sorted = state[sort_idx]
        uplim_bool_sorted = uplim_bool[sort_idx]
        Fr_sorted = Fr[sort_idx]
        Fr_unc_sorted = Fr_unc[sort_idx]
        Lr_orig_sorted = Lr_orig[sort_idx]
        Lr_all_sorted = Lr_all[sort_idx]
        Lr_unc_orig_sorted = Lr_unc_orig[sort_idx]


        # If there is a gap of more than 100 days, split into separate sections (e.g. GX 339-4 2002-3 vs 2010-11 outbursts)
        # This is because low-level "zero" points can integrate to quite a lot of energy if they are very extended in time, and we don't want to artificially inflate the energy estimate by including long periods of non-detections between outbursts. 
        # By splitting into separate sections, we can apply the gap-augmentation method separately to each section, which prevents this issue.
        time_diffs = np.diff(t)
        gap_indices = np.where(time_diffs > 100)[0]

        # Split indices into segments
        split_points = gap_indices + 1  # +1 because diff is between i and i+1
        segments = np.split(np.arange(len(t)), split_points)


        ## Initialise energy estimates for this source (summing over all segments)
        E_linear_trapz_source = 0 # method 1
        E_linear_akima_source = 0 # method 2
        E_log_trapz_source = 0 # method 3
        E_log_akima_source = 0 # method 4
        E_log_akima_alt_source = 0 # method 5
        
      
        # Loop over each subset
        for i, seg in enumerate(segments):
            print(f"Subset {i+1}")


            # Get the data for this segment
            t = t_sorted[seg] 
            state = state_sorted[seg] 
            uplim_bool = uplim_bool_sorted[seg] 
            Fr = Fr_sorted[seg] 
            Fr_unc = Fr_unc_sorted[seg] 
            Lr_orig = Lr_orig_sorted[seg] 
            Lr_all = Lr_all_sorted[seg] 
            Lr_unc_orig = Lr_unc_orig_sorted[seg] 


            # ---------- Prepare the Data  ----------

            ## Filter for HS and QS states
            mask = np.isin(state, ["HS", "QS"])
            t_hsqs = t[mask]
            Lr_hsqs = Lr_orig[mask]
            uplim_bool_hsqs = uplim_bool[mask]
            if len(t_hsqs) == 0:
                print(f"No HS/QS data for {name}. Skipping...")
                continue

            ## Set uplims to 0 -- conservative
            Lr_hsqs[uplim_bool_hsqs] = 0.0 # just HS/QS data
            Lr_all[uplim_bool] = 0.0 # all data
            
            ## Sort the data (although it should already be sorted)
            sort_idx = np.argsort(t_hsqs)
            t_hsqs = t_hsqs[sort_idx]
            Lr_hsqs = Lr_hsqs[sort_idx]
            uplim_bool_hsqs = uplim_bool_hsqs[sort_idx]
            # Also sort the full data
            sort_idx = np.argsort(t)
            t = t[sort_idx]
            Lr_all = Lr_all[sort_idx]


            ## Forward and backward passed – insert 0-flux point before/after large gap (using t_gap)
            t_hsqs_aug, Lr_hsqs_aug = augment_with_gap_zeros(t_hsqs, Lr_hsqs, t_gap, t_gap_first) # just HS/QS data
            t_all_aug, Lr_all_aug   = augment_with_gap_zeros(t, Lr_all,   t_gap, t_gap_first) # all data
            t_hsqs_aug_sec = t_hsqs_aug * 86400  # convert days to seconds


            # ---------- Prepare Data for Log Interpolation ----------

            # Replace 0s with minL, for the HS/QS data set
            Lr_nozeros = np.where(Lr_hsqs_aug == 0, minL, Lr_hsqs_aug)
            logLr = np.log10(Lr_nozeros)

            # Replace 0s with minL, for the full data set
            Lr_all_nozeros = np.where(Lr_all_aug == 0, minL, Lr_all_aug)
            logLr_all = np.log10(Lr_all_nozeros)


            # ---------- Create Integration Windows for Method 5 ----------
            # I.e. HS/QS regions in time to integrate over 

            mask = (state == "HS") | (state == "QS")
            idxs = np.flatnonzero(mask)
            
            # Group consecutive indices into contiguous HS/QS segments
            splits = np.nonzero(np.diff(idxs) != 1)[0] + 1
            groups = np.split(idxs, splits)

            # Create extended windows (start - t_gap, end + t_gap)
            # Note, however, if the next/previous non-HS/QS point is closer than t_gap, only extend halfway to that point 
            windows = []
            n = len(t)
            for grp in groups:

                # First/last times for this HS/QS group
                t_first = float(t[grp[0]])
                t_last  = float(t[grp[-1]])

                # Find previous non-HS/QS point (to the left), if any
                prev_idx = grp[0] - 1
                prev_exists = (prev_idx >= 0)
                if prev_exists:
                    t_prev = float(t[prev_idx])
                    half_before = 0.5 * (t_first - t_prev) # half distance between first in group and previous non-HS/QS point
                    start_extend = min(t_gap, half_before)
                else: # no previous non-HS/QS point; use t_gap_first as fallback
                    if grp[0]==0: start_extend = t_gap_first  # if first point in data, use t_gap_first
                    else: start_extend = t_gap

                # Find next non-HS/QS point (to the right), if any 
                next_idx = grp[-1] + 1
                next_exists = (next_idx < n)
                if next_exists:
                    t_next = float(t[next_idx])
                    half_after = 0.5 * (t_next - t_last)
                    end_extend = min(t_gap, half_after)
                else: # no next non-HS/QS point; fall back to t_gap
                    end_extend = t_gap

                start = t_first - start_extend
                end   = t_last  + end_extend
                windows.append((start, end))

            # Merge overlapping/adjacent windows
            windows.sort(key=lambda x: x[0])
            merged = []
            cur_s, cur_e = windows[0]
            for s, e in windows[1:]:
                if s <= cur_e:   # overlap or touching -> merge
                    cur_e = max(cur_e, e)
                else:
                    merged.append((cur_s, cur_e))
                    cur_s, cur_e = s, e
            merged.append((cur_s, cur_e))

            # clip to interpolator domain
            tmin, tmax = t_all_aug.min(), t_all_aug.max()
            clipped = [(max(s, tmin), min(e, tmax)) for s, e in merged]
            clipped = [(s, e) for s, e in clipped if e > s]



            # ---------- Integration Method ----------

            # Ljet function
            def compute_Ljet(Lr, K):
                return K * ((Lr / 1.28 * 8.6 / 1e30) ** (12 / 17))
            # Get the normalisation K based on source class
            K =  1.3482006e+37 if source_class in ["BH", "candidateBH"] else 1.2976430775e+37


            ##########################################

            # ---------- Method 1 ----------
            # Trapezoidal integration on linear scale
            
            E_linear_trapz = 0
            for j in range(len(t_hsqs_aug) - 1):
                dt_sec = (t_hsqs_aug[j+1] - t_hsqs_aug[j]) * 86400 # convert dt to seconds
                Lr_avg = (Lr_hsqs_aug [j] + Lr_hsqs_aug[j+1]) / 2 # take middle Lr
                Ljet_avg = compute_Ljet(Lr_avg, K) # in erg/s
                E_linear_trapz += Ljet_avg * dt_sec # in erg


            # ---------- Method 2 ----------
            # Akima interpolation on linear scale using quad
            # Alternative to quad is usig np.trapz, but quad is more accurate for integration
            
            # Interpolate the HS/QS data
            akima_interp = Akima1DInterpolator(t_hsqs_aug, Lr_hsqs_aug)
            def Ljet_func_linear(t_):
                val = akima_interp(t_)
                val = np.clip(val, 0.0, None) # clip to avoid negative values due to spline overshooting
                return compute_Ljet(val, K) # in erg/s
            # Integrate using quad
            E_linear_akima, _ = quad(Ljet_func_linear, t_hsqs_aug[0], t_hsqs_aug[-1], limit=500, epsabs=1e-30, epsrel=1e-4)
            E_linear_akima *= 86400  # convert from days to seconds
            # Equivalent to:
            # akima_interp = Akima1DInterpolator(t_hsqs_aug_sec, Lr_hsqs_aug)
            # E_linear_akima, _ = quad(Ljet_func_linear, t_hsqs_aug_sec[0], t_hsqs_aug_sec[-1], limit=200, epsabs=1e-30, epsrel=1e-4)
        


            # ---------- Method 3 ----------
            # Trapezoidal integration on log scale
            """
            Note that below, for a particular point, we are doing (where alpha is a constant... alpha = K * (8.6 / 1.28 / 1e30)**(12/17) ):
            Ljet_avg = alpha * Lr_avg**(12/17)
            Ljet_avg 
            = 10 ** [(logLjet[j] + logLjet[j+1]) / 2] 
            =  10 ** [ ( np.log10(alpha * Lr_nozeros[j]**(12/17) ) + np.log10(alpha * Lr_nozeros[j + 1]**(12/17) ) ) / 2]
            = 10** [0.5 * np.log10( alpha**2 * Lr_nozeros[j]**(12/17)  * Lr_nozeros[j + 1]**(12/17) ) ] 
            = 10** np.log10[ (alpha**2 * Lr_nozeros[j]**(12/17)  * Lr_nozeros[j + 1]**(12/17))**0.5 ]
            =  (alpha**2 * Lr_nozeros[j]**(12/17)  * Lr_nozeros[j + 1]**(12/17))**0.5 
            = alpha * [(Lr_nozeros[j] * Lr_nozeros[j + 1])**0.5]**(12/17)

            This is equivalent to first taking the average of Lr in log space and then calculating Ljet:
            Ljet_avg 
            = alpha * [10**(0.5* (logLr[j] + logLr[j+1]) )]**(12/17) 
            = alpha * [10**(0.5* ( np.log10(Lr_nozeros[j]) + np.log10(Lr_nozeros[j+1]) ) )]**(12/17)
            =  alpha * [10**( np.log10[(Lr_nozeros[j] * Lr_nozeros[j+1])**0.5 ] )]**(12/17) 
            =  alpha * [ (Lr_nozeros[j] * Lr_nozeros[j+1])**0.5 ]**(12/17)

            So instead of (Lr_nozeros[j] + Lr_nozeros[j + 1])*0.5 as we do in method 1, we are doing (Lr_nozeros[j] * Lr_nozeros[j + 1])**0.5
            """
            logLjet = np.log10(compute_Ljet(Lr_nozeros, K))
            E_log_trapz = 0
            for j in range(len(t_hsqs_aug) - 1):
                dt_sec = (t_hsqs_aug[j+1] - t_hsqs_aug[j]) * 86400
                logL_avg = (logLjet[j] + logLjet[j+1]) / 2
                Ljet_avg = 10 ** logL_avg
                E_log_trapz += Ljet_avg * dt_sec



            # ---------- Method 4 ----------
            # Akima interpolation on log scale using quad
            
            # Interpolate the HS/QS data on log scale
            akima_log_interp = Akima1DInterpolator(t_hsqs_aug, logLr)
            def Ljet_func_log(t_):
                logLr_val = akima_log_interp(t_) # get logLr
                Lr_val = 10 ** logLr_val # convert back to linear scale to get Lr
                return compute_Ljet(Lr_val, K) # erg/s
            # Integrate using quad
            E_log_akima, _ = quad(Ljet_func_log, t_hsqs_aug[0], t_hsqs_aug[-1], limit=500, epsabs=1e-30, epsrel=1e-4)
            E_log_akima *= 86400  # convert from days to seconds


            # ---------- Method 5 ----------
            # Interpolate the whole curve on log scale, but only add the contribution from the HS/QS ranges. 
            
            # Interpolate the full data on log scale
            akima_log_interp_all = Akima1DInterpolator(t_all_aug, logLr_all)
            def Ljet_func_log_all(t_):
                logLr_val = akima_log_interp_all(t_)   # use the ALL-data interpolator here
                Lr_val = 10 ** logLr_val
                return compute_Ljet(Lr_val, K)
            # Integrate over clipped windows
            E_log_akima_alt = 0.0
            for s, e in clipped:
                val, err = quad(Ljet_func_log_all, s, e, limit=500, epsabs=1e-30, epsrel=1e-4)
                E_log_akima_alt += val
            E_log_akima_alt *= 86400.0 # convert to seconds


            ##########################################

            # ---------- Print Results ----------
            
            # Print individual source results
            print(f"\n{name} [{source_class}] Energy Estimates:")
            print(f"  Linear Trapezoidal (method 1):         {E_linear_trapz:.3e} erg")
            print(f"  Linear Akima Interpolation (method 2): {E_linear_akima:.3e} erg")
            print(f"  Log Trapezoidal (method 3):            {E_log_trapz:.3e} erg")
            print(f"  Log Akima Interpolation (method 4):    {E_log_akima:.3e} erg")
            print(f"  Log Akima Interpolation Alt (method 5): {E_log_akima_alt:.3e} erg")


            # Accumulate total for the source
            E_linear_trapz_source += E_linear_trapz
            E_linear_akima_source += E_linear_akima
            E_log_trapz_source += E_log_trapz
            E_log_akima_source += E_log_akima
            E_log_akima_alt_source += E_log_akima_alt


            # Accumulate totals for all sources
            total_E_linear_trapz += E_linear_trapz
            total_E_linear_akima += E_linear_akima
            total_E_log_trapz += E_log_trapz
            total_E_log_akima += E_log_akima
            total_E_log_akima_alt += E_log_akima_alt


            # ---------- Prepare Plotting ----------

            xrange  = [np.min(t_all_aug) - 2, np.max(t_all_aug) + 2] 
            t_dense = np.linspace(t_hsqs_aug[0], t_hsqs_aug[-1], 1000)
            t_dense_all = np.linspace(t_all_aug[0], t_all_aug[-1], 1000)
            t_dense_sec = t_dense * 86400  # convert days to seconds

            # For plotting, get the original HS/QS data
            is_hsqs = np.isin(state, ["HS", "QS"])
            t_hsqs_orig = t[is_hsqs]
            Lr_hsqs_orig = Lr_orig[is_hsqs]
            uplim_bool_hsqs = uplim_bool[is_hsqs]

            
            # ---------- Plot Method 1 ----------
            plt.figure(figsize=(18, 7))
            plt.title(f"{name} -- Trapezoidal Integration on Linear Scale (Method 1)")

            # Plot original data
            plt.plot(t[~uplim_bool], Lr_orig[~uplim_bool], 'o', color='lightblue', label='Other States')
            plt.plot(t[uplim_bool], Lr_orig[uplim_bool], 'v', color='lightblue')
            plt.plot(t_hsqs_orig[~uplim_bool_hsqs], Lr_hsqs_orig[~uplim_bool_hsqs], 'o', color='darkblue', label='HS/QS (Original)')
            plt.plot(t_hsqs_orig[uplim_bool_hsqs], Lr_hsqs_orig[uplim_bool_hsqs], 'v', color='darkblue')

            # Plot altered HS/QS data -- i.e. with gap-augmentation
            plt.plot(t_hsqs_aug, Lr_hsqs_aug , 'o--', color='red', label='HS/QS (Altered)')

            plt.xlim(xrange)
            plt.xlabel("Time [MJD]")
            plt.ylabel("Radio Luminosity [erg/s]")
            plt.legend()
            plt.tight_layout()
            plt.show()


            # ---------- Plot Method 2 ----------
            plt.figure(figsize=(18, 7))
            plt.title(f"{name} – Quad Integration on Linear Scale (Method 2)")

            # Plot original data
            plt.plot(t[~uplim_bool], Lr_orig[~uplim_bool], 'o', color='lightblue', label='Other States')
            plt.plot(t[uplim_bool], Lr_orig[uplim_bool], 'v', color='lightblue')
            plt.plot(t_hsqs_orig[~uplim_bool_hsqs], Lr_hsqs_orig[~uplim_bool_hsqs], 'o', color='darkblue', label='HS/QS (Original)')
            plt.plot(t_hsqs_orig[uplim_bool_hsqs], Lr_hsqs_orig[uplim_bool_hsqs], 'v', color='darkblue')

            # Plot altered HS/QS data -- i.e. with gap-augmentation
            plt.plot(t_hsqs_aug, Lr_hsqs_aug, 'o', color='red', label='HS/QS (Altered)')
            
            # Plot interpolated results
            Lr_dense = akima_interp(t_dense)
            plt.plot(t_dense, Lr_dense, '-', label="Akima Interpolation (method 2)", color='purple')

            # Plot interpolated results method 4 for comparison -- i.e. when interpolation is on a log scale instead
            logLr_dense = akima_log_interp(t_dense)
            Lr_dense = 10 ** logLr_dense
            plt.plot(t_dense, Lr_dense, '-', label="Akima Interpolation (method 4)", color='pink')
            
            plt.xlim(xrange)
            plt.xlabel("Time [MJD]")
            plt.ylabel("Radio Luminosity [erg/s]")
            plt.legend()
            plt.tight_layout()
            plt.show()



            # ---------- Plot Method 3 ----------
            plt.figure(figsize=(18, 7))
            plt.title(f"{name} – Trapezoidal Integration on Log Scale (Method 3)")

            # Plot original data
            plt.plot(t[~uplim_bool], Lr_orig[~uplim_bool], 'o', color='lightblue', label='Other States')
            plt.plot(t[uplim_bool], Lr_orig[uplim_bool], 'v', color='lightblue')
            plt.plot(t_hsqs_orig[~uplim_bool_hsqs], Lr_hsqs_orig[~uplim_bool_hsqs], 'o', color='darkblue', label='HS/QS (Original)')
            plt.plot(t_hsqs_orig[uplim_bool_hsqs], Lr_hsqs_orig[uplim_bool_hsqs], 'v', color='darkblue')

            # Plot altered HS/QS data -- i.e. with gap-augmentation
            plt.plot(t_hsqs_aug, Lr_nozeros, 'o--', color='red', label='HS/QS (Altered)')
            plt.xlim(xrange)
            plt.xlabel("Time [MJD]")
            plt.ylabel("Radio Luminosity [erg/s]")
            plt.yscale('log') # log scale
            plt.legend()
            plt.tight_layout()
            plt.show()



            # ---------- Plot Method 4 ----------
            plt.figure(figsize=(18, 7))
            plt.title(f"{name} – Quad Integration on Log Scale (Method 4)")
            
            # Plot original data
            plt.plot(t[~uplim_bool], np.log10(Lr_orig[~uplim_bool]), 'o', color='lightblue', label='Other States')
            plt.plot(t[uplim_bool], np.log10(Lr_orig[uplim_bool]), 'v', color='lightblue')
            plt.plot(t_hsqs_orig[~uplim_bool_hsqs], np.log10(Lr_hsqs_orig[~uplim_bool_hsqs]), 'o', color='darkblue', label='HS/QS (Original)')
            plt.plot(t_hsqs_orig[uplim_bool_hsqs], np.log10(Lr_hsqs_orig[uplim_bool_hsqs]), 'v', color='darkblue')
            
            # Plot augmented HS/QS data -- i.e. with gap-augmentation
            plt.plot(t_hsqs_aug, np.log10(Lr_nozeros), 'o', color='red', label='HS/QS (Altered)')

            # Plot interpolated results
            logLr_dense = akima_log_interp(t_dense)
            plt.plot(t_dense, logLr_dense, '-', label="Akima Interpolation (method 4)", color='purple')
            
            plt.xlim(xrange)
            plt.xlabel("Time [MJD]")
            plt.ylabel("Radio Luminosity [erg/s]")
            plt.legend()
            plt.tight_layout()
            plt.show()


            # ---------- Plot Method 4, on a linear scale instead of log scale ----------
            plt.figure(figsize=(18, 7))
            plt.title(f"{name} – Quad Integration on Log Scale (Method 4), Transformed Back to Linear Scale")
            
            # Plot original data
            plt.plot(t[~uplim_bool], Lr_orig[~uplim_bool], 'o', color='lightblue', label='Other States')
            plt.plot(t[uplim_bool], Lr_orig[uplim_bool], 'v', color='lightblue')
            plt.plot(t_hsqs_orig[~uplim_bool_hsqs], Lr_hsqs_orig[~uplim_bool_hsqs], 'o', color='darkblue', label='HS/QS (Original)')
            plt.plot(t_hsqs_orig[uplim_bool_hsqs], Lr_hsqs_orig[uplim_bool_hsqs], 'v', color='darkblue')
            
            # Plot augmented HS/QS data -- i.e. with gap-augmentation
            plt.plot(t_hsqs_aug, Lr_nozeros, 'o', color='red', label='HS/QS (Altered)')

            # Plot interpolated results
            logLr_dense = akima_log_interp(t_dense)
            Lr_dense = 10 ** logLr_dense
            plt.plot(t_dense, Lr_dense, '-', label="Akima Interpolation (method 4)", color='purple')
            
            #plt.yscale('log')
            plt.xlim(xrange)       
            plt.xlabel("Time [MJD]")
            plt.ylabel("Radio Luminosity [erg/s]")
            plt.legend()
            plt.tight_layout()
            plt.show()



            # ---------- Plot Method 5 ----------
            fig, ax = plt.subplots(figsize=(18, 7))
            plt.title(f"{name} – Quad Integration (alt method) on Log Scale (Method 5)")
            
            # Plot original data
            plt.plot(t[~uplim_bool], np.log10(Lr_orig[~uplim_bool]), 'o', color='lightblue', label='All Data')
            plt.plot(t[uplim_bool], np.log10(Lr_orig[uplim_bool]), 'v', color='lightblue')
            plt.plot(t_hsqs_orig[~uplim_bool_hsqs], np.log10(Lr_hsqs_orig[~uplim_bool_hsqs]), 'o', color='darkblue', label='HS/QS (Original)')
            plt.plot(t_hsqs_orig[uplim_bool_hsqs], np.log10(Lr_hsqs_orig[uplim_bool_hsqs]), 'v', color='darkblue')
            
            # Plot augmented HS/QS data
            plt.plot(t_all_aug, np.log10(Lr_all_nozeros), 'o', color='red', label='HS/QS (Altered)')


            # Plot interpolated results
            logLr_dense = akima_log_interp_all(t_dense_all)
            plt.plot(t_dense_all, logLr_dense, '-', label="Akima Interpolation (Method 5)", color='purple')

            # Plot the integration windows
            # clipped windows used for integration
            integration_windows = clipped   # list of (t_start, t_end)
            for i, (t0, t1) in enumerate(integration_windows):
                ax.axvspan(
                    t0, t1,
                    color="orange",
                    alpha=0.15,
                    zorder=0,
                    label="HS/QS Integration Region" if i == 0 else None
                )
                
            ax.set_xlim(xrange)
            plt.xlabel("Time [MJD]")
            plt.ylabel("Radio Luminosity [erg/s]")
            plt.legend()
            plt.tight_layout()
            plt.show()


        
        # End of loop over subsets for this source
        print(f"Total for {name}:")
        print(f"  Linear Trapezoidal:         {E_linear_trapz_source:.3e} erg")
        print(f"  Linear Akima Interpolation: {E_linear_akima_source:.3e} erg")
        print(f"  Log Trapezoidal:            {E_log_trapz_source:.3e} erg")
        print(f"  Log Akima Interpolation:    {E_log_akima_source:.3e} erg")
        print(f"  Log Akima Interpolation (Alt): {E_log_akima_alt_source:.3e} erg")
        
        print("\n----------------------------------------\n")
        print("----------------------------------------\n")
        print("\n----------------------------------------\n")



    # ---------- TOTAL ENERGY ESTIMATES ----------

    # Final printout of totals
    print("\n--- TOTAL ENERGY ESTIMATES (All Sources) ---")
    print(f"Total Linear Trapezoidal:         {total_E_linear_trapz:.3e} erg")
    print(f"Total Linear Akima Interpolation: {total_E_linear_akima:.3e} erg")
    print(f"Total Log Trapezoidal:            {total_E_log_trapz:.3e} erg")
    print(f"Total Log Akima Interpolation:    {total_E_log_akima:.3e} erg")
    print(f"Total Log Akima Interpolation (Alt): {total_E_log_akima_alt:.3e} erg")