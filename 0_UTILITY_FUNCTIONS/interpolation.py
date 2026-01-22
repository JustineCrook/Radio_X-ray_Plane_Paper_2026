## INTERPOLATION FUNCTIONS FOR RADIO:X-RAY PLANE GENERATIONS

##############################################################################################################

## IMPORTS
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline, PchipInterpolator, Akima1DInterpolator
from scipy.interpolate import interp1d
import warnings

import plotly.graph_objects as go



"""
Weighted averages:
- https://www.amherst.edu/system/files/media/1871/weighted%20average.pdf
- https://seismo.berkeley.edu/~kirchner/Toolkits/Toolkit_12.pdf .... extra errors due to scatter
"""



##############################################################################################################
## HELPER FUNCTIONS FOR SCIPY INTERPOLATION



# When points are extremely close in x value, then the interpolation fails
# In this case, average these points
# Threshold is in days
def merge_close_points(x, y, y_unc_l, y_unc_u, uplim, threshold=0.05, weighted_average=True):
    """
    Merge consecutive points in time (x) that are within a given threshold.
    If upper limits (uplim) and detections are mixed, only keep detections.
    Uncertainties are averaged or propagated depending on `weighted_average`.
    """
    
    x, y, y_unc_l, y_unc_u, uplim = map(np.asarray, (x, y, y_unc_l, y_unc_u, uplim))
    
    merged_x, merged_y  = [], []
    merged_y_unc_l, merged_y_unc_u = [], []
    merged_uplim = []

    i = 0
    while i < len(x):
        # Start new group
        group_x = [x[i]]
        group_y = [y[i]]
        group_y_unc_l = [y_unc_l[i]]
        group_y_unc_u = [y_unc_u[i]]
        group_uplim = [uplim[i]]

        # Gather consecutive points within the threshold
        while i + 1 < len(x) and (x[i+1] - x[i]) <= threshold:
            i += 1
            group_x.append(x[i])
            group_y.append(y[i])
            group_y_unc_l.append(y_unc_l[i])
            group_y_unc_u.append(y_unc_u[i])
            group_uplim.append(uplim[i])

        # Filter out upper limits, if both detections and limits exist in the group
        if True in group_uplim and False in group_uplim:
            keep_indices = [j for j, is_uplim in enumerate(group_uplim) if not is_uplim]
            group_x = [group_x[j] for j in keep_indices]
            group_y = [group_y[j] for j in keep_indices]
            group_y_unc_l = [group_y_unc_l[j] for j in keep_indices]
            group_y_unc_u = [group_y_unc_u[j] for j in keep_indices]
            group_uplim = [False]  

        # Compute mean only if more than one point remains 
        if len(group_x) > 1:

            group_x = np.array(group_x)
            group_y = np.array(group_y)
            group_y_unc_l = np.array(group_y_unc_l)
            group_y_unc_u = np.array(group_y_unc_u)

            if weighted_average:
                avg_unc = (group_y_unc_l + group_y_unc_u) / 2
                weights = 1 / (avg_unc**2)
                merged_y.append( np.sum(weights * group_y) / np.sum(weights))
                # Use the same weighting for x -- i.e. bias it towards the x of the more reliable y values 
                merged_x.append(np.sum(weights * group_x) / np.sum(weights))
                # Get upper and lower uncertainties separately
                weights_u = 1 / (group_y_unc_u**2)
                weights_l = 1 / (group_y_unc_l**2)
                merged_y_unc_u.append( 1 / np.sqrt(np.sum(weights_u)) )
                merged_y_unc_l.append( 1 / np.sqrt(np.sum(weights_l)) )

            else:
                merged_x.append(np.mean(group_x))
                merged_y.append(np.mean(group_y)) 
                merged_y_unc_l.append(np.sqrt(np.sum(np.array(group_y_unc_l) ** 2)) / len(group_y_unc_l))
                merged_y_unc_u.append(np.sqrt(np.sum(np.array(group_y_unc_u) ** 2)) / len(group_y_unc_u))
        
        else: 
            # Single point -- no merging needed
            merged_x.append(group_x[0])
            merged_y.append(group_y[0])
            merged_y_unc_l.append(group_y_unc_l[0])
            merged_y_unc_u.append(group_y_unc_u[0])

        # Set uplim: If all True, keep True; otherwise, it's already False from above
        merged_uplim.append(all(group_uplim))
        i += 1

    return (np.array(merged_x), np.array(merged_y), np.array(merged_y_unc_l), np.array(merged_y_unc_u), np.array(merged_uplim))


##############################################################################################################
## RUNNER FUNCTION FOR MANUAL LINEAR INTERPOLATION -- IN LOG SPACE


"""
Note that when propagating uncertainties, we ignore the uncertainty of the time variable. 

My formula y = y1 + (x - x1) * (y2 - y1) / (x2 - x1) is equivalent to:
y = y1 - y1*(x-x1)/(x2-x1) + y2*(x-x1)/(x2-x1)
y = y1 * [1- (x-x1)/(x2-x1)] + y2*(x-x1)/(x2-x1)
y = y1 * [(x2 -x1 -x +x1)/(x2-x1)] + y2*(x-x1)/(x2-x1)
y = y1 * [(x2 -x)/(x2-x1)] + y2*(x-x1)/(x2-x1)
y = y1 * [(x-x2)/(x1-x2)] + y2*(x-x1)/(x2-x1)
... which is the same as Andrew's formula from White 2017 eqn (14).

My propagation of errors formula:
dy^2 = (1 - (x - x1)/(x2 - x1))**2 * dy1**2 + ((x - x1)/(x2 - x1))**2 *dy2**2
dy^2 =  [ (x1 - x2 + (x - x1) ) / (x1 - x2)]**2 * dy1**2 + ((x - x1)/(x2 - x1))**2 *dy2**2
dy^2 =  ((x - x2)/(x1 - x2)**2 * dy1**2 + ((x - x1)/(x2 - x1))**2 *dy2**2
... whichis the same as White 2017 eqn (15).
"""
##TODO: Deal with case when there is only one data point.
def manual_linear_interpolation(xray_dates, xray_flux, xray_flux_unc_l, xray_flux_unc_u, xray_uplims, radio_dates, verbose=True, plot=True):


    # If some point are extremely close, then it messes up the interpolation (since it thinks this is a steep rise/decay). 
    # Simple solution: average these X-ray data points.
    xray_dates, xray_flux, xray_flux_unc_l, xray_flux_unc_u, xray_uplims = merge_close_points(xray_dates, xray_flux, xray_flux_unc_l, xray_flux_unc_u, xray_uplims, threshold=0.1)


    # x-axis values
    x = xray_dates
    # x data for plotting:
    x_plot = np.linspace( x[0] , x[-1] , 2*int(1*(x[-1] - x[0])) ) 
    if verbose: print("Number of plotting points: ", 2*int(1*(x[-1] - x[0])) )
    # x data for prediction 
    x_predict = radio_dates

    # Get logged values
    y = np.log10(xray_flux) 
    yerr_l = np.log10(xray_flux) - np.log10(xray_flux - xray_flux_unc_l)
    yerr_u = np.log10(xray_flux + xray_flux_unc_u) - np.log10(xray_flux)

    # Get error for each data point
    #y_err = 0.5*(yerr_l  + yerr_u)
    y_err = np.maximum(yerr_l, yerr_u) # maximum for corresponding elements <<< conservative approach

    
    ## Conduct filtering of predictions, and implement uncertainty scheme based on distance to nearest X-ray data point
    y_predict_uplim_bool = np.zeros(len( x_predict ), dtype=bool)
    y_predict = np.zeros(len( x_predict ))
    y_predict_err_l = np.zeros(len( x_predict ))
    y_predict_err_u = np.zeros(len( x_predict ))


    for i, radio_date in enumerate(x_predict):

        if np.isnan(radio_date): continue

        # Find indices of X-ray observations before and after the radio observation
        before_idxs = np.where(xray_dates < radio_date)[0]
        after_idxs = np.where(xray_dates > radio_date)[0]


        if len(before_idxs) != 0 :
            before_idx = before_idxs[-1]  # Last X-ray point before radio
            t_before = xray_dates[before_idx]
            f_before = y[before_idx]
            f_unc_before= y_err[before_idx]
            f_unc_before_l = yerr_l[before_idx]
            f_unc_before_u = yerr_u[before_idx]
            before_uplim = xray_uplims[before_idx] # whether this X-ray data point is an uplim
    
        
        if len(after_idxs) != 0: 
            after_idx = after_idxs[0]     # First X-ray point after radio
            t_after = xray_dates[after_idx]
            f_after = y[after_idx]
            f_unc_after = y_err[after_idx]
            f_unc_after_l = yerr_l[after_idx]
            f_unc_after_u = yerr_u[after_idx]
            after_uplim = xray_uplims[after_idx] # whether this X-ray data point is an uplim


        if len(before_idxs) != 0 and len(after_idxs) !=0 :
            ## Do interpolation
            
            # y = y1 + (x - x1) * (y2 - y1) / (x2 - x1)
            alpha = (radio_date - t_before) / (t_after - t_before) # (x - x1)/(x2 - x1)
            log_interp_flux = f_before + alpha * (f_after - f_before)
            y_predict[i] = log_interp_flux
            
            # Using propagation of uncertainties:
            # dy = np.sqrt( (1 - (x - x1)/(x2 - x1))**2 * dy1**2 + ((x - x1)/(x2 - x1))**2 *dy2**2 )
            #log_interp_flux_unc = np.sqrt((1 - alpha)**2 * f_unc_before**2 + alpha**2 * f_unc_after**2)
            #y_predict_err_l[i], y_predict_err_u[i] = log_interp_flux_unc, log_interp_flux_unc
            log_interp_flux_unc_l = np.sqrt((1 - alpha)**2 * f_unc_before_l**2 + alpha**2 * f_unc_after_l**2)
            log_interp_flux_unc_u = np.sqrt((1 - alpha)**2 * f_unc_before_u**2 + alpha**2 * f_unc_after_u**2)
            y_predict_err_l[i], y_predict_err_u[i] = log_interp_flux_unc_l, log_interp_flux_unc_u



        # linear interp doesn't do extrapolation
        # If radio date is less than a day from the nearest point, use that value
        if len(before_idxs) == 0:
            t_before =0
            y_predict[i] = f_after
            y_predict_err_l[i] = f_unc_after
            y_predict_err_u[i] = f_unc_after
            
        if len(after_idxs) ==0 :
            t_after = np.inf
            y_predict[i] = f_before  
            y_predict_err_l[i] = f_unc_before
            y_predict_err_u[i] = f_unc_before




        # Time to nearest X-ray point, in days
        time_to_nearest = min(
            abs(radio_date - t_before),
            abs(radio_date - t_after)
        )  

        time_to_furthest = max(
            abs(radio_date - t_before),
            abs(radio_date - t_after)
        ) 
        

        # Reject the predictions that are too far away -- i.e. make them NaN
        # This will include the extrapolated points further than 1.5 day away
        if (time_to_nearest > 1.0 and time_to_furthest > 15) or time_to_nearest> 10:  
            if verbose: print(f"Rejected based on distance: radio_MJD = {radio_date}; time_to_nearest: {time_to_nearest}; time_to_furthest: {time_to_furthest}" )
            y_predict[i] = np.nan
            y_predict_err_l[i] = np.nan 
            y_predict_err_u[i] = np.nan
            continue

        # If the closest data point is an uplim, then treat the interpolated point as an uplim
        # Uplim error: dy = log10(x) - log10(x-dx) = log10(3/2) = 0.1760912591
        if abs(radio_date - t_before) < abs(radio_date - t_after): # closest data point is the one before
            if before_uplim == True: 
                if verbose: print("Upper limit")
                y_predict_uplim_bool[i] = True
                y_predict_err_l[i] =  np.log10(3/2) # 1 sigma in log space
                y_predict_err_u[i] = 0
                continue
            
        else: # closest point is the one after
            if after_uplim == True:
                if verbose: print("Upper limit")
                y_predict_uplim_bool[i] = True
                y_predict_err_l[i] = np.log10(3/2)
                y_predict_err_u[i] = 0
                continue
        
        if verbose: print(f"{radio_date}: time_to_nearest: {time_to_nearest} & time_to_furthest: {time_to_furthest}; y_predict_err: {y_predict_err_l[i]}")

    if verbose:
        print()
        print("LOG ERRORS:")
        print(y_predict_err_l)
        print()



    # Convert to linear space 
    y_predict_linear = 10**y_predict
    upper_linear_errors = 10**(y_predict + y_predict_err_u) - y_predict_linear
    lower_linear_errors = y_predict_linear - 10**(y_predict - y_predict_err_l) 

    if plot:

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9))
        
        # Plot the actual data
        ax1.errorbar(x, y, yerr=y_err, uplims = xray_uplims, label='Data',  color="red")
        ax2.errorbar(x, xray_flux, yerr=[xray_flux_unc_l, xray_flux_unc_u], uplims= xray_uplims, label='Data', color="red")
        ax1.set_ylabel("Log flux")
        ax2.set_ylabel("Flux")

        # Plot radio MJDs:
        for i, date in enumerate(radio_dates):
            if i==0: 
                ax1.axvline(date, linestyle='-', color='black', alpha=0.1, label="Radio dates")
                ax2.axvline(date, linestyle='-', color='black', alpha=0.1, label="Radio dates")
            else: 
                ax1.axvline(date, linestyle='-', color='black', alpha=0.1)
                ax2.axvline(date, linestyle='-', color='black', alpha=0.1)


        # Plot the interpolated points
        ax1.errorbar(x_predict[~y_predict_uplim_bool], y_predict[~y_predict_uplim_bool], yerr= y_predict_err_l[~y_predict_uplim_bool], alpha=0.5, fmt='.', color='black', label='Interpolated at radio times')
        ax1.scatter(x_predict[y_predict_uplim_bool], y_predict[y_predict_uplim_bool], marker='v', color="black")
        
        # Back to linear space
        ax2.errorbar(x_predict[~y_predict_uplim_bool], y_predict_linear [~y_predict_uplim_bool], yerr= [lower_linear_errors[~y_predict_uplim_bool], upper_linear_errors[~y_predict_uplim_bool]], alpha=0.5, fmt='.', color='black', label='Interpolated at radio times')
        ax2.scatter(x_predict[y_predict_uplim_bool], y_predict_linear [y_predict_uplim_bool], marker='v', color="black")

        ax1.set_xlabel("MJD")
        ax2.set_xlabel("MJD")

        handles, labels = ax2.get_legend_handles_labels()
        order = [labels.index("Data")] + [i for i, lbl in enumerate(labels) if lbl != "Data"]
        ax2.legend([handles[i] for i in order], [labels[i] for i in order], fontsize=10)
        plt.tight_layout()
        plt.show()


    return y_predict_linear, lower_linear_errors, upper_linear_errors, y_predict_uplim_bool


##############################################################################################################
## RUNNER FUNCTION FOR SCIPY INTERPOLATION



# I interpolate X-ray points using akima on a log scale
# I include uplims, but if the interpolate point is closest to an uplim, then it is also considered an uplim
# Note that akima cannot extrapolate
# Run MC repeats to get the error on the interpolated data points, since we cannot propagate uncertainties in this scheme
# Take the max of this and an uncertainty based on distance to the nearest X-ray point

def interp_data_scipy_MC(xray_dates, xray_flux, xray_flux_unc_l, xray_flux_unc_u, xray_uplims, radio_dates, testing=False, dt1 = 3.0, dt2 = 10.0, plotly = False, source_name="", verbose=True, plot=True): # dt1 and dt were previously 10,15

    if verbose: 
        print("dt1, dt2: ", dt1, dt2)
        print()

    # For reproducibility
    np.random.seed(42)


    if len(xray_dates) == 1:
        xray_date = xray_dates[0]
        idx_closest = np.argmin(np.abs(radio_dates - xray_date))
        radio_date = radio_dates[idx_closest]
        if np.abs(radio_date- xray_date)<1.0:
            y_predict_linear, lower_linear_errors, upper_linear_errors, y_predict_uplim_bool = np.full(len(radio_dates), np.nan), np.full(len(radio_dates), np.nan), np.full(len(radio_dates), np.nan), np.zeros(len(radio_dates), dtype=bool)
            y_predict_linear[idx_closest] = xray_flux[0]
            lower_linear_errors[idx_closest] = xray_flux_unc_l[0]
            upper_linear_errors[idx_closest] = xray_flux_unc_u[0]
            y_predict_uplim_bool[idx_closest] =xray_uplims[0]
            return y_predict_linear, lower_linear_errors, upper_linear_errors, y_predict_uplim_bool
        else: 
            return np.full(len(radio_dates), np.nan), np.full(len(radio_dates), np.nan), np.full(len(radio_dates), np.nan), np.full(len(radio_dates), np.nan) 


    # If some point are extremely close, then it messes up the interpolation (since it thinks this is a steep rise/decay). 
    # Simple solution: average these X-ray data points.
    xray_dates, xray_flux, xray_flux_unc_l, xray_flux_unc_u, xray_uplims = merge_close_points(xray_dates, xray_flux, xray_flux_unc_l, xray_flux_unc_u, xray_uplims, threshold=0.1)
   

    # x-axis values
    x = xray_dates
    # x data for plotting:
    x_plot = np.linspace( x[0] , x[-1] , 2*int(1*(x[-1] - x[0])) ) 
    if verbose: print("Number of plotting points: ", 2*int(1*(x[-1] - x[0])) )
    # x data for prediction 
    x_predict = radio_dates

    # Get logged values
    y = np.log10(xray_flux) 
    yerr_l = np.log10(xray_flux) - np.log10(xray_flux - xray_flux_unc_l)
    yerr_u = np.log10(xray_flux + xray_flux_unc_u) - np.log10(xray_flux)

    # Get error for each data point
    #y_err = 0.5*(yerr_l  + yerr_u)
    y_err = np.maximum(yerr_l, yerr_u) # maximum for corresponding elements <<< conservative approach

    all_y_predict = []
    all_y_plot = []

    for i in range(10000):

        # Sample the fluxes, treating the uplims as detections
        y_sample = np.random.normal(y, y_err)

        func = Akima1DInterpolator(x, y_sample)

        y_plot = func(x_plot)
        all_y_plot.append(y_plot)
        y_predict = func(x_predict)
        all_y_predict.append(y_predict)


    # Get the results -- using mean and std
    #results_predict = np.mean(all_y_predict, axis=0)
    #std_results_predict = np.std(all_y_predict, axis=0, ddof=1)
    #print(results_predict)
    #print(std_results_predict)

    # Get the results -- using median and 16th/84th percentiles
    results_predict = np.median(all_y_predict, axis=0)
    err_predict_low = results_predict - np.percentile(all_y_predict, 16, axis=0)
    err_predict_high = np.percentile(all_y_predict, 84, axis=0) - results_predict 
    results_plot = np.median(all_y_plot, axis=0)
    

    ## Conduct filtering of predictions, and implement uncertainty scheme based on distance to nearest X-ray data point
    y_predict_uplim_bool = np.zeros(len( results_predict ), dtype=bool)
    y_predict = results_predict
    y_predict_err_l = err_predict_low.copy()
    y_predict_err_u = err_predict_high.copy()


    for i, radio_date in enumerate(x_predict):

        if np.isnan(radio_date): continue

        # Find indices of X-ray observations before and after the radio observation
        before_idxs = np.where(xray_dates < radio_date)[0]
        after_idxs = np.where(xray_dates > radio_date)[0]


        if len(before_idxs) != 0 :
            before_idx = before_idxs[-1]  # Last X-ray point before radio
            t_before = xray_dates[before_idx]
            f_before = y[before_idx]
            f_unc_before= y_err[before_idx]
            before_uplim = xray_uplims[before_idx] # whether this X-ray data point is an uplim
    
        
        if len(after_idxs) != 0: 
            after_idx = after_idxs[0]     # First X-ray point after radio
            t_after = xray_dates[after_idx]
            f_after = y[after_idx]
            f_unc_after = y_err[after_idx]
            after_uplim = xray_uplims[after_idx] # whether this X-ray data point is an uplim


        # Akima doesn't do extrapolation
        # If radio date is less than a day from the nearest point, use that value
        if len(before_idxs) == 0:
            t_before =0
            y_predict[i] = f_after
            y_predict_err_l[i] = f_unc_after
            y_predict_err_u[i] = f_unc_after
            before_uplim = after_uplim  # just to be safe
    
            
        if len(after_idxs) ==0 :
            t_after = np.inf
            y_predict[i] = f_before  
            y_predict_err_l[i] = f_unc_before
            y_predict_err_u[i] = f_unc_before
            after_uplim = before_uplim  # just to be safe



        # Time to nearest X-ray point, in days
        time_to_nearest = min(
            abs(radio_date - t_before),
            abs(radio_date - t_after)
        )  

        time_to_furthest = max(
            abs(radio_date - t_before),
            abs(radio_date - t_after)
        ) 
        

        # Reject the predictions that are too far away -- i.e. make them NaN
        # This will include the extrapolated points further than 1.0 day away
        # Default dt1 = 10.0, dt2 = 15.0

        # Workaround for now:
        if source_name=="SAX J1808.4-3658": dt1, dt2 = 6.5, 6.5
        if (time_to_nearest > dt1) or (time_to_nearest > 1.0 and time_to_furthest > dt2):  
            if verbose: print(f"Rejected based on distance: radio_MJD = {radio_date}; time_to_nearest: {time_to_nearest}; time_to_furthest: {time_to_furthest}" )
            y_predict[i] = np.nan
            y_predict_err_l[i] = np.nan 
            y_predict_err_u[i] = np.nan
            continue


        
        # If the one of the surrounding data points is an uplim (and the closest one is not a detection less than one day away), then treat the interpolated point as an uplim
        # Uplim error: dy = log10(x) - log10(x-dx) = log10(3/2) = 0.1760912591
        nearest_is_before = abs(radio_date - t_before) < abs(radio_date - t_after)

        uplim_nearest = before_uplim if nearest_is_before else after_uplim
        uplim_other   = after_uplim  if nearest_is_before else before_uplim

        if uplim_nearest or (uplim_other and time_to_nearest > 1.0):
            if verbose: print("Upper limit")
            y_predict_uplim_bool[i] = True
            y_predict_err_l[i] = np.log10(3/2)  # 1σ in log space
            y_predict_err_u[i] = 0
            continue

        
        if verbose: print(f"{radio_date}: time_to_nearest: {time_to_nearest} & time_to_furthest: {time_to_furthest}; y_predict_err: {y_predict_err_l[i]}")

    if verbose:
        print()
        print("LOG ERRORS:")
        print(y_predict_err_l)
        print()


    # Convert to linear space 
    y_predict_linear = 10**y_predict
    upper_linear_errors = 10**(y_predict + y_predict_err_u) - y_predict_linear
    lower_linear_errors = y_predict_linear - 10**(y_predict - y_predict_err_l) 

    if plot:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9))

        # Plot the actual data
        #ax1.errorbar(x[~xray_uplims], y[~xray_uplims], yerr=[yerr_l[~xray_uplims], yerr_u[~xray_uplims]], marker='.', label='Data', linestyle='none', color="red")
        ax1.errorbar(x[~xray_uplims], y[~xray_uplims], yerr=y_err[~xray_uplims], marker='.', label='Data', linestyle='none', color="red")
        ax1.scatter(x[xray_uplims], y[xray_uplims], marker='v', label="3-sigma upper limits", color="red")
        ax2.errorbar(x[~xray_uplims], xray_flux[~xray_uplims], yerr=[xray_flux_unc_l[~xray_uplims], xray_flux_unc_u[~xray_uplims]], marker='.', label='Data', linestyle='none', color="red")
        ax2.scatter(x[xray_uplims], xray_flux[xray_uplims], marker="v", label="3-sigma upper limits", color="red" )
        ax1.set_ylabel("Log flux")
        ax2.set_ylabel("Flux")

        # Plot radio MJDs:
        for i, date in enumerate(radio_dates):
            if i==0: 
                ax1.axvline(date, linestyle='-', color='black', alpha=0.1, label="Radio dates")
                ax2.axvline(date, linestyle='-', color='black', alpha=0.1, label="Radio dates")
            else: 
                ax1.axvline(date, linestyle='-', color='black', alpha=0.1)
                ax2.axvline(date, linestyle='-', color='black', alpha=0.1)



        # Plot the xplot results
        ax1.plot(x_plot, results_plot , '-', label='akima',alpha=0.5) 
        ax2.plot(x_plot, 10**results_plot , '-', label='akima',alpha=0.5)

        # Plot the interpolated points
        ax1.errorbar(x_predict[~y_predict_uplim_bool], y_predict[~y_predict_uplim_bool], yerr= y_predict_err_l[~y_predict_uplim_bool], alpha=0.5, fmt='.', color='black', label='Interpolated at radio times')
        ax1.scatter(x_predict[y_predict_uplim_bool], y_predict[y_predict_uplim_bool], marker='v', color="black")
        
        # Plot back in linear space
        ax2.errorbar(x_predict[~y_predict_uplim_bool], y_predict_linear [~y_predict_uplim_bool], yerr= [lower_linear_errors[~y_predict_uplim_bool], upper_linear_errors[~y_predict_uplim_bool]], alpha=0.5, fmt='.', color='black', label='Interpolated at radio times')
        ax2.scatter(x_predict[y_predict_uplim_bool], y_predict_linear [y_predict_uplim_bool], marker='v', color="black")

        ax1.set_xlabel("MJD")
        ax2.set_xlabel("MJD")

        handles, labels = ax2.get_legend_handles_labels()
        order = [labels.index("Data")] + [i for i, lbl in enumerate(labels) if lbl != "Data"]
        ax2.legend([handles[i] for i in order], [labels[i] for i in order], fontsize=10)
        plt.tight_layout()
        plt.show()

    if plotly:
        fig = go.Figure()

        # 1. X-ray detections with error bars
        fig.add_trace(go.Scatter(
            x=x[~xray_uplims],
            y=y[~xray_uplims],
            error_y=dict(type='data', array=y_err[~xray_uplims], visible=True),
            mode='markers',
            marker=dict(color='red'),
            name='Data'
        ))

        # 2. X-ray upper limits (plotted as downward triangles)
        fig.add_trace(go.Scatter(
            x=x[xray_uplims],
            y=y[xray_uplims],
            mode='markers',
            marker=dict(symbol='triangle-down', color='red'),
            name='3-sigma upper limits'
        ))

        # 3. Vertical lines at radio dates
        for i, date in enumerate(radio_dates):
            show_legend = (i == 0)
            fig.add_trace(go.Scatter(
                x=[date, date],
                y=[min(y)-1, max(y)+1],  # Adjust Y range as needed
                mode='lines',
                line=dict(color='black', dash='solid', width=1),
                opacity=0.1,
                name='Radio dates' if show_legend else None,
                showlegend=show_legend
            ))

        # 4. Fitted/interpolated xplot line
        fig.add_trace(go.Scatter(
            x=x_plot,
            y=results_plot,
            mode='lines',
            line=dict(color='blue', dash='solid'),
            name='akima',
            opacity=0.5
        ))

        # 5. Interpolated points (detections)
        fig.add_trace(go.Scatter(
            x=x_predict[~y_predict_uplim_bool],
            y=y_predict[~y_predict_uplim_bool],
            error_y=dict(type='data', array=y_predict_err_l[~y_predict_uplim_bool], visible=True),
            mode='markers',
            marker=dict(color='black'),
            name='Interpolated at radio times',
            opacity=0.5
        ))

        # 6. Interpolated points (upper limits)
        fig.add_trace(go.Scatter(
            x=x_predict[y_predict_uplim_bool],
            y=y_predict[y_predict_uplim_bool],
            mode='markers',
            marker=dict(symbol='triangle-down', color='black'),
            showlegend=False
        ))

        # Axes and layout
        fig.update_layout(
            yaxis_title='Log flux',
            xaxis_title='MJD or time',
            template='simple_white',
            legend=dict(borderwidth=1),
            height=600,
            width=1200
        )

        fig.show()



    if testing: return y_predict_linear, lower_linear_errors, upper_linear_errors, y_predict_uplim_bool, x_plot, 10**results_plot 

    else: return y_predict_linear, lower_linear_errors, upper_linear_errors, y_predict_uplim_bool
    


