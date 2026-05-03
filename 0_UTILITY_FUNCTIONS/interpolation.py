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
NOTES:

Weighted averages:
- https://www.amherst.edu/system/files/media/1871/weighted%20average.pdf
- https://seismo.berkeley.edu/~kirchner/Toolkits/Toolkit_12.pdf .... extra errors due to scatter


I can freely convert between errors in log and linear space. The only difference that may occur is due to floating point precision. 
y = log10(x)
x = 10**y

dx_u = 10**(y + dy_u) - 10**y
dx_l = 10**y - 10**(y - dy_l) 

dy_u = log10(x + dx_u) - log10(x) = log10( (x + dx_u)/x ) = log10(1 + dx_u/x) = log10(1 + (10**(y + dy_u) - 10**y)/10**y) = log10(1 + 10**dy_u - 1) = log10(10**dy_u) = dy_u 
dy_l = log10(x) - log10(x - dx_l) = log10(x/(x-dx_l)) = log10(1/(1 - dx_l/x)) = log10(1/(1 - (10**y - 10**(y - dy_l))/10**y)) = log10(1/(1 - (1 - 10**(-dy_l)))) = log10(1/10**(-dy_l)) = log10(10**dy_l) = dy_l
"""



##############################################################################################################
## HELPER FUNCTIONS FOR SCIPY INTERPOLATION



def merge_close_points(x, y, y_unc_l, y_unc_u, uplim, threshold=0.05, weighted_average=True):
    """
    When points are extremely close in x value, then the akima interpolation fails. In this case, I need to average the close data points.

    Merge consecutive points in time (x) that are within a given threshold.
    If upper limits (uplim) and detections are mixed, only keep detections.
    Uncertainties are averaged or propagated depending on `weighted_average`.
    
    Parameters
    ----------
    - x: array-like, x values (e.g., MJDs)
    - y: array-like, y values (e.g., fluxes)
    - y_unc_l: array-like, lower uncertainties on y
    - y_unc_u: array-like, upper uncertainties on y
    - uplim: array-like of bool, whether each point is an upper limit
    - threshold: float, maximum allowed difference in x to consider points as "close"; this is in the same units as x (e.g., days)
    """

    # Convert inputs to numpy arrays for easier processing    
    x, y, y_unc_l, y_unc_u, uplim = map(np.asarray, (x, y, y_unc_l, y_unc_u, uplim))
    
    # Initialise lists to hold merged results
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

        # Filter out upper limits if both detections and limits exist in the group
        if True in group_uplim and False in group_uplim:
            keep_indices = [j for j, is_uplim in enumerate(group_uplim) if not is_uplim] # indices of detections
            group_x = [group_x[j] for j in keep_indices]
            group_y = [group_y[j] for j in keep_indices]
            group_y_unc_l = [group_y_unc_l[j] for j in keep_indices]
            group_y_unc_u = [group_y_unc_u[j] for j in keep_indices]
            group_uplim = [False]  
        # Else either all are detections or all are uplims, so we keep them as they are (i.e. if all are uplims, we keep all of them as uplims)

        # Compute mean only if more than one point remains 
        if len(group_x) > 1:

            group_x = np.array(group_x)
            group_y = np.array(group_y)
            group_y_unc_l = np.array(group_y_unc_l)
            group_y_unc_u = np.array(group_y_unc_u)

            if weighted_average:
                avg_unc = (group_y_unc_l + group_y_unc_u) / 2
                weights = 1 / (avg_unc**2) # weights are inverse variance
                merged_y.append( np.sum(weights * group_y) / np.sum(weights))
                # Use the same weighting for x -- i.e. bias it towards the x of the more reliable y values 
                merged_x.append(np.sum(weights * group_x) / np.sum(weights))
                # Get upper and lower uncertainties separately
                # Technically, we should have:
                # merged_y_unc_u = 1 / np.sqrt(np.sum(weights)) and merged_y_unc_l = 1 / np.sqrt(np.sum(weights)), but the code below retains the asymmetry in the uncertainties. 
                weights_u = 1 / (group_y_unc_u**2)
                weights_l = 1 / (group_y_unc_l**2)
                merged_y_unc_u.append( 1 / np.sqrt(np.sum(weights_u)) )
                merged_y_unc_l.append( 1 / np.sqrt(np.sum(weights_l)) )

            else: # do a normal average
                merged_x.append(np.mean(group_x))
                merged_y.append(np.mean(group_y))
                # Propagate the uncertainties -- use vectorisation 
                merged_y_unc_l.append(np.sqrt(np.sum(np.array(group_y_unc_l) ** 2)) / len(group_y_unc_l))
                merged_y_unc_u.append(np.sqrt(np.sum(np.array(group_y_unc_u) ** 2)) / len(group_y_unc_u))
        

        else: # Single point -- no merging needed
            merged_x.append(group_x[0])
            merged_y.append(group_y[0])
            merged_y_unc_l.append(group_y_unc_l[0])
            merged_y_unc_u.append(group_y_unc_u[0])

        # Set uplim value: If all True, keep True. If all False, keep False. Otherwise, if there was a mix of True and False originally, it's already False from above.
        # all(group_uplim) checks whether every element in group_uplim is truthy. If all elements evaluate to True, it returns True. If any element is False, it returns False.
        merged_uplim.append(all(group_uplim))

        # Go to next group
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
... which is the same as White 2017 eqn (15).
"""

def manual_linear_interpolation(xray_dates, xray_flux, xray_flux_unc_l, xray_flux_unc_u, xray_uplims, radio_dates, verbose=True, plot=True, dt1 = 3.0, dt2 = 10.0 ):
    """
    Perform a manual linear interpolation in log space to get the predicted X-ray flux at the radio dates.

    Parameters
    ----------
    - xray_dates: array-like, MJDs of X-ray observations
    - xray_flux: array-like, X-ray flux values
    - xray_flux_unc_l: array-like, Lower uncertainties for X-ray flux
    - xray_flux_unc_u: array-like, Upper uncertainties for X-ray flux
    - xray_uplims: array-like, Boolean array indicating upper limits
    - radio_dates: array-like, MJDs of radio observations
    - verbose: bool, whether to print verbose output
    - plot: bool, whether to create plots
    - dt1: float, time window for filtering predictions
    - dt2: float, time window for filtering predictions
    
    Returns
    ----------
    - y_predict_linear: array, predicted X-ray flux at radio dates (in linear space)
    - lower_linear_errors: array, lower uncertainties on predicted X-ray flux (in linear space)
    - upper_linear_errors: array, upper uncertainties on predicted X-ray flux (in linear space)
    - y_predict_uplim_bool: array of bool, whether the predicted point is an upper limit
    Note that if the prediction is filtered out, the predicted flux and its uncertainties will be set to NaN.
    
    """

    # Do not interpolate if there are no X-ray data points -- just return NaN for all radio dates
    if len(xray_dates)==0:
        return np.full(len(radio_dates), np.nan), np.full(len(radio_dates), np.nan), np.full(len(radio_dates), np.nan), np.zeros(len(radio_dates), dtype=bool)

    # If there is only one X-ray data point, then we cannot do interpolation. 
    # Instead, if the radio date is within 1 day of the X-ray date, we use the X-ray flux as the predicted value (with the same uncertainties). 
    # Otherwise, we set it to NaN.    
    elif len(xray_dates) == 1:
        xray_date = xray_dates[0]
        idx_closest = np.argmin(np.abs(radio_dates - xray_date))
        radio_date = radio_dates[idx_closest]
        if np.abs(radio_date- xray_date)<1.0: # it is within 1 day
            # Create arrays of NaN for all radio dates, and then fill in the value for the closest radio date
            y_predict_linear, lower_linear_errors, upper_linear_errors, y_predict_uplim_bool = np.full(len(radio_dates), np.nan), np.full(len(radio_dates), np.nan), np.full(len(radio_dates), np.nan), np.zeros(len(radio_dates), dtype=bool)
            y_predict_linear[idx_closest] = xray_flux[0]
            lower_linear_errors[idx_closest] = xray_flux_unc_l[0]
            upper_linear_errors[idx_closest] = xray_flux_unc_u[0]
            y_predict_uplim_bool[idx_closest] =xray_uplims[0]
            return y_predict_linear, lower_linear_errors, upper_linear_errors, y_predict_uplim_bool
        else: # NaN for all radio dates
            return np.full(len(radio_dates), np.nan), np.full(len(radio_dates), np.nan), np.full(len(radio_dates), np.nan), np.full(len(radio_dates), np.nan) 



    # If some point are extremely close, then it messes up the akima interpolation (since it thinks this is a steep rise/decay). 
    # Simple solution: average these X-ray data points. For consistency with the akima interpolation, we do the same.
    xray_dates, xray_flux, xray_flux_unc_l, xray_flux_unc_u, xray_uplims = merge_close_points(xray_dates, xray_flux, xray_flux_unc_l, xray_flux_unc_u, xray_uplims, threshold=0.1)


    # x-axis values
    x = xray_dates
    # x data for prediction -- i.e. the radio dates
    x_predict = radio_dates

    # Get logged values, since we run the interpolation in log space.
    y = np.log10(xray_flux) 
    yerr_l = np.log10(xray_flux) - np.log10(xray_flux - xray_flux_unc_l)
    yerr_u = np.log10(xray_flux + xray_flux_unc_u) - np.log10(xray_flux)

    # Get uncertainty for each data point
    #y_err = 0.5*(yerr_l  + yerr_u)
    y_err = np.maximum(yerr_l, yerr_u) # maximum for corresponding elements <<< conservative approach

    
    # Initialise arrays to hold the predicted values and their uncertainties
    y_predict = np.zeros(len( x_predict ))
    y_predict_err_l = np.zeros(len( x_predict ))
    y_predict_err_u = np.zeros(len( x_predict ))
    y_predict_uplim_bool = np.zeros(len( x_predict ), dtype=bool)


    # Loop through each radio date
    for i, radio_date in enumerate(x_predict):

        if np.isnan(radio_date): continue

        # Find indices of X-ray observations before and after the radio observation.
        # np.where returns a tuple of arrays, where the first array contains the indices of the elements that satisfy the condition.
        before_idxs = np.where(xray_dates < radio_date)[0]
        after_idxs = np.where(xray_dates > radio_date)[0]

        if len(before_idxs) != 0 : # there are X-ray points before the radio date
            before_idx = before_idxs[-1]  # Last X-ray point before radio date ... this works because the data is ordered in time
            t_before = xray_dates[before_idx]
            f_before = y[before_idx]
            f_unc_before= y_err[before_idx]
            f_unc_before_l = yerr_l[before_idx]
            f_unc_before_u = yerr_u[before_idx]
            before_uplim = xray_uplims[before_idx] # whether this X-ray data point is an uplim
    
        
        if len(after_idxs) != 0: # there are X-ray points after the radio date
            after_idx = after_idxs[0]     # First X-ray point after radio date ... this works because the data is ordered in time
            t_after = xray_dates[after_idx]
            f_after = y[after_idx]
            f_unc_after = y_err[after_idx]
            f_unc_after_l = yerr_l[after_idx]
            f_unc_after_u = yerr_u[after_idx]
            after_uplim = xray_uplims[after_idx] # whether this X-ray data point is an uplim


        if len(before_idxs) != 0 and len(after_idxs) !=0 : # there are X-ray points both before and after the radio date, so we can do interpolation
            
            # y = y1 + (x - x1) * (y2 - y1) / (x2 - x1)
            alpha = (radio_date - t_before) / (t_after - t_before) # (x - x1)/(x2 - x1)
            log_interp_flux = f_before + alpha * (f_after - f_before)
            y_predict[i] = log_interp_flux
            
            # Using propagation of uncertainties:
            # dy = np.sqrt( (1 - (x - x1)/(x2 - x1))**2 * dy1**2 + ((x - x1)/(x2 - x1))**2 *dy2**2 ) = np.sqrt( (1 - alpha)**2 * dy1**2 + alpha**2 *dy2**2 )
            # If using the same uncertainty for upper and lower:
            # log_interp_flux_unc = np.sqrt((1 - alpha)**2 * f_unc_before**2 + alpha**2 * f_unc_after**2)
            # y_predict_err_l[i], y_predict_err_u[i] = log_interp_flux_unc, log_interp_flux_unc
            # If using the upper and lower uncertainty separately:
            log_interp_flux_unc_l = np.sqrt((1 - alpha)**2 * f_unc_before_l**2 + alpha**2 * f_unc_after_l**2)
            log_interp_flux_unc_u = np.sqrt((1 - alpha)**2 * f_unc_before_u**2 + alpha**2 * f_unc_after_u**2)
            y_predict_err_l[i], y_predict_err_u[i] = log_interp_flux_unc_l, log_interp_flux_unc_u


        # linear interp doesn't do extrapolation for radio dates outside the range of the X-ray dates.
        # Assign the X-ray data point that is closest. 
        # This will later be filtered so that we only use an extrapolated radio date if it is less than 1 day from the nearest X-ray point.
        if len(before_idxs) == 0: # radio date is before all X-ray dates; there are data point after
            t_before =0 # assign very small value
            y_predict[i] = f_after
            y_predict_err_l[i] = f_unc_after_l
            y_predict_err_u[i] = f_unc_after_u
        if len(after_idxs) ==0 : # radio date is after all X-ray dates; there are data point before
            t_after = 6e20 # assign a very large value
            y_predict[i] = f_before  
            y_predict_err_l[i] = f_unc_before_l
            y_predict_err_u[i] = f_unc_before_u


        # Time to nearest X-ray point, in days
        time_to_nearest = min(
            abs(radio_date - t_before),
            abs(radio_date - t_after)
        )  
        # Time to furthest X-ray point, in days
        time_to_furthest = max(
            abs(radio_date - t_before),
            abs(radio_date - t_after)
        ) 
        

        # Reject the predictions that are too far away -- i.e. make them NaN
        # For the extrapolated points, time_to_furthest is always greater than dt2, so the condition for rejecting extrapolated points is that time_to_nearest is greater than 1.0 day.
        if (time_to_nearest > dt1) or (time_to_nearest > 1.0 and time_to_furthest > dt2):  
            if verbose: print(f"Rejected based on distance: radio_MJD = {radio_date}; time_to_nearest: {time_to_nearest}; time_to_furthest: {time_to_furthest}" )
            y_predict[i] = np.nan
            y_predict_err_l[i] = np.nan 
            y_predict_err_u[i] = np.nan
            continue

        # Deal with upper limits: 
        # Treat a predicted point as an upper limit if:
        # (1) the closest data point is an uplim, 
        # (2) OR one of the surrounding data points is an uplim and the closest data point is more than 1 day away.
        # Uplim error: dy = log10(x) - log10(x-dx) = log10(3/2) = 0.1760912591
        
        # Determine which of the two surrounding data points is closer to the radio date
        # nearest_is_before is True if the closest data point is the one before the radio date, and False if the closest data point is the one after the radio date.
        nearest_is_before = abs(radio_date - t_before) < abs(radio_date - t_after)
        # Assign the nearest and furthest data points and their uplim status based on which one is closer to the radio date
        uplim_nearest = before_uplim if nearest_is_before else after_uplim
        uplim_other   = after_uplim  if nearest_is_before else before_uplim

        if uplim_nearest or (uplim_other and time_to_nearest > 1.0):
            if verbose: print("Upper limit")
            y_predict_uplim_bool[i] = True
            y_predict_err_l[i] = np.log10(3/2)  # 1σ in log space
            y_predict_err_u[i] = 0
            continue
        
        if verbose: print(f"{radio_date}: time_to_nearest: {time_to_nearest} & time_to_furthest: {time_to_furthest}; y_predict_err_l: {y_predict_err_l[i]}")

    if verbose:
        print()
        print("LOG ERRORS:")
        print(y_predict_err_l)
        print()



    # Convert back to linear space 
    y_predict_linear = 10**y_predict # x = 10**y
    upper_linear_errors = 10**(y_predict + y_predict_err_u) - y_predict_linear
    lower_linear_errors = y_predict_linear - 10**(y_predict - y_predict_err_l) 
    # Note that for upper limits, dx_l = x - 10**( log10(x) - log10(3/2)) = x - (2/3)x = (1/3)x.

    if plot:

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9))
        
        # Plot the actual log data (after merging close points)
        ax1.errorbar(x, y, yerr=y_err, uplims = xray_uplims, label='Data',  color="red")
        ax1.set_ylabel("Log flux")

        # Plot the actual linear data (after merging close points)
        ax2.errorbar(x, xray_flux, yerr=[xray_flux_unc_l, xray_flux_unc_u], uplims= xray_uplims, label='Data', color="red")
        ax2.set_ylabel("Flux")

        # Plot radio MJDs:
        for i, date in enumerate(radio_dates):
            if i==0: 
                ax1.axvline(date, linestyle='-', color='black', alpha=0.1, label="Radio dates")
                ax2.axvline(date, linestyle='-', color='black', alpha=0.1, label="Radio dates")
            else: 
                ax1.axvline(date, linestyle='-', color='black', alpha=0.1)
                ax2.axvline(date, linestyle='-', color='black', alpha=0.1)


        # Plot the interpolated points on the log plot (using y_predict and y_predict_err_l and y_predict_err_u)
        ax1.errorbar(x_predict[~y_predict_uplim_bool], y_predict[~y_predict_uplim_bool], yerr= y_predict_err_l[~y_predict_uplim_bool], alpha=0.5, fmt='.', color='black', label='Interpolated at radio times')
        ax1.scatter(x_predict[y_predict_uplim_bool], y_predict[y_predict_uplim_bool], marker='v', color="black")
        
        # Plot the interpolated points on the linear plot (using y_predict_linear and lower_linear_errors and upper_linear_errors)
        ax2.errorbar(x_predict[~y_predict_uplim_bool], y_predict_linear [~y_predict_uplim_bool], yerr= [lower_linear_errors[~y_predict_uplim_bool], upper_linear_errors[~y_predict_uplim_bool]], alpha=0.5, fmt='.', color='black', label='Interpolated at radio times')
        ax2.scatter(x_predict[y_predict_uplim_bool], y_predict_linear [y_predict_uplim_bool], marker='v', color="black")

        # Set labels and legends
        ax1.set_xlabel("MJD")
        ax2.set_xlabel("MJD")
        handles, labels = ax2.get_legend_handles_labels()
        order = [labels.index("Data")] + [i for i, lbl in enumerate(labels) if lbl != "Data"]
        ax2.legend([handles[i] for i in order], [labels[i] for i in order], fontsize=10)
        plt.tight_layout()
        plt.show()


    # Return the interpolated values in linear flux space.
    return y_predict_linear, lower_linear_errors, upper_linear_errors, y_predict_uplim_bool


##############################################################################################################
## RUNNER FUNCTION FOR SCIPY INTERPOLATION



def interp_data_scipy_MC(xray_dates, xray_flux, xray_flux_unc_l, xray_flux_unc_u, xray_uplims, radio_dates, testing=False, dt1 = 3.0, dt2 = 10.0, plotly = False, verbose=True, plot=True): # dt1 and dt were previously 10,15
    """
    Interpolate X-ray points using akima on a log scale
    I include uplims, but if the interpolated point is closest to an uplim, then it is also considered an uplim.

    Run MC repeats to get the error on the interpolated data points, since we cannot propagate uncertainties in this scheme.
    
    Parameters
    ----------
    - xray_dates: array-like, MJDs of X-ray observations
    - xray_flux: array-like, X-ray flux values
    - xray_flux_unc_l: array-like, Lower uncertainties for X-ray flux
    - xray_flux_unc_u: array-like, Upper uncertainties for X-ray flux
    - xray_uplims: array-like, Boolean array indicating upper limits
    - radio_dates: array-like, MJDs of radio observations
    - testing: bool, whether to run in testing mode (with fewer MC iterations and less dense plotting points)
    - dt1: float, time window for filtering predictions
    - dt2: float, time window for filtering predictions
    - plotly: bool, whether to create interactive plots with plotly
    - verbose: bool, whether to print verbose output
    - plot: bool, whether to create static plots with matplotlib
    """

    if verbose: 
        print("dt1, dt2: ", dt1, dt2)
        print()

    # For reproducibility
    np.random.seed(42)


    # Do not interpolate if there are no X-ray data points -- just return NaN for all radio dates
    if len(xray_dates)==0:
        return np.full(len(radio_dates), np.nan), np.full(len(radio_dates), np.nan), np.full(len(radio_dates), np.nan), np.zeros(len(radio_dates), dtype=bool)
     
    # If there is only one X-ray data point, then we cannot do interpolation. 
    # Instead, if the radio date is within 1 day of the X-ray date, we use the X-ray flux as the predicted value (with the same uncertainties). 
    # Otherwise, we set it to NaN.
    elif len(xray_dates) == 1:
        xray_date = xray_dates[0]
        idx_closest = np.argmin(np.abs(radio_dates - xray_date))
        radio_date = radio_dates[idx_closest]
        if np.abs(radio_date- xray_date)<1.0: # it is within 1 day
            # Create arrays of NaN for all radio dates, and then fill in the value for the closest radio date
            y_predict_linear, lower_linear_errors, upper_linear_errors, y_predict_uplim_bool = np.full(len(radio_dates), np.nan), np.full(len(radio_dates), np.nan), np.full(len(radio_dates), np.nan), np.zeros(len(radio_dates), dtype=bool)
            y_predict_linear[idx_closest] = xray_flux[0]
            lower_linear_errors[idx_closest] = xray_flux_unc_l[0]
            upper_linear_errors[idx_closest] = xray_flux_unc_u[0]
            y_predict_uplim_bool[idx_closest] =xray_uplims[0]
            return y_predict_linear, lower_linear_errors, upper_linear_errors, y_predict_uplim_bool
        else: # NaN for all radio dates
            return np.full(len(radio_dates), np.nan), np.full(len(radio_dates), np.nan), np.full(len(radio_dates), np.nan), np.full(len(radio_dates), np.nan) 


    # If some point are extremely close, then it messes up the akima interpolation (since it thinks this is a steep rise/decay). 
    # Simple solution: average these X-ray data points.
    xray_dates, xray_flux, xray_flux_unc_l, xray_flux_unc_u, xray_uplims = merge_close_points(xray_dates, xray_flux, xray_flux_unc_l, xray_flux_unc_u, xray_uplims, threshold=0.1)
   

    # x-axis values:
    x = xray_dates
    
    # x data for prediction:
    x_predict = radio_dates

    # x data for plotting:
    x_plot = np.linspace( x[0] , x[-1] , 2*int(1*(x[-1] - x[0])) ) 
    if verbose: print("Number of plotting points: ", 2*int(1*(x[-1] - x[0])) )

    # Get logged values, since we run the interpolation in log space.
    y = np.log10(xray_flux) 
    yerr_l = np.log10(xray_flux) - np.log10(xray_flux - xray_flux_unc_l)
    yerr_u = np.log10(xray_flux + xray_flux_unc_u) - np.log10(xray_flux)

    # Get error for each data point
    #y_err = 0.5*(yerr_l  + yerr_u)
    y_err = np.maximum(yerr_l, yerr_u) # maximum for corresponding elements <<< conservative approach

    # MC simulations:
    # For each simulation, we sample the log fluxes from a normal distribution with mean = observed log flux and sigma = uncertainty (treating uplims as detections). 
    # We then run the interpolation to get the predicted log fluxes at the radio dates. 
    # After running all simulations, we can get the final predicted log fluxes and their uncertainties by taking the median and 16th/84th percentiles of the predicted log fluxes across all simulations.
    # Initialise arrays for the MC simulations
    all_y_predict = [] # at the radio dates
    all_y_plot = [] # for plotting the interpolated curve and its uncertainty
    for i in range(10000):

        # Sample the log fluxes, treating the uplims as detections
        y_sample = np.random.normal(y, y_err)

        # Run the interpolation, inputting the x-ray dates and the sampled log fluxes. 
        func = Akima1DInterpolator(x, y_sample)

        # Calculate the predicted log fluxes at the radio dates.
        y_predict = func(x_predict)
        all_y_predict.append(y_predict)

        # Calculate the predicted log fluxes at the plotting dates (for visualisation purposes).
        y_plot = func(x_plot)
        all_y_plot.append(y_plot)
        
        

    # Get the results -- using median and 16th/84th percentiles
    # Should be basically the same as the mean and std as we used symmetric uncertainties.
    results_predict = np.median(all_y_predict, axis=0)
    err_predict_low = results_predict - np.percentile(all_y_predict, 16, axis=0)
    err_predict_high = np.percentile(all_y_predict, 84, axis=0) - results_predict 
    results_plot = np.median(all_y_plot, axis=0)

    # Get the results -- using mean and std
    #results_predict = np.mean(all_y_predict, axis=0)
    #std_results_predict = np.std(all_y_predict, axis=0, ddof=1)
 


    # Initialise arrays to hold the predicted values and their uncertainties
    y_predict_uplim_bool = np.zeros(len( results_predict ), dtype=bool)
    # Set the predicted values and their uncertainties to the results from the MC simulations.
    y_predict = results_predict.copy()
    y_predict_err_l = err_predict_low.copy()
    y_predict_err_u = err_predict_high.copy()


    # Loop through each radio date
    for i, radio_date in enumerate(x_predict):

        if np.isnan(radio_date): continue

        # Find indices of X-ray observations before and after the radio observation
        # np.where returns a tuple of arrays, where the first array contains the indices of the elements that satisfy the condition.
        before_idxs = np.where(xray_dates < radio_date)[0]
        after_idxs = np.where(xray_dates > radio_date)[0]


        if len(before_idxs) != 0 : # there are X-ray points before the radio date
            before_idx = before_idxs[-1]  # Last X-ray point before radio... this works because the data is ordered in time
            t_before = xray_dates[before_idx]
            f_before = y[before_idx]
            f_unc_before_l = yerr_l[before_idx]
            f_unc_before_u = yerr_u[before_idx]
            f_unc_before= y_err[before_idx]
            before_uplim = xray_uplims[before_idx] # whether this X-ray data point is an uplim
    
        
        if len(after_idxs) != 0: # there are X-ray points after the radio date
            after_idx = after_idxs[0]     # First X-ray point after radio... this works because the data is ordered in time
            t_after = xray_dates[after_idx]
            f_after = y[after_idx]
            f_unc_after_l = yerr_l[after_idx]
            f_unc_after_u = yerr_u[after_idx]
            f_unc_after= y_err[after_idx]
            after_uplim = xray_uplims[after_idx] # whether this X-ray data point is an uplim


        # By default, akima doesn't do extrapolation for radio dates outside the range of the X-ray dates.
        # Assign the X-ray data point that is closest. 
        # This will later be filtered so that we only use an extrapolated radio date if it is less than 1 day from the nearest X-ray point.
        if len(before_idxs) == 0:  # radio date is before all X-ray dates; there are data point after
            t_before =0 # assign very small value
            y_predict[i] = f_after
            y_predict_err_l[i] = f_unc_after_l
            y_predict_err_u[i] = f_unc_after_u
            before_uplim = after_uplim  # just to be safe
        if len(after_idxs) ==0 : # radio date is after all X-ray dates; there are data point before
            t_after = 6e20 # assign a very large value
            y_predict[i] = f_before  
            y_predict_err_l[i] = f_unc_before_l
            y_predict_err_u[i] = f_unc_before_u
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
        if (time_to_nearest > dt1) or (time_to_nearest > 1.0 and time_to_furthest > dt2):  
            if verbose: print(f"Rejected based on distance: radio_MJD = {radio_date}; time_to_nearest: {time_to_nearest}; time_to_furthest: {time_to_furthest}" )
            y_predict[i] = np.nan
            y_predict_err_l[i] = np.nan 
            y_predict_err_u[i] = np.nan
            continue


        # Deal with upper limits: 
        # Treat a predicted point as an upper limit if:
        # (1) the closest data point is an uplim, 
        # (2) OR one of the surrounding data points is an uplim and the closest data point is more than 1 day away.
        # Uplim error: dy = log10(x) - log10(x-dx) = log10(3/2) = 0.1760912591

        # Determine which of the two surrounding data points is closer to the radio date
        # nearest_is_before is True if the closest data point is the one before the radio date, and False if the closest data point is the one after the radio date.
        nearest_is_before = abs(radio_date - t_before) < abs(radio_date - t_after)
        # Assign the nearest and furthest data points and their uplim status based on which one is closer to the radio date
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


    # Convert back to linear space 
    y_predict_linear = 10**y_predict # x = 10**y
    upper_linear_errors = 10**(y_predict + y_predict_err_u) - y_predict_linear
    lower_linear_errors = y_predict_linear - 10**(y_predict - y_predict_err_l) 
    # Note that for upper limits, dx_l = x - 10**( log10(x) - log10(3/2)) = x - (2/3)x = (1/3)x.


    if plot:

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9))

        # Plot the actual log data (after merging close points); I use the y_err here which is the maximum of the upper and lower errors, since this is what we used for the MC sampling.
        #ax1.errorbar(x[~xray_uplims], y[~xray_uplims], yerr=[yerr_l[~xray_uplims], yerr_u[~xray_uplims]], marker='.', label='Data', linestyle='none', color="red")
        ax1.errorbar(x[~xray_uplims], y[~xray_uplims], yerr=y_err[~xray_uplims], marker='.', label='Data', linestyle='none', color="red")
        ax1.scatter(x[xray_uplims], y[xray_uplims], marker='v', label="3-sigma upper limits", color="red")
        ax1.set_ylabel("Log flux")

        # Plot the actual linear data (after merging close points)
        ax2.errorbar(x[~xray_uplims], xray_flux[~xray_uplims], yerr=[xray_flux_unc_l[~xray_uplims], xray_flux_unc_u[~xray_uplims]], marker='.', label='Data', linestyle='none', color="red")
        ax2.scatter(x[xray_uplims], xray_flux[xray_uplims], marker="v", label="3-sigma upper limits", color="red" )
        ax2.set_ylabel("Flux")

        # Plot radio MJDs:
        for i, date in enumerate(radio_dates):
            if i==0: 
                ax1.axvline(date, linestyle='-', color='black', alpha=0.1, label="Radio dates")
                ax2.axvline(date, linestyle='-', color='black', alpha=0.1, label="Radio dates")
            else: 
                ax1.axvline(date, linestyle='-', color='black', alpha=0.1)
                ax2.axvline(date, linestyle='-', color='black', alpha=0.1)


        # Plot the xplot results on log scale
        ax1.plot(x_plot, results_plot , '-', label='akima',alpha=0.5)
        # Plot the xplot results on linear scale 
        ax2.plot(x_plot, 10**results_plot , '-', label='akima',alpha=0.5)

        # Plot the predicted points (i.e. at the radio times) on the log plot (using y_predict and y_predict_err_l and y_predict_err_u)
        ax1.errorbar(x_predict[~y_predict_uplim_bool], y_predict[~y_predict_uplim_bool], yerr= [y_predict_err_l[~y_predict_uplim_bool], y_predict_err_u[~y_predict_uplim_bool]], alpha=0.5, fmt='.', color='black', label='Interpolated at radio times')
        ax1.scatter(x_predict[y_predict_uplim_bool], y_predict[y_predict_uplim_bool], marker='v', color="black")
        
        # Plot the predicted points (i.e. at the radio times) on the linear plot (using y_predict_linear and the corresponding errors)
        ax2.errorbar(x_predict[~y_predict_uplim_bool], y_predict_linear [~y_predict_uplim_bool], yerr= [lower_linear_errors[~y_predict_uplim_bool], upper_linear_errors[~y_predict_uplim_bool]], alpha=0.5, fmt='.', color='black', label='Interpolated at radio times')
        ax2.scatter(x_predict[y_predict_uplim_bool], y_predict_linear [y_predict_uplim_bool], marker='v', color="black")


        # Set labels and legends
        ax1.set_xlabel("MJD")
        ax2.set_xlabel("MJD")
        handles, labels = ax2.get_legend_handles_labels()
        order = [labels.index("Data")] + [i for i, lbl in enumerate(labels) if lbl != "Data"]
        ax2.legend([handles[i] for i in order], [labels[i] for i in order], fontsize=10)
        plt.tight_layout()
        plt.show()


    if plotly:
        
        fig = go.Figure()

        # Plot the actual log data (after merging close points); I use the y_err here which is the maximum of the upper and lower errors, since this is what we used for the MC sampling.
        # Detections
        fig.add_trace(go.Scatter(
            x=x[~xray_uplims],
            y=y[~xray_uplims],
            error_y=dict(type='data', arrayminus=yerr_l[~xray_uplims], array=yerr_u[~xray_uplims], visible=True),
            mode='markers',
            marker=dict(color='red'),
            name='Data'
        ))
        # Upper limits (plotted as downward triangles)
        fig.add_trace(go.Scatter(
            x=x[xray_uplims],
            y=y[xray_uplims],
            mode='markers',
            marker=dict(symbol='triangle-down', color='red'),
            name='3-sigma upper limits'
        ))

        # Plot vertical lines at radio dates
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

        # Plot the xplot results (i.e. interpolation curve)
        fig.add_trace(go.Scatter(
            x=x_plot,
            y=results_plot,
            mode='lines',
            line=dict(color='blue', dash='solid'),
            name='akima',
            opacity=0.5
        ))

        # Plot the predicted points (i.e. at the radio times) 
        # Detections
        fig.add_trace(go.Scatter(
            x=x_predict[~y_predict_uplim_bool],
            y=y_predict[~y_predict_uplim_bool],
            error_y=dict(type='data', array=y_predict_err_u[~y_predict_uplim_bool], arrayminus=y_predict_err_l[~y_predict_uplim_bool], visible=True),
            mode='markers',
            marker=dict(color='black'),
            name='Interpolated at radio times',
            opacity=0.5
        ))
        # Upper limits
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


    # Return the results on a linear scale.
    if testing: return y_predict_linear, lower_linear_errors, upper_linear_errors, y_predict_uplim_bool, x_plot, 10**results_plot 
    else: return y_predict_linear, lower_linear_errors, upper_linear_errors, y_predict_uplim_bool
    


##############################################################################################################