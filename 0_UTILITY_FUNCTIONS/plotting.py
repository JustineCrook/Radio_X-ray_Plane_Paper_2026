## PLOTTING FUNCTIONS

##############################################################################################################

## IMPORTS
import numpy as np
import ast
import matplotlib.pyplot as plt
import pandas as pd
pd.set_option('display.max_rows', 1000)
from io import StringIO
import matplotlib as mpl
from matplotlib.animation import FuncAnimation
from matplotlib.colors import Normalize
from IPython.display import HTML
import itertools
import os
import plotly.graph_objects as go
from astropy.time import Time
import matplotlib.dates as mdates
from matplotlib.patches import Polygon
from decimal import Decimal, ROUND_FLOOR, ROUND_CEILING
import seaborn as sns
from matplotlib.ticker import FuncFormatter
from scipy.stats import ks_2samp
from scipy.stats import anderson_ksamp
import matplotlib.pyplot as plt
from ndtest import ks2d2s

import sys
sys.path.append(os.path.abspath("../0_UTILITY_FUNCTIONS/"))
from get_data import *




## PLOT FORMATTING
plt.rcParams['axes.formatter.useoffset'] = False  # Disable offset mode
plt.rcParams['axes.formatter.use_locale'] = False  # Locale settings can also influence formatting, but this line is optional.
plt.rcParams['axes.formatter.limits'] = (-5, 5)   # Range outside which scientific notation is used; set high to avoid sci-notation.
mpl.rcParams.update(mpl.rcParamsDefault)
mpl.rcParams['font.size'] = 14
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['ytick.right'] = True
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['ytick.minor.visible'] = True
mpl.rcParams['ytick.labelsize'] = 'small'
mpl.rcParams['ytick.major.width'] = 1
mpl.rcParams['ytick.minor.width'] = 1
mpl.rcParams['ytick.major.size'] = 6.0
mpl.rcParams['ytick.minor.size'] = 3.0
mpl.rcParams['xtick.top'] = True
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['xtick.labelsize'] = 'small'
mpl.rcParams['xtick.minor.visible'] = True
mpl.rcParams['xtick.major.width'] = 1
mpl.rcParams['xtick.minor.width'] = 1
mpl.rcParams['xtick.major.size'] = 6.0
mpl.rcParams['xtick.minor.size'] = 3.0
mpl.rcParams['axes.linewidth'] = 1.5


## USEFUL VARIABLES
colours_random = [ "red", "blue","green", "pink", "cyan", "yellow", "brown", "orange", "black", 
           "purple", "lime", "teal", "magenta", "gold", "navy", "olive", "maroon", "turquoise", 
           "violet", "indigo", "salmon", "coral", "khaki", "lavender", "chartreuse"]
colours = {
    '1A 1744-361': "red",
    '4U 1543-47': "gold",
    '4U 1630-47': "darkcyan",
    'Cen X-4': "#006400",
    'Cir X-1':  "chocolate", 
    'EXO 1846-031': "pink",
    'GRS 1739-278': "indianred", 
    "GRS 1915+105": "blue",
    'GX 339-4': "maroon",       
    'H1743-322': "orange",
    'IGR J17091-3624': "black",
    'MAXI J1348-630': "dodgerblue",
    'MAXI J1631-479': "mediumseagreen",
    'MAXI J1803-298': "olive",
    'MAXI J1807+132': "darkgoldenrod", 
    'MAXI J1810-222': "teal",
    "MAXI J1816-195":"cyan" , 
    'MAXI J1820+070': "magenta",
    'SAX J1808.4-3658': '#5ac2be',
    'SAX J1810.8-2609': "slategray",
    'Swift J1727.8-1613': "hotpink", 
    'Swift J1728.9-3613': "firebrick",   
    'Swift J1842.5-1124': "mediumorchid",
    'Swift J1858.6-0814': "lightgreen",
    'XTE J1701-462': "purple", 
    'Vela X-1': "#8000ff"
}

# Other colours: "darkblue", "crimson","mediumvioletred", "royalblue", 
# Define a shape set and loop through it if needed
shapes = [".", "*", "s", "d", "p", "x", "D", "h", "H"]
shapes_plotly = ["circle", "star", "square", "diamond", "cross", "x", "pentagon", "hexagon", "hexagon2"]

# Cycle through shapes to match length of colours
shapes = list(itertools.islice(itertools.cycle(shapes), len(colours)))
shapes_plotly = list(itertools.islice(itertools.cycle(shapes_plotly), len(colours)))

state_markers = {"HS": "o", "SS": "x", "IMS": "s", "QS": "*", "Unclear": "P"}
state_markersizes = {"HS": 2, "SS": 4, "IMS": 4, "QS": 6, "Unclear": 3}
state_markersizes_large = {"HS": 4, "SS": 6, "IMS": 6, "QS": 7, "Unclear": 6}
state_markersizes_extra_large = {"HS": 15, "SS": 30, "IMS": 20, "QS": 30, "Unclear": 30}


min_Lr = 1.5e25
max_Lr = 3e32 
max_Lr_2 = 2e31

min_Lx = 2e30
max_Lx = 3e39

colour_NS = "#0303D6"
colour_BH = "#D40404"



##############################################################################################################
## HELPER FUNCTIONS

def plot2mjd(t):
    '''Convert from matplotlib plot date to mjd'''
    return Time(t, format="plot_date", scale='utc').mjd


def mjd2plot(mjd):
    '''Convert from mjd to matplotlib plot'''
    return Time(mjd, format="mjd", scale='utc').plot_date

def mjd2utc(mjd):
    # Convert MJD to UTC using astropy
    t = Time(mjd, format='mjd', scale='utc')
    return t.iso  # Returns in ISO format (YYYY-MM-DD HH:MM:SS.sss)

def iso2mjd(iso_dates):
    # Convert ISO dates to MJD using astropy Time
    times = Time(iso_dates, format='isot', scale='utc')
    return times.mjd

def FormatAxis(ax, start_mjd, end_mjd, interval = 60):
    '''Function for putting UTC on top of axis'''

    ax[0].set_xlabel('Observing Date (UTC)', fontfamily='serif')    
    ax[0].xaxis.set_major_locator(mdates.DayLocator(interval=interval))
    ax[0].set_xlim(Time(start_mjd, format='mjd').datetime, Time(end_mjd, format='mjd').datetime)
    ax[0].xaxis.set_label_position('top') 
    xformatter = mdates.DateFormatter('%Y-%m-%d')
    plt.gcf().axes[0].xaxis.set_major_formatter(xformatter) #ax[0].xaxis.set_major_formatter(xformatter)
    ax[0].tick_params(axis='x', which='major',rotation=10,labeltop=True, labelbottom=False)

    # Format secondary x-axis
    mjd_ax = ax[-1].secondary_xaxis('bottom', functions=(plot2mjd, mjd2plot))
    mjd_ax.set_xlabel('Observing Date (MJD)', fontfamily='serif')  
    mjd_ax.tick_params(which='major', direction='in', length = 0.0, width = 0.0)
    plt.draw()

    # Extract the labels
    mjd_ticks = []
    labels = ax[0].get_xticklabels(which='major')
    for lab in labels:
        mjd_ticks.append(lab.get_text() + 'T00:00:00')

    # Line up MJD and Datetime labels 
    mjd_ticks = (Time(mjd_ticks, format='isot').mjd).astype(int)
    mjd_ax.set_xticks(mjd_ticks, labels = mjd_ticks)




##############################################################################################################
## PLOTTING LIGHT CURVES FOR INDIVIDUAL SOURCES


## Plotting unpaired flux data
## This is done after the data is checked and processed
def plot_xray_lightcurve(xray_data, start_mjd=0, end_mjd=70000, line_mjd = []):

    xray_MJDs = xray_data["t_xray"].to_numpy()
    mask = (xray_MJDs >= start_mjd) & (xray_MJDs < end_mjd) # plot only certain range of dates
    xray_MJDs = xray_MJDs[mask]
    Fx = xray_data["Fx"].to_numpy()[mask] 
    uplims_bool = xray_data["Fx_uplim_bool"].to_numpy()[mask] 
    Fx_unc_l = xray_data["Fx_unc_l"].to_numpy()[mask] 
    Fx_unc_u = xray_data["Fx_unc_u"].to_numpy()[mask] 

    fig=plt.figure(figsize=(13,5))
    ax = fig.add_subplot(1,1,1)
    plt.xlabel("Date [MJD]")
    plt.ylabel("Flux [erg/s/cm^2]")
    plt.errorbar(xray_MJDs[~uplims_bool], Fx[~uplims_bool], yerr=[Fx_unc_l[~uplims_bool], Fx_unc_u[~uplims_bool]], fmt="o", ms=4)
    plt.errorbar(xray_MJDs[uplims_bool], Fx[uplims_bool], yerr=[Fx_unc_l[uplims_bool], Fx_unc_u[uplims_bool]], fmt="v", ms=4)
    ax.set_yscale("log", base=10)
    if line_mjd!=[]: 
        for line in line_mjd: plt.axvline(x = line, color = 'r', alpha=0.5)
    plt.show()

    # Normal scale on the y-axis
    fig=plt.figure(figsize=(13,5))
    ax = fig.add_subplot(1,1,1)
    plt.xlabel("Date [MJD]")
    plt.ylabel("Flux [erg/s/cm^2]")
    plt.errorbar(xray_MJDs[~uplims_bool], Fx[~uplims_bool], yerr=[Fx_unc_l[~uplims_bool], Fx_unc_u[~uplims_bool]], fmt="o", ms=4)
    plt.errorbar(xray_MJDs[uplims_bool], Fx[uplims_bool], fmt="v", ms=4)
    if line_mjd!=[]: 
        for line in line_mjd: plt.axvline(x = line, color = 'r', alpha=0.5)
    plt.show()



## Plotting unpaired flux data
## This is done after the data is checked and processed
def plot_radio_lightcurve(radio_data, start_mjd=0, end_mjd=70000, line_mjd = []):

    radio_MJDs = radio_data["t_radio"].to_numpy()
    mask = (radio_MJDs >= start_mjd) & (radio_MJDs < end_mjd) # plot only certain range of dates
    radio_MJDs = radio_MJDs[mask]
    Fr = radio_data["Fr"].to_numpy()[mask] 
    uplims_bool = radio_data["Fr_uplim_bool"].to_numpy()[mask] 
    Fr_unc = radio_data["Fr_unc"].to_numpy()[mask] 

    fig=plt.figure(figsize=(13,5))
    ax = fig.add_subplot(1,1,1)
    plt.xlabel("Date [MJD]")
    plt.ylabel("Flux [mJy]")
    plt.errorbar(radio_MJDs[~uplims_bool], Fr[~uplims_bool], yerr=Fr_unc[~uplims_bool], fmt="o", ms=4)
    plt.errorbar(radio_MJDs[uplims_bool], Fr[uplims_bool], fmt="v", ms=4)
    ax.set_yscale("log", base=10)
    if line_mjd!=[]: 
        for line in line_mjd: plt.axvline(x = line, color = 'r', alpha=0.5)
    plt.show()

    # Normal scale on the y-axis
    fig=plt.figure(figsize=(13,5))
    ax = fig.add_subplot(1,1,1)
    plt.xlabel("Date [MJD]")
    plt.ylabel("Flux [mJy]")
    plt.errorbar(radio_MJDs[~uplims_bool], Fr[~uplims_bool], yerr=Fr_unc[~uplims_bool], fmt="o", ms=4)
    plt.errorbar(radio_MJDs[uplims_bool], Fr[uplims_bool], fmt="v", ms=4)
    if line_mjd!=[]: 
        for line in line_mjd: plt.axvline(x = line, color = 'r', alpha=0.5)
    plt.show()





##############################################################################################################
## PLOTTING LIGHT CURVES FOR ALL SOURCES


def plot_all_lightcurves(all_df, log=True, show_errorbars=False,highlight_name=None, save_name=None):
    """
    Plot flux and luminosity for all the data as a function of time. 

    all_df are either all the radio data or all the xray data
    """

    if "t_xray" in all_df.columns:
        xrays = True
    else: xrays = False

    fig_flux, ax_flux = plt.subplots(figsize=(20, 6))
    fig_lum, ax_lum = plt.subplots(figsize=(20, 6))

    names = np.unique(all_df["name"])
    source_classes = []
    
    for i, name in enumerate(names):

        if name== "GX 339-4": zorder = 5
        elif name == highlight_name: zorder = 4
        else: zorder = 1
        
        if highlight_name!=None and name != highlight_name: c = "grey"
        else: c = colours.get(name, 'black')

        filtered_data = all_df[all_df['name'] == name]
        source_class = filtered_data["class"].iloc[0]
        source_classes.append(source_class)

        if xrays:
            MJDs = filtered_data["t_xray"].to_numpy()
            uplims_bool = filtered_data["Fx_uplim_bool"].to_numpy()
            states = filtered_data["Xstate"].to_numpy()
            F = filtered_data["Fx"].to_numpy()
            F_unc_l = filtered_data["Fx_unc_l"].to_numpy()
            F_unc_u = filtered_data["Fx_unc_u"].to_numpy()
            L = filtered_data["Lx"].to_numpy()
            L_unc_l = filtered_data["Lx_unc_l"].to_numpy()
            L_unc_u = filtered_data["Lx_unc_u"].to_numpy()
            
        else: 
            MJDs = filtered_data["t_radio"].to_numpy()
            uplims_bool = filtered_data["Fr_uplim_bool"].to_numpy()
            states = filtered_data["Rstate"].to_numpy()
            F = filtered_data["Fr"].to_numpy()
            F_unc_l = filtered_data["Fr_unc"].to_numpy().copy()
            F_unc_u = filtered_data["Fr_unc"].to_numpy().copy()
            L = filtered_data["Lr"].to_numpy()
            L_unc_l = filtered_data["Lr_unc"].to_numpy().copy()
            L_unc_u = filtered_data["Lr_unc"].to_numpy().copy()
    
        dates = Time(MJDs, format='mjd').datetime

        ## Set uncertainties to zero if we don't want to show error bars
        if show_errorbars==False:
            F_unc_u[:], F_unc_l[~uplims_bool] = 0, 0
            L_unc_u[:], L_unc_l[~uplims_bool] = 0, 0

        ## Make the arrows for upper limits 1/4 of the value
        F_unc_u[uplims_bool], F_unc_l[uplims_bool] = 0, F[uplims_bool]/4 
        L_unc_u[uplims_bool], L_unc_l[uplims_bool] = 0, L[uplims_bool]/4 

            
        for state in np.unique(states):
            mask = (states == state) # True for the state being considered
            marker = state_markers.get(state, ".")  # default is '.' if state doesn't correspond to any of those defined above
            ms = state_markersizes_large.get(state, 3)

            ## Plot the luminosity plot    
            y, yunc_l, yunc_u = L, L_unc_l, L_unc_u
            plot, caps, bars = ax_lum.errorbar(dates[mask], y[mask], yerr=[yunc_l[mask], yunc_u[mask]], color=c, fmt=marker, ms=ms, uplims=uplims_bool[mask], capsize=0.5, ecolor="black", elinewidth=0.4, zorder=zorder)
            #for cap in caps:
            #    cap.set_color('black')      
            #    cap.set_markeredgewidth(0.2)  
            #    cap.set_markersize(3.5) 
            for bar in bars:
                bar.set_color('black')            

            ## Plot the flux plot
            y, yunc_l, yunc_u = F, F_unc_l, F_unc_u
            plot, caps, bars = ax_flux.errorbar(dates[mask], y[mask], yerr=[yunc_l[mask], yunc_u[mask]], color=c, fmt=marker, ms=ms, uplims=uplims_bool[mask], capsize=0.5, ecolor="black", elinewidth=0.4, zorder=zorder)
            #for cap in caps:
            #    cap.set_color('black')      
            #    cap.set_markeredgewidth(0.2)  
            #    cap.set_markersize(3.5) 
            for bar in bars:
                bar.set_color('black')            

    if log:
        ax_lum.set_yscale("log", base=10)
        ax_flux.set_yscale("log", base=10)

    if xrays: 
        ax_lum.set_ylabel(r"1–10 keV Unabsorbed X-ray Luminosity [erg s$^{-1}$]")
        ax_flux.set_ylabel(r"1–10 keV Unabsorbed X-ray Flux [erg s$^{-1}$ cm$^{-2}$]")
    else: 
        ax_lum.set_ylabel(r"1.28 GHz Radio Luminosity [erg s$^{-1}$]")
        ax_flux.set_ylabel("1.28 GHz Radio Flux [mJy]")

    # For the names array, replace "-" with "–"
    names_text = names.copy()
    names_text = [name.replace("-", "–") for name in names]
    # For the source_class array, replace "candidateBH" with "BH"
    source_classes = ["BH" if sc == "candidateBH" else sc for sc in source_classes]

    for ax in [ax_lum, ax_flux]:

        ax = np.atleast_1d(ax)

        # Format x-axes
        if xrays: all_mjd = all_df["t_xray"].to_numpy()
        else: all_mjd = all_df["t_radio"].to_numpy()
        min_mjd = np.min(all_mjd )
        max_mjd = np.max(all_mjd) 
        FormatAxis(ax, start_mjd = min_mjd-20, end_mjd = max_mjd+20, interval = 200)

        # y-axis limits
        if ax == ax_lum:
            if xrays: 
                min_y = min_Lx
                max_y = max_Lx
            else:
                min_y = min_Lr
                max_y = max_Lr
            ax[0].set_ylim(min_y, max_y)


        # Create state legend (within plot) in black
        state_legend_handles = [plt.Line2D([0], [0], marker=marker, color='black', linestyle='None', markersize=6, label=state) for state, marker in state_markers.items()]
        state_legend = ax[0].legend(handles=state_legend_handles, loc="upper right", title="States", fontsize=10)
        ax[0].add_artist(state_legend)  

        
        # Create source legend (at bottom) with dots
        if highlight_name:
            source_legend_handles = [
            plt.Line2D(
                [0], [0],
                marker='o',
                color=colours.get(name, 'grey') if name == highlight_name else 'grey',
                linestyle='None',
                markersize=8,
                label=f"{name_text} ({source_class})",
            )
            for (name, name_text, source_class) in zip(names, names_text, source_classes)
            ]
        else: source_legend_handles = [plt.Line2D([0], [0], marker='o', color=colours.get(name, 'black'), linestyle='None', markersize=8, label=f"{name_text} ({source_class})") for (name, name_text, source_class) in zip(names, names_text, source_classes)] 
        ax[0].legend(loc="upper center", handles=source_legend_handles, bbox_to_anchor=(0.5, -0.1), ncol=int(np.ceil(len(names) / 5)), title="Sources",columnspacing=0.6,handletextpad=0.05,labelspacing=0.3,borderpad=0.3)
    
    if save_name!=None: 
        fig_lum.savefig(f"../FIGURES/{save_name}_lum.png", dpi=600,bbox_inches="tight")
        fig_flux.savefig(f"../FIGURES/{save_name}_flux.png", dpi=600,bbox_inches="tight")

        # Also save as a pdf
        fig_lum.savefig(f"../FIGURES/{save_name}_lum.pdf", dpi=600,bbox_inches="tight")
        fig_flux.savefig(f"../FIGURES/{save_name}_flux.pdf", dpi=600,bbox_inches="tight")

    plt.show()





##############################################################################################################
## PLOTTING HISTOGRAMS FOR ALL SOURCES


def add_arrows_to_uplims_region(ax, det_bars, uplims_bars, arrow_color="grey"):
    """
    Helper function to add fixed-size arrows to only the uplims portion of stacked bars in a histogram.
    """

    # Fixed arrow dimensions (height scaled to y-axis range)
    y_min, y_max = ax.get_ylim()
    arrow_height = (y_max - y_min) * 0.045  # 4.5% of y-axis range
    arrow_width_ratio = 0.5  # As fraction of bar width
    
    for i, (det_bar, uplims_bar) in enumerate(zip(det_bars, uplims_bars)):
        # Get bar dimensions
        bar_x = det_bar.get_x()
        bar_width = det_bar.get_width()
        
        # Calculate the heights
        det_height = det_bar.get_height()
        uplims_height = uplims_bar.get_height()
        
        # Skip if no uplims data in this bin
        if uplims_height <= 0:
            continue
        
        # The uplims region starts at det_height and goes to det_height + uplims_height
        # Add small padding from the edges
        edge_padding = arrow_height * 0.12  # 12% of arrow height as padding
        uplims_bottom = det_height + edge_padding
        uplims_top = det_height + uplims_height - edge_padding
        uplims_region_height = uplims_top - uplims_bottom
        
        # Skip if the uplims region is too small to fit even one arrow
        if uplims_region_height < arrow_height:
            continue
        
        # Calculate arrow width
        arrow_width = bar_width * arrow_width_ratio
        
        # Calculate how many arrows fit vertically with some padding
        padding = arrow_height * 0.1  # Small padding between arrows
        effective_arrow_height = arrow_height + padding
        
        # Number of arrows that fit in the uplims region
        n_arrows = int(uplims_region_height / effective_arrow_height)
        if n_arrows < 1:
            n_arrows = 1
        
        # Center the arrows horizontally in the bar
        arrow_x_center = bar_x + bar_width / 2
        
        # Calculate y positions for arrows, centered in the uplims region
        if n_arrows == 1:
            y_positions = [uplims_bottom + uplims_region_height / 2]
        else:
            # Evenly space arrows in the uplims region
            y_start = uplims_bottom + arrow_height / 2
            y_end = uplims_top - arrow_height / 2
            y_positions = np.linspace(y_start, y_end, n_arrows)
        
        for y_pos in y_positions:
            # Create left-pointing arrow shape
            arrow_tip_x = arrow_x_center - arrow_width / 2
            arrow_base_x = arrow_x_center + arrow_width / 2
            
            # Define arrow vertices
            arrow_vertices = [
                (arrow_tip_x, y_pos),  # Tip (leftmost point)
                (arrow_base_x, y_pos + arrow_height/2),  # Top right
                (arrow_base_x, y_pos + arrow_height/4),  # Top base right
                (arrow_base_x, y_pos - arrow_height/4),  # Bottom base right
                (arrow_base_x, y_pos - arrow_height/2),  # Bottom right
            ]
            
            # Create and add the filled arrow polygon
            arrow = Polygon(arrow_vertices, facecolor=arrow_color, edgecolor='black', linewidth=0.5, alpha=0.1)
            ax.add_patch(arrow)





def plot_luminosity_histogram_stacked(all_df, n_bins=20,  save_name=None):
    """
    Plot histograms of the flux/luminosity distributions. 
    """   
    medium_grey = (0.55, 0.55, 0.55, 0.4) # medium grey

    types = all_df["class"].unique().tolist()

    if "t_xray" in all_df.columns: 
        xrays = True
        states = all_df["Xstate"].unique().tolist()
        uplims_bool = all_df["Fx_uplim_bool"].to_numpy()
        F = all_df["Fx"].to_numpy()
        L = all_df["Lx"].to_numpy()
        min_x = min_Lx
        max_x = max_Lx
    else: 
        xrays = False
        states = all_df["Rstate"].unique().tolist()
        uplims_bool = all_df["Fr_uplim_bool"].to_numpy()
        F = all_df["Fr"].to_numpy()
        L = all_df["Lr"].to_numpy()
        min_x = min_Lr
        max_x = max_Lr
    

    F_det = F[~uplims_bool]
    F_uplim = F[uplims_bool]
    L_det = L[~uplims_bool]
    L_uplim = L[uplims_bool]

    mean_L_det = np.mean(L_det) if L_det.size > 0 else np.nan
    mean_L_uplim = np.mean(L_uplim) if L_uplim.size > 0 else np.nan
    median_L_det = np.median(L_det) if L_det.size > 0 else np.nan
    median_L_uplim = np.median(L_uplim) if L_uplim.size > 0 else np.nan
    print("Mean Ldet: ", mean_L_det )
    print("Mean Luplim: ", mean_L_uplim)
    print("Median Ldet: ", median_L_det)
    print("Median Luplim: ", median_L_uplim)

    ## Create labels with median values for the legend
    label = []
    label.append(f"Detections\nmedian = {median_L_det:.2e}".replace('e+', ' × 10$^{').replace('e-', ' × 10$^{-') + "}$ erg s$^{-1}$")
    label.append(f"Upper limits\nmedian = {median_L_uplim:.2e}".replace('e+', ' × 10$^{').replace('e-', ' × 10$^{-') + "}$ erg s$^{-1}$") 



    ## Make the plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))  
    
    ## Create logarithmic bins
    log_bins = np.logspace(np.log10(min_x), np.log10(max_x), n_bins + 1)

    ## Stack the histogram
    counts, bins, patches = ax.hist(
        [L_det, L_uplim],
        bins=log_bins,
        stacked=True,
        color=[medium_grey, 'white'], 
        edgecolor='black',
        label= label
    )

    ## Add arrows to uplims region only
    if len(patches) > 1:  add_arrows_to_uplims_region(ax, patches[0], patches[1], arrow_color=medium_grey)


    ## Create source type legend
    # Replace "candidateBH" with "candidate BH"
    types = ["candidate BH" if typ == "candidateBH" else typ for typ in types]
    type_legend_handles = [plt.Line2D([0], [0], color='none', linestyle='None', markersize=1, marker=".",  label=typ) for typ in types]
    phantom = plt.Line2D([0], [0], color='none', label='\u200A' * 65) 
    type_legend_handles.append(phantom)
    type_legend = ax.legend(handles=type_legend_handles, loc="upper left", title="Types", handlelength=0, fontsize=10)
    ax.add_artist(type_legend)

    ## Create state legend (within plot) in black
    state_legend_handles = [plt.Line2D([0], [0], color='none', linestyle='None', markersize=1, marker=".", label=state) for state in states] 
    phantom = plt.Line2D([0], [0], color='none', label='\u200A' * 48)  
    state_legend_handles.append(phantom)
    state_legend = ax.legend(handles=state_legend_handles, loc="upper left",bbox_to_anchor=(0.175, 1.0), title="States", handlelength=0, fontsize=10)
    ax.add_artist(state_legend)  


    ax.set_xscale('log')
    if xrays: ax.set_xlabel(r'1–10 keV Unabsorbed Luminosity [erg s$^{-1}$]')
    else: ax.set_xlabel(r'1.28 GHz Radio Luminosity [erg s$^{-1}$]')
    ax.set_ylabel('Frequency')
    ax.legend(fontsize=10, loc="upper right")
    plt.xlim([min_x, max_x])
    plt.tight_layout()

    if save_name!=None: 
        plt.savefig(f"../FIGURES/{save_name}.png", dpi=600)
        plt.savefig(f"../FIGURES/{save_name}.pdf", dpi=600)
    
    plt.show()




##############################################################################################################
## PLOTTING LRLX PLANE


## Plotting the LrLx data
# colourby examples: "state", "source_name: MAXI J1348-630","alpha"
# mew: marker edge width
# mfc: marker face colour
# mec: marker edge colour
# elinewidth: edge line width (i.e. the error bars)
def plot_Lr_Lx(paired_data, colourby="", states=["HS", "IMS", "SS", "QS", "Unclear"], types=["BH", "candidateBH", "NS", "NS HMXRB"], lx=np.array([]), lr=np.array([])):

    mask = paired_data["state"].isin(states) & paired_data["class"].isin(types)

    x = paired_data["Lx"][mask]
    xerr =[paired_data["Lx_unc_l"][mask], paired_data["Lx_unc_u"][mask]] # x-axis errors
    y = paired_data["Lr"][mask]
    yerr = paired_data["Lr_unc"][mask] # y-axis errors

    # Boolean arrays to specify directional limits for each data point
    uplims = paired_data["Fr_uplim_bool"][mask]    # Upper y-limit arrow (down)
    xuplims = paired_data["Fx_uplim_bool"][mask]   # Right x-limit arrow (down)

    fig= plt.figure(figsize=(9,6))
    ax = fig.add_subplot(1,1,1)

    # Plot additional data
    if lr.size > 0 and lx.size > 0 and len(lr) == len(lx): plt.plot(lx, lr, '.', color="grey", ms=2, label = "Rescaled Bahramian & Rushton 2022 data")
    #plt.ylim([float(10**(25)), float(2*10**(31))])
    #plt.xlim([float(10**(30)), float(10**(39))])

    plot, caps, bars = plt.errorbar(x, y, yerr=yerr, xerr=xerr, fmt='.', ms=0.05, mec='blue', mfc='blue', uplims=uplims,  xuplims=xuplims, capsize=0.5, ecolor="black", elinewidth=0.4, zorder=3)
    for cap in caps:
        cap.set_color('black')      # Set cap color
        cap.set_markeredgewidth(0.2)  # Set edge width
        cap.set_markersize(3) 
    for bar in bars:
        bar.set_color('black')

    if colourby in ["class", "Xphase", "Rphase", "state", "name"]:  # discrete values
        colourby_values = paired_data[colourby][mask].unique()
        for i, colourby_value in enumerate(colourby_values):
            colour_mask = (paired_data[colourby][mask] == colourby_value) # Create a mask for the current 'colourby' value
            size=35
            if shapes[i]=="s": size=20
            if colourby_value=="candidateBH": colourby_value = "candidate BH"
            plt.scatter(x[colour_mask], y[colour_mask], color=colours_random[i], label=colourby_value, s=size, marker=shapes[i])
        if colourby=="name": plt.legend(fontsize=9, loc='center left', bbox_to_anchor=(1, 0.5))
        else: plt.legend(fontsize=9)
            

    elif colourby in ["alpha", "t"]: # continuous values

        dict = {"alpha": "spectral index", "t": "time"}

        values = paired_data[colourby][mask]
        if values.isna().all():
            print("No data to colour by.")
            plt.close()
            return
        sc = plt.scatter(x, y, c=values,cmap="viridis", s=50, marker=".")
        cbar = plt.colorbar(sc, ax=ax, label=dict[colourby], pad=0.02, location='right')
        values_min, values_max = values.min(), values.max()
        cbar.set_ticks([values_min, (values_min + values_max) / 2, values_max])
        cbar.set_ticklabels([f"{values_min:.2f}", f"{(values_min + values_max) / 2:.2f}", f"{values_max:.2f}"])
        cbar.ax.tick_params(labelsize=9)


    # Code to highlight a particular source -- colourby = "name: "
    elif colourby.startswith("source_name"): 
        name = colourby.split(":")[1].strip()
        mask_names = (paired_data["name"][mask]==name)
        plt.scatter(x[mask_names], y[mask_names], color="red", marker="o", label=name, s=15)
        plt.legend(fontsize=9)

    else:
        plt.scatter(x, y, color="blue", marker=".", s=15)


    plt.xlabel('1-10 keV X-ray luminosity [erg/s]')
    plt.ylabel('1.28 GHz radio luminosity [erg/s]')
    ax.set_yscale("log", base=10)
    ax.set_xscale("log", base=10)
    ax.xaxis.set_major_locator(plt.LogLocator(base=10.0, numticks=10))
    ax.xaxis.set_minor_locator(plt.LogLocator(base=10.0, subs="auto", numticks=10))
    plt.show()



def plot_Lr_Lx_plotly(paired_data, colourby="", states=["HS", "IMS", "SS", "QS", "unclear"], types=["BH", "candidateBH", "NS", "candidateNS"]):

    mask = paired_data["state"].isin(states) & paired_data["class"].isin(types)
    x = paired_data["Lx"][mask]
    xerr_lower = paired_data["Lx_unc_l"][mask]
    xerr_upper = paired_data["Lx_unc_u"][mask]
    y = paired_data["Lr"][mask]
    yerr = paired_data["Lr_unc"][mask]
    
    uplims = paired_data["Fr_uplim_bool"][mask]
    xuplims = paired_data["Fx_uplim_bool"][mask]

    fig = go.Figure()

    # Plot error bars on detections (no limits)
    
    detections = ~uplims | ~xuplims  # Points to plot: either x detections, y detections, or both
    fig.add_trace(go.Scatter(
        x=x[detections],
        y=y[detections],
        error_x=dict(
            type='data',
            symmetric=False,
            array=np.where(~xuplims[detections], xerr_upper[detections], 0),
            arrayminus=np.where(~xuplims[detections], xerr_lower[detections], 0),
            color='black',
            visible=True  # Ensure x error bars are visible only when applicable
        ) ,  
        error_y=dict(
            type='data',
            array=np.where(~uplims[detections], yerr[detections], 0),
            color='black',
            visible=True  # Ensure y error bars are visible only when applicable
        ) ,  
        mode='markers',
        marker=dict(size=0, color='blue'),
        showlegend=False,
        hoverinfo="skip"  # Don't show hover info
    ))


    # Plot error bars with arrowheads for the upper limits 
    for i in range(len(x)):

        if uplims[i]:  # Y-axis upper limit
            fig.add_trace(go.Scatter(
                x=[x[i]],  # Use x position as is
                y=[y[i]],  # Use y position
                error_y=dict(
                    type='data',
                    arrayminus=[yerr[i]],
                    visible=True, 
                    width =0, # no cap
                    color='black'
                ),
                mode='markers',
                marker=dict(size=0, color='blue'),
                showlegend=False, 
                hoverinfo="skip" # don't show hover info
            ))
            # Add a small arrowhead at the end of the error bar
            fig.add_trace(go.Scatter(
                x=[x[i]],
                y=[y[i] - yerr[i]],  # Position arrow at the top of the error bar
                mode='markers',
                marker=dict(size=5, color='black', symbol='triangle-down'),
                showlegend=False, 
                hoverinfo="skip" # don't show hover info
            ))

        if xuplims[i]:  # X-axis upper limit
            fig.add_trace(go.Scatter(
                x=[x[i]],  # Use x position
                y=[y[i]],  # Use y position as is
                error_x=dict(
                    type='data',
                    arrayminus=[xerr_lower[i]],
                    visible=True,
                    width=0, # no cap
                    color='black'
                ),
                mode='markers',
                marker=dict(size=0, color='blue'),
                showlegend=False,
                hoverinfo="skip" # don't show hover info
            ))
            # Add a small arrowhead at the end of the error bar
            fig.add_trace(go.Scatter(
                x=[x[i] - xerr_lower[i]],  # Position arrow at the end of the error bar
                y=[y[i]],
                mode='markers',
                marker=dict(size=5, color='black', symbol='triangle-left'),
                showlegend=False,
                hoverinfo="skip" # don't show hover info
            ))


    # Colour by categories
    if colourby in ["class", "Xphase", "Rphase", "state", "name"]:  # discrete values
        unique_values = paired_data[colourby][mask].unique()
        for i, value in enumerate(unique_values):
            category_mask = paired_data[colourby][mask] == value
            fig.add_trace(go.Scatter(
                x=x[category_mask],
                y=y[category_mask],
                mode='markers',
                marker=dict(size=7.5, color=colours_random[i], symbol=shapes_plotly[i]),
                name=str(value)
            ))

    # Colour by continuous values
    elif colourby in ["alpha"]: 
        values = paired_data[colourby][mask]
        if values.isna().all():
            print("No data to colour by.")
            return
        fig.add_trace(go.Scatter(
            x=x,
            y=y,
            mode='markers',
            marker=dict(size=7.5, color=values, colorscale='Viridis', showscale=True),
            name=f"Coloured by {colourby}",
            hoverinfo='x+y+text',  # Include x, y, and text in the hover
            text=values  # Display values as text in the hover
        ))

    # Highlight specific name
    elif colourby.startswith("name"): 
        name = colourby.split(":")[1].strip()
        highlight_mask = paired_data["name"] == name
        fig.add_trace(go.Scatter(
            x=x[highlight_mask],
            y=y[highlight_mask],
            mode='markers',
            marker=dict(size=8, color='red', symbol='circle'),
            name=f'Highlight: {name}', 
        ))

    # Default scatter
    else:
        fig.add_trace(go.Scatter(
            x=x,
            y=y,
            mode='markers',
            marker=dict(size=6, color='blue')
        ))
    

    fig.update_layout(
        xaxis=dict(title='1-10 keV X-ray luminosity [erg/s]', type='log',),
        yaxis=dict(title='1.28 GHz radio luminosity [erg/s]', type='log'),
        template='plotly_white',
        width=1200,
        height=700
    )

    fig.show()




##############################################################################################################
## PLOTTING LRLX TIME EVOLUTION ANIMATION


def plot_time_evolution_timed(paired_data):
    data = paired_data.sort_values("t").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$L_X$ (erg/s)")
    ax.set_ylabel(r"$L_R$ (erg/s)")

    x_min, x_max = data["Lx"].min(), data["Lx"].max()
    y_min, y_max = data["Lr"].min(), data["Lr"].max()
    ax.set_xlim(x_min * 0.8, x_max * 1.2)
    ax.set_ylim(y_min * 0.8, y_max * 1.2)

    norm = Normalize(vmin=data["t"].min(), vmax=data["t"].max())
    scat = ax.scatter([], [], c=[], cmap="viridis", norm=norm, s=20)

    t = data["t"].to_numpy()
    dt = np.diff(t, prepend=t[0])  # dt[0]=0
    slowing_factor = 2.5
    # Convert dt to milliseconds with scaling; protect against all-zero dt
    dt_max = dt.max() if dt.max() > 0 else 1.0
    intervals_ms = (dt / dt_max) * 1000 * slowing_factor
    intervals_ms[0] = intervals_ms[1] if len(intervals_ms) > 1 else 200  # nicer first step

    def update(frame):
        # show points up to and including frame
        shown = data.iloc[: frame + 1]

        scat.set_offsets(shown[["Lx", "Lr"]].to_numpy())
        scat.set_array(shown["t"].to_numpy())

        ax.set_title(f"Time: {shown['t'].iloc[-1]:.2f}")

        # THIS is what makes timing proportional to time:
        ani.event_source.interval = float(intervals_ms[frame])

        return (scat,)

    ani = FuncAnimation(
        fig,
        update,
        frames=len(data),
        interval=float(intervals_ms[0]),
        repeat=False,
        blit=False,
    )

    plt.close(fig) 

    
    return HTML(ani.to_jshtml())


##############################################################################################################
## LRLX PLOTS FOR PAPER


def plot_Lr_Lx_plot1(paired_data, states=["HS", "IMS", "SS", "QS", "Unclear"], types=["BH", "candidateBH", "NS"], show_error_bars=False, save_name=None):

    mask = paired_data["state"].isin(states) & paired_data["class"].isin(types)

    paired_data_filtered = paired_data[mask].copy()

    x = paired_data_filtered["Lx"]
    xerr =[paired_data_filtered["Lx_unc_l"].copy(), paired_data_filtered["Lx_unc_u"].copy()] # x-axis errors
    y = paired_data_filtered["Lr"]
    yerr = paired_data_filtered["Lr_unc"].copy() # y-axis errors

    # Boolean arrays to specify directional limits for each data point
    uplims = paired_data_filtered["Fr_uplim_bool"]    # Upper y-limit arrow (down)
    xuplims = paired_data_filtered["Fx_uplim_bool"]  # Right x-limit arrow (down)

    if show_error_bars==False:
        xerr[0][~xuplims] = np.nan
        xerr[1][~xuplims] = np.nan
        yerr[~uplims] = np.nan

    
    unique_names = paired_data_filtered["name"].unique() 
    # Get the indices corresponding to unique names
    n_names = len(unique_names)
    # Get the source classes for the unique names
    source_classes = [paired_data_filtered[paired_data_filtered["name"] == name]["class"].iloc[0] for name in unique_names]

    fig= plt.figure(figsize=(9,6))
    ax = fig.add_subplot(1,1,1)

    plot, caps, bars = plt.errorbar(x, y, yerr=yerr, xerr=xerr, fmt='.', ms=0.05, mec='blue', mfc='blue', uplims=uplims,  xuplims=xuplims, capsize=0.5, ecolor="black", elinewidth=0.4)
    for cap in caps:
        cap.set_color('black')      # Set cap color
        cap.set_markeredgewidth(0.2)  # Set edge width
        cap.set_markersize(3) 
    for bar in bars:
        bar.set_color('black')


    # Loop over classes and names
    for state in paired_data_filtered["state"].unique():
    
        data = paired_data_filtered[paired_data_filtered["state"] == state]
        marker = state_markers.get(state, '.')
        size = state_markersizes_extra_large.get(state, 3)
        
        for name in data["name"].unique():
            subset = data[data["name"] == name]
            if subset.empty: continue
            c = colours.get(name, 'black')
            cls = subset["class"].iloc[0]
            
            ax.scatter(
                subset["Lx"], subset["Lr"],
                label=f"{name} ({cls})",
                marker=marker,
                s=size,
                color=c, zorder=3
            )

    # Create source type legend
    types = ["candidate BH" if typ == "candidateBH" else typ for typ in types]
    type_legend_handles = [plt.Line2D([0], [0], color='none', linestyle='None', markersize=1, marker=".",  label=typ) for typ in types]
    phantom = plt.Line2D([0], [0], color='none', label='\u200A' * 65) 
    type_legend_handles.append(phantom)
    type_legend = ax.legend(handles=type_legend_handles, loc="upper left", title="Types", handlelength=0, fontsize=10)
    ax.add_artist(type_legend)



    # Create state legend (within plot) in black
    markers = [state_markers.get(state, '.') for state in states]
    state_legend_handles = [plt.Line2D([0], [0], marker=marker, color='black', linestyle='None', markersize=6, label=state) for state, marker in zip(states,markers)] 
    phantom = plt.Line2D([0], [0], color='none', label='\u200A' * 48)  
    state_legend_handles.append(phantom)
    state_legend = ax.legend(handles=state_legend_handles, loc="upper left",bbox_to_anchor=(0.18, 1.0), title="States", fontsize=10)
    ax.add_artist(state_legend)  

    
    # For the names array, replace "-" with "–"
    names_text = unique_names.copy()
    names_text = [name.replace("-", "–") for name in unique_names]
    # For the source_class array, replace "candidateBH" with "BH"
    source_classes = ["BH" if sc == "candidateBH" else sc for sc in source_classes]


    # Create source legend (at bottom) with dots
    source_legend_handles = [plt.Line2D([0], [0], marker='o', color=colours.get(name, 'black'), linestyle='None', markersize=8, label=f"{name_text} ({source_class})") for (name, name_text, source_class) in zip(unique_names, names_text, source_classes)] 
    ncol=4
    leg = plt.legend(loc="upper center", handles=source_legend_handles, bbox_to_anchor=(0.45, -0.12), ncol=ncol, title="Sources", fontsize=10,columnspacing=0.6,handletextpad=0.05,labelspacing=0.3,borderpad=0.3)

    plt.xlim([min_Lx,max_Lx])
    plt.ylim([min_Lr,max_Lr_2])

    #plt.xlabel(r'1–10 keV Unabsorbed X-ray Luminosity [erg s$^{-1}$]')
    #plt.ylabel(r'1.28 GHz Radio Luminosity [erg s$^{-1}$]')
    plt.xlabel(r'$L_X$ [erg s$^{-1}$]')
    plt.ylabel(r'$L_R$ [erg s$^{-1}$]')
    ax.set_yscale("log", base=10)
    ax.set_xscale("log", base=10)
    ax.xaxis.set_major_locator(plt.LogLocator(base=10.0, numticks=10))
    ax.xaxis.set_minor_locator(plt.LogLocator(base=10.0, subs="auto", numticks=10))


    if save_name!=None: 
        plt.savefig(f"../FIGURES/{save_name}.png", dpi=600, bbox_inches="tight", bbox_extra_artists=[leg])
        plt.savefig(f"../FIGURES/{save_name}.pdf", dpi=600, bbox_inches="tight", bbox_extra_artists=[leg])

    plt.show()




    
 

# Just HS and QS
# Colour by class
# Versus Bahramian data
def plot_Lr_Lx_plot2(paired_data, show_bahramian = True, save_name=None):

    states = ["HS", "QS"]
    mask = paired_data["state"].isin(states) 

    paired_data_filtered = paired_data[mask].copy()

    x = paired_data_filtered["Lx"]
    xerr =[paired_data_filtered["Lx_unc_l"], paired_data_filtered["Lx_unc_u"]] # x-axis errors
    y = paired_data_filtered["Lr"]
    yerr = paired_data_filtered["Lr_unc"] # y-axis errors

    # Boolean arrays to specify directional limits for each data point
    uplims = paired_data_filtered["Fr_uplim_bool"]    # Upper y-limit arrow (down)
    xuplims = paired_data_filtered["Fx_uplim_bool"]  # Right x-limit arrow (down)


    fig= plt.figure(figsize=(9,6))
    ax = fig.add_subplot(1,1,1)

    plot, caps, bars = plt.errorbar(x, y, yerr=yerr, xerr=xerr, fmt='.', ms=0.05, mec='blue', mfc='blue', uplims=uplims,  xuplims=xuplims, capsize=0.5, ecolor="black", elinewidth=0.4)
    for cap in caps:
        cap.set_color('black')      # Set cap color
        cap.set_markeredgewidth(0.2)  # Set edge width
        cap.set_markersize(3) 
    for bar in bars:
        bar.set_color('black')


    # Get Bahramian data
    lr_bah, lx_bah, source_classes_bah = get_bahramian_data()
    
    colours = ["#D40404", "#a10000ff", "#0303D6"]
    types = ["BH", "candidateBH", "NS"]
    for i, cls in enumerate(types):
        mask = (paired_data_filtered["class"] == cls) 
        mask_bah = (source_classes_bah == cls) 
        plt.scatter(x[mask], y[mask], color=colours[i], s=25, marker="o", zorder=3)
        if show_bahramian: plt.scatter(lx_bah[mask_bah], lr_bah[mask_bah], color=colours[i], s=7, marker=".")
    
    plt.xlim([min_Lx,max_Lx])
    plt.ylim([min_Lr,max_Lr_2])


    if show_bahramian:
        legend_handles = [plt.Line2D([0], [0], marker=marker, color='black', linestyle='None', markersize=6, label=data_type) for data_type, marker in zip(["Our data", "Rescaled Bahramian & Rushton 2022 detections"],["o", "."])]
        legend = ax.legend(handles=legend_handles, loc="lower right", fontsize=10)
        ax.add_artist(legend) 



    # Create source type legend
    types = ["candidate BH" if typ == "candidateBH" else typ for typ in types]
    type_legend_handles = [plt.Line2D([0], [0], color=colour, linestyle='None', markersize=6, label=typ, marker="o") for typ, colour in zip(types, colours)]
    phantom = plt.Line2D([0], [0], color='none', label='\u200A' * 65) 
    type_legend_handles.append(phantom)
    type_legend = ax.legend(handles=type_legend_handles, loc="upper left", title="Types", fontsize=10)
    ax.add_artist(type_legend)


    # Create state legend (within plot) in black
    state_legend_handles = [plt.Line2D([0], [0], color='none', linestyle='None', markersize=1, marker=".", label=state) for state in states] 
    phantom = plt.Line2D([0], [0], color='none', label='\u200A' * 48)  
    state_legend_handles.append(phantom)
    state_legend = ax.legend(handles=state_legend_handles, loc="upper left",bbox_to_anchor=(0.22, 1.0), title="States", handlelength=0, fontsize=10)
    ax.add_artist(state_legend)  


    #plt.xlabel(r'1–10 keV Unabsorbed X-ray Luminosity [erg s$^{-1}$]')
    #plt.ylabel(r'1.28 GHz Radio Luminosity [erg s$^{-1}$]')
    plt.xlabel(r'$L_X$ [erg s$^{-1}$]')
    plt.ylabel(r'$L_R$ [erg s$^{-1}$]')
    ax.set_yscale("log", base=10)
    ax.set_xscale("log", base=10)
    ax.xaxis.set_major_locator(plt.LogLocator(base=10.0, numticks=10))
    ax.xaxis.set_minor_locator(plt.LogLocator(base=10.0, subs="auto", numticks=10))


    if save_name!=None: 
        plt.savefig(f"../FIGURES/{save_name}.png", dpi=600, bbox_inches="tight")
        plt.savefig(f"../FIGURES/{save_name}.pdf", dpi=600, bbox_inches="tight")

    plt.show()





# Highlight a particular source
# Show as dots all the other observations of that source type
def plot_Lr_Lx_plot3(paired_data, source_name= "MAXI J1348-630", save_name=None, show_standard_track= False):


    ## SOURCE DATA

    states = ["HS", "QS"]
    mask = (paired_data["name"] == source_name) & (paired_data["state"].isin(states))

    paired_data_filtered = paired_data[mask].copy()

    x = paired_data_filtered["Lx"]
    xerr =[paired_data_filtered["Lx_unc_l"], paired_data_filtered["Lx_unc_u"]] # x-axis errors
    y = paired_data_filtered["Lr"]
    yerr = paired_data_filtered["Lr_unc"] # y-axis errors

    dist = paired_data_filtered["D"].to_numpy()[0]

    # Boolean arrays to specify directional limits for each data point
    uplims = paired_data_filtered["Fr_uplim_bool"]    # Upper y-limit arrow (down)
    xuplims = paired_data_filtered["Fx_uplim_bool"]  # Right x-limit arrow (down)


    ## OTHER DATA
    ## TODO: Make class functionality more general

    # types = paired_data_filtered["class"].iloc[0]
    types = ["BH", "candidateBH"]
    mask = (paired_data["class"].isin(types)) & (paired_data["state"].isin(states))
    other_data_filtered = paired_data[mask].copy()
    x_other = other_data_filtered["Lx"]
    y_other = other_data_filtered["Lr"]

    colour = colours.get(source_name, 'black')



    fig= plt.figure(figsize=(9,6))
    ax = fig.add_subplot(1,1,1)

    plot, caps, bars = plt.errorbar(x, y, yerr=yerr, xerr=xerr, fmt='o', ms=5, mec=colour, mfc=colour, uplims=uplims,  xuplims=xuplims, capsize=0.5, ecolor="black", elinewidth=0.4, zorder=100)
    for cap in caps:
        cap.set_color('black')      # Set cap color
        cap.set_markeredgewidth(0.2)  # Set edge width
        cap.set_markersize(3) 
    for bar in bars:
        bar.set_color('black')


    plt.scatter(x_other, y_other, s=7, marker=".",color="grey")


    # Create source type legend
    types = ["candidate BH" if typ == "candidateBH" else typ for typ in types]
    type_legend_handles = [plt.Line2D([0], [0], color='none', linestyle='None', markersize=1, marker=".",  label=typ) for typ in types]
    phantom = plt.Line2D([0], [0], color='none', label='\u200A' * 65) 
    type_legend_handles.append(phantom)
    type_legend = ax.legend(handles=type_legend_handles, loc="upper left", title="Types", handlelength=0, fontsize=10)
    ax.add_artist(type_legend)



    # Create state legend (within plot) in black
    states = paired_data_filtered["state"].unique() 
    markers = [state_markers.get(state, '.') for state in states]
    state_legend_handles = [plt.Line2D([0], [0], marker=marker, color='black', linestyle='None', markersize=6, label=state) for state, marker in zip(states,markers)] 
    phantom = plt.Line2D([0], [0], color='none', label='\u200A' * 48)  
    state_legend_handles.append(phantom)
    state_legend = ax.legend(handles=state_legend_handles, loc="upper left",bbox_to_anchor=(0.18, 1.0), title="States", fontsize=10)
    ax.add_artist(state_legend)  



    # Filter GX339 to get the region of interest
    t0 = 58964
    t1 = 59083
    lx = paired_data["Lx"]
    t = paired_data["t"]    
    lr = paired_data["Lr"]
    lx_delta = paired_data["Fx_uplim_bool"]
    lr_delta = paired_data["Fr_uplim_bool"]
    states = paired_data["state"]
    gx339_mask = (paired_data["name"]=="GX 339-4") & (lx > 2.7e34) & ( (t<t0) | (t>t1) ) & (lx_delta==False) & (states.isin(["HS", "QS"]))
    lr_gx339, lx_gx339,  delta_gx339 = lr[gx339_mask], lx[gx339_mask], lr_delta[gx339_mask]
    gx339_color = colours.get("GX 339-4", 'black')
    # Standard track fit parameters
    lx0 = 5.205841778483798e+35
    lr0 = 5.141600494192072e+28
    alpha = 0.336
    beta = 0.583
    model_axis = np.linspace(min_Lx, max_Lx, 100) # L_X values
    if show_standard_track:
        # The best fit:
        casefit = lr0 * (10**(alpha)) * ((model_axis/lx0)**(beta)) 
        ax.errorbar(model_axis, casefit, fmt='--', color='black', lw=1.5)
        ax.errorbar(lx_gx339, lr_gx339, fmt='o', ms=3.5, mec='none', mfc=gx339_color, uplims=delta_gx339, capsize=0.5, ecolor="black", elinewidth=0.4, zorder=8)



    # Create source legend (at bottom) with dots
    ##TODO: Make surce type labelling more general
    source_name_text = source_name.replace("-", "–") 
    if show_standard_track: legend_handles = [plt.Line2D([0], [0], marker=marker, color=c, linestyle='None', markersize=6, label=data_type) for data_type, marker, c in zip([f"{source_name_text} ({dist} kpc)", "Other BH / candidateBH", "GX 339–4 'standard' track"],["o", ".", "."], [colour, "grey", gx339_color])]
    else: legend_handles = [plt.Line2D([0], [0], marker=marker, color=c, linestyle='None', markersize=6, label=data_type) for data_type, marker, c in zip([f"{source_name_text} ({dist} kpc)", "Other BH / candidate BH"],["o", "."], [colour, "grey"])]
    legend = ax.legend(handles=legend_handles, loc="lower right", fontsize=10)
    ax.add_artist(legend) 



    plt.xlim([min_Lx,max_Lx])
    plt.ylim([min_Lr,max_Lr_2])

    #plt.xlabel(r'1–10 keV Unabsorbed X-ray Luminosity [erg s$^{-1}$]')
    #plt.ylabel(r'1.28 GHz Radio Luminosity [erg s$^{-1}$]')
    plt.xlabel(r'$L_X$ [erg s$^{-1}$]')
    plt.ylabel(r'$L_R$ [erg s$^{-1}$]')
    ax.set_yscale("log", base=10)
    ax.set_xscale("log", base=10)
    ax.xaxis.set_major_locator(plt.LogLocator(base=10.0, numticks=10))
    ax.xaxis.set_minor_locator(plt.LogLocator(base=10.0, subs="auto", numticks=10))



    if save_name!=None: 
        plt.savefig(f"../FIGURES/{save_name}.png", dpi=600, bbox_inches="tight")
        plt.savefig(f"../FIGURES/{save_name}.pdf", dpi=600, bbox_inches="tight")

    plt.show()



##############################################################################################################
## LRLX PLOTS FOR PAPER -- BH VS NS DETECTIONS


def plot_LrLx_BH_vs_NS_detections(df):

    ## Get the BH and NS data
    mask_NS = df["class"].isin(["NS"]) 
    mask_BH = df["class"].isin(["BH", "candidateBH"]) 
    x_NS = df["Lx"][mask_NS]
    y_NS = df["Lr"][mask_NS]
    x_BH = df["Lx"][mask_BH]
    y_BH =  df["Lr"][mask_BH]

    ## Convert to log scale
    log_x_NS = np.log10(x_NS).to_numpy()
    log_y_NS = np.log10(y_NS).to_numpy()
    log_x_BH = np.log10(x_BH).to_numpy()
    log_y_BH =  np.log10(y_BH).to_numpy()


    ## KS Tests
    pval_Lx = ks_2samp(log_x_NS, log_x_BH).pvalue
    pval_Lr = ks_2samp(log_y_NS, log_y_BH).pvalue
    #pval_2D, D_2D = ks2d2s(log_x_NS, log_y_NS, log_x_BH, log_y_BH, extra=True)

    ## Define log bins
    nbins = 20
    log_bins_x = np.linspace(np.log10(min_Lx), np.log10(max_Lx), nbins + 1)
    log_bins_y = np.linspace(np.log10(min_Lr), np.log10(max_Lr_2), nbins + 1)

    ## Setup figure and axes
    fig = plt.figure(figsize=(9,6))
    gs = fig.add_gridspec(2, 2, width_ratios=(5, 1.5), height_ratios=(1.5, 5), wspace=0.05, hspace=0.05)

    ax_main = fig.add_subplot(gs[1, 0])
    ax_top = fig.add_subplot(gs[0, 0], sharex=ax_main)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)

    ## Main scatter + KDE
    ax_main.scatter(log_x_NS, log_y_NS, color=colour_NS, alpha=0.6, label="NS")
    ax_main.scatter(log_x_BH, log_y_BH, color=colour_BH, alpha=0.6, label="BH & candidate BH")

    levels = [0.05, 0.25, 0.6, 0.9]

    sns.kdeplot(x=log_x_NS, y=log_y_NS, ax=ax_main, levels=levels, color=colour_NS, linewidths=1, fill=False)
    sns.kdeplot(x=log_x_BH, y=log_y_BH, ax=ax_main, levels=levels, color=colour_BH, linewidths=1, fill=False)

    ## Create state legend (within plot) in black
    states = ["HS","QS"]
    state_legend_handles = [plt.Line2D([0], [0], color='none', linestyle='None', markersize=1, marker=".", label=state) for state in states] 
    phantom = plt.Line2D([0], [0], color='none', label='\u200A' * 48)  
    state_legend_handles.append(phantom)
    state_legend = ax_main.legend(handles=state_legend_handles, loc="upper left",bbox_to_anchor=(0.0, 0.79), title="States", handlelength=0, fontsize=10)
    ax_main.add_artist(state_legend)  

    ## Create types legend
    ax_main.legend( fontsize=9,loc="upper left", title="Types")

    ## Add 2D KS p-value to legend
    #ax_main.text(0.98, 0.02, f"2D KS: p = {pval_2D:.2g}", transform=ax_main.transAxes, ha='right', va='bottom', fontsize=10, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))


    ## Top histogram (Lx)
    ax_top.hist(log_x_NS, bins=log_bins_x, density=True, alpha=0.1, color=colour_NS)
    ax_top.hist(log_x_BH, bins=log_bins_x, density=True, alpha=0.1, color=colour_BH)
    sns.kdeplot(log_x_NS, fill=True, alpha=0.2, linewidth=2, color=colour_NS, ax=ax_top)
    sns.kdeplot(log_x_BH, fill=True, alpha=0.2, linewidth=2, color=colour_BH, ax=ax_top)
    ax_top.set_ylabel("Density")
    ax_top.text(0.98, 0.9, f"KS: p = {pval_Lx:.2g}", transform=ax_top.transAxes, ha='right', va='top', fontsize=10, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    plt.setp(ax_top.get_xticklabels(), visible=False)


    ## Right histogram (Lr)
    ax_right.hist(log_y_NS, bins=log_bins_y, density=True, alpha=0.1, color=colour_NS, orientation='horizontal')
    ax_right.hist(log_y_BH, bins=log_bins_y, density=True, alpha=0.1, color=colour_BH, orientation='horizontal')
    sns.kdeplot(y=log_y_NS, fill=True, alpha=0.2, linewidth=2, color=colour_NS, ax=ax_right)
    sns.kdeplot(y=log_y_BH, fill=True, alpha=0.2, linewidth=2, color=colour_BH, ax=ax_right)
    ax_right.set_xlabel("Density")
    ax_right.text(0.92, 0.97, f"KS: p = {pval_Lr:.2g}", transform=ax_right.transAxes, ha='right', va='top', fontsize=10, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    plt.setp(ax_right.get_yticklabels(), visible=False)


    ## Format the x-axis to mimic log scale
    log_xmin = np.log10(min_Lx)
    log_xmax = np.log10(max_Lx)
    log_start_x = int(np.floor(log_xmin))
    log_end_x = int(np.ceil(log_xmax))
    major_ticks_x = np.arange(log_start_x, log_end_x)
    minor_ticks_x = []
    for decade in major_ticks_x:
        minor_ticks_x.extend([decade + np.log10(i) for i in range(2, 10)])
    ax_main.set_xticks(major_ticks_x)
    ax_main.set_xticks(minor_ticks_x, minor=True)
    ax_main.xaxis.set_major_formatter(FuncFormatter(lambda val, _: f"$10^{{{int(val)}}}$"))
    ax_main.xaxis.set_minor_formatter(FuncFormatter(lambda val, _: ""))


    ## Format y-axis
    log_ymin = np.log10(min_Lr)
    log_ymax = np.log10(max_Lr_2)
    log_start_y = int(np.floor(log_ymin))
    log_end_y = int(np.ceil(log_ymax))
    major_ticks_y = np.arange(log_start_y, log_end_y)
    minor_ticks_y = []
    for decade in major_ticks_y:
        minor_ticks_y.extend([decade + np.log10(i) for i in range(2, 10)])
    ax_main.set_yticks(major_ticks_y)
    ax_main.set_yticks(minor_ticks_y, minor=True)
    ax_main.yaxis.set_major_formatter(FuncFormatter(lambda val, _: f"$10^{{{int(val)}}}$"))
    ax_main.yaxis.set_minor_formatter(FuncFormatter(lambda val, _: ""))


    ## Set axis limits and labels
    ax_main.set_xlim([np.log10(min_Lx), np.log10(max_Lx)])
    ax_main.set_ylim([np.log10(min_Lr), np.log10(max_Lr_2)])
    ax_main.set_xlabel(r'$L_X$ [erg s$^{-1}$]')
    ax_main.set_ylabel(r'$L_R$ [erg s$^{-1}$]')

    save_name = "BH_NS_distributions_interp"
    plt.savefig(f"../FIGURES/{save_name}.png", dpi=600,bbox_inches="tight")
    plt.savefig(f"../FIGURES/{save_name}.pdf", dpi=600,bbox_inches="tight")

    plt.show()


