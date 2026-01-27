import linmix
import numpy as np
np.set_printoptions(linewidth=200)
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.font_manager as font_manager
from IPython.display import display, clear_output
import pandas as pd
import shutil
import ast
from joblib import Parallel, delayed
from scipy.stats import truncnorm
from tqdm import tqdm
from IPython.display import display

import sys
import os
sys.path.append(os.path.abspath("../0_UTILITY_FUNCTIONS/"))
from get_LrLx_data import *
from get_data import *
from plotting import colours, min_Lr, max_Lr, max_Lr_2, min_Lx, max_Lx

##############################################################################################################
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

mpl.rcParams["text.usetex"] = False

label_size = 10
font_size = 20

Lr_med = 4.54e+28 #erg/s
Lx_med = 5.47e+35 #erg/s


##############################################################################################################
## HELPER FUNCTIONS



def hist_plotter(ax, edges, counts, C='k', LW=2, normalised_to_one=True, label='', ALPHA=1.0):
    """
    Function that can plot a histogram using the output of np.histogram and a defined axis. I.e. it uses manual step plotting (rather than plt.hist). 
    It draws flat-topped rectangles (like a histogram) using line segments.
    """
    
    if normalised_to_one: counts = counts/float(max(counts))
    
    ax.plot([edges[0]]*2, [0, counts[0]], lw=LW, color=C, label=label, alpha=ALPHA)
    ax.plot([edges[-1]]*2, [counts[-1], 0], lw=LW, color=C, alpha=ALPHA)
    for i in range(len(edges)-1):
        ax.plot([edges[i], edges[i+1]], [counts[i]]*2, lw=LW, color=C, alpha=ALPHA)
        if i != 0:
            ax.plot([edges[i]]*2, [counts[i-1],counts[i]], lw=LW, color=C, alpha=ALPHA)      
    


def plotting_helper(all_data=[], onerun_data=[], n_bins_all=1000, n_bins_onerun=100, param_name="", ax =None, show_legend=True):
    
    if ax is None:
        fig = plt.figure(figsize=(8.,12))
        ax = fig.add_subplot(111)

    ax.tick_params(labelsize=label_size, width=2, length=8, axis='both', which='major', pad=3)
    ax.tick_params(labelsize=label_size, length=4, width=1, axis='both', which='minor', pad=3)
    ax.set_xlabel(param_name, fontsize=font_size)
    ax.set_ylabel('P('+param_name+')', fontsize=font_size, labelpad=12)
    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_ticks_position('both')
    ax.get_yaxis().set_tick_params(direction='in', which='both')
    ax.get_xaxis().set_tick_params(direction='in', which='both')

    
    if len(all_data)>0:
        all_counts, all_edges = np.histogram(all_data, bins=n_bins_all, density=True)
        hist_plotter(ax=ax, edges=all_edges, counts=all_counts, LW=2, normalised_to_one=False, C='b', label="All runs")
    if len(onerun_data)>1:
        counts, edges = np.histogram(onerun_data, bins=n_bins_onerun, density=True)
        hist_plotter(ax=ax, edges=edges, counts=counts, LW=2, normalised_to_one=False, C='k', label="First run")

    if show_legend: ax.legend(fontsize=10)



def sample_from_distribution(dist_type, param1, param2):
    """
    Sample distributions for the distances.
    """

    ## Truncated normal distribution
    # param1 is the mean and param2 is the standard deviation
    if dist_type == "gauss":
        #return np.random.normal(param1, param2)
        upper = np.inf # right‐hand truncation (no upper bound)
        lower = 0 # left‐hand truncation at 0
        a, b = (lower - param1) / param2, (upper - param1) / param2
        return truncnorm.rvs(a, b, loc=param1, scale=param2, size=1)[0]
    
    ## Uniform distribution
    # param1 is the lower bound and param2 is the upper bound
    elif dist_type == "uniform":
        return np.random.uniform(param1, param2)



def sample_from_distribution_nruns(dist_type, param1, param2, seed=5, N_runs=1):
    """
    Sample distributions for the distances, as above, except we sample multiple runs at once.
    """

    ## Truncated normal distribution
    # param1 is the mean and param2 is the standard deviation
    if dist_type == "gauss":
        #return np.random.normal(param1, param2)
        upper = np.inf # right‐hand truncation (no upper bound)
        lower = 0 # left‐hand truncation at 0
        a, b = (lower - param1) / param2, (upper - param1) / param2
        return truncnorm.rvs(a, b, loc=param1, scale=param2, size=N_runs, random_state=seed)
    
    ## Uniform distribution
    # param1 is the lower bound and param2 is the upper bound
    elif dist_type == "uniform":
        return np.random.uniform(param1, param2, size=N_runs)


def plot_distance_distribution(distance_samples_all, unique_names, unique_D, unique_D_prob):

    N_sources = distance_samples_all.shape[1]

    for i in range(N_sources):
        samples = distance_samples_all[:, i]  # Shape (N_runs,)
        name = unique_names[i]
        D = unique_D[i]
        D_prob = unique_D_prob[i]

        # Compute statistics
        mean_val = np.mean(samples)
        std_val = np.std(samples)
        min_val = np.min(samples)
        max_val = np.max(samples)

        # Plot histogram
        plt.figure(figsize=(6, 4))
        plt.hist(samples, bins=30, alpha=0.75, edgecolor='black')

        # Title with source name, D, and D_prob
        plt.title(f"{name}\nD = {D:.2f}, D_prob = {D_prob}")

        # Legend with stats
        stats_text = (
            f"mean = {mean_val:.2f}\n"
            f"std = {std_val:.2f}\n"
            f"min = {min_val:.2f}\n"
            f"max = {max_val:.2f}"
        )
        plt.legend([stats_text], loc='best')

        plt.xlabel("Sampled Distance")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.show()






##############################################################################################################
## RUN A SINGLE LINMIX ITERATION
"""
NOTES:

L = F * (4*pi*D^2)
logL = logF + log(4*pi) + 2logD
ΔlogL = ΔlogF (if we ignore the error on D, as it is folded in when we sample distances for each iteration)

For a particular iteration: 
Li = F * (4*pi*D^2) * (Di/D)^2 = L * (Di/D)^2
logLi = logL + 2log(Di/D) 
ΔlogLi = ΔlogL (the error remains the same, it cancels out)
So, to scale the fitted luminosities by a factor (Di/D)^2 for iteration i, we need to add 2*log(Di/D) in log space.
In other words: d_corr  = 2*log(Di/D)
"""


def run_mcmc_iteration(j, x, y, dx, dy, delta, K, dir, seed, silence=True, xycov=None,  min_iter=5000):
    """
    - j is the iteration number

    IMPORTANT: 
    After convergence is reached, the second halves of all chains are concatenated and stored in the .chain attribute as a numpy recarray.
    """

    ## For reproducibility
    np.random.seed(seed)

    ## Run the MCMC
    lm = linmix.LinMix(x=x, y=y, xsig=dx, ysig=dy, delta=delta, K=K, parallelize=False, seed=seed)
    lm.run_mcmc(miniter=min_iter,maxiter=100000,silent=silence)

    if not silence: 
        # linmix runs nchains = 4 chains, but discards the first half of samples from each chain
        # So the resultant concatenated chain has length =  nchains * niter /2
        print(f"Length of chain: {len(lm.chain)}")
   
    
    ## Extract the fitted parameters for each chain for this iteration
    alphas = []
    betas = []
    sigmas = []
    for i in range(len(lm.chain)):
        alphas.append(lm.chain[i]['alpha']) # alpha = y-intercept = c, i.e. log(Xi)
        betas.append(lm.chain[i]['beta']) # m, i.e. slope beta
        sigmas.append(np.sqrt(lm.chain[i]['sigsqr'])) # "spread" of data around best-fit
    
    ## Save all the results
    np.savetxt(f'{dir}/alphas_{j}.txt', alphas)
    np.savetxt(f'{dir}/betas_{j}.txt', betas)
    np.savetxt(f'{dir}/sigmas_{j}.txt', sigmas)


    ## Calculate the percentiles, modes, and means
    all_summary_results = [
    np.percentile(alphas, 16), np.percentile(alphas, 50), np.percentile(alphas, 84), np.mean(alphas),
    np.percentile(betas, 10), np.percentile(betas, 16), np.percentile(betas, 50), np.percentile(betas, 84), np.percentile(betas, 90), np.mean(betas),
    np.percentile(sigmas, 16), np.percentile(sigmas, 50), np.percentile(sigmas, 84), np.mean(sigmas)
    ]
    ## Save the summary results
    np.savetxt(f'{dir}/summary_results_{j}.txt', all_summary_results)




##############################################################################################################
## RUN LINMIX FOR N_RUNS ITERATIONS WITH DISTANCE CORRECTIONS


def run_linmix(N_runs=1, K=3, seed=42, parallel=True, min_iter=5000,
               names = None, interp=False, type_source=None, include_Fr_uplims=True,
               verbose= False, gx_339_filtered=False):
    """
    Runner function to run linmix for N_runs iterations, applying distance corrections each time.
    The code is structured such that the first iteration is always run with the best predictions for the distances.

    Use names = ["...", "..."] if we do not want to include all the sources.
    """

    ##############################
    ## GET THE DATA
    # Important: The Lx upper limits have been excluded, as linmix does not have the functionality to fit these
    # Get the luminosity (results using the best distance estimates)
    lr0, lx0, lr, dlr, delta_radio, lx, dlx_l, dlx_u, delta_xrays, source_names, unique_names, unique_D, unique_D_prob, t = get_data_arrays(names = names, interp=interp, rerun = False, save=False, incl_Fr_uplims=include_Fr_uplims, type_source=type_source)
    delta = delta_radio.astype(int) # Convert boolean to int (1 for detection, 0 for upper limit)

    
    ##############################
    ## PREPARE THE DATA FOR LINMIX
    """
    NOTES:
    - In the equations below, the uncertainties are the same as they would be without normalisation (it cancels out).
    - We can choose to take the average of the upper and lower uncertainty: 
        0.5*(upper - lower) = 0.5*(lower_er + upper_er) = 0.5*[log10(x+dx_u) - log10(x-dx_l)] = 0.5*[ (log10(x+dx_u) - log10(x)) + (log10(x) - log10(x-dx_l))]                         
        OR we can take the max of the upper and lower uncertainty (conservative approach)... which is what we do.
    """

    # x:
    lx_scaled = lx/lx0
    log_lx_scaled = np.log10(lx_scaled) # x
    log_dlx_l_scaled = np.log10(lx / (lx - dlx_l)) # = log(lx/lx0) - log( (lx - dlx_l)/ lx0 ) 
    log_dlx_u_scaled = np.log10( ( lx + dlx_u) / lx ) # = log( (lx + dlx_u)/ lx0 ) - log(lx/lx0)
    log_dlx_scaled = np.maximum(log_dlx_l_scaled , log_dlx_u_scaled )

    # y:
    lr_scaled = lr/lr0
    log_lr_scaled = np.log10(lr_scaled) # y
    log_dlr_l_scaled = np.log10(lr / (lr - dlr))
    log_dlr_u_scaled = np.log10( ( lr + dlr) / lr )
    log_dlr_scaled = np.maximum(log_dlr_l_scaled , log_dlr_u_scaled )


    ##############################

    # If filtering for GX 339-4 region of interest
    if gx_339_filtered:
        t0 = 58964
        t1 = 59083
        mask_excl = (source_names=="GX 339-4") & ( (lx <= 2.7e34) | ( (t>=t0) & (t<=t1) ) )
        log_lx_scaled = log_lx_scaled[~mask_excl]
        log_lr_scaled = log_lr_scaled[~mask_excl]
        log_dlx_scaled = log_dlx_scaled[~mask_excl]
        log_dlr_scaled = log_dlr_scaled[~mask_excl]
        delta = delta[~mask_excl]
        source_names = source_names[~mask_excl]


    ##############################

    ## Make directory to store inputs
    dir_input = "./input_files"
    if os.path.exists(dir_input):
        shutil.rmtree(dir_input)  # delete the directory and its contents
    os.makedirs(dir_input) 


    ## Make directory to store MCMC results
    if interp: dir = "./MCMC_parfiles_interp"
    else: dir = "./MCMC_parfiles_paired"
    if type_source=="BH": 
        dir+="_BH"
    elif type_source=="NS":
        dir+="_NS"
    if names!=None:
        dir+=f"_{names}"
    if include_Fr_uplims==False: 
        dir+="_no_Fr_uplims"
    if gx_339_filtered:
        dir+="_GX339_filtered"
    dir +=f"_Nruns_{N_runs}"
    print("DIRECTORY: ", dir)
    if os.path.exists(dir):
        shutil.rmtree(dir)  # delete the directory and its contents
    os.makedirs(dir) 


    ##############################
    ## GET THE DISTANCE CORRECTIONS FOR EACH RUN

    np.random.seed(seed) # so that the results are reproducible
    
    ## Get an array of luminosity correction factors for each iteration. This will have length N_runs * n_data
    d_corr_all = []
    ## Get the distance used for each iteration and each unique source
    distance_samples_all = []


    for j in range(N_runs):

        if j==0: # for the first iteration, use the "best" distance estimates, i.e. no corrections
            distance_samples_all.append(unique_D)
            d_corr = np.zeros(len(log_lx_scaled)) 
        
        else:
            # Apply a correction on the luminosities due to distance, in log space:
            # Each time, re-draw the assumed distance for each source
            # If L = F * (4*pi*D^2), then Li = F * (4*pi*D^2) * (Di/D)^2 = L * (Di/D)^2 ...  dcorr = 2*log(Di/D) in log space
        
            # Sample distance (Di) for each unique source
            distance_samples = np.array([sample_from_distribution(*ast.literal_eval(dist)) for dist in unique_D_prob])
            distance_samples_all.append(distance_samples)
            # Calculate 2* np.log10(Di/D) fo each source
            distance_ratios = distance_samples / unique_D
            d_corr = 2.*np.log10(distance_ratios)
            # Map the dcorr to the source name
            d_corr_dict = dict(zip(unique_names, d_corr))
            # Assign the correction to the data points (since there are multiple data points per source)
            # source_names is the array of source names for each data point.
            d_corr = np.array([d_corr_dict[name] for name in source_names])
    
        d_corr_all.append(d_corr)

    if verbose: # Plot distribution for each of the sources
        plot_distance_distribution(np.array(distance_samples_all), unique_names, unique_D, unique_D_prob)

     
    ##############################
    ## RUN LINMIX FOR EACH ITERATION

    if N_runs==1: silence=False
    else: silence=True

    # Run iterations in parallel
    if parallel:
        Parallel(n_jobs=-1)(delayed(run_mcmc_iteration)(j, x = log_lx_scaled + d_corr, y = log_lr_scaled+d_corr, dx = log_dlx_scaled, dy = log_dlr_scaled, 
                                                        delta = delta, K = K, dir= dir, seed=seed, silence=silence, min_iter= min_iter) for j, d_corr in enumerate(tqdm(d_corr_all)))
    
    # Run iterations sequentially
    else: 
        for j, d_corr in enumerate(d_corr_all):
            run_mcmc_iteration(j, x = log_lx_scaled + d_corr, y = log_lr_scaled+d_corr, dx = log_dlx_scaled, dy = log_dlr_scaled, 
                               delta = delta, K = K, dir= dir, seed=seed, silence=silence, min_iter= min_iter)





##############################################################################################################
## GET THE LINMIX RESULTS AFTER RUNNING


def show_and_plot_results(dir, ax, lr0, lx0, colour, colour_line, show_alt_uncertainty_methods=False, best_fit_fmt = '-', plot_unc= True, best_fit_legend_only_beta=False):
    """
    NOTES:
    y = alpha + beta * x_plot = log(lr/lr0) .... log space linreg results
    lr_plot = lr0 * 10**(a + b * x_plot) =  lr0 * (10 ** a) * ((lx_plot / lx0) ** b) 

    LR = LR0 * Xi * (LX / LX0)^beta 
    => log(LR/LR0) = log(Xi) + beta * log(LX/LX0) = alpha + beta * log(LX/LX0)
    Therefore, LR = LR0  * 10**(alpha) * (LX / LX0)^beta , where alpha is the intercept
    
    If all_alphas has shape (n,), then all_alphas[:, None] has shape (n,1) ... where n = N_runs*n_chains
    If x_plot has shape (m,), then x_plot[None, :] has shape (1,m) ... where m = 100
    """

    ## From the first run
    # These arrays have length n_chains
    onerun_alphas = np.loadtxt(f'{dir}/alphas_0.txt')
    onerun_betas = np.loadtxt(f'{dir}/betas_0.txt')
    onerun_sigmas = np.loadtxt(f'{dir}/sigmas_0.txt')

    ## From all runs
    # These arrays have shape N_runs*n_chains
    all_alphas, all_betas, all_sigmas, = [], [], []
    all_summary_results = []
    N_runs = len(os.listdir(dir)) // 4  # each run has 4 files
    for i in range(N_runs):
        all_alphas.append(np.loadtxt(f'{dir}/alphas_{i}.txt'))
        all_betas.append(np.loadtxt(f'{dir}/betas_{i}.txt'))
        all_sigmas.append(np.loadtxt(f'{dir}/sigmas_{i}.txt'))
        all_summary_results.append(np.loadtxt(f'{dir}/summary_results_{i}.txt'))
    # Create single arrays for all runs (length N_runs*n_chains)
    all_alphas = np.concatenate(all_alphas)
    all_betas = np.concatenate(all_betas)
    all_sigmas = np.concatenate(all_sigmas)
    print(all_alphas.shape, all_betas.shape, all_sigmas.shape)


    ## Mean, median, 16th, and 84th percentiles calculated from every run 
    # These arrays have length N_runs
    # The following takes the first, second, third etc. elements from each inner array and puts these into their corresponding separate arrays
    alpha_16,  alpha_50 , alpha_84 , alpha_mean , beta_10 ,beta_16 ,beta_50 , beta_84 , beta_90, beta_mean, sigma_16, sigma_50, sigma_84, sigma_mean = map(list, zip(*all_summary_results ))
    alpha_50, beta_50, sigma_50 = np.array(alpha_50), np.array(beta_50), np.array(sigma_50)


    ######################################
    ### GET PARAMETER ESTIMATES

    
    ## Results using all values together
    # 1 sigma is +-34% from the mean 
    print()
    print("USING ALL RESULTS TOGETHER:")
    def get_percentiles(samples, string=True):
        p16, p50, p84 = np.percentile(samples, [16, 50, 84])
        if string: return f"{p50:.3f}  -{(p50-p16):.3f}  +{(p84-p50):.3f}"
        else: return p50, p50-p16, p84-p50
    def get_means(samples):
        mean = np.mean(samples)
        p16, p84 = np.percentile(samples, [16, 84])
        return f"{mean:.3f}  -{(mean - p16):.3f}  +{(p84 - mean):.3f}"
    print ("The fitted values and uncertainties based on the mean, 16th, and 84th percentile of all values")
    print("Alpha =", get_means(all_alphas))
    print("Beta  =", get_means(all_betas))
    print("Sigma =", get_means(all_sigmas))
    print ("The fitted values and uncertainties based on the median, 16th, and 84th percentile of all values" )
    print("Alpha =", get_percentiles(all_alphas))
    print("Beta  =", get_percentiles(all_betas))
    print("Sigma =", get_percentiles(all_sigmas))
    print("The 90% lower limit on beta from all values")
    print("Beta >", f"{np.percentile(all_betas, 10):.3f}")
    print("The 90% upper limit on beta from all values")
    print("Beta <", f"{np.percentile(all_betas, 90):.3f}")
    print()


    ## Results using the mean values of the median, 16th, 84th percentiles calculated from each individual run
    print("USING PERCENTILE RESULTS FROM EACH REPEAT:")
    print( "The fitted values based on the mean of the median, 16th, and 84th percentile of the runs")
    def get_mean_of_percentiles(p16_array, p50_array, p84_array):
        p16_mean = np.mean(p16_array)
        p50_mean = np.mean(p50_array)
        p84_mean = np.mean( p84_array)
        return f"{p50_mean:.3f}  -{(p50_mean - p16_mean):.3f}  +{(p84_mean - p50_mean):.3f}"
    print( 'Alpha = ', get_mean_of_percentiles(alpha_16, alpha_50, alpha_84) )
    print( 'Beta = ', get_mean_of_percentiles(beta_16, beta_50, beta_84) )
    print( 'Sigma = ', get_mean_of_percentiles(sigma_16, sigma_50, sigma_84) )
    print ("The 90% lower limit on beta from run results")
    print ('Beta > ', np.round(np.mean(beta_10),2)) # report lower limit on the slope at the 90% confidence level
    print ("The 90% upper limit on beta from run results")
    print ('Beta < ', np.round(np.mean(beta_90),2)) # report upper limit on the slope at the 90% confidence level
    print()


    ## Final parameter estimates to use 
    beta, dbeta_l, dbeta_u = get_percentiles(all_betas, string=False)
    dbeta = np.max([dbeta_u, dbeta_l])
    alpha, dalpha_l, dalpha_u = get_percentiles(all_alphas, string=False)
    dalpha = np.max([dalpha_u, dalpha_l])
    sigma, dsigma_l, dsigma_u = get_percentiles(all_sigmas, string=False)
    dsigma = np.max([dsigma_u, dsigma_l])



    ######################################
    ### PLOT POSTERIOR PARAMETER DISTRIBUTIONS

    ## Plot comparison between a single run and combined output of all runs
    # Will highlight if there is any systematic bias introduced by the assumed distance in a single run.
    fig_histo, axs = plt.subplots(1, 3, figsize=(12, 4))  # 3 panels horizontally
    plotting_helper(all_data=all_betas, onerun_data=onerun_betas, param_name=r'$\beta$', ax=axs[1], show_legend=False) # (slope)
    plotting_helper(all_data=all_alphas, onerun_data=onerun_alphas, param_name=r'$\alpha$', ax=axs[0], show_legend=False)  # (offset)
    plotting_helper(all_data=all_sigmas, onerun_data=onerun_sigmas, param_name=r'$\sigma$', ax=axs[2], show_legend=True) # (scatter)
    fig_histo.tight_layout()
    display(fig_histo)
    plt.close(fig_histo)




    ######################################
    ### PLOT THE RESULTS

    log_lx_plot = np.linspace(np.log10(min_Lx),np.log10(max_Lx),100,endpoint=True) # log Lx grid for plotting
    lx_plot = 10.0**log_lx_plot # Lx values
    x_plot = np.log10(lx_plot / lx0)  # X grid in log-space
    

    ## PLOT BEST FIT
    lr_best_fit = lr0 * (10**(alpha)) * ((lx_plot/lx0)**(beta)) 
    if best_fit_legend_only_beta:
        ax.errorbar(lx_plot, lr_best_fit, fmt=best_fit_fmt, color=colour_line, lw=2, label=r'$\beta$'+f'={beta:.3f}+{dbeta_u:.3f}/-{dbeta_l:.3f}', zorder=5)
    else:
        ax.errorbar(lx_plot, lr_best_fit, fmt=best_fit_fmt, color=colour_line, lw=2, label=r'$\beta$'+f'={beta:.3f}+{dbeta_u:.3f}/-{dbeta_l:.3f}\n'+r'$\alpha$'+f'={alpha:.3f}+{dalpha_u:.3f}/-{dalpha_l:.3f}\n'+r'$\sigma_ε$'+f'={sigma:.3f}+{dsigma_u:.3f}/-{dsigma_l:.3f}', zorder=5)
    

    ## PLOT UNCERTAINTIES

    if plot_unc:
        ## Get all the results (using all posterior samples; i.e. N_runs*n_chains samples)
        y = all_alphas[:, None] + all_betas[:, None] * x_plot[None, :] # y has shape (n,m)
        y16, y50, y84 = np.percentile(y, [16, 50, 84], axis=0) # each has shape (m,) 
        lr_plot_16 = lr0 * 10**y16
        lr_plot_50 = lr0 * 10**y50
        lr_plot_84 = lr0 * 10**y84
        # Add posterior predictive band (includes ε ~ N(0, σ^2) in dex) 
        rng = np.random.default_rng(123)
        eps = rng.normal(0.0, all_sigmas[:, None], size=(all_alphas.size, x_plot.size))  # shape: (n, m)
        y_with_scatter = y + eps
        y_with_scatter_16, y_with_scatter_50, y_with_scatter_84 = np.percentile(y_with_scatter, [16, 50, 84], axis=0)
        lr_with_scatter_plot_16 = lr0 * 10**y_with_scatter_16
        lr_with_scatter_plot_50 = lr0 * 10**y_with_scatter_50
        lr_with_scatter_plot_84 = lr0 * 10**y_with_scatter_84

        # Parameter-uncertainty-only band
        ax.plot(lx_plot, lr_plot_16, '--', color=colour_line, lw=1, zorder=6) # label='68% band (params)'
        ax.plot(lx_plot, lr_plot_84, '--', color=colour_line, lw=1, zorder=6)
        ax.fill_between(lx_plot, lr_plot_16, lr_plot_84, facecolor= colour_line, alpha=0.07)
        # Uncertainty band with scatter
        ax.fill_between(lx_plot, lr_with_scatter_plot_16, lr_with_scatter_plot_84, alpha=0.07, facecolor=colour_line) # , label='68% band (predictive)'


    if show_alt_uncertainty_methods:

        ## ALTERNATIVE: Jakob's method
        # The four combinations of the two 1-sigma limits on slope and normalisations
        # HOWEVER: This method ignores the covariance between the slope and intercept
        case1 = lr0 * (10**(alpha +dalpha_u)) * ((lx_plot/lx0)**(beta-dbeta_l))
        case2 = lr0 * (10**(alpha -dalpha_l)) * ((lx_plot/lx0)**(beta+dbeta_u)) 
        case3 = lr0 * (10**(alpha -dalpha_l)) * ((lx_plot/lx0)**(beta-dbeta_l)) 
        case4 = lr0 * (10**(alpha +dalpha_u)) * ((lx_plot/lx0)**(beta+dbeta_u)) 
        maxline = []
        minline = []
        for i in range(len(case1)):
            maxline.append(max(case1[i], case2[i], case3[i], case4[i]))
            minline.append(min(case1[i], case2[i], case3[i], case4[i]))
        ax.plot(lx_plot, minline, '--', color='purple', lw=1, label=f'$1$-$\sigma$ contours')
        ax.plot(lx_plot, maxline, '--', color='purple', lw=1)
        ax.fill_between(lx_plot, minline, maxline, facecolor='purple', alpha=0.07)


        ## ALTERNATIVE: Plot the percentiles of the median results (length N_runs)
        y = alpha_50[:, None] + beta_50[:, None]*x_plot[None, :]
        lr_plot_16 = lr0* 10**(np.percentile(y, 16, axis=0))
        lr_plot_84 = lr0* 10**(np.percentile(y, 84, axis=0))
        # Add scatter (using the median sigma from each run)
        y_with_scatter_u = y + sigma_50[:, None] 
        y_with_scatter_l = y - sigma_50[:, None] 
        lr_with_scatter_plot_16 = lr0* 10**(np.percentile(y_with_scatter_l, 16, axis=0))
        lr_with_scatter_plot_84 = lr0* 10**(np.percentile(y_with_scatter_u, 84, axis=0))
        ax.plot(lx_plot, lr_plot_16, '--', color='grey', lw=1, label=f'$1$-$\sigma$ contours')
        ax.plot(lx_plot, lr_plot_84, '--', color= 'grey', lw=1)
        ax.fill_between(lx_plot, lr_plot_16, lr_plot_84, facecolor='grey', alpha=0.07)
        ax.fill_between(lx_plot, lr_with_scatter_plot_16, lr_with_scatter_plot_84, facecolor='grey', hatch='///',alpha=0.05, label=f'$1$-$\sigma$ with scatter')





def linmix_results(N_runs =1, names = None, interp=False, type_source=None, include_Fr_uplims=True, save_name=None, show_alt_uncertainty_methods=False, gx_339_filtered=False):
    """
    Plot the results after running linmix with distance corrections -- plot all results together.

    Use names = ["...", "..."] if we do not want to include all the sources.
    """


    ##############################
    ## GET THE DIR

    if interp: dir = "./MCMC_parfiles_interp"
    else: dir = "./MCMC_parfiles_paired"
    if type_source=="BH": 
        dir+="_BH"
        colour = "#D40404"
        colour_line = "#a10000ff"
    elif type_source=="NS":
        dir+="_NS"
        colour = "#0303D6"
        colour_line = "#020286ff"
    if names!=None:
        dir+=f"_{names}"
        colour_source = colours.get(names[0], "#000000ff")
        colour = colour_source 
        colour_line = colour_source
    if include_Fr_uplims==False: 
        dir+="_no_Fr_uplims"
    if gx_339_filtered:
        dir+="_GX339_filtered"
    dir +=f"_Nruns_{N_runs}"
    print("DIRECTORY: ", dir)



    ##############################
    ## PLOT RESULTS


    fig = plt.figure(figsize=(9,6), constrained_layout=True)
    ax = fig.add_subplot(1,1,1)
     

    ## PLOT DATA
    lr0, lx0, lr, dlr, delta_radio, lx, dlx_l, dlx_u, delta_xrays, source_names, unique_names, unique_D, unique_D_prob, t = get_data_arrays(names = names, interp=interp, rerun = False, save=False, incl_Fr_uplims=include_Fr_uplims, type_source=type_source, gx_339_filtered=gx_339_filtered)
    plot, caps, bars = ax.errorbar(lx, lr, yerr=dlr, xerr=[dlx_l, dlx_u], fmt='o', ms=5, mec=colour, mfc=colour, uplims=~delta_radio,  xuplims=~delta_xrays, capsize=0.5, ecolor="black", elinewidth=0.4, zorder=3)
    for cap in caps:
        cap.set_color('black')      # Set cap color
        cap.set_markeredgewidth(0.2)  # Set edge width
        cap.set_markersize(3) 
    for bar in bars:
        bar.set_color('black')
    

    show_and_plot_results(dir, ax, lr0, lx0, colour, colour_line, show_alt_uncertainty_methods)


    ## PLOT LAYOUT
    
    ## Create source type legend
    if type_source=="BH": 
        type_sources = ["BH", "candidateBH"]
    elif type_source==None:
        type_sources = ["BH", "candidateBH", "NS"]
    else: 
        type_sources = ["NS"]
    type_legend_handles = [plt.Line2D([0], [0], color='none', linestyle='None', markersize=1, marker=".",  label=typ) for typ in type_sources]
    phantom = plt.Line2D([0], [0], color='none', label='\u200A' * 65) 
    type_legend_handles.append(phantom)
    type_legend = ax.legend(handles=type_legend_handles, loc="upper left", title="Types", handlelength=0, fontsize=10)
    ax.add_artist(type_legend)


    ## Create state legend (within plot) in black
    states = ["HS", "QS"]
    state_legend_handles = [plt.Line2D([0], [0], color='none', linestyle='None', markersize=1, marker=".", label=state) for state in states] 
    phantom = plt.Line2D([0], [0], color='none', label='\u200A' * 48)  
    state_legend_handles.append(phantom)
    state_legend = ax.legend(handles=state_legend_handles, bbox_to_anchor=(0.27, 1.0), title="States", handlelength=0, fontsize=10)
    ax.add_artist(state_legend)  
    

    handles, labels = ax.get_legend_handles_labels()
    # Ensure the 'Best fit' label is identified first
    sorted_pairs = sorted(zip(labels, handles), key=lambda x: 0 if r'$\beta$' in x[0] else 1)
    sorted_labels, sorted_handles = zip(*sorted_pairs)
    ax.legend(sorted_handles, sorted_labels, fontsize=10, bbox_to_anchor=(0.28, 0.82), title="Best Fit")

    plt.xlim([min_Lx,max_Lx])
    plt.ylim([min_Lr,max_Lr_2])
    #ax.set_xlabel(r"1–10 keV Unabsorbed X-ray Luminosity [erg s$^{-1}$]")
    #ax.set_ylabel(r'1.28 GHz Radio Luminosity [erg s$^{-1}$]')
    ax.set_xlabel(r'$L_X$ [erg s$^{-1}$]')
    ax.set_ylabel(r'$L_R$ [erg s$^{-1}$]')
    ax.set_xscale('log', base=10)
    ax.set_yscale('log', base=10)   
    ax.xaxis.set_major_locator(plt.LogLocator(base=10.0, numticks=10))
    ax.xaxis.set_minor_locator(plt.LogLocator(base=10.0, subs="auto", numticks=10))

    if save_name!=None: 
        plt.savefig(f"../FIGURES/{save_name}.png", dpi=600)
        plt.savefig(f"../FIGURES/{save_name}.pdf", dpi=600)

    plt.show()






def linmix_results_BH_vs_NS(N_runs=1, interp=True, save_name="BH_vs_NS"):
    """
    Plot the results after running linmix with distance corrections -- plot the BH vs NS results.

    Use names = ["...", "..."] if we do not want to include all the sources.
    """

    if interp: dir = "./MCMC_parfiles_interp"
    else: dir = "./MCMC_parfiles_paired"



    ##############################
    ## PLOT RESULTS


    fig = plt.figure(figsize=(9,6), constrained_layout=True)
    ax = fig.add_subplot(1,1,1)

    for type_source in ["BH", "NS", "NS_no_Fr_uplims"]:
        
        print(type_source)

        if type_source=="BH": 
            dir_new = dir + "_BH"
            colour = "#D40404"
            colour_line = "#a10000ff"
        elif type_source=="NS":
            dir_new = dir + "_NS"
            colour = "#0303D6"
            colour_line = "#020286ff"
        elif type_source=="NS_no_Fr_uplims":
            dir_new = dir + "_NS_no_Fr_uplims"
        dir_new +=f"_Nruns_{N_runs}"
        
        ## PLOT DATA
        if type_source!="NS_no_Fr_uplims":
    
            lr0, lx0, lr, dlr, delta_radio, lx, dlx_l, dlx_u, delta_xrays, source_names, unique_names, unique_D, unique_D_prob, t = get_data_arrays(names = None, interp=interp, rerun = False, save=False, incl_Fr_uplims=True, type_source=type_source)
            plot, caps, bars = ax.errorbar(lx, lr, yerr=dlr, xerr=[dlx_l, dlx_u], fmt='o', ms=5, mec=colour, mfc=colour, uplims=~delta_radio,  xuplims=~delta_xrays, capsize=0.5, ecolor="black", elinewidth=0.4, zorder=3)
            for cap in caps:
                cap.set_color('black')      # Set cap color
                cap.set_markeredgewidth(0.2)  # Set edge width
                cap.set_markersize(3) 
            for bar in bars:
                bar.set_color('black')
    
        ## PLOT RESULTS
        if type_source=="NS_no_Fr_uplims": show_and_plot_results(dir_new, ax, lr0, lx0, colour, colour_line, plot_unc = False, best_fit_fmt = "-.", show_alt_uncertainty_methods=False)
        else: show_and_plot_results(dir_new, ax, lr0, lx0, colour, colour_line, plot_unc = True, best_fit_fmt = '-', show_alt_uncertainty_methods=False)


    ## PLOT LAYOUT
    
    # Create state legend (within plot) in black
    states = ["HS", "QS"]
    state_legend_handles = [plt.Line2D([0], [0], color='none', linestyle='None', markersize=1, marker=".", label=state) for state in states] 
    phantom = plt.Line2D([0], [0], color='none', label='\u200A' * 48)  
    state_legend_handles.append(phantom)
    state_legend = ax.legend(handles=state_legend_handles, bbox_to_anchor=(0.128, 0.578),title="States", handlelength=0, fontsize=10)
    ax.add_artist(state_legend)  


    # Create type legend (within plot) in black
    types = ["BH & candidate BH", "NS"]
    colours = ["#D40404", "#0303D6"]
    type_legend_handles = [plt.Line2D([0], [0], marker='o', color=colour, linestyle='None', markersize=6, label=type_source) for type_source, colour in zip(types,colours)] 
    phantom = plt.Line2D([0], [0], color='none', label='\u200A' * 48)  
    type_legend_handles.append(phantom)
    type_legend = ax.legend(handles=type_legend_handles, loc="upper left",bbox_to_anchor=(0.285, 1.0), title="Types", fontsize=10)
    ax.add_artist(type_legend) 


    handles, labels = ax.get_legend_handles_labels()
    # Ensure the 'Best fit' label is identified first
    sorted_pairs = sorted(zip(labels, handles), key=lambda x: 0 if r'$\beta$' in x[0] else 1)
    sorted_labels, sorted_handles = zip(*sorted_pairs)
    ax.legend(sorted_handles, sorted_labels, fontsize=10, loc="upper left", title="Best fits")   

    plt.xlim([min_Lx,max_Lx])
    plt.ylim([min_Lr,max_Lr_2])
    #ax.set_xlabel(r"1–10 keV Unabsorbed X-ray Luminosity [erg s$^{-1}$]")
    #ax.set_ylabel(r'1.28 GHz Radio Luminosity [erg s$^{-1}$]')
    ax.set_xlabel(r'$L_X$ [erg s$^{-1}$]')
    ax.set_ylabel(r'$L_R$ [erg s$^{-1}$]')
    ax.set_xscale('log', base=10)
    ax.set_yscale('log', base=10)   
    ax.xaxis.set_major_locator(plt.LogLocator(base=10.0, numticks=10))
    ax.xaxis.set_minor_locator(plt.LogLocator(base=10.0, subs="auto", numticks=10))

    if save_name!=None: 
        plt.savefig(f"../FIGURES/{save_name}.png", dpi=600)
        plt.savefig(f"../FIGURES/{save_name}.pdf", dpi=600)

    plt.show()





def linmix_results_individual_sources(N_runs=1, names=None , interp=True, save_name="individual_sources"):
    """
    Plot the results after running linmix with distance corrections -- plot the individual sources.
    """

    if interp: dir = "./MCMC_parfiles_interp"
    else: dir = "./MCMC_parfiles_paired"

    ##############################
    ## PLOT RESULTS


    fig = plt.figure(figsize=(9,6), constrained_layout=True)
    ax = fig.add_subplot(1,1,1)

    distances = []

    for name in names:
        
        print(name)

        dir_new = dir+f"_['{name}']"
        dir_new +=f"_Nruns_{N_runs}"
        print(dir_new)

        colour_source = colours.get(name, "#000000ff")
        colour = colour_source 
        colour_line = colour_source 
        

        ## PLOT DATA
        lr0, lx0, lr, dlr, delta_radio, lx, dlx_l, dlx_u, delta_xrays, source_names, unique_names, unique_D, unique_D_prob, t = get_data_arrays(names = [name], interp=interp, rerun = False, save=False, incl_Fr_uplims=True, type_source=None)
        distances.append(unique_D[0])

        plot, caps, bars = ax.errorbar(lx, lr, yerr=dlr, xerr=[dlx_l, dlx_u], fmt='o', ms=5, mec=colour, mfc=colour, uplims=~delta_radio,  xuplims=~delta_xrays, capsize=0.5, ecolor="black", elinewidth=0.4, zorder=3)
        for cap in caps:
            cap.set_color('black')      # Set cap color
            cap.set_markeredgewidth(0.2)  # Set edge width
            cap.set_markersize(3) 
        for bar in bars:
            bar.set_color('black')
    
        ## PLOT RESULTS
        show_and_plot_results(dir_new, ax, lr0, lx0, colour, colour_line, plot_unc = True, best_fit_fmt = '-', show_alt_uncertainty_methods=False, best_fit_legend_only_beta=True)


    ## PLOT LAYOUT
    
    # Create state legend (within plot) in black
    states = ["HS", "QS"]
    state_legend_handles = [plt.Line2D([0], [0], color='none', linestyle='None', markersize=1, marker=".", label=state) for state in states] 
    phantom = plt.Line2D([0], [0], color='none', label='\u200A' * 48)  
    state_legend_handles.append(phantom)
    state_legend = ax.legend(handles=state_legend_handles, bbox_to_anchor=(0.13, 1.0),title="States", handlelength=0, fontsize=10)
    ax.add_artist(state_legend)  



    # Create name legend (within plot) in black
    all_colours = [colours.get(name, "#000000ff") for name in names]
    # For the names array, replace "-" with "–"
    names_text = unique_names.copy()
    names_text = [name.replace("-", "–") for name in names]
    name_legend_handles = [plt.Line2D([0], [0], marker='o', color=colour, linestyle='None', markersize=6, label=rf"{name} ($D$ = {D} kpc)") for name, colour, D in zip(names_text, all_colours, np.array(distances))]
    phantom = plt.Line2D([0], [0], color='none', label='\u200A' * 48)
    name_legend_handles.append(phantom)
    name_legend = ax.legend(handles=name_legend_handles, loc="lower right", fontsize=10)
    ax.add_artist(name_legend)


    handles, labels = ax.get_legend_handles_labels()
    # Ensure the 'Best fit' label is identified first
    sorted_pairs = sorted(zip(labels, handles), key=lambda x: 0 if r'$\beta$' in x[0] else 1)
    sorted_labels, sorted_handles = zip(*sorted_pairs)
    ax.legend(sorted_handles, sorted_labels, fontsize=10, bbox_to_anchor=(0.41, 1.0), title="Best fits")     


    plt.xlim([min_Lx,max_Lx])
    plt.ylim([min_Lr,max_Lr_2])
    #ax.set_xlabel(r"1–10 keV Unabsorbed X-ray Luminosity [erg s$^{-1}$]")
    #ax.set_ylabel(r'1.28 GHz Radio Luminosity [erg s$^{-1}$]')
    ax.set_xlabel(r'$L_X$ [erg s$^{-1}$]')
    ax.set_ylabel(r'$L_R$ [erg s$^{-1}$]')
    ax.set_xscale('log', base=10)
    ax.set_yscale('log', base=10)   
    ax.xaxis.set_major_locator(plt.LogLocator(base=10.0, numticks=10))
    ax.xaxis.set_minor_locator(plt.LogLocator(base=10.0, subs="auto", numticks=10))

    if save_name!=None: 
        plt.savefig(f"../FIGURES/{save_name}.png", dpi=600)
        plt.savefig(f"../FIGURES/{save_name}.pdf", dpi=600)

    plt.show()





