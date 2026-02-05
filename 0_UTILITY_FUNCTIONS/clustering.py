import pandas as pd
import numpy as np
np.set_printoptions(precision=12)

from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import fcluster

from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.cluster import DBSCAN
from sklearn.mixture import BayesianGaussianMixture
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib.cm as cm
from matplotlib.colors import LogNorm
from matplotlib.patches import Ellipse
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.colors as mcolors
import matplotlib as mpl


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


from plotting import min_Lr, max_Lr, max_Lr_2, min_Lx, max_Lx  



##########################################################
## PLOTTING DATA

# Plotting for checking
def plotLrLx(data):

    Lx = data[:,0]
    Lr = data[:,1]
    
    fig= plt.figure(figsize=(7,5))
    ax = fig.add_subplot(1,1,1)
    plt.plot(Lx, Lr, '.')
    ax.set_yscale("log", base=10)
    ax.set_xscale("log", base=10)
    plt.xlabel(r"$L_X$ [erg s$^{-1}$]")
    plt.ylabel(r"$L_R$ [erg s$^{-1}$]")
    plt.xlim([min_Lx, max_Lx])
    plt.ylim([min_Lr, max_Lr_2])
    plt.show()


def plotLrLx_tranformed(transformed_data):
    
    fig= plt.figure(figsize=(7,5))
    ax = fig.add_subplot(1,1,1)
    plt.scatter(transformed_data[:, 0], transformed_data[:, 1], s=8)
    plt.xlabel(r"Transformed $L_X$")
    plt.ylabel(r"Transformed $L_R$")
    plt.show()





##########################################################
## PREPROCESSING FUNCTIONS


def transform1(data, testing = False):
    """
    Transform the data before clustering, using the same strategy as the Gallo papers:
    - Step 1: Take the logarithms of the luminosities
    - Step 2: Standardise
    - Step 3: Apply PCA
    - Step 4: Standardise again
    """

    Lx = data[:,0]
    Lr = data[:,1]
    
    ## Log-transform
    log_Lx = np.log10(Lx)
    log_Lr = np.log10(Lr)

    data = np.column_stack((log_Lx, log_Lr))


    ## Standardise
    original_means = np.mean(data, axis=0) # = np.array([np.mean(log_Lx), np.mean(log_Lr)])
    original_stds = np.std(data, axis=0, ddof =1) # = np.array([np.std(log_Lx), np.std(log_Lr)])
    data_centered = (data - original_means) / original_stds
    if testing:
        print(data_centered[:5])
        print("Should be close to zero...", np.mean(data_centered, axis =0)) # should be close to zero
        print("Should be close to one...", np.std(data_centered, ddof=1, axis =0)) # should be close to one

        # Alternative (equivalent) for the standardisation above:
        x_prime = (log_Lx - np.mean(log_Lx)) / np.std(log_Lx)
        y_prime = (log_Lr - np.mean(log_Lr)) / np.std(log_Lr)
        data_centered_alt = np.column_stack((x_prime, y_prime))
        print(data_centered_alt[:5])



    ## PCA transformation
    # PCA identifies a set of orthogonal axes (principal components) that maximise variance in the data. These axes are linear combinations of the original axes (variables).
    # The data is then transformed by projecting it onto these new principal axes. This operation can be thought of as rotating the data around its mean to align it with the new axes.
    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(data_centered)

    ## Standardise again (which is equivalent to PCA with whiten=True)
    pca_data_mean = np.mean(pca_data, axis=0)
    pca_data_std = np.std(pca_data, axis=0, ddof=0)
    pca_data_scaled = (pca_data - pca_data_mean) / pca_data_std
    if testing:
        print()
        print("Should be close to zero...", np.mean(pca_data_scaled, axis =0)) # should be close to zero
        print("Should be close to one...", np.std(pca_data_scaled, ddof=1, axis =0)) # should be close to one

    
    if testing: return pca_data_scaled, original_means, original_stds, pca, pca_data_mean, pca_data_std 
    else: return pca_data_scaled 




def transform1_back(pca_data_scaled, original_means, original_stds, pca_model, pca_data_mean, pca_data_std):
    """
    For checking the transformation above.
    """
    
    ## Reverse standardisation in the PCA space
    pca_data = pca_data_scaled * pca_data_std + pca_data_mean

    ## Reverse PCA rotation
    data_centered = pca_data @ pca_model.components_  

    ## Reverse mean centering
    data_original = data_centered*original_stds + original_means

    ## Reverse log-transform
    log_Lx = data_original[:,0]
    log_Lr = data_original[:,1]
    Lx = 10**log_Lx
    Lr = 10**log_Lr
    data_original = np.column_stack((Lx, Lr))


    return data_original





def transform1_alt(data, standardise):
    """
    The following should be equivalent to the function above, with the option of whether to standardise. 
    
    Steps: 
    - (Standardise)
    - PCA with whiten=True
    """

    Lx = data[:,0]
    Lr = data[:,1]

    # Log-transform
    log_Lx = np.log10(Lx)
    log_Lr = np.log10(Lr)

    data = np.column_stack((log_Lx, log_Lr))

    if standardise:
        scaler = StandardScaler()
        data = scaler.fit_transform(data)
    

    data = PCA(n_components=2, whiten=True).fit_transform(data)


    return data




def standardise(data):
    """
    As a check, see what happens when I just standardise and do not run PCA.
    """

    Lx = data[:,0]
    Lr = data[:,1]

    # Log-transform
    log_Lx = np.log10(Lx)
    log_Lr = np.log10(Lr)

    data = np.column_stack((log_Lx, log_Lr))
    
    original_means = np.array([np.mean(log_Lx), np.mean(log_Lr)])
    original_stds = np.array([np.std(log_Lx), np.std(log_Lr)])
    data= (data - original_means) / original_stds

    return data







##########################################################
## Affinity propagation


def affinity_propagation(data, transformed_data, preference, damping, logged = False, show_results1 = True, show_results2=True, source_classes = ["BH", "candidateBH"], states=["HS", "QS"], save_name = None):

    af = AffinityPropagation(preference=preference, damping =damping, random_state=0).fit(transformed_data)
    cluster_centers_indices = af.cluster_centers_indices_
    labels = af.labels_

    n_clusters_ = len(cluster_centers_indices)
    print("Estimated number of clusters: %d" % n_clusters_)

    if n_clusters_ > 1:
        silhouette = metrics.silhouette_score(transformed_data, labels, metric="sqeuclidean")
        print("Silhouette Coefficient: %0.3f" % silhouette)
    else:
        print("Only one cluster found. Silhouette score is not applicable.")

    if n_clusters_ <= 4:
        cluster_colors = ["orange", "purple", "green", "pink"][:n_clusters_]
        cmap = None
        norm = None
    else:
        cmap = plt.colormaps["viridis"].resampled(n_clusters_)
        norm = mcolors.Normalize(vmin=0, vmax=n_clusters_ - 1)
        cluster_colors = [cmap(norm(k)) for k in range(n_clusters_)]


    if show_results1:

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9,3))

        for k in range(n_clusters_):
            col = cluster_colors[k]
            class_members = labels == k
            if not np.any(class_members):
                continue
            cluster_center = transformed_data[cluster_centers_indices[k]]
            ax1.scatter(
                transformed_data[class_members, 0], transformed_data[class_members, 1], color=col, marker="."
            )
            ax1.scatter(
                cluster_center[0], cluster_center[1], s=14, color=col, marker="o"
            )
            for x in transformed_data[class_members]:
                ax1.plot(
                    [cluster_center[0], x[0]], [cluster_center[1], x[1]], color=col
                )

        ax1.set_title("Estimated number of clusters: %d" % n_clusters_)
        if cmap:
            ax2.scatter(data[:, 0], data[:, 1], c=labels, cmap=cmap, norm=norm, s=25)
        else:
            # Manual colors for ≤4 clusters
            for k in range(n_clusters_):
                ax2.scatter(data[labels == k, 0], data[labels == k, 1], color=cluster_colors[k], s=25)
        if logged:
            ax2.set_xlabel("log Lx")
            ax2.set_ylabel("log Lr")
        else:
            ax2.set_xlabel("Lx")
            ax2.set_ylabel("Lr")
            ax2.set_yscale("log", base=10)
            ax2.set_xscale("log", base=10)
        plt.show()


    if show_results2:
        
        fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)  # Main plot

        # Main scatter plot
        if cmap:
            ax.scatter(data[:, 0], data[:, 1], c=labels, cmap=cmap, norm=norm, s=25)
        else:
            # Manual colors for ≤4 clusters
            for k in range(n_clusters_):
                ax.scatter(data[labels == k, 0], data[labels == k, 1], color=cluster_colors[k], s=25)
        if logged:
            ax.set_xlabel("log Lx")
            ax.set_ylabel("log Lr")
        else:
            ax.set_xlabel(r"1–10 keV Unabsorbed X-ray Luminosity [erg s$^{-1}$]")
            ax.set_ylabel(r"1.28 GHz Radio Luminosity [erg s$^{-1}$]")
            #ax.set_xlabel(r"$L_X$ [erg s$^{-1}$]")
            #ax.set_ylabel(r"$L_R$ [erg s$^{-1}$]")
            ax.set_xscale("log", base=10)
            ax.set_yscale("log", base=10)


        # Add inset below legends (top left quadrant)
        # bbox_to_anchor=(0.03, 0.48, 0.32, 0.32),
        ax1 = inset_axes(ax, width="100%", height="100%", loc="lower right", bbox_to_anchor=(0.67, 0.07, 0.32, 0.32), bbox_transform=ax.transAxes, borderpad=0)



        for k in range(n_clusters_):
            col = cluster_colors[k]
            class_members = labels == k
            cluster_center = transformed_data[cluster_centers_indices[k]]
            ax1.scatter(
                transformed_data[class_members, 0], transformed_data[class_members, 1], color=col, marker="."
            )
            ax1.scatter(
                cluster_center[0], cluster_center[1], s=14, color=col, marker="o"
            )
            for x in transformed_data[class_members]:
                ax1.plot(
                    [cluster_center[0], x[0]], [cluster_center[1], x[1]], color=col
                )
        ax1.tick_params(labelsize=6)
        ax1.set_xlabel("$x$", fontsize=9, labelpad=2)
        ax1.set_ylabel("$y$", fontsize=9, labelpad=2)


        # ----- Legends and Boxes -----

        # Legend: Types
        source_classes = ["BH candidate" if sc == "candidateBH" else sc for sc in source_classes] # for the source_class array, replace "candidateBH" with "BH"
        type_legend_handles = [plt.Line2D([0], [0], color='none', linestyle='None', markersize=1, marker=".", label=typ) for typ in source_classes]
        phantom = plt.Line2D([0], [0], color='none', label='\u200A' * 65)  # spacing hack
        type_legend_handles.append(phantom)
        type_legend = ax.legend(handles=type_legend_handles, loc="upper left", title="Types", handlelength=0, fontsize=10)
        ax.add_artist(type_legend)

        # Legend: States (positioned right of Types)
        state_legend_handles = [plt.Line2D([0], [0], color='none', linestyle='None', markersize=1, marker=".", label=state) for state in states]
        phantom = plt.Line2D([0], [0], color='none', label='\u200A' * 48)
        state_legend_handles.append(phantom)
        state_legend = ax.legend(handles=state_legend_handles, loc="upper left", bbox_to_anchor=(0.15, 1.0), title="States", handlelength=0, fontsize=10)
        ax.add_artist(state_legend)

        # Legend: Hyperparameters (positioned right of States)
        pref_str = "/" if preference is None else f"{preference:>4}"
        hyperparam_handle = [plt.Line2D([0], [0], color='none', linestyle='None', label=r"$\lambda$="+f"{damping:>3}\n"+r"$p$="+f"{pref_str}")]  # invisible
        #phantom = plt.Line2D([0], [0], color='none', label='\u200A' * 1)
        #hyperparam_handle.append(phantom)
        hyperparam_box = ax.legend(handles=hyperparam_handle,bbox_to_anchor=(0.1058, 0.815), handlelength=0, fontsize=10)
        ax.add_artist(hyperparam_box)

        # Axis limits
        ax.set_xlim([min_Lx, max_Lx])
        ax.set_ylim([min_Lr, max_Lr_2])


        if save_name: 
            plt.savefig(f"../FIGURES/{save_name}.png", dpi=600,bbox_inches="tight")
            plt.savefig(f"../FIGURES/{save_name}.pdf", dpi=600,bbox_inches="tight")
    

        plt.show()


    return labels





##########################################################
## CENTROID-BASED: K-MEANS


# Iterate over different k values, plot the result, and save the minimised cost function.
# Initialisation: "random", k++""
def run_kmeans(data, transformed_data, initialisation):

    k_values = np.arange(1, 4)
    
    minimal_cost_function = []
    all_labels=[]
    all_centroids=[]

    for i, k_val in enumerate(k_values):

        # n_init is the number of times the k-means algorithm is run with different centroid seeds
        kmeans = KMeans(n_clusters=k_val, init=initialisation, n_init=4, random_state=0) 
        kmeans.fit(transformed_data)
        
        labels = kmeans.labels_ # OR labels = kmeans.fit_predict(transformed_data)
        all_labels.append(labels)
        centroids = kmeans.cluster_centers_
        all_centroids.append(centroids)
        minimal_cost_function.append(kmeans.inertia_)


        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed clusters
        if k_val!=1: print(f"Silhouette for k={k_val}: ", silhouette_score(transformed_data, labels))

        # Plot results
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9,3))
        ax1.set_title("k-means with %s clusters" % k_val)
        ax1.scatter(transformed_data[:, 0], transformed_data[:, 1], c=labels, alpha=0.3, cmap="viridis")
        ax1.scatter(centroids[:, 0], centroids[:, 1], c=np.arange(0,k_val), marker='^', s=100, cmap="viridis")
        ax1.set_xlabel("x")
        ax1.set_ylabel("y")
        ax2.set_title("k-means with %s clusters" % k_val)
        ax2.scatter(data[:,0], data[:,1], c=labels, s=5)
        ax2.set_xlabel("Lx")
        ax2.set_ylabel("Lr")
        ax2.set_yscale("log", base=10)
        ax2.set_xscale("log", base=10)
        plt.show()

    # Determining the right k with elbow analysis
    # Plot the minimal cost function as a function of k
    # https://blog.cambridgespark.com/how-to-determine-the-optimal-number-of-clusters-for-k-means-clustering-14f27070048f
    plt.figure(1, figsize=(4, 3))
    plt.scatter(k_values, minimal_cost_function, c="k")
    plt.xlabel("Number of clusters k")
    plt.ylabel("Minimal cost function")
    plt.tight_layout()
    plt.show()


    # Determining the right k with the silhouette factor
    # https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html

    for index, k in enumerate(k_values[1:]): # can't include when k=1
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9,4))

        # The 1st subplot is the silhouette plot
        # The silhouette coefficient can range from -1, 1 
        ax1.set_xlim([-0.1, 1])
        ax1.set_ylim([0, len(transformed_data) + (k + 1) * 10]) 

        
        silhouette_avg = silhouette_score(transformed_data, all_labels[index+1])
        print("For n_clusters =", k, ", the average silhouette_score is :", silhouette_avg)

        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(transformed_data, all_labels[index+1])

        y_lower = 10
        for i in range(k):
            # Aggregate the silhouette scores for samples belonging to cluster i, and sort them
            ith_cluster_silhouette_values = sample_silhouette_values[all_labels[index+1] == i]
            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(i) / k)
            ax1.fill_betweenx(
                np.arange(y_lower, y_upper),
                0,
                ith_cluster_silhouette_values,
                facecolor=color,
                edgecolor=color,
                alpha=0.7,
            )

            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        #ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        # Second plot showing the actual clusters formed
        colors = cm.nipy_spectral(all_labels[index+1].astype(float) / k)
        ax2.scatter(transformed_data[:, 0], transformed_data[:, 1], marker=".", s=30, lw=0, alpha=0.7, c=colors, edgecolor="k")

        # Draw white circles at cluster centers
        ax2.scatter(
            all_centroids[index+1][:, 0],
            all_centroids[index+1][:, 1],
            marker="o",
            c="white",
            alpha=1,
            s=200,
            edgecolor="k",
        )

        for i, c in enumerate(all_centroids[index+1]):
            ax2.scatter(c[0], c[1], marker="$%d$" % i, alpha=1, s=50, edgecolor="k")

        #ax2.set_title("Visualisation of the clustered data.")

        plt.suptitle(
            "Silhouette analysis for KMeans clustering on sample data with n_clusters = %d"
            % k,
            fontsize=14,
            fontweight="bold",
        )

    plt.show()



##########################################################
## HIERARCHICAL CLUSTERING -- agglomerative clustering

def hierarchical_clustering(data, transformed_data, plotting_dendo = False):
    linkage_methods = ['ward', 'single', 'complete', 'average']

    for i, link_method in enumerate(linkage_methods):
        Z = linkage(transformed_data, link_method)

        if plotting_dendo:
            plt.figure(i + 1)
            plt.title('Hierarchical clustering, %s linkage' % link_method)
            plt.xlabel('Object index or (cluster size)')
            plt.ylabel('Distance')
            dendrogram(
                Z,
                truncate_mode='lastp',  # show only the last p merged clusters
                p=12,  # show only the last p merged clusters
                leaf_rotation=90.,
                leaf_font_size=12.,
                show_contracted=True,  # to get a distribution impression in truncated branches
            ) 
            plt.show()

        max_distance = Z[-1, 2] # gives the distance between the two final clusters merged, which is the largest distance in the hierarchical clustering
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9,3))
        max_d = max_distance * 0.7 # form clusters where the distance between them is smaller than 70% of the largest observed merge distance
        clusters = fcluster(Z, max_d, criterion='distance')
        ax1.set_title("Clusters with %s linkage" % link_method)
        ax1.scatter(transformed_data[:,0], transformed_data[:,1], c=clusters, cmap='viridis', s=5)
        ax1.set_xlabel("x")
        ax1.set_ylabel("y")
        ax2.set_title("Clusters with %s linkage" % link_method)
        ax2.scatter(data[:,0], data[:,1], c=clusters, cmap='viridis', s=5)
        ax2.set_xlabel("Lx")
        ax2.set_ylabel("Lr")
        ax2.set_yscale("log", base=10)
        ax2.set_xscale("log", base=10)
        plt.show()

        
##########################################################
## Density-based clustering: DBSCAN

def dbscan(data, transformed_data, eps = 0.5, min_samples=10, metric='euclidean'):

    if metric =='mahalanobis':
        # Estimate covariance matrix from data
        cov = np.cov(transformed_data.T)
        db = DBSCAN(eps=eps, min_samples=min_samples, metric=metric, metric_params={'V': cov}).fit(transformed_data)
    else: db = DBSCAN(eps=eps, min_samples=min_samples, metric=metric).fit(transformed_data)
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print("Estimated number of clusters: %d" % n_clusters_)
    print("Estimated number of noise points: %d" % n_noise_)
    print(f"Silhouette Coefficient: {metrics.silhouette_score(transformed_data, labels):.3f}")

    unique_labels = set(labels)
    core_samples_mask = np.zeros_like(labels, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True

    colours = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9,3))
    for k, col in zip(unique_labels, colours):

        if k == -1: col = [0, 0, 0, 1] # black for noise

        class_member_mask = labels == k

        xy = transformed_data[class_member_mask & core_samples_mask]
        ax1.plot(
            xy[:, 0],
            xy[:, 1],
            "o",
            markerfacecolor=tuple(col),
            markeredgecolor="k",
            markersize=8,
        )

        xy = transformed_data[class_member_mask & ~core_samples_mask]
        ax1.plot(
            xy[:, 0],
            xy[:, 1],
            "o",
            markerfacecolor=tuple(col),
            markeredgecolor="k",
            markersize=3,
        )

        xy = data[class_member_mask & core_samples_mask]
        ax2.plot(
            xy[:, 0],
            xy[:, 1],
            "o",
            markerfacecolor=tuple(col),
            markeredgecolor="k",
            markersize=8,
        )

        xy = data[class_member_mask & ~core_samples_mask]
        ax2.plot(
            xy[:, 0],
            xy[:, 1],
            "o",
            markerfacecolor=tuple(col),
            markeredgecolor="k",
            markersize=3,
        )
        ax2.set_yscale("log", base=10)
        ax2.set_xscale("log", base=10)

    plt.title(f"Estimated number of clusters: {n_clusters_}")
    plt.show()







##########################################################
## Distribution-based clustering: GMM

def plot_centroids(ax, centroids, weights=None):
    # The weight is the proportion int the dataset
    sizes = weights * 500 if weights is not None else 50  # scale marker size by weight
    ax.scatter(centroids[:, 0], centroids[:, 1], 
               marker='x', s=sizes, linewidths=2, 
               color="red", zorder=10, alpha=0.9)


# covariance_types = ["spherical", "tied", "diag", "full"]
# Note that I could alternatively use fit_predict
def GMM(data, transformed_data, covariance_type, testing=False):

    k_values = np.arange(1, 8)
    models = [None for n in k_values]
    AIC = [None for n in k_values]
    BIC = [None for n in k_values]
    for i in range(len(k_values)):
        models[i] = GaussianMixture(n_components=k_values[i], covariance_type=covariance_type, n_init=20, random_state=2, max_iter=1000)
        models[i].fit(transformed_data)
        AIC[i] = models[i].aic(transformed_data) 
        BIC[i] = models[i].bic(transformed_data) 

    i_best = np.argmin(BIC)
    gmm_best = models[i_best]
    print("Best fit converged:", gmm_best.converged_)
    print("BIC: n_components =  %i" % k_values[i_best])    

    if testing:
        print("TEST: Plotting for n_components = 2")
        gmm_best = models[1]

    fig = plt.figure(figsize=(3,3))
    ax = fig.add_subplot(111)
    ax.plot(k_values, AIC, '-k', label='AIC')
    ax.plot(k_values, BIC, ':k', label='BIC')
    ax.legend(loc=1)
    ax.set_xlabel('N components')
    plt.setp(ax.get_yticklabels(), fontsize=7)

    # Plot best results
    labels = gmm_best.predict(transformed_data)
    aic = AIC[i_best]
    bic = BIC[i_best]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.scatter(transformed_data[:, 0], transformed_data[:, 1], c=labels, cmap='viridis', s=5)
    ax1.text(0.05, 0.95, f"AIC: {aic:.2f}\nBIC: {bic:.2f}", transform=ax1.transAxes, fontsize=10, 
                verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))

    ax2.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', s=5)
    ax2.set_xlabel("Lx")
    ax2.set_ylabel("Lr")
    ax2.set_yscale("log", base=10)
    ax2.set_xscale("log", base=10)
    plt.show()



    
    # Plot contour plot for best GMM
    mins = transformed_data.min(axis=0) - 0.1
    maxs = transformed_data.max(axis=0) + 0.1
    xx, yy = np.meshgrid(np.linspace(mins[0], maxs[0], 1000),np.linspace(mins[1], maxs[1], 1000))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = -gmm_best.score_samples(grid_points)
    Z = Z.reshape(xx.shape)
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    vmin, vmax = Z.min(), Z.max()
    plt.contourf(xx, yy, Z, norm=LogNorm(vmin=vmin, vmax=vmax), levels=np.logspace(np.log10(vmin), np.log10(vmax), 12))
    plt.plot(transformed_data[:, 0], transformed_data[:, 1], 'k.', markersize=2)
    plot_centroids(ax, gmm_best.means_, gmm_best.weights_)
    plt.tight_layout()
    plt.show()



##########################################################
## Distribution-based clustering: Bayesian GMM

# covariance_types = ["spherical", "tied", "diag", "full"]
def bayesian_GMM(data, transformed_data, covariance_type):


    # Set n_components to the maximum
    bgm = BayesianGaussianMixture(n_components=10, covariance_type=covariance_type, weight_concentration_prior=0.1, n_init=20, random_state=2, max_iter=1000)
    labels = bgm.fit_predict(transformed_data)

    effective_components = np.sum(bgm.weights_ > 0.01)  # Count components with non-negligible weight
    print(f"Effective number of components: {effective_components}")


    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9,3))
    ax1.scatter(transformed_data[:,0], transformed_data[:,1], c=labels, cmap='viridis', s=5)
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax2.scatter(data[:,0], data[:,1], c=labels, cmap='viridis', s=5)
    ax2.set_xlabel("Lx")
    ax2.set_ylabel("Lr")
    ax2.set_yscale("log", base=10)
    ax2.set_xscale("log", base=10)
    plt.show()

    

