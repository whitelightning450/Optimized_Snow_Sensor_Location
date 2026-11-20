#imports
import numpy as np
from sklearn import mixture
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import BallTree
from sklearn.preprocessing import MinMaxScaler


#Define functions
def neighbors(num_neighbors,dat,gmm):
    """
    Finds the nearest neighbors to the cluster centers defined by a Gaussian Mixture Model (GMM).

    Parameters
    ----------
    num_neighbors : int
        Number of nearest neighbors to find for each GMM cluster center.
    dat: Pandas dataframe
        Copy of the dataframe that will be used (created in GMM function)
    gmm : sklearn.mixture._gaussian_mixture.GaussianMixture
        The fitted Gaussian Mixture Model object.

    Returns
    -------
    np.ma.array
        An array containing the indices of the nearest neighbors for each GMM cluster center.
    """
    
    # Find the closest locations using a Ball Tree structure for efficient nearest-neighbor search.
    tree = BallTree(dat)
    _distance, inx_initial = tree.query(gmm.means_, k=num_neighbors)
    
    # Retrieve the indices of the nearest neighbors.
    dat_indices = np.array(dat.index.to_list())
    snow_indices = dat_indices[inx_initial]

    return np.ma.array(snow_indices)

def modified_sample_loc(num_neighbors,d,dat,gmm):
    """
    Identifies suitable sample locations based on nearest neighbors within a "go" area.

    Parameters
    ----------
    num_neighbors : int
        Number of neighbors to consider for each GMM cluster center.
    d : Pandas DataFrame
        The original dataframe containing the full dataset, including the "No_go" column.
    dat : Pandas DataFrame
        A copy of the dataframe containing only the features used in GMM clustering.
    gmm : sklearn.mixture._gaussian_mixture.GaussianMixture
        The fitted Gaussian Mixture Model object.

    Returns
    -------
    tuple
        A tuple containing:
        - A boolean indicating whether the process was successful.
        - A list of indices for the final sample locations.
        - An array of the original nearest neighbors for each GMM cluster center.
    """
    
    # Get the nearest neighbors for the GMM cluster centers.
    snow_neighbors = neighbors(num_neighbors,dat,gmm)

    # Initialize an empty array and dictionary to store neighbor information.
    nbr_inx = np.empty([len(snow_neighbors),num_neighbors], object)
    nbr_dict = {}
    
    # Loop through each cluster center's neighbors and store their "No_go" status.
    for i in range(len(snow_neighbors)):
        for val, j in enumerate(snow_neighbors[i]):
            nbr_dict[j] = (d['No_go'].loc[j])
            nbr_inx[i,val] = {j: nbr_dict[j]}
    print(nbr_inx)

    # Find the nearest neighbor within the "go" region for each cluster center.
    final_inx = []
    for i in range(len(nbr_inx)):
        j = 0
        while next(iter(nbr_inx[i,j].values())) != 1:
            j+=1
            if j >= num_neighbors:
                print('Warning: GMM location #',(i+1),'had no suitable neighbors. Skipping this location.')
                j=0
                i=i+1
                break
        else:
            print(nbr_inx[i,j])
            print('neighbor# = ',j)
            final_inx.append(next(iter(nbr_inx[i,j].keys())))
    return len(final_inx) == len(snow_neighbors), final_inx, snow_neighbors[:,0]


def gmm_step(d, ss, n, f, num_neighbors):
    """
    Fits a Gaussian Mixture Model (GMM) to a sub-region and identifies suitable sample locations.

    Parameters
    ----------
    d : Pandas DataFrame
        The full dataset.
    ss : float
        Subsampling fraction (e.g., 0.5 for 50% of the data).
    n : int
        Number of components for the GMM.
    f : list[str]
        List of features to be used in the GMM.
    num_neighbors : int
        Number of neighbors to consider for each GMM cluster center.

    Returns
    -------
    Pandas DataFrame
        A dataframe containing the GMM locations and their corresponding features.
    """
    
    # Create a copy of the dataframe with only the selected features.
    dat = d[f].copy()

    # Subsample the data with a fixed random seed for reproducibility.
    dat = dat.sample(frac = ss, random_state=42)

    #min_max scale to normalize the data
    min_max_scaler = MinMaxScaler()
    dat[f] = min_max_scaler.fit_transform(dat[f])

    #fit a GMM on the scaled data
    gmm = mixture.GaussianMixture(n_components = n, covariance_type='spherical', random_state=42)
    gmm.fit(dat)
    print('point numbers:', n)
    print('gmm.means_')
    print(gmm.means_)
    
    # Check if the GMM converged.
    if gmm.converged_ == False:
        print('Error: GMM did not converge')
    
    # Select valid sensor locations.
    success, inx, _orig = modified_sample_loc(num_neighbors, d, dat, gmm)

    if not success:
        raise ValueError(
            f"Only {len(inx)} of {n} GMM clusters have valid sensor locations."
        )

    return d.loc[inx].copy()


def GP_step(x, y, f, step_list, delta_list, rmse_list, score_list, bias_list, measured=False):
    """
    Parameters
    ----------
    x : Pandas DataFrame
        The full dataset of the region.
    y : Pandas DataFrame
        Dataframe containing GMM locations and corresponding snow depth measurements.
    f : list[str]
        List of features to be used in the GPR model.
    measured : bool, optional
        If True, snow depths measured outside the test region will be used. Default is False.

    Returns
    -------
    Pandas DataFrame
        A dataframe containing GPR predictions for snow depth, along with performance metrics.
    """

    # Create a copy of the y dataframe and drop rows with NaN snow_depth values.
    y_nonans = y.copy().dropna()

    # Split the data into training and test sets based on the GMM locations.
    x_train = y_nonans[f]
    y_train = y_nonans.true_snow_depth.values.reshape(-1,1)
    if measured == True:
        x_test = x.drop(x.iloc[y.index].index.values)
    else:
        x_test=x.copy()

    #set kernel and length scales
    
    #Run gpr with gridsearch to determine optimal parameters
    gpr_cv = GaussianProcessRegressor( random_state=42) 
    param_grid = [{'n_restarts_optimizer':[50,100,200]}]
    
    param_grid = [{'kernel': [RBF(l) for l in np.logspace(-2, 0, 10)], 
                                  'n_restarts_optimizer':[10,50,100]}]

    gp_gs = GridSearchCV(estimator=gpr_cv, param_grid=param_grid, scoring='neg_mean_absolute_error')
    gp_gs.fit(x_train, y_train)
    print(gp_gs.best_params_)
    print(gp_gs.best_score_)
    score = gp_gs.best_score_
    allscores=gp_gs.cv_results_['mean_test_score']
    print(allscores)

    #Estimate snow depth and std at everey cell in the raster except for the training sites
    y_predict = gp_gs.best_estimator_.predict(x_test[f])
    
    # Correct negative predictions to zero.
    y_predict[y_predict <0]=0
    
    #define the output df with snow estimates and std values
    out = x_test.copy()
    
    out['snow_depth_est']=y_predict
    out['depth_delta'] = out['snow_depth_est'] - out['true_snow_depth']

    print("learned kernel params")
    print(gp_gs.get_params())
    
    # Calculate and print performance metrics (average delta, RMSE, bias).
    avg_delta = out['depth_delta'].mean() #mean bias error
    print('avearge_delta:', avg_delta)
    rmse = root_mean_squared_error(out['snow_depth_est'],out['true_snow_depth'])
    print('rmse:', rmse)
#     bias = sum(out['depth_delta'])
    bias =out['depth_delta'].mean() #MBE
    print('bias:', bias)
    
    # Append metrics to respective lists for tracking across iterations.
    step_list.append(len(y_train))
    delta_list.append(avg_delta)
    rmse_list.append(rmse)
    score_list.append(score)
    bias_list.append(bias)
    
    return out, step_list, delta_list, rmse_list, score_list, bias_list

