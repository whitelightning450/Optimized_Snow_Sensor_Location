# Optimized Snow Sensor Location

Python functions for selecting representative snow measurement locations and estimating snow depth using Gaussian-based machine learning.

This project supports research into improving snowpack monitoring in complex mountain terrain with a limited number of field observations.

## Research Background

Mountain snow depth varies with elevation, terrain orientation, vegetation, and wind exposure. Traditional monitoring stations cannot capture all this spatial variability.

The workflow described by Liljestrand et al. (2025) combines:

- Snow-free lidar-derived terrain characteristics
- Gaussian Mixture Models (GMMs) for representative sampling
- Field snow-depth measurements
- Gaussian Process Regression (GPR) for spatial prediction

The study evaluated this approach in Franklin Basin near the Utah–Idaho border, including Boss Canyon, Peterson Hollow, and Hell's Kitchen Canyon.

## Research Workflow

### 1. Prepare terrain features

Derive physiographic variables from snow-free lidar and associated terrain data.

The published study used:

- Elevation
- Northness
- Eastness
- Canopy height
- Wind shelter

Northness and eastness describe slope orientation and steepness. Terrain features are normalized before GMM fitting.

### 2. Identify representative sampling locations

A Gaussian Mixture Model groups the terrain feature space into a specified number of components.

The model identifies representative cluster centers, then uses nearest-neighbor searching to select actual candidate locations.

Sampling restrictions can exclude unsuitable terrain. In the current implementation, the `No_go` column uses:

- `1`: permitted sampling location
- `0`: restricted sampling location

This convention is specific to the current code.

The published study investigated excluding terrain with slopes of 30 degrees or greater as a simplified avalanche-risk screening criterion. This threshold does not guarantee field safety.

### 3. Obtain snow-depth observations

Collect snow-depth measurements at the selected locations.

The published workflow used snow probe observations. Snow-on lidar-derived snow depth was also used to evaluate model predictions.

### 4. Estimate snow depth

Gaussian Process Regression relates the measured snow depths to the terrain features and predicts snow depth across the target domain.

The implementation uses scikit-learn's GaussianProcessRegressor and GridSearchCV.

### 5. Evaluate predictions

Compare modeled snow depth with reference observations and calculate prediction errors.

The current code calculates:

- Mean prediction difference (bias)
- Root mean squared error (RMSE)
- Cross-validation score

The published study also reports additional evaluation metrics, including Kling-Gupta efficiency (KGE). These are not all implemented in this repository.

## Repository Structure

```text
Optimized_Snow_Sensor_Location/
├── README.md
├── .gitignore
└── src/
    └── funcs.py
```

The repository currently contains reusable Python functions. It does not include the complete preprocessing pipeline, example datasets, notebooks, or an executable end-to-end workflow.

## Python Functions

All functions are located in `src/funcs.py`.

| Function | Description |
| --- | --- |
| `neighbors()` | Finds nearest candidate locations to fitted GMM cluster centers using a BallTree. |
| `modified_sample_loc()` | Selects permitted sampling locations using the `No_go` column. |
| `gmm_step()` | Subsamples and scales terrain features, fits a GMM, and returns selected locations. |
| `GP_step()` | Fits a Gaussian Process regression model, predicts snow depth, and calculates evaluation metrics. |

## Dependencies

Core Python dependencies used by `src/funcs.py`:

- NumPy
- pandas (for input DataFrames)
- scikit-learn

The code has been checked with scikit-learn 1.7.2.

## Input Data Requirements

### GMM sensor selection

`gmm_step(d, ss, n, f, num_neighbors)` requires:

- `d`: DataFrame of candidate locations and terrain features
- `ss`: fraction of candidate rows used for model fitting
- `n`: number of GMM components
- `f`: list of terrain feature column names
- `num_neighbors`: number of candidate neighbors examined per cluster

The input DataFrame must contain the requested feature columns and a `No_go` column. Its index identifies candidate locations.

The function returns a DataFrame containing selected locations or raises a `ValueError` if suitable locations cannot be found for every cluster.

### Gaussian Process prediction

`GP_step(x, y, f, step_list, delta_list, rmse_list, score_list, bias_list, measured=False)` requires:

- `x`: prediction-domain DataFrame
- `y`: DataFrame of measurement locations
- `f`: predictor feature column names
- Metric lists for recording results

The measurement DataFrame must contain `true_snow_depth`. The prediction-domain DataFrame must also contain `true_snow_depth` for the current evaluation calculations.

The function returns the prediction DataFrame and updated metric lists.

## Current Limitations

- Input datasets and preprocessing scripts are not included.
- The full published workflow cannot be reproduced from this repository alone.
- GMM sampling requires a precomputed restriction mask.
- The Gaussian Process grid search may be computationally expensive.
- Real-world results depend on the input data, sampling design, and validation approach.
- Field safety requires assessment beyond the simplified terrain restrictions used in the model.

## Reference

Liljestrand, D., Johnson, R., Neilson, B., Strong, P., and Cotter, E. (2025). Leveraging snow probe data, lidar, and machine learning for snow depth estimation in complex-terrain environments. *The Cryosphere*, 19, 3123–3138.

https://doi.org/10.5194/tc-19-3123-2025
