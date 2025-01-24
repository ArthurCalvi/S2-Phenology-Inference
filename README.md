# S2-Phenology-Inference

This repository contains a set of scripts and modules to estimate phenology from Sentinel-2 data. The workflow includes preparing inference configurations, collecting and downloading DEM tiles, performing inference in parallel (SLURM-compatible), and visualizing outputs. 

## Directory Structure

```
arthurcalvi-s2-phenology-inference/
├── README.md
├── jobs/
│   ├── run_comparison.sh
│   ├── run_infenrence.sh
│   └── run_prepare_inference.sh
├── src_inference/
│   ├── collect_dem_info.py
│   ├── download_dem.py
│   ├── download_dem_chronos.py
│   ├── inference.py
│   ├── plotting.py
│   ├── prepare_inference.py
│   ├── run_inference.py
│   └── utils.py
└── test/
    ├── test_dem_pipeline.py
    ├── test_fit_periodic_robustness_to_nan_with_weights.py
    ├── test_folder_inference.py
    ├── test_geodataframe_to_raster.py
    ├── test_prepare_inference.py
    ├── test_run_inference.py
    ├── test_tile_inference.py
    ├── test_tile_inference_chronos.py
    └── test_window_inference.py
```

### Contents

1. **jobs/**  
   Contains SLURM job scripts to:
   - Prepare inference runs.
   - Execute the inference pipeline on HPC (JeanZay or similar).
   - Compare or evaluate results.

2. **src_inference/**  
   Main source code for phenology inference:
   - **collect_dem_info.py**: Collect DEM information (bounds, transformations) for each tile.  
   - **download_dem.py** and **download_dem_chronos.py**: Scripts that download and prepare DEM data using third-party libraries (e.g., `multidem`).  
   - **inference.py**: Main inference logic, including classes for tile-based inference, window-based inference, folder-level inference, etc.  
   - **plotting.py**: Visual tools to plot phenology predictions over an RGB basemap.  
   - **prepare_inference.py**: Prepare inference configurations (tile lists, metadata, etc.) for subsequent SLURM job array runs.  
   - **run_inference.py**: Entry point to run inference on a single tile, using the prepared configurations.  
   - **utils.py**: Shared utility functions for time-series modeling, data transformations, spectral index computation, and more.

3. **test/**  
   Unit and integration tests ensuring the pipeline works correctly:
   - **test_dem_pipeline.py**: Tests DEM collection and downloading procedures.  
   - **test_fit_periodic_robustness_to_nan_with_weights.py**: Verifies the robustness of periodic function fitting in the presence of NaNs or varying weights.  
   - **test_folder_inference.py**, **test_tile_inference.py**, **test_window_inference.py**: Validate inference logic for folder-level, tile-level, and window-level approaches.  
   - **test_prepare_inference.py**, **test_run_inference.py**: Test the scripts that build and execute the final inference pipeline.  
   - **test_geodataframe_to_raster.py**: Example test for converting vector data to raster.

## Quick Start

1. **Clone the repository**:
   ```
   git clone https://github.com/<user>/arthurcalvi-s2-phenology-inference.git
   ```
2. **Install dependencies**. Ensure you have Python 3.9+ available with relevant libraries (e.g., NumPy, rasterio, pandas, scikit-learn, etc.). You can use:
   ```
   pip install -r requirements.txt
   ```
   or set up a conda environment with the necessary packages.

3. **Prepare your data**:  
   - Store Sentinel-2 mosaic files in a structured directory:  
     ```
     [year]/[mm-dd_plus_minus_30_days]/s2/s2_{tile_id}.tif
     ```
   - Provide a corresponding DEM directory with `dem_{tile_id}.tif` for each tile.

4. **Configure and run inference**:
   - **Step A: Prepare**  
     Use the script "prepare_inference.py" to detect tiles, gather years, build the config files, and store everything in an output folder:
     ```
     python src_inference/prepare_inference.py \
         --mosaic-dir /path/to/mosaics \
         --dem-dir /path/to/dem/files \
         --output-dir /path/to/output \
         --model-path /path/to/model.pkl \
         --years 2021 2022 \
         --window-size 1024
     ```
     This command generates a set of JSON configurations for each tile, plus a summary file for scheduling jobs.

   - **Step B: Run the inference**  
     Depending on your platform, you can run inference with SLURM or locally. For SLURM:
     ```
     sbatch jobs/run_infenrence.sh
     ```
     or, for a single tile:
     ```
     python src_inference/run_inference.py \
         --config-dir /path/to/phenology_inference/configs \
         --tile-idx 0
     ```

5. **Output**:  
   - A TIFF file containing the probability map (or classification) is generated for each tile.  
   - You can visualize these outputs via "plotting.py" or additional scripts in the "test/" folder.

## Overview of the Workflow

1. **DEM Pipeline**  
   - "collect_dem_info.py" scans for all mosaic tiles and collects bounding info.  
   - "download_dem.py" uses bounding info and downloads DEM data (SRTM30 by default).  
   - Each tile’s DEM is reprojected and clipped to match the mosaic geometry.

2. **Time Series & Features**  
   - For each tile, the code reads reflectance bands from multiple dates.  
   - Spectral indices (e.g., NDVI, EVI, NBR, CRSWIR) are computed using the functions in "utils.py".  
   - An iterative robust fitting (Fourier harmonics) extracts key temporal features (amplitude, phase).

3. **Model Inference**  
   - A trained classifier or regressor (e.g., a scikit-learn model stored as "model.pkl") processes the extracted features.  
   - The script outputs a georeferenced probability/phenology map per tile.

4. **Post-Processing & Visualization**  
   - "plotting.py" enables quick side-by-side plots of RGB, DEM shading, and the inferred phenology map for quality control.

## Tests

- **Pytest or unittest**: All tests in "test/" can be run as:
  ```
  pytest test/
  ```
  or
  ```
  python -m unittest discover test
  ```
  This ensures data reading, transformation, and inference routines behave as expected.

## Dependencies

- Python >= 3.9  
- **rasterio** (reading/writing geospatial rasters)  
- **numpy, pandas** (array and data management)  
- **joblib** (parallelization)  
- **scikit-learn** (machine learning for classification/regression)  
- **matplotlib** (visualization)  
- **tqdm** (progress bars)

## Contributing

1. **Fork** the repository and create a feature branch.  
2. **Add** or **edit** functionalities in "src_inference/" or "test/" while maintaining style and clarity.  
3. **Submit** a pull request, tagging relevant maintainers for code review.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- CNRS and other institutional resources used for HPC-based runs.  
- Remote sensing libraries and open-source communities that facilitate data processing.

