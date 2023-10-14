# Deep-regression-for-repeated-measurements

This repository contains code and results for simulations and real data analyses referenced in the associated research paper. 
It is structured into two main directories: `Simulation` and `RealData`.

## Directory Structure
- 📂 **.**
  - 📂 **Simulation**
    - 📂 **case1** ... **case6**
      - 📜 `data.py`
      - 📜 `train.py`
      - 📂 **data**
      - 📂 **res**
    - 📂 **resultsv**
    - 📜 `plot.ipynb`
  - 📂 **RealData**
    - 📂 **Airline**
      - 📜 `process.r`
      - 📜 `nmtrain.ipynb`
      - 📜 `fulltrain.ipynb`
      - 📜 `vis.ipynb`
      - 📂 **data**
      - 📂 **resultsv**
      - 📂 **bestnet**
    - 📂 **Argo**
      - 📜 `process.r`
      - 📜 `nmtrain.ipynb`
      - 📜 `fulltrain.ipynb`
      - 📂 **data**
      - 📂 **resultsv**
      - 📂 **bestnet**
     
  
## Description
### Simulation

- **function/**: 
  - `trainfun.py`: Contains utility functions, including neural network training functions.

- **Cases (1-6)**: Each `Case` directory corresponds to one of the 6 simulations from the article:
  - `data.py`: Script for data generation.
  - `train.py`: Script for model training.
  - `data/`: Stores generated data.
  - `res/`: Stores training results.

- **plot.ipynb**: Visualizes final results from the simulations.

### Real Data Analyses

- **Airline**:
  - `process.r`: Processes the data, which is then stored in the `data/` directory.
  - `nmtrain.ipynb`: Training on various sample sizes and sampling frequencies.
  - `fulltrain.ipynb`: Training on the full dataset.
  - `resultsv/` is required for training, while final models are saved in `bestnet/`.
  - `vis.ipynb`: Visualizes final results, including fit plots and multidimensional scaling plot.
  
- **Argo**:
  - `process.r`: Data download and processing script, which would be saved in `data/`. 
  - `nmtrain.ipynb`: Training on various sample sizes and sampling frequencies with visualization of estimations.
  - `fulltrain.ipynb`: Training on the full dataset with visualization of the estimation.
  - `resultsv/` is required for training, while final models are saved in `bestnet/`.


## How to Use

### Utilizing GPU Resources in Training

In the training scripts located within the `Simulation` directory or `RealData` directory, we utilize a variable named `nocuda` to manage the usage of GPU resources during the execution. Here’s how the `nocuda` variable affects the computation:

- **`nocuda = -1`**: The code will run on the CPU.
- **`nocuda = 0`** or **`nocuda = 1`**: The code will run on GPU 0 or GPU 1, respectively.
- **`nocuda = 100`**: The script will automatically select the GPU (between 0 and 1) with more available memory to run the code.
- **`nocuda = 9`**: Use this option if you intend to manually specify the GPU. Add the following lines at the top of your `.py` or `.ipynb` files, where `"i"` should be replaced with your chosen GPU identifier (0 or 1).
  ```python
  import os
  os.environ["CUDA_VISIBLE_DEVICES"] = "i"  ## i = 0 or 1
  
Note: Ensure that these lines are commented out if you’re using any nocuda value other than 9, to prevent any potential issues or errors during execution.

Ensure that you configure the nocuda variable as per your computational resource availability and requirements before initiating the training scripts.

### Simulation

1. Navigate to the desired `case` directory within the `Simulation` folder.
2. Run `data.py` to generate the required data.
3. Execute `train.py` to start the model training process. 
4. After training, results will be saved in the `res/` sub-directory.
5. For visualizing the results, launch Jupyter and use `plot.ipynb`.

### Real Data Analyses 

#### Airline

1. Navigate to the `Airline` directory under `RealData`.
2. Process the data using the provided R script: `process.r`. The [original data](https://community.amstat.org/jointscsgsection/dataexpo/dataexpo2009) and preprocessed data have been already placed in the respective folders for ease of use. Hence, you can skip this step to save time.
3. Launch Jupyter Notebook and open the training notebooks:
- For training on various sample sizes and sampling frequencies, please use `nmtrain.ipynb`.

- For training on the full dataset, please use `fulltrain.ipynb`.

4. The finalized models are saved in the `bestnet/` directory.
5. For visualization, please run `vis.ipynb`.


#### Argo

1. Navigate to either the `Airline` or `Argo` directory under `RealData`.
2. Process the data using the provided R script: `process.r`. The data processing script `process.r` makes use of the [`argoFloats`](https://cran.r-project.org/web/packages/argoFloats/index.html) R package. The `argoFloats` package provides a suite of functions for obtaining and working with Argo oceanographic measurement data. This includes capabilities for downloading, plotting, and managing Argo float datasets, which are pivotal for ocean and climate studies. For more information, please see [here](https://github.com/ArgoCanada/argoFloats). 
The package accesses data primarily from the [Argo Data](https://data-argo.ifremer.fr), you might also explore and directly download data from the [Argo Data website](https://data-argo.ifremer.fr) as per your specific requirements.
For a straightforward and user-friendly experience, the results generated by the `process.r` script are conveniently stored in the respective `data` folder.
Hence, you can skip this step to save time.

3. Launch Jupyter Notebook and open the training notebooks:
- For training on various sample sizes and sampling frequencies and seeing the result, please run `nmtrain.ipynb`.

- For training on the full dataset and seeing the result, please run `fulltrain.ipynb`.

4. You can find the finalized models in the `bestnet/` directory.
