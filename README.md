# Deep-regression-for-repeated-measurements

This repository contains code and results for simulations and real data analyses referenced in [the associated research paper](https://www.tandfonline.com/doi/full/10.1080/01621459.2025.2458344). 

Shunxing Yan, Fang Yao, & Hang Zhou. (2025). Deep Regression for Repeated Measurements. Journal of the American Statistical Association.

It is structured into two main directories: `Simulation` and `RealData`.

## Directory Structure
- 📂 **.**
  - 📂 **Simulation**
    - 📂 **Case1**, **Case2**, **Case3**, **Case4/10d**, **Case4/20d**, **Case4/30d**, **CaseS1**, **CaseS2**, **CaseS3**
      - 📜 `data.py`
      - 📜 `train.py`
      - 📜 `localinear.py`
      - 📜 `RKHS.py`
      - 📜 `spline.py`
      - 📂 **data**
      - 📂 **res**
    - 📂 **resultsv**
    - 📜 `result.ipynb`
  - 📂 **RealData**
    - 📂 **Airline**
      - 📜 `process.r`
      - 📜 `nmtrain.ipynb`
      - 📜 `fulltrain.ipynb`
      - 📜 `result.ipynb`
      - 📂 **data**
      - 📂 **resultsv**
      - 📂 **bestnet**
    - 📂 **PM2.5**
      - 📜 `process.ipynb`
      - 📜 `train.ipynb`
      - 📜 `AM&SIM.Rmd`
      - 📂 **data**
      - 📂 **resultsv**

## Description
### Simulation

- **Cases (1-3), Case4 (10d-30d) and Supplementary Cases (S1-S3)**: Each `Case` directory corresponds to one of the simulations from the paper:
  - `data.py`: Script for data generation.
  - `train.py`: Trains DNN estimators.
  - `localinear.py`: Trains local linear estimators.
  - `RKHS.py`: Trains RKHS estimators.
  - `spline.py`: Trains regression spline estimators for examples with $d\leq 5$.
  - `data/`: Stores generated data.
  - `res/`: Stores training results.

- **result.ipynb**: Visualizes and summarizes final results from the simulations.
- **resultsv/**: Stores intermediate results during the training process.

### Real Data Analyses

- **Airline**:
  - `process.r`: Processes the data, which is then stored in the `data/` directory.
  - `nmtrain.ipynb`: Training on various sample sizes and sampling frequencies.
  - `fulltrain.ipynb`: Training on the full dataset.
  - `resultsv/` is required for training, while final models are saved in `bestnet/`.
  - `result.ipynb`: Visualizes final results, including fit plots and multidimensional scaling plot.
  
- **PM2.5**:
  - `process.ipynb`: Preprocesses the data and saves the processed data as `.csv` files in the `data/` directory.
  - `train.ipynb`: Trains DNN estimator, linear regression, RKHS regression, and local linear regression models.
  - `AM&SIM.Rmd`: Trains additive model and single index model.
  - `data/`: Stores raw and processed data.

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

1. Navigate to the desired `Case` directory within the `Simulation` folder.
2. Run `data.py` to generate the required data.
3. Execute `train.py`, `localinear.py`, `RKHS.py`, and `spline.py` to train the respective estimator results. 
4. After training, results will be saved in the `res/` directory.
5. For visualizing the results, launch Jupyter and use `result.ipynb`.


### Real Data Analyses 

#### Airline

1. Navigate to the `Airline` directory under `RealData`.

2. The processed data is readily available for usage, stored in the `datahao` sub-directory. Hence, if you want to save time, you can skip the processing step.  If you want to process the data independently: Please download the 2008 dataset from [[here](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/HG7NV7)](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/HG7NV7), place the `2008.csv` file in `data\`, and then run `process.r`.

3. Launch Jupyter Notebook and open the training notebooks:
- For training on various sample sizes and sampling frequencies, please use `nmtrain.ipynb`.

- For training on the full dataset, please use `fulltrain.ipynb`.

4. The finalized models are saved in the `bestnet/` directory.

5. For visualization, please run `result.ipynb`.


#### PM2.5

1. Navigate to the `PM2.5` directory under `RealData`.

2. Process the data using `process.ipynb`. The dataset is publicly available in the paper by Zheng, X. & Chen, S. X. (2024) and can be downloaded from [this repository](https://github.com/FlyHighest/Dynamic-Synthetic-Control). The `data` folder has contained both the raw data and the preprocessed data. You can directly use the preprocessed data, or you can also redownload the data and use the provided code to preprocess.
     - [Zheng, X. & Chen, S. X. (2024), ‘Dynamic synthetic control method for evaluating treatment effects in auto-regressive processes’, Journal of the Royal Statistical Society Series B: Statistical Methodology 86(1), 155–176.](https://academic.oup.com/jrsssb/article-abstract/86/1/155/7331057)

3. Launch Jupyter Notebook and open the training notebooks:
   - For training DNN estimator, linear regression, RKHS regression, and local linear regression and seeing the results, use `train.ipynb`.

4. For training additive model and single index model, use `AM&SIM.Rmd`. The additive model training uses the R package ‘PLSiMCpp’ (Wu et al. 2022), and the single index model training uses the R package ‘gam’ (Hastie 2023). The references for these packages are:
   - [Wu, S., Zhang, Q., Li, Z. & Liang, H. (2022), PLSiMCpp: Methods for Partial Linear Single Index Model. R package version 1.0.4.](https://cran.r-project.org/web/packages/PLSiMCpp/index.html)
   - [Hastie, T. (2023), gam: Generalized Additive Models. R package version 1.22-3.](https://cran.r-project.org/web/packages/gam/index.html)

