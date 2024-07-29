## Description

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





### PM2.5

1. Navigate to the `PM2.5` directory under `RealData`.

2. Process the data using `process.ipynb`. The dataset is publicly available in the paper by Zheng, X. & Chen, S. X. (2024) and can be downloaded from [this repository](https://github.com/FlyHighest/Dynamic-Synthetic-Control). The `data` folder has contained both the raw data and the preprocessed data. You can directly use the preprocessed data, or you can also redownload the data and use the provided code to preprocess.
   - Zheng, X. & Chen, S. X. (2024), ‘Dynamic synthetic control method for evaluating treatment effects in auto-regressive processes’, Journal of the Royal Statistical Society Series B: Statistical Methodology 86(1), 155–176.

3. Launch Jupyter Notebook and open the training notebooks:
   - For training DNN estimator, linear regression, RKHS regression, and local linear regression and seeing the results, use `train.ipynb`.

4. For training additive model and single index model, use `AM&SIM.Rmd`. The additive model training uses the R package ‘PLSiMCpp’ (Wu et al. 2022), and the single index model training uses the R package ‘gam’ (Hastie 2023). The references for these packages are:
   - [Wu, S., Zhang, Q., Li, Z. & Liang, H. (2022), PLSiMCpp: Methods for Partial Linear Single Index Model. R package version 1.0.4.](https://cran.r-project.org/web/packages/PLSiMCpp/index.html)
   - [Hastie, T. (2023), gam: Generalized Additive Models. R package version 1.22-3.](https://cran.r-project.org/web/packages/gam/index.html)

