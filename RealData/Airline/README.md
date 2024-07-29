## Description

- **Airline**:
  - `process.r`: Processes the data, which is then stored in the `data/` directory.
  - `nmtrain.ipynb`: Training on various sample sizes and sampling frequencies.
  - `fulltrain.ipynb`: Training on the full dataset.
  - `resultsv/` is required for training, while final models are saved in `bestnet/`.
  - `result.ipynb`: Visualizes final results, including fit plots and multidimensional scaling plot.
  


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




### Airline

1. Navigate to the `Airline` directory under `RealData`.

2. The processed data is readily available for usage, stored in the `datahao` sub-directory. Hence, if you want to save time, you can skip the processing step.  If you can want to process the data independently: Please download the 2008 dataset from [here](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/HG7NV7), place the `2008.csv` file in `data\`, and then run `process.r`.

3. Launch Jupyter Notebook and open the training notebooks:
- For training on various sample sizes and sampling frequencies, please use `nmtrain.ipynb`.

- For training on the full dataset, please use `fulltrain.ipynb`.

4. The finalized models are saved in the `bestnet/` directory.

5. For visualization, please run `result.ipynb`.


