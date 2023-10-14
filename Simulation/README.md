## Description

- **Cases (1-6)**: Each `Case` directory corresponds to one of the 6 simulations from the article:
  - `data.py`: Script for data generation.
  - `train.py`: Script for model training.
  - `data/`: Stores generated data.
  - `res/`: Stores training results.

- **plot.ipynb**: Visualizes final results from the simulations.


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
4. After training, results will be saved in the `res/` directory.
5. For visualizing the results, launch Jupyter and use `plot.ipynb`.
