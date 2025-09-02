# Stock-Price-Prediction
# Stock-Price-Prediction using PyTorch

## Description
This project implements a **time series prediction model** using **PyTorch**, integrating data preprocessing, model training, and evaluation.  
The workflow includes:  
- Data loading and normalization using **Pandas** and **MinMaxScaler**.  
- Custom `Dataset` class for PyTorch data pipeline.  
- Model design with **LSTM/MLP** layers for sequence prediction.  
- Visualization of prediction results using **Matplotlib** and **Seaborn**.  

## Dataset Information
- Input: Tabular stock series data (CSV or other structured formats).  
- Features: Numerical values for each time step.  
- Target: The variable to be predicted (e.g., stock price, sensor value).  
- Preprocessing: Min–Max scaling applied to all numeric features.  

## Code Information
- **`Dataset` class** for loading and preprocessing the dataset.  
- **Model definition** in PyTorch (`torch.nn.Module`).  
- **Training loop** with loss computation and optimizer updates.  
- **Evaluation metrics**: MAE, MSE.  
- **Visualization scripts** for plotting predictions vs. ground truth.  

## Usage Instructions
1. **Clone this repository**  
   ```bash
   git clone https://github.com/jiaoXUe250/Stock-Price-Prediction.git
   ```  

2. **Prepare dataset**  
   - Place your CSV file inside the `data/` folder.  
   - Update the dataset path in the script.  

3. **Run the code**  
   ```bash
   python code.py
   ```  

4. **View results**  
   - Metrics (MAE, MSE) will be printed in the console.  
   - Prediction plots will be saved in the `results/` folder.  

## Requirements
```txt
pandas
numpy
torch
scikit-learn
matplotlib
seaborn
```  
Install them with:  
```bash
pip install -r requirements.txt
```  

## Methodology
1. **Data Loading**: Load CSV data using Pandas.  
2. **Scaling**: Apply Min–Max scaling for feature normalization.  
3. **Dataset Preparation**: Create a sliding window for sequence input.  
4. **Model Training**: Train the PyTorch model using an optimizer and loss function.  
5. **Evaluation**: Compute MAE, MSE, and visualize predictions.

## Materials & Methods
Computing Infrastructure:
Operating System: Windows 10
Hardware: Intel i7 CPU, 16GB RAM, NVIDIA RTX 3060 GPU (12GB VRAM)
Software: Python 3.9, PyTorch 1.13+, CUDA 11.6 (if GPU available)
Model Implementation: PyTorch nn.Module with Transformer layers.


## License
This project is licensed under the MIT License.



