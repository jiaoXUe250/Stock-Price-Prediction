# Stock-Price-Prediction
# Time Series Prediction using PyTorch

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
   git clone https://github.com/yourusername/time-series-pytorch.git
   cd time-series-pytorch
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

## Citations
If you use this code in academic research, please cite:  
```
@misc{time-series-pytorch,
  author = {Your Name},
  title = {Time Series Prediction using PyTorch},
  year = {2025},
  howpublished = {\url{https://github.com/yourusername/time-series-pytorch}}
}
```  

## License
This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.  

## Contribution Guidelines
- Fork the repository.  
- Create a new branch for your feature or bug fix.  
- Submit a pull request with a detailed description.  

