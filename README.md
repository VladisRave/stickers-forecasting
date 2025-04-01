# Kaggle Time Series Forecasting Project

![Python](https://img.shields.io/badge/Python-3.11%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.6.0-red)
![XGBoost](https://img.shields.io/badge/XGBoost-2.1.4-green)

This repository contains exploratory data analysis (EDA) and the implementation of two forecasting models for time series data. The project compares XGBoost (RMSE = 100) and an LSTM neural network (RMSE = 80) for stickers sales prediction from Forecasting Sticker Sales contest. This is my first project on time series prediction.

## Features
- Comprehensive EDA with three interpolation methods
- XGBoost baseline model implementation
- LSTM neural network with PyTorch
- Model comparison and evaluation
- Kaggle submission pipeline

## Installation

### Prerequisites
- Python 3.11+
- CUDA-enabled GPU (recommended)
- NVIDIA drivers compatible with PyTorch 2.6.0

```bash
# Clone repository
git clone https://github.com/yourusername/time-series-forecasting.git
cd time-series-forecasting

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/MacOS
# venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## Project Structure
```
├── data/                   # Raw and processed datasets
│   ├── data_after_eda.csv/                
│   ├── test.csv/                
│   └── train.csv/          
├── results/                # Training artifacts
│   ├── best_model.pth      
│   └── scaler_params.pkl   
├── notebooks/              # Jupyter notebooks
│   ├── data_analysis.ipynb 
│   └── train.ipynb         
├── requirements.txt        
└── README.md               
```

## Model Details

### XGBoost Baseline
- Used for initial benchmarking
- Feature engineering with time-series characteristics
- RMSE: 100

### LSTM Architecture
```python
class LSTMModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(x)
        output = self.dropout(output[:, -1, :])
        return self.fc(output)
```

**Hyperparameters:**
```python
num_layers = 6
window_size = 7
hidden_size = 512
```

## Results

| Metric   | XGBoost | LSTM    |
|----------|---------|---------|
| RMSE     | 100.37  | 87.74   |
| MAE      | 59.34   | 50.32   |
| MAPE     | 26.71%  | 7.35%   |
| SMAPE    | 24.42%  | 7.44%   |

**Kaggle Submission:**
- Best competition score: 0.99 (LSTM model)

## Notebooks

### `data_analysis.ipynb`
- Contains full EDA for handling NaN values in the `num_sold` column.
- Analyzes three interpolation methods: linear, spline, and polynomial.
- Includes visualizations for different approaches.

### `train.ipynb`
- Implements feature engineering to improve model accuracy.
- Trains an XGBoost model as a benchmark before applying LSTM.
- Evaluates multiple interpolation methods for best results.
- Implements LSTM architecture for time series forecasting.
