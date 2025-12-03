# S&P 500 Stock Predictor Dashboard

A web-based stock prediction dashboard built with Taipy that uses machine learning models to predict S&P 500 stock prices. The application provides an interactive interface for analyzing historical stock data and comparing predictions from multiple ML models.

![Python](https://img.shields.io/badge/python-3.11-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.18.0-orange.svg)
![Taipy](https://img.shields.io/badge/Taipy-4.0.0-red.svg)

## Features

- **Multiple ML Models**: Compare predictions from three different algorithms:
  - Linear Regression
  - K-Nearest Neighbors (KNN)
  - Neural Network (Feedforward Dense Network)
  
- **Interactive Dashboard**: Built with Taipy for a modern, responsive web interface

- **Country-based Filtering**: Filter companies by country of origin

- **Historical Data Visualization**: Interactive Plotly charts showing actual vs predicted stock prices

- **Configurable Date Ranges**: Select custom date ranges for predictions

## Tech Stack

- **Python 3.11**
- **Taipy 4.0.0** - GUI framework and orchestration
- **TensorFlow 2.18.0** - Neural network implementation
- **Scikit-learn 1.5.2** - Linear Regression and KNN models
- **Plotly 5.24.1** - Interactive visualizations
- **Pandas** - Data processing

## Installation

### Prerequisites

- Python 3.11 or higher
- pip package manager
- Git

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/mikeananou/SP500PredictorDashboard.git
   cd SP500PredictorDashboard
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv .venv
   ```

3. **Activate the virtual environment**
   
   On Windows:
   ```bash
   .venv\Scripts\activate
   ```
   
   On macOS/Linux:
   ```bash
   source .venv/bin/activate
   ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the Application

1. **Start the dashboard**
   ```bash
   python main.py
   ```

2. **Access the application**
   
   Open your web browser and navigate to:
   ```
   http://localhost:5000
   ```

3. **Using the Dashboard**
   - Select a country from the dropdown
   - Choose a company from the available options
   - Set your desired date range for predictions
   - Click "Analyze" to generate predictions
   - Compare results from all three ML models

### Stopping the Application

Press `Ctrl+C` in the terminal to stop the server.

## Project Structure

```
SP500PredictorDashboard/
├── main.py                    # Main application file
├── requirements.txt           # Python dependencies
├── data/
│   ├── sp500_companies.csv   # Company metadata
│   ├── sp500_index.csv       # S&P 500 index data
│   └── sp500_stocks.csv      # Historical stock prices
├── images/
│   ├── flags/                # Country flag images
│   └── icons/                # UI icons
└── README.md                 # This file
```

## Data

The application uses three CSV files:

- **sp500_companies.csv**: Contains company information (Symbol, Name, Country)
- **sp500_stocks.csv**: Historical stock data (Date, Symbol, Open, High, Low, Close, Volume)
- **sp500_index.csv**: S&P 500 index historical data

### Currently Supported Companies

- **United States**: Apple (AAPL), Microsoft (MSFT), Alphabet (GOOGL), Amazon (AMZN), Tesla (TSLA)
- **Canada**: Lululemon Athletica (LULU)

## Model Details

### Linear Regression
A simple linear model that assumes a linear relationship between input features and stock prices.

### K-Nearest Neighbors (KNN)
A non-parametric method that predicts based on the k-nearest training examples in the feature space.

### Neural Network
A feedforward neural network with:
- 64 hidden units
- Dense layers
- RMSprop optimizer
- 10 epochs training
- Batch size of 32

## Configuration

Key constants can be modified in `main.py`:

```python
# Model hyperparameters
NN_BATCH_SIZE = 32
NN_EPOCHS = 10
NN_HIDDEN_UNITS = 64

# Server configuration
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 5000
```

## Development

### Code Structure

- **Constants Section** (Lines 60-97): Centralized configuration
- **Data Loading** (Lines 99-129): CSV loading with error handling
- **Prediction Functions**: Consolidated prediction logic
- **GUI Builder**: Taipy-based interface construction
- **Orchestration**: Task and scenario management

### Recent Improvements

✅ Extracted constants for better maintainability  
✅ Added comprehensive error handling  
✅ Added type hints to all functions  
✅ Consolidated prediction logic  
✅ Renamed RNN to Neural Network for accuracy  
✅ Changed to single company selection  
✅ Fixed graph clearing on country change  

## Troubleshooting

### Port Already in Use
```bash
# Windows
Get-Process python | Stop-Process -Force

# macOS/Linux
killall python
```

### Taipy Cache Issues
If you encounter "Task not found" errors:
```bash
# Remove Taipy cache directories
rm -rf .taipy user_data
```

### TensorFlow Issues
The application will fall back gracefully if TensorFlow is not available, but Neural Network predictions will be disabled.

## Future Enhancements

- [ ] Kaggle dataset integration for daily automatic updates
- [ ] Expand company coverage to full S&P 500
- [ ] Add more ML models (LSTM, Random Forest, etc.)
- [ ] Export predictions to CSV
- [ ] Model performance metrics (RMSE, MAE, R²)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License.

## Author

**Akakpo Mike Ananou**

## Acknowledgments

- Built as part of CS7050 Data Warehousing & Data Mining course at Kennesaw State University
- Stock data sourced from S&P 500 historical records
- Powered by the Taipy framework

## Support

For issues or questions, please open an issue on the [GitHub repository](https://github.com/mikeananou/SP500PredictorDashboard/issues).

---

**Note**: Stock predictions are for educational purposes only and should not be used as financial advice.
