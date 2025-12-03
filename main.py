# Stock Data Dashboard Application
# Author: Akakpo Mike Ananou

# Ensure we're in the correct directory for relative imports
import os
import sys
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir:
        os.chdir(script_dir)
        if script_dir not in sys.path:
            sys.path.insert(0, script_dir)

# GUI imports
import taipy as tp
import taipy.gui.builder as tgb
from taipy.gui import Icon
from taipy import Config
# machine learning imports
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
try:
    from tensorflow.keras import models
    from tensorflow.keras import layers
    TENSORFLOW_AVAILABLE = True
except Exception as e:
    print(f"Warning: TensorFlow not available: {e}")
    print("Neural Network predictions will not work, but other features will still function.")
    TENSORFLOW_AVAILABLE = False
    # Create dummy classes to prevent errors
    class models:
        class Sequential:
            def __init__(self): pass
            def add(self, *args): pass
            def compile(self, *args, **kwargs): pass
            def fit(self, *args, **kwargs): pass
            def predict(self, *args, **kwargs): return [[0.0]]
    class layers:
        class Dense:
            def __init__(self, *args, **kwargs): pass
# data imports
import datetime
import plotly.graph_objects as go
import os
import platform
from typing import List, Tuple, Optional
import numpy as np

# Check for GPU (nvidia-smi only works on Linux/Unix, use different approach on Windows)
if platform.system() != "Windows":
    if os.system("nvidia-smi") == 0:
        # if GPU is compatible - install cuDF Pandas
        try:
            import cudf.pandas
            cudf.pandas.install()
        except ImportError:
            pass
# Always use Pandas on CPU for Windows or if GPU not available
import pandas as pd

###################################
# CONSTANTS
###################################

# File paths
STOCK_DATA_PATH = "data/sp500_stocks.csv"
COMPANY_DATA_PATH = "data/sp500_companies.csv"
LOGO_PATH = "images/icons/logo.png"
ID_CARD_ICON_PATH = "images/icons/id-card.png"
LIN_ICON_PATH = "images/icons/lin.png"
KNN_ICON_PATH = "images/icons/knn.png"
NN_ICON_PATH = "images/icons/rnn.png"
FLAGS_PATH = "images/flags/"

# Model hyperparameters
NN_BATCH_SIZE = 32
NN_EPOCHS = 10
NN_HIDDEN_UNITS = 64
NN_LEARNING_RATE = 'rmsprop'
NUM_FEATURES = 6

# Server configuration
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 5000

# UI layout
LOGO_WIDTH = "10vw"
ICON_WIDTH = "3vw"
LAYOUT_COUNTRY_COMPANY = "20 80"
LAYOUT_PREDICTIONS = "4 72 4 4 4 4 4 4"

# Default selections
DEFAULT_COUNTRY = "Canada"
DEFAULT_COMPANY = "LULU"

###################################
# DATA LOADING
###################################

def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load stock and company data from CSV files with error handling.
    
    Returns:
        Tuple of (stock_data, company_data) DataFrames
        
    Raises:
        FileNotFoundError: If data files are not found
        pd.errors.EmptyDataError: If data files are empty
    """
    try:
        stock_data = pd.read_csv(STOCK_DATA_PATH)
        company_data = pd.read_csv(COMPANY_DATA_PATH)
        
        if stock_data.empty:
            raise ValueError(f"Stock data file is empty: {STOCK_DATA_PATH}")
        if company_data.empty:
            raise ValueError(f"Company data file is empty: {COMPANY_DATA_PATH}")
            
        print(f"Successfully loaded {len(stock_data)} stock records and {len(company_data)} companies")
        return stock_data, company_data
        
    except FileNotFoundError as e:
        print(f"Error: Data file not found - {e}")
        raise
    except pd.errors.EmptyDataError as e:
        print(f"Error: Data file is empty - {e}")
        raise
    except Exception as e:
        print(f"Error loading data: {e}")
        raise

###################################
# GLOBAL VARIABLES
###################################

# Load datasets with error handling
try:
    stock_data, company_data = load_data()
except Exception as e:
    print(f"Fatal error: Could not load required data files. Application cannot start.")
    sys.exit(1)

# country names and icons [for slider]
country_names = company_data["Country"].unique().tolist()
country_names = [(i, Icon(FLAGS_PATH + i + ".png", i)) for i in country_names]

# company names [for slider]
company_names = company_data[["Symbol", "Shortname"]
    ].sort_values("Shortname").values.tolist()

# start and finish dates
dates = [
    stock_data["Date"].min(),
    stock_data["Date"].max()
]

# initial country and company selection
country = DEFAULT_COUNTRY
company = DEFAULT_COMPANY

# initial prediction values
lin_pred = 0
knn_pred = 0
nn_pred = 0

# initial graph values
graph_data = None
figure = None

# Helper function for safe company display
def get_company_display(company_value, company_data_df):
    """Get display name for selected company."""
    if not company_value:
        return "No Selection"
    symbol = company_value
    name_match = company_data_df[company_data_df['Symbol'] == symbol]['Shortname']
    if len(name_match) > 0:
        return f"{symbol} | {name_match.values[0]}"
    return symbol

company_display = "No Selection"

###################################
# GRAPHIC USER INTERFACE
###################################

# create web page
with tgb.Page() as page:
    # create horizontal group of elements
    # aligned to the center
    with tgb.part("text-center"):
        tgb.image(LOGO_PATH, width=LOGO_WIDTH)
        tgb.text(
            "# S&P 500 Stock Value Over Time",
            mode="md"
            )
        # create date range selector
        tgb.date_range(
            "{dates}",
            label_start="Start Date",
            label_end="End Date"
            )
        # create vertical group of 2 elements
        # taking 20% and 80% of the view power
        with tgb.layout(LAYOUT_COUNTRY_COMPANY):
            tgb.selector(
                label="country",
                class_name="fullwidth",
                value="{country}",
                lov="{country_names}",
                dropdown=True,
                value_by_id=True
                )
            tgb.selector(
                label="company",
                class_name="fullwidth",
                value="{company}",
                lov="{company_names}",
                dropdown=True,
                value_by_id=True
                )
        # create chart
        tgb.chart(figure="{figure}")
        # vertical group of 8 elements
        with tgb.part("text-left"):
            with tgb.layout(LAYOUT_PREDICTIONS):
                # company name and symbol
                tgb.image(
                    ID_CARD_ICON_PATH,
                    width=ICON_WIDTH
                    )
                tgb.text("{company_display}", mode="md")
                # linear regression prediction
                tgb.image(
                    LIN_ICON_PATH,
                    width=ICON_WIDTH
                    )
                tgb.text("{lin_pred}", mode="md")
                # KNN prediction
                tgb.image(
                    KNN_ICON_PATH,
                    width=ICON_WIDTH
                    )
                tgb.text("{knn_pred}", mode="md")
                # Neural Network prediction
                tgb.image(
                    NN_ICON_PATH,
                    width=ICON_WIDTH
                    )
                tgb.text("{nn_pred}", mode="md")
                
###################################
# FUNCTIONS
###################################

def build_company_names(country: str) -> List[List[str]]:
    """
    Filter companies by their country of origin.
    
    Args:
        country: String with country name
        
    Returns:
        List of [Symbol, Shortname] pairs from the input country
    """
    company_names = company_data[["Symbol", "Shortname"]][
        company_data["Country"] == country
    ].sort_values("Shortname").values.tolist()
    
    return company_names

def build_graph_data(dates: List[str], company: str) -> pd.DataFrame:
    """
    Filter global stock data by dates and company.
    
    Args:
        dates: List with a start date and a finish date
        company: Company symbol
        
    Returns:
        DataFrame of stock values for the symbol within the date range
    """
    if not company:
        return pd.DataFrame()
        
    temp_data = stock_data[["Date", "Adj Close", "Symbol"]][
        # filter by dates
        (stock_data["Date"] > str(dates[0])) &
        (stock_data["Date"] < str(dates[1]))
    ]
    
    # reconstruct temp_data with empty data frame
    graph_data = pd.DataFrame()
    # fetch dates column
    graph_data["Date"] = temp_data["Date"].unique()
    
    # fetch company values into column
    graph_data[company] = temp_data["Adj Close"][
        temp_data["Symbol"] == company
    ].values

    return graph_data

def display_graph(graph_data: pd.DataFrame) -> go.Figure:
    """
    Draw stock value graphs.
    
    Args:
        graph_data: DataFrame of stock values to plot
        
    Returns:
        Plotly Figure with visualized graph_data
    """
    figure = go.Figure()
    # fetch symbols from column names
    symbols = graph_data.columns[1:]
    
    # draw historic data for each symbol
    for i in symbols:
        figure.add_trace(go.Scatter(
            x=graph_data["Date"],
            y=graph_data[i],
            name=i,
            showlegend=True
            ))
        
    # add titles
    figure.update_layout(
        xaxis_title="Date",
        yaxis_title="Stock Value"
        )
    
    return figure

def split_data(stock_data: pd.DataFrame, dates: List[str], symbol: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Arrange data for training and prediction.
    
    Args:
        stock_data: Pandas DataFrame with stock data
        dates: List with a start date and a finish date
        symbol: String that represents a company symbol
        
    Returns:
        Tuple of (features, targets, eval_features) as numpy arrays
        
    Raises:
        ValueError: If insufficient data is available
    """
    temp_data = stock_data[
        # filter dates and symbol
        (stock_data["Symbol"] == symbol) &
        (stock_data["Date"] > str(dates[0])) &
        (stock_data["Date"] < str(dates[1]))
    ].drop(["Date", "Symbol"], axis=1)
    
    if len(temp_data) < 2:
        raise ValueError(f"Insufficient data for symbol {symbol} in the given date range")
    
    # fetch evaluation sample
    eval_features = temp_data.values[-1].copy()
    # unsqueeze dimensions
    eval_features = eval_features.reshape(1, -1)
    # fetch features and targets
    features = temp_data.values[:-1].copy()
    targets = temp_data["Adj Close"].shift(-1).values[:-1]
    
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    
    # Prevent division by zero
    std[std == 0] = 1.0
    
    # normalize features
    features = (features - mean) / std
    # normalize evaluation sample
    eval_features = (eval_features - mean) / std

    return features, targets, eval_features

def predict_stock(model, dates: List[str], company: str, model_name: str = "Model") -> float:
    """
    Obtain prediction using any machine learning model.
    
    Args:
        model: Trained ML model with fit() and predict() methods
        dates: List with a start date and a finish date
        company: Company symbol
        model_name: Name of the model for logging purposes
        
    Returns:
        Floating point prediction rounded to 3 decimal places
    """
    if not company:
        print(f"{model_name}: No company selected")
        return 0.0
        
    try:
        x, y, eval_x = split_data(stock_data, dates, company)
        
        # Special handling for neural network
        if model_name == "Neural Network" and TENSORFLOW_AVAILABLE:
            model.fit(x, y, batch_size=NN_BATCH_SIZE, epochs=NN_EPOCHS, verbose=0)
            pred = model.predict(eval_x, verbose=0)
            return round(float(pred[0][0]), 3)
        else:
            model.fit(x, y)
            pred = model.predict(eval_x)
            return round(float(pred[0]), 3)
            
    except ValueError as e:
        print(f"{model_name} prediction error: {e}")
        return 0.0
    except Exception as e:
        print(f"{model_name} unexpected error: {e}")
        return 0.0

def get_lin(dates: List[str], company: str) -> float:
    """
    Obtain prediction with Linear Regression.
    
    Args:
        dates: List with a start date and a finish date
        company: Company symbol
        
    Returns:
        Floating point prediction
    """
    return predict_stock(lin_model, dates, company, "Linear Regression")

def get_knn(dates: List[str], company: str) -> float:
    """
    Obtain prediction with K Nearest Neighbors.
    
    Args:
        dates: List with a start date and a finish date
        company: Company symbol
        
    Returns:
        Floating point prediction
    """
    return predict_stock(knn_model, dates, company, "KNN")

def get_nn(dates: List[str], company: str) -> float:
    """
    Obtain prediction with Neural Network.
    
    Args:
        dates: List with a start date and a finish date
        company: Company symbol
        
    Returns:
        Floating point prediction
    """
    if not TENSORFLOW_AVAILABLE:
        return 0.0
    return predict_stock(nn_model, dates, company, "Neural Network")

###################################
# BACKEND
###################################

# configure data nodes
country_cfg = Config.configure_data_node(
    id="country"
)
company_names_cfg = Config.configure_data_node(
    id="company_names"
)
dates_cfg = Config.configure_data_node(
    id="dates"
)
company_cfg = Config.configure_data_node(
    id="company"
)
graph_data_cfg = Config.configure_data_node(
    id="graph_data"
)
lin_pred_cfg = Config.configure_data_node(
    id="lin_pred"
)
knn_pred_cfg = Config.configure_data_node(
    id="knn_pred"
)
nn_pred_cfg = Config.configure_data_node(
    id="nn_pred"
)

# configure tasks
get_lin_cfg = Config.configure_task(
    input = [dates_cfg, company_cfg],
    output = lin_pred_cfg,
    function = get_lin,
    id = "get_lin",
    skippable = True
    )

get_knn_cfg = Config.configure_task(
    input = [dates_cfg, company_cfg],
    output = knn_pred_cfg,
    function = get_knn,
    id = "get_knn",
    skippable = True
    )

get_nn_cfg = Config.configure_task(
    input = [dates_cfg, company_cfg],
    output = nn_pred_cfg,
    function = get_nn,
    id = "get_nn",
    skippable = True
    )

build_graph_data_cfg = Config.configure_task(
    input = [dates_cfg, company_cfg],
    output = graph_data_cfg,
    function = build_graph_data,
    id = "build_graph_data",
    skippable = True
    )

build_company_names_cfg = Config.configure_task(
    input = country_cfg,
    output = company_names_cfg,
    function = build_company_names,
    id = "build_company_names",
    skippable = True
    )

# configure scenario
scenario_cfg = Config.configure_scenario(
    task_configs = [
        build_company_names_cfg, 
        build_graph_data_cfg,
        get_lin_cfg,
        get_knn_cfg,
        get_nn_cfg
    ],
    id="scenario"
    )

def on_init(state) -> None:
    """
    Built-in Taipy function that runs once when the application first loads.
    
    Args:
        state: Taipy state object
    """
    # write inputs to scenario
    state.scenario.country.write(state.country)
    state.scenario.dates.write(state.dates)
    state.scenario.company.write(state.company)
    # update scenario
    state.scenario.submit(wait=True)
    # fetch updated outputs
    state.graph_data = state.scenario.graph_data.read()
    state.company_names = state.scenario.company_names.read()
    state.lin_pred = state.scenario.lin_pred.read()
    state.knn_pred = state.scenario.knn_pred.read()
    state.nn_pred = state.scenario.nn_pred.read()
    # update company display
    state.company_display = get_company_display(state.company, company_data)
    # initialize the figure for the chart
    if state.graph_data is not None and not state.graph_data.empty:
        state.figure = display_graph(state.graph_data)

def on_change(state, name: str, value) -> None:
    """
    Built-in Taipy function that runs every time a GUI variable is changed by user.
    
    Args:
        state: Taipy state object
        name: Name of the variable that changed
        value: New value of the variable
    """
    if name == "country":
        print(name, "was modified", value)
        # update scenario with new country selection
        state.scenario.country.write(state.country)
        state.scenario.submit(wait=True)
        state.company_names = state.scenario.company_names.read()
        # Clear company selection and reset graph/predictions when country changes
        state.company = ""
        state.company_display = "No Selection"
        state.graph_data = None
        state.figure = None
        state.lin_pred = 0
        state.knn_pred = 0
        state.nn_pred = 0
    
    if name == "company" or name == "dates":
        print(name, "was modified", value)
        # update scenario with new company or dates selection
        state.scenario.dates.write(state.dates)
        state.scenario.company.write(state.company)
        state.scenario.submit(wait=True)
        state.graph_data = state.scenario.graph_data.read()
        state.lin_pred = state.scenario.lin_pred.read()
        state.knn_pred = state.scenario.knn_pred.read()
        state.nn_pred = state.scenario.nn_pred.read()
        # update company display
        state.company_display = get_company_display(state.company, company_data)
        # update the figure when graph_data changes
        if state.graph_data is not None and not state.graph_data.empty:
            state.figure = display_graph(state.graph_data)
    
    if name == "graph_data":
        # display updated graph data
        if state.graph_data is not None and not state.graph_data.empty:
            state.figure = display_graph(state.graph_data)

def build_neural_network(n_features: int):
    """
    Create a feedforward Neural Network for stock prediction.
    
    Note: This is a Dense Neural Network, not a Recurrent Neural Network (RNN).
    For time series data, consider using LSTM or GRU layers instead.
    
    Args:
        n_features: Integer with the number of features within x and eval_x
        
    Returns:
        TensorFlow Sequential model
    """
    if not TENSORFLOW_AVAILABLE:
        return models.Sequential()  # Return dummy model
    
    model = models.Sequential()
    model.add(layers.Dense(NN_HIDDEN_UNITS, activation='relu', input_shape=(n_features,)))
    model.add(layers.Dense(NN_HIDDEN_UNITS, activation='relu'))
    model.add(layers.Dense(1, activation='linear'))
    model.compile(optimizer=NN_LEARNING_RATE, loss='mse', metrics=['mae'])
    return model

if __name__ == "__main__":
    # create machine learning models
    lin_model = LinearRegression()
    knn_model = KNeighborsRegressor()
    nn_model = build_neural_network(NUM_FEATURES)
    
    # run Taipy orchestrator to manage scenarios
    tp.Orchestrator().run()
    # initialize scenario
    scenario = tp.create_scenario(scenario_cfg)
    # initialize GUI and display page
    gui = tp.Gui(page)
    # run application - pass scenario and other variables to state
    gui.run(
        title = "S&P 500 Prediction Dashboard",
        # Set host and port for production deployment
        host=DEFAULT_HOST,
        port=int(os.environ.get("PORT", DEFAULT_PORT)),
        # Disable reloader in production
        use_reloader = False,
        # Make scenario available in state
        scenario=scenario
        )
