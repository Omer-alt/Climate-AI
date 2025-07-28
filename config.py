"""
Configuration file for the drought analysis project.

Centralizes file paths, parameters, and constants to make the main scripts
cleaner and easier to maintain.
"""
import os

# --- File Paths ---
# Note: Adjust these paths relative to where you run the main script.
DATASET_DIR = os.path.join(os.path.dirname(__file__), '..', 'Dataset')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'models_output')
CRU_RAW_PATH = os.path.join(DATASET_DIR, "CRUPATH")
GPCC_RAW_PATH = os.path.join(DATASET_DIR, "GPCCPATH")
WORLD_SHAPEFILE_URL = "https://naciscdn.org/naturalearth/50m/cultural/ne_50m_admin_0_countries.zip"

# --- Geographic & Time Parameters ---
ECCAS_ISO3_CODES = ["AGO", "BDI", "CMR", "COG", "GAB", "GNQ", "CAF", "COD", "RWA", "STP", "TCD"]
STUDY_PERIOD_START = '1928-01-01'
STUDY_PERIOD_END = '2017-12-31'
SUBPERIODS = {
    'p1': slice('1928-01-01', '1957-12-31'),
    'p2': slice('1958-01-01', '1987-12-31'),
    'p3': slice('1988-01-01', '2017-12-31')
}

# --- Diagnostic Analysis Parameters ---
SPI_SCALE = 6
DROUGHT_THRESHOLD = -1.0
K_CLUSTERS = 7

# --- Prognostic (Forecasting) Analysis Parameters ---
FORECASTING_TRAIN_PERIOD = slice('1986-01-01', '2015-12-31')
FORECASTING_TEST_PERIOD = slice('2016-01-01', '2017-12-31')
FORECASTING_LAGS = 3
# Limit number of samples for SVR training to speed up execution. 'all' to use full dataset.
SVR_TRAINING_SUBSET = 100000 

# --- ANN Training Parameters ---
ANN_EPOCHS = 100
ANN_BATCH_SIZE = 256
ANN_LEARNING_RATE = 0.001
ANN_PATIENCE = 10 # For early stopping

# --- Genetic Algorithm (RGA-SVR) Parameters ---
RGA_POPULATION_SIZE = 20
RGA_GENERATIONS = 5
RGA_CXPB = 0.7  # Crossover probability
RGA_MUTPB = 0.3 # Mutation probability

# --- Plotting Parameters ---
COLOR_PALETTES = {
    'Arid':      ['#55372A', '#4A2C17', '#3D2010'],
    'Critical':  ['#8B0000', '#6E0000', '#4D0000'],
    'Severe':    ['#D92525', '#C21313', '#A10D0D'],
    'Warning':   ['#F37750', '#EE4D23', '#D93A0D'],
    'Moderate':  ['#FDCB9A', '#FAAC64', '#F78C2D'],
    'Attention': ['#FDD869', '#FBC11F', '#E9A200'],
    'Normal':    ['#FEFDE2', '#FCF7B8', '#FAEF8A']
}
LEGEND_ORDER = ['Normal', 'Attention', 'Moderate', 'Warning', 'Severe', 'Critical', 'Arid']
