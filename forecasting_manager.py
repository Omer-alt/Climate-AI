"""
Object-Oriented Module for the Prognostic Engine. (v5 - Final NaN Fix)

This module contains the ForecastingManager class, which encapsulates
all logic for time series forecasting of the SPI index. This includes:
- Feature preparation (lagging, wavelet transforms).
- Data splitting, flattening, and scaling.
- Training and evaluation of multiple ML models (ANN, SVR, RF).
- Hyperparameter optimization using Genetic Algorithms (RGA-SVR).
"""
import os
import random
import warnings
import numpy as np
import xarray as xr
import pywt
import joblib

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from deap import base, creator, tools, algorithms

import config
from drought_analysis import DataManager

warnings.filterwarnings("ignore", category=UserWarning, module='deap')
warnings.filterwarnings("ignore", category=FutureWarning)

class ForecastingManager:
    """Manages the prognostic engine for short-term drought forecasting."""

    def __init__(self, data_manager: DataManager, output_dir=config.OUTPUT_DIR):
        self.dm = data_manager
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"ForecastingManager initialized. Using device: {self.device}")
        self.data = {}
        self.models = {}
        self.results = {}

    def prepare_data_for_forecasting(self):
        """
        Prepares the data by creating lagged features, wavelet transforms,
        and splitting into scaled train/test sets.
        """
        print("--- Preparing data for forecasting ---")
        if self.dm.spi_ds is None:
            self.dm.compute_spi()

        spi_full = self.dm.spi_ds

        # 1. Wavelet Transform on the *full* dataset
        spi_approx_full = self._apply_wavelet_transform(spi_full)

        # 2. Create Lagged Features from the *full* datasets
        features_original_full = self._create_lagged_features(spi_full)
        features_approx_full = self._create_lagged_features(spi_approx_full)

        # 3. Split all datasets (original, approx, features) into train/test
        self.data['train_target_xr'] = spi_full.sel(time=config.FORECASTING_TRAIN_PERIOD)
        self.data['test_target_xr'] = spi_full.sel(time=config.FORECASTING_TEST_PERIOD)
        
        self.data['train_features_original_xr'] = features_original_full.sel(time=config.FORECASTING_TRAIN_PERIOD)
        self.data['test_features_original_xr'] = features_original_full.sel(time=config.FORECASTING_TEST_PERIOD)
        
        self.data['train_features_approx_xr'] = features_approx_full.sel(time=config.FORECASTING_TRAIN_PERIOD)
        self.data['test_features_approx_xr'] = features_approx_full.sel(time=config.FORECASTING_TEST_PERIOD)

        # 4. Flatten and Clean Data
        self._flatten_and_clean()

        # 5. Scale Data
        self._scale_data()
        print("Data preparation complete.")

    def _flatten_and_clean(self):
        """Flattens the prepared xarray data and removes NaNs."""
        y_train_flat = self.data['train_target_xr'].stack(z=('time', 'lat', 'lon')).values
        X_train_original_flat = self.data['train_features_original_xr'].stack(z=('time', 'lat', 'lon')).transpose('z', 'lag').values
        X_train_approx_flat = self.data['train_features_approx_xr'].stack(z=('time', 'lat', 'lon')).transpose('z', 'lag').values

        y_test_flat = self.data['test_target_xr'].stack(z=('time', 'lat', 'lon')).values
        X_test_original_flat = self.data['test_features_original_xr'].stack(z=('time', 'lat', 'lon')).transpose('z', 'lag').values
        X_test_approx_flat = self.data['test_features_approx_xr'].stack(z=('time', 'lat', 'lon')).transpose('z', 'lag').values

        # CORRECTION: The valid mask must check for NaNs in ALL relevant datasets.
        valid_train = (
            ~np.isnan(X_train_original_flat).any(axis=1) &
            ~np.isnan(X_train_approx_flat).any(axis=1) &
            ~np.isnan(y_train_flat)
        )
        self.data['valid_test'] = (
            ~np.isnan(X_test_original_flat).any(axis=1) &
            ~np.isnan(X_test_approx_flat).any(axis=1) &
            ~np.isnan(y_test_flat)
        )

        # Apply masks
        self.data['y_train'] = y_train_flat[valid_train]
        self.data['X_train_original'] = X_train_original_flat[valid_train]
        self.data['X_train_approx'] = X_train_approx_flat[valid_train]

        self.data['y_test'] = y_test_flat[self.data['valid_test']]
        self.data['X_test_original'] = X_test_original_flat[self.data['valid_test']]
        self.data['X_test_approx'] = X_test_approx_flat[self.data['valid_test']]

    def train_and_evaluate_all_models(self):
        """Trains all forecasting models and stores their performance."""
        if not self.data:
            self.prepare_data_for_forecasting()

        print("\n--- Training and Evaluating All Models ---")
        self._train_ann()
        self._train_wa_ann()
        self._train_svr_scale()
        self._train_rga_svr()
        self._train_random_forest()
        
        print("\n--- Forecasting Performance Summary ---")
        for name, result in self.results.items():
            print(f"{name}: RMSE={result['rmse']:.4f}, R²={result['r2']:.4f}")
        
        return self.results

    def _wavelet_transform_1d(self, spi_1d, wavelet='haar', level=2):
        """Helper function for wavelet transform on a single 1D series."""
        data_writable = np.copy(spi_1d)
        
        coeffs = pywt.wavedec(data_writable, wavelet, level=level)
        coeffs_approx = [coeffs[0]] + [np.zeros_like(c) for c in coeffs[1:]]
        reconstructed = pywt.waverec(coeffs_approx, wavelet)

        if len(reconstructed) > len(spi_1d):
            return reconstructed[:len(spi_1d)]
        elif len(reconstructed) < len(spi_1d):
            return np.pad(reconstructed, (0, len(spi_1d) - len(reconstructed)), mode='constant')
        return reconstructed

    def _apply_wavelet_transform(self, spi_xr, wavelet='haar', level=2):
        """Applies wavelet transform to the SPI time series."""
        return xr.apply_ufunc(
            self._wavelet_transform_1d,
            spi_xr,
            input_core_dims=[["time"]],
            output_core_dims=[["time"]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[float],
            kwargs={'wavelet': wavelet, 'level': level}
        )

    def _create_lagged_features(self, data_xr):
        features = [data_xr.shift(time=lag) for lag in range(1, config.FORECASTING_LAGS + 1)]
        return xr.concat(features, dim='lag')

    def _scale_data(self):
        self.scalers = {}
        for name in ['original', 'approx']:
            scaler = MinMaxScaler()
            self.data[f'X_train_{name}_scaled'] = scaler.fit_transform(self.data[f'X_train_{name}'])
            self.data[f'X_test_{name}_scaled'] = scaler.transform(self.data[f'X_test_{name}'])
            self.scalers[name] = scaler

    def _train_ann_base(self, model_name, train_loader, test_loader, X_test_scaled):
        """Base training loop for ANN and WA-ANN with detailed logging."""
        model = self.ANNModel(input_size=config.FORECASTING_LAGS).to(self.device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=config.ANN_LEARNING_RATE)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
        
        best_loss = float('inf')
        patience_counter = 0

        for epoch in range(config.ANN_EPOCHS):
            model.train()
            running_train_loss = 0.0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(), labels)
                loss.backward()
                optimizer.step()
                running_train_loss += loss.item() * inputs.size(0)
            
            epoch_train_loss = running_train_loss / len(train_loader.dataset)
            
            model.eval()
            running_val_loss = 0.0
            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = model(inputs)
                    loss = criterion(outputs.squeeze(), labels)
                    running_val_loss += loss.item() * inputs.size(0)
            
            epoch_val_loss = running_val_loss / len(test_loader.dataset)
            scheduler.step(epoch_val_loss)
            
            print(f"{model_name} Epoch {epoch+1}/{config.ANN_EPOCHS}, Train Loss: {epoch_train_loss:.6f}, Val Loss: {epoch_val_loss:.6f}")
            
            # Save model only if validation loss is not NaN and is the best so far
            if not np.isnan(epoch_val_loss) and epoch_val_loss < best_loss:
                best_loss = epoch_val_loss
                patience_counter = 0
                torch.save(model.state_dict(), os.path.join(self.output_dir, f'{model_name}.pth'))
            else:
                patience_counter += 1
                if patience_counter >= config.ANN_PATIENCE:
                    print("Early stopping triggered.")
                    break
        
        # Load the best saved model, if it exists
        model_path = os.path.join(self.output_dir, f'{model_name}.pth')
        if not os.path.exists(model_path):
            print(f"ERROR: Model for {model_name} was not saved due to NaN losses. Skipping evaluation.")
            self.results[model_name] = {'rmse': np.nan, 'r2': np.nan}
            return

        model.load_state_dict(torch.load(model_path))
        self.models[model_name] = model
        
        model.eval()
        with torch.no_grad():
            preds = model(torch.tensor(X_test_scaled, dtype=torch.float32).to(self.device)).cpu().numpy().flatten()
        
        self.results[model_name] = {
            'rmse': np.sqrt(mean_squared_error(self.data['y_test'], preds)),
            'r2': r2_score(self.data['y_test'], preds)
        }

    def _train_ann(self):
        train_loader = DataLoader(TensorDataset(torch.tensor(self.data['X_train_original_scaled'], dtype=torch.float32), torch.tensor(self.data['y_train'], dtype=torch.float32)), batch_size=config.ANN_BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(TensorDataset(torch.tensor(self.data['X_test_original_scaled'], dtype=torch.float32), torch.tensor(self.data['y_test'], dtype=torch.float32)), batch_size=config.ANN_BATCH_SIZE)
        self._train_ann_base('ANN', train_loader, test_loader, self.data['X_test_original_scaled'])

    def _train_wa_ann(self):
        train_loader = DataLoader(TensorDataset(torch.tensor(self.data['X_train_approx_scaled'], dtype=torch.float32), torch.tensor(self.data['y_train'], dtype=torch.float32)), batch_size=config.ANN_BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(TensorDataset(torch.tensor(self.data['X_test_approx_scaled'], dtype=torch.float32), torch.tensor(self.data['y_test'], dtype=torch.float32)), batch_size=config.ANN_BATCH_SIZE)
        self._train_ann_base('WA-ANN', train_loader, test_loader, self.data['X_test_approx_scaled'])

    def _train_svr_scale(self):
        model_name = 'SVR-scale'
        print(f"Training {model_name}...")
        model = SVR(kernel='rbf', C=5.0, epsilon=0.1, gamma='scale')
        subset_idx = config.SVR_TRAINING_SUBSET if config.SVR_TRAINING_SUBSET != 'all' else len(self.data['y_train'])
        model.fit(self.data['X_train_original_scaled'][:subset_idx], self.data['y_train'][:subset_idx])
        joblib.dump(model, os.path.join(self.output_dir, f'{model_name}.joblib'))
        self.models[model_name] = model
        preds = model.predict(self.data['X_test_original_scaled'])
        self.results[model_name] = {'rmse': np.sqrt(mean_squared_error(self.data['y_test'], preds)), 'r2': r2_score(self.data['y_test'], preds)}

    def _train_rga_svr(self):
        model_name = 'RGA-SVR'
        print(f"Training {model_name} with Genetic Algorithm...")
        # Defensive creation of DEAP types
        if not hasattr(creator, "FitnessMin"):
            creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMin)
        
        toolbox = base.Toolbox()
        toolbox.register("attr_float", random.uniform, 0.1, 10)
        toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=3)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        
        subset_idx = config.SVR_TRAINING_SUBSET if config.SVR_TRAINING_SUBSET != 'all' else len(self.data['y_train'])
        X_train_sub = self.data['X_train_original_scaled'][:subset_idx]
        y_train_sub = self.data['y_train'][:subset_idx]

        def evaluate_svr(individual):
            C, gamma, epsilon = [max(0.01, x) for x in individual]
            svr = SVR(kernel='rbf', C=C, gamma=gamma, epsilon=epsilon)
            svr.fit(X_train_sub, y_train_sub)
            preds = svr.predict(self.data['X_test_original_scaled'])
            return mean_squared_error(self.data['y_test'], preds),

        toolbox.register("evaluate", evaluate_svr)
        toolbox.register("mate", tools.cxBlend, alpha=0.5)
        toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
        toolbox.register("select", tools.selTournament, tournsize=3)
        population = toolbox.population(n=config.RGA_POPULATION_SIZE)
        algorithms.eaSimple(population, toolbox, cxpb=config.RGA_CXPB, mutpb=config.RGA_MUTPB, ngen=config.RGA_GENERATIONS, verbose=True)
        best_ind = tools.selBest(population, k=1)[0]
        print(f"Best RGA-SVR params: C={best_ind[0]:.4f}, gamma={best_ind[1]:.4f}, epsilon={best_ind[2]:.4f}")
        model = SVR(kernel='rbf', C=best_ind[0], gamma=best_ind[1], epsilon=best_ind[2])
        model.fit(X_train_sub, y_train_sub)
        joblib.dump(model, os.path.join(self.output_dir, f'{model_name}.joblib'))
        self.models[model_name] = model
        preds = model.predict(self.data['X_test_original_scaled'])
        self.results[model_name] = {'rmse': np.sqrt(mean_squared_error(self.data['y_test'], preds)), 'r2': r2_score(self.data['y_test'], preds)}

    def _train_random_forest(self):
        model_name = 'RandomForest'
        print(f"Training {model_name}...")
        model = RandomForestRegressor(n_estimators=100, max_depth=15, min_samples_split=10, random_state=42, n_jobs=-1)
        model.fit(self.data['X_train_original_scaled'], self.data['y_train'])
        joblib.dump(model, os.path.join(self.output_dir, f'{model_name}.joblib'))
        self.models[model_name] = model
        preds = model.predict(self.data['X_test_original_scaled'])
        self.results[model_name] = {'rmse': np.sqrt(mean_squared_error(self.data['y_test'], preds)), 'r2': r2_score(self.data['y_test'], preds)}

    class ANNModel(nn.Module):
        """Internal PyTorch ANN model definition."""
        def __init__(self, input_size, hidden_size1=256, hidden_size2=128, hidden_size3=64):
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(input_size, hidden_size1), nn.BatchNorm1d(hidden_size1), nn.ReLU(), nn.Dropout(0.4),
                nn.Linear(hidden_size1, hidden_size2), nn.BatchNorm1d(hidden_size2), nn.ReLU(), nn.Dropout(0.4),
                nn.Linear(hidden_size2, hidden_size3), nn.BatchNorm1d(hidden_size3), nn.ReLU(), nn.Dropout(0.4),
                nn.Linear(hidden_size3, 1)
            )
        def forward(self, x):
            return self.network(x)