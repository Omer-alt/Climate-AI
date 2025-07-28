"""
Object-Oriented Framework for Spatio-Temporal Drought Analysis.

This module contains three core classes:
1.  DataManager: Handles loading, preprocessing, and calculation of climate data.
2.  ClusteringManager: Manages the diagnostic engine, including feature engineering,
    K-Means clustering, and hotspot identification.
3.  PlottingManager: Provides a suite of configurable methods for visualizing
    geospatial data, time series, and analysis results.
"""

import os
import warnings
from itertools import groupby

import numpy as np
import xarray as xr
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import ListedColormap
from shapely.geometry import Point
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from climate_indices import indices, compute

# Ignore warnings from climate_indices about fitting parameters
warnings.filterwarnings("ignore", message="Fitting parameters to a constant series")

import config  # Import parameters from the config file


class DataManager:
    """
    Handles data loading, preprocessing, and core climate calculations.
    
    This class is responsible for all data I/O and transformations,
    providing clean, analysis-ready datasets to other components.
    """

    def __init__(self, data_source: str):
        """
        Initializes the DataManager for a specific data source.

        Args:
            data_source (str): The name of the data source, either 'GPCC' or 'CRU'.
        """
        if data_source.upper() not in ['GPCC', 'CRU']:
            raise ValueError("data_source must be 'GPCC' or 'CRU'")
        self.data_source = data_source.upper()
        self.raw_path = config.GPCC_RAW_PATH if self.data_source == 'GPCC' else config.CRU_RAW_PATH
        self.precip_var = 'precip' if self.data_source == 'GPCC' else 'pre'
        
        self.ds = None
        self.eccas_mask = None
        self.spi_ds = None

    def load_and_preprocess(self, force_reprocess=False):
        """
        Loads, masks, and preprocesses the dataset for the ECCAS region.

        This method performs the following steps:
        1. Loads the raw NetCDF dataset.
        2. Selects the study period.
        3. Loads ECCAS country geometries.
        4. Creates a spatial mask for the ECCAS region.
        5. Applies the mask to the dataset.

        Args:
            force_reprocess (bool): If True, re-creates the mask even if it exists.
        
        Returns:
            xr.Dataset: The preprocessed dataset masked for the ECCAS region.
        """
        print(f"--- Loading and preprocessing for {self.data_source} ---")
        self.ds = xr.open_dataset(self.raw_path).sel(time=slice(config.STUDY_PERIOD_START, config.STUDY_PERIOD_END))
        
        world = gpd.read_file(config.WORLD_SHAPEFILE_URL)
        eccas_gdf = world[world["ISO_A3"].isin(config.ECCAS_ISO3_CODES)]
        eccas_union = eccas_gdf.geometry.union_all()
        
        lon, lat = np.meshgrid(self.ds.lon.values, self.ds.lat.values)
        mask = np.zeros_like(lon, dtype=bool)
        for i in range(lat.shape[0]):
            for j in range(lat.shape[1]):
                mask[i, j] = eccas_union.contains(Point(lon[i, j], lat[i, j]))
        
        self.eccas_mask = xr.DataArray(mask, dims=('lat', 'lon'), coords={'lat': self.ds.lat, 'lon': self.ds.lon})
        self.ds = self.ds.where(self.eccas_mask)
        print(f"Preprocessing complete for {self.data_source}.")
        return self.ds

    def compute_spi(self, scale: int = config.SPI_SCALE):
        """
        Computes the Standardized Precipitation Index (SPI) for the dataset.

        Args:
            scale (int): The time scale in months for the SPI calculation.

        Returns:
            xr.DataArray: The computed SPI data.
        """
        if self.ds is None:
            self.load_and_preprocess()

        print(f"Computing SPI-{scale} for {self.data_source}...")

        def spi_1d(arr_1d):
            return indices.spi(
                values=arr_1d,
                scale=scale,
                distribution=indices.Distribution.gamma,
                data_start_year=int(config.STUDY_PERIOD_START[:4]),
                calibration_year_initial=int(config.STUDY_PERIOD_START[:4]),
                calibration_year_final=int(config.STUDY_PERIOD_END[:4]),
                periodicity=compute.Periodicity.monthly,
            )

        self.spi_ds = xr.apply_ufunc(
            spi_1d,
            self.ds[self.precip_var],
            input_core_dims=[["time"]],
            output_core_dims=[["time"]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[float],
        ).rename(f"spi{scale}_{self.data_source.lower()}")
        
        print(f"SPI-{scale} computation finished.")
        return self.spi_ds

class ClusteringManager:
    """
    Manages the diagnostic engine for drought and aridity analysis.
    
    This class implements the K-Means clustering workflow, including
    custom feature engineering to resolve the "aridity blindspot".
    """
    def __init__(self, data_manager: DataManager):
        """
        Initializes the ClusteringManager with a DataManager instance.

        Args:
            data_manager (DataManager): An initialized DataManager object
                                        containing the data to be analyzed.
        """
        self.dm = data_manager
        if self.dm.spi_ds is None:
            self.dm.compute_spi()

    def _compute_metrics(self, period_slice):
        """Computes all necessary metrics for a given time period."""
        precip_sub = self.dm.ds[self.dm.precip_var].sel(time=period_slice)
        spi_sub = self.dm.spi_ds.sel(time=period_slice)
        
        map_metric = (precip_sub.mean(dim='time') * 12).rename('mean_annual_precip')
        ds_metric = (-spi_sub).where(spi_sub < config.DROUGHT_THRESHOLD, 0).sum(dim='time').rename('drought_severity')
        
        def max_consecutive_drought(spi_1d, thresh):
            return max((len(list(g)) for k, g in groupby(spi_1d < thresh) if k), default=0)
            
        mdd_metric = xr.apply_ufunc(
            max_consecutive_drought, spi_sub, 
            input_core_dims=[["time"]], vectorize=True, dask="parallelized", 
            output_dtypes=[int], kwargs={'thresh': config.DROUGHT_THRESHOLD}
        ).rename('max_drought_duration')
        
        return map_metric, ds_metric, mdd_metric

    def perform_clustering(self, period_slice, n_clusters=config.K_CLUSTERS):
        """
        Performs the full clustering analysis for a specific period.

        Args:
            period_slice (slice): A time slice object for the analysis period.
            n_clusters (int): The number of clusters for K-Means.

        Returns:
            dict: A dictionary containing clustering results (labels, scores, etc.).
        """
        print(f"Performing clustering for period: {period_slice.start}-{period_slice.stop}...")
        map_m, ds_m, mdd_m = self._compute_metrics(period_slice)
        
        lat, lon = self.dm.ds.lat.values, self.dm.ds.lon.values
        lon_grid, lat_grid = np.meshgrid(lon, lat)
        lat_flat, lon_flat = lat_grid.flatten(), lon_grid.flatten()
        mask_flat = self.dm.eccas_mask.values.flatten()
        
        # Feature engineering: Aridity Index + Drought Metrics
        arid_feature = -np.log1p(map_m.values.flatten())
        features = np.column_stack((lat_flat, lon_flat, arid_feature, ds_m.values.flatten(), mdd_m.values.flatten()))
        all_metrics = np.column_stack((map_m.values.flatten(), ds_m.values.flatten(), mdd_m.values.flatten()))
        
        valid_mask = mask_flat & ~np.isnan(features).any(axis=1)
        features_clean = features[valid_mask]
        metrics_clean = all_metrics[valid_mask]
        
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features_clean)
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit(features_scaled)
        labels = kmeans.labels_
        
        # Structure results
        labels_full = np.full(lat_flat.shape, np.nan)
        labels_full[valid_mask] = labels
        labels_da = xr.DataArray(labels_full.reshape(lat_grid.shape), coords={'lat': lat, 'lon': lon}, dims=['lat', 'lon'])
        
        # Calculate cluster means and assign labels
        cluster_means = {i: {
            'mean_map': np.mean(metrics_clean[labels == i, 0]),
            'mean_ds': np.mean(metrics_clean[labels == i, 1]),
            'mean_mdd': np.mean(metrics_clean[labels == i, 2])
        } for i in range(n_clusters)}
        
        arid_cluster_id = min(cluster_means, key=lambda k: cluster_means[k]['mean_map'])
        drought_clusters = sorted(
            [(k, v) for k, v in cluster_means.items() if k != arid_cluster_id],
            key=lambda item: item[1]['mean_ds'], reverse=True
        )
        
        cluster_labels_map = {arid_cluster_id: 'Arid'}
        drought_labels = ['Critical', 'Severe', 'Warning', 'Moderate', 'Attention', 'Normal']
        for i, (cluster_id, _) in enumerate(drought_clusters):
            cluster_labels_map[cluster_id] = drought_labels[i]

        return {
            'labels_da': labels_da,
            'cluster_labels': cluster_labels_map,
            'cluster_means': cluster_means,
            'silhouette_score': silhouette_score(features_scaled, labels)
        }

class PlottingManager:
    """Provides a collection of methods for visualizing climate data and analysis results."""

    def __init__(self, eccas_shapefile):
        """
        Initializes the PlottingManager.

        Args:
            eccas_shapefile (gpd.GeoDataFrame): GeoDataFrame of ECCAS countries.
        """
        self.eccas_gdf = eccas_shapefile
        self.projection = ccrs.PlateCarree()

    def _setup_map_axis(self, ax, title=""):
        """Configures a standard map axis with borders and coastlines."""
        ax.set_title(title, fontsize=14)
        ax.coastlines()
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        gl = ax.gridlines(crs=self.projection, draw_labels=True, linewidth=0.5, color='gray', alpha=0.5)
        gl.top_labels = gl.right_labels = False
        gl.xlabel_style = gl.ylabel_style = {'size': 10}
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

    def plot_precipitation_comparison(self, ds1, var1, ds2, var2, time_index):
        """
        Plots a side-by-side comparison of precipitation from two datasets.
        
        Args:
            ds1 (xr.Dataset): First dataset (e.g., CRU).
            var1 (str): Precipitation variable name in ds1.
            ds2 (xr.Dataset): Second dataset (e.g., GPCC).
            var2 (str): Precipitation variable name in ds2.
            time_index (int): The time index to plot.
        """
        fig, axes = plt.subplots(1, 2, figsize=(20, 8), subplot_kw={'projection': self.projection})
        time_str = ds1.time.isel(time=time_index).dt.strftime('%B %Y').item()
        fig.suptitle(f"Precipitation Comparison for {time_str}", fontsize=18)
        
        # Plot 1
        self._setup_map_axis(axes[0], f"{ds1.attrs.get('title', 'Dataset 1')}")
        ds1[var1].isel(time=time_index).plot(ax=axes[0], cmap='viridis', cbar_kwargs={'label': 'Precipitation (mm/month)'})

        # Plot 2
        self._setup_map_axis(axes[1], f"{ds2.attrs.get('title', 'Dataset 2')}")
        ds2[var2].isel(time=time_index).plot(ax=axes[1], cmap='viridis', cbar_kwargs={'label': 'Precipitation (mm/month)'})
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    def plot_clustering_results(self, all_results):
        """
        Generates the main 2x3 grid plot of clustering results over time.

        Args:
            all_results (dict): A nested dictionary with results structured as
                                {dataset_name: {period_name: results_dict}}.
        """
        fig, axes = plt.subplots(2, 3, figsize=(24, 12), subplot_kw={'projection': self.projection})
        
        # Temporal ranking for color shading
        ranks = self._calculate_temporal_ranks(all_results)

        for i, dataset_name in enumerate(['GPCC', 'CRU']):
            for j, period_name in enumerate(config.SUBPERIODS.keys()):
                ax = axes[i, j]
                res = all_results[dataset_name][period_name]
                
                # Create colormap for this specific plot
                colors_list = [None] * config.K_CLUSTERS
                for cluster_id, label in res['cluster_labels'].items():
                    rank = ranks[dataset_name][label].get(period_name, 0)
                    color = config.COLOR_PALETTES[label][rank]
                    colors_list[int(cluster_id)] = color
                cmap = ListedColormap([c for c in colors_list if c is not None])
                
                # Plotting
                self._setup_map_axis(ax, f"{dataset_name} {period_name.upper()}\nSilhouette: {res['silhouette_score']:.3f}")
                res['labels_da'].plot(ax=ax, cmap=cmap, add_colorbar=False, levels=np.arange(config.K_CLUSTERS + 1) - 0.5)
                
                # Add legend
                self._add_cluster_legend(ax, res, ranks[dataset_name], period_name)

        plt.tight_layout(rect=[0, 0.1, 1, 0.95])
        self._add_global_color_legend(fig)
        plt.show()

    def _calculate_temporal_ranks(self, all_results):
        """Helper to rank clusters over time for color shading."""
        ds_values = {}
        for ds_name in ['GPCC', 'CRU']:
            ds_values[ds_name] = {}
            for label in config.COLOR_PALETTES:
                ds_values[ds_name][label] = {}
                for period, res in all_results[ds_name].items():
                    label_to_id = {v: k for k, v in res['cluster_labels'].items()}
                    if label in label_to_id:
                        cluster_id = label_to_id[label]
                        val = res['cluster_means'][cluster_id]['mean_map' if label == 'Arid' else 'mean_ds']
                        ds_values[ds_name][label][period] = val
        
        ranks = {}
        for ds_name in ['GPCC', 'CRU']:
            ranks[ds_name] = {}
            for label in config.COLOR_PALETTES:
                value_list = sorted(ds_values[ds_name].get(label, {}).items(), key=lambda x: x[1])
                ranks[ds_name][label] = {period: rank for rank, (period, _) in enumerate(value_list)}
        return ranks

    def _add_cluster_legend(self, ax, results, ranks, period_name):
        """Adds a detailed legend to a single map axis."""
        patches = []
        label_to_id = {v: k for k, v in results['cluster_labels'].items()}
        for label in config.LEGEND_ORDER:
            if label in label_to_id:
                cluster_id = label_to_id[label]
                rank = ranks[label].get(period_name, 0)
                color = config.COLOR_PALETTES[label][rank]
                metrics = results['cluster_means'][cluster_id]
                
                text = f"{label}\nDS: {metrics['mean_ds']:.1f}; MDD: {metrics['mean_mdd']:.1f}"
                if label == 'Arid':
                    text = f"{label}\nMAP: {metrics['mean_map']:.1f} mm/yr"
                
                patches.append(mpatches.Patch(color=color, label=text))
        
        ax.legend(handles=patches, loc='upper right', bbox_to_anchor=(1.55, 1), title='Cluster Info', fontsize='small')

    def _add_global_color_legend(self, fig):
        """Adds the horizontal color ramp legend at the bottom of the figure."""
        legend_ax = fig.add_axes([0.15, 0.05, 0.7, 0.05])
        legend_ax.set_axis_off()
        total_units = (len(config.LEGEND_ORDER) * 3) + (len(config.LEGEND_ORDER) - 1)
        box_width = 1.0 / total_units
        x_pos = 0
        for category in config.LEGEND_ORDER:
            cat_start_x = x_pos
            for color in config.COLOR_PALETTES[category]:
                rect = mpatches.Rectangle((x_pos, 0.2), box_width, 0.6, fc=color, ec='none')
                legend_ax.add_patch(rect)
                x_pos += box_width
            legend_ax.text(cat_start_x + (1.5 * box_width), -0.1, category, ha='center', va='top', fontsize=10)
            x_pos += box_width

        fig.text(0.5, 0.02, 'Lighter shade = Lower severity/aridity rank | Darker shade = Higher severity/aridity rank', 
                 ha='center', va='bottom', fontsize=10, style='italic')
        fig.text(0.5, 0.10, 'Severity/Aridity Category & Temporal Rank', ha='center', va='center', fontsize=12, weight='bold')
