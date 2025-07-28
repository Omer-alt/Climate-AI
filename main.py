"""
Main execution script for the drought analysis project.

This script orchestrates the entire workflow:
1.  Diagnostic Engine : Initializes DataManagers, runs clustering analysis,
    and visualizes historical drought/aridity hotspots.
2.  Prognostic Engine : Initializes ForecastingManager, prepares data,
    trains a suite of ML models, and evaluates their performance.
"""
import geopandas as gpd
from drought_analysis import DataManager, ClusteringManager, PlottingManager
from forecasting_manager import ForecastingManager
import config

def run_diagnostic_engine():
    """Runs the full diagnostic analysis and generates plots."""
    print("========== RUNNING DIAGNOSTIC ENGINE ==========")
    gpcc_dm = DataManager(data_source='GPCC')
    cru_dm = DataManager(data_source='CRU')
    
    world = gpd.read_file(config.WORLD_SHAPEFILE_URL)
    eccas_gdf = world[world["ISO_A3"].isin(config.ECCAS_ISO3_CODES)]
    plotter = PlottingManager(eccas_gdf)

    all_results = {'GPCC': {}, 'CRU': {}}
    for dm in [gpcc_dm, cru_dm]:
        cluster_manager = ClusteringManager(data_manager=dm)
        for period_name, period_slice in config.SUBPERIODS.items():
            result = cluster_manager.perform_clustering(period_slice)
            all_results[dm.data_source][period_name] = result
    
    plotter.plot_clustering_results(all_results)
    print("========== DIAGNOSTIC ENGINE COMPLETE ==========\n")

def run_prognostic_engine():
    """Runs the full forecasting model comparison."""
    print("========== RUNNING PROGNOSTIC ENGINE ==========")
    # Forecasting is typically done on the higher-resolution dataset
    gpcc_dm = DataManager(data_source='GPCC')
    
    forecaster = ForecastingManager(data_manager=gpcc_dm)
    forecaster.train_and_evaluate_all_models()
    
    print("========== PROGNOSTIC ENGINE COMPLETE ==========")

def main():
    """Main function to run both analysis engines."""
    run_diagnostic_engine()
    run_prognostic_engine()

if __name__ == "__main__":
    main()