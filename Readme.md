# The Aridity Blindspot: A Machine Learning Framework for Spatio-Temporal Drought Analysis and Forecasting

![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Frameworks](https://img.shields.io/badge/Frameworks-PyTorch_|_Scikit--learn_|_Xarray_|_GeoPandas_|_climate_indices-orange.svg)
---

### Abstract

This project introduces a comprehensive machine learning framework to address a critical, often-overlooked challenge in climate science: the "aridity blindspot." Standard drought indicators, when used in unsupervised learning, frequently fail to distinguish between temporary water deficits (drought) and permanent water scarcity (aridity). This flaw is particularly pronounced in climatologically diverse regions like Central Africa, leading to inaccurate risk assessments.

Our framework is composed of two primary engines:
1.  The **Diagnostic Engine** resolves the aridity blindspot by integrating a novel Aridity Index into a K-Means clustering algorithm. Applied to a 90-year climate record for the ECCAS region, this engine successfully disentangles chronic dryness from anomalous drought, producing the first accurate spatio-temporal maps of evolving climate stress hotspots.
2.  The **Prognostic Engine** complements this analysis by establishing a robust short-term drought forecasting system. It conducts a rigorous comparative study of state-of-the-art models (ANN, SVR, RGA-SVR, RF) to deliver high-fidelity SPI-6 predictions.

Ultimately, this project delivers a transferable and validated framework that provides actionable insights for policymakers, improves climate risk assessment, and can be adapted for water resource management in other data-sparse, vulnerable regions worldwide.

---

## Repository Structure

The project is organized into a modular structure to ensure clarity and ease of maintenance.

```
.
├── Assets/
├── Dataset/
├── models_output/
├── climate_notebook.ipynb
├── config.py
├── drought_analysis.py
├── forecasting_manager.py
├── main.py
└── requirements.txt
```

-   `Assets/`: Contains static assets like images and utility files used in the project.
-   `Dataset/`: Houses the raw climate datasets (e.g., GPCC, CRU). *Note: Due to their size, these files are typically git-ignored.*
-   `models_output/`: Default directory for saving trained model artifacts (`.pth`, `.joblib`) and other outputs generated during execution.
-   `climate_notebook.ipynb`: A comprehensive Jupyter Notebook presenting the entire implementation in a simplified, cell-by-cell format. Ideal for understanding the workflow and visualizing results.
-   `config.py`: Central configuration file for all parameters, paths, and constants. Allows for easy tuning of the models and analysis settings.
-   `drought_analysis.py`: Implements the **Diagnostic Engine**. This module handles the spatio-temporal analysis of drought, providing a clear vision of the water stress problem in the ECCAS region.
-   `forecasting_manager.py`: Implements the **Prognostic Engine**. This module is dedicated to forecasting the SPI-6 index using various models (SVR, RGA-SVR, ANN, WA-ANN, RF).
-   `main.py`: The main entry point to run the diagnostic and prognostic engines.
-   `requirements.txt`: A list of all Python dependencies required to run the project.

---

## Getting Started

### Prerequisites

-   Python 3.9+
-   `pip` or `conda` for package management.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Omer-alt/Climate-AI.git
    cd Climate-AI
    ```

2.  **Create a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the dependencies:**
    Use the provided `requirements.txt` file to install all necessary packages.
    ```bash
    pip install -r requirements.txt
    ```
    > **Note on Geospatial Libraries:** Packages like `Cartopy` and `geopandas` can sometimes be challenging to install with `pip`. If you encounter issues, using a `conda` environment is highly recommended.
    > ```bash
    > conda create --name climate_env python=3.11
    > conda activate climate_env
    > conda install --file requirements.txt
    > ```

4.  **Download Datasets:**
    Place the CRU and GPCC NetCDF files into the `Dataset/` directory.

---

## Usage

The `main.py` script serves as the main entry point for running the analysis. It is designed to be modular, allowing you to run either the diagnostic or prognostic engine independently.

-   **To run the complete analysis (both engines):**
    ```bash
    python main.py
    ```

-   **To run only a specific engine:**
    Open `main.py` and comment out the engine you do not wish to run in the `main()` function:
    ```python
    def main():
        """Main function to run both analysis engines."""
        run_diagnostic_engine()
        # run_prognostic_engine() # Comment this out to run only diagnostics

    if __name__ == "__main__":
        main()
    ```


---

## License

This project is licensed under the MIT License - see the `LICENSE.md` file for details.

---
⭐️ If you find this repository helpful, we’d be thrilled if you could give it a star! 