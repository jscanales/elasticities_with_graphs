# elasticities-with-graphs
 

## Overview
This repository contains the code supporting the analysis presented in the working paper 
"IDENTIFYING ELASTICITIES IN AUTOCORRELATED TIME SERIES USING CAUSAL GRAPHS" by Tiedemann et al. 
The project is structured into three primary components:

### 1. Data Processing
The data component consists of two main aspects:

1. Data Acquisition.

    1. For some of the variabes used in the analysis, this just involves using downloaded .csv 
    files (e.g., natural gas prices) or with a slight transformation (e.g., holidays).

    2. For other variables, the process requires querying online sources and constructing the data series. This involves setting up accounts for external services, such as ENTSO-E Transparency and NASA's MERRA-2, and loading the retrieved data into a local database.

2. Data Handling.
    * The code provides functions to interact with these data sources and use them in analysis, transforming the raw data files into pandas-compatible objects.
    * Variables are either read directly from .csv files or retrieved from the database where the data was previously stored.

* Key Files: 
    - database.py: Contains functions for creating and managing the database.
    - model.py: Includes classes responsible for retrieving and processing data for analysis. The Dummy class handles binary variables, while the HourlySeries class manages continuous variables. Both are initialized with a pandas.DatetimeIndex object and offer methods such as .constant() and .price() to retrieve the respective variables.


### 2. The core code of the project
This part of the project builds on the data handling to perform the primary analysis. The code implements the analytical framework and methods used in the paper. The code is written using both functional and object-oriented programming. The code for the analysis is prepared over several files, but not executed there.

* Key Files:
    - model.py: Builds on database.py and utilizes the linearmodels package to manage input variables and prepare them for the Two-Stage Least Squares (2SLS) Instrumental Variable (IV) analysis.
    - estimation.py: Implements the different identification strategies used in the project, as well as additional estimation functions (e.g., autoregressive analysis).
    - synthetic.py: Generates synthetic time series data for wind generation, electricity demand, supply, and prices using parameters derived from real-world data and an underlying structural model.
    - simulations.py: Conducts multiple simulations using the synthetic data and analyzes the outcomes across various experimental runs to account for random variation and identify broader patterns.

### 3. Analysis and Visualization
This component includes code for executing the analysis and visualizing the results.

* Key Files:
    - application_plots.ipynb: Executes and visualizes the analysis based on real-world data, that is, data from the German wholesale electricity market between 2017 and 2021. However, both the data and the code are present to allow an extension of the analysis to some other European markets, albeit with less context-specific controls.
    - simulation_plots.ipynb: Runs simulations, generates synthetic data, and visualizes the results to validate the theoretical arguments.


## Running the Code
There are three primary ways to utilize this repository:

1. Reproduce the Paper's Results: Start with the final .csv files in the 'results' folder to reproduce the exact plots featured in the paper. This will yield identical results to those reported.

2. Run Simulations and Visualize Results: Start with the downloaded data available in the repository and run simulations to generate new synthetic datasets before visualizing the results. The outcomes will be similar to (1), with differences arising from the random seeds used during synthetic data generation. On average, results will align closely with those in the paper, beyond minor variations due to random sampling error.

3. Full Data Acquisition and Analysis: If desired, you can generate the data from scratch by setting up the required accounts and querying the data sources. This approach involves creating a local database and storing the retrieved data. While theoretically possible, the availability of data from third-party sources is not guaranteed, and the process can be time-consuming. Ultimately, the results should match those obtained in (2), with the same caveats regarding randomness.
    * If you plan on retrieving the data yourself or expanding it, you should first make sure to have set up the file 'data/config.py'.