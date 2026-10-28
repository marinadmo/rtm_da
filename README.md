# Project Overview

This repository contains scripts and configuration files for running RTM (Radiative Transfer Model) simulations, data assimilation with EnKF, and generating figures from the results.

---

## Folder Structure

### `config.py`
Example file defining paths and variable names.

### `plotting_codes/`
Scripts to generate figures from the results.

### `rtm_dal_param/`
Contains RTM code and DAL parameterization scripts.

- **`rtm_atms_fcts.py`**: Main RTM script that supports different ice emissivity parameterizations.
- **`main_fct_rtm_tb.py`**: A set of functions to run the RTM, read data, and save results.
- **`dynamic_tiepoints/` and `pmr_codes/`**: Folders containing scripts needed for DAL parameterization (e.g., computation of coefficients).
- **`steps_da/`**: Scripts for data assimilation.

---

## Quick Start Guide

### 1. Configure the `config.py` File
Define your configuration file with:
- Paths to data and results
- Experiment name
- Other required parameters

---

### 2. Load Necessary Modules
Load the required Python or `conda` environment and EnKF-C module:

```bash
conda activate production-10-2022
module load enkfc/2.9.9

### 3. Run assimilation experiments
- Produce the RTM Tbs from model data
- Prepare data and perform assimilation wiht EnKF
- Generate figures
