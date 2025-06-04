# Pedestrian Intent Prediction for Autonomous Mobile Robots

## Overview

This project aims to predict pedestrian intent for autonomous mobile robots using advanced machine learning techniques. By processing RGB and depth data, the system analyzes pedestrian actions and intentions to support safer, more efficient navigation in dynamic environments.

## Features

- **Dataset Management**: Tools for handling, validating, and preparing datasets.
- **Action Recognition**: Annotation tools for labeling pedestrian actions to train recognition models.
- **Intent Prediction**: Frameworks for labeling and analyzing pedestrian intentions.
- **Data Extraction**: Utilities for extracting data from various formats, including ROS bag files.
- **Visualization**: Tools to visualize combined RGB and depth data for better interpretability.

## Installation

### Requirements

- Python 3.8+
- Anaconda (recommended)
- Python packages:
  - `opencv-python`
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `sphinx`

### Installation Steps

1. **Clone the repository**:

   ```bash
   git clone https://github.com/FelipeZerokun/intent_prediction.git
   cd intent_prediction
   ```

2. **Create and activate a virtual environment**:

   ```bash
   conda create -n intent_pred_pytorch python=3.8
   conda activate intent_pred_pytorch
   ```

3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Build the documentation** (optional):

   ```bash
   cd docs
   make html
   ```

## Usage

Documentation and usage examples will be available in the `docs` directory after building the HTML documentation.

## License

This project is licensed under the [MIT License](LICENSE).

## Contact

For questions or collaborations, please contact **Felipe Zerokun** via [GitHub](https://github.com/FelipeZerokun).
