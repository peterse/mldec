# ML Decoder Experiment Monitor

A Streamlit dashboard for monitoring ML decoder training experiments.

## Features

- View all experiments, runs, and jobs in a hierarchical structure
- Track completion status for runs and jobs
- View performance metrics from completed runs
- Visualize metrics through interactive charts
- See summary statistics across all experiments

## Installation

1. Ensure you have Python 3.8+ installed
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the Streamlit app:

```bash
streamlit run app.py
```

This will start the dashboard and automatically open it in your default web browser.

## Dashboard Structure

- **Top-level metrics**: Summary of experiments, completed and running runs
- **Experiment tabs**: Each experiment has its own tab showing:
  - Runs with timestamp and completion status
  - Progress bars for job completion
  - Detailed job status tables
  - Results from completed runs with visualizations

## Path Configuration

The dashboard is configured to look at the following path:
```
C:\Users\peter\Desktop\projects\mldec\src\mldec\tune_results
```

To change this path, edit the `RESULTS_DIR` variable in `app.py`. 