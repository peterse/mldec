import streamlit as st
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import re
import time

# Set page configuration
st.set_page_config(
    page_title="ML Decoder Experiment Monitor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define paths
RESULTS_DIR = "C:/Users/peter/Desktop/projects/mldec/src/mldec/tune_results"

# Functions to get experiment status
def get_experiment_data():
    """Get all experiments and their status"""
    experiments = []
    
    # Get all experiment directories
    for exp_name in os.listdir(RESULTS_DIR):
        exp_path = os.path.join(RESULTS_DIR, exp_name)
        if not os.path.isdir(exp_path):
            continue
            
        # Get all runs in this experiment
        runs = []
        total_runs = 0
        completed_runs = 0
        
        for run_name in os.listdir(exp_path):
            run_path = os.path.join(exp_path, run_name)
            if not os.path.isdir(run_path) or not run_name.startswith("run_"):
                continue
                
            # Parse date from run name
            run_date_str = run_name.replace("run_", "")
            try:
                run_date = datetime.strptime(run_date_str[:19], "%Y-%m-%d-%H-%M-%S")
                run_date_display = run_date.strftime("%Y-%m-%d %H:%M:%S")
            except:
                run_date_display = run_name
            
            # Check if run is completed (has CSV file)
            csv_files = [f for f in os.listdir(run_path) if f.endswith('.csv')]
            is_completed = len(csv_files) > 0
            
            # Count jobs and completed jobs
            jobs = []
            total_jobs = 0
            completed_jobs = 0
            
            # Look for job directories (zjob_*)
            for item in os.listdir(run_path):
                job_path = os.path.join(run_path, item)
                if os.path.isdir(job_path) and item.startswith("zjob_"):
                    total_jobs += 1
                    has_config = os.path.exists(os.path.join(job_path, "hyper_config.json"))
                    has_results = any(f.endswith('.csv') for f in os.listdir(job_path))
                    job_status = "Completed" if has_config or has_results else "Running"
                    
                    if job_status == "Completed":
                        completed_jobs += 1
                    
                    jobs.append({
                        "job_name": item,
                        "status": job_status,
                        "path": job_path
                    })
            
            # Add run data
            total_runs += 1
            if is_completed:
                completed_runs += 1
                
            runs.append({
                "run_name": run_name,
                "display_name": run_date_display,
                "status": "Completed" if is_completed else "Running",
                "jobs": jobs,
                "total_jobs": total_jobs,
                "completed_jobs": completed_jobs,
                "path": run_path
            })
        
        # Sort runs by name (newest first)
        runs.sort(key=lambda x: x["run_name"], reverse=True)
        
        # Add experiment data
        experiments.append({
            "name": exp_name,
            "runs": runs,
            "total_runs": total_runs,
            "completed_runs": completed_runs,
            "path": exp_path
        })
    
    return experiments

# Define color scheme for status
def get_status_color(status):
    if status == "Completed":
        return "#00cc96"  # Green
    elif status == "Running":
        return "#ff9f1c"  # Orange
    else:
        return "#ef553b"  # Red

# Main app
def main():
    st.title("ML Decoder Experiment Monitor 🧠")
    
    # Add refresh rate selector in sidebar
    st.sidebar.title("Settings")
    refresh_rate = st.sidebar.selectbox(
        "Auto-refresh interval",
        options=[0, 30, 60, 300, 600],
        format_func=lambda x: "Disabled" if x == 0 else f"Every {x} seconds",
        index=1
    )
    
    # Add last updated time
    last_update = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.sidebar.info(f"Last updated: {last_update}")
    
    # Add manual refresh button
    if st.sidebar.button("Refresh Now"):
        st.experimental_rerun()
    
    # Get experiment data
    experiments = get_experiment_data()
    
    # Display summary metrics
    col1, col2, col3 = st.columns(3)
    total_experiments = len(experiments)
    total_runs = sum(exp["total_runs"] for exp in experiments)
    completed_runs = sum(exp["completed_runs"] for exp in experiments)
    running_runs = total_runs - completed_runs
    
    col1.metric("Total Experiments", total_experiments)
    col2.metric("Completed Runs", completed_runs)
    col3.metric("Running Runs", running_runs)
    
    # Create tabs for each experiment
    if experiments:
        tabs = st.tabs([exp["name"] for exp in experiments])
        
        for i, exp in enumerate(experiments):
            with tabs[i]:
                st.subheader(f"Experiment: {exp['name']}")
                
                # Display run information
                for run in exp["runs"]:
                    with st.expander(
                        f"{run['display_name']} - {run['status']} ({run['completed_jobs']}/{run['total_jobs']} jobs completed)",
                        True if run["status"] == "Running" else False
                    ):
                        # Progress bar for jobs
                        if run["total_jobs"] > 0:
                            progress = run["completed_jobs"] / run["total_jobs"]
                            st.progress(progress)
                        
                        # Job details
                        if run["jobs"]:
                            job_data = []
                            for job in run["jobs"]:
                                job_data.append({
                                    "Job": job["job_name"],
                                    "Status": job["status"]
                                })
                            
                            df = pd.DataFrame(job_data)
                            
                            # Use custom styling for status
                            def color_status(val):
                                color = get_status_color(val)
                                return f'background-color: {color}; color: white;'
                            
                            styled_df = df.style.applymap(color_status, subset=['Status'])
                            st.dataframe(styled_df, use_container_width=True)
                        else:
                            st.info("No jobs found for this run")
                        
                        # If there's a CSV file, load and display it
                        csv_files = [f for f in os.listdir(run["path"]) if f.endswith('.csv')]
                        if csv_files:
                            st.subheader("Results")
                            csv_path = os.path.join(run["path"], csv_files[0])
                            try:
                                results_df = pd.read_csv(csv_path)
                                st.dataframe(results_df, use_container_width=True)
                                
                                # Plot if there are numeric columns that look like metrics
                                metrics_cols = [col for col in results_df.columns if any(x in col.lower() for x in ['acc', 'loss', 'score', 'val'])]
                                if metrics_cols:
                                    st.subheader("Performance Metrics")
                                    fig, ax = plt.subplots(figsize=(10, 6))
                                    for col in metrics_cols:
                                        if pd.api.types.is_numeric_dtype(results_df[col]):
                                            sns.lineplot(data=results_df, x=results_df.index, y=col, label=col, ax=ax)
                                    plt.legend()
                                    st.pyplot(fig)
                            except Exception as e:
                                st.error(f"Error loading results: {e}")
    else:
        st.warning("No experiments found in the specified directory.")
    
    # Set up auto-refresh
    if refresh_rate > 0:
        time.sleep(4)  # Small delay to ensure the UI has updated
        st.empty()  # Creates an empty element to be replaced
        time.sleep(refresh_rate)  # Wait for the specified refresh rate
        st.experimental_rerun()  # Rerun the app

if __name__ == "__main__":
    main() 