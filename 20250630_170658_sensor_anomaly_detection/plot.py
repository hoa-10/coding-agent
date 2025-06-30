import matplotlib.pyplot as plt
import json
import os
import numpy as np

def load_results(run_dir):
    """Loads results from the final_info.json file in the specified run directory."""
    filepath = os.path.join(run_dir, "final_info.json")
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            return json.load(f)
    return None

def plot_metrics(all_results, metric_keys, title_suffix, filename_suffix):
    """Generates and saves bar plots for specified metrics across all runs."""
    labels = list(all_results.keys())
    
    # Define colors for each run for consistency
    colors = plt.cm.get_cmap('tab10', len(labels))

    fig, axes = plt.subplots(len(metric_keys), 1, figsize=(10, 5 * len(metric_keys)), sharex=True)
    if len(metric_keys) == 1:
        axes = [axes] # Ensure axes is iterable even for a single subplot

    for i, metric_key in enumerate(metric_keys):
        metric_values = [all_results[label][metric_key] for label in labels]
        
        bars = axes[i].bar(labels, metric_values, color=colors(np.arange(len(labels))))
        axes[i].set_ylabel(metric_key.replace('_', ' ').title())
        axes[i].set_title(f'{metric_key.replace("_", " ").title()} {title_suffix}')
        axes[i].grid(axis='y', linestyle='--', alpha=0.7)
        axes[i].set_ylim(0, 1.05) # Metrics are typically between 0 and 1

        # Add value labels on top of bars
        for bar in bars:
            yval = bar.get_height()
            axes[i].text(bar.get_x() + bar.get_width()/2, yval + 0.02, round(yval, 3), ha='center', va='bottom')

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'plots/metrics_{filename_suffix}.png')
    plt.close()
    print(f"Generated plots/metrics_{filename_suffix}.png")

if __name__ == "__main__":
    # Ensure plots directory exists
    os.makedirs('plots', exist_ok=True)

    # Define the runs and their display labels for the plots
    labels = {
        "run_0": "Baseline (RF, WS=10, S=1)",
        "run_1": "RF, n_est=200",
        "run_2": "RF, WS=20",
        "run_3": "RF, S=5",
        "run_4": "Logistic Regression",
        "run_5": "RF, n_est=200, WS=20"
    }

    all_run_results = {}
    for run_dir_name, display_label in labels.items():
        results = load_results(os.path.join(os.getcwd(), run_dir_name))
        if results:
            all_run_results[display_label] = results
        else:
            print(f"Warning: Results for {run_dir_name} not found. Skipping this run.")

    if not all_run_results:
        print("No results found to plot. Ensure experiment runs have completed and 'final_info.json' exists in run_X directories.")
    else:
        # Metrics to plot
        val_metrics = ['val_accuracy', 'val_precision', 'val_recall', 'val_f1']
        test_metrics = ['test_accuracy', 'test_precision', 'test_recall', 'test_f1']

        # Generate plots for validation metrics
        plot_metrics(all_run_results, val_metrics, "on Validation Set", "val_set")

        # Generate plots for test metrics
        plot_metrics(all_run_results, test_metrics, "on Test Set", "test_set")

    print("\nPlotting complete.")
