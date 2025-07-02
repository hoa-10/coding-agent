import matplotlib.pyplot as plt
import json
import os
import numpy as np

def plot_results():
    # Define the runs and their labels for plotting
    labels = {
        'run_0': 'SVC (Baseline)',
        'run_1': 'RandomForest',
        'run_2': 'LogisticRegression (Balanced)',
        'run_3': 'SVC Linear (Balanced)',
        'run_4': 'RandomForest (Balanced)',
        'run_5': 'SVC RBF (Balanced)'
    }

    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score']
    
    # Create a directory for plots if it doesn't exist
    plots_dir = 'plots'
    os.makedirs(plots_dir, exist_ok=True)

    # Iterate through each run to create a separate plot
    for run_dir, run_label in labels.items():
        results_file_path = os.path.join(run_dir, 'final_info.json')
        
        if os.path.exists(results_file_path):
            with open(results_file_path, 'r') as f:
                results = json.load(f)
                evaluation_metrics = results['evaluation_metrics']
                
                # Extract metrics for the current run
                metric_values = [evaluation_metrics.get(metric, 0.0) for metric in metrics_to_plot]
                
                # Create a new figure for each run
                fig, ax = plt.subplots(figsize=(8, 6)) # Adjust figure size as needed

                x = np.arange(len(metrics_to_plot)) # the label locations
                bar_width = 0.5 # Width of the bars

                ax.bar(x, metric_values, bar_width, color='skyblue') # Use a single color for clarity per run
                
                ax.set_xlabel('Metrics', fontsize=12)
                ax.set_ylabel('Score', fontsize=12)
                ax.set_title(f'Performance for {run_label}', fontsize=14)
                ax.set_xticks(x)
                ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics_to_plot], rotation=45, ha="right")
                ax.set_ylim(0, 1.1) # Metrics are typically between 0 and 1
                ax.grid(axis='y', linestyle='--', alpha=0.7)

                # Add value labels on top of bars
                for i, v in enumerate(metric_values):
                    ax.text(x[i], v + 0.05, f'{v:.2f}', ha='center', va='bottom', fontsize=10)

                plt.tight_layout()
                
                # Save the plot with a unique filename for each run
                plot_file_name = f'{run_dir}_performance.png'
                plot_file_path = os.path.join(plots_dir, plot_file_name)
                plt.savefig(plot_file_path)
                print(f"Plot for {run_label} saved to {plot_file_path}")
                plt.close(fig) # Close the figure to free memory
        else:
            print(f"Warning: Results file not found for {run_dir} at {results_file_path}. Skipping plot generation for this run.")

if __name__ == "__main__":
    plot_results()
