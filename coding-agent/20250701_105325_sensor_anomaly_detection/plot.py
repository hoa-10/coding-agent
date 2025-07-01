import matplotlib.pyplot as plt
import pandas as pd
import json
import os

def load_results(run_dir):
    """Loads results from final_info.json in the specified run directory."""
    file_path = os.path.join(run_dir, 'final_info.json')
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            return json.load(f)
    return None

def plot_results(results_df, labels, output_dir='plots'):
    """Generates and saves bar plots for evaluation metrics."""
    os.makedirs(output_dir, exist_ok=True)

    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    
    # Ensure the 'Run Label' column is ordered correctly for plotting
    # by using a categorical type with the order from the labels dictionary
    results_df['Run Label'] = pd.Categorical(results_df['Run Label'], categories=list(labels.values()), ordered=True)
    results_df = results_df.sort_values('Run Label')

    for metric in metrics:
        plt.figure(figsize=(10, 6))
        bars = plt.bar(results_df['Run Label'], results_df[metric], color='skyblue')
        plt.xlabel('Experiment Run')
        plt.ylabel(metric.replace('_', ' ').title())
        plt.title(f'Comparison of {metric.replace("_", " ").title()} Across Experiments')
        plt.ylim(0.0, 1.05) # Set y-axis limit for metrics
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        # Add value labels on top of bars
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval + 0.02, round(yval, 3), ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{metric}_comparison.png'))
        plt.close()
        print(f"Saved {metric}_comparison.png to {output_dir}")

if __name__ == "__main__":
    # Define the run directories and their corresponding labels for plotting
    # Ensure these match the actual run directories created by experiment.py
    run_dirs = {
        'run_0': 'Baseline',
        'run_1': 'Smaller Window (30/15)',
        'run_2': 'Increased Overlap (60/45)',
        'run_3': 'Increased n_estimators (200)',
        'run_4': 'Larger Anomaly Magnitude',
        'run_5': 'Logistic Regression'
    }

    all_results = []

    # Load results for each run
    for run_folder, label in run_dirs.items():
        results = load_results(run_folder)
        if results:
            metrics = results['evaluation_metrics']
            params = results['pipeline_parameters']
            model_info = {
                'model_type': results.get('model_type', 'RandomForest'), # Default to RF if not specified
                'model_hyperparameters': results['model_hyperparameters']
            }
            
            row = {
                'Run Label': label,
                'accuracy': metrics['accuracy'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1_score': metrics['f1_score'],
                'window_size': params['window_size'],
                'overlap': params['overlap'],
                'anomaly_perturbation_factor_std': str(params['anomaly_perturbation_factor_std']),
                'model_type': model_info['model_type'],
                'model_hyperparameters': str(model_info['model_hyperparameters'])
            }
            all_results.append(row)
        else:
            print(f"Warning: Results not found for {run_folder}. Skipping.")

    if not all_results:
        print("No results found to plot. Please ensure run directories and final_info.json files exist.")
    else:
        results_df = pd.DataFrame(all_results)
        print("\nCollected Results:")
        print(results_df.to_string()) # Use to_string() to prevent truncation

        # Generate plots
        plot_results(results_df, run_dirs) # Pass run_dirs as labels

        print("\nPlotting complete. Check the 'plots' directory for generated images.")
