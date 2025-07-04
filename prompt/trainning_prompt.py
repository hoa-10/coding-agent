def coding_instruct_prompt(data_info_path, idea):
    return f"""
**Instructions for Implementation LLM: Generate Machine Learning Pipeline Prompt**

Your task is to analyze the dataset information located at `{data_info_path}`and the research idea: '{idea}'. Based on this analysis, generate a detailed, clear, and concise prompt that instructs another LLM (the instruct LLM) to write a high-quality Python script 
this is the required code you must to instruct llm to write when loading the dataset:
 ```python
 import numpy as np

 data = np.load('pect_ndt_full_dataset.npz')
 X_train = data['X_train']
 y_train = data['y_train']
 X_valid = data['X_valid']
 y_valid = data['y_valid']
 X_scan = data['X_scan']
 X_in_corr = data['X_in_corr']
 Xc = data['Xc']
 Xg = data['Xg']
 m = data['m']
 st = data['st']
---
### Steps to Follow:
1. **you will analyze and identify all this task**:
   - Identify key characteristics such as number of samples, features, data types, missing values, correlations, or temporal patterns.
   - Determine the task type (e.g., classification, regression, clustering) if specified, or infer it from the dataset and idea.

2. **Understand the Research Idea**:
   - Carefully analyze the idea: '{idea}' to understand the deep learning task and its objectives (e.g., what to predict or optimize).

3. **Design the Pipeline**:
   - **Model Selection**: Choose an appropriate model based on the task and dataset 
   - **Hyperparameters**: Select reasonable hyperparameters based on dataset size and complexity.
   - **Preprocessing**: Specify preprocessing steps (e.g., impute missing values, scale features, encode categorical variables, feature engineering if needed).
   - **Training**: Define the training process, including data splitting (e.g., 80% train, 10% validation, 10% test) and any validation techniques (e.g., cross-validation).
   - **Evaluation Metrics**: Select multiple benchmarks suitable for the task 
   - Ensure llm instruct this prompt code that limitate rarely library , which harly install 
4. **Generate the Prompt**:
   - Create a prompt for the instruct LLM to write a Python script named `run_pipeline.py`.
   - The prompt must include:
     - **Task Overview**: Briefly describe the task and goal based on the idea.
     - **Data Loading**: Specify how to load the dataset from its path and any specific parameters.
     - **Preprocessing Steps**: List all preprocessing steps in sequence.
     - **Model Configuration**: Define the model and its hyperparameters.
     - **Training Process**: Explain how to train the model, including data splits and validation.
     - **Evaluation**: Instruct to compute multiple evaluation metrics  on the test set.
     - **Save Results**: Direct the LLM to:
       - Create a "result" folder if it does not exist (e.g., using `os.makedirs`).
       - Save a JSON file (e.g., `results.json`) in the "result" folder containing:
         - All evaluation metrics computed on the test set.
         - Hyperparameters used in the model.
       - Ensure the script is modular and adaptable.
IMPORTANT REQUIREMENTS:
- Do NOT create dummy/fake/synthetic data
- Do NOT generate random data if the data file doesn't exist
- Only work with the actual provided data file
- If the data file is missing, print an error message and exit gracefully
- Focus on analyzing the real data provided, not simulated data
### Additional Requirements:
- The prompt must be detailed, specific to the dataset and idea, and follow a common structure for training AI models.
- Do not include notes or comments in the generated source code.
- Ensure the script uses multiple evaluation benchmarks as appropriate for the task.
- The final script must save results in a "result" folder in a structured format (e.g., JSON).
- prompt must be ensure that llm won"t use the random data or random seed in the code, so that the results are reproducible.

"""
