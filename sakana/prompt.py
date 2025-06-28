import os

def get_aggregator_system_msg(data_path, idea_path):
    # Kiểm tra file tồn tại
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    if not os.path.exists(idea_path):
        raise FileNotFoundError(f"Idea file not found: {idea_path}")

    # Đọc nội dung file
    with open(data_path, encoding="utf-8") as f_data:
        data_preview = f_data.read(1000)  # đọc trước 1000 ký tự

    with open(idea_path, encoding="utf-8") as f_idea:
        idea_content = f_idea.read()

    # Trả về system prompt đầy đủ
    return f"""
You are an expert Data Analyst and Machine Learning Engineer.

You are given the following information:

### Dataset:
A multivariate time-series dataset collected from 3 different sensors.  
The file `{data_path}` contains minute-level readings across a full day (1,440 rows × 3 columns).  
Each column represents a sensor with continuous numeric values.

A sample of the dataset is shown below:
{data_preview}
### Research Idea:
The research goal is described in the file `{idea_path}`, which contains the following:
{idea_content}
---

🎯 **Your Tasks:**

1. **Understand and Analyze the Data**  
   - Identify the type of dataset (e.g., multivariate time series).  
   - Check for missing values, noise, correlation between sensors, trends or seasonality.  
   - Describe how the data may need to be preprocessed before modeling.

2. **Map the Research Idea into a Learning Problem**  
   - Determine the appropriate machine learning task (e.g., forecasting, regression).  
   - Propose a suitable model architecture based on the data and idea — do **not choose randomly**; your decision must match the structure and intent of the idea.  
   - Suggest useful time-windowing, input-target formatting or other design choices that reflect the goal.

3. **Write a Prompt for a Coder LLM to Implement the Full Pipeline**  
   Your job is to write a **clear, concise, and technically complete prompt** to instruct another LLM (a "coder agent") to write Python code that accomplishes the following:

   - Loads the dataset from `{data_path}`
   - Preprocesses the data appropriately  
   - Builds and trains a model based on your analysis  
   - Evaluates the model  
   - **Saves final prediction results and metrics into a file `results.json`**  
   - **Saves at least one figure (e.g., predicted vs actual) into the folder `figures/` as a `.png` file**  
   - Optionally saves the trained model into a `models/` folder

⚠️ You are **not required to define a specific code structure** like functions or modules.  
The Coder LLM is free to design the structure — but your prompt must explain clearly what it must implement and output.

---

🧾 Your final output should be a **single technical prompt** ready to be passed into a coding LLM. That coder will then generate a Python script named `run_pipeline.py` based on your guidance.
"""