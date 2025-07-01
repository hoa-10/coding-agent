import os 
from base_code.coding_loop_enhance import generate_instruct_prompt
from base_code.processing_data import auto_analyze_with_retry
from dotenv import load_dotenv
from datetime import datetime
from perform_experiment import perform_experiments
from aider.coders import Coder
import pandas as pd
from aider.io import InputOutput
from aider.models import Model
os.environ['PYTHONIOENCODING'] = 'utf-8'
import json
import sys
import shutil
import os
import os.path as osp
load_dotenv()
def print_time():
    print(datetime.now().strftime("[%Y-%m-%d %H:%M:%S]"))

def do_idea(base_dir, results_dir, idea, model="gpt-3.5-turbo", log_file=False):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    idea_name = f"{timestamp}_{idea['Name']}"
    folder_name = osp.join(results_dir, idea_name)
    assert not osp.exists(folder_name), f"Folder {folder_name} already exists."
    shutil.copytree(base_dir, folder_name, dirs_exist_ok=True)

    with open(osp.join(base_dir, "results.json"), "r") as f:
        baseline_results = json.load(f)

    exp_file = osp.join(folder_name, "experiment.py")
    vis_file = osp.join(folder_name, "plot.py")
    notes = osp.join(folder_name, "notes.txt")
    

    with open(notes, "w", encoding='utf-8') as f:
        f.write(f"# Title: {idea['Title']}\n")
        f.write(f"# Experiment description: {idea['Experiment']}\n")
        f.write(f"## Run 0: Baseline\n")
        f.write(f"Results: {baseline_results}\n")
        f.write(f"Description: Baseline results.\n")

    if log_file:
        log_path = osp.join(folder_name, "log.txt")
        log = open(log_path, "a", encoding='utf-8')
        sys.stdout = log
        sys.stderr = log

    print_time()
    print(f"*Starting idea: {idea_name}*")
    fnames = [exp_file, vis_file, notes]
    io = InputOutput(yes=True, chat_history_file=f"{folder_name}/{idea_name}_aider.txt")
    
    main_model = Model(model)
    exp_exists = os.path.exists(exp_file)
    exp_size = os.path.getsize(exp_file) if exp_exists else 0
    

    coder = Coder.create(
        main_model=main_model,
        fnames=fnames,
        io=io,
        stream=False,
        use_git=False,
        edit_format="diff",
        auto_commits=False, 
        dirty_commits=True,  
    )

    print_time()
    print(f"*Starting Experiments*")
    
    try:
        with open(exp_file, 'r', encoding='utf-8') as f:
            initial_content = f.read()
        print(f"Initial experiment.py content length: {len(initial_content)}")
        print(f"Initial experiment.py preview:\n{initial_content[:200]}...")
    except Exception as e:
        print(f"Could not read initial experiment.py: {e}")
    
    try:
        success = perform_experiments(idea, folder_name, coder, baseline_results)
    except Exception as e:
        print(f"Error during experiments: {e}")
        print(f"Experiments failed for idea {idea_name}")
        return False

    if not success:
        print(f"Experiments failed for idea {idea_name}")
        return False

    return True

##################################
data = "sensor_data.csv"
data_info = pd.read_csv(data)
idea = json.load(open("idea.json", encoding="utf-8"))
results_dir = "coding-agent"
os.makedirs(results_dir, exist_ok=True)

# Step 1: Auto analyze with retry
print("Step 1: Starting auto analysis...")
auto_success = auto_analyze_with_retry()
if not auto_success:
    print("❌ Auto analysis failed. Stopping workflow.")
    sys.exit(1)

print("✅ Auto analysis completed successfully!")

# Step 2: Generate instruct prompt
print("Step 2: Generating instruct prompt...")
prompt_success = generate_instruct_prompt(data, data_info, idea)
if not prompt_success:
    print("❌ Instruct prompt generation failed. Stopping workflow.")
    sys.exit(1)

print("✅ Instruct prompt generated successfully!")

# Step 3: Execute the main experiment
print("Step 3: Starting main experiment...")
experiment_success = do_idea(
    base_dir="result",
    results_dir=results_dir, 
    idea=idea,
    model="gemini/gemini-2.5-flash",  
    log_file=True
)

if experiment_success:
    print("✅ Complete workflow finished successfully!")
else:
    print("❌ Experiment failed!")