from datetime import datetime
from perform_experiment import perform_experiments
from aider.coders import Coder
from aider.io import InputOutput
from aider.models import Model
import os
import json
import sys
import shutil
import os
import os.path as osp
os.environ['PYTHONIOENCODING'] = 'utf-8'

def print_time():
    print(datetime.now().strftime("[%Y-%m-%d %H:%M:%S]"))

def do_idea(base_dir, results_dir, idea, model="gpt-3.5-turbo", log_file=False):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    idea_name = f"{timestamp}_{idea['Name']}"
    folder_name = osp.join(results_dir, idea_name)
    assert not osp.exists(folder_name), f"Folder {folder_name} already exists."
    shutil.copytree(base_dir, folder_name, dirs_exist_ok=True)

    with open(osp.join(base_dir, "run_0", "final_info.json"), "r") as f:
        baseline_results = json.load(f)
    if isinstance(baseline_results, dict):
        baseline_results = {k: v["means"] for k, v in baseline_results.items()}

    exp_file = osp.join(folder_name, "experiment.py")
    vis_file = osp.join(folder_name, "plot.py")
    notes = osp.join(folder_name, "notes.txt")
    
    # FIX: Thêm encoding='utf-8' khi ghi file
    with open(notes, "w", encoding='utf-8') as f:
        f.write(f"# Title: {idea['Title']}\n")
        f.write(f"# Experiment description: {idea['Experiment']}\n")
        f.write(f"## Run 0: Baseline\n")
        f.write(f"Results: {baseline_results}\n")
        f.write(f"Description: Baseline results.\n")

    if log_file:
        log_path = osp.join(folder_name, "log.txt")
        # FIX: Thêm encoding='utf-8' cho log file
        log = open(log_path, "a", encoding='utf-8')
        sys.stdout = log
        sys.stderr = log

    print_time()
    print(f"*Starting idea: {idea_name}*")
    fnames = [exp_file, vis_file, notes]
    io = InputOutput(yes=True, chat_history_file=f"{folder_name}/{idea_name}_aider.txt")
    
    # Use OpenAI model with more specific configuration
    main_model = Model(model)
    
    # Kiểm tra xem experiment.py có tồn tại và có nội dung không
    exp_exists = os.path.exists(exp_file)
    exp_size = os.path.getsize(exp_file) if exp_exists else 0
    

    coder = Coder.create(
        main_model=main_model,
        fnames=fnames,
        io=io,
        stream=False,
        use_git=False,
        edit_format="diff",
        auto_commits=False,  # Tắt auto commits để tránh conflicts
        dirty_commits=True,  # Cho phép commits với dirty state
    )

    print_time()
    print(f"*Starting Experiments*")
    
    # In ra nội dung experiment.py ban đầu để debug
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