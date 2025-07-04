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

data = "pect_ndt_full_dataset.npz"
idea = json.load(open("idea.json", encoding="utf-8"))
results_dir = "coding-agent"


print("Step 1: Starting auto analysis...")
auto_success = auto_analyze_with_retry()
if not auto_success:
    print("❌ Auto analysis failed. Stopping workflow.")
    sys.exit(1)