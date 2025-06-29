import os 
from dotenv import load_dotenv
load_dotenv()

os.environ['PYTHONIOENCODING'] = 'utf-8'
from aider_coding.test_experiment import do_idea
import json
idea = json.load(open("idea.json", encoding="utf-8"))
results_dir = "base_code"
os.makedirs(results_dir, exist_ok=True)
# Chạy experiment
success = do_idea(
    base_dir="experiment_base",
    results_dir=results_dir, 
    idea=idea,
    model="gemini/gemini-2.5-flash",  # hoặc "gpt-4", "gpt-3.5-turbo"
    log_file=True
)
if success:
    print("Experiment completed successfully!")
else:
    print("Experiment failed!")