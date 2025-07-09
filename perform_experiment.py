import json
import os
import os.path as osp
import shutil
import subprocess
import sys
from subprocess import TimeoutExpired
import hashlib
import time

# Set encoding globally
os.environ['PYTHONIOENCODING'] = 'utf-8'

MAX_ITERS = 6
MAX_RUNS = 2
MAX_STDERR_OUTPUT = 1500

coder_prompt = """
Your task is to implement the following idea: {title}

Proposed Experiment: {idea}

Available Runs: Up to {max_runs} (you do not need to use all).
###requirment:
you need to load dataset with path : data_path = r'C:\\Users\\user\\Desktop\\coding-agent\\pect_ndt_full_dataset.npz'
Instructions:
Propose Experiments:
Based on the idea and baseline results, propose and justify a sequence of experiments with specific parameter settings.
you incapable to plan for all of process, you just need plan for first experiment., each result after should be analyzed by you to adjust other best approach for suitable
Implement in experiment.py:
Modify the provided experiment.py to include your list of parameter settings.
Parse --out_dir=run_i to select the i-th experiment's settings.
Save results to final_info.json in out_dir for each run.
Important:
Do not re-run the baseline; use provided results.
Modify experiment.py directly; do not just describe changes.
Use only --out_dir in the command; no additional args.
Baseline Results: {baseline_results}
Running: We will run python experiment.py --out_dir=run_i for each proposed experiment.
IMPORTANT: You must modify the experiment.py file to implement your changes. The file already contains baseline code that you should modify.
You can then implement the next thing on your list.

PLEASE MAKE SURE TO ACTUALLY MODIFY THE EXPERIMENT.PY FILE WITH YOUR CHANGES. DO NOT JUST DESCRIBE WHAT TO DO."""

def ensure_result_file_exists(folder_name, run_num):
    """Create a dummy result file if experiment didn't create one"""
    result_file = osp.join(folder_name, f"run_{run_num}", "final_info.json")
    if not osp.exists(result_file):
        os.makedirs(osp.dirname(result_file), exist_ok=True)
        # Create dummy results
        dummy_results = {"error": "experiment did not save results", "status": "failed"}
        try:
            with open(result_file, "w", encoding='utf-8') as f:
                json.dump(dummy_results, f)
        except Exception as e:
            print(f"Could not create dummy result file: {e}")
    return result_file



def run_experiment(folder_name, run_num, timeout=3600):  # Reduced timeout
    cwd = osp.abspath(folder_name)
    
    # COPY CODE SO WE CAN SEE IT.
    try:
        shutil.copy(
            osp.join(folder_name, "experiment.py"),
            osp.join(folder_name, f"run_{run_num}.py"),
        )
    except Exception as e:
        print(f"Could not copy experiment.py: {e}")

    # LAUNCH COMMAND
    command = [
        sys.executable,  # Use current Python executable
        "experiment.py",
        f"--out_dir=run_{run_num}",
    ]
    
    try:
        result = subprocess.run(
            command, 
            cwd=cwd, 
            stderr=subprocess.PIPE, 
            stdout=subprocess.PIPE, 
            text=True, 
            timeout=timeout,
            encoding='utf-8'
        )

        if result.stdout:
            print(f"Run {run_num} stdout:", result.stdout)
            
        if result.stderr:
            print(f"Run {run_num} stderr:", result.stderr, file=sys.stderr)

        if result.returncode != 0:
            print(f"Run {run_num} failed with return code {result.returncode}")
            stderr_output = result.stderr
            if len(stderr_output) > MAX_STDERR_OUTPUT:
                stderr_output = "..." + stderr_output[-MAX_STDERR_OUTPUT:]
            
            # Still try to create result file
            ensure_result_file_exists(cwd, run_num)
            next_prompt = f"Run failed with the following error {stderr_output}. Please fix the experiment.py to ensure it saves results to final_info.json"
            return 1, next_prompt
        else:
            # Check if final_info.json exists
            result_file = ensure_result_file_exists(cwd, run_num)
            
            try:
                with open(result_file, "r", encoding='utf-8') as f:
                    results = json.load(f)
            except Exception as e:
                print(f"Error reading results: {e}")
                results = {"error": f"could not read results: {e}"}
            
            # Handle different result formats
            if isinstance(results, dict) and any("means" in str(v) for v in results.values()):
                results = {k: v["means"] if isinstance(v, dict) and "means" in v else v for k, v in results.items()}

            next_prompt = f"""Run {run_num} completed.  
Here are the latest results:  
{results}  


1. **Analyze** these results—compare current vs. previous metrics.  
2. **Decide** if and how to tweak your experiment (hyper‑parameters, architecture, data processing, etc.) to improve the key metric(s).  
3. **Edit only** the `experiment.py` file to implement your chosen change.  
   - Do **not** add any new command‑line arguments.  
   - Keep all logic for saving your run’s outputs to `final_info.json` in the target `--out_dir`.  
4. **Write** a verbose block of text (for `notes.txt`) that documents:
   - Which run this is (`Run {run_num}`),  
   - A clear description of the experiment you just ran,  
   - What you will change and **why**, linked to the metric differences,  
   - The exact command we will use next:  
     ```
     python experiment.py --out_dir=run_{run_num + 1}
     ```  
5. If **no further improvement** is expected or you’ve exhausted your plan, respond with `ALL_COMPLETED`.  

**IMPORTANT**  
- After your edit, we’ll launch only:  
  ```bash
  python experiment.py --out_dir=run_{run_num + 1}
"""
            
        return result.returncode, next_prompt
        
    except TimeoutExpired:
        print(f"Run {run_num} timed out after {timeout} seconds")
        if osp.exists(osp.join(cwd, f"run_{run_num}")):
            try:
                shutil.rmtree(osp.join(cwd, f"run_{run_num}"))
            except:
                pass
        next_prompt = f"Run timed out after {timeout} seconds. Please optimize your experiment.py to run faster."
        return 1, next_prompt
    except Exception as e:
        print(f"Unexpected error in run {run_num}: {e}")
        ensure_result_file_exists(cwd, run_num)
        next_prompt = f"Unexpected error: {e}. Please fix the experiment.py"
        return 1, next_prompt


# RUN PLOTTING
def run_plotting(folder_name, timeout=300):  # Reduced timeout
    cwd = osp.abspath(folder_name)
    command = [sys.executable, "plot.py"]
    
    try:
        result = subprocess.run(
            command, 
            cwd=cwd, 
            stderr=subprocess.PIPE, 
            stdout=subprocess.PIPE, 
            text=True, 
            timeout=timeout,
            encoding='utf-8'
        )

        if result.stdout:
            print("Plot stdout:", result.stdout)
            
        if result.stderr:
            print("Plot stderr:", result.stderr, file=sys.stderr)

        if result.returncode != 0:
            print(f"Plotting failed with return code {result.returncode}")
            next_prompt = f"Plotting failed with the following error {result.stderr}"
        else:
            next_prompt = ""
        return result.returncode, next_prompt
        
    except TimeoutExpired:
        print(f"Plotting timed out after {timeout} seconds")
        next_prompt = f"Plotting timed out after {timeout} seconds"
        return 1, next_prompt
    except Exception as e:
        print(f"Plotting error: {e}")
        next_prompt = f"Plotting error: {e}"
        return 1, next_prompt


def file_hash(filepath):
    """Calculate MD5 hash of a file"""
    try:
        with open(filepath, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()
    except Exception as e:
        print(f"Error calculating hash for {filepath}: {e}")
        return None


def read_file_content(filepath):
    """Read file content for comparison"""
    try:
        with open(filepath, "r", encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None


# PERFORM EXPERIMENTS
# Modify the plotting phase in perform_experiments function
def perform_experiments(idea, folder_name, coder, baseline_results, gui_mode=False, plot_feedback_callback=None) -> bool:
    current_iter = 0
    run = 1
    next_prompt = coder_prompt.format(
        title=idea["Title"],
        idea=idea["Experiment"],
        max_runs=MAX_RUNS,
        baseline_results=baseline_results,
    )
    
    print(f"Starting experiments for: {idea['Title']}")
    
    exp_file = os.path.join(folder_name, "experiment.py")
    
    while run < MAX_RUNS + 1:
        if current_iter >= MAX_ITERS:
            print(f"Max iterations reached for run {run}")
            break
            
        try:
            old_hash = file_hash(exp_file)
            old_content = read_file_content(exp_file)
            
            print(f"Getting coder output for run {run}, iteration {current_iter}")
            print(f"File hash before coder: {old_hash}")
            
            coder_out = coder.run(next_prompt)
            print(f"Coder output: {coder_out}")
            
            time.sleep(1)
            
            new_hash = file_hash(exp_file)
            new_content = read_file_content(exp_file)
            
            print(f"File hash after coder: {new_hash}")
            
            if old_hash == new_hash or old_content == new_content:
                print("WARNING: experiment.py was not changed by AI coder!")
                print("This might indicate the AI didn't actually modify the file.")
                
                force_edit_prompt = f"""
The experiment.py file was not modified in your previous response. You MUST actually edit the experiment.py file to implement the changes.
After each run, only save the results of the current run to the final_info.json file in the output directory. Do not aggregate or overwrite results from other runs.
Current experiment.py content preview:
{new_content[:500]}...
> - When running the command: python experiment.py --out_dir=run_i, only perform the experiment for run number i (that is, index i-1 in your list of parameter settings).

Please MODIFY the experiment.py file now to implement the next experiment in your plan.
For example, if you're changing from 1 hidden layer to 2 hidden layers, you need to actually change the model architecture in the code.

DO NOT just describe what to do - ACTUALLY EDIT THE FILE.

If you need to make major changes, you can rewrite the entire file using the 'whole' edit format.
"""
                print("Forcing file edit with more specific prompt...")
                
                try:
                    current_format = getattr(coder, 'edit_format', 'diff')
                    if current_format == 'diff':
                        print("Switching to 'whole' edit format for major changes...")
                        coder.edit_format = 'whole'
                    
                    coder_out = coder.run(force_edit_prompt)
                    
                except Exception as format_error:
                    print(f"Error with format change: {format_error}")
                    coder_out = coder.run(force_edit_prompt)
                
                time.sleep(1)
                final_hash = file_hash(exp_file)
                final_content = read_file_content(exp_file)
                
                if old_hash == final_hash or old_content == final_content:
                    print("ERROR: AI coder still didn't modify the file after forced prompt!")
                    current_iter += 1
                    continue
                else:
                    print("SUCCESS: File was modified after forced prompt")

            if "ALL_COMPLETED" in coder_out:
                print("All experiments completed!")
                break
                
        except Exception as e:
            print(f"Coder error: {e}")
            current_iter += 1
            continue
            
        print(f"Running experiment {run}")
        return_code, next_prompt = run_experiment(folder_name, run)
        
        if return_code == 0:
            print(f"Run {run} successful")
            run += 1
            current_iter = 0
        else:
            print(f"Run {run} failed, retrying")
            current_iter += 1
            
    if current_iter >= MAX_ITERS and run < MAX_RUNS + 1:
        print("Not all experiments completed due to max iterations.")
        return False

    print("Starting plotting phase...")
    
    initial_plot_prompt = """
Great job! Please modify `plot.py` to generate the most relevant plots for the final writeup. 
In particular, be sure to fill in the "labels" dictionary with the correct names for each run that you want to plot.
Only the runs in the `labels` dictionary will be plotted, so make sure to include all relevant runs.

Create plots that clearly show:
- Performance comparisons between different runs
- Key metrics and their trends
- Clear legends and labels
- Professional appearance

We will be running the command `python plot.py` to generate the plots.
"""
    
    try:
        coder.run(initial_plot_prompt)
        
        # Generate initial plots
        print("\n" + "="*60)
        print("📊 GENERATING INITIAL PLOTS")
        print("="*60)
        
        return_code, error_msg = run_plotting(folder_name)
        
        if return_code != 0:
            print(f"❌ Initial plotting failed: {error_msg}")
            print("🔧 Trying to fix plotting errors...")
            
            fix_prompt = f"""
Plotting failed with error: {error_msg}
Please fix the plot.py file to resolve this error.
Make sure to check:
- All required imports are present
- Data files exist and are readable
- Plot syntax is correct
- Output directory is accessible
"""
            try:
                coder.run(fix_prompt)
                return_code, error_msg = run_plotting(folder_name)
                
                if return_code != 0:
                    print(f"❌ Still failed after fix attempt: {error_msg}")
                    return False
                    
            except Exception as e:
                print(f"Failed to fix plotting error: {e}")
                return False
        
        print("✅ Initial plots generated successfully!")
        print(f"📁 Plots available in folder: {folder_name}")
        
        # If GUI mode, return True and let GUI handle plot feedback
        if gui_mode:
            return True
            
        # Terminal mode: Continue with input() as before
        while True:
            print("\n" + "="*50)
            print("👤 USER FEEDBACK")
            print("="*50)
            
            user_feedback = input("📝 Your feedback on the plots (type 'ok' if satisfied, or describe changes needed): ").strip()
            
            if user_feedback.lower() == 'ok':
                print("✅ Plots approved by user!")
                break
            
            print(f"📝 Applying user feedback: {user_feedback}")
            
            feedback_prompt = f"""
The user has reviewed the generated plots and provided this feedback:

USER FEEDBACK:
{user_feedback}

Please modify the plot.py file to address this feedback. Consider:
- Changing colors, styles, or plot types
- Modifying titles, labels, legends
- Adding or removing plot elements
- Improving readability and visual appeal
- Making data comparisons clearer
- Adding statistical information if needed
- Changing figure sizes or layouts

Make sure the modified plot.py still:
- Reads the results from all run directories
- Generates meaningful visualizations
- Saves plots with appropriate filenames
- Works with the existing data structure

ACTUALLY MODIFY THE PLOT.PY FILE - don't just describe what to do.
"""
            
            try:
                print("🔧 Applying feedback with Aider...")
                coder.run(feedback_prompt)
                print("✅ Feedback applied, regenerating plots...")
                
                return_code, error_msg = run_plotting(folder_name)
                
                if return_code != 0:
                    print(f"❌ Plot regeneration failed: {error_msg}")
                    retry = input("❓ Would you like to try again? (yes/no): ")
                    if retry.lower() != "yes":
                        break
                else:
                    print("✅ Plots regenerated successfully!")
                    
            except Exception as e:
                print(f"❌ Error applying feedback: {e}")
                retry = input("❓ Would you like to try again? (yes/no): ")
                if retry.lower() != "yes":
                    break
                    
    except Exception as e:
        print(f"Plotting phase error: {e}")
        
    # Notes phase
    try:
        next_prompt = """
Please modify `notes.txt` with a description of what each plot shows along with the filename of the figure. Please do so in-depth.

Somebody else will be using `notes.txt` to write a report on this in the future.
"""
        coder.run(next_prompt)
    except Exception as e:
        print(f"Notes phase error: {e}")

    return True

def apply_plot_feedback(folder_name, coder, feedback):
    try:
        print(f"📝 Applying plot feedback: {feedback}")
        
        feedback_prompt = f"""
The user has reviewed the generated plots and provided this feedback:

USER FEEDBACK:
{feedback}

Please modify the plot.py file to address this feedback. Consider:
- Changing colors, styles, or plot types
- Modifying titles, labels, legends
- Adding or removing plot elements
- Improving readability and visual appeal
- Making data comparisons clearer
- Adding statistical information if needed
- Changing figure sizes or layouts

Make sure the modified plot.py still:
- Reads the results from all run directories
- Generates meaningful visualizations
- Saves plots with appropriate filenames
- Works with the existing data structure

ACTUALLY MODIFY THE PLOT.PY FILE - don't just describe what to do.
"""
        
        coder.run(feedback_prompt)
        print("✅ Plot feedback applied, regenerating plots...")

        return_code, error_msg = run_plotting(folder_name)
        
        if return_code == 0:
            print("✅ Plots regenerated successfully!")
            return True
        else:
            print(f"❌ Plot regeneration failed: {error_msg}")
            return False
            
    except Exception as e:
        print(f"❌ Error applying plot feedback: {e}")
        return False

