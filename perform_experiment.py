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

MAX_ITERS = 4
MAX_RUNS = 5
MAX_STDERR_OUTPUT = 1500

coder_prompt = """Your goal is to implement the following idea: {title}.
The proposed experiment is as follows: {idea}.
You are given a total of up to {max_runs} runs to complete the necessary experiments. You do not need to use all {max_runs}.

IMPORTANT: Your experiment.py MUST save results to final_info.json in the output directory. 
First, plan the list of experiments you would like to run. For example, if you are sweeping over a specific hyperparameter, plan each value you would like to test for each run.
After each run, only save the results of the current run to the final_info.json file in the output directory. Do not aggregate or overwrite results from other runs.
Note that we already provide the vanilla baseline results, so you do not need to re-run it.
Ensure code
For reference, the baseline results are as follows:
{baseline_results}
> - When running the command: python experiment.py --out_dir=run_i, only perform the experiment for run number i (that is, index i-1 in your list of parameter settings).

After you complete each change, we will run the command `python experiment.py --out_dir=run_i' where i is the run number and evaluate the results.
YOUR PROPOSED CHANGE MUST USE THIS COMMAND FORMAT, DO NOT ADD ADDITIONAL COMMAND LINE ARGS.

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

            next_prompt = f"""Run {run_num} completed. Here are the results:
{results}

Decide if you need to re-plan your experiments given the result (you often will not need to).

Someone else will be using `notes.txt` to perform a writeup on this in the future.
Please include *all* relevant information for the writeup on Run {run_num}, including an experiment description and the run number. Be as verbose as necessary.
> - When running the command: python experiment.py --out_dir=run_i, only perform the experiment for run number i (that is, index i-1 in your list of parameter settings).
Then, implement the next thing on your list by MODIFYING the experiment.py file.
We will then run the command `python experiment.py --out_dir=run_{run_num + 1}'.
YOUR PROPOSED CHANGE MUST USE THIS COMMAND FORMAT, DO NOT ADD ADDITIONAL COMMAND LINE ARGS.
If you are finished with experiments, respond with 'ALL_COMPLETED'.
After each run, only save the results of the current run to the final_info.json file in the output directory. Do not aggregate or overwrite results from other runs.
REMEMBER: Your experiment.py MUST save results to final_info.json in the output directory.
IMPORTANT: YOU MUST ACTUALLY MODIFY THE EXPERIMENT.PY FILE, NOT JUST DESCRIBE THE CHANGES."""
            
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
def perform_experiments(idea, folder_name, coder, baseline_results) -> bool:
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
            # Lưu hash và nội dung trước khi AI coder chạy
            old_hash = file_hash(exp_file)
            old_content = read_file_content(exp_file)
            
            print(f"Getting coder output for run {run}, iteration {current_iter}")
            print(f"File hash before coder: {old_hash}")
            
            # Chạy AI coder
            coder_out = coder.run(next_prompt)
            print(f"Coder output: {coder_out}")
            
            # Đợi một chút để đảm bảo file được ghi
            time.sleep(1)
            
            # Kiểm tra hash và nội dung sau khi AI coder chạy
            new_hash = file_hash(exp_file)
            new_content = read_file_content(exp_file)
            
            print(f"File hash after coder: {new_hash}")
            
            if old_hash == new_hash or old_content == new_content:
                print("WARNING: experiment.py was not changed by AI coder!")
                print("This might indicate the AI didn't actually modify the file.")
                
                # Thử yêu cầu AI coder một lần nữa với prompt cụ thể hơn
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
                
                # Thử thay đổi edit format nếu diff không hoạt động
                try:
                    # Lưu edit format hiện tại
                    current_format = getattr(coder, 'edit_format', 'diff')
                    
                    # Nếu đang dùng diff và không hiệu quả, thử whole
                    if current_format == 'diff':
                        print("Switching to 'whole' edit format for major changes...")
                        coder.edit_format = 'whole'
                    
                    coder_out = coder.run(force_edit_prompt)
                    
                except Exception as format_error:
                    print(f"Error with format change: {format_error}")
                    coder_out = coder.run(force_edit_prompt)
                
                # Kiểm tra lại sau khi force edit
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

    # Plotting phase
    print("Starting plotting phase")
    current_iter = 0
    next_prompt = """
Great job! Please modify `plot.py` to generate the most relevant plots for the final writeup. 
In particular, be sure to fill in the "labels" dictionary with the correct names for each run that you want to plot.
Only the runs in the `labels` dictionary will be plotted, so make sure to include all relevant runs.
We will be running the command `python plot.py` to generate the plots.
"""
    while current_iter < MAX_ITERS:
        try:
            coder.run(next_prompt)
            return_code, next_prompt = run_plotting(folder_name)
            if return_code == 0:
                break
            current_iter += 1
        except Exception as e:
            print(f"Plotting phase error: {e}")
            current_iter += 1
            
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