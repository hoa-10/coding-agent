import argparse
import json
import os
import os.path
import re
import shutil
import subprocess
import sys
import traceback
from rich import print
from prompt import AGGREGATOR_SYSTEM_MSG
from llm import create_client, get_response_from_llm

MAX_FIGURES = 12

def build_aggregator_prompt(combined_summaries_str, idea_text):
    return f"""
We have three JSON summaries of scientific experiments: baseline, research, ablation.
They may contain lists of figure descriptions, code to generate the figures, and paths to the .npy files containing the numerical results.
Our goal is to produce final, publishable figures.

--- RESEARCH IDEA ---
```
{idea_text}
```

IMPORTANT:
- The aggregator script must load existing .npy experiment data from the "exp_results_npy_files" fields (ONLY using full and exact file paths in the summary JSONs) for thorough plotting.
- It should call os.makedirs("figures", exist_ok=True) before saving any plots.
- Aim for a balance of empirical results, ablations, and diverse, informative visuals in 'figures/' that comprehensively showcase the finalized research outcomes.
- If you need .npy paths from the summary, only copy those paths directly (rather than copying and parsing the entire summary).

Your generated Python script must:
1) Load or refer to relevant data and .npy files from these summaries. Use the full and exact file paths in the summary JSONs.
2) Synthesize or directly create final, scientifically meaningful plots for a final research paper (comprehensive and complete), referencing the original code if needed to see how the data was generated.
3) Carefully combine or replicate relevant existing plotting code to produce these final aggregated plots in 'figures/' only, since only those are used in the final paper.
4) Do not hallucinate data. Data must either be loaded from .npy files or copied from the JSON summaries.
5) The aggregator script must be fully self-contained, and place the final plots in 'figures/'.
6) This aggregator script should produce a comprehensive and final set of scientific plots for the final paper, reflecting all major findings from the experiment data.
7) Make sure that every plot is unique and not duplicated from the original plots. Delete any duplicate plots if necessary.
8) Each figure can have up to 3 subplots using fig, ax = plt.subplots(1, 3).
9) Use a font size larger than the default for plot labels and titles to ensure they are readable in the final PDF paper.


Below are the summaries in JSON:

{combined_summaries_str}

Respond with a Python script in triple backticks.
"""

def extract_code_snippet(text: str) -> str:
    pattern = r"```(?:python)?(.*?)```"
    matches = re.findall(pattern, text, flags=re.DOTALL)
    return matches[0].strip() if matches else text.strip()

def run_aggregator_scripts(
    aggregator_code, aggregator_script_path, base_model, script_name):
    if not aggregator_code.strip():
        print("No aggregator code was provided. Skipping aggregator script run.")
        return ""
    with open(aggregator_script_path, "w") as f:
        f.write(aggregator_code)

    print(
        f"Aggregator script written to '{aggregator_script_path}'. Attempting to run it..."
    )
    aggregator_out = ""
    try:
        result = subprocess.run(
            [sys.executable, script_name],
            cwd = base_model,
            check = True,
            stdout = subprocess.PIPE,
            stderr = subprocess.PIPE,
            text=True,
        )
        aggregator_out = result.stdout + "\n" + result.stderr
        print("Aggregator script ran successfully.")
    except subprocess.CalledProcessError as e:
        aggregator_out = (e.stdout or "") + "\n" + (e.stderr or "")
        print("Error: aggregator script returned a non-zero exit code.")
        print(e)
    except Exception as e:
        aggregator_out = str(e)
        print("Error while running aggregator script.")
        print(e)

    return aggregator_out
    

def aggregate_plot(
        base_fodel: str, model: str= "o1-mini", n_reflections: int = 5
) -> None:
    file_name = "auto_plot_aggregator.py"
    aggregator_script_path = os.path.join(base_fodel, file_name)
    figure_dir = os.path.join(base_fodel, "figures")

    if os.path.exists(aggregator_script_path):
        os.remove(aggregator_script_path)
    if os.path.exists(figure_dir):
        shutil.rmtree(figure_dir)
        print(f"Removed existing figure directory")

    idea_text = load_idea_text(base_fodel)
    exp_summaries = load_exp_summaries(base_fodel)
    filtered_summaries_for_plot_agg = filter_experiment_summaries(
        exp_summaries, step_name="plot_aggregation"
    )
    combined_summaries_str = json.dump(filtered_summaries_for_plot_agg, indent=2)

    aggregator_prompt = build_aggregator_prompt(combined_summaries_str, idea_text)

    client, model_name = create_client(model)
    response , msg_history = None, []
    try:
        response, msg_history = get_response_from_llm(
            prompt = aggregator_prompt,
            client=client,
            model = model_name,
            system_message= AGGREGATOR_SYSTEM_MSG,
            print_debug= False,
            msg_history= msg_history
        )
    except Exception:
        traceback.print_exc()
        print("Failed to get aggregator script from LLM.")
        return
    aggregator_code = extract_code_snippet(response)
    if not aggregator_code.strip():
        print(
            "No Python code block was found in LLM response. Full response:\n", response
        )
        return
    aggregator_out = run_aggregator_scripts(
        aggregator_code, aggregator_script_path, base_fodel, file_name
    )
    for i in range(n_reflections):
        # Check number of figures
        figure_count = 0
        if os.path.exists(figure_dir):
            figure_count = len(
                [
                    f
                    for f in os.listdir(figure_dir)
                    if os.path.isfile(os.path.join(figure_dir, f))
                ]
            )
        print(f"[{i + 1} / {n_reflections}]: Number of figures: {figure_count}")
        reflection_prompt = f"""We have run your aggregator script and it produced {figure_count} figure(s). The script's output is:
```
{aggregator_out}
```

Please criticize the current script for any flaws including but not limited to:
- Are these enough plots for a final paper submission? Don't create more than {MAX_FIGURES} plots.
- Have you made sure to both use key numbers and generate more detailed plots from .npy files?
- Does the figure title and legend have informative and descriptive names? These plots are the final versions, ensure there are no comments or other notes.
- Can you aggregate multiple plots into one figure if suitable?
- Do the labels have underscores? If so, replace them with spaces.
- Make sure that every plot is unique and not duplicated from the original plots.

If you believe you are done, simply say: "I am done". Otherwise, please provide an updated aggregator script in triple backticks."""
        print("[green]Reflection prompt:[/green] ", reflection_prompt)
        try:
            reflection_response, msg_history = get_response_from_llm(
                prompt=reflection_prompt,
                client=client,
                model=model_name,
                system_message=AGGREGATOR_SYSTEM_MSG,
                print_debug=False,
                msg_history=msg_history,
            )

        except Exception:
            traceback.print_exc()
            print("Failed to get reflection from LLM.")
            return

        # Early-exit check
        if figure_count > 0 and "I am done" in reflection_response:
            print("LLM indicated it is done with reflections. Exiting reflection loop.")
            break

        aggregator_new_code = extract_code_snippet(reflection_response)

        # If new code is provided and differs, run again
        if (
            aggregator_new_code.strip()
            and aggregator_new_code.strip() != aggregator_code.strip()
        ):
            aggregator_code = aggregator_new_code
            aggregator_out = run_aggregator_scripts(
                aggregator_code, aggregator_script_path, base_fodel, file_name
            )
        else:
            print(
                f"No new aggregator script was provided or it was identical. Reflection step {i+1} complete."
            )

def main():
    # Manually define the arguments
    folder = "experiment"
    model = "o1-mini"  # Or any other model you'd like to use
    reflections = 5  # Or any other number of reflections

    # Call the function with these arguments
    aggregate_plot(base_folder=folder, model=model, n_reflections=reflections)

if __name__ == "__main__":
    main()
