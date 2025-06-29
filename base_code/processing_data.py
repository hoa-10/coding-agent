import json
import os
import os.path as osp
import shutil
import subprocess
import sys
from subprocess import TimeoutExpired
import hashlib
import time
import re
from prompt.analyze_data import analyze_dataset_prompt
from prompt.trainning_prompt import coding_instruct_prompt
import google.generativeai as genai
from dotenv import load_dotenv
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY")) 
model = genai.GenerativeModel("gemini-2.5-flash")

os.environ['PYTHONIOENCODING'] = 'utf-8'
data = "sensor_data.csv"

def extract_python_code(prompts):
    match = re.search(r"```(?:python)?\s*(.*?)```", prompts, re.DOTALL)
    code = match.group(1).strip() if match else prompts.strip()
    return code

def save_and_run_code(code, filename="analysis/generated_analysis.py"):
    try:
        with open(filename, "w", encoding="utf-8") as f:
            f.write(code)
        print(f"🏃 Running {filename}...")
        result = subprocess.run(
            [sys.executable, filename],
            capture_output=True,
            text=True,
            encoding='utf-8',
            timeout=120
        )
        
        if result.returncode == 0:
            print("✅ Code executed successfully!")
            if result.stdout:
                print("📊 Output:")
                print(result.stdout)
            return True
        else:
            print("❌ Code execution failed:")
            print("Error:", result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print("⏰ Code execution timeout")
        return False
    except Exception as e:
        print(f"❌ Error running code: {e}")
        return False

def generate_and_run_analysis():
    try:
        prompt = analyze_dataset_prompt(data)
        response = model.generate_content(prompt)
        
        if response and response.text:
            code = extract_python_code(response.text)
            
            if code:
                success = save_and_run_code(code)
                
                if success:
                    print("🎉 Data analysis completed!")
                    
                    if os.path.exists("analysis/results.json"):
                        print("📊 results.json file created")
                        with open("analysis/results.json", "r", encoding="utf-8") as f:
                            results = json.load(f)
                        print("📈 Results preview:", str(results)[:200] + "...")
                    
                    return True
                else:
                    print("💥 Code execution failed")
                    return False
            else:
                print("❌ No Python code found in response")
                print("📄 Full response:")
                print(response.text)
                return False
        else:
            print("❌ No response from Gemini")
            return False
            
    except Exception as e:
        print(f"❌ Error calling Gemini: {e}")
        return False

def auto_analyze_with_gemini():
    print("🔍 Starting automatic data analysis with Gemini...")

    os.makedirs("analysis", exist_ok=True)
    os.makedirs("analysis/figures", exist_ok=True)
    
    success = generate_and_run_analysis()
    
    if success:
        print("🎉 Analysis completed successfully!")
    else:
        print("💥 Analysis failed!")
    
    return success

def auto_analyze_with_retry(max_retries=3):
    for attempt in range(max_retries):
        print(f"🔄 Attempt {attempt + 1}/{max_retries}")
        
        success = auto_analyze_with_gemini()
        
        if success:
            return True
        else:
            if attempt < max_retries - 1:
                print("⏳ Waiting 5 seconds before retry...")
                time.sleep(5)
    
    print("💥 Failed after all attempts")
    return False

if __name__ == "__main__":
    auto_analyze_with_retry()