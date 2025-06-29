# Sửa các import này
import sys
import os
# Thêm parent directory vào path để có thể import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from prompt.trainning_prompt import coding_instruct_prompt
from dotenv import load_dotenv
from aider.coders import Coder
import subprocess
from aider.models import Model
from aider.io import InputOutput
from base_code.processing_data import extract_python_code  
import pandas as pd
import google.generativeai as genai
import json
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY")) 
model = genai.GenerativeModel("gemini-2.5-flash")

os.environ['PYTHONIOENCODING'] = 'utf-8'
data = "sensor_data.csv"

def run_code_with_error_fix(file_path, max_attempts=3):

    for attempt in range(max_attempts):
        print(f"\n🔄 Attempt {attempt + 1}/{max_attempts}")
        
        result = subprocess.run(['python', file_path], 
                              capture_output=True, text=True, encoding='utf-8')
        
        if result.returncode == 0:
            print("✅ Code executed successfully!")
            if result.stdout:
                print(f"📄 Output:\n{result.stdout}")
            return True
        else:
            error_msg = result.stderr
            print(f"❌ Code execution failed")
            print(f"🔴 Error: {error_msg}")
            
            if attempt < max_attempts - 1:
                print("🔧 Using Aider to fix the error...")
                fixed = fix_with_aider(file_path, error_msg)
                if not fixed:
                    print("⚠️ Aider could not fix the error")
                    break
            else:
                print("💥 Max attempts reached")
                return False
    
    return False

def fix_with_aider(file_path, error_msg):
    try:
        fnames = [file_path]
        aider_model = Model("gemini/gemini-2.5-flash")
        io = InputOutput(yes=True, chat_history_file=None)
        
        coder = Coder.create(
            main_model=aider_model,
            fnames=fnames,
            io=io,
            use_git=False,
            edit_format="diff"
        )
        
        fix_prompt = f"""
The Python script is failing with this error:

ERROR:
{error_msg}

Please analyze the error and fix the code. Common fixes needed:
- Fix import issues (e.g., np.warnings -> warnings)
- Fix file path issues
- Fix syntax errors
- Fix missing imports
- Handle missing columns or data issues

Please fix the code so it runs successfully.
"""
        print("🔧 Sending fix request to Aider...")
        reply = coder.run(fix_prompt)
        print(f"🛠️ Aider response: {reply}")
        
        return True
        
    except Exception as e:
        print(f"❌ Aider fix failed: {str(e)}")
        return False

def generate_instruct_prompt(data_info_path, data_info, idea):
    os.makedirs("result", exist_ok=True)

    base_prompt = coding_instruct_prompt(data_info_path, data_info, idea)
    print("🔄 Generated instruction prompt")
    
    response = model.generate_content(base_prompt)
    
    if response and response.text:
        instruct_prompt = response.text
        with open("result/instruct_prompt.txt", "w", encoding="utf-8") as f:
            f.write(instruct_prompt)
            
        print("📝 Generated instruct prompt for code generation")
        
        code_response = model.generate_content(instruct_prompt)
        
        if code_response and code_response.text:
            code = extract_python_code(code_response.text)
            if code:
                print("🔍 Extracted Python code")
                file_path = 'result/experiment.py'
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(code)
                print(f"💾 Code saved to {file_path}")
                
                success = run_code_with_error_fix(file_path, max_attempts=3)
                
                if success:
                    print("🎉 Data analysis completed!")
                    
                    if os.path.exists("result/results.json"):
                        print("📊 results.json file created")
                        with open("result/results.json", "r", encoding="utf-8") as f:
                            results = json.load(f)
                        print("📈 Results preview:", str(results)[:200] + "...")
                    return True
                else:
                    print("💥 Code execution failed after all attempts")
                    return False
            else:
                print("❌ No Python code found in response")
                return False
        else:
            print("❌ No code response from Gemini")
            return False
    else:
        print("❌ No response from Gemini")
        return False

def main():
    data_info = pd.read_csv(data)
    with open("idea.json","r", encoding="utf-8") as f:
        idea = json.load(f)
    success = generate_instruct_prompt(data, data_info, idea)
    if success:
        print("✅ Analysis completed successfully!")
    else:
        print("❌ Analysis failed.")

if __name__ == "__main__":
    main()