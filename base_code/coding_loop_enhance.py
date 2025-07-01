import sys
import os

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

def get_human_feedback(file_path, code_content):
    """Get human feedback on the generated code"""
    print("\n" + "="*60)
    print("🔍 HUMAN REVIEW REQUIRED")
    print("="*60)
    print(f"📁 File: {file_path}")
    print("\n📝 Generated Code:")
    print("-" * 40)
    print(code_content)
    print("-" * 40)
    
    while True:
        print("\n🤔 Please review the code above:")
        print("1. ✅ Approve - Code looks good")
        print("2. 📝 Provide feedback - Code needs changes")
        print("3. 👀 View in editor - Open file in VS Code")
        print("4. ❌ Reject - Start over")
        
        choice = input("\n👤 Your choice (1/2/3/4): ").strip()
        
        if choice == "1":
            return "approved", ""
        elif choice == "2":
            feedback = input("\n💬 Please provide your feedback/suggestions:\n👤 ")
            if feedback.strip():
                return "feedback", feedback
            else:
                print("⚠️ Please provide some feedback!")
        elif choice == "3":
            try:
                subprocess.run(["code", file_path], check=True)
                print("📂 File opened in VS Code. Please review and come back.")
                input("⏸️ Press Enter when you're ready to continue...")
            except:
                print("❌ Could not open VS Code. Please check the file manually.")
        elif choice == "4":
            return "rejected", ""
        else:
            print("❌ Invalid choice. Please select 1, 2, 3, or 4.")

def apply_human_feedback(file_path, feedback, max_attempts=2):
    """Use Aider to apply human feedback to the code"""
    print(f"\n🔧 Applying human feedback with Aider...")
    
    for attempt in range(max_attempts):
        try:
            fnames = [file_path]
            aider_model = Model("gpt-4o-mini")  # Using more stable model
            io = InputOutput(yes=True, chat_history_file=None)
            
            coder = Coder.create(
                main_model=aider_model,
                fnames=fnames,
                io=io,
                use_git=False,
                edit_format="diff"
            )
            
            feedback_prompt = f"""
Based on human feedback, please modify the code according to these requirements:

HUMAN FEEDBACK:
{feedback}

Please:
1. Analyze the current code
2. Understand the human feedback
3. Make the necessary changes to improve the code
4. Ensure the code still functions correctly
5. Keep the same overall structure unless feedback suggests otherwise

Make the changes thoughtfully and explain what you're doing.
"""
            
            print(f"🔄 Attempt {attempt + 1}/{max_attempts} - Sending feedback to Aider...")
            reply = coder.run(feedback_prompt)
            print(f"🛠️ Aider response: {reply}")
            
            # Test the modified code
            print("🧪 Testing modified code...")
            result = subprocess.run(['python', file_path], 
                                  capture_output=True, text=True, encoding='utf-8')
            
            if result.returncode == 0:
                print("✅ Modified code runs successfully!")
                return True
            else:
                print(f"❌ Modified code has errors: {result.stderr}")
                if attempt < max_attempts - 1:
                    print("🔄 Trying again...")
                
        except Exception as e:
            print(f"❌ Aider feedback application failed: {str(e)}")
            
    return False

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
        aider_model = Model("gpt-4o-mini")  # Using more stable model
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
                
                # NEW: Human feedback loop
                max_feedback_rounds = 3
                for round_num in range(max_feedback_rounds):
                    print(f"\n🔄 Human Review Round {round_num + 1}/{max_feedback_rounds}")
                    
                    # Get human feedback
                    feedback_status, feedback_text = get_human_feedback(file_path, code)
                    
                    if feedback_status == "approved":
                        print("✅ Code approved by human reviewer!")
                        break
                    elif feedback_status == "rejected":
                        print("❌ Code rejected. Regenerating...")
                        # Regenerate code with additional context
                        retry_prompt = instruct_prompt + f"\n\nPrevious attempt was rejected. Please generate a completely different approach."
                        code_response = model.generate_content(retry_prompt)
                        if code_response and code_response.text:
                            code = extract_python_code(code_response.text)
                            with open(file_path, 'w', encoding='utf-8') as f:
                                f.write(code)
                        continue
                    elif feedback_status == "feedback":
                        print("📝 Applying human feedback...")
                        success = apply_human_feedback(file_path, feedback_text)
                        if success:
                            # Read the updated code
                            with open(file_path, 'r', encoding='utf-8') as f:
                                code = f.read()
                            print("✅ Code updated based on feedback!")
                        else:
                            print("❌ Failed to apply feedback")
                            continue
                else:
                    print("⚠️ Maximum feedback rounds reached")
                
                # Run the final code
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