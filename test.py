import gradio as gr
import json
import os
import sys
import threading
import time
from base_code.coding_loop_enhance import run_code as run_code_func
from datetime import datetime
import shutil
import subprocess
import os.path as osp
import numpy as np
from typing import Dict, List, Tuple, Optional
import base64
from PIL import Image
from base_code.coding_loop_enhance import step2_generate_code as generate_code_from_idea

# Import các module cần thiết
from base_code.processing_data import auto_analyze_with_retry, extract_python_code
from base_code.coding_loop_enhance import apply_user_feedback, step2_generate_code
from prompt.trainning_prompt import coding_instruct_prompt
from perform_experiment import perform_experiments
from aider.coders import Coder
from aider.io import InputOutput
from aider.models import Model
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
gemini_model = genai.GenerativeModel("gemini-2.5-flash")

data = "pect_ndt_full_dataset.npz"
idea = json.load(open("idea.json", encoding="utf-8"))

class AIResearchAssistant:
    def __init__(self):
        self.data_file = r'C:\Users\user\Desktop\coding-agent\pect_ndt_full_dataset.npz'
        self.results_dir = "coding-agent"
        self.current_idea = None
        self.current_code = ""
        self.chat_history = []
        self.plot_chat_history = []
        self.analysis_complete = False
        self.code_generation_complete = False
        self.experiment_complete = False
        self.experiment_running = False
        self.logs = []
        self.current_experiment_folder = None
        self.experiment_coder = None
        
        # Tạo các thư mục cần thiết
        self.setup_directories()
        
    def setup_directories(self):
        """Tạo các thư mục cần thiết"""
        directories = [
            "result",
            "analysis", 
            "analysis/figures",
            self.results_dir
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            self.log(f"📁 Created/checked directory: {directory}")
    
    def log(self, message):
        """Thêm log và in ra console"""
        timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
        log_message = f"{timestamp} {message}"
        self.logs.append(log_message)
        print(log_message)
        return log_message
    
    def get_logs(self):
        """Lấy logs gần đây nhất"""
        return "\n".join(self.logs[-50:])
    
    def load_idea(self, idea_json):
        """Load idea từ JSON"""
        try:
            if not idea_json:
                self.log("❌ No idea JSON provided")
                return False, "Error: No idea JSON provided"
            
            if isinstance(idea_json, str):
                self.current_idea = json.loads(idea_json)
            else:
                self.current_idea = idea_json
                
            self.log(f"✅ Loaded idea: {self.current_idea['Name']}")
            return True, f"Loaded idea: {self.current_idea['Name']}"
        except Exception as e:
            self.log(f"❌ Error loading idea: {str(e)}")
            return False, f"Error: {str(e)}"
    
    def add_message(self, role, content):
        """Thêm tin nhắn vào lịch sử chat"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.chat_history.append({
            "role": role,
            "content": content,
            "timestamp": timestamp
        })
    
    def add_plot_message(self, role, content):
        """Thêm tin nhắn vào lịch sử plot chat"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.plot_chat_history.append({
            "role": role,
            "content": content,
            "timestamp": timestamp
        })
    
    def get_chat_history(self):
        """Lấy lịch sử chat định dạng cho Gradio"""
        formatted = []
        for msg in self.chat_history:
            formatted.append({
                "role": msg["role"],
                "content": msg["content"]
            })
        return formatted
    
    def get_plot_chat_history(self):
        """Lấy lịch sử plot chat định dạng cho Gradio"""
        formatted = []
        for msg in self.plot_chat_history:
            formatted.append({
                "role": msg["role"],
                "content": msg["content"]
            })
        return formatted
    
    def step1_auto_analyze(self):

        self.log("🔍 Step 1: Starting auto analysis...")
        
        try:
            success = auto_analyze_with_retry()
            
            if success:
                self.analysis_complete = True
                self.log("✅ Step 1 completed: Auto analysis successful!")
                return True, "Auto analysis completed successfully"
            else:
                self.log("❌ Step 1 failed: Auto analysis failed")
                return False, "Auto analysis failed"
        except Exception as e:
            self.log(f"❌ Error in auto analysis: {str(e)}")
            return False, f"Error: {str(e)}"
    # Thêm method này vào class AIResearchAssistant (sau method step2_generate_code):
    # Sửa method run_code trong class AIResearchAssistant:
    def run_code(self):
        """Chạy code đã generate"""
        if not self.current_code:
            self.log("⚠️ No code to run")
            return False, "No code to run"
        
        self.log("🚀 Running generated code...")
        
        try:
            file_path = 'result/experiment.py'
            
            # Đảm bảo file tồn tại
            if not os.path.exists(file_path):
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(self.current_code)
            
            
            success = run_code_func(file_path)
            
            if success:
                self.log("🎉 Code ran successfully!")
                self.code_generation_complete = True
                
                # Kiểm tra results
                if os.path.exists("result/results.json"):
                    self.log("📊 Results saved to results.json")
                    return True, "Code ran successfully and results generated"
                else:
                    self.log("⚠️ Code ran but no results.json found")
                    return True, "Code ran successfully but no results found"
            else:
                self.log("❌ Code execution failed")
                return False, "Code execution failed"
        except Exception as e:
            self.log(f"❌ Error running code: {str(e)}")
            return False, f"Error: {str(e)}"
    def apply_feedback(self, feedback):

        if not self.current_code:
            self.log("⚠️ No code to apply feedback to")
            return False, "No code to apply feedback to", self.current_code
        
        self.log(f"📝 Applying user feedback: {feedback}")
        
        if feedback.lower() == "ok":
            self.log("✅ User approved the code. Ready to run.")
            self.code_generation_complete = True
            return True, "Code approved. Ready to run experiments.", self.current_code
        
        try:
            file_path = 'result/experiment.py'
            
            feedback_applied = apply_user_feedback(file_path, feedback)
            
            if feedback_applied:
                with open(file_path, 'r', encoding='utf-8') as f:
                    updated_code = f.read()
                
                self.current_code = updated_code
                self.log("✅ Code updated based on feedback!")
                return True, "Code updated successfully based on feedback", updated_code
            else:
                self.log("❌ Failed to apply user feedback")
                return False, "Failed to apply feedback", self.current_code
        except Exception as e:
            self.log(f"❌ Error applying feedback: {str(e)}")
            return False, f"Error: {str(e)}", self.current_code
    def step2_generate_code(self):
        if not self.analysis_complete:
            self.log("⚠️ Please complete Step 1 first")
            return False, "Please complete Step 1 first", None
        
        if not self.current_idea:
            self.log("⚠️ Please load an idea first")
            return False, "Please load an idea first", None
        
        self.log("🔧 Step 2: Generating code from idea...")
        
        try:
            success, message, code = generate_code_from_idea(
                data_file=self.data_file,
                idea_dict=self.current_idea,
                output_dir="result"
            )
            
            if success and code:
                self.current_code = code
                self.log("✅ Code generated successfully!")
                return True, message, code
            else:
                self.log(f"❌ Code generation failed: {message}")
                return False, message, None
                
        except Exception as e:
            self.log(f"❌ Error in step2_generate_code: {str(e)}")
            return False, f"Error: {str(e)}", None
    
    def step3_run_experiment(self, model="gemini/gemini-2.5-flash"):
        """Bước 3: Chạy thí nghiệm"""
        if not self.code_generation_complete:
            self.log("⚠️ Please complete Step 2 first")
            return False, "Please complete Step 2 first"
        
        
        if self.experiment_running:
            self.log("⚠️ Experiment is already running")
            return False, "Experiment is already running"
        
        self.log(f"🧪 Step 3: Running experiment with model: {model}")
        self.experiment_running = True
        
        def run_experiment_thread():
            try:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                idea_name = f"{timestamp}_{self.current_idea['Name']}"
                folder_name = os.path.join(self.results_dir, idea_name)
                
                self.log(f"📁 Creating experiment folder: {folder_name}")
                
                if os.path.exists(folder_name):
                    shutil.rmtree(folder_name)
                
                shutil.copytree("result", folder_name, dirs_exist_ok=True)
                
                with open(os.path.join("result", "results.json"), "r") as f:
                    baseline_results = json.load(f)
                
                exp_file = os.path.join(folder_name, "experiment.py")
                vis_file = os.path.join(folder_name, "plot.py")
                notes = os.path.join(folder_name, "notes.txt")
                
                with open(notes, "w", encoding='utf-8') as f:
                    f.write(f"# Title: {self.current_idea['Title']}\n")
                    f.write(f"# Experiment description: {self.current_idea['Experiment']}\n")
                    f.write(f"## Run 0: Baseline\n")
                    f.write(f"Results: {baseline_results}\n")
                    f.write(f"Description: Baseline results.\n")
                
                fnames = [exp_file, vis_file, notes]
                io = InputOutput(yes=True, chat_history_file=f"{folder_name}/{idea_name}_aider.txt")
                
                main_model = Model(model)
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
                
                self.log("🔬 Starting experiments with AI assistant...")
                
                success = perform_experiments(
                    self.current_idea, 
                    folder_name, 
                    coder, 
                    baseline_results,
                    gui_mode=True
                )
                
                if success:
                    self.experiment_complete = True
                    self.current_experiment_folder = folder_name
                    self.experiment_coder = coder
                    # Reset notification flag for new experiments
                    if hasattr(self, '_plots_notified'):
                        delattr(self, '_plots_notified')
                    self.log(f"🎉 Experiments completed successfully!")
                    self.log(f"📊 Plots generated in {folder_name}")
                    self.log("👀 Please review the plots in the Plots tab")
                    
                    self.add_plot_message("assistant", "🎉 Experiments completed! I've generated plots for your analysis. Please review them and provide feedback if needed, or type 'ok' if you're satisfied with the results.")
                else:
                    self.log("❌ Experiments failed")
                
                self.experiment_running = False
            except Exception as e:
                self.log(f"❌ Error in experiment: {str(e)}")
                self.experiment_running = False
        
        thread = threading.Thread(target=run_experiment_thread)
        thread.daemon = True
        thread.start()
        
        return True, "Experiment started. This may take some time..."
    
    def get_plot_images(self):
        # Auto-detect latest experiment folder if not set
        if not self.current_experiment_folder:
            self.current_experiment_folder = self.get_latest_experiment_folder()
            if not self.current_experiment_folder:
                return []
        
        plot_files = []
        plot_extensions = ['.png', '.jpg', '.jpeg', '.svg', '.pdf', '.bmp', '.tiff']
        
        try:
            # Recursively search for image files in experiment folder and all subfolders
            for root, dirs, files in os.walk(self.current_experiment_folder):
                for file in files:
                    if any(file.lower().endswith(ext) for ext in plot_extensions):
                        full_path = os.path.join(root, file)
                        plot_files.append(full_path)
                        self.log(f"📊 Found plot: {file} in {os.path.relpath(root, self.current_experiment_folder)}")
            
            # Sort files by modification time (newest first) for better display order
            plot_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            
        except Exception as e:
            self.log(f"❌ Error getting plot files: {e}")
        
        self.log(f"📊 Total plots found: {len(plot_files)}")
        return plot_files
    
    def get_latest_experiment_folder(self):
        """Auto-detect the latest experiment folder"""
        try:
            if not os.path.exists(self.results_dir):
                return None
            
            # Find folders matching the experiment pattern
            folders = []
            for item in os.listdir(self.results_dir):
                folder_path = os.path.join(self.results_dir, item)
                if os.path.isdir(folder_path):
                    # Check if it looks like an experiment folder (contains experiment.py)
                    if os.path.exists(os.path.join(folder_path, "experiment.py")):
                        folders.append((folder_path, os.path.getmtime(folder_path)))
            
            if folders:
                # Return the most recently modified folder
                latest_folder = max(folders, key=lambda x: x[1])[0]
                self.log(f"📁 Auto-detected latest experiment folder: {os.path.basename(latest_folder)}")
                return latest_folder
        except Exception as e:
            self.log(f"❌ Error detecting latest experiment folder: {e}")
        
        return None
        
    def apply_plot_feedbacks(self, feedback):
        """Áp dụng feedback cho plots"""
        if not self.current_experiment_folder or not self.experiment_coder:
            return False, "No experiment plots available"
        
        if feedback.lower() == "ok":
            self.log("✅ Plots approved by user!")
            return True, "Plots approved! Analysis complete."
        
        try:
            self.log(f"📝 Applying plot feedback: {feedback}")
            
            from perform_experiment import apply_plot_feedback
            success = apply_plot_feedback(
                self.current_experiment_folder,
                self.experiment_coder,
                feedback
            )
            
            if success:
                self.log("✅ Plot feedback applied and plots regenerated")
                return True, "Plots updated based on feedback. Please review the new plots."
            else:
                self.log("❌ Failed to apply plot feedback")
                return False, "Failed to apply plot feedback"
        except Exception as e:
            self.log(f"❌ Error applying plot feedback: {str(e)}")
            return False, f"Error: {str(e)}"

def create_interface():
    assistant = AIResearchAssistant()
    
    def update_interface():
        status = "⏳ Analysis in progress..." if not assistant.analysis_complete else "✅ Analysis complete"
        status += " | "
        status += "⏳ Code generation in progress..." if not assistant.code_generation_complete else "✅ Code generation complete"
        status += " | "
        status += "⏳ Experiment in progress..." if assistant.experiment_running else ("✅ Experiment complete" if assistant.experiment_complete else "⏳ Waiting for experiment")
        
        code_status = ""
        if assistant.current_code:
            code_status = "✅ Code generated. Provide feedback or type 'ok' to approve."
        else:
            code_status = "⏳ Waiting for code generation..."
        
        return status, code_status, assistant.get_chat_history(), assistant.get_plot_chat_history(), assistant.get_logs()
    
    # INTERFACE DESIGN
    with gr.Blocks(title="AI Research Assistant", theme=gr.themes.Soft(), css="""
        .plot-gallery { border: 2px solid #e0e0e0; border-radius: 8px; padding: 10px; }
        .chat-container { height: 400px; overflow-y: auto; }
        .status-box { background: #f0f8ff; padding: 10px; border-radius: 8px; margin: 10px 0; }
    """) as demo:
        
        gr.Markdown("# 🧪 AI Research Assistant")
        gr.Markdown("An interactive platform for AI-driven research and experimentation with advanced plot visualization")
        
        # Status Display
        status_text = gr.Textbox(
            label="🔄 Current Status",
            value="Ready to start",
            interactive=False,
            elem_classes="status-box"
        )
        
        with gr.Tabs() as tabs:
            with gr.TabItem("📝 Research Setup"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 📋 Research Idea")
                        
                        # Example idea
                        example_idea = {
                            "Name": "advanced_ndt_analysis",
                            "Title": "Advanced NDT Signal Analysis with Deep Learning",
                            "Experiment": "Implement advanced signal processing and deep learning techniques for non-destructive testing data analysis, including feature extraction, anomaly detection, and predictive modeling with comprehensive visualization."
                        }
                        
                        idea_json = gr.Code(
                            label="Idea JSON",
                            value=json.dumps(example_idea, indent=2),
                            language="json",
                            interactive=True,
                            lines=10
                        )
                        
                        load_idea_btn = gr.Button("📥 Load Idea", variant="primary", size="lg")
                        
                        # Model selection
                        gr.Markdown("### 🤖 Model Selection")
                        model_dropdown = gr.Dropdown(
                            choices=[
                                "gemini/gemini-2.5-flash",
                                "gpt-3.5-turbo",
                                "gpt-4",
                                "claude-3-sonnet-20240229"
                            ],
                            value="gemini/gemini-2.5-flash",
                            label="AI Model"
                        )
                    
                    with gr.Column(scale=2):
                        gr.Markdown("### 🚀 Research Workflow")
                        
                        # Step buttons
                        with gr.Row():
                            step1_btn = gr.Button("🔍 Step 1: Auto Analysis", variant="secondary", size="lg")
                            step2_btn = gr.Button("🔧 Step 2: Generate Code", variant="secondary", size="lg")
                            step3_btn = gr.Button("🧪 Step 3: Run Experiment", variant="secondary", size="lg")
                        
                        # Progress indicator
                        gr.Markdown("### 📊 Progress")
                        progress_md = gr.Markdown("⏳ Step 1: Pending | ⏳ Step 2: Pending | ⏳ Step 3: Pending")
                        
                        # Logs display
                        logs_text = gr.Textbox(
                            label="📋 System Logs",
                            interactive=False,
                            lines=20,
                            max_lines=100,

                        )
            
            # TAB 2: Code Development
            with gr.TabItem("💻 Code Development"):
                with gr.Row():
                    with gr.Column(scale=2):
                        # Code display
                        code_display = gr.Code(
                            label="Generated Code",
                            language="python",
                            interactive=False,
                            lines=25,
                        )
                        
                        code_status = gr.Textbox(
                            label="Code Status",
                            value="⏳ Waiting for code generation...",
                            interactive=False
                        )
                        
                        # Code actions
                        with gr.Row():
                            run_code_btn = gr.Button("▶️ Run Code", variant="primary", size="lg")
                            regenerate_btn = gr.Button("🔄 Regenerate Code", variant="secondary", size="lg")
                    
                    with gr.Column(scale=1):
                        gr.Markdown("### 💬 Code Feedback & Interaction")
                        
                        chatbot = gr.Chatbot(
                            value=[],
                            label="Chat with AI",
                            height=400,
                            type="messages",
                        )
                        
                        # Feedback input
                        with gr.Row():
                            feedback_input = gr.Textbox(
                                placeholder="Provide feedback on the code or type 'ok' to approve",
                                label="Feedback",
                                lines=3,
                                scale=4
                            )
                            send_btn = gr.Button("📤 Send", variant="primary", scale=1)
            
            # TAB 3: Plots & Visualization
            with gr.TabItem("📊 Plots & Visualization"):
                gr.Markdown("### 🎨 Generated Plots and Analysis")
                
                with gr.Row():
                    with gr.Column(scale=2):
                        # Gallery để hiển thị plots
                        plot_gallery = gr.Gallery(
                            label="Generated Plots",
                            show_label=True,
                            elem_id="plot_gallery",
                            columns=2,
                            rows=2,
                            height=600,
                            object_fit="contain",
                        )
                        
                        # Plot control buttons
                        with gr.Row():
                            refresh_plots_btn = gr.Button("🔄 Refresh Plots", variant="secondary", size="lg")
                            download_plots_btn = gr.Button("📥 Download All Plots", variant="secondary", size="lg")
                        
                        # Plot info
                        plot_info = gr.Textbox(
                            label="Plot Information",
                            placeholder="Plot details will appear here...",
                            interactive=False,
                            lines=5
                        )
                        
                    with gr.Column(scale=1):
                        gr.Markdown("### 💬 Plot Feedback")
                        
                        # Chat cho plot feedback
                        plot_chatbot = gr.Chatbot(
                            value=[],
                            label="Plot Feedback Chat",
                            height=400,
                            type="messages",
                        )
                        
                        # Input cho plot feedback
                        with gr.Row():
                            plot_feedback_input = gr.Textbox(
                                placeholder="Provide feedback on plots or type 'ok' to approve",
                                label="Plot Feedback",
                                lines=3,
                                scale=4
                            )
                            send_plot_btn = gr.Button("📤 Send", variant="primary", scale=1)
                        
                        # Plot feedback suggestions
                        gr.Markdown("### 💡 Feedback Suggestions")
                        feedback_suggestions = gr.Markdown("""
                        **Common feedback examples:**
                        - "Make the title larger and more descriptive"
                        - "Change colors to be more colorblind-friendly"
                        - "Add error bars to show confidence intervals"
                        - "Use logarithmic scale for better visibility"
                        - "Add a legend with better positioning"
                        - "Include statistical significance markers"
                        - "Change plot size to be more readable"
                        """)
            
            # TAB 4: Results & Analysis
            with gr.TabItem("📈 Results & Analysis"):
                gr.Markdown("### 📊 Experiment Results and Final Analysis")
                
                with gr.Row():
                    with gr.Column(scale=2):
                        # Results display
                        results_text = gr.Code(
                            label="Experiment Results (JSON)",
                            language="json",
                            interactive=False,
                            lines=15,
                        )
                        
                        # Notes display
                        notes_text = gr.Textbox(
                            label="Analysis Notes",
                            interactive=False,
                            lines=10,
                        )
                        
                    with gr.Column(scale=1):
                        # Control buttons
                        with gr.Column():
                            refresh_results_btn = gr.Button("🔄 Refresh Results", variant="secondary", size="lg")
                            export_results_btn = gr.Button("📤 Export Results", variant="primary", size="lg")
                        
                        # Summary stats
                        summary_stats = gr.Textbox(
                            label="Summary Statistics",
                            placeholder="Summary will appear here...",
                            interactive=False,
                            lines=8
                        )
                        
                        # Experiment info
                        experiment_info = gr.Textbox(
                            label="Experiment Info",
                            placeholder="Experiment details will appear here...",
                            interactive=False,
                            lines=6
                        )
        
        # EVENT HANDLERS
        
        # Load idea
        def handle_load_idea(idea_text):
            success, message = assistant.load_idea(idea_text)
            status, code_status, chat_history, plot_chat_history, logs = update_interface()
            
            # Update progress
            progress = "✅ Idea Loaded | ⏳ Step 1: Pending | ⏳ Step 2: Pending | ⏳ Step 3: Pending"
            
            return status, logs, message, progress
        
        # Step 1: Auto Analysis
        def handle_step1():
            success, message = assistant.step1_auto_analyze()
            status, code_status, chat_history, plot_chat_history, logs = update_interface()
            
            # Update progress
            if success:
                progress = "✅ Idea Loaded | ✅ Step 1: Complete | ⏳ Step 2: Pending | ⏳ Step 3: Pending"
            else:
                progress = "✅ Idea Loaded | ❌ Step 1: Failed | ⏳ Step 2: Pending | ⏳ Step 3: Pending"
            
            return status, logs, message, progress
        
        # Step 2: Generate Code
        def handle_step2():
            success, message, code = assistant.step2_generate_code()
            status, code_status, chat_history, plot_chat_history, logs = update_interface()
            

            if success:
                progress = "✅ Idea Loaded | ✅ Step 1: Complete | ✅ Step 2: Code Generated | ⏳ Step 3: Pending"
                assistant.add_message("assistant", "✅ I've generated code based on your research idea. Please review it and provide feedback, or type 'ok' if you're satisfied.")
                chat_history = assistant.get_chat_history()
            else:
                progress = "✅ Idea Loaded | ✅ Step 1: Complete | ❌ Step 2: Failed | ⏳ Step 3: Pending"
            
            return status, code if code else "", code_status, chat_history, logs, progress

        # Apply feedback to code
        def handle_feedback(feedback):
            if not feedback.strip():
                return "", *update_interface()
            
            assistant.add_message("user", feedback)
            
            success, message, updated_code = assistant.apply_feedback(feedback)
            
            if success:
                assistant.add_message("assistant", message)
                if feedback.lower() == "ok":
                    assistant.add_message("assistant", "✅ Code approved! You can now proceed to Step 3.")
                else:
                    assistant.add_message("assistant", "I've updated the code based on your feedback. Please review again or type 'ok' to approve.")
            else:
                assistant.add_message("assistant", f"❌ {message}")
            
            status, code_status, chat_history, plot_chat_history, logs = update_interface()
            return "", updated_code, code_status, chat_history, logs

        def handle_run_code():
            success, message = assistant.run_code()  # Thêm assistant.
            assistant.add_message("assistant", message)
            status, code_status, chat_history, plot_chat_history, logs = update_interface()
            return status, code_status, chat_history, logs
        # Step 3: Run experiment
        def handle_step3(model):
            success, message = assistant.step3_run_experiment(model)
            assistant.add_message("assistant", message)
            status, code_status, chat_history, plot_chat_history, logs = update_interface()
            
            # Update progress
            if success:
                progress = "✅ Idea Loaded | ✅ Step 1: Complete | ✅ Step 2: Code Approved | 🔄 Step 3: Running..."
            else:
                progress = "✅ Idea Loaded | ✅ Step 1: Complete | ✅ Step 2: Code Approved | ❌ Step 3: Failed"
            
            return status, chat_history, logs, progress
        
        # Refresh plots
        def refresh_plots():
            plot_files = assistant.get_plot_images()
            
            # Get detailed plot info
            if plot_files:
                info = f"📊 Found {len(plot_files)} plot files:\n"
                for plot_file in plot_files:
                    # Show relative path from experiment folder for better context
                    if assistant.current_experiment_folder:
                        rel_path = os.path.relpath(plot_file, assistant.current_experiment_folder)
                        info += f"  • {rel_path}\n"
                    else:
                        info += f"  • {os.path.basename(plot_file)}\n"
            else:
                info = "📭 No plot files found."
                if assistant.current_experiment_folder:
                    info += f"\nSearched in: {assistant.current_experiment_folder}"
                else:
                    info += "\nNo experiment folder selected."
            
            return plot_files, info
        
        # Download plots
        def download_plots():
            """Create a zip file containing all plots for download"""
            plot_files = assistant.get_plot_images()
            
            if not plot_files:
                return "📭 No plots available for download"
            
            try:
                import zipfile
                from datetime import datetime
                
                # Create zip filename with timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                zip_filename = f"plots_{timestamp}.zip"
                zip_path = os.path.join(os.getcwd(), zip_filename)
                
                with zipfile.ZipFile(zip_path, 'w') as zipf:
                    for plot_file in plot_files:
                        if os.path.exists(plot_file):
                            # Use relative path as archive name for better organization
                            if assistant.current_experiment_folder:
                                arcname = os.path.relpath(plot_file, assistant.current_experiment_folder)
                            else:
                                arcname = os.path.basename(plot_file)
                            zipf.write(plot_file, arcname)
                
                info = f"✅ Created {zip_filename} with {len(plot_files)} plot files\n📁 Location: {zip_path}"
                assistant.log(info)
                return info
                
            except Exception as e:
                error_msg = f"❌ Error creating plot archive: {str(e)}"
                assistant.log(error_msg)
                return error_msg
        
        # Handle plot feedback
        def handle_plot_feedback(feedback):
            if not feedback.strip():
                return "", assistant.get_plot_images(), assistant.get_plot_chat_history()
            
            assistant.add_plot_message("user", feedback)
            
            success, message = assistant.apply_plot_feedbacks(feedback)
            
            if success:
                assistant.add_plot_message("assistant", message)
                if feedback.lower() == "ok":
                    assistant.add_plot_message("assistant", "🎉 Excellent! All plots have been approved. Your research analysis is now complete!")
                else:
                    assistant.add_plot_message("assistant", "I've updated the plots based on your feedback. Please review the new plots.")
            else:
                assistant.add_plot_message("assistant", f"❌ {message}")
            
            # Refresh plots nếu thành công
            updated_plots = assistant.get_plot_images() if success else []
            plot_chat_history = assistant.get_plot_chat_history()
            
            return "", updated_plots, plot_chat_history
        
        # Refresh results
        def handle_refresh_results():
            results = "No results available yet."
            notes = "No notes available yet."
            summary = "No summary available yet."
            exp_info = "No experiment info available yet."
            
            try:
                # Check for latest experiment folder
                if assistant.current_experiment_folder and os.path.exists(assistant.current_experiment_folder):
                    folder_name = assistant.current_experiment_folder
                    
                    # Load results
                    results_file = os.path.join(folder_name, "results.json")
                    if os.path.exists(results_file):
                        with open(results_file, 'r', encoding='utf-8') as f:
                            results = json.dumps(json.load(f), indent=2)
                    
                    # Load notes
                    notes_file = os.path.join(folder_name, "notes.txt")
                    if os.path.exists(notes_file):
                        with open(notes_file, 'r', encoding='utf-8') as f:
                            notes = f.read()
                    
                    # Generate summary
                    summary = f"Experiment folder: {os.path.basename(folder_name)}\n"
                    summary += f"Experiment completed: {assistant.experiment_complete}\n"
                    summary += f"Plot files: {len(assistant.get_plot_images())}\n"
                    
                    # Experiment info
                    if assistant.current_idea:
                        exp_info = f"Title: {assistant.current_idea['Title']}\n"
                        exp_info += f"Name: {assistant.current_idea['Name']}\n"
                        exp_info += f"Description: {assistant.current_idea['Experiment']}\n"
                        
            except Exception as e:
                results = f"Error loading results: {str(e)}"
            
            return results, notes, summary, exp_info
        
        # CONNECT EVENTS
        
        # Setup tab
        load_idea_btn.click(
            handle_load_idea,
            inputs=[idea_json],
            outputs=[status_text, logs_text, status_text, progress_md]
        )
        
        step1_btn.click(
            handle_step1,
            outputs=[status_text, logs_text, status_text, progress_md]
        )
        
        step2_btn.click(
            handle_step2,
            outputs=[status_text, code_display, code_status, chatbot, logs_text, progress_md]
        )
        
        step3_btn.click(
            handle_step3,
            inputs=[model_dropdown],
            outputs=[status_text, chatbot, logs_text, progress_md]
        )
        
        # Code tab
        send_btn.click(
            handle_feedback,
            inputs=[feedback_input],
            outputs=[feedback_input, code_display, code_status, chatbot, logs_text]
        )
        
        feedback_input.submit(
            handle_feedback,
            inputs=[feedback_input],
            outputs=[feedback_input, code_display, code_status, chatbot, logs_text]
        )
        
        run_code_btn.click(
            handle_run_code,
            outputs=[status_text, code_status, chatbot, logs_text]
        )
        
        regenerate_btn.click(
            handle_step2,
            outputs=[status_text, code_display, code_status, chatbot, logs_text, progress_md]
        )
        
        # Plot tab
        refresh_plots_btn.click(
            refresh_plots,
            outputs=[plot_gallery, plot_info]
        )
        
        download_plots_btn.click(
            download_plots,
            outputs=[plot_info]
        )
        
        send_plot_btn.click(
            handle_plot_feedback,
            inputs=[plot_feedback_input],
            outputs=[plot_feedback_input, plot_gallery, plot_chatbot]
        )
        
        plot_feedback_input.submit(
            handle_plot_feedback,
            inputs=[plot_feedback_input],
            outputs=[plot_feedback_input, plot_gallery, plot_chatbot]
        )
        
        # Results tab
        refresh_results_btn.click(
            handle_refresh_results,
            outputs=[results_text, notes_text, summary_stats, experiment_info]
        )
        
        # Auto-refresh plots when experiment completes
        def check_experiment_status():
            if assistant.experiment_complete and assistant.current_experiment_folder:
                plots = assistant.get_plot_images()
                
                # Add automatic notification when plots are found
                if plots and not hasattr(assistant, '_plots_notified'):
                    assistant.add_plot_message("assistant", f"🎉 Found {len(plots)} plots from your completed experiment! Review them and provide feedback, or type 'ok' if they look good.")
                    assistant._plots_notified = True
                
                plot_chat = assistant.get_plot_chat_history()
                return plots, plot_chat
            return [], []
        
        # Periodic refresh for experiment completion
        demo.load(
            check_experiment_status,
            outputs=[plot_gallery, plot_chatbot],
        )
    
    return demo
# Launch app
if __name__ == "__main__":
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    
    os.makedirs("result", exist_ok=True)
    os.makedirs("analysis", exist_ok=True)
    os.makedirs("coding-agent", exist_ok=True)
    
    demo = create_interface()
    demo.launch(
        server_name="localhost",
        server_port=7860,
        share=False,
        inbrowser=True,
        show_error=True
    )