"""
Datarus Agent - Multi-step reasoning agent implementation
"""

import os
import re
import json
import requests
from typing import Dict, List, Optional
from uuid import uuid4

from src.core.notebook_manager import NotebookManager
from src.utils.logger import ConsoleLogger


class DatarusAgent:
    """Main agent for Datarus multi-step reasoning pipeline"""
        
    def __init__(self, api_key: str, challenge_config: Dict, config: Dict = None):
        self.api_key = api_key
        self.config = config or {}
        self.step_counter = 0
        self.is_correction = False
        self.consecutive_errors = 0
        self.logger = ConsoleLogger()
        self.step_history = {}
        
        # Initialize Docker configuration
        docker_config = self.config.get('docker', {})
        remote_serverip = docker_config.get('remote_server_ip', os.getenv("REMOTE_SERVER_IP", "localhost"))
        container_prefix = docker_config.get('container_prefix', 'datarus_container')
        docker_image = docker_config.get('image', 'jupyter/tensorflow-notebook')
        
        # Use UUID for unique container naming
        rand_uuid = str(uuid4())
        container_name = f"{container_prefix}_{rand_uuid[:8]}"
        
        # Initialize notebook manager with proper config
        self.notebook_manager = NotebookManager(container_name, remote_serverip, config)
        self.notebook_manager.ensure_container(docker_image)
        
        # Create notebook with unique name
        notebook_name = f"datarus_notebook_{rand_uuid[:8]}"
        self.notebook_manager.load_or_create_notebook(notebook_name)
        
        # Initialize environment
        self._initialize_environment(challenge_config)

    def _initialize_environment(self, challenge_config: Dict):
        """Initialize notebook environment with datasets and libraries"""
        self.logger.print_step("Environment Initialization", "Setting up notebook and dataset", "magenta")
        
        # Standard environment initialization
        ENV_INIT_SCRIPT = """import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn import *"""
        
        step_name = "Environment Initialization"
        self.notebook_manager.create_step(step_name)
        
        exit_code, outputs = self.notebook_manager.add_cell_to_step('code', ENV_INIT_SCRIPT, 'setup', step_name)
        if exit_code != 0:
            self.logger.print_error(f"Environment initialization failed")
            raise RuntimeError("Environment initialization failed")
        
        if "dataset_generation_code" in challenge_config:
            exit_code, outputs = self.notebook_manager.add_cell_to_step(
                'code', challenge_config["dataset_generation_code"], 'data_gen', step_name
            )
            if exit_code != 0:
                self.logger.print_error(f"Dataset generation failed")
                raise RuntimeError("Dataset generation failed")
        
        outputs = self.notebook_manager.get_outputs_by_step(step_name)
        if outputs['has_error']:
            self.logger.print_error(f"Dataset generation error: {outputs['error']}")
            raise RuntimeError(f"Dataset generation error: {outputs['error']}")
        
        self.logger.print_status("Environment initialized successfully", "success")

    def chat(self, messages: List[Dict[str, str]]) -> Dict:
        """Send request to Datarus model using configuration"""
        messages = self.clean_chat_history(messages)
        
        # Deduplicate consecutive identical messages
        dedupe_messages = []
        for msg in messages:
            if not dedupe_messages or msg['content'] != dedupe_messages[-1]['content']:
                dedupe_messages.append(msg)
        
        self.logger.print_step("LLM Request", f"Sending request with {len(dedupe_messages)} messages", "cyan")
        
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        
        # Get model configuration
        model_config = self.config.get('model', {})
        endpoint = model_config.get('endpoint', os.getenv('MODEL_ENDPOINT', 'http://localhost:8000/v1/chat/completions'))
        model_name = model_config.get('model_name', 'Datarus/Datarus-R1-14B-preview')
        temperature = model_config.get('temperature', 0.7)
        max_tokens = model_config.get('max_tokens', 2048)
        stream = model_config.get('stream', False)
        stop_tokens = model_config.get('stop_tokens', ["</step>", "</stop_analysis>", "<｜end▁of▁sentence｜>"])
        
        payload = {
            "model": model_name,
            "messages": dedupe_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
            "stop": stop_tokens
        }

        response = requests.post(
            endpoint,
            headers=headers,
            json=payload,
            timeout=60
        )

        response.raise_for_status()
        return response.json()

    def execute_code(self, code: str, thought: str) -> str:
        """Execute code in notebook and return output"""
        if not self.is_correction:
            self.step_counter += 1
            step_name = f"Step {self.step_counter}"
            # Truncate thought for display
            print_thought = thought[:1024 - 3] + "..." if len(thought) >= 1024 else thought
            self.logger.print_step(f"Executing Code - {step_name}", print_thought, "blue")
            self.logger.print_code(code)
            
            self.notebook_manager.create_step(step_name)
            self.notebook_manager.add_cell_to_step('markdown', thought, 'thought', step_name, execute=False)
            exit_code, outputs = self.notebook_manager.add_cell_to_step('code', code, 'code', step_name)
        else:
            step_name = f"Step {self.step_counter}"
            self.logger.print_correction(f"Retrying {step_name} after error")
            self.logger.print_code(code)
            exit_code, outputs = self.notebook_manager.update_cell_content(step_name, 'code', 'code', code)
        
        outputs_dict = self.notebook_manager.get_outputs_by_step(step_name)
        
        if outputs_dict['has_error']:
            self.is_correction = True
            self.consecutive_errors += 1
            self.logger.print_error(outputs_dict['error'])
            self.step_history[self.step_counter] = {'status': 'error', 'output': outputs_dict['error']}
            
            # Use max_consecutive_errors from config
            max_errors = self.config.get('pipeline', {}).get('max_consecutive_errors', 5)
            if self.consecutive_errors >= max_errors:
                raise RuntimeError(f"Too many consecutive errors at step {self.step_counter}")
            return outputs_dict['error']
        else:
            self.is_correction = False
            self.consecutive_errors = 0
            result = outputs_dict['stream']
            
            # Use output_truncate_length from config
            truncate_length = self.config.get('pipeline', {}).get('output_truncate_length', 10000)
            if len(result) > truncate_length:
                result = result[:truncate_length] + "\n... [OUTPUT TRUNCATED]"
            self.logger.print_result(result[:500] + "..." if len(result) > 500 else result)
            
            self.step_history[self.step_counter] = {'status': 'success', 'output': result}
            
            return result

    def clean_chat_history(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Remove failed attempts from chat history, keeping only the last error attempt.
        This maintains conversation coherence while preventing error accumulation.
        """
        if not self.step_history:
            return messages
        
        # Collect all error steps
        error_steps = [step_num for step_num in sorted(self.step_history.keys()) 
                       if self.step_history[step_num]['status'] == 'error']
        
        # Need at least 2 errors to start removing (keep the last one)
        if len(error_steps) < 2:
            return messages
        
        # Remove all error steps except the last one
        steps_to_remove = set(error_steps[:-1])
        
        cleaned_messages = []
        i = 0
        
        while i < len(messages):
            msg = messages[i]
            
            # Check if this message contains a step we want to remove
            should_remove = False
            for step_num in steps_to_remove:
                step_name = f"Step {step_num}"
                if step_name in msg['content'] and '</step>' in msg['content']:
                    should_remove = True
                    break
            
            if not should_remove:
                # Keep this message
                cleaned_messages.append(msg)
                i += 1
            else:
                # This is an error step we're removing
                # Also skip the next message if it's a user message
                i += 1  # Skip current message
                
                # Check if next message exists and is from user
                if i < len(messages) and messages[i].get('role') == 'user':
                    # Skip the user's follow-up message too
                    i += 1
        
        return cleaned_messages

    def parse_response(self, text: str):
        """
        Parse Datarus model response to extract steps and final answer.
        Handles various response formats with robust error recovery.
        """
        steps = []
        final_answer = None
        
        # Preprocessing: Clean up common issues
        text = self._preprocess_response(text)
        
        # Try to extract well-formed steps first
        steps = self._extract_steps(text)
        
        # If no well-formed steps found, try to recover from malformed XML
        if not steps:
            steps = self._extract_steps_fallback(text)
        
        # Extract final answer
        final_answer = self._extract_final_answer(text)
        
        return steps, final_answer

    def _preprocess_response(self, text: str) -> str:
        """Preprocess response to fix common formatting issues"""
        # Remove leading </think> or <think> tags
        text = re.sub(r'^</?think>\s*', '', text, flags=re.IGNORECASE)
        
        # Fix case sensitivity issues by standardizing tags
        replacements = [
            (r'<Step>', '<step>'), (r'</Step>', '</step>'),
            (r'<Thought>', '<thought>'), (r'</Thought>', '</thought>'),
            (r'<Action>', '<action>'), (r'</Action>', '</action>'),
            (r'<Action_input>', '<action_input>'), (r'</Action_input>', '</action_input>'),
        ]
        for old, new in replacements:
            text = re.sub(old, new, text)
        
        # Fix incomplete XML structures
        if '</thought>' in text and '<thought>' not in text:
            # Add opening thought tag at appropriate position
            thought_close_pos = text.index('</thought>')
            # Look for <step> before this position
            step_start = text.rfind('<step>', 0, thought_close_pos)
            if step_start != -1:
                insert_pos = step_start + len('<step>')
                text = text[:insert_pos] + '<thought>' + text[insert_pos:]
            else:
                text = '<thought>' + text
        
        # Ensure action tags have content (default to python_executor)
        text = re.sub(r'<action>\s*</action>', '<action>python_executor</action>', text)
        
        return text

    def _extract_steps(self, text: str) -> List[tuple]:
        """Extract well-formed steps from the response"""
        steps = []
        
        # Find all step blocks
        step_pattern = r'<step>(.*?)</step>'
        step_matches = re.finditer(step_pattern, text, re.DOTALL)
        
        for step_match in step_matches:
            step_content = step_match.group(1)
            
            # Extract components
            thought = self._extract_tag_content(step_content, 'thought')
            action = self._extract_tag_content(step_content, 'action') or 'python_executor'
            action_input = self._extract_tag_content(step_content, 'action_input')
            
            if action_input:  # Must have code to be valid
                action_input = self._clean_code_block(action_input)
                if action_input:  # Check again after cleaning
                    steps.append((thought, action, action_input))
        
        return steps

    def _extract_steps_fallback(self, text: str) -> List[tuple]:
        """
        Fallback extraction for malformed XML.
        Tries to extract steps even from poorly formatted responses.
        """
        steps = []
        
        # Look for action_input tags even without proper step tags
        action_input_pattern = r'<action_input>(.*?)</action_input>'
        action_input_matches = re.finditer(action_input_pattern, text, re.DOTALL)
        
        for match in action_input_matches:
            action_input = self._clean_code_block(match.group(1))
            if action_input:
                # Try to find associated thought before this action_input
                before_text = text[:match.start()]
                thought = self._extract_tag_content(before_text, 'thought') or "Executing code"
                
                # Try to find associated action
                action = self._extract_tag_content(before_text, 'action') or 'python_executor'
                
                steps.append((thought, action, action_input))
        
        return steps

    def _extract_tag_content(self, text: str, tag: str) -> str:
        """Extract content from the last occurrence of a tag"""
        pattern = f'<{tag}>(.*?)</{tag}>'
        matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
        return matches[-1].strip() if matches else ""

    def _extract_final_answer(self, text: str) -> Optional[str]:
        """Extract final answer from stop_analysis block"""
        stop_pattern = r'<stop_analysis>(.*?)</stop_analysis>'
        stop_match = re.search(stop_pattern, text, re.DOTALL | re.IGNORECASE)
        
        if stop_match:
            return self._extract_tag_content(stop_match.group(1), 'answer')
        
        return None

    def _clean_code_block(self, code: str) -> str:
        """
        Clean code block of markdown formatting and extra whitespace.
        Handles various formatting styles.
        """
        if not code:
            return ""
        
        code = code.strip()
        
        # Remove markdown code blocks
        if code.startswith('```'):
            # Find the end of first line (language specifier)
            first_newline = code.find('\n')
            if first_newline != -1:
                code = code[first_newline + 1:]
            else:
                # No newline, just remove the ```
                code = code[3:]
        
        # Remove trailing ```
        if code.endswith('```'):
            code = code[:-3]
        
        # Final strip
        code = code.strip()
        
        return code

    def process_query(self, query: str) -> str:
        """Process user query through multi-step reasoning"""
        print_query = query[:1024 - 3] + "..." if len(query) >= 1024 else query
        self.logger.print_step("Processing Query", print_query + "...", "magenta")
        
        # System prompt for the model
        system_prompt = """You are an intelligent data scientist assistant. Answer the user questions step by step.

You have access to the following tool:

Tool: python_executor
Description: Executes Python code and returns the output. You must use print() to capture outputs. Available libraries include pandas, numpy, scikit-learn, matplotlib, seaborn, and other common data science libraries.
Input: Valid Python code
Output: Output of the Python code execution

IMPORTANT: Start your response directly with <step> tags. Do not use <think> tags.

Use EXACTLY this format for your responses:

<step>
<thought>explain what you're going to do</thought>
<action>python_executor</action>
<action_input>
# Your Python code here
print("output")
</action_input>
</step>

After receiving an observation, continue with more steps as needed. When you have the final answer:

<stop_analysis>
<thought>summarize what you found</thought>
<answer>the final answer to the question</answer>
</stop_analysis>

Remember:
- If you encounter an error, analyze it and try a different approach
- Use print() statements to see outputs
- Start immediately with <step> tags"""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]
        
        # Get max_iterations from config
        max_iterations = self.config.get('pipeline', {}).get('max_iterations', 30)
        
        for i in range(max_iterations):
            self.logger.print_status(f"Iteration {i+1}/{max_iterations}", "info")
            
            response = self.chat(messages)
            content = response['choices'][0]['message']['content']
            
            # Handle </think> prefix if present
            if content.startswith("</think>"):
                content = content[8:]
            
            self.logger.print_llm_response(content)
            
            # Fix incomplete XML if needed
            if "<step>" in content and "</step>" not in content:
                content += "</step>"

            steps, final_answer = self.parse_response(content)
            
            if steps:
                for thought, action, code in steps:
                    self.logger.print_thought(thought)
                    
                    if action.lower() == "python_executor":
                        observation = self.execute_code(code, thought)
                        
                        # Format step XML
                        step_xml = f"""<step>
<thought>{thought}</thought>
<action>{action}</action>
<action_input>{code}</action_input>
<observation>{observation}</observation>
</step>"""
                        
                        messages.append({"role": "assistant", "content": step_xml})
                        
                        # Check for errors and provide appropriate response
                        if observation.startswith("Error:"):
                            messages.append({"role": "user", "content": "The code resulted in an error. Please analyze the error and try a different approach."})
                        else:
                            messages.append({"role": "user", "content": "Continue with the analysis."})
            
            if final_answer:
                self.logger.print_step("Final Answer", final_answer, "green")
                return final_answer
            
            # Fallback handling for malformed responses
            if not steps and not final_answer:
                self.logger.print_status("No valid XML format detected, prompting to continue...", "warning")
                messages.append({"role": "assistant", "content": content})
                messages.append({"role": "user", "content": "Please continue the analysis using the XML format with <step> tags. Remember to start with <step> directly, not <think>."})
        
        self.logger.print_error("Analysis could not be completed within the iteration limit.")
        return "Analysis could not be completed within the iteration limit."
