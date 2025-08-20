#!/usr/bin/env python3
"""
Datarus Multi-Step Reasoning Pipeline - Main Entry Point
"""

import os
import sys
import json
import argparse
import random
from pathlib import Path
from uuid import uuid4

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.agent import DatarusAgent
from src.utils.config_loader import load_config
from src.utils.logger import ConsoleLogger


def main():
    parser = argparse.ArgumentParser(
        description='Datarus Multi-Step Reasoning Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--challenge', type=str,
                       help='Path to challenge JSON file')
    parser.add_argument('--query', type=str,
                       help='Direct query to process')
    parser.add_argument('--output-notebook', type=str,
                       help='Path to save the final notebook')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug output')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    if args.debug:
        config.setdefault('development', {})['debug'] = True
    
    # Initialize console
    console = ConsoleLogger()
    console.print("\n[bold yellow]🔍 Datarus Data Analysis Agent[/]")
    console.print("=" * 80)
    
    # Load challenge or get query
    challenge_config = None
    query = None
    
    if args.challenge:
        with open(args.challenge, "r") as f:
            challenge_config = json.load(f)
        console.print(f"[bold green]Loaded challenge: {challenge_config.get('domain', 'Unknown')}[/]")
        
        random_num = random.randint(1000, 9999)
        challenge_data_name = f"challenge_data_{random_num}.pkl"
        challenge_config["challenge_data_name"] = challenge_data_name
        
        load_raw_df = """
import pandas as pd
with open('/tmp/challenge_data.pkl', 'rb') as file:
    data_object = pickle.load(file)"""
        
        challenge_config["dataset_generation_code"] = challenge_config["dataset_generation_code"].replace(
            "challenge_data.pkl", challenge_data_name
        )
        challenge_config["dataset_generation_code"] += "\n" + load_raw_df.replace(
            "challenge_data.pkl", challenge_data_name
        )
        
        query = f"""Please analyze the following dataset and complete the task.

{challenge_config.get('scenario', '')}

{challenge_config.get('objective', '')}

The data has been loaded as 'data_object'. 

Requirements:
1. Start your response with <step> tags immediately
2. Explore the data thoroughly
3. Build and evaluate appropriate models
4. Provide clear results and conclusions

Begin your analysis now."""
            
    elif args.query:
        query = args.query
    else:
        console.print("Enter your query:")
        query = input("> ")
    
    if not query:
        console.print_error("No query provided")
        sys.exit(1)
    
    # Initialize agent
    console.print("\n[bold cyan]Initializing agent...[/]")
    api_key = os.getenv('DATARUS_API_KEY', '')
    agent = DatarusAgent(api_key, challenge_config, config)
    
    # Display query
    print_query = query[:1024 - 3] + "..." if len(query) >= 1024 else query
    console.print("\n[bold magenta]Query:[/]")
    console.print_panel(print_query, "magenta")
    
    # Process query
    try:
        result = agent.process_query(query)
        console.print(f"\n{'='*80}")
        console.print("[bold green]FINAL RESULT:[/]")
        console.print_panel(result, "green")
        console.print(f"{'='*80}")
        
        # Save notebook if requested
        if args.output_notebook:
            output_path = args.output_notebook
            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            agent.notebook_manager.save_notebook_to_file(output_path)
            console.print(f"Notebook saved to: {output_path}", "success")
            
    except Exception as e:
        console.print_error(f"Pipeline failed: {str(e)}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)
    finally:
        console.print("\n[bold yellow]Cleaning up...[/]")
        agent.notebook_manager.cleanup()


if __name__ == "__main__":
    main()
