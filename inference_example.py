#!/usr/bin/env python3
"""
GFlowNet for StrategyQA - Inference Example
Author: Shuhao Zhang
Date: 2025/9/30

This script demonstrates how to use the trained GFlowNet model for inference.
"""

import json
import logging
import sys
import os
from typing import List, Dict, Any

# Add project path
sys.path.append(os.path.dirname(__file__))

# Import project modules
from main import GFlowNetProject
from gfn.sampler import GFlowNetSampler
from gfn.model import GFlowNetPolicy
from llm.generate import LLMGenerator
from llm.promt import ReasoningChain

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GFlowNetInference:
    """GFlowNet Inference Class"""
    
    def __init__(self, config_path: str = "config_real.json"):
        """
        Initialize inference system
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.project = None
        self.sampler = None
        self.llm_generator = None
        self.policy_net = None
        
        self._load_config()
        self._initialize_components()
    
    def _load_config(self):
        """Load configuration"""
        try:
            with open(self.config_path, 'r') as f:
                self.config = json.load(f)
            logger.info(f"Configuration loaded from {self.config_path}")
        except FileNotFoundError:
            logger.error(f"Configuration file {self.config_path} not found")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in configuration file: {e}")
            raise
    
    def _initialize_components(self):
        """Initialize inference components"""
        logger.info("Initializing inference components...")
        
        # Initialize LLM generator
        self.llm_generator = LLMGenerator(
            model_name=self.config.get('model_name', 'microsoft/DialoGPT-medium'),
            device=self.config.get('device', 'auto')
        )
        
        # Initialize policy network
        self.policy_net = GFlowNetPolicy(
            hidden_dim=self.config.get('hidden_dim', 256),
            num_layers=self.config.get('num_layers', 2),
            dropout=self.config.get('dropout', 0.1),
            max_steps=self.config.get('max_steps', 5)
        )
        
        # Initialize sampler
        self.sampler = GFlowNetSampler(
            policy_net=self.policy_net,
            llm_generator=self.llm_generator,
            max_steps=self.config.get('max_steps', 5),
            temperature=self.config.get('temperature', 1.0),
            num_candidates=self.config.get('num_candidates', 3)
        )
        
        logger.info("Inference components initialized successfully")
    
    def generate_reasoning(self, question: str, show_steps: bool = True) -> Dict[str, Any]:
        """
        Generate reasoning chain for a question
        Args:
            question: Input question
            show_steps: Whether to show intermediate steps
        Returns:
            Dictionary containing reasoning results
        """
        logger.info(f"Generating reasoning for: {question}")
        
        try:
            # Sample trajectory
            final_state, trajectory_data = self.sampler.sample_trajectory(question)
            
            # Extract reasoning steps
            reasoning_steps = final_state.steps
            final_answer = final_state.final_answer if hasattr(final_state, 'final_answer') else "No answer generated"
            
            # Prepare result
            result = {
                'question': question,
                'reasoning_steps': reasoning_steps,
                'final_answer': final_answer,
                'num_steps': len(reasoning_steps),
                'trajectory_data': trajectory_data if show_steps else None
            }
            
            if show_steps:
                logger.info("Reasoning steps:")
                for i, step in enumerate(reasoning_steps, 1):
                    logger.info(f"  Step {i}: {step}")
                logger.info(f"Final answer: {final_answer}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating reasoning: {e}")
            return {
                'question': question,
                'error': str(e),
                'reasoning_steps': [],
                'final_answer': "Error occurred",
                'num_steps': 0
            }
    
    def batch_inference(self, questions: List[str]) -> List[Dict[str, Any]]:
        """
        Perform batch inference on multiple questions
        Args:
            questions: List of questions
        Returns:
            List of reasoning results
        """
        logger.info(f"Performing batch inference on {len(questions)} questions")
        
        results = []
        for i, question in enumerate(questions, 1):
            logger.info(f"Processing question {i}/{len(questions)}")
            result = self.generate_reasoning(question, show_steps=False)
            results.append(result)
        
        return results
    
    def interactive_mode(self):
        """Run interactive inference mode"""
        logger.info("Starting interactive inference mode")
        logger.info("Type 'quit' to exit, 'help' for commands")
        
        while True:
            try:
                question = input("\nEnter your question: ").strip()
                
                if question.lower() == 'quit':
                    break
                elif question.lower() == 'help':
                    print("\nCommands:")
                    print("  - Enter any question to get reasoning")
                    print("  - 'quit' to exit")
                    print("  - 'help' to show this message")
                    continue
                elif not question:
                    continue
                
                # Generate reasoning
                result = self.generate_reasoning(question, show_steps=True)
                
                # Display result
                print(f"\nQuestion: {result['question']}")
                print(f"Number of reasoning steps: {result['num_steps']}")
                print(f"Final answer: {result['final_answer']}")
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Error in interactive mode: {e}")
        
        logger.info("Interactive mode ended")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='GFlowNet Inference Example')
    parser.add_argument('--config', type=str, default='config_real.json',
                       help='Configuration file path')
    parser.add_argument('--question', type=str, default=None,
                       help='Single question to process')
    parser.add_argument('--questions_file', type=str, default=None,
                       help='File containing questions (one per line)')
    parser.add_argument('--interactive', action='store_true',
                       help='Run in interactive mode')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file for results (JSON format)')
    
    args = parser.parse_args()
    
    try:
        # Initialize inference system
        inference = GFlowNetInference(args.config)
        
        if args.interactive:
            # Interactive mode
            inference.interactive_mode()
        
        elif args.question:
            # Single question
            result = inference.generate_reasoning(args.question, show_steps=True)
            
            if args.output:
                with open(args.output, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
                logger.info(f"Result saved to {args.output}")
        
        elif args.questions_file:
            # Batch processing from file
            with open(args.questions_file, 'r', encoding='utf-8') as f:
                questions = [line.strip() for line in f if line.strip()]
            
            results = inference.batch_inference(questions)
            
            if args.output:
                with open(args.output, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                logger.info(f"Results saved to {args.output}")
            else:
                # Print summary
                for result in results:
                    print(f"Q: {result['question']}")
                    print(f"A: {result['final_answer']} ({result['num_steps']} steps)")
                    print()
        
        else:
            # Default: run example questions
            example_questions = [
                "Did the Hopi Indians use a symbol that was similar to the swastika?",
                "Is it true that the Great Wall of China is visible from space?",
                "Do penguins have knees?",
                "Can you make a diamond from coal?",
                "Is the tomato a fruit or a vegetable?"
            ]
            
            logger.info("Running example questions...")
            results = inference.batch_inference(example_questions)
            
            # Print results
            for result in results:
                print(f"\nQuestion: {result['question']}")
                print(f"Answer: {result['final_answer']}")
                print(f"Reasoning steps: {result['num_steps']}")
                if result['reasoning_steps']:
                    print("Steps:")
                    for i, step in enumerate(result['reasoning_steps'], 1):
                        print(f"  {i}. {step}")
                print("-" * 50)
    
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
