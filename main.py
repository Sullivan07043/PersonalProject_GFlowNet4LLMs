#!/usr/bin/env python3
"""
GFlowNet for StrategyQA Project Main Entry
Author: Shuhao Zhang
Date: 2025/9/30
"""

import argparse
import logging
import os
import sys
from typing import Dict, Any, Optional
import json
from datetime import datetime

# Add project path
sys.path.append(os.path.dirname(__file__))

# Import project modules
from data.dataset import StrategyQADataset
from llm.generate import MockLLMGenerator
from llm.promt import ReasoningChain

# Add paths to resolve import issues
sys.path.append(os.path.join(os.path.dirname(__file__), 'gfn'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'eval'))

from gfn.model import MockGFlowNetPolicy, GFlowNetState
from gfn.sampler import MockGFlowNetSampler
from gfn.losses import GFlowNetLoss, MockRewardFunction
from gfn.train import MockGFlowNetTrainer
from evaluation.eval import MockGFlowNetEvaluator
from evaluation.metrics import MockEvaluationMetrics, ReasoningDiversity

# Setup logging will be done in main() function

class GFlowNetProject:
    """GFlowNet Project Main Class"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize project
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.setup_directories()
        self.initialize_components()
    
    def setup_directories(self):
        """Create necessary directories"""
        directories = [
            'checkpoints',
            'results',
            'data/cache'
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            logging.getLogger(__name__).info(f"Created directory: {directory}")
    
    def initialize_components(self):
        """Initialize project components"""
        logging.getLogger(__name__).info("Initializing project components...")
        
        # Choose to use real components or mock components based on config
        use_mock = self.config.get('use_mock', True)
        
        if use_mock:
            logging.getLogger(__name__).info("Using mock components for testing")
            self.llm_generator = MockLLMGenerator()
            self.policy_net = MockGFlowNetPolicy()
            self.sampler = MockGFlowNetSampler()
            self.loss_fn = GFlowNetLoss()
            self.reward_fn = MockRewardFunction()
            self.trainer = MockGFlowNetTrainer()
            self.evaluator = MockGFlowNetEvaluator()
        else:
            logging.getLogger(__name__).info("Using real components")
            # Initialize real components
            from llm.generate import LLMGenerator
            from gfn.model import GFlowNetPolicy
            from gfn.sampler import GFlowNetSampler
            from gfn.losses import RewardFunction
            from gfn.train import GFlowNetTrainer
            from evaluation.eval import GFlowNetEvaluator
            from evaluation.metrics import EvaluationMetrics
            
            # Initialize real LLM generator
            self.llm_generator = LLMGenerator(
                model_name=self.config.get('model_name', 'microsoft/DialoGPT-medium'),
                device=self.config.get('device', 'auto')
            )
            
            # Initialize real GFlowNet policy
            self.policy_net = GFlowNetPolicy(
                hidden_dim=self.config.get('hidden_dim', 512),
                num_layers=self.config.get('num_layers', 3),
                dropout=self.config.get('dropout', 0.1),
                max_steps=self.config.get('max_steps', 10)
            )
            
            # Initialize real loss function and reward function
            self.loss_fn = GFlowNetLoss(
                use_tb=self.config.get('use_tb', True),
                use_fm=self.config.get('use_fm', False),
                tb_weight=self.config.get('tb_weight', 1.0),
                fm_weight=self.config.get('fm_weight', 0.1)
            )
            
            self.reward_fn = RewardFunction(
                correctness_weight=self.config.get('correctness_weight', 1.0),
                length_penalty=self.config.get('length_penalty', 0.01),
                diversity_bonus=self.config.get('diversity_bonus', 0.1)
            )
            
            # Initialize real sampler
            self.sampler = GFlowNetSampler(
                policy_net=self.policy_net,
                llm_generator=self.llm_generator,
                max_steps=self.config.get('max_steps', 10),
                temperature=self.config.get('temperature', 1.0),
                num_candidates=self.config.get('num_candidates', 5)
            )
            
            # Initialize real trainer
            self.trainer = GFlowNetTrainer(
                policy_net=self.policy_net,
                llm_generator=self.llm_generator,
                sampler=self.sampler,
                loss_fn=self.loss_fn,
                reward_fn=self.reward_fn,
                learning_rate=self.config.get('learning_rate', 1e-4),
                device=self.config.get('device', 'auto')
            )
            
            # Initialize real evaluator
            self.evaluator = GFlowNetEvaluator(
                sampler=self.sampler,
                llm_generator=self.llm_generator,
                metrics=EvaluationMetrics(),
                diversity_metrics=ReasoningDiversity()
            )
        
        logging.getLogger(__name__).info("Component initialization completed")
    
    def load_dataset(self, split: str = "train_filtered", max_samples: Optional[int] = None):
        """Load dataset"""
        logging.getLogger(__name__).info(f"Loading dataset: {split}")
        
        dataset = StrategyQADataset(split=split)
        processed_data = dataset.preprocess(max_samples=max_samples)
        dataloader = dataset.get_dataloader(
            batch_size=self.config.get('batch_size', 8),
            shuffle=True
        )
        
        logging.getLogger(__name__).info(f"Dataset loading completed: {len(processed_data)} samples")
        return dataset, dataloader
    
    def train(self, dataloader, num_epochs: int = 10):
        """Train model"""
        logging.getLogger(__name__).info(f"Starting training, {num_epochs} epochs")
        
        # Start training
        history = self.trainer.train(dataloader, num_epochs=num_epochs)
        
        # Save training history
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        history_path = f"results/training_history_{timestamp}.json"
        
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        
        logging.getLogger(__name__).info(f"Training completed, history saved to: {history_path}")
        return history
    
    def evaluate(self, dataloader, num_samples_per_question: int = 5):
        """Evaluate model"""
        logging.getLogger(__name__).info("Starting model evaluation")
        
        # Evaluate dataset
        results = self.evaluator.evaluate_dataset(
            dataloader, 
            num_samples_per_question=num_samples_per_question
        )
        
        # Compare with baseline
        comparison = self.evaluator.compare_with_baseline(
            dataloader, 
            baseline_method="random"
        )
        
        # Save evaluation results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = f"results/evaluation_results_{timestamp}.json"
        
        evaluation_data = {
            'results': results,
            'comparison': comparison,
            'config': self.config,
            'timestamp': timestamp
        }
        
        with open(results_path, 'w') as f:
            json.dump(evaluation_data, f, indent=2)
        
        logging.getLogger(__name__).info(f"Evaluation completed, results saved to: {results_path}")
        return results, comparison
    
    def run_experiment(self, experiment_name: str = "default"):
        """Run complete experiment"""
        logging.getLogger(__name__).info(f"Starting experiment: {experiment_name}")
        
        # Load dataset
        dataset, train_dataloader = self.load_dataset(
            split=self.config.get('train_split', 'train_filtered'),
            max_samples=self.config.get('max_train_samples', 100)
        )
        
        _, eval_dataloader = self.load_dataset(
            split=self.config.get('eval_split', 'train_filtered'),
            max_samples=self.config.get('max_eval_samples', 50)
        )
        
        # Train model
        if self.config.get('do_training', True):
            training_history = self.train(
                train_dataloader, 
                num_epochs=self.config.get('num_epochs', 5)
            )
        else:
            training_history = None
        
        # Evaluate model
        if self.config.get('do_evaluation', True):
            eval_results, comparison = self.evaluate(
                eval_dataloader,
                num_samples_per_question=self.config.get('num_samples_per_question', 3)
            )
        else:
            eval_results, comparison = None, None
        
        # Generate experiment report
        self.generate_report(experiment_name, training_history, eval_results, comparison)
        
        logging.getLogger(__name__).info(f"Experiment {experiment_name} completed")
    
    def generate_report(self, experiment_name: str, training_history, eval_results, comparison):
        """Generate experiment report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = f"results/experiment_report_{experiment_name}_{timestamp}.md"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"# GFlowNet for StrategyQA Experiment Report\n\n")
            f.write(f"**Experiment Name**: {experiment_name}\n")
            f.write(f"**Time**: {timestamp}\n")
            f.write(f"**Configuration**: {json.dumps(self.config, indent=2, ensure_ascii=False)}\n\n")
            
            if training_history:
                f.write("## Training Results\n\n")
                f.write(f"- Final Loss: {training_history['losses'][-1]:.4f}\n")
                f.write(f"- Final Reward: {training_history['rewards'][-1]:.4f}\n")
                f.write(f"- Final Accuracy: {training_history['accuracies'][-1]:.4f}\n\n")
            
            if eval_results:
                f.write("## Evaluation Results\n\n")
                f.write("### Main Metrics\n")
                for metric, value in eval_results['metrics'].items():
                    f.write(f"- {metric}: {value:.4f}\n")
                
                f.write("\n### Diversity Metrics\n")
                for metric, value in eval_results['diversity'].items():
                    f.write(f"- {metric}: {value:.4f}\n")
            
            if comparison:
                f.write("\n## Baseline Comparison\n\n")
                f.write("### Improvement Percentage\n")
                for metric, improvement in comparison['improvements'].items():
                    f.write(f"- {metric}: {improvement:+.2f}%\n")
        
        logging.getLogger(__name__).info(f"Experiment report generated: {report_path}")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='GFlowNet for StrategyQA Project')
    
    # Basic parameters
    parser.add_argument('--mode', type=str, default='experiment', 
                       choices=['train', 'eval', 'experiment', 'test'],
                       help='Run mode')
    parser.add_argument('--config', type=str, default=None,
                       help='Configuration file path')
    parser.add_argument('--experiment_name', type=str, default='default',
                       help='Experiment name')
    
    # Data parameters
    parser.add_argument('--train_split', type=str, default='train_filtered',
                       help='Training data split')
    parser.add_argument('--eval_split', type=str, default='train_filtered',
                       help='Evaluation data split')
    parser.add_argument('--max_train_samples', type=int, default=100,
                       help='Maximum training samples')
    parser.add_argument('--max_eval_samples', type=int, default=50,
                       help='Maximum evaluation samples')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size')
    
    # Training parameters
    parser.add_argument('--num_epochs', type=int, default=5,
                       help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate')
    
    # Evaluation parameters
    parser.add_argument('--num_samples_per_question', type=int, default=3,
                       help='Number of samples per question')
    
    # Other parameters
    parser.add_argument('--use_mock', action='store_true', default=False,
                       help='Use mock components')
    parser.add_argument('--do_training', action='store_true', default=True,
                       help='Whether to perform training')
    parser.add_argument('--do_evaluation', action='store_true', default=True,
                       help='Whether to perform evaluation')
    parser.add_argument('--model_name', type=str, default='microsoft/DialoGPT-medium',
                       help='LLM model name')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cpu, cuda)')
    parser.add_argument('--hidden_dim', type=int, default=512,
                       help='Hidden dimension for GFlowNet policy')
    parser.add_argument('--num_layers', type=int, default=3,
                       help='Number of layers in GFlowNet policy')
    parser.add_argument('--dropout', type=float, default=0.1,
                       help='Dropout rate')
    parser.add_argument('--max_steps', type=int, default=10,
                       help='Maximum reasoning steps')
    parser.add_argument('--temperature', type=float, default=1.0,
                       help='Sampling temperature')
    parser.add_argument('--num_candidates', type=int, default=5,
                       help='Number of action candidates')
    parser.add_argument('--use_tb', action='store_true', default=True,
                       help='Use trajectory balance loss')
    parser.add_argument('--use_fm', action='store_true', default=False,
                       help='Use flow matching loss')
    parser.add_argument('--tb_weight', type=float, default=1.0,
                       help='Trajectory balance loss weight')
    parser.add_argument('--fm_weight', type=float, default=0.1,
                       help='Flow matching loss weight')
    parser.add_argument('--correctness_weight', type=float, default=1.0,
                       help='Correctness reward weight')
    parser.add_argument('--length_penalty', type=float, default=0.01,
                       help='Length penalty for reward')
    parser.add_argument('--diversity_bonus', type=float, default=0.1,
                       help='Diversity bonus for reward')
    
    return parser.parse_args()

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration file"""
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    return {}

def main():
    """Main function"""
    args = parse_arguments()
    
    # Setup logging
    # Create logs directory first
    os.makedirs('logs', exist_ok=True)
    
    # Generate timestamped log filename
    log_filename = f"logs/gfn_project_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    # Clear any existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ],
        force=True  # Force reconfiguration
    )
    logger = logging.getLogger(__name__)
    
    # Load configuration
    config = load_config(args.config)
    
    # Add command line arguments to config
    config.update(vars(args))
    
    logger.info("=" * 50)
    logger.info("Starting GFlowNet for StrategyQA Project")
    logger.info("=" * 50)
    logger.info(f"Run mode: {args.mode}")
    logger.info(f"Experiment name: {args.experiment_name}")
    logger.info(f"Using mock components: {args.use_mock}")
    
    # Create project instance
    project = GFlowNetProject(config)
    
    try:
        if args.mode == 'test':
            # Test mode: run all module tests
            logging.getLogger(__name__).info("Running test mode...")
            
            # Test data loading
            dataset, dataloader = project.load_dataset(max_samples=10)
            logging.getLogger(__name__).info("Data loading test passed")
            
            # Test training
            if args.do_training:
                history = project.train(dataloader, num_epochs=2)
                logging.getLogger(__name__).info("Training test passed")
            
            # Test evaluation
            if args.do_evaluation:
                results, comparison = project.evaluate(dataloader, num_samples_per_question=2)
                logging.getLogger(__name__).info("Evaluation test passed")
            
            logging.getLogger(__name__).info("All tests passed!")
            
        elif args.mode == 'train':
            # Training mode
            logging.getLogger(__name__).info("Running training mode...")
            dataset, dataloader = project.load_dataset(
                split=args.train_split,
                max_samples=args.max_train_samples
            )
            project.train(dataloader, num_epochs=args.num_epochs)
            
        elif args.mode == 'eval':
            # Evaluation mode
            logging.getLogger(__name__).info("Running evaluation mode...")
            dataset, dataloader = project.load_dataset(
                split=args.eval_split,
                max_samples=args.max_eval_samples
            )
            project.evaluate(dataloader, num_samples_per_question=args.num_samples_per_question)
            
        elif args.mode == 'experiment':
            # Complete experiment mode
            logging.getLogger(__name__).info("Running complete experiment mode...")
            project.run_experiment(args.experiment_name)
        
        logging.getLogger(__name__).info("=" * 50)
        logging.getLogger(__name__).info("Project run completed")
        logging.getLogger(__name__).info("=" * 50)
        
    except Exception as e:
        logging.getLogger(__name__).error(f"Project run failed: {e}")
        raise

if __name__ == "__main__":
    main()
