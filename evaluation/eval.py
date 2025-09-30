import torch
from torch.utils.data import DataLoader
import logging
import os
import json
from typing import Dict, List, Any, Optional
from tqdm import tqdm
import sys

# Add project path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from data.dataset import StrategyQADataset
from llm.generate import MockLLMGenerator
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'gfn'))
from gfn.sampler import MockGFlowNetSampler
from .metrics import MockEvaluationMetrics, ReasoningDiversity

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GFlowNetEvaluator:
    """GFlowNet Evaluator"""
    
    def __init__(self, 
                 sampler,
                 llm_generator,
                 metrics,
                 diversity_metrics):
        """
        Initialize evaluator
        Args:
            sampler: Sampler
            llm_generator: LLM generator
            metrics: Metrics calculator
            diversity_metrics: Diversity calculator
        """
        self.sampler = sampler
        self.llm_generator = llm_generator
        self.metrics_calculator = metrics
        self.diversity_calculator = diversity_metrics
    
    def evaluate_dataset(self, 
                        dataloader: DataLoader, 
                        num_samples_per_question: int = 5) -> Dict[str, Any]:
        """
        Evaluate entire dataset
        Args:
            dataloader: Data loader
            num_samples_per_question: Number of samples per question
        Returns:
            Evaluation results
        """
        logger.info("Starting dataset evaluation...")
        
        # Reset metrics
        self.metrics_calculator.reset()
        
        all_reasoning_chains = []
        all_questions = []
        all_ground_truths = []
        
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating batches")):
            questions = batch['questions']
            ground_truths = batch['answer_texts']
            
            batch_reasoning_chains = []
            batch_predictions = []
            batch_rewards = []
            batch_reasoning_lengths = []
            batch_diversity_scores = []
            
            for i, (question, ground_truth) in enumerate(zip(questions, ground_truths)):
                # Generate multiple reasoning chains for each question
                question_chains = []
                question_predictions = []
                question_rewards = []
                question_lengths = []
                
                for _ in range(num_samples_per_question):
                    # Sample trajectory
                    final_state, trajectory_data = self.sampler.sample_trajectory(question, ground_truth)
                    
                    # Generate complete reasoning chain
                    chain = self.llm_generator.generate_reasoning_chain(question)
                    
                    question_chains.append(chain.steps)
                    question_predictions.append(chain.final_answer)
                    question_rewards.append(final_state.reward)
                    question_lengths.append(len(chain.steps))
                
                # Compute diversity score
                diversity_score = self.diversity_calculator.compute_ngram_diversity(question_chains)
                
                # Select best prediction (based on reward)
                best_idx = max(range(len(question_rewards)), key=lambda i: question_rewards[i])
                best_prediction = question_predictions[best_idx]
                best_reward = question_rewards[best_idx]
                best_length = question_lengths[best_idx]
                
                # Add to batch data
                batch_reasoning_chains.extend(question_chains)
                batch_predictions.append(best_prediction)
                batch_rewards.append(best_reward)
                batch_reasoning_lengths.append(best_length)
                batch_diversity_scores.append(diversity_score)
                
                # Record all data
                all_reasoning_chains.extend(question_chains)
                all_questions.append(question)
                all_ground_truths.append(ground_truth)
            
            # Add batch data to metrics calculator
            self.metrics_calculator.add_batch(
                batch_predictions,
                ground_truths,
                batch_rewards,
                batch_reasoning_lengths,
                batch_diversity_scores
            )
        
        # Compute all metrics
        metrics = self.metrics_calculator.compute_all_metrics()
        
        # Compute overall diversity
        overall_diversity = self.diversity_calculator.compute_ngram_diversity(all_reasoning_chains)
        semantic_diversity = self.diversity_calculator.compute_semantic_diversity(all_reasoning_chains)
        structural_diversity = self.diversity_calculator.compute_structural_diversity(all_reasoning_chains)
        
        # Build evaluation results
        evaluation_results = {
            'metrics': metrics,
            'diversity': {
                'ngram_diversity': overall_diversity,
                'semantic_diversity': semantic_diversity,
                'structural_diversity': structural_diversity
            },
            'num_questions': len(all_questions),
            'num_samples_per_question': num_samples_per_question,
            'total_samples': len(all_questions) * num_samples_per_question
        }
        
        return evaluation_results
    
    def compare_with_baseline(self, 
                             dataloader: DataLoader,
                             baseline_method: str = "random") -> Dict[str, Any]:
        """
        Compare with baseline method
        Args:
            dataloader: Data loader
            baseline_method: Baseline method ("random", "greedy")
        Returns:
            Comparison results
        """
        logger.info(f"Comparing with baseline method {baseline_method}...")
        
        # Evaluate GFlowNet method
        gfn_results = self.evaluate_dataset(dataloader)
        
        # Evaluate baseline method
        baseline_results = self._evaluate_baseline(dataloader, baseline_method)
        
        # Compute improvements
        improvements = {}
        for metric in gfn_results['metrics']:
            gfn_value = gfn_results['metrics'][metric]
            baseline_value = baseline_results['metrics'][metric]
            improvement = (gfn_value - baseline_value) / max(baseline_value, 1e-8) * 100
            improvements[metric] = improvement
        
        return {
            'gfn_results': gfn_results,
            'baseline_results': baseline_results,
            'improvements': improvements
        }
    
    def _evaluate_baseline(self, dataloader: DataLoader, method: str) -> Dict[str, Any]:
        """Evaluate baseline method"""
        metrics_calculator = MockEvaluationMetrics()
        metrics_calculator.reset()
        
        for batch in dataloader:
            questions = batch['questions']
            ground_truths = batch['answer_texts']
            
            predictions = []
            rewards = []
            reasoning_lengths = []
            diversity_scores = []
            
            for question, ground_truth in zip(questions, ground_truths):
                if method == "random":
                    # Random prediction
                    import random
                    prediction = "Yes" if random.random() > 0.5 else "No"
                    reward = 1.0 if prediction.lower() == ground_truth.lower() else 0.0
                    reasoning_length = random.randint(2, 5)
                    diversity_score = 0.5
                elif method == "greedy":
                    # Greedy prediction (always predict "Yes")
                    prediction = "Yes"
                    reward = 1.0 if prediction.lower() == ground_truth.lower() else 0.0
                    reasoning_length = 3
                    diversity_score = 0.3
                else:
                    raise ValueError(f"Unknown baseline method: {method}")
                
                predictions.append(prediction)
                rewards.append(reward)
                reasoning_lengths.append(reasoning_length)
                diversity_scores.append(diversity_score)
            
            metrics_calculator.add_batch(predictions, ground_truths, rewards, reasoning_lengths, diversity_scores)
        
        metrics = metrics_calculator.compute_all_metrics()
        
        return {
            'metrics': metrics,
            'method': method
        }
    
    def save_results(self, results: Dict[str, Any], save_path: str):
        """Save evaluation results"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Evaluation results saved to: {save_path}")

class MockGFlowNetEvaluator:
    """Mock GFlowNet Evaluator (for testing)"""
    
    def __init__(self):
        self.sampler = MockGFlowNetSampler()
        self.llm_generator = MockLLMGenerator()
        self.metrics_calculator = MockEvaluationMetrics()
        self.diversity_calculator = ReasoningDiversity()
    
    def evaluate_dataset(self, dataloader: DataLoader, num_samples_per_question: int = 3) -> Dict[str, Any]:
        """Mock evaluate dataset"""
        logger.info("Starting mock dataset evaluation...")
        
        self.metrics_calculator.reset()
        
        all_reasoning_chains = []
        
        for batch in tqdm(dataloader, desc="Evaluating batches"):
            questions = batch['questions']
            ground_truths = batch['answer_texts']
            
            batch_predictions = []
            batch_rewards = []
            batch_reasoning_lengths = []
            batch_diversity_scores = []
            
            for question, ground_truth in zip(questions, ground_truths):
                # Mock generate reasoning chains
                question_chains = []
                for _ in range(num_samples_per_question):
                    chain = self.llm_generator.generate_reasoning_chain(question)
                    question_chains.append(chain.steps)
                
                # Mock prediction and reward
                prediction = "Yes" if "swastika" in question.lower() else "No"
                reward = 1.0 if prediction.lower() == ground_truth.lower() else 0.0
                reasoning_length = len(question_chains[0])
                diversity_score = 0.7
                
                batch_predictions.append(prediction)
                batch_rewards.append(reward)
                batch_reasoning_lengths.append(reasoning_length)
                batch_diversity_scores.append(diversity_score)
                
                all_reasoning_chains.extend(question_chains)
            
            self.metrics_calculator.add_batch(
                batch_predictions,
                ground_truths,
                batch_rewards,
                batch_reasoning_lengths,
                batch_diversity_scores
            )
        
        # Compute metrics
        metrics = self.metrics_calculator.compute_all_metrics()
        
        # Compute diversity
        overall_diversity = self.diversity_calculator.compute_ngram_diversity(all_reasoning_chains)
        
        return {
            'metrics': metrics,
            'diversity': {
                'ngram_diversity': overall_diversity,
                'semantic_diversity': 0.6,
                'structural_diversity': 0.8
            },
            'num_questions': len(questions),
            'num_samples_per_question': num_samples_per_question
        }
    
    def compare_with_baseline(self, dataloader: DataLoader, baseline_method: str = "random") -> Dict[str, Any]:
        """Mock compare with baseline"""
        gfn_results = self.evaluate_dataset(dataloader)
        
        # Mock baseline results
        baseline_results = {
            'metrics': {
                'accuracy': 0.5,
                'average_reward': 0.5,
                'average_reasoning_length': 3.0,
                'diversity_score': 0.4
            },
            'method': baseline_method
        }
        
        # Compute improvements
        improvements = {}
        for metric in gfn_results['metrics']:
            gfn_value = gfn_results['metrics'][metric]
            baseline_value = baseline_results['metrics'][metric]
            improvement = (gfn_value - baseline_value) / max(baseline_value, 1e-8) * 100
            improvements[metric] = improvement
        
        return {
            'gfn_results': gfn_results,
            'baseline_results': baseline_results,
            'improvements': improvements
        }

# Usage example
if __name__ == "__main__":
    # Create dataset
    dataset = StrategyQADataset(split="train_filtered")
    processed_data = dataset.preprocess(max_samples=20)  # Small sample test
    dataloader = dataset.get_dataloader(batch_size=4, shuffle=False)
    
    # Create evaluator
    evaluator = MockGFlowNetEvaluator()
    
    # Evaluate dataset
    results = evaluator.evaluate_dataset(dataloader, num_samples_per_question=2)
    
    print("Evaluation results:")
    print(f"Accuracy: {results['metrics']['accuracy']:.4f}")
    print(f"Average reward: {results['metrics']['average_reward']:.4f}")
    print(f"Average reasoning length: {results['metrics']['average_reasoning_length']:.4f}")
    print(f"Diversity score: {results['metrics']['diversity_score']:.4f}")
    
    print(f"\nDiversity metrics:")
    print(f"N-gram diversity: {results['diversity']['ngram_diversity']:.4f}")
    print(f"Semantic diversity: {results['diversity']['semantic_diversity']:.4f}")
    print(f"Structural diversity: {results['diversity']['structural_diversity']:.4f}")
    
    # Compare with baseline
    comparison = evaluator.compare_with_baseline(dataloader, baseline_method="random")
    
    print(f"\nComparison with random baseline:")
    for metric, improvement in comparison['improvements'].items():
        print(f"{metric}: {improvement:+.2f}%")
