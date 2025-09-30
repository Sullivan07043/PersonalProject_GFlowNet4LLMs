import torch
import numpy as np
from typing import List, Dict, Any, Tuple
from collections import Counter
import logging

logger = logging.getLogger(__name__)

class EvaluationMetrics:
    """Evaluation metrics calculation class"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all metrics"""
        self.predictions = []
        self.ground_truths = []
        self.rewards = []
        self.reasoning_lengths = []
        self.diversity_scores = []
    
    def add_batch(self, 
                  predictions: List[str], 
                  ground_truths: List[str],
                  rewards: List[float] = None,
                  reasoning_lengths: List[int] = None,
                  diversity_scores: List[float] = None):
        """
        Add a batch of evaluation data
        Args:
            predictions: List of predicted answers
            ground_truths: List of ground truth answers
            rewards: List of rewards
            reasoning_lengths: List of reasoning lengths
            diversity_scores: List of diversity scores
        """
        self.predictions.extend(predictions)
        self.ground_truths.extend(ground_truths)
        
        if rewards:
            self.rewards.extend(rewards)
        if reasoning_lengths:
            self.reasoning_lengths.extend(reasoning_lengths)
        if diversity_scores:
            self.diversity_scores.extend(diversity_scores)
    
    def compute_accuracy(self) -> float:
        """Compute accuracy"""
        if not self.predictions or not self.ground_truths:
            return 0.0
        
        correct = sum(1 for pred, gt in zip(self.predictions, self.ground_truths) 
                     if pred.lower() == gt.lower())
        return correct / len(self.predictions)
    
    def compute_precision_recall_f1(self) -> Dict[str, float]:
        """Compute precision, recall and F1 score"""
        if not self.predictions or not self.ground_truths:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        
        # Convert to binary labels
        pred_binary = [1 if pred.lower() == 'yes' else 0 for pred in self.predictions]
        gt_binary = [1 if gt.lower() == 'yes' else 0 for gt in self.ground_truths]
        
        # Compute TP, FP, FN
        tp = sum(1 for p, g in zip(pred_binary, gt_binary) if p == 1 and g == 1)
        fp = sum(1 for p, g in zip(pred_binary, gt_binary) if p == 1 and g == 0)
        fn = sum(1 for p, g in zip(pred_binary, gt_binary) if p == 0 and g == 1)
        
        # Compute metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def compute_average_reward(self) -> float:
        """Compute average reward"""
        if not self.rewards:
            return 0.0
        return sum(self.rewards) / len(self.rewards)
    
    def compute_average_reasoning_length(self) -> float:
        """Compute average reasoning length"""
        if not self.reasoning_lengths:
            return 0.0
        return sum(self.reasoning_lengths) / len(self.reasoning_lengths)
    
    def compute_diversity_score(self) -> float:
        """Compute diversity score"""
        if not self.diversity_scores:
            return 0.0
        return sum(self.diversity_scores) / len(self.diversity_scores)
    
    def compute_all_metrics(self) -> Dict[str, float]:
        """Compute all metrics"""
        return {
            'accuracy': self.compute_accuracy(),
            'precision': self.compute_precision_recall_f1()['precision'],
            'recall': self.compute_precision_recall_f1()['recall'],
            'f1': self.compute_precision_recall_f1()['f1'],
            'average_reward': self.compute_average_reward(),
            'average_reasoning_length': self.compute_average_reasoning_length(),
            'diversity_score': self.compute_diversity_score()
        }

class ReasoningDiversity:
    """Reasoning diversity calculation class"""
    
    def __init__(self):
        pass
    
    def compute_ngram_diversity(self, reasoning_chains: List[List[str]], n: int = 2) -> float:
        """
        Compute n-gram diversity
        Args:
            reasoning_chains: List of reasoning chains
            n: n-gram size
        Returns:
            Diversity score
        """
        if not reasoning_chains:
            return 0.0
        
        all_ngrams = set()
        total_ngrams = 0
        
        for chain in reasoning_chains:
            for step in chain:
                words = step.lower().split()
                for i in range(len(words) - n + 1):
                    ngram = ' '.join(words[i:i+n])
                    all_ngrams.add(ngram)
                    total_ngrams += 1
        
        if total_ngrams == 0:
            return 0.0
        
        return len(all_ngrams) / total_ngrams
    
    def compute_semantic_diversity(self, reasoning_chains: List[List[str]]) -> float:
        """
        Compute semantic diversity (simplified version)
        Args:
            reasoning_chains: List of reasoning chains
        Returns:
            Semantic diversity score
        """
        if not reasoning_chains:
            return 0.0
        
        # Simple vocabulary diversity calculation
        all_words = set()
        total_words = 0
        
        for chain in reasoning_chains:
            for step in chain:
                words = step.lower().split()
                all_words.update(words)
                total_words += len(words)
        
        if total_words == 0:
            return 0.0
        
        return len(all_words) / total_words
    
    def compute_structural_diversity(self, reasoning_chains: List[List[str]]) -> float:
        """
        Compute structural diversity
        Args:
            reasoning_chains: List of reasoning chains
        Returns:
            Structural diversity score
        """
        if not reasoning_chains:
            return 0.0
        
        # Compute reasoning chain length distribution
        lengths = [len(chain) for chain in reasoning_chains]
        
        if not lengths:
            return 0.0
        
        # Compute entropy of length distribution
        length_counts = Counter(lengths)
        total_chains = len(reasoning_chains)
        
        entropy = 0.0
        for count in length_counts.values():
            prob = count / total_chains
            if prob > 0:
                entropy -= prob * np.log2(prob)
        
        return entropy

class MockEvaluationMetrics:
    """Mock evaluation metrics (for testing)"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset metrics"""
        self.predictions = []
        self.ground_truths = []
        self.rewards = []
        self.reasoning_lengths = []
        self.diversity_scores = []
    
    def add_batch(self, 
                  predictions: List[str], 
                  ground_truths: List[str],
                  rewards: List[float] = None,
                  reasoning_lengths: List[int] = None,
                  diversity_scores: List[float] = None):
        """Add batch data"""
        self.predictions.extend(predictions)
        self.ground_truths.extend(ground_truths)
        
        if rewards:
            self.rewards.extend(rewards)
        if reasoning_lengths:
            self.reasoning_lengths.extend(reasoning_lengths)
        if diversity_scores:
            self.diversity_scores.extend(diversity_scores)
    
    def compute_accuracy(self) -> float:
        """Compute accuracy"""
        if not self.predictions or not self.ground_truths:
            return 0.0
        
        correct = sum(1 for pred, gt in zip(self.predictions, self.ground_truths) 
                     if pred.lower() == gt.lower())
        return correct / len(self.predictions)
    
    def compute_average_reward(self) -> float:
        """Compute average reward"""
        if not self.rewards:
            return 0.0
        return sum(self.rewards) / len(self.rewards)
    
    def compute_average_reasoning_length(self) -> float:
        """Compute average reasoning length"""
        if not self.reasoning_lengths:
            return 0.0
        return sum(self.reasoning_lengths) / len(self.reasoning_lengths)
    
    def compute_diversity_score(self) -> float:
        """Compute diversity score"""
        if not self.diversity_scores:
            return 0.0
        return sum(self.diversity_scores) / len(self.diversity_scores)
    
    def compute_all_metrics(self) -> Dict[str, float]:
        """Compute all metrics"""
        return {
            'accuracy': self.compute_accuracy(),
            'average_reward': self.compute_average_reward(),
            'average_reasoning_length': self.compute_average_reasoning_length(),
            'diversity_score': self.compute_diversity_score()
        }

# Usage example
if __name__ == "__main__":
    # Test evaluation metrics
    metrics = MockEvaluationMetrics()
    
    # Add test data
    predictions = ["Yes", "No", "Yes", "No", "Yes"]
    ground_truths = ["Yes", "Yes", "No", "No", "Yes"]
    rewards = [1.0, 0.0, 0.0, 1.0, 1.0]
    reasoning_lengths = [3, 2, 4, 3, 2]
    diversity_scores = [0.8, 0.6, 0.9, 0.7, 0.8]
    
    metrics.add_batch(predictions, ground_truths, rewards, reasoning_lengths, diversity_scores)
    
    # Compute metrics
    all_metrics = metrics.compute_all_metrics()
    
    print("Evaluation metrics:")
    for metric, value in all_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Test diversity calculation
    diversity_calculator = ReasoningDiversity()
    
    reasoning_chains = [
        ["Step 1: Understanding", "Step 2: Analysis", "Step 3: Conclusion"],
        ["Step 1: Question analysis", "Step 2: Research", "Step 3: Answer"],
        ["Step 1: Understanding", "Step 2: Analysis", "Step 3: Conclusion"]
    ]
    
    ngram_diversity = diversity_calculator.compute_ngram_diversity(reasoning_chains, n=2)
    semantic_diversity = diversity_calculator.compute_semantic_diversity(reasoning_chains)
    structural_diversity = diversity_calculator.compute_structural_diversity(reasoning_chains)
    
    print(f"\nDiversity metrics:")
    print(f"  N-gram diversity: {ngram_diversity:.4f}")
    print(f"  Semantic diversity: {semantic_diversity:.4f}")
    print(f"  Structural diversity: {structural_diversity:.4f}")
