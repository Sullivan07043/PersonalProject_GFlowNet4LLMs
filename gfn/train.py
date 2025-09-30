import torch
import torch.optim as optim
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
from llm.promt import ReasoningChain
from gfn.model import MockGFlowNetPolicy, GFlowNetState
from gfn.sampler import MockGFlowNetSampler
from gfn.losses import GFlowNetLoss, MockRewardFunction

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GFlowNetTrainer:
    """GFlowNet Trainer"""
    
    def __init__(self,
                 policy_net,
                 llm_generator,
                 sampler,
                 loss_fn,
                 reward_fn,
                 learning_rate: float = 1e-4,
                 device: str = "cpu"):
        """
        Initialize trainer
        Args:
            policy_net: Policy network
            llm_generator: LLM generator
            sampler: Sampler
            loss_fn: Loss function
            reward_fn: Reward function
            learning_rate: Learning rate
            device: Device
        """
        self.policy_net = policy_net
        self.llm_generator = llm_generator
        self.sampler = sampler
        self.loss_fn = loss_fn
        self.reward_fn = reward_fn
        self.device = device
        
        # Optimizer
        if hasattr(policy_net, 'parameters'):
            self.optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
        else:
            self.optimizer = None
        
        # Training history
        self.train_history = {
            'losses': [],
            'rewards': [],
            'accuracies': []
        }
    
    def train_epoch(self, dataloader: DataLoader, epoch: int) -> Dict[str, float]:
        """
        Train one epoch
        Args:
            dataloader: Data loader
            epoch: Current epoch
        Returns:
            Training metrics
        """
        total_loss = 0.0
        total_reward = 0.0
        total_accuracy = 0.0
        num_batches = 0
        
        # Set training mode
        if hasattr(self.policy_net, 'train'):
            self.policy_net.train()
        
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch}")):
            try:
                # Get batch data
                questions = batch['questions']
                answers = batch['answer_texts']
                
                # Sample trajectories
                trajectories = self.sampler.batch_sample(questions, answers)
                
                # Compute loss
                batch_loss = 0.0
                batch_reward = 0.0
                batch_accuracy = 0.0
                
                for i, (final_state, trajectory_data) in enumerate(trajectories):
                    # Compute trajectory loss
                    trajectory_loss = self._compute_trajectory_loss(trajectory_data, final_state.reward)
                    batch_loss += trajectory_loss
                    
                    # Accumulate reward and accuracy
                    batch_reward += final_state.reward
                    batch_accuracy += 1.0 if final_state.reward > 0.5 else 0.0
                
                # Average
                batch_loss /= len(trajectories)
                batch_reward /= len(trajectories)
                batch_accuracy /= len(trajectories)
                
                # Backward pass
                if self.optimizer is not None:
                    self.optimizer.zero_grad()
                    batch_loss.backward()
                    self.optimizer.step()
                
                # Accumulate statistics
                total_loss += batch_loss.item()
                total_reward += batch_reward
                total_accuracy += batch_accuracy
                num_batches += 1
                
                # Log progress
                if batch_idx % 10 == 0:
                    logger.info(f"Batch {batch_idx}: Loss={batch_loss:.4f}, Reward={batch_reward:.4f}, Acc={batch_accuracy:.4f}")
                
            except Exception as e:
                logger.error(f"Training batch {batch_idx} failed: {e}")
                continue
        
        # Compute average metrics
        avg_loss = total_loss / max(num_batches, 1)
        avg_reward = total_reward / max(num_batches, 1)
        avg_accuracy = total_accuracy / max(num_batches, 1)
        
        return {
            'loss': avg_loss,
            'reward': avg_reward,
            'accuracy': avg_accuracy
        }
    
    def _compute_trajectory_loss(self, trajectory_data: List[Dict], terminal_reward: float) -> torch.Tensor:
        """Compute trajectory loss"""
        if not trajectory_data:
            return torch.tensor(0.0, requires_grad=True)
        
        # Extract trajectory information
        log_probs = [step['action_log_prob'] for step in trajectory_data]
        rewards = [0.1] * len(trajectory_data)  # Simple step rewards
        
        # Build trajectory data
        trajectory_dict = {
            'log_probs': log_probs,
            'rewards': rewards,
            'terminal_reward': terminal_reward
        }
        
        # Compute loss
        losses = self.loss_fn(trajectory_dict)
        return losses['total_loss']
    
    def train(self, 
              dataloader: DataLoader, 
              num_epochs: int = 10,
              save_dir: str = "./checkpoints") -> Dict[str, List[float]]:
        """
        Complete training process
        Args:
            dataloader: Data loader
            num_epochs: Number of epochs
            save_dir: Save directory
        Returns:
            Training history
        """
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        logger.info(f"Starting training, {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            # Train one epoch
            metrics = self.train_epoch(dataloader, epoch)
            
            # Record history
            self.train_history['losses'].append(metrics['loss'])
            self.train_history['rewards'].append(metrics['reward'])
            self.train_history['accuracies'].append(metrics['accuracy'])
            
            # Print progress
            logger.info(f"Epoch {epoch}: Loss={metrics['loss']:.4f}, Reward={metrics['reward']:.4f}, Acc={metrics['accuracy']:.4f}")
            
            # Save checkpoint
            if epoch % 5 == 0:
                self.save_checkpoint(save_dir, epoch, metrics)
        
        # Save final model
        self.save_checkpoint(save_dir, num_epochs - 1, metrics, is_final=True)
        
        return self.train_history
    
    def save_checkpoint(self, save_dir: str, epoch: int, metrics: Dict, is_final: bool = False):
        """Save checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'metrics': metrics,
            'train_history': self.train_history
        }
        
        if is_final:
            filename = "final_model.json"
        else:
            filename = f"checkpoint_epoch_{epoch}.json"
        
        filepath = os.path.join(save_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(checkpoint, f, indent=2)
        
        logger.info(f"Checkpoint saved: {filepath}")
    
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate model"""
        total_reward = 0.0
        total_accuracy = 0.0
        num_samples = 0
        
        # Set evaluation mode
        if hasattr(self.policy_net, 'eval'):
            self.policy_net.eval()
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluation"):
                questions = batch['questions']
                answers = batch['answer_texts']
                
                # Sample trajectories
                trajectories = self.sampler.batch_sample(questions, answers)
                
                for final_state, _ in trajectories:
                    total_reward += final_state.reward
                    total_accuracy += 1.0 if final_state.reward > 0.5 else 0.0
                    num_samples += 1
        
        avg_reward = total_reward / max(num_samples, 1)
        avg_accuracy = total_accuracy / max(num_samples, 1)
        
        return {
            'reward': avg_reward,
            'accuracy': avg_accuracy
        }

class MockGFlowNetTrainer:
    """Mock GFlowNet Trainer (for testing)"""
    
    def __init__(self):
        self.policy_net = MockGFlowNetPolicy()
        self.llm_generator = MockLLMGenerator()
        self.sampler = MockGFlowNetSampler()
        self.loss_fn = GFlowNetLoss()
        self.reward_fn = MockRewardFunction()
        
        self.train_history = {
            'losses': [],
            'rewards': [],
            'accuracies': []
        }
    
    def train_epoch(self, dataloader: DataLoader, epoch: int) -> Dict[str, float]:
        """Mock training one epoch"""
        total_loss = 0.0
        total_reward = 0.0
        total_accuracy = 0.0
        num_batches = 0
        
        for batch in dataloader:
            questions = batch['questions']
            answers = batch['answer_texts']
            
            # Mock sampling
            trajectories = self.sampler.batch_sample(questions, answers)
            
            batch_loss = 0.0
            batch_reward = 0.0
            batch_accuracy = 0.0
            
            for final_state, _ in trajectories:
                # Mock loss computation
                batch_loss += 0.5 + (1.0 - final_state.reward) * 0.5
                batch_reward += final_state.reward
                batch_accuracy += 1.0 if final_state.reward > 0.5 else 0.0
            
            batch_loss /= len(trajectories)
            batch_reward /= len(trajectories)
            batch_accuracy /= len(trajectories)
            
            total_loss += batch_loss
            total_reward += batch_reward
            total_accuracy += batch_accuracy
            num_batches += 1
        
        avg_loss = total_loss / max(num_batches, 1)
        avg_reward = total_reward / max(num_batches, 1)
        avg_accuracy = total_accuracy / max(num_batches, 1)
        
        return {
            'loss': avg_loss,
            'reward': avg_reward,
            'accuracy': avg_accuracy
        }
    
    def train(self, dataloader: DataLoader, num_epochs: int = 5) -> Dict[str, List[float]]:
        """Mock complete training process"""
        logger.info(f"Starting mock training, {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            metrics = self.train_epoch(dataloader, epoch)
            
            self.train_history['losses'].append(metrics['loss'])
            self.train_history['rewards'].append(metrics['reward'])
            self.train_history['accuracies'].append(metrics['accuracy'])
            
            logger.info(f"Epoch {epoch}: Loss={metrics['loss']:.4f}, Reward={metrics['reward']:.4f}, Acc={metrics['accuracy']:.4f}")
        
        return self.train_history

# Usage example
if __name__ == "__main__":
    # Create dataset
    dataset = StrategyQADataset(split="train_filtered")
    processed_data = dataset.preprocess(max_samples=50)  # Small sample test
    dataloader = dataset.get_dataloader(batch_size=4, shuffle=True)
    
    # Create trainer
    trainer = MockGFlowNetTrainer()
    
    # Start training
    history = trainer.train(dataloader, num_epochs=3)
    
    print("\nTraining completed!")
    print(f"Final loss: {history['losses'][-1]:.4f}")
    print(f"Final reward: {history['rewards'][-1]:.4f}")
    print(f"Final accuracy: {history['accuracies'][-1]:.4f}")
    
    # Plot training curves (simple version)
    print("\nTraining history:")
    for i, (loss, reward, acc) in enumerate(zip(history['losses'], history['rewards'], history['accuracies'])):
        print(f"Epoch {i}: Loss={loss:.4f}, Reward={reward:.4f}, Acc={acc:.4f}")
