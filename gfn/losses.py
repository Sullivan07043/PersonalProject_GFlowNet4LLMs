import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any, Tuple, Optional
import math
import logging

logger = logging.getLogger(__name__)

class TrajectoryBalanceLoss(nn.Module):
    """Trajectory Balance Loss Function"""
    
    def __init__(self, 
                 log_z_init: float = 0.0,
                 log_z_target: float = 0.0,
                 tb_lambda: float = 1.0):
        """
        Initialize Trajectory Balance Loss
        Args:
            log_z_init: Initial state log Z value
            log_z_target: Target log Z value
            tb_lambda: Loss weight
        """
        super().__init__()
        self.log_z_init = nn.Parameter(torch.tensor(log_z_init))
        self.log_z_target = log_z_target
        self.tb_lambda = tb_lambda
    
    def forward(self, 
                trajectory_log_probs: List[torch.Tensor],
                trajectory_rewards: List[float],
                terminal_reward: float) -> torch.Tensor:
        """
        Compute Trajectory Balance Loss
        Args:
            trajectory_log_probs: Log probabilities for each step in trajectory
            trajectory_rewards: Rewards for each step in trajectory
            terminal_reward: Terminal state reward
        Returns:
            Loss value
        """
        if not trajectory_log_probs:
            return torch.tensor(0.0, requires_grad=True)
        
        # Compute total log probability of trajectory
        total_log_prob = sum(trajectory_log_probs)
        
        # Compute reward
        total_reward = sum(trajectory_rewards) + terminal_reward
        
        # Trajectory Balance Loss
        # L_TB = (log Z + log P_F(τ) - log R(τ))^2
        loss = (self.log_z_init + total_log_prob - math.log(max(total_reward, 1e-8))) ** 2
        
        return self.tb_lambda * loss

class FlowMatchingLoss(nn.Module):
    """Flow Matching Loss Function"""
    
    def __init__(self, lambda_fm: float = 1.0):
        """
        Initialize Flow Matching Loss
        Args:
            lambda_fm: Loss weight
        """
        super().__init__()
        self.lambda_fm = lambda_fm
    
    def forward(self, 
                state_flows: List[torch.Tensor],
                action_flows: List[torch.Tensor],
                rewards: List[float]) -> torch.Tensor:
        """
        Compute Flow Matching Loss
        Args:
            state_flows: State flow values
            action_flows: Action flow values
            rewards: Reward values
        Returns:
            Loss value
        """
        if not state_flows or not action_flows:
            return torch.tensor(0.0, requires_grad=True)
        
        total_loss = 0.0
        
        for i, (state_flow, action_flow, reward) in enumerate(zip(state_flows, action_flows, rewards)):
            # Flow matching constraint: F(s) = Σ_a F(s,a) + R(s)
            flow_constraint = state_flow - (action_flow.sum() + reward)
            total_loss += flow_constraint ** 2
        
        return self.lambda_fm * total_loss

class GFlowNetLoss(nn.Module):
    """GFlowNet Comprehensive Loss Function"""
    
    def __init__(self, 
                 use_tb: bool = True,
                 use_fm: bool = False,
                 tb_weight: float = 1.0,
                 fm_weight: float = 0.1):
        """
        Initialize GFlowNet Loss
        Args:
            use_tb: Whether to use trajectory balance loss
            use_fm: Whether to use flow matching loss
            tb_weight: Trajectory balance loss weight
            fm_weight: Flow matching loss weight
        """
        super().__init__()
        self.use_tb = use_tb
        self.use_fm = use_fm
        self.tb_weight = tb_weight
        self.fm_weight = fm_weight
        
        if use_tb:
            self.tb_loss = TrajectoryBalanceLoss()
        if use_fm:
            self.fm_loss = FlowMatchingLoss()
    
    def forward(self, 
                trajectory_data: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Compute comprehensive loss
        Args:
            trajectory_data: Trajectory data dictionary
        Returns:
            Loss dictionary
        """
        losses = {}
        
        if self.use_tb:
            tb_loss = self.tb_loss(
                trajectory_data.get('log_probs', []),
                trajectory_data.get('rewards', []),
                trajectory_data.get('terminal_reward', 0.0)
            )
            losses['tb_loss'] = self.tb_weight * tb_loss
        
        if self.use_fm:
            fm_loss = self.fm_loss(
                trajectory_data.get('state_flows', []),
                trajectory_data.get('action_flows', []),
                trajectory_data.get('rewards', [])
            )
            losses['fm_loss'] = self.fm_weight * fm_loss
        
        # Compute total loss
        total_loss = sum(losses.values())
        losses['total_loss'] = total_loss
        
        return losses

class RewardFunction:
    """Reward Function Class"""
    
    def __init__(self, 
                 correctness_weight: float = 1.0,
                 length_penalty: float = 0.01,
                 diversity_bonus: float = 0.1):
        """
        Initialize reward function
        Args:
            correctness_weight: Correctness weight
            length_penalty: Length penalty
            diversity_bonus: Diversity bonus
        """
        self.correctness_weight = correctness_weight
        self.length_penalty = length_penalty
        self.diversity_bonus = diversity_bonus
    
    def compute_reward(self, 
                      predicted_answer: str,
                      gold_answer: str,
                      reasoning_length: int,
                      diversity_score: float = 0.0) -> float:
        """
        Compute reward
        Args:
            predicted_answer: Predicted answer
            gold_answer: Gold answer
            reasoning_length: Reasoning length
            diversity_score: Diversity score
        Returns:
            Reward value
        """
        # Correctness reward
        correctness_reward = 1.0 if predicted_answer.lower() == gold_answer.lower() else 0.0
        
        # Length penalty
        length_penalty = self.length_penalty * reasoning_length
        
        # Diversity reward
        diversity_reward = self.diversity_bonus * diversity_score
        
        # Total reward
        total_reward = (self.correctness_weight * correctness_reward - 
                       length_penalty + 
                       diversity_reward)
        
        return max(total_reward, 0.0)  # Ensure reward is non-negative
    
    def compute_step_reward(self, 
                           step: str,
                           previous_steps: List[str]) -> float:
        """
        Compute single step reward
        Args:
            step: Current step
            previous_steps: Previous steps
        Returns:
            Step reward
        """
        # Simple step reward: based on step length and repetition
        step_length = len(step.split())
        
        # Repetition penalty
        repetition_penalty = 0.0
        for prev_step in previous_steps:
            if step.lower() in prev_step.lower() or prev_step.lower() in step.lower():
                repetition_penalty += 0.1
        
        # Step reward
        step_reward = min(step_length / 10.0, 1.0) - repetition_penalty
        
        return max(step_reward, 0.0)

class MockRewardFunction:
    """Mock Reward Function (for testing)"""
    
    def __init__(self):
        pass
    
    def compute_reward(self, 
                      predicted_answer: str,
                      gold_answer: str,
                      reasoning_length: int,
                      diversity_score: float = 0.0) -> float:
        """Mock compute reward"""
        # Simple correctness reward
        if predicted_answer.lower() == gold_answer.lower():
            return 1.0
        else:
            return 0.0
    
    def compute_step_reward(self, 
                           step: str,
                           previous_steps: List[str]) -> float:
        """Mock compute step reward"""
        # Simple step reward
        return 0.1

# Usage example
if __name__ == "__main__":
    # Test trajectory balance loss
    tb_loss = TrajectoryBalanceLoss()
    
    # Mock trajectory data
    trajectory_log_probs = [torch.tensor(-0.5), torch.tensor(-0.3), torch.tensor(-0.2)]
    trajectory_rewards = [0.1, 0.1, 0.1]
    terminal_reward = 1.0
    
    loss = tb_loss(trajectory_log_probs, trajectory_rewards, terminal_reward)
    print(f"Trajectory Balance Loss: {loss}")
    
    # Test reward function
    reward_fn = MockRewardFunction()
    
    # Test final reward
    final_reward = reward_fn.compute_reward("Yes", "Yes", 5, 0.5)
    print(f"Final Reward: {final_reward}")
    
    # Test step reward
    step_reward = reward_fn.compute_step_reward("This is a reasoning step", [])
    print(f"Step Reward: {step_reward}")
    
    # Test comprehensive loss
    gfn_loss = GFlowNetLoss(use_tb=True, use_fm=False)
    
    trajectory_data = {
        'log_probs': trajectory_log_probs,
        'rewards': trajectory_rewards,
        'terminal_reward': terminal_reward
    }
    
    losses = gfn_loss(trajectory_data)
    print(f"Comprehensive Loss: {losses}")
    
    print("\nAll tests completed!")
