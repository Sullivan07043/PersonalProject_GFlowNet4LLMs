import torch
import random
from typing import List, Dict, Any, Tuple, Optional
import logging
from model import GFlowNetState, MockGFlowNetPolicy
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from llm.generate import MockLLMGenerator
from llm.promt import ReasoningChain

logger = logging.getLogger(__name__)

class GFlowNetSampler:
    """GFlowNet Sampler"""
    
    def __init__(self, 
                 policy_net,
                 llm_generator,
                 max_steps: int = 10,
                 temperature: float = 1.0,
                 num_candidates: int = 5):
        """
        Initialize GFlowNet Sampler
        Args:
            policy_net: Policy network
            llm_generator: LLM generator
            max_steps: Maximum reasoning steps
            temperature: Sampling temperature
            num_candidates: Number of candidate actions
        """
        self.policy_net = policy_net
        self.llm_generator = llm_generator
        self.max_steps = max_steps
        self.temperature = temperature
        self.num_candidates = num_candidates
    
    def sample_trajectory(self, 
                         question: str, 
                         gold_answer: str = None) -> Tuple[GFlowNetState, List[Dict[str, Any]]]:
        """
        Sample a complete trajectory
        Args:
            question: Question
            gold_answer: Gold answer (for computing reward)
        Returns:
            Terminal state and trajectory data
        """
        # Initialize state
        state = GFlowNetState(question)
        trajectory_data = []
        
        for step_idx in range(self.max_steps):
            # Generate candidate actions
            candidates = self.llm_generator.generate_step_candidates(
                ReasoningChain(question, state.steps), 
                self.num_candidates
            )
            
            # Validate candidates
            if not candidates:
                logger.warning(f"No candidates generated at step {step_idx}, terminating")
                break
            
            if len(candidates) != self.num_candidates:
                logger.warning(f"Expected {self.num_candidates} candidates, got {len(candidates)}")
            
            # Use policy network to select action
            action_idx, action_probs = self.policy_net.sample_action(
                state.get_text_representation(),
                candidates,
                self.temperature
            )
            
            if action_idx == -1:
                logger.warning(f"Policy network failed to select action at step {step_idx}, terminating")
                break
            
            # Validate action index
            if action_idx >= len(candidates):
                logger.error(f"Action index {action_idx} out of bounds for {len(candidates)} candidates")
                action_idx = 0  # Use first candidate as fallback
            
            # Execute action
            selected_action = candidates[action_idx]
            state.add_step(selected_action)
            
            # Record trajectory data
            # Validate action_probs and action_idx consistency
            if action_probs.numel() == 0:
                logger.warning(f"Empty action_probs at step {step_idx}")
                action_log_prob = torch.tensor(0.0)
            elif action_idx >= len(action_probs):
                logger.warning(f"Action index {action_idx} out of bounds for action_probs of length {len(action_probs)}")
                action_log_prob = torch.tensor(0.0)
            else:
                action_log_prob = torch.log(action_probs[action_idx])
            
            step_data = {
                'step_idx': step_idx,
                'state': state.get_text_representation(),
                'candidates': candidates,
                'selected_action': selected_action,
                'action_idx': action_idx,
                'action_probs': action_probs,
                'action_log_prob': action_log_prob
            }
            trajectory_data.append(step_data)
            
            # Check if should terminate
            if self._should_terminate(state, step_idx):
                break
        
        # Generate final answer
        final_answer = self._generate_final_answer(state)
        state.set_terminal()
        
        # Compute reward
        if gold_answer:
            reward = self._compute_reward(final_answer, gold_answer, len(state.steps))
            state.reward = reward
        
        return state, trajectory_data
    
    def _should_terminate(self, state: GFlowNetState, step_idx: int) -> bool:
        """Check if should terminate"""
        # Reached maximum steps
        if step_idx >= self.max_steps - 1:
            return True
        
        # Check termination probability
        if hasattr(self.policy_net, 'get_termination_probability'):
            term_prob = self.policy_net.get_termination_probability(state.get_text_representation())
            return random.random() < term_prob
        
        return False
    
    def _generate_final_answer(self, state: GFlowNetState) -> str:
        """Generate final answer"""
        # Use LLM to generate complete reasoning chain
        chain = self.llm_generator.generate_reasoning_chain(state.question)
        return chain.final_answer if chain.final_answer else "Unknown"
    
    def _compute_reward(self, predicted_answer: str, gold_answer: str, reasoning_length: int) -> float:
        """Compute reward"""
        # Simple correctness reward
        if predicted_answer.lower() == gold_answer.lower():
            return 1.0
        else:
            return 0.0
    
    def batch_sample(self, 
                    questions: List[str], 
                    gold_answers: List[str] = None) -> List[Tuple[GFlowNetState, List[Dict[str, Any]]]]:
        """
        Batch sample trajectories
        Args:
            questions: List of questions
            gold_answers: List of gold answers
        Returns:
            List of trajectories
        """
        trajectories = []
        
        for i, question in enumerate(questions):
            gold_answer = gold_answers[i] if gold_answers else None
            trajectory = self.sample_trajectory(question, gold_answer)
            trajectories.append(trajectory)
        
        return trajectories

class MockGFlowNetSampler:
    """Mock GFlowNet Sampler (for testing)"""
    
    def __init__(self, max_steps: int = 5):
        self.max_steps = max_steps
        self.policy_net = MockGFlowNetPolicy()
        self.llm_generator = MockLLMGenerator()
    
    def sample_trajectory(self, 
                         question: str, 
                         gold_answer: str = None) -> Tuple[GFlowNetState, List[Dict[str, Any]]]:
        """Mock sample trajectory"""
        # Initialize state
        state = GFlowNetState(question)
        trajectory_data = []
        
        # Mock reasoning steps
        mock_steps = [
            "Let me understand what this question is asking.",
            "I need to think through this step by step.",
            "Based on my knowledge, I can provide an answer."
        ]
        
        for step_idx, step in enumerate(mock_steps):
            # Mock candidate actions
            candidates = [f"Option {i+1}: {step}" for i in range(3)]
            
            # Mock action selection
            action_idx = step_idx % len(candidates)
            selected_action = candidates[action_idx]
            
            # Execute action
            state.add_step(selected_action)
            
            # Record trajectory data
            step_data = {
                'step_idx': step_idx,
                'state': state.get_text_representation(),
                'candidates': candidates,
                'selected_action': selected_action,
                'action_idx': action_idx,
                'action_probs': torch.tensor([0.4, 0.3, 0.3]),
                'action_log_prob': torch.log(torch.tensor(0.4))
            }
            trajectory_data.append(step_data)
        
        # Set terminal state
        final_answer = "Yes" if "swastika" in question.lower() else "No"
        state.set_terminal()
        
        # Compute reward
        if gold_answer:
            reward = 1.0 if final_answer.lower() == gold_answer.lower() else 0.0
            state.reward = reward
        
        return state, trajectory_data
    
    def batch_sample(self, 
                    questions: List[str], 
                    gold_answers: List[str] = None) -> List[Tuple[GFlowNetState, List[Dict[str, Any]]]]:
        """Mock batch sampling"""
        trajectories = []
        for i, question in enumerate(questions):
            gold_answer = gold_answers[i] if gold_answers else None
            trajectory = self.sample_trajectory(question, gold_answer)
            trajectories.append(trajectory)
        return trajectories

# Usage example
if __name__ == "__main__":
    # Test using mock sampler
    sampler = MockGFlowNetSampler()
    
    # Test question
    question = "Did the Hopi Indians use a symbol that was similar to the swastika?"
    gold_answer = "Yes"
    
    # Sample trajectory
    final_state, trajectory_data = sampler.sample_trajectory(question, gold_answer)
    
    print("Sampling Results:")
    print(f"Final State: {final_state}")
    print(f"Final Reward: {final_state.reward}")
    print(f"Trajectory Length: {len(trajectory_data)}")
    
    print("\nTrajectory Details:")
    for i, step_data in enumerate(trajectory_data):
        print(f"Step {i+1}:")
        print(f"  Selected Action: {step_data['selected_action']}")
        print(f"  Action Probabilities: {step_data['action_probs']}")
        print(f"  Log Probability: {step_data['action_log_prob']}")
    
    # Test batch sampling
    questions = [
        "Did the Hopi Indians use a symbol that was similar to the swastika?",
        "Is the capital of France located in Europe?"
    ]
    gold_answers = ["Yes", "Yes"]
    
    batch_trajectories = sampler.batch_sample(questions, gold_answers)
    
    print(f"\nBatch Sampling Results:")
    print(f"Number of Sampled Trajectories: {len(batch_trajectories)}")
    
    for i, (state, data) in enumerate(batch_trajectories):
        print(f"Question {i+1}: Reward = {state.reward}, Steps = {len(data)}")
