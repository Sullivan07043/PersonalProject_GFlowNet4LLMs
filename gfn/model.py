import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any, Tuple, Optional
import math
from transformers import AutoTokenizer, AutoModel
import logging

logger = logging.getLogger(__name__)

class GFlowNetPolicy(nn.Module):
    """GFlowNet Policy Network"""
    
    def __init__(self, 
                 hidden_dim: int = 512,
                 num_layers: int = 3,
                 dropout: float = 0.1,
                 max_steps: int = 10):
        """
        Initialize GFlowNet Policy Network
        Args:
            hidden_dim: Hidden layer dimension
            num_layers: Number of network layers
            dropout: Dropout rate
            max_steps: Maximum reasoning steps
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_steps = max_steps
        
        # Text encoder (using pre-trained BERT)
        self.text_encoder = AutoModel.from_pretrained("bert-base-uncased")
        self.text_encoder.requires_grad_(False)  # Freeze pre-trained weights
        
        # State encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(self.text_encoder.config.hidden_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Action encoder (candidate step encoding)
        self.action_encoder = nn.Sequential(
            nn.Linear(self.text_encoder.config.hidden_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Policy network
        self.policy_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)  # Output logits
        )
        
        # Termination probability network
        self.termination_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)  # Output termination probability
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights"""
        for module in [self.state_encoder, self.action_encoder, self.policy_net, self.termination_net]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.constant_(layer.bias, 0)
    
    def encode_text(self, text: str) -> torch.Tensor:
        """Encode text using BERT tokenizer and model"""
        # Initialize tokenizer if not exists
        if not hasattr(self, 'tokenizer'):
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        
        # Tokenize and encode
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=512
        )
        
        # Get BERT embeddings
        with torch.no_grad():
            outputs = self.text_encoder(**inputs)
            # Use mean pooling of all token embeddings
            embeddings = outputs.last_hidden_state.mean(dim=1)
        
        return embeddings
    
    def encode_state(self, state_text: str) -> torch.Tensor:
        """Encode state (partial reasoning chain)"""
        text_embedding = self.encode_text(state_text)
        return self.state_encoder(text_embedding)
    
    def encode_action(self, action_text: str) -> torch.Tensor:
        """Encode action (candidate step)"""
        text_embedding = self.encode_text(action_text)
        return self.action_encoder(text_embedding)
    
    def forward(self, 
                state_text: str, 
                action_candidates: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        Args:
            state_text: Current state text
            action_candidates: List of candidate actions
        Returns:
            action_logits: Action logits
            termination_prob: Termination probability
        """
        # Validate inputs
        if not action_candidates:
            logger.warning("No action candidates provided")
            return torch.tensor([]), torch.sigmoid(self.termination_net(self.encode_state(state_text)))
        
        # Encode state
        state_embedding = self.encode_state(state_text)
        
        # Encode all candidate actions
        action_embeddings = []
        for action in action_candidates:
            action_embedding = self.encode_action(action)
            action_embeddings.append(action_embedding)
        
        if not action_embeddings:
            # If no candidate actions, return empty tensor
            return torch.tensor([]), torch.sigmoid(self.termination_net(state_embedding))
        
        # Stack action embeddings
        action_embeddings = torch.stack(action_embeddings, dim=1)  # [batch, num_actions, hidden_dim]
        
        # Compute policy logits
        batch_size, num_actions, hidden_dim = action_embeddings.shape
        
        # Ensure state_embedding has correct shape [batch_size, hidden_dim]
        if state_embedding.dim() == 1:
            state_embedding = state_embedding.unsqueeze(0)  # Add batch dimension
        
        state_expanded = state_embedding.unsqueeze(1).expand(-1, num_actions, -1)
        
        # Concatenate state and action embeddings
        combined = torch.cat([state_expanded, action_embeddings], dim=-1)
        
        # Reshape to [batch * num_actions, hidden_dim * 2]
        combined = combined.view(-1, hidden_dim * 2)
        
        # Compute logits
        action_logits = self.policy_net(combined).view(batch_size, num_actions)
        
        # Compute termination probability
        termination_prob = torch.sigmoid(self.termination_net(state_embedding))
        
        return action_logits, termination_prob
    
    def get_action_probabilities(self, 
                                state_text: str, 
                                action_candidates: List[str]) -> torch.Tensor:
        """Get action probability distribution"""
        action_logits, _ = self.forward(state_text, action_candidates)
        if action_logits.numel() == 0:
            return torch.tensor([])
        return F.softmax(action_logits, dim=-1)
    
    def sample_action(self, 
                     state_text: str, 
                     action_candidates: List[str], 
                     temperature: float = 1.0) -> Tuple[int, torch.Tensor]:
        """Sample action"""
        # Validate inputs
        if not action_candidates:
            logger.warning("No action candidates provided for sampling")
            return -1, torch.tensor([])
        
        action_logits, _ = self.forward(state_text, action_candidates)
        if action_logits.numel() == 0:
            logger.warning("Empty action logits returned from forward pass")
            return -1, torch.tensor([])
        
        # Apply temperature
        action_logits = action_logits / temperature
        
        # Sample
        action_probs = F.softmax(action_logits, dim=-1)
        
        # Ensure action_probs has the correct shape [num_actions]
        if action_probs.dim() > 1:
            action_probs = action_probs.squeeze(0)  # Remove batch dimension if present
        
        # Validate dimensions
        if len(action_probs) != len(action_candidates):
            logger.error(f"Action probs length {len(action_probs)} != candidates length {len(action_candidates)}")
            # Create uniform distribution as fallback
            action_probs = torch.ones(len(action_candidates)) / len(action_candidates)
        
        # Ensure probabilities are valid
        if torch.any(torch.isnan(action_probs)) or torch.any(torch.isinf(action_probs)):
            logger.warning("Invalid probabilities detected, using uniform distribution")
            action_probs = torch.ones(len(action_candidates)) / len(action_candidates)
        
        # Sample action
        try:
            action = torch.multinomial(action_probs, 1).item()
        except RuntimeError as e:
            logger.error(f"Multinomial sampling failed: {e}, using uniform sampling")
            action = torch.randint(0, len(action_candidates), (1,)).item()
        
        return action, action_probs

class MockGFlowNetPolicy:
    """Mock GFlowNet Policy Network (for testing)"""
    
    def __init__(self, max_steps: int = 10):
        self.max_steps = max_steps
    
    def get_action_probabilities(self, 
                                state_text: str, 
                                action_candidates: List[str]) -> torch.Tensor:
        """Mock get action probability distribution"""
        if not action_candidates:
            return torch.tensor([])
        
        # Simple uniform distribution
        num_actions = len(action_candidates)
        return torch.ones(num_actions) / num_actions
    
    def sample_action(self, 
                     state_text: str, 
                     action_candidates: List[str], 
                     temperature: float = 1.0) -> Tuple[int, torch.Tensor]:
        """Mock sample action"""
        if not action_candidates:
            return -1, torch.tensor([])
        
        # Randomly select action
        import random
        action = random.randint(0, len(action_candidates) - 1)
        probs = self.get_action_probabilities(state_text, action_candidates)
        
        return action, probs
    
    def get_termination_probability(self, state_text: str) -> float:
        """Mock get termination probability"""
        # Simple termination logic: terminate if state contains "answer is"
        if "answer is" in state_text.lower():
            return 0.9
        else:
            return 0.1

class GFlowNetState:
    """GFlowNet State Class"""
    
    def __init__(self, question: str, steps: List[str] = None):
        self.question = question
        self.steps = steps or []
        self.is_terminal = False
        self.reward = 0.0
    
    def add_step(self, step: str):
        """Add reasoning step"""
        self.steps.append(step)
    
    def get_text_representation(self) -> str:
        """Get text representation of state"""
        text = f"Question: {self.question}\n\n"
        for i, step in enumerate(self.steps, 1):
            text += f"Step {i}: {step}\n"
        return text
    
    def is_terminal_state(self) -> bool:
        """Check if terminal state"""
        return self.is_terminal or len(self.steps) >= 10  # Maximum steps
    
    def set_terminal(self, reward: float = 0.0):
        """Set as terminal state"""
        self.is_terminal = True
        self.reward = reward
    
    def __len__(self):
        """Return number of steps"""
        return len(self.steps)
    
    def __str__(self):
        return self.get_text_representation()

# Usage example
if __name__ == "__main__":
    # Test using mock policy network
    policy = MockGFlowNetPolicy()
    
    # Test state
    question = "Did the Hopi Indians use a symbol that was similar to the swastika?"
    state = GFlowNetState(question, ["Step 1: Understanding the question"])
    
    # Candidate actions
    candidates = [
        "The swastika is a geometric symbol used in many cultures.",
        "I need to research Native American symbols.",
        "Let me think about Hopi culture and traditions."
    ]
    
    # Get action probabilities
    probs = policy.get_action_probabilities(state.get_text_representation(), candidates)
    print(f"Action Probabilities: {probs}")
    
    # Sample action
    action, action_probs = policy.sample_action(state.get_text_representation(), candidates)
    print(f"Sampled Action: {action}")
    print(f"Selected Candidate: {candidates[action]}")
    
    # Termination probability
    term_prob = policy.get_termination_probability(state.get_text_representation())
    print(f"Termination Probability: {term_prob}")
    
    # Test state operations
    state.add_step("Step 2: Researching the topic")
    print(f"\nUpdated State:\n{state}")
    print(f"Is Terminal State: {state.is_terminal_state()}")
    
    # Set as terminal state
    state.set_terminal(reward=1.0)
    print(f"Terminal State Reward: {state.reward}")
