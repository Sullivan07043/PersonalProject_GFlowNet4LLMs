import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from typing import List, Dict, Any, Optional
import logging
import sys
import os
sys.path.append(os.path.dirname(__file__))
from promt import PromptTemplate, ReasoningChain

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMGenerator:
    """LLM Reasoning Chain Generator"""
    
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium", device: str = "auto"):
        """
        Initialize LLM Generator
        Args:
            model_name: Model name
            device: Device ("auto", "cpu", "cuda")
        """
        self.model_name = model_name
        self.device = self._get_device(device)
        self.tokenizer = None
        self.model = None
        self.prompt_template = PromptTemplate()
        
        # Generation configuration
        self.generation_config = GenerationConfig(
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=None,  # Will be set after loading model
            eos_token_id=None,  # Will be set after loading model
        )
        
        self._load_model()
    
    def _get_device(self, device: str) -> str:
        """Get device"""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return device
    
    def _load_model(self):
        """Load model and tokenizer"""
        try:
            logger.info(f"Loading model: {self.model_name}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            # Set pad_token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model with better error handling
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    device_map=self.device if self.device == "cuda" else None,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True  # Reduce memory usage
                )
            except Exception as e:
                logger.warning(f"Failed to load model on {self.device}, falling back to CPU: {e}")
                self.device = "cpu"
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    dtype=torch.float32,
                    device_map=None,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True
                )
            
            if self.device != "cuda":
                self.model = self.model.to(self.device)
            
            # Update generation configuration
            self.generation_config.pad_token_id = self.tokenizer.pad_token_id
            self.generation_config.eos_token_id = self.tokenizer.eos_token_id
            
            logger.info(f"Model loaded successfully, device: {self.device}")
            
        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            raise
    
    def generate_reasoning_chain(self, question: str, use_few_shot: bool = False) -> ReasoningChain:
        """
        Generate complete reasoning chain
        Args:
            question: Question
            use_few_shot: Whether to use few-shot learning
        Returns:
            Reasoning chain object
        """
        # Generate prompt
        prompt = self.prompt_template.format_reasoning_prompt(question, use_few_shot)
        
        # Generate text
        generated_text = self._generate_text(prompt)
        
        # Extract reasoning steps and answer
        steps = self.prompt_template.extract_reasoning_steps(generated_text)
        answer = self.prompt_template.extract_final_answer(generated_text)
        
        # Create reasoning chain object
        chain = ReasoningChain(question, steps, answer)
        
        return chain
    
    def generate_step_candidates(self, partial_chain: ReasoningChain, num_candidates: int = 5) -> List[str]:
        """
        Generate candidate steps for partial reasoning chain
        Args:
            partial_chain: Partial reasoning chain
            num_candidates: Number of candidates
        Returns:
            List of candidate steps
        """
        # Build prompt
        prompt = partial_chain.get_partial_text(len(partial_chain))
        prompt += f"\nStep {len(partial_chain) + 1}:"
        
        candidates = []
        max_attempts = num_candidates * 3  # Allow up to 3x attempts to get enough candidates
        attempt = 0
        
        while len(candidates) < num_candidates and attempt < max_attempts:
            attempt += 1
            
            # Generate single candidate
            generated_text = self._generate_text(prompt, max_new_tokens=100)
            
            # Extract step content
            step_content = generated_text.replace(prompt, "").strip()
            
            # Only add non-empty, unique candidates
            if step_content and step_content not in candidates:
                candidates.append(step_content)
            elif not step_content:
                # If generation failed, add a fallback candidate
                fallback_candidate = f"Step {len(partial_chain) + 1}: Analyzing the question further."
                if fallback_candidate not in candidates:
                    candidates.append(fallback_candidate)
        
        # Ensure we have exactly num_candidates
        while len(candidates) < num_candidates:
            fallback_candidate = f"Step {len(partial_chain) + 1}: Reasoning step {len(candidates) + 1}."
            candidates.append(fallback_candidate)
        
        # Return only the requested number
        return candidates[:num_candidates]
    
    def _generate_text(self, prompt: str, max_new_tokens: int = None) -> str:
        """
        Generate text
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum new tokens
        Returns:
            Generated text
        """
        try:
            # Validate prompt
            if not prompt or not prompt.strip():
                logger.warning("Empty prompt provided")
                return "Empty reasoning step"
            
            # Encode input with proper error handling
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                padding=True, 
                truncation=True,
                max_length=512  # Limit input length
            )
            
            # Move inputs to device safely
            try:
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            except Exception as e:
                logger.error(f"Failed to move inputs to device {self.device}: {e}")
                # Fallback to CPU
                self.device = "cpu"
                self.model = self.model.to("cpu")
                inputs = {k: v.to("cpu") for k, v in inputs.items()}
            
            # Set generation parameters
            gen_config = self.generation_config
            if max_new_tokens:
                # Create new config with only max_new_tokens to avoid conflict
                config_dict = gen_config.to_dict()
                if 'max_new_tokens' in config_dict:
                    del config_dict['max_new_tokens']
                if 'max_length' in config_dict:
                    del config_dict['max_length']
                gen_config = GenerationConfig(**config_dict, max_new_tokens=max_new_tokens)
            
            # Generate with error handling
            with torch.no_grad():
                try:
                    outputs = self.model.generate(
                        **inputs,
                        generation_config=gen_config,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id
                    )
                except RuntimeError as e:
                    if "CUDA" in str(e):
                        logger.error(f"CUDA error during generation: {e}")
                        # Clear CUDA cache and retry on CPU
                        torch.cuda.empty_cache()
                        self.device = "cpu"
                        self.model = self.model.to("cpu")
                        inputs = {k: v.to("cpu") for k, v in inputs.items()}
                        outputs = self.model.generate(
                            **inputs,
                            generation_config=gen_config,
                            pad_token_id=self.tokenizer.pad_token_id,
                            eos_token_id=self.tokenizer.eos_token_id
                        )
                    else:
                        raise e
            
            # Decode output
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            return generated_text
            
        except Exception as e:
            logger.error(f"Text generation failed: {e}")
            # Return a fallback response instead of empty string
            return f"Generated reasoning step: {prompt.split('Step')[-1].strip() if 'Step' in prompt else 'Continuing analysis'}"
    
    def batch_generate(self, questions: List[str], use_few_shot: bool = False) -> List[ReasoningChain]:
        """
        Batch generate reasoning chains
        Args:
            questions: List of questions
            use_few_shot: Whether to use few-shot learning
        Returns:
            List of reasoning chains
        """
        chains = []
        for question in questions:
            try:
                chain = self.generate_reasoning_chain(question, use_few_shot)
                chains.append(chain)
            except Exception as e:
                logger.error(f"Failed to generate reasoning chain (question: {question}): {e}")
                # Create empty reasoning chain as placeholder
                chains.append(ReasoningChain(question))
        
        return chains
    
    def evaluate_chain(self, chain: ReasoningChain, gold_answer: str) -> float:
        """
        Evaluate reasoning chain
        Args:
            chain: Reasoning chain
            gold_answer: Gold answer
        Returns:
            Reward score (0.0 or 1.0)
        """
        if not chain.is_complete():
            return 0.0
        
        # Simple answer matching evaluation
        predicted_answer = chain.final_answer.lower()
        gold_answer = gold_answer.lower()
        
        if predicted_answer == gold_answer:
            return 1.0
        else:
            return 0.0

class MockLLMGenerator:
    """Mock LLM Generator (for testing)"""
    
    def __init__(self):
        self.prompt_template = PromptTemplate()
    
    def generate_reasoning_chain(self, question: str, use_few_shot: bool = False) -> ReasoningChain:
        """Mock generate reasoning chain"""
        # Simple mock logic
        if "swastika" in question.lower():
            steps = [
                "Let me understand what this question is asking. I need to determine if the Hopi Indians used a symbol similar to the swastika.",
                "The swastika is a geometric symbol that has been used in many cultures throughout history, including in Native American cultures.",
                "The Hopi are a Native American tribe, and they did indeed use symbols similar to the swastika in their traditional art and ceremonies."
            ]
            answer = "Yes"
        else:
            steps = [
                "Let me understand what this question is asking.",
                "I need to think through this step by step.",
                "Based on my knowledge, I can provide an answer."
            ]
            answer = "Yes"  # Default answer
        
        return ReasoningChain(question, steps, answer)
    
    def generate_step_candidates(self, partial_chain: ReasoningChain, num_candidates: int = 5) -> List[str]:
        """Mock generate candidate steps"""
        candidates = []
        for i in range(num_candidates):
            candidates.append(f"Mock reasoning step {i+1} for the current question.")
        
        # Ensure we return exactly the requested number
        return candidates[:num_candidates]
    
    def evaluate_chain(self, chain: ReasoningChain, gold_answer: str) -> float:
        """Mock evaluate reasoning chain"""
        if not chain.is_complete():
            return 0.0
        
        predicted_answer = chain.final_answer.lower()
        gold_answer = gold_answer.lower()
        
        if predicted_answer == gold_answer:
            return 1.0
        else:
            return 0.0

# Usage example
if __name__ == "__main__":
    # Test using mock generator
    generator = MockLLMGenerator()
    
    # Test question
    question = "Did the Hopi Indians use a symbol that was similar to the swastika?"
    
    # Generate reasoning chain
    chain = generator.generate_reasoning_chain(question, use_few_shot=True)
    
    print("Generated Reasoning Chain:")
    print(chain.get_full_text())
    print(f"\nIs Chain Complete: {chain.is_complete()}")
    print(f"Number of Reasoning Steps: {len(chain)}")
    
    # Test candidate step generation
    partial_chain = ReasoningChain(question, ["Step 1: Understanding the question"])
    candidates = generator.generate_step_candidates(partial_chain, num_candidates=3)
    
    print(f"\nCandidate Steps:")
    for i, candidate in enumerate(candidates, 1):
        print(f"Candidate {i}: {candidate}")
    
    # Test evaluation
    reward = generator.evaluate_chain(chain, "Yes")
    print(f"\nEvaluation Reward: {reward}")
