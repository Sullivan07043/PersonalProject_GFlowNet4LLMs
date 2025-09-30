from typing import List, Dict, Any
import re

class PromptTemplate:
    """Prompt Template Class"""
    
    def __init__(self):
        self.templates = {
            "reasoning_chain": self._get_reasoning_chain_template(),
            "final_answer": self._get_final_answer_template(),
            "few_shot": self._get_few_shot_template()
        }
    
    def _get_reasoning_chain_template(self) -> str:
        """Reasoning chain generation template"""
        return """You are a helpful assistant that solves complex reasoning problems step by step.

Question: {question}

Please think through this question step by step. For each step, explain your reasoning clearly.

Step 1: Let me understand what this question is asking.
Step 2: [Your reasoning step]
Step 3: [Continue reasoning]
...

Based on my reasoning above, the answer is:"""

    def _get_final_answer_template(self) -> str:
        """Final answer extraction template"""
        return """Based on the reasoning above, the final answer is: {answer} (Yes/No)"""

    def _get_few_shot_template(self) -> str:
        """Few-shot learning template"""
        return """You are a helpful assistant that solves complex reasoning problems step by step.

Here are some examples:

Example 1:
Question: Did the Hopi Indians use a symbol that was similar to the swastika?
Step 1: Let me understand what this question is asking. I need to determine if the Hopi Indians used a symbol similar to the swastika.
Step 2: The swastika is a geometric symbol that has been used in many cultures throughout history, including in Native American cultures.
Step 3: The Hopi are a Native American tribe, and they did indeed use symbols similar to the swastika in their traditional art and ceremonies.
Based on my reasoning above, the answer is: Yes

Example 2:
Question: Is the capital of France located in Europe?
Step 1: Let me understand what this question is asking. I need to determine if Paris (the capital of France) is in Europe.
Step 2: France is a country in Western Europe.
Step 3: Paris is the capital city of France, so it must be located in Europe.
Based on my reasoning above, the answer is: Yes

Now solve this question:
Question: {question}
Step 1: Let me understand what this question is asking.
Step 2: [Your reasoning step]
Step 3: [Continue reasoning]
...

Based on my reasoning above, the answer is:"""

    def format_reasoning_prompt(self, question: str, use_few_shot: bool = False) -> str:
        """Format reasoning prompt"""
        if use_few_shot:
            return self.templates["few_shot"].format(question=question)
        else:
            return self.templates["reasoning_chain"].format(question=question)
    
    def format_final_answer_prompt(self, answer: str) -> str:
        """Format final answer prompt"""
        return self.templates["final_answer"].format(answer=answer)
    
    def extract_reasoning_steps(self, text: str) -> List[str]:
        """Extract reasoning steps from generated text"""
        steps = []
        # Find all lines starting with "Step X:"
        step_pattern = r'Step \d+:\s*(.+?)(?=Step \d+:|Based on|$)'
        matches = re.findall(step_pattern, text, re.DOTALL | re.IGNORECASE)
        
        for match in matches:
            step_text = match.strip()
            if step_text:
                steps.append(step_text)
        
        return steps
    
    def extract_final_answer(self, text: str) -> str:
        """Extract final answer from generated text"""
        # Find content after "Based on my reasoning above, the answer is:"
        answer_pattern = r'Based on my reasoning above, the answer is:\s*(.+?)(?:\n|$)'
        match = re.search(answer_pattern, text, re.IGNORECASE)
        
        if match:
            answer = match.group(1).strip()
            # Clean answer, keep only Yes/No
            if 'yes' in answer.lower():
                return 'Yes'
            elif 'no' in answer.lower():
                return 'No'
            else:
                return answer
        else:
            # If standard format not found, try other patterns
            if 'yes' in text.lower() and 'no' not in text.lower():
                return 'Yes'
            elif 'no' in text.lower() and 'yes' not in text.lower():
                return 'No'
            else:
                return 'Unknown'

class ReasoningChain:
    """Reasoning Chain Class"""
    
    def __init__(self, question: str, steps: List[str] = None, final_answer: str = None):
        self.question = question
        self.steps = steps or []
        self.final_answer = final_answer
        self.reward = 0.0  # For GFlowNet training
    
    def add_step(self, step: str):
        """Add reasoning step"""
        self.steps.append(step)
    
    def set_final_answer(self, answer: str):
        """Set final answer"""
        self.final_answer = answer
    
    def get_full_text(self) -> str:
        """Get complete reasoning chain text"""
        text = f"Question: {self.question}\n\n"
        for i, step in enumerate(self.steps, 1):
            text += f"Step {i}: {step}\n"
        if self.final_answer:
            text += f"\nBased on my reasoning above, the answer is: {self.final_answer}"
        return text
    
    def get_partial_text(self, step_count: int) -> str:
        """Get partial reasoning chain text (for GFlowNet state representation)"""
        text = f"Question: {self.question}\n\n"
        for i, step in enumerate(self.steps[:step_count], 1):
            text += f"Step {i}: {step}\n"
        return text
    
    def is_complete(self) -> bool:
        """Check if reasoning chain is complete"""
        return len(self.steps) > 0 and self.final_answer is not None
    
    def __len__(self):
        """Return number of reasoning steps"""
        return len(self.steps)

# Usage example
if __name__ == "__main__":
    # Create prompt template
    prompt_template = PromptTemplate()
    
    # Test question
    question = "Did the Hopi Indians use a symbol that was similar to the swastika?"
    
    # Generate prompt
    prompt = prompt_template.format_reasoning_prompt(question, use_few_shot=True)
    print("Generated Prompt:")
    print(prompt)
    print("\n" + "="*50 + "\n")
    
    # Mock generated reasoning chain
    generated_text = """Step 1: Let me understand what this question is asking. I need to determine if the Hopi Indians used a symbol similar to the swastika.
Step 2: The swastika is a geometric symbol that has been used in many cultures throughout history, including in Native American cultures.
Step 3: The Hopi are a Native American tribe, and they did indeed use symbols similar to the swastika in their traditional art and ceremonies.
Based on my reasoning above, the answer is: Yes"""
    
    # Extract reasoning steps
    steps = prompt_template.extract_reasoning_steps(generated_text)
    print("Extracted Reasoning Steps:")
    for i, step in enumerate(steps, 1):
        print(f"Step {i}: {step}")
    
    # Extract final answer
    answer = prompt_template.extract_final_answer(generated_text)
    print(f"\nExtracted Final Answer: {answer}")
    
    # Create reasoning chain object
    chain = ReasoningChain(question, steps, answer)
    print(f"\nComplete Reasoning Chain:\n{chain.get_full_text()}")
    print(f"\nIs Chain Complete: {chain.is_complete()}")
    print(f"Number of Reasoning Steps: {len(chain)}")
