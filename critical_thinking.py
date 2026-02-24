# """
# Critical Thinking Workflow Module for Sentinel-LLM

# This module implements advanced reasoning patterns for processing user prompts
# using Large Language Models (LLMs). It provides two primary thinking modes:

# 1. Chain of Thought (CoT): Sequential step-by-step reasoning
# 2. Tree of Thought (ToT): Multi-path exploration with evaluation and selection

# The module is designed to integrate with LLM APIs (e.g., OpenAI, Groq, Llama 3.3 70B)
# through a generic call_model() interface.
# """

# import asyncio
# from typing import Dict, List, Optional, Callable, Any
# from dataclasses import dataclass, field
# import json


# # ============================================================
# # DATA STRUCTURES
# # ============================================================

# @dataclass
# class ReasoningStep:
#     """Represents a single step in Chain of Thought reasoning."""
#     step_number: int
#     thought: str
#     reasoning: str
#     conclusion: Optional[str] = None


# @dataclass
# class ThoughtBranch:
#     """Represents a branch in Tree of Thought reasoning."""
#     branch_id: int
#     path: List[str]
#     evaluation_score: float = 0.0
#     is_valid: bool = True
#     reasoning: str = ""


# @dataclass
# class ThinkingResult:
#     """Container for the final result of a thinking process."""
#     mode: str  # 'cot' or 'tot'
#     original_prompt: str
#     final_answer: str
#     reasoning_trace: List[Any]
#     metadata: Dict[str, Any] = field(default_factory=dict)


# # ============================================================
# # MODEL INTERACTION
# # ============================================================

# async def call_model(prompt: str, **kwargs) -> str:
#     """
#     Placeholder function to simulate or connect to an LLM API.
    
#     In production, this would connect to services like:
#     - OpenAI API (GPT-4, GPT-3.5)
#     - Groq API
#     - Llama 3.3 70B API (via Groq)
#     - Local LLM via llama-cpp-python
#     - Any other LLM service
    
#     Parameters
#     ----------
#     prompt : str
#         The input prompt to send to the LLM
#     **kwargs : dict
#         Additional parameters like temperature, max_tokens, etc.
        
#     Returns
#     -------
#     str
#         The model's response text
        
#     Example
#     -------
#     To integrate with OpenAI:
#     >>> import openai
#     >>> response = await openai.ChatCompletion.acreate(
#     ...     model="gpt-4",
#     ...     messages=[{"role": "user", "content": prompt}]
#     ... )
#     >>> return response.choices[0].message.content
#     """
#     # Simulated response for demonstration
#     # In production, replace this with actual API calls
#     await asyncio.sleep(0.1)  # Simulate API latency
    
#     # Simple simulation that echoes understanding of the prompt
#     MAX_PROMPT_PREVIEW = 50  # Character limit for prompt preview in simulation
#     return f"[Simulated LLM Response to: '{prompt[:MAX_PROMPT_PREVIEW]}...']"


# # ============================================================
# # CHAIN OF THOUGHT (CoT) IMPLEMENTATION
# # ============================================================

# class ChainOfThoughtProcessor:
#     """
#     Implements Chain of Thought reasoning pattern.
    
#     CoT breaks down complex problems into sequential steps, where each step
#     builds upon the previous one, culminating in a final conclusion.
#     """
    
#     def __init__(self, model_fn: Optional[Callable] = None):
#         """
#         Initialize the CoT processor.
        
#         Parameters
#         ----------
#         model_fn : Callable, optional
#             Custom model function. If None, uses the default call_model.
#         """
#         self.model_fn = model_fn or call_model
        
#     async def process(
#         self,
#         prompt: str,
#         num_steps: int = 5,
#         include_reflection: bool = True
#     ) -> ThinkingResult:
#         """
#         Process a prompt using Chain of Thought reasoning.
        
#         Parameters
#         ----------
#         prompt : str
#             The user's question or problem to solve
#         num_steps : int
#             Number of reasoning steps to generate
#         include_reflection : bool
#             Whether to include a final reflection step
            
#         Returns
#         -------
#         ThinkingResult
#             Complete reasoning trace and final answer
#         """
#         reasoning_steps: List[ReasoningStep] = []
        
#         # Step 1: Initial decomposition
#         decomposition_prompt = f"""
# You are a critical thinking assistant. Break down this problem into logical steps.

# User Query: {prompt}

# Task: Identify the key aspects that need to be considered to answer this question.
# Provide 3-5 key thinking points.
# """
#         decomposition = await self.model_fn(decomposition_prompt)
        
#         # Step 2: Sequential reasoning through each step
#         context = f"Original question: {prompt}\n\n"
        
#         for i in range(num_steps):
#             step_prompt = f"""
# {context}

# Step {i + 1} of {num_steps}:
# Based on the above, what is the next logical consideration or insight?
# Provide a clear, focused reasoning step.

# Format your response as:
# Thought: [Your thinking]
# Reasoning: [Why this matters]
# """
#             step_response = await self.model_fn(step_prompt)
            
#             step = ReasoningStep(
#                 step_number=i + 1,
#                 thought=step_response,
#                 reasoning=f"Sequential reasoning step {i + 1}"
#             )
#             reasoning_steps.append(step)
            
#             # Build context for next step
#             context += f"\nStep {i + 1}: {step_response}\n"
        
#         # Step 3: Synthesis
#         synthesis_prompt = f"""
# {context}

# Now synthesize all the above reasoning steps into a final, coherent answer to:
# {prompt}

# Provide a clear, well-reasoned conclusion.
# """
#         final_answer = await self.model_fn(synthesis_prompt)
        
#         # Optional reflection
#         if include_reflection:
#             reflection_prompt = f"""
# Given the reasoning process and conclusion:
# {final_answer}

# Briefly reflect: Are there any gaps or alternative perspectives?
# """
#             reflection = await self.model_fn(reflection_prompt)
#             reasoning_steps.append(
#                 ReasoningStep(
#                     step_number=len(reasoning_steps) + 1,
#                     thought=reflection,
#                     reasoning="Final reflection",
#                     conclusion=final_answer
#                 )
#             )
        
#         return ThinkingResult(
#             mode="chain_of_thought",
#             original_prompt=prompt,
#             final_answer=final_answer,
#             reasoning_trace=reasoning_steps,
#             metadata={
#                 "num_steps": num_steps,
#                 "included_reflection": include_reflection,
#                 "decomposition": decomposition
#             }
#         )


# # ============================================================
# # TREE OF THOUGHT (ToT) IMPLEMENTATION
# # ============================================================

# class TreeOfThoughtProcessor:
#     """
#     Implements Tree of Thought reasoning pattern.
    
#     ToT explores multiple reasoning paths in parallel, evaluates each path,
#     and selects or synthesizes the best solution.
#     """
    
#     def __init__(self, model_fn: Optional[Callable] = None):
#         """
#         Initialize the ToT processor.
        
#         Parameters
#         ----------
#         model_fn : Callable, optional
#             Custom model function. If None, uses the default call_model.
#         """
#         self.model_fn = model_fn or call_model
    
#     async def _generate_branches(
#         self,
#         prompt: str,
#         num_branches: int = 3
#     ) -> List[ThoughtBranch]:
#         """
#         Generate multiple initial reasoning branches.
        
#         Parameters
#         ----------
#         prompt : str
#             The original problem
#         num_branches : int
#             Number of parallel branches to explore
            
#         Returns
#         -------
#         List[ThoughtBranch]
#             Initial thought branches
#         """
#         branches: List[ThoughtBranch] = []
        
#         generation_prompt = f"""
# You are exploring different approaches to solve this problem:
# {prompt}

# Generate {num_branches} distinct reasoning approaches or perspectives.
# Each should be fundamentally different in its approach.

# Provide {num_branches} numbered approaches.
# """
#         approaches_text = await self.model_fn(generation_prompt)
        
#         # Create branches (in production, you'd parse the model output)
#         for i in range(num_branches):
#             branch = ThoughtBranch(
#                 branch_id=i,
#                 path=[f"Approach {i + 1}: {approaches_text}"],
#                 reasoning=f"Initial exploration path {i + 1}"
#             )
#             branches.append(branch)
        
#         return branches
    
#     async def _expand_branch(
#         self,
#         branch: ThoughtBranch,
#         prompt: str,
#         depth: int = 2
#     ) -> ThoughtBranch:
#         """
#         Expand a single branch with additional reasoning steps.
        
#         Parameters
#         ----------
#         branch : ThoughtBranch
#             The branch to expand
#         prompt : str
#             Original problem context
#         depth : int
#             How many levels to expand
            
#         Returns
#         -------
#         ThoughtBranch
#             Expanded branch with more reasoning steps
#         """
#         current_path = "\n".join(branch.path)
        
#         for level in range(depth):
#             expansion_prompt = f"""
# Original problem: {prompt}

# Current reasoning path:
# {current_path}

# Continue this reasoning path with the next logical step.
# Build upon what came before.
# """
#             next_step = await self.model_fn(expansion_prompt)
#             branch.path.append(f"Step {level + 2}: {next_step}")
#             current_path = "\n".join(branch.path)
        
#         return branch
    
#     async def _evaluate_branch(
#         self,
#         branch: ThoughtBranch,
#         prompt: str
#     ) -> float:
#         """
#         Evaluate the quality of a reasoning branch.
        
#         Parameters
#         ----------
#         branch : ThoughtBranch
#             The branch to evaluate
#         prompt : str
#             Original problem for context
            
#         Returns
#         -------
#         float
#             Quality score (0.0 to 1.0)
#         """
#         evaluation_prompt = f"""
# Original problem: {prompt}

# Reasoning path:
# {chr(10).join(branch.path)}

# Evaluate this reasoning path on:
# 1. Logical coherence (0-10)
# 2. Relevance to the problem (0-10)
# 3. Depth of insight (0-10)

# Provide a single overall score from 0-10.
# """
#         evaluation_text = await self.model_fn(evaluation_prompt)
        
#         # Simple simulation: extract numeric score (in production, parse properly)
#         # For now, assign based on branch_id as simulation
#         score = 0.7 + (branch.branch_id * 0.05)
#         score = min(1.0, score)
        
#         return score
    
#     async def process(
#         self,
#         prompt: str,
#         num_branches: int = 3,
#         expansion_depth: int = 2,
#         selection_strategy: str = "best"
#     ) -> ThinkingResult:
#         """
#         Process a prompt using Tree of Thought reasoning.
        
#         Parameters
#         ----------
#         prompt : str
#             The user's question or problem to solve
#         num_branches : int
#             Number of parallel reasoning branches to explore
#         expansion_depth : int
#             How many steps to expand each branch
#         selection_strategy : str
#             'best' = pick highest scored branch
#             'synthesis' = combine insights from all branches
            
#         Returns
#         -------
#         ThinkingResult
#             Complete reasoning trace and final answer
#         """
#         # Phase 1: Generate initial branches
#         branches = await self._generate_branches(prompt, num_branches)
        
#         # Phase 2: Expand each branch
#         expanded_branches = []
#         for branch in branches:
#             expanded = await self._expand_branch(branch, prompt, expansion_depth)
#             expanded_branches.append(expanded)
        
#         # Phase 3: Evaluate branches
#         for branch in expanded_branches:
#             score = await self._evaluate_branch(branch, prompt)
#             branch.evaluation_score = score
        
#         # Phase 4: Selection or synthesis
#         if selection_strategy == "best":
#             # Select the highest-scored branch
#             best_branch = max(expanded_branches, key=lambda b: b.evaluation_score)
            
#             synthesis_prompt = f"""
# Original problem: {prompt}

# Best reasoning path (score: {best_branch.evaluation_score:.2f}):
# {chr(10).join(best_branch.path)}

# Based on this reasoning, provide a final answer to the original problem.
# """
#             final_answer = await self.model_fn(synthesis_prompt)
            
#             metadata = {
#                 "selection_strategy": "best",
#                 "selected_branch_id": best_branch.branch_id,
#                 "selected_score": best_branch.evaluation_score
#             }
            
#         else:  # synthesis strategy
#             # Combine insights from all branches
#             all_paths = "\n\n".join([
#                 f"Path {b.branch_id + 1} (score: {b.evaluation_score:.2f}):\n" + 
#                 "\n".join(b.path)
#                 for b in expanded_branches
#             ])
            
#             synthesis_prompt = f"""
# Original problem: {prompt}

# Multiple reasoning paths explored:
# {all_paths}

# Synthesize the best insights from all paths into a comprehensive answer.
# """
#             final_answer = await self.model_fn(synthesis_prompt)
            
#             metadata = {
#                 "selection_strategy": "synthesis",
#                 "all_scores": [b.evaluation_score for b in expanded_branches]
#             }
        
#         return ThinkingResult(
#             mode="tree_of_thought",
#             original_prompt=prompt,
#             final_answer=final_answer,
#             reasoning_trace=expanded_branches,
#             metadata=metadata
#         )


# # ============================================================
# # MAIN CRITICAL THINKING PROCESSOR
# # ============================================================

# class CriticalThinkingProcessor:
#     """
#     Main interface for critical thinking workflows.
    
#     This class provides a unified interface to apply different reasoning
#     patterns (Chain of Thought, Tree of Thought) to user prompts.
#     """
    
#     def __init__(self, model_fn: Optional[Callable] = None):
#         """
#         Initialize the critical thinking processor.
        
#         Parameters
#         ----------
#         model_fn : Callable, optional
#             Custom model function. If None, uses the default call_model.
#         """
#         self.model_fn = model_fn or call_model
#         self.cot_processor = ChainOfThoughtProcessor(model_fn)
#         self.tot_processor = TreeOfThoughtProcessor(model_fn)
    
#     async def process_with_mode(
#         self,
#         prompt: str,
#         mode: str = "cot",
#         **kwargs
#     ) -> ThinkingResult:
#         """
#         Process a prompt using the specified thinking mode.
        
#         Parameters
#         ----------
#         prompt : str
#             User's question or problem
#         mode : str
#             Reasoning mode: 'cot', 'tot', 'chain_of_thought', or 'tree_of_thought'
#         **kwargs : dict
#             Additional parameters for the specific mode
            
#         Returns
#         -------
#         ThinkingResult
#             Complete reasoning result
            
#         Raises
#         ------
#         ValueError
#             If mode is not recognized
#         """
#         mode_normalized = mode.lower()
#         if mode_normalized in ("cot", "chain_of_thought"):
#             return await self.cot_processor.process(prompt, **kwargs)
#         elif mode_normalized in ("tot", "tree_of_thought"):
#             return await self.tot_processor.process(prompt, **kwargs)
#         else:
#             raise ValueError(
#                 f"Unknown mode: {mode}. Use 'cot', 'tot', "
#                 "'chain_of_thought', or 'tree_of_thought'."
#             )
    
#     async def auto_process(self, prompt: str) -> ThinkingResult:
#         """
#         Automatically select and apply the best thinking mode.
        
#         This method analyzes the prompt and chooses between CoT and ToT
#         based on problem characteristics.
        
#         Parameters
#         ----------
#         prompt : str
#             User's question or problem
            
#         Returns
#         -------
#         ThinkingResult
#             Complete reasoning result
#         """
#         # Heuristic: Use ToT for open-ended or creative problems
#         # Use CoT for analytical or sequential problems
        
#         analysis_prompt = f"""
# Analyze this problem:
# {prompt}

# Is this problem:
# A) Sequential/analytical (best solved step-by-step)
# B) Open-ended/creative (benefits from exploring multiple approaches)

# Respond with just A or B.
# """
#         analysis = await self.model_fn(analysis_prompt)
        
#         # Simple heuristic (in production, parse the response properly)
#         # Short prompts (< 20 words) typically benefit from direct CoT
#         SHORT_PROMPT_THRESHOLD = 20  # word count threshold for mode selection
#         mode = "cot" if "A" in analysis or len(prompt.split()) < SHORT_PROMPT_THRESHOLD else "tot"
        
#         print(f"Auto-selected mode: {mode.upper()}")
#         return await self.process_with_mode(prompt, mode=mode)
    
#     def format_result(self, result: ThinkingResult) -> str:
#         """
#         Format a thinking result for human-readable display.
        
#         Parameters
#         ----------
#         result : ThinkingResult
#             The thinking result to format
            
#         Returns
#         -------
#         str
#             Formatted string representation
#         """
#         output = []
#         output.append("=" * 80)
#         output.append(f"CRITICAL THINKING ANALYSIS ({result.mode.upper()})")
#         output.append("=" * 80)
#         output.append(f"\nOriginal Prompt: {result.original_prompt}\n")
        
#         if result.mode == "chain_of_thought":
#             output.append("REASONING STEPS:")
#             output.append("-" * 80)
#             for step in result.reasoning_trace:
#                 output.append(f"\nStep {step.step_number}:")
#                 output.append(f"  {step.thought}")
#                 if step.conclusion:
#                     output.append(f"  Conclusion: {step.conclusion}")
        
#         elif result.mode == "tree_of_thought":
#             output.append("REASONING BRANCHES:")
#             output.append("-" * 80)
#             for branch in result.reasoning_trace:
#                 output.append(f"\nBranch {branch.branch_id + 1} "
#                             f"(Score: {branch.evaluation_score:.2f}):")
#                 for path_step in branch.path:
#                     output.append(f"  • {path_step}")
        
#         output.append("\n" + "=" * 80)
#         output.append("FINAL ANSWER:")
#         output.append("-" * 80)
#         output.append(result.final_answer)
#         output.append("=" * 80)
        
#         if result.metadata:
#             output.append("\nMetadata:")
#             output.append(json.dumps(result.metadata, indent=2))
        
#         return "\n".join(output)


# # ============================================================
# # MAIN INTERFACE FUNCTION
# # ============================================================

# async def think(
#     prompt: str,
#     mode: str = "auto",
#     model_fn: Optional[Callable] = None,
#     verbose: bool = True,
#     **kwargs
# ) -> ThinkingResult:
#     """
#     Main interface function for critical thinking processing.
    
#     This is the primary entry point for using the critical thinking module.
    
#     Parameters
#     ----------
#     prompt : str
#         The user's question or problem to analyze
#     mode : str
#         Reasoning mode: 'cot', 'tot', or 'auto'
#         - 'cot': Chain of Thought (sequential reasoning)
#         - 'tot': Tree of Thought (multi-path exploration)
#         - 'auto': Automatically select best mode
#     model_fn : Callable, optional
#         Custom function to call LLM. If None, uses default.
#     verbose : bool
#         If True, prints formatted output
#     **kwargs : dict
#         Additional parameters for specific modes
        
#     Returns
#     -------
#     ThinkingResult
#         Complete thinking result with reasoning trace
        
#     Examples
#     --------
#     >>> # Use Chain of Thought
#     >>> result = await think(
#     ...     "How can we reduce carbon emissions?",
#     ...     mode="cot"
#     ... )
    
#     >>> # Use Tree of Thought
#     >>> result = await think(
#     ...     "What are creative solutions to urban traffic?",
#     ...     mode="tot",
#     ...     num_branches=4
#     ... )
    
#     >>> # Auto-select mode
#     >>> result = await think(
#     ...     "Should AI systems have rights?",
#     ...     mode="auto"
#     ... )
#     """
#     processor = CriticalThinkingProcessor(model_fn)
    
#     if mode.lower() == "auto":
#         result = await processor.auto_process(prompt)
#     else:
#         result = await processor.process_with_mode(prompt, mode=mode, **kwargs)
    
#     if verbose:
#         print(processor.format_result(result))
    
#     return result


# # ============================================================
# # EXAMPLE USAGE
# # ============================================================

# async def main():
#     """
#     Example usage demonstrating the critical thinking module.
#     """
#     print("\n🧠 Critical Thinking Workflow Demo\n")
    
#     # Example 1: Chain of Thought
#     print("\n" + "=" * 80)
#     print("EXAMPLE 1: Chain of Thought")
#     print("=" * 80)
    
#     result1 = await think(
#         "What are the key factors to consider when designing a sustainable city?",
#         mode="cot",
#         num_steps=4,
#         verbose=True
#     )
    
#     # Example 2: Tree of Thought
#     print("\n\n" + "=" * 80)
#     print("EXAMPLE 2: Tree of Thought")
#     print("=" * 80)
    
#     result2 = await think(
#         "How might artificial intelligence reshape education in the next decade?",
#         mode="tot",
#         num_branches=3,
#         expansion_depth=2,
#         selection_strategy="synthesis",
#         verbose=True
#     )
    
#     # Example 3: Auto mode
#     print("\n\n" + "=" * 80)
#     print("EXAMPLE 3: Auto Mode Selection")
#     print("=" * 80)
    
#     result3 = await think(
#         "Is consciousness an emergent property of complex systems?",
#         mode="auto",
#         verbose=True
#     )


# if __name__ == "__main__":
#     asyncio.run(main())

# ==============================================================================
# SENTINEL-X: INDUSTRIAL REASONING ENGINE (v12.0)
# ==============================================================================
# PURPOSE: High-Fidelity Complex Problem Solving (Non-Security)
# ------------------------------------------------------------------------------
# ARCHITECTURE:
# 1. VERIFIED CHAIN OF THOUGHT (vCoT): Step-by-step with self-correction.
# 2. STRATEGIC TREE OF THOUGHT (sToT): Multi-path exploration with pruning.
# 3. INDUSTRIAL INTEGRATION: Uses shared cloud_clients.py.
# ==============================================================================

import asyncio
import json
import logging
import os
import sys
import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

# ==============================================================================
# 0. ROBUST IMPORT (Auto-Discovery)
# ==============================================================================
# Automatically finds cloud_clients.py in parent/sibling directories
current_dir = os.path.dirname(os.path.abspath(__file__))
search_paths = [
    current_dir,
    os.path.abspath(os.path.join(current_dir, "..")),
    os.path.abspath(os.path.join(current_dir, "..", "models")),
    os.path.abspath(os.path.join(current_dir, "models")),
]
client_found = False
for path in search_paths:
    if path not in sys.path: sys.path.insert(0, path)
    try:
        from backend.models.cloud_clients import CloudModelClient
        client_found = True
        break
    except ImportError:
        pass

if not client_found:
    print("[!] CRITICAL: 'cloud_clients.py' not found. Ensure it exists nearby.")
    exit(1)

# ==============================================================================
# 1. DATA STRUCTURES (REASONING FOCUSED)
# ==============================================================================

@dataclass
class ReasoningStep:
    """A single verified step in the reasoning chain."""
    step_id: int
    thought_content: str
    verification_passed: bool
    correction: Optional[str] = None
    confidence: int = 0  # 0-100

@dataclass
class StrategicBranch:
    """A parallel solution path in the Tree of Thought."""
    branch_id: int
    approach_name: str
    steps_taken: List[str]
    feasibility_score: int  # 0-100
    impact_score: int       # 0-100
    final_score: float      # Weighted average

@dataclass
class ReasoningResult:
    """The final output package."""
    mode: str
    target_model: str
    input_problem: str
    final_answer: str
    trace: List[Any]  # Can be Steps or Branches
    execution_time: float

# ==============================================================================
# 2. THE INDUSTRIAL REASONING ENGINE
# ==============================================================================

class IndustrialReasoningEngine:
    def __init__(self):
        self.client = CloudModelClient()
        logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
        self.logger = logging.getLogger("SentinelReasoning")

    async def _call_target(self, target: str, prompt: str) -> str:
        """Route request to the specific model."""
        try:
            if target == "groq": return await self.client.call_groq(prompt)
            if target == "llama70b": return await self.client.call_llama70b(prompt)
            if target == "qwen": return await self.client.call_qwenvl(prompt)
            return "ERROR: Unknown Target"
        except Exception as e:
            return f"ERROR_EXCEPTION: {str(e)}"

    # --------------------------------------------------------------------------
    # MODE A: VERIFIED CHAIN OF THOUGHT (vCoT)
    # --------------------------------------------------------------------------
    # This is "Critical Thinking" with a supervisor. 
    # It generates a thought, then immediately asks: "Is this logic sound?"
    # --------------------------------------------------------------------------
    async def solve_with_verification(self, target: str, problem: str, max_steps: int = 4) -> ReasoningResult:
        print(f"    [~] Initiating Verified CoT on {target}...")
        
        trace = []
        current_context = f"Problem: {problem}\n"
        start_time = asyncio.get_event_loop().time()

        for i in range(max_steps):
            # 1. Generate The Thought
            step_prompt = f"""
            {current_context}
            [TASK]: Provide the next logical step to solve this problem. 
            Do not jump to the conclusion. Be precise.
            [OUTPUT]: Just the step.
            """
            thought = await self._call_target(target, step_prompt)

            # 2. Verify The Thought (Self-Correction)
            verify_prompt = f"""
            [CONTEXT]: {current_context}
            [PROPOSED STEP]: {thought}
            [TASK]: Critical Review. Is this step logically sound and factual? 
            If YES, output "CONFIRMED". If NO, output the correction.
            """
            review = await self._call_target(target, verify_prompt)
            
            # 3. Process Verification
            is_valid = "CONFIRMED" in review.upper()
            final_content = thought if is_valid else f"{thought} (Correction: {review})"
            
            trace.append(ReasoningStep(
                step_id=i+1,
                thought_content=final_content,
                verification_passed=is_valid,
                correction=review if not is_valid else None,
                confidence=95 if is_valid else 40
            ))
            
            current_context += f"Step {i+1}: {final_content}\n"
            print(f"      -> Step {i+1}: {'✓' if is_valid else '⚠'} {final_content[:60]}...")

        # 4. Final Synthesis
        final_prompt = f"{current_context}\n\n[TASK]: Based on the verified steps above, provide the final solution."
        final_answer = await self._call_target(target, final_prompt)

        return ReasoningResult(
            mode="Verified_CoT",
            target_model=target,
            input_problem=problem,
            final_answer=final_answer,
            trace=trace,
            execution_time=asyncio.get_event_loop().time() - start_time
        )

    # --------------------------------------------------------------------------
    # MODE B: STRATEGIC TREE OF THOUGHT (sToT)
    # --------------------------------------------------------------------------
    # This explores multiple parallel solutions, scores them, and picks the winner.
    # --------------------------------------------------------------------------
    async def solve_with_strategy_tree(self, target: str, problem: str) -> ReasoningResult:
        print(f"    [~] Building Strategy Tree on {target}...")
        start_time = asyncio.get_event_loop().time()
        
        # 1. Generate 3 Distinct Approaches
        init_prompt = f"""
        Problem: {problem}
        [TASK]: Propose 3 distinct, competing strategies to solve this.
        Format: 
        1. [Strategy Name]: [Brief Description]
        2. [Strategy Name]: [Brief Description]
        3. [Strategy Name]: [Brief Description]
        """
        raw_strategies = await self._call_target(target, init_prompt)
        
        # Simple parser to mock branches (In production, use Regex)
        # We assume the model follows instructions roughly
        branches = []
        strategies = raw_strategies.split("\n")[:3] # Take first 3 lines roughly
        
        for i, strat in enumerate(strategies):
            if len(strat) < 5: continue 
            
            # 2. Evaluate Each Branch (The Pruning Phase)
            eval_prompt = f"""
            Problem: {problem}
            Proposed Strategy: {strat}
            [TASK]: Rate this strategy on two metrics (0-100):
            1. Feasibility (How easy is it?)
            2. Impact (How effective is it?)
            Output format: Feasibility: XX, Impact: XX
            """
            eval_res = await self._call_target(target, eval_prompt)
            
            # Extract scores (naive parsing for demo)
            try:
                feasibility = int(re.search(r"Feasibility:? (\d+)", eval_res).group(1))
                impact = int(re.search(r"Impact:? (\d+)", eval_res).group(1))
            except:
                feasibility, impact = 50, 50 # Fallback
            
            final_score = (feasibility * 0.4) + (impact * 0.6) # Impact matters more
            
            branches.append(StrategicBranch(
                branch_id=i+1,
                approach_name=strat[:50],
                steps_taken=[strat],
                feasibility_score=feasibility,
                impact_score=impact,
                final_score=final_score
            ))
            print(f"      -> Branch {i+1}: Score {final_score:.1f} ({strat[:40]}...)")

        # 3. Select Best Branch
        if not branches:
            best_branch = None
            final_answer = "Error: No valid strategies generated."
        else:
            best_branch = max(branches, key=lambda b: b.final_score)
            
            # 4. Refine Best Solution
            refine_prompt = f"""
            Problem: {problem}
            Selected Strategy: {best_branch.approach_name}
            [TASK]: This strategy was rated highest. Flesh it out into a detailed final answer.
            """
            final_answer = await self._call_target(target, refine_prompt)

        return ReasoningResult(
            mode="Strategic_ToT",
            target_model=target,
            input_problem=problem,
            final_answer=final_answer,
            trace=branches,
            execution_time=asyncio.get_event_loop().time() - start_time
        )

    # --------------------------------------------------------------------------
    # MAIN ANALYZER LOOP
    # --------------------------------------------------------------------------
    async def run_reasoning_campaign(self, problem: str):
        # We test on one 'Strong' model (Llama 3.3 70B) and one 'Fast' model (Groq) for contrast
        targets = ["llama70b", "groq"] 
        
        print(f"\n{'='*80}")
        print(f"SENTINEL-X REASONING CORE | PROBLEM: {problem[:40]}...")
        print(f"{'='*80}\n")
        
        for model in targets:
            print(f">>> REASONING AGENT: {model.upper()}")
            
            # 1. Run Verified CoT (Deep Dive)
            cot_result = await self.solve_with_verification(model, problem)
            print(f"    [+] vCoT Result: {cot_result.final_answer[:100]}...")
            
            # 2. Run Strategic ToT (Broad Search)
            tot_result = await self.solve_with_strategy_tree(model, problem)
            print(f"    [+] sToT Winner: Branch {max(tot_result.trace, key=lambda x:x.final_score).branch_id} (Score: {max(tot_result.trace, key=lambda x:x.final_score).final_score})")
            print("-" * 40)

# ==============================================================================
# ENTRY POINT
# ==============================================================================
async def main():
    engine = IndustrialReasoningEngine()
    
    # A complex problem that requires structured thinking, not just knowledge retrieval.
    test_problem = "We need to migrate a legacy monolithic banking database (SQL) to a microservices cloud architecture (NoSQL) with ZERO downtime. How do we do it?"
    
    await engine.run_reasoning_campaign(test_problem)

if __name__ == "__main__":
    asyncio.run(main())