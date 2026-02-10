# """
# Test suite for critical_thinking.py module

# This file contains tests to validate the Chain of Thought (CoT) and
# Tree of Thought (ToT) implementations in the critical thinking module.
# """

# import asyncio
# import sys
# import os

# # Ensure root is on path
# sys.path.insert(0, os.path.dirname(__file__))

# from critical_thinking import (
#     think,
#     CriticalThinkingProcessor,
#     ChainOfThoughtProcessor,
#     TreeOfThoughtProcessor,
#     call_model,
#     ThinkingResult
# )


# # ============================================================
# # TEST QUERIES
# # ============================================================

# TEST_QUERIES_COT = [
#     "What are the main causes of climate change?",
#     "How does encryption protect data privacy?",
#     "What steps are needed to start a small business?",
# ]

# TEST_QUERIES_TOT = [
#     "What are innovative solutions to reduce plastic waste?",
#     "How can we balance economic growth with environmental protection?",
#     "What creative approaches could improve mental health support?",
# ]

# TEST_QUERIES_AUTO = [
#     "Should genetic engineering of humans be allowed?",
#     "Calculate the compound interest on $1000 at 5% over 10 years.",
#     "What makes a good leader?",
# ]


# # ============================================================
# # TEST FUNCTIONS
# # ============================================================

# async def test_cot_processor():
#     """Test Chain of Thought processor."""
#     print("\n" + "=" * 80)
#     print("TEST: Chain of Thought Processor")
#     print("=" * 80)
    
#     processor = ChainOfThoughtProcessor()
    
#     for query in TEST_QUERIES_COT:
#         print(f"\nQuery: {query}")
#         result = await processor.process(query, num_steps=3)
        
#         assert result is not None, "Result should not be None"
#         assert result.mode == "chain_of_thought", "Mode should be CoT"
#         assert result.original_prompt == query, "Prompt should match"
#         assert len(result.reasoning_trace) > 0, "Should have reasoning steps"
#         assert result.final_answer, "Should have final answer"
        
#         print(f"✓ Passed - {len(result.reasoning_trace)} steps generated")
    
#     print("\n✅ All Chain of Thought tests passed!")


# async def test_tot_processor():
#     """Test Tree of Thought processor."""
#     print("\n" + "=" * 80)
#     print("TEST: Tree of Thought Processor")
#     print("=" * 80)
    
#     processor = TreeOfThoughtProcessor()
    
#     NUM_BRANCHES = 3  # Expected number of branches to generate
    
#     for query in TEST_QUERIES_TOT:
#         print(f"\nQuery: {query}")
#         result = await processor.process(query, num_branches=NUM_BRANCHES, expansion_depth=2)
        
#         assert result is not None, "Result should not be None"
#         assert result.mode == "tree_of_thought", "Mode should be ToT"
#         assert result.original_prompt == query, "Prompt should match"
#         assert len(result.reasoning_trace) == NUM_BRANCHES, f"Should have {NUM_BRANCHES} branches"
#         assert result.final_answer, "Should have final answer"
        
#         # Check branch properties
#         for branch in result.reasoning_trace:
#             assert branch.evaluation_score >= 0.0, "Score should be >= 0"
#             assert branch.evaluation_score <= 1.0, "Score should be <= 1"
#             assert len(branch.path) > 0, "Branch should have reasoning path"
        
#         print(f"✓ Passed - {len(result.reasoning_trace)} branches evaluated")
    
#     print("\n✅ All Tree of Thought tests passed!")


# async def test_main_interface():
#     """Test the main think() interface function."""
#     print("\n" + "=" * 80)
#     print("TEST: Main Interface Function")
#     print("=" * 80)
    
#     # Test explicit CoT mode
#     print("\nTesting explicit CoT mode...")
#     result = await think(
#         "What is the water cycle?",
#         mode="cot",
#         num_steps=3,
#         verbose=False
#     )
#     assert result.mode == "chain_of_thought", "Should use CoT mode"
#     print("✓ CoT mode works")
    
#     # Test explicit ToT mode
#     print("\nTesting explicit ToT mode...")
#     result = await think(
#         "How can we improve education?",
#         mode="tot",
#         num_branches=2,
#         verbose=False
#     )
#     assert result.mode == "tree_of_thought", "Should use ToT mode"
#     print("✓ ToT mode works")
    
#     # Test auto mode
#     print("\nTesting auto mode selection...")
#     result = await think(
#         "What are prime numbers?",
#         mode="auto",
#         verbose=False
#     )
#     assert result is not None, "Should return result"
#     assert result.mode in ["chain_of_thought", "tree_of_thought"], "Should select valid mode"
#     print(f"✓ Auto mode works (selected: {result.mode})")
    
#     print("\n✅ All main interface tests passed!")


# async def test_critical_thinking_processor():
#     """Test the CriticalThinkingProcessor class."""
#     print("\n" + "=" * 80)
#     print("TEST: CriticalThinkingProcessor Class")
#     print("=" * 80)
    
#     processor = CriticalThinkingProcessor()
    
#     # Test process_with_mode
#     print("\nTesting process_with_mode...")
#     result = await processor.process_with_mode(
#         "What is photosynthesis?",
#         mode="cot",
#         num_steps=2
#     )
#     assert result.mode == "chain_of_thought", "Should process with CoT"
#     print("✓ process_with_mode works")
    
#     # Test auto_process
#     print("\nTesting auto_process...")
#     result = await processor.auto_process("Explain quantum computing")
#     assert result is not None, "Should return result"
#     print("✓ auto_process works")
    
#     # Test format_result
#     print("\nTesting format_result...")
#     formatted = processor.format_result(result)
#     assert len(formatted) > 0, "Should produce formatted output"
#     assert "CRITICAL THINKING ANALYSIS" in formatted, "Should have header"
#     assert "FINAL ANSWER" in formatted, "Should have final answer section"
#     print("✓ format_result works")
    
#     print("\n✅ All CriticalThinkingProcessor tests passed!")


# async def test_custom_model_function():
#     """Test using a custom model function."""
#     print("\n" + "=" * 80)
#     print("TEST: Custom Model Function")
#     print("=" * 80)
    
#     # Define a custom model function
#     async def custom_model(prompt, **kwargs):
#         """Custom model that returns a specific format."""
#         await asyncio.sleep(0.05)
#         return f"Custom response to: {prompt[:30]}..."
    
#     processor = CriticalThinkingProcessor(model_fn=custom_model)
#     result = await processor.process_with_mode(
#         "Test query",
#         mode="cot",
#         num_steps=2
#     )
    
#     assert result is not None, "Should work with custom model"
#     assert "Custom response" in result.final_answer, "Should use custom model"
#     print("✓ Custom model function works")
    
#     print("\n✅ Custom model function test passed!")


# async def test_error_handling():
#     """Test error handling for invalid inputs."""
#     print("\n" + "=" * 80)
#     print("TEST: Error Handling")
#     print("=" * 80)
    
#     processor = CriticalThinkingProcessor()
    
#     # Test invalid mode
#     print("\nTesting invalid mode...")
#     try:
#         await processor.process_with_mode("test", mode="invalid_mode")
#         assert False, "Should raise ValueError"
#     except ValueError as e:
#         assert "Unknown mode" in str(e), "Should have proper error message"
#         print("✓ Invalid mode raises ValueError")
    
#     print("\n✅ Error handling tests passed!")


# async def test_data_structures():
#     """Test data structure creation and properties."""
#     print("\n" + "=" * 80)
#     print("TEST: Data Structures")
#     print("=" * 80)
    
#     # Test that call_model can be called
#     print("\nTesting call_model...")
#     response = await call_model("test prompt")
#     assert response is not None, "call_model should return response"
#     assert isinstance(response, str), "Response should be string"
#     print("✓ call_model function works")
    
#     print("\n✅ Data structure tests passed!")


# # ============================================================
# # TEST RUNNER
# # ============================================================

# async def run_all_tests():
#     """Run all test suites."""
#     print("\n" + "#" * 80)
#     print("### CRITICAL THINKING MODULE TEST SUITE")
#     print("#" * 80)
    
#     try:
#         await test_data_structures()
#         await test_cot_processor()
#         await test_tot_processor()
#         await test_main_interface()
#         await test_critical_thinking_processor()
#         await test_custom_model_function()
#         await test_error_handling()
        
#         print("\n" + "#" * 80)
#         print("### ALL TESTS PASSED! ✅")
#         print("#" * 80)
        
#     except Exception as e:
#         print("\n" + "#" * 80)
#         print("### TEST FAILED! ❌")
#         print("#" * 80)
#         print(f"\nError: {e}")
#         import traceback
#         traceback.print_exc()
#         raise


# # ============================================================
# # ENTRY POINT
# # ============================================================

# if __name__ == "__main__":
#     asyncio.run(run_all_tests())

import asyncio
import unittest

from critical_thinking import IndustrialReasoningEngine

class TestCriticalThinking(unittest.TestCase):

    def setUp(self):
        self.engine = IndustrialReasoningEngine()

    def test_solve_with_verification(self):
        result = asyncio.run(self.engine.solve_with_verification("groq", "2+2=?", max_steps=1))
        self.assertIsNotNone(result.final_answer)
        print(f"Verified CoT: {result.final_answer}")

    def test_solve_with_strategy_tree(self):
        result = asyncio.run(self.engine.solve_with_strategy_tree("groq", "2+2=?"))
        self.assertIsNotNone(result.final_answer)
        print(f"Strategy ToT: {result.final_answer}")

if __name__ == "__main__":
    unittest.main()