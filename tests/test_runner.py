"""Tests for the runner module."""

import os
import asyncio
import unittest
from unittest import mock
import json
import pytest
from types import SimpleNamespace

from agentoptim.evalset import EvalSet
from agentoptim.runner import (
    EvalResult,
    EvalResults,
    run_evalset,
    evaluate_question,
)


class TestRunner(unittest.TestCase):
    """Test suite for runner functionality."""
    
    def setUp(self):
        """Set up the test environment."""
        # Sample data for tests
        self.test_evalset = EvalSet(
            id="test-evalset-id",
            name="Test EvalSet",
            template="""
            Given this conversation:
            {{ conversation }}
            
            Please answer the following yes/no question about the final assistant response:
            {{ eval_question }}
            
            Return a JSON object with the following format:
            {"judgment": true} for yes or {"judgment": false} for no.
            """,
            questions=["Is the response helpful?", "Is the response clear?"],
            description="Test description"
        )
        
        self.test_conversation = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "How do I reset my password?"},
            {"role": "assistant", "content": "To reset your password, please go to the login page and click on 'Forgot Password'. You'll receive an email with instructions to create a new password."}
        ]
        
        # Mock API response
        self.mock_api_success_response = {
            "choices": [
                {
                    "message": {
                        "content": '{"reasoning": "The response provides clear instructions on how to reset a password, including going to the login page, clicking on Forgot Password, and checking spam folders if needed.", "judgment": true, "confidence": 0.85}'
                    }
                }
            ]
        }
        
        self.mock_api_failure_response = {
            "error": "API connection error"
        }
        
    @mock.patch('agentoptim.runner.get_evalset')
    @mock.patch('agentoptim.runner.call_llm_api')
    async def test_run_evalset_success(self, mock_call_llm_api, mock_get_evalset):
        """Test running an EvalSet successfully."""
        # Mock get_evalset to return our test EvalSet
        mock_get_evalset.return_value = self.test_evalset
        
        # Mock call_llm_api to return success responses
        mock_call_llm_api.return_value = self.mock_api_success_response
        
        # Run the EvalSet and check results
        result = await run_evalset(
            evalset_id="test-evalset-id",
            conversation=self.test_conversation,
            judge_model="test-model",
            max_parallel=2
        )
        
        # Check basic success structure
        self.assertEqual(result.get("status"), "success")
        self.assertEqual(result.get("evalset_id"), "test-evalset-id")
        self.assertEqual(result.get("evalset_name"), "Test EvalSet")
        self.assertEqual(result.get("judge_model"), "test-model")
        
        # Check that results array exists and has expected length
        self.assertIn("results", result)
        results = result.get("results", [])
        self.assertEqual(len(results), 2)  # Should have one result per question
        
        # Check summary 
        self.assertIn("summary", result)
        summary = result.get("summary", {})
        self.assertEqual(summary.get("total_questions"), 2)
        self.assertEqual(summary.get("yes_count"), 2)  # Both should be "yes" from our mock
        
        # Check formatted message is present
        self.assertIn("formatted_message", result)
    
    @mock.patch('agentoptim.runner.get_evalset')
    async def test_run_evalset_missing_evalset(self, mock_get_evalset):
        """Test running an EvalSet with a non-existent ID."""
        # Mock get_evalset to return None (evalset not found)
        mock_get_evalset.return_value = None
        
        # Mock the format_error function to return expected format
        with mock.patch('agentoptim.runner.format_error', 
                        return_value={"status": "error", "message": "EvalSet not found"}):
            
            # Run the EvalSet and check error response
            result = await run_evalset(
                evalset_id="nonexistent-id",
                conversation=self.test_conversation
            )
            
            # Check error structure
            self.assertEqual(result.get("status"), "error")
            self.assertIn("not found", result.get("message", ""))
    
    @mock.patch('agentoptim.runner.get_evalset')
    async def test_run_evalset_invalid_conversation(self, mock_get_evalset):
        """Test running an EvalSet with an invalid conversation."""
        # Mock get_evalset to return our test EvalSet
        mock_get_evalset.return_value = self.test_evalset
        
        # Mock the format_error function to return expected format
        with mock.patch('agentoptim.runner.format_error', 
                        return_value={"status": "error", "message": "Conversation must be non-empty"}):
            
            # Test with empty conversation
            result = await run_evalset(
                evalset_id="test-evalset-id",
                conversation=[]
            )
            self.assertEqual(result.get("status"), "error")
            self.assertIn("non-empty", result.get("message", ""))
        
        # Mock the format_error function again with different error message
        with mock.patch('agentoptim.runner.format_error', 
                        return_value={"status": "error", "message": "Each conversation message must have role and content"}):
            
            # Test with invalid conversation structure
            result = await run_evalset(
                evalset_id="test-evalset-id",
                conversation=[{"invalid": "structure"}]
            )
            self.assertEqual(result.get("status"), "error")
            self.assertIn("role", result.get("message", ""))
    
    @mock.patch('agentoptim.runner.call_llm_api')
    async def test_evaluate_question_success(self, mock_call_llm_api):
        """Test evaluating a single question successfully."""
        # Mock call_llm_api to return success response
        mock_call_llm_api.return_value = self.mock_api_success_response
        
        # Test with reasoning (default)
        result = await evaluate_question(
            conversation=self.test_conversation,
            question="Is the response helpful?",
            template=self.test_evalset.template,
            judge_model="test-model"
        )
        
        # Check result structure with reasoning
        self.assertIsInstance(result, EvalResult)
        self.assertEqual(result.question, "Is the response helpful?")
        self.assertEqual(result.judgment, True)  # Should be True from our mock
        self.assertIsNotNone(result.confidence)
        self.assertIsNotNone(result.reasoning)  # Reasoning should be present
        self.assertIsNone(result.error)
        
        # Test with omit_reasoning=True
        result_no_reasoning = await evaluate_question(
            conversation=self.test_conversation,
            question="Is the response helpful?",
            template=self.test_evalset.template,
            judge_model="test-model",
            omit_reasoning=True
        )
        
        # Check result structure without reasoning
        self.assertIsInstance(result_no_reasoning, EvalResult)
        self.assertEqual(result_no_reasoning.question, "Is the response helpful?")
        self.assertEqual(result_no_reasoning.judgment, True)
        self.assertIsNotNone(result_no_reasoning.confidence)
        self.assertIsNone(result_no_reasoning.error)
    
    @mock.patch('agentoptim.runner.call_llm_api')
    async def test_evaluate_question_api_error(self, mock_call_llm_api):
        """Test evaluating a question with API error."""
        # Mock call_llm_api to return failure response
        mock_call_llm_api.return_value = self.mock_api_failure_response
        
        # Evaluate a question
        result = await evaluate_question(
            conversation=self.test_conversation,
            question="Is the response helpful?",
            template=self.test_evalset.template,
            judge_model="test-model"
        )
        
        # Check result structure for error
        self.assertIsInstance(result, EvalResult)
        self.assertEqual(result.question, "Is the response helpful?")
        self.assertIsNone(result.judgment)
        self.assertIsNone(result.confidence)
        self.assertIsNotNone(result.error)
        self.assertIn("API connection error", result.error)
    
    # Using unittest approach for consistency with other tests
    @mock.patch('agentoptim.runner.call_llm_api')
    async def test_evaluate_question_missing_fields(self, mock_call_llm_api):
        """Test evaluating a question with missing confidence and reasoning fields."""
        # Skipping this test in unittest mode - we'll add a pytest-specific test below
        pass
    
    def test_eval_result_model(self):
        """Test the EvalResult model functionality."""
        # Create an EvalResult
        result = EvalResult(
            question="Is the response helpful?",
            judgment=True,
            confidence=0.85,
            reasoning="The response provides clear instructions."
        )
        
        # Check basic properties
        self.assertEqual(result.question, "Is the response helpful?")
        self.assertEqual(result.judgment, True)
        self.assertEqual(result.confidence, 0.85)
        self.assertEqual(result.reasoning, "The response provides clear instructions.")
        self.assertIsNone(result.error)
        
        # Create an error result
        error_result = EvalResult(
            question="Is the response helpful?",
            error="API error"
        )
        
        # Check error properties
        self.assertEqual(error_result.question, "Is the response helpful?")
        self.assertIsNone(error_result.judgment)
        self.assertIsNone(error_result.confidence)
        self.assertIsNone(error_result.reasoning)
        self.assertEqual(error_result.error, "API error")
    
    def test_eval_results_model(self):
        """Test the EvalResults model functionality."""
        # Create some EvalResults
        results = [
            EvalResult(question="Q1", judgment=True, confidence=0.85, reasoning="Reasoning 1"),
            EvalResult(question="Q2", judgment=False, confidence=0.75, reasoning="Reasoning 2"),
            EvalResult(question="Q3", error="API error")
        ]
        
        # Create an EvalResults
        eval_results = EvalResults(
            evalset_id="test-evalset-id",
            evalset_name="Test EvalSet",
            results=results,
            conversation=self.test_conversation,
            summary={
                "total_questions": 3,
                "successful_evaluations": 2,
                "yes_count": 1,
                "no_count": 1,
                "error_count": 1,
                "yes_percentage": 50.0
            }
        )
        
        # Check basic properties
        self.assertEqual(eval_results.evalset_id, "test-evalset-id")
        self.assertEqual(eval_results.evalset_name, "Test EvalSet")
        self.assertEqual(len(eval_results.results), 3)
        self.assertEqual(eval_results.results[0].question, "Q1")
        self.assertEqual(eval_results.conversation, self.test_conversation)
        self.assertEqual(eval_results.summary.get("yes_percentage"), 50.0)


# Create a run function for async tests
def run_async_test(coroutine):
    """Run an async coroutine as a test."""
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(coroutine)


# Override the TestRunner.run method to enable async tests
def async_test(test_case):
    """Decorator for async test methods."""
    def wrapper(*args, **kwargs):
        return run_async_test(test_case(*args, **kwargs))
    return wrapper


# Apply the decorator to async test methods
TestRunner.test_run_evalset_success = async_test(TestRunner.test_run_evalset_success)
TestRunner.test_run_evalset_missing_evalset = async_test(TestRunner.test_run_evalset_missing_evalset)
TestRunner.test_run_evalset_invalid_conversation = async_test(TestRunner.test_run_evalset_invalid_conversation)
TestRunner.test_evaluate_question_success = async_test(TestRunner.test_evaluate_question_success)
TestRunner.test_evaluate_question_api_error = async_test(TestRunner.test_evaluate_question_api_error)
TestRunner.test_evaluate_question_missing_fields = async_test(TestRunner.test_evaluate_question_missing_fields)


# Add a standalone pytest test for evaluation with missing fields
@pytest.mark.asyncio
async def test_run_evalset_omit_reasoning():
    """Test running an EvalSet with omit_reasoning."""
    import agentoptim.runner
    from agentoptim.runner import get_evalset
    
    # Create a mock EvalSet get function to avoid file I/O
    original_get_evalset = agentoptim.runner.get_evalset
    
    def mock_get_evalset(evalset_id):
        return SimpleNamespace(
            id="test-evalset-id",
            name="Test EvalSet",
            template="Given this conversation: {{ conversation }} Answer: {{ eval_question }}",
            questions=[
                "Is the response helpful?",
                "Is the response accurate?"
            ]
        )
    
    # Create a mock evaluate_question function
    original_evaluate_question = agentoptim.runner.evaluate_question
    
    async def mock_evaluate_question(conversation, question, template, judge_model, omit_reasoning=False):
        # Return different results based on omit_reasoning
        if omit_reasoning:
            # No reasoning when omitted
            return EvalResult(
                question=question,
                judgment=True,
                confidence=0.9,
                reasoning=None
            )
        else:
            # Include reasoning by default
            return EvalResult(
                question=question,
                judgment=True,
                confidence=0.9,
                reasoning="This is a good response because it answers the question directly."
            )
    
    # Apply the patches
    agentoptim.runner.get_evalset = mock_get_evalset
    agentoptim.runner.evaluate_question = mock_evaluate_question
    
    try:
        # Define a test conversation
        conversation = [
            {"role": "user", "content": "How do I reset my password?"},
            {"role": "assistant", "content": "Go to the login page and click 'Forgot Password'."}
        ]
        
        # Test with omit_reasoning=False (default)
        result_with_reasoning = await run_evalset(
            evalset_id="test-evalset-id",
            conversation=conversation,
            judge_model="test-model",
            max_parallel=1,
            omit_reasoning=False
        )
        
        # Test with omit_reasoning=True
        result_without_reasoning = await run_evalset(
            evalset_id="test-evalset-id",
            conversation=conversation,
            judge_model="test-model",
            max_parallel=1,
            omit_reasoning=True
        )
        
        # Verify results with reasoning
        assert result_with_reasoning["status"] == "success"
        assert len(result_with_reasoning["results"]) == 2
        assert "reasoning" in result_with_reasoning["results"][0]
        assert result_with_reasoning["results"][0]["reasoning"] is not None
        
        # Verify formatted message includes reasoning
        assert "Reasoning" in result_with_reasoning["formatted_message"]
        
        # Verify results without reasoning
        assert result_without_reasoning["status"] == "success"
        assert len(result_without_reasoning["results"]) == 2
        assert "reasoning" not in result_without_reasoning["results"][0]
        
        # Verify formatted message doesn't include reasoning
        assert "Reasoning" not in result_without_reasoning["formatted_message"]
    
    finally:
        # Restore original functions
        agentoptim.runner.get_evalset = original_get_evalset
        agentoptim.runner.evaluate_question = original_evaluate_question


@pytest.mark.asyncio
async def test_evaluate_question_missing_fields_standalone():
    """Test handling of responses with missing confidence and reasoning fields."""
    # Create a test conversation
    conversation = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "How do I reset my password?"},
        {"role": "assistant", "content": "To reset your password, go to the login page."}
    ]
    
    # Template similar to what's used in the TestRunner
    template = """
    Given this conversation:
    {{ conversation }}
    
    Please answer the following yes/no question about the final assistant response:
    {{ eval_question }}
    
    Return a JSON object with the following format:
    {"judgment": true} for yes or {"judgment": false} for no.
    """
    
    # Mock response with only judgment field
    mock_response = {
        "choices": [
            {
                "message": {
                    "content": '{"judgment": true}'
                }
            }
        ]
    }
    
    # Patch the API call
    with mock.patch('agentoptim.runner.call_llm_api') as mock_call_llm_api:
        mock_call_llm_api.return_value = mock_response
        
        # Test with reasoning enabled (default)
        result = await evaluate_question(
            conversation=conversation,
            question="Is the response helpful?",
            template=template,
            judge_model="test-model"
        )
        
        # Check result structure - should use default values
        assert isinstance(result, EvalResult)
        assert result.question == "Is the response helpful?"
        assert result.judgment is True
        assert result.confidence == 0.7  # Default confidence
        assert result.reasoning == "No reasoning provided by evaluation model"  # Default reasoning
        
        # Test with omit_reasoning=True
        result_no_reasoning = await evaluate_question(
            conversation=conversation,
            question="Is the response helpful?",
            template=template,
            judge_model="test-model",
            omit_reasoning=True
        )
        
        # Check result structure - reasoning should be None
        assert isinstance(result_no_reasoning, EvalResult)
        assert result_no_reasoning.question == "Is the response helpful?"
        assert result_no_reasoning.judgment is True
        assert result_no_reasoning.confidence == 0.7  # Default confidence
        assert result_no_reasoning.reasoning is None  # Reasoning should be None
        assert result_no_reasoning.error is None


@pytest.mark.asyncio
async def test_run_evalset_max_parallel_limits():
    """Test run_evalset with different max_parallel values to ensure they limit concurrency."""
    import agentoptim.runner
    from agentoptim.runner import get_evalset
    
    # Create a mock EvalSet
    test_evalset = SimpleNamespace(
        id="test-evalset-id",
        name="Test EvalSet with Many Questions",
        template="Given this conversation: {{ conversation }} Answer: {{ eval_question }}",
        questions=[f"Question {i}" for i in range(10)]  # Create 10 questions
    )
    
    # Mock the get_evalset function
    original_get_evalset = agentoptim.runner.get_evalset
    
    def mock_get_evalset(evalset_id):
        return test_evalset
    
    # Create a mock evaluate_question function that tracks calls
    original_evaluate_question = agentoptim.runner.evaluate_question
    
    # For tracking concurrent executions
    execution_tracker = {
        "current": 0,
        "max": 0,
        "calls": 0
    }
    
    async def mock_evaluate_question(conversation, question, template, judge_model, omit_reasoning=False):
        # Increment tracking variables
        execution_tracker["current"] += 1
        execution_tracker["calls"] += 1
        execution_tracker["max"] = max(execution_tracker["max"], execution_tracker["current"])
        
        # Simulate work
        await asyncio.sleep(0.1)
        
        # Create result
        result = EvalResult(
            question=question,
            judgment=True,
            confidence=0.9,
            reasoning="Reasoning for " + question
        )
        
        # Decrement current executions
        execution_tracker["current"] -= 1
        
        return result
    
    # Apply the patches
    agentoptim.runner.get_evalset = mock_get_evalset
    agentoptim.runner.evaluate_question = mock_evaluate_question
    
    try:
        # Define a test conversation
        conversation = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"}
        ]
        
        # Test with max_parallel=1 (should execute sequentially)
        execution_tracker["current"] = 0
        execution_tracker["max"] = 0
        execution_tracker["calls"] = 0
        
        await run_evalset(
            evalset_id="test-evalset-id",
            conversation=conversation,
            judge_model="test-model",
            max_parallel=1
        )
        
        assert execution_tracker["max"] == 1
        assert execution_tracker["calls"] == 10
        
        # Test with max_parallel=3 (should have up to 3 concurrent tasks)
        execution_tracker["current"] = 0
        execution_tracker["max"] = 0
        execution_tracker["calls"] = 0
        
        await run_evalset(
            evalset_id="test-evalset-id",
            conversation=conversation,
            judge_model="test-model",
            max_parallel=3
        )
        
        assert execution_tracker["max"] == 3
        assert execution_tracker["calls"] == 10
        
        # Test with max_parallel=10 (should process all tasks concurrently)
        execution_tracker["current"] = 0
        execution_tracker["max"] = 0
        execution_tracker["calls"] = 0
        
        await run_evalset(
            evalset_id="test-evalset-id",
            conversation=conversation,
            judge_model="test-model",
            max_parallel=10
        )
        
        assert execution_tracker["max"] == 10
        assert execution_tracker["calls"] == 10
        
        # Test with max_parallel larger than questions count
        execution_tracker["current"] = 0
        execution_tracker["max"] = 0
        execution_tracker["calls"] = 0
        
        await run_evalset(
            evalset_id="test-evalset-id",
            conversation=conversation,
            judge_model="test-model",
            max_parallel=20
        )
        
        assert execution_tracker["max"] == 10  # Should be limited by the number of questions
        assert execution_tracker["calls"] == 10
        
    finally:
        # Restore original functions
        agentoptim.runner.get_evalset = original_get_evalset
        agentoptim.runner.evaluate_question = original_evaluate_question


@pytest.mark.asyncio
async def test_run_evalset_invalid_max_parallel():
    """Test run_evalset with invalid max_parallel values."""
    import agentoptim.runner
    from agentoptim.runner import get_evalset
    
    # Create a mock EvalSet
    test_evalset = SimpleNamespace(
        id="test-evalset-id",
        name="Test EvalSet with Many Questions",
        template="Given this conversation: {{ conversation }} Answer: {{ eval_question }}",
        questions=[f"Question {i}" for i in range(3)]  # Create 3 questions
    )
    
    # Mock the get_evalset function
    original_get_evalset = agentoptim.runner.get_evalset
    
    def mock_get_evalset(evalset_id):
        return test_evalset
    
    # Create a mock format_error function
    original_format_error = agentoptim.runner.format_error
    
    def mock_format_error(message):
        return {"status": "error", "message": message}
    
    # Create a mock evaluate_question function
    original_evaluate_question = agentoptim.runner.evaluate_question
    
    async def mock_evaluate_question(conversation, question, template, judge_model, omit_reasoning=False):
        return EvalResult(
            question=question,
            judgment=True,
            confidence=0.9,
            reasoning="Reasoning for " + question
        )
    
    # Apply the patches
    agentoptim.runner.get_evalset = mock_get_evalset
    agentoptim.runner.format_error = mock_format_error
    agentoptim.runner.evaluate_question = mock_evaluate_question
    
    try:
        # Define a test conversation
        conversation = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"}
        ]
        
        # Test with max_parallel=0 (should return an error)
        result_zero = await run_evalset(
            evalset_id="test-evalset-id",
            conversation=conversation,
            judge_model="test-model",
            max_parallel=0
        )
        
        # Check that we get an error for max_parallel=0
        assert result_zero["status"] == "error"
        assert "max_parallel" in result_zero["message"].lower()
        
        # Test with max_parallel=-1 (should return an error)
        result_negative = await run_evalset(
            evalset_id="test-evalset-id",
            conversation=conversation,
            judge_model="test-model",
            max_parallel=-1
        )
        
        # Check that we get an error for max_parallel=-1
        assert result_negative["status"] == "error"
        assert "max_parallel" in result_negative["message"].lower()
        
        # Test with max_parallel=1 (should work)
        result_valid = await run_evalset(
            evalset_id="test-evalset-id",
            conversation=conversation,
            judge_model="test-model",
            max_parallel=1
        )
        
        # Check that normal max_parallel works
        assert result_valid["status"] == "success"
        assert len(result_valid["results"]) == 3
        
    finally:
        # Restore original functions
        agentoptim.runner.get_evalset = original_get_evalset
        agentoptim.runner.format_error = original_format_error
        agentoptim.runner.evaluate_question = original_evaluate_question


@pytest.mark.asyncio
async def test_run_evalset_empty_questions():
    """Test run_evalset with an EvalSet that has no questions."""
    import agentoptim.runner
    from agentoptim.runner import get_evalset
    
    # Create a mock EvalSet with no questions
    test_evalset = SimpleNamespace(
        id="empty-questions-evalset",
        name="Empty Questions EvalSet",
        template="Given this conversation: {{ conversation }} Answer: {{ eval_question }}",
        questions=[]  # Empty questions list
    )
    
    # Mock the get_evalset function
    original_get_evalset = agentoptim.runner.get_evalset
    
    def mock_get_evalset(evalset_id):
        return test_evalset
    
    # Mock the format_error function
    original_format_error = agentoptim.runner.format_error
    
    def mock_format_error(message):
        return {"status": "error", "message": message}
    
    # Apply the patches
    agentoptim.runner.get_evalset = mock_get_evalset
    agentoptim.runner.format_error = mock_format_error
    
    try:
        # Define a test conversation
        conversation = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"}
        ]
        
        # Run the evalset with empty questions
        result = await run_evalset(
            evalset_id="empty-questions-evalset",
            conversation=conversation,
            judge_model="test-model"
        )
        
        # Check that the result has the expected structure
        assert result["status"] == "success"
        assert result["evalset_id"] == "empty-questions-evalset"
        assert result["evalset_name"] == "Empty Questions EvalSet"
        assert result["results"] == []  # Should have empty results
        
        # Check that the summary has zeros
        assert result["summary"]["total_questions"] == 0
        assert result["summary"]["yes_count"] == 0
        assert result["summary"]["no_count"] == 0
        
        # Check that the formatted message acknowledges empty questions
        assert "Total questions: 0" in result["formatted_message"]
    
    finally:
        # Restore original functions
        agentoptim.runner.get_evalset = original_get_evalset
        agentoptim.runner.format_error = original_format_error


@pytest.mark.asyncio
async def test_run_evalset_with_partial_failures():
    """Test run_evalset when some questions have evaluation errors."""
    import agentoptim.runner
    from agentoptim.runner import get_evalset
    
    # Create a mock EvalSet
    test_evalset = SimpleNamespace(
        id="partial-failures-evalset",
        name="Partial Failures EvalSet",
        template="Given this conversation: {{ conversation }} Answer: {{ eval_question }}",
        questions=[
            "Question that will succeed 1", 
            "Question that will fail", 
            "Question that will succeed 2"
        ]
    )
    
    # Mock the get_evalset function
    original_get_evalset = agentoptim.runner.get_evalset
    
    def mock_get_evalset(evalset_id):
        return test_evalset
    
    # Create a mock evaluate_question function that fails for specific questions
    original_evaluate_question = agentoptim.runner.evaluate_question
    
    async def mock_evaluate_question(conversation, question, template, judge_model, omit_reasoning=False):
        if "fail" in question:
            return EvalResult(
                question=question,
                error="Simulated API error for testing"
            )
        else:
            return EvalResult(
                question=question,
                judgment=True,
                confidence=0.9,
                reasoning="Reasoning for " + question
            )
    
    # Apply the patches
    agentoptim.runner.get_evalset = mock_get_evalset
    agentoptim.runner.evaluate_question = mock_evaluate_question
    
    try:
        # Define a test conversation
        conversation = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"}
        ]
        
        # Run the evalset with partial failures
        result = await run_evalset(
            evalset_id="partial-failures-evalset",
            conversation=conversation,
            judge_model="test-model"
        )
        
        # Check that the result has the expected structure
        assert result["status"] == "success"
        assert result["evalset_id"] == "partial-failures-evalset"
        assert result["evalset_name"] == "Partial Failures EvalSet"
        
        # Check that we have the right number of results
        assert len(result["results"]) == 3
        
        # Check that the successful and failed results are marked correctly
        success1 = next(r for r in result["results"] if r["question"] == "Question that will succeed 1")
        failed = next(r for r in result["results"] if r["question"] == "Question that will fail")
        success2 = next(r for r in result["results"] if r["question"] == "Question that will succeed 2")
        
        assert success1["judgment"] is True
        assert success1["error"] is None
        
        assert failed["judgment"] is None
        assert "Simulated API error" in failed["error"]
        
        assert success2["judgment"] is True
        assert success2["error"] is None
        
        # Check that the summary has correct counts
        assert result["summary"]["total_questions"] == 3
        assert result["summary"]["successful_evaluations"] == 2
        assert result["summary"]["yes_count"] == 2
        assert result["summary"]["no_count"] == 0
        assert result["summary"]["error_count"] == 1
        
        # Check that the formatted message mentions the error
        assert "Errors: 1" in result["formatted_message"]
    
    finally:
        # Restore original functions
        agentoptim.runner.get_evalset = original_get_evalset
        agentoptim.runner.evaluate_question = original_evaluate_question


@pytest.mark.asyncio
async def test_run_evalset_with_invalid_json_response():
    """Test run_evalset when the LLM returns invalid JSON."""
    import agentoptim.runner
    from agentoptim.runner import get_evalset, call_llm_api
    
    # Create a mock EvalSet
    test_evalset = SimpleNamespace(
        id="invalid-json-evalset",
        name="Invalid JSON EvalSet",
        template="Given this conversation: {{ conversation }} Answer: {{ eval_question }}",
        questions=["Is the response helpful?"]
    )
    
    # Mock the get_evalset function
    original_get_evalset = agentoptim.runner.get_evalset
    
    def mock_get_evalset(evalset_id):
        return test_evalset
    
    # Create a mock call_llm_api function that returns invalid JSON
    original_call_llm_api = agentoptim.runner.call_llm_api
    
    async def mock_call_llm_api(prompt=None, model=None, temperature=None, max_tokens=None, 
                               logit_bias=None, messages=None, omit_reasoning=False):
        # Return a response with invalid JSON
        return {
            "choices": [
                {
                    "message": {
                        "content": """The assistant's response is helpful.
                        
                        {
                          "reasoning": "The assistant provides a clear answer to the user's question with specific steps to take.",
                          "judgment": true,
                          "confidence": 0.85
                        """  # Missing closing brace
                    }
                }
            ]
        }
    
    # Apply the patches
    agentoptim.runner.get_evalset = mock_get_evalset
    agentoptim.runner.call_llm_api = mock_call_llm_api
    
    try:
        # Define a test conversation
        conversation = [
            {"role": "user", "content": "How do I reset my password?"},
            {"role": "assistant", "content": "Go to login page and click 'Forgot Password'."}
        ]
        
        # Run the evalset with invalid JSON response
        result = await run_evalset(
            evalset_id="invalid-json-evalset",
            conversation=conversation,
            judge_model="test-model"
        )
        
        # Check that the overall result still has success status
        assert result["status"] == "success"
        assert result["evalset_id"] == "invalid-json-evalset"
        
        # Check that we have the right number of results
        assert len(result["results"]) == 1
        
        # The regex fallback is finding a judgment in the invalid JSON
        eval_result = result["results"][0]
        assert eval_result["question"] == "Is the response helpful?"
        
        # Our runner.py actually has a smart fallback that uses regex to extract judgments
        # from invalid JSON. In this case it finds "judgment": true in the invalid JSON
        # and correctly identifies it as a True judgment.
        assert eval_result["judgment"] is True
        
        # The runner may set the reasoning to None when using regex fallback with omit_reasoning=False
        assert eval_result["reasoning"] is None
        
        # Check that the summary has the right question count
        assert result["summary"]["total_questions"] == 1
        
        # The regex fallback actually works in this case, so there's no error_count
        assert result["summary"]["error_count"] == 0
        
        # We should have a successful evaluation since the regex fallback worked
        assert result["summary"]["successful_evaluations"] == 1
        
        # The regex fallback works so well there's no error in the formatted message!
        # Let's just verify the formatted_message exists and has the expected structure
        assert "Evaluation Results for" in result["formatted_message"]
        assert "Yes responses: 1" in result["formatted_message"]
    
    finally:
        # Restore original functions
        agentoptim.runner.get_evalset = original_get_evalset
        agentoptim.runner.call_llm_api = original_call_llm_api


@pytest.mark.asyncio
async def test_run_evalset_with_rate_limit_errors():
    """Test run_evalset with rate limit errors from the LLM API."""
    import agentoptim.runner
    from agentoptim.runner import get_evalset, call_llm_api
    
    # Create a mock EvalSet with multiple questions
    test_evalset = SimpleNamespace(
        id="rate-limit-evalset",
        name="Rate Limit EvalSet",
        template="Given this conversation: {{ conversation }} Answer: {{ eval_question }}",
        questions=[f"Question {i}" for i in range(5)]  # 5 questions
    )
    
    # Mock the get_evalset function
    original_get_evalset = agentoptim.runner.get_evalset
    
    def mock_get_evalset(evalset_id):
        return test_evalset
    
    # Create a mock call_llm_api function that simulates rate limit errors
    # for specific questions
    original_call_llm_api = agentoptim.runner.call_llm_api
    
    # Counter to track which question is being processed
    call_counter = {"count": 0}
    
    async def mock_call_llm_api(prompt=None, model=None, temperature=None, max_tokens=None, 
                               logit_bias=None, messages=None, omit_reasoning=False):
        # Increment the counter for each call
        call_counter["count"] += 1
        
        # Simulate rate limit error for every other call
        if call_counter["count"] % 2 == 0:
            return {
                "error": "Rate limit exceeded. Please try again later.",
                "details": "You are sending requests too quickly. Please slow down."
            }
        else:
            # Return a successful response for odd-numbered calls
            return {
                "choices": [
                    {
                        "message": {
                            "content": '{"reasoning": "The response is clear and helpful.", "judgment": true, "confidence": 0.85}'
                        }
                    }
                ]
            }
    
    # Apply the patches
    agentoptim.runner.get_evalset = mock_get_evalset
    agentoptim.runner.call_llm_api = mock_call_llm_api
    
    try:
        # Define a test conversation
        conversation = [
            {"role": "user", "content": "How do I reset my password?"},
            {"role": "assistant", "content": "Go to login page and click 'Forgot Password'."}
        ]
        
        # Run the evalset with rate limit errors
        result = await run_evalset(
            evalset_id="rate-limit-evalset",
            conversation=conversation,
            judge_model="test-model",
            max_parallel=2  # Allow parallel processing
        )
        
        # Check that the overall result still has success status
        assert result["status"] == "success"
        assert result["evalset_id"] == "rate-limit-evalset"
        
        # We should have all 5 results
        assert len(result["results"]) == 5
        
        # Check that we have a mix of successful and failed evaluations
        successful_evals = [r for r in result["results"] if r["error"] is None]
        failed_evals = [r for r in result["results"] if r["error"] is not None]
        
        # We should have some successful and some failed evaluations
        assert len(successful_evals) > 0
        assert len(failed_evals) > 0
        
        # Check that the failures are due to rate limiting
        for failed in failed_evals:
            assert "Rate limit" in failed["error"]
        
        # Check that the summary counts are correct
        assert result["summary"]["total_questions"] == 5
        assert result["summary"]["successful_evaluations"] == len(successful_evals)
        assert result["summary"]["error_count"] == len(failed_evals)
        
        # Verify the formatted message mentions errors
        assert f"Errors: {len(failed_evals)}" in result["formatted_message"]
        
    finally:
        # Restore original functions
        agentoptim.runner.get_evalset = original_get_evalset
        agentoptim.runner.call_llm_api = original_call_llm_api


if __name__ == "__main__":
    unittest.main()