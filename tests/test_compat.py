"""Tests for the compatibility layer.

Note: The compat module is intentionally deprecated and these tests will be removed in v2.1.0.
They are kept to ensure backwards compatibility during the transition period.
"""

import os
import asyncio
import unittest
from unittest import mock
import json
import pytest

# Mark all tests as intentionally using deprecated features
pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

# Suppress specific deprecation warnings for these imports
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    from agentoptim.compat import (
        convert_evaluation_to_evalset,
        evaluation_to_evalset_id,
        dataset_to_conversations,
        experiment_results_to_evalset_results
    )

from agentoptim.evalset import EvalSet, list_evalsets


class TestCompat(unittest.TestCase):
    """Test suite for compatibility layer functionality."""
    
    def setUp(self):
        """Set up the test environment."""
        # Mock data for tests
        self.test_name = "Test Evaluation"
        self.test_template = """
        Input: {input}
        Response: {response}
        Question: {question}
        
        Answer yes (1) or no (0) in JSON format: {"judgment": 1 or 0}
        """
        self.test_questions = ["Question 1", "Question 2"]
        self.test_description = "Test description"
        
        # Sample dataset items
        self.test_dataset_items = [
            {"input": "Input 1", "expected_output": "Output 1"},
            {"input": "Input 2", "expected_output": "Output 2"}
        ]
        
        # Sample conversation
        self.test_conversation = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "How do I reset my password?"},
            {"role": "assistant", "content": "To reset your password, please go to the login page and click on 'Forgot Password'."}
        ]
    
    @mock.patch('agentoptim.compat.manage_evalset')
    async def test_convert_evaluation_to_evalset(self, mock_manage_evalset):
        """Test converting an evaluation to an EvalSet."""
        # Mock the manage_evalset function to return success
        mock_manage_evalset.return_value = {
            "status": "success",
            "evalset": {
                "id": "new-evalset-id",
                "name": self.test_name,
                "questions": self.test_questions
            }
        }
        
        # Convert the evaluation
        result = await convert_evaluation_to_evalset(
            name=self.test_name,
            template=self.test_template,
            questions=self.test_questions,
            description=self.test_description
        )
        
        # Check the result
        self.assertEqual(result.get("status"), "success")
        self.assertEqual(result.get("evalset", {}).get("name"), self.test_name)
        
        # Check that manage_evalset was called with the converted template
        args, kwargs = mock_manage_evalset.call_args
        self.assertEqual(kwargs.get("name"), self.test_name)
        self.assertEqual(kwargs.get("questions"), self.test_questions)
        self.assertEqual(kwargs.get("description"), self.test_description)
        
        # The template should be converted to use {{ conversation }} and {{ eval_question }}
        template = kwargs.get("template")
        self.assertIn("{{ conversation }}", template)
        self.assertIn("{{ eval_question }}", template)
    
    @mock.patch('agentoptim.compat.list_evalsets')
    async def test_evaluation_to_evalset_id_found(self, mock_list_evalsets):
        """Test finding a corresponding EvalSet for an evaluation ID."""
        # Create a mock EvalSet that references the evaluation ID
        mock_evalset = EvalSet(
            id="evalset-1",
            name="Converted Evaluation",
            template="Template with {{ conversation }} and {{ eval_question }}",
            questions=["Q1", "Q2"],
            description="Converted from evaluation eval-123"
        )
        
        # Mock list_evalsets to return our mock EvalSet
        mock_list_evalsets.return_value = [mock_evalset]
        
        # Try to find the corresponding EvalSet
        evalset_id = await evaluation_to_evalset_id("eval-123")
        
        # Check that we found the right EvalSet
        self.assertEqual(evalset_id, "evalset-1")
    
    @mock.patch('agentoptim.compat.list_evalsets')
    async def test_evaluation_to_evalset_id_not_found(self, mock_list_evalsets):
        """Test when no corresponding EvalSet is found."""
        # Mock list_evalsets to return an empty list
        mock_list_evalsets.return_value = []
        
        # Try to find a corresponding EvalSet for a non-existent evaluation
        evalset_id = await evaluation_to_evalset_id("non-existent-eval")
        
        # Check that no EvalSet was found
        self.assertIsNone(evalset_id)
    
    async def test_dataset_to_conversations(self):
        """Test converting dataset items to conversations."""
        # Convert the dataset items to conversations
        conversations = await dataset_to_conversations(
            dataset_id="dataset-1",
            items=self.test_dataset_items
        )
        
        # Check that we got the right number of conversations
        self.assertEqual(len(conversations), 2)
        
        # Check the structure of the first conversation
        self.assertEqual(len(conversations[0]), 3)  # system + user + assistant
        self.assertEqual(conversations[0][0]["role"], "system")
        self.assertEqual(conversations[0][1]["role"], "user")
        self.assertEqual(conversations[0][1]["content"], "Input 1")
        self.assertEqual(conversations[0][2]["role"], "assistant")
        self.assertEqual(conversations[0][2]["content"], "Output 1")
    
    async def test_experiment_results_to_evalset_results(self):
        """Test converting experiment results to EvalSet results."""
        # Convert experiment results to EvalSet results
        results = await experiment_results_to_evalset_results(
            experiment_id="exp-1",
            evaluation_id="eval-1",
            evalset_id="evalset-1"
        )
        
        # Check the basic structure
        self.assertEqual(results.get("status"), "success")
        self.assertEqual(results.get("evalset_id"), "evalset-1")
        self.assertIn("summary", results)
        self.assertIn("results", results)


# Create a run function for async tests
def run_async_test(coroutine):
    """Run an async coroutine as a test."""
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(coroutine)


# Override the TestCompat methods to enable async tests
def async_test(test_case):
    """Decorator for async test methods."""
    def wrapper(*args, **kwargs):
        return run_async_test(test_case(*args, **kwargs))
    return wrapper


# Apply the decorator to async test methods
TestCompat.test_convert_evaluation_to_evalset = async_test(TestCompat.test_convert_evaluation_to_evalset)
TestCompat.test_evaluation_to_evalset_id_found = async_test(TestCompat.test_evaluation_to_evalset_id_found)
TestCompat.test_evaluation_to_evalset_id_not_found = async_test(TestCompat.test_evaluation_to_evalset_id_not_found)
TestCompat.test_dataset_to_conversations = async_test(TestCompat.test_dataset_to_conversations)
TestCompat.test_experiment_results_to_evalset_results = async_test(TestCompat.test_experiment_results_to_evalset_results)


if __name__ == "__main__":
    unittest.main()