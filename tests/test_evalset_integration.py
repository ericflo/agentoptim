"""
Integration tests for the new EvalSet architecture.

These tests verify that the EvalSet architecture works end-to-end with real-world examples.
"""

import pytest

# Mark all tests in this file as both integration tests and evalset tests
pytestmark = [
    pytest.mark.integration,
    pytest.mark.evalset
]

import pytest
import os
import shutil
import asyncio
import uuid
from datetime import datetime
from unittest.mock import patch, MagicMock

# Import the new EvalSet architecture components
from agentoptim.evalset import manage_evalset
from agentoptim.runner import run_evalset
from agentoptim.compat import (
    convert_evaluation_to_evalset,
    evaluation_to_evalset_id,
    dataset_to_conversations
)
from agentoptim.utils import DATA_DIR


# Create a test directory to avoid interfering with real data
TEST_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "test_evalset_data")


@pytest.fixture
def temp_data_dir():
    """Create a temporary data directory for tests."""
    # Set up the test data directory
    os.makedirs(TEST_DATA_DIR, exist_ok=True)
    
    # Also set up subdirectories
    os.makedirs(os.path.join(TEST_DATA_DIR, "evalsets"), exist_ok=True)
    os.makedirs(os.path.join(TEST_DATA_DIR, "evaluations"), exist_ok=True)
    
    # Patch the DATA_DIR constant for all relevant modules
    with patch('agentoptim.utils.DATA_DIR', TEST_DATA_DIR):
        with patch('agentoptim.evalset.DATA_DIR', TEST_DATA_DIR):
            with patch('agentoptim.runner.DATA_DIR', TEST_DATA_DIR):
                with patch('agentoptim.compat.DATA_DIR', TEST_DATA_DIR):
                    yield TEST_DATA_DIR
    
    # Clean up after the test
    if os.path.exists(TEST_DATA_DIR):
        shutil.rmtree(TEST_DATA_DIR)


class TestEvalSetBasics:
    """Test basic functionality of the EvalSet architecture."""
    
    @pytest.mark.asyncio
    async def test_create_and_retrieve_evalset(self, temp_data_dir):
        """Test creating and retrieving an EvalSet."""
        
        # Generate a unique test name
        test_uuid = str(uuid.uuid4())[:8]
        test_name = f"EvalSet Test {test_uuid}"
        
        # Create a new EvalSet
        evalset_result = manage_evalset(
            action="create",
            name=f"{test_name}",
            template="""
            Given this conversation:
            {{ conversation }}
            
            Please answer the following yes/no question about the final assistant response:
            {{ eval_question }}
            
            Return a JSON object with the following format:
            {"judgment": 1} for yes or {"judgment": 0} for no.
            """,
            questions=[
                "Is the response helpful?",
                "Does the response directly address the user's question?",
                "Is the response clear and easy to understand?",
                "Does the response provide complete information?"
            ],
            description=f"Test EvalSet for {test_name}"
        )
        
        # Verify EvalSet was created
        assert evalset_result is not None
        assert "evalset" in evalset_result
        assert "id" in evalset_result["evalset"]
        
        evalset_id = evalset_result["evalset"]["id"]
        
        # Retrieve the EvalSet
        get_result = manage_evalset(
            action="get",
            evalset_id=evalset_id
        )
        
        # Verify EvalSet can be retrieved
        assert get_result is not None
        assert "evalset" in get_result
        assert get_result["evalset"]["id"] == evalset_id
        assert len(get_result["evalset"]["questions"]) == 4
        
        # List all EvalSets
        list_result = manage_evalset(
            action="list"
        )
        
        # Verify the list includes our EvalSet
        assert list_result is not None
        assert "evalsets" in list_result
        assert any(evalset["id"] == evalset_id for evalset in list_result["evalsets"])
        
        # Update the EvalSet
        update_result = manage_evalset(
            action="update",
            evalset_id=evalset_id,
            questions=[
                "Is the response helpful?",
                "Does the response directly address the user's question?",
                "Is the response clear and easy to understand?",
                "Does the response provide complete information?",
                "Is the response accurate?"
            ]
        )
        
        # Verify the update was successful
        assert update_result is not None
        assert "evalset" in update_result
        assert update_result["evalset"]["id"] == evalset_id
        assert len(update_result["evalset"]["questions"]) == 5
        
        # Delete the EvalSet
        delete_result = manage_evalset(
            action="delete",
            evalset_id=evalset_id
        )
        
        # Verify the deletion was successful
        assert delete_result is not None
        assert "status" in delete_result
        assert delete_result["status"] == "success"
        
        # Verify the EvalSet is no longer retrievable
        try:
            manage_evalset(
                action="get",
                evalset_id=evalset_id
            )
            # Should not reach here
            assert False, "EvalSet should have been deleted"
        except Exception as e:
            # Expected to raise an exception
            assert "not found" in str(e).lower()


class TestEvalSetEvaluation:
    """Test evaluating conversations with the EvalSet architecture."""
    
    @pytest.mark.asyncio
    async def test_evaluate_conversation(self, temp_data_dir):
        """Test evaluating a conversation with an EvalSet."""
        
        # Generate a unique test name
        test_uuid = str(uuid.uuid4())[:8]
        test_name = f"Evaluation Test {test_uuid}"
        
        # Create a new EvalSet
        evalset_result = manage_evalset(
            action="create",
            name=f"{test_name}",
            template="""
            Given this conversation:
            {{ conversation }}
            
            Please answer the following yes/no question about the final assistant response:
            {{ eval_question }}
            
            Return a JSON object with the following format:
            {"judgment": 1} for yes or {"judgment": 0} for no.
            """,
            questions=[
                "Is the response helpful?",
                "Does the response directly address the user's question?"
            ],
            description=f"Test EvalSet for {test_name}"
        )
        
        evalset_id = evalset_result["evalset"]["id"]
        
        # Define a test conversation
        conversation = [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": "How do I reset my password?"},
            {"role": "assistant", "content": "To reset your password, please go to the login page and click on 'Forgot Password'. You'll receive an email with instructions to create a new password."}
        ]
        
        # Mock the LLM call since we're testing the integration, not the LLM response
        mock_response = {
            "result": {
                "choices": [
                    {
                        "message": {"content": '{"judgment": 1}'},
                        "logprobs": {"content": [{"token": '{"judgment": 1}', "logprob": -0.05}]}
                    }
                ]
            }
        }
        
        # Patch the runner._call_judge function to return mock responses
        with patch('agentoptim.runner._call_judge', return_value=mock_response):
            # Run the evaluation
            eval_result = await run_evalset(
                evalset_id=evalset_id,
                conversation=conversation,
                model="test-model",
                max_parallel=1
            )
            
            # Verify the evaluation result structure
            assert eval_result is not None
            assert "status" in eval_result
            assert eval_result["status"] == "success"
            assert "evalset_id" in eval_result
            assert eval_result["evalset_id"] == evalset_id
            assert "summary" in eval_result
            assert "results" in eval_result
            
            # Verify summary statistics
            summary = eval_result["summary"]
            assert "total_questions" in summary
            assert summary["total_questions"] == 2
            assert "successful_evaluations" in summary
            assert summary["successful_evaluations"] == 2
            assert "yes_count" in summary
            assert summary["yes_count"] == 2
            assert "yes_percentage" in summary
            assert summary["yes_percentage"] == 100.0
            
            # Verify results
            results = eval_result["results"]
            assert len(results) == 2
            for result in results:
                assert "question" in result
                assert "judgment" in result
                assert result["judgment"] is True  # All mocked to be yes
                assert "logprob" in result


class TestCompatibilityLayer:
    """Test the compatibility layer for transitioning from old to new architecture."""
    
    @pytest.mark.asyncio
    async def test_convert_evaluation(self, temp_data_dir):
        """Test converting an old evaluation to a new EvalSet."""
        # Since we can't easily mock the old evaluation system,
        # we'll create a simulated old evaluation format directly
        
        # Create a mock old evaluation
        old_evaluation = {
            "id": "eval-" + str(uuid.uuid4()),
            "name": "Old Evaluation",
            "template": "Input: {input}\nResponse: {response}\nQuestion: {question}\n\nAnswer yes (1) or no (0) in JSON format: {\"judgment\": 1 or 0}",
            "questions": [
                "Is the response helpful?",
                "Does the response directly address the question?"
            ],
            "description": "Test old evaluation format"
        }
        
        # Mock the import of old evaluation
        with patch('agentoptim.compat.get_evaluation', return_value=old_evaluation):
            # Convert to new EvalSet
            conversion_result = await convert_evaluation_to_evalset(
                name="Converted Evaluation",
                template=old_evaluation["template"],
                questions=old_evaluation["questions"],
                description="Converted from old evaluation format"
            )
            
            # Verify conversion result
            assert conversion_result is not None
            assert "evalset" in conversion_result
            assert "id" in conversion_result["evalset"]
            assert conversion_result["evalset"]["name"] == "Converted Evaluation"
            
            # The conversion should have transformed placeholders
            template = conversion_result["evalset"]["template"]
            assert "{{ conversation }}" in template or "{{conversation}}" in template
            assert "{{ eval_question }}" in template or "{{eval_question}}" in template
            
            # And kept the same questions
            assert len(conversion_result["evalset"]["questions"]) == 2
            assert "Is the response helpful?" in conversion_result["evalset"]["questions"]
            
            # Test the mapping function
            with patch('agentoptim.compat.get_evaluation', return_value=old_evaluation):
                with patch('agentoptim.compat._get_evalset_id_for_evaluation', return_value=conversion_result["evalset"]["id"]):
                    mapping_result = await evaluation_to_evalset_id(old_evaluation["id"])
                    assert mapping_result is not None
                    assert mapping_result == conversion_result["evalset"]["id"]


class TestRealWorldScenarios:
    """Test real-world scenarios with the EvalSet architecture."""
    
    @pytest.mark.asyncio
    async def test_comparing_responses(self, temp_data_dir):
        """Test comparing different responses using the EvalSet architecture."""
        
        # Generate a unique test name
        test_uuid = str(uuid.uuid4())[:8]
        test_name = f"Comparison Test {test_uuid}"
        
        # Create an EvalSet for response quality
        evalset_result = manage_evalset(
            action="create",
            name=f"{test_name}",
            template="""
            Given this conversation:
            {{ conversation }}
            
            Please answer the following yes/no question about the final assistant response:
            {{ eval_question }}
            
            Return a JSON object with the following format:
            {"judgment": 1} for yes or {"judgment": 0} for no.
            """,
            questions=[
                "Is the response helpful?",
                "Does the response directly address the user's question?",
                "Is the response clear and easy to understand?",
                "Does the response provide complete information?"
            ],
            description=f"Evaluation criteria for response quality"
        )
        
        evalset_id = evalset_result["evalset"]["id"]
        
        # Define two conversations with different quality responses
        good_response = [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": "How do I reset my password?"},
            {"role": "assistant", "content": "To reset your password, please go to the login page and click on 'Forgot Password'. You'll receive an email with instructions to create a new password. If you don't receive the email within a few minutes, check your spam folder. If you still need help, contact our support team."}
        ]
        
        poor_response = [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": "How do I reset my password?"},
            {"role": "assistant", "content": "Contact support."}
        ]
        
        # Create two different mocked responses
        def mock_judge_call(prompt, model, **kwargs):
            # Different responses based on input prompt
            if "Contact support." in prompt:
                # Poor response gets low scores (mostly no)
                return {
                    "result": {
                        "choices": [
                            {
                                "message": {"content": '{"judgment": 0}'},
                                "logprobs": {"content": [{"token": '{"judgment": 0}', "logprob": -0.05}]}
                            }
                        ]
                    }
                }
            else:
                # Good response gets high scores (mostly yes)
                return {
                    "result": {
                        "choices": [
                            {
                                "message": {"content": '{"judgment": 1}'},
                                "logprobs": {"content": [{"token": '{"judgment": 1}', "logprob": -0.05}]}
                            }
                        ]
                    }
                }
        
        # Patch the judge call
        with patch('agentoptim.runner._call_judge', side_effect=mock_judge_call):
            # Evaluate good response
            good_result = await run_evalset(
                evalset_id=evalset_id,
                conversation=good_response,
                model="test-model",
                max_parallel=1
            )
            
            # Evaluate poor response
            poor_result = await run_evalset(
                evalset_id=evalset_id,
                conversation=poor_response,
                model="test-model",
                max_parallel=1
            )
            
            # Verify the results show a difference
            assert good_result["summary"]["yes_percentage"] > poor_result["summary"]["yes_percentage"]
            
            # Good response should have higher yes count
            assert good_result["summary"]["yes_count"] > poor_result["summary"]["yes_count"]