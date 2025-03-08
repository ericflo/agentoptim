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
    
    # Patch the DATA_DIR constant for utils and EVALSETS_DIR for evalset
    with patch('agentoptim.utils.DATA_DIR', TEST_DATA_DIR):
        with patch('agentoptim.evalset.EVALSETS_DIR', os.path.join(TEST_DATA_DIR, "evalsets")):
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
        assert "items" in list_result
        assert any(evalset["id"] == evalset_id for evalset in list_result["items"])
        
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
        
        # Verify the EvalSet is no longer in the list
        list_after_delete = manage_evalset(action="list")
        assert not any(evalset["id"] == evalset_id for evalset in list_after_delete.get("items", []))


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
        
        # Mock the LLM API call
        mock_api_response = {
            "choices": [
                {
                    "message": {
                        "content": '{"judgment": 1}'
                    },
                    "logprobs": {
                        "content": [
                            {"token": "{", "logprob": -0.1},
                            {"token": "\"", "logprob": -0.2},
                            {"token": "judgment", "logprob": -0.3},
                            {"token": "\"", "logprob": -0.2},
                            {"token": ":", "logprob": -0.1},
                            {"token": " ", "logprob": -0.1},
                            {"token": "1", "logprob": -0.5},
                            {"token": "}", "logprob": -0.1}
                        ]
                    }
                }
            ]
        }
        
        # Patch the LLM API call
        with patch('agentoptim.runner.call_llm_api', return_value=mock_api_response):
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
    
    @pytest.mark.skip(reason="Compatibility layer is not needed in the v2.0 architecture")
    @pytest.mark.asyncio
    async def test_convert_evaluation(self, temp_data_dir):
        """Test converting an old evaluation to a new EvalSet.
        
        Note: This test is skipped because we've fully migrated to the new architecture
        and no longer need the compatibility layer.
        """
        # Skip this test - legacy compatibility test
        pass


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
        
        # Create different mock responses for good and poor responses
        # Mock for good response gets all yes judgments
        good_response_mock = {
            "choices": [
                {
                    "message": {
                        "content": '{"judgment": 1}'
                    },
                    "logprobs": {
                        "content": [
                            {"token": "{", "logprob": -0.1},
                            {"token": "\"", "logprob": -0.2},
                            {"token": "judgment", "logprob": -0.3},
                            {"token": "\"", "logprob": -0.2},
                            {"token": ":", "logprob": -0.1},
                            {"token": " ", "logprob": -0.1},
                            {"token": "1", "logprob": -0.5},
                            {"token": "}", "logprob": -0.1}
                        ]
                    }
                }
            ]
        }
        
        # Mock for poor response gets all no judgments
        poor_response_mock = {
            "choices": [
                {
                    "message": {
                        "content": '{"judgment": 0}'
                    },
                    "logprobs": {
                        "content": [
                            {"token": "{", "logprob": -0.1},
                            {"token": "\"", "logprob": -0.2},
                            {"token": "judgment", "logprob": -0.3},
                            {"token": "\"", "logprob": -0.2},
                            {"token": ":", "logprob": -0.1},
                            {"token": " ", "logprob": -0.1},
                            {"token": "0", "logprob": -0.5},
                            {"token": "}", "logprob": -0.1}
                        ]
                    }
                }
            ]
        }
        
        # Use side_effect to return different responses based on input
        def mock_api_call(*args, **kwargs):
            # Check if the prompt contains the poor response text
            prompt = args[0] if args else kwargs.get('prompt', '')
            if "Contact support" in prompt:
                return poor_response_mock
            else:
                return good_response_mock
        
        # Patch the LLM API call with the side_effect function
        with patch('agentoptim.runner.call_llm_api', side_effect=mock_api_call):
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