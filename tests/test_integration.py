"""
Integration tests for AgentOptim v2.0.

These tests verify that the different components of the system work together
correctly through full workflows.
"""

import pytest
import os
import shutil
import asyncio
from unittest.mock import patch, MagicMock

from agentoptim import manage_evalset, run_evalset
from agentoptim.evalset import EvalSet, EVALSETS_DIR
from agentoptim.utils import DATA_DIR


# Create a test directory to avoid interfering with real data
TEST_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "test_data")


@pytest.fixture
def temp_data_dir():
    """Create a temporary data directory for tests."""
    # Set up the test data directory
    os.makedirs(TEST_DATA_DIR, exist_ok=True)
    
    # Also set up subdirectories for EvalSets
    os.makedirs(os.path.join(TEST_DATA_DIR, "evalsets"), exist_ok=True)
    
    # Patch the DATA_DIR constant and the derived EVALSETS_DIR
    with patch('agentoptim.utils.DATA_DIR', TEST_DATA_DIR):
        with patch('agentoptim.evalset.EVALSETS_DIR', os.path.join(TEST_DATA_DIR, "evalsets")):
            yield TEST_DATA_DIR
    
    # Clean up after the test
    if os.path.exists(TEST_DATA_DIR):
        shutil.rmtree(TEST_DATA_DIR)


class TestFullWorkflow:
    """Test the full workflow for version 2.0."""
    
    @pytest.mark.asyncio
    async def test_basic_workflow(self, temp_data_dir):
        """Test a basic workflow from EvalSet creation to running an evaluation."""
        
        # 1. Create an EvalSet with yes/no questions
        evalset_result = manage_evalset(
            action="create",
            name="Test EvalSet",
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
                "Is the response clear and concise?", 
                "Does the response directly address the question?",
                "Is the response accurate?"
            ],
            description="Test EvalSet for integration testing"
        )
        
        # Verify EvalSet was created
        assert evalset_result is not None
        assert evalset_result.get("status") == "success"
        assert "evalset" in evalset_result
        
        evalset_id = evalset_result.get("evalset", {}).get("id")
        assert evalset_id is not None
        
        # 2. Retrieve the EvalSet
        get_result = manage_evalset(
            action="get",
            evalset_id=evalset_id
        )
        
        # Verify EvalSet can be retrieved
        assert get_result is not None
        assert get_result.get("status") == "success"
        assert get_result.get("evalset", {}).get("id") == evalset_id
        assert len(get_result.get("evalset", {}).get("questions", [])) == 4
        
        # 3. Define a conversation to evaluate
        conversation = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "How do I reset my password?"},
            {"role": "assistant", "content": "To reset your password, please go to the login page and click on 'Forgot Password'. You'll receive an email with instructions to create a new password."}
        ]
        
        # 4. Run an evaluation with a mock
        # Mock the LLM API call to return a successful response with yes judgment
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
        
        # Apply patch for the LLM API call
        with patch('agentoptim.runner.call_llm_api', return_value=mock_api_response):
            # Run the evaluation
            result = await run_evalset(
                evalset_id=evalset_id,
                conversation=conversation,
                model="mock-model",
                max_parallel=2
            )
            
            # Verify evaluation results
            assert result is not None
            assert result.get("status") == "success"
            assert result.get("evalset_id") == evalset_id
            assert result.get("model") == "mock-model"
            
            # Check results array
            assert "results" in result
            results = result.get("results", [])
            assert len(results) == 4  # Should have one result per question
            
            # Check summary
            assert "summary" in result
            summary = result.get("summary", {})
            assert summary.get("total_questions") == 4
            assert summary.get("yes_count") == 4  # All should be "yes" from our mock
            
            # Check formatted message is present
            assert "formatted_message" in result
        
        # 5. Test the list function
        list_result = manage_evalset(action="list")
        
        # Verify list results
        assert list_result is not None
        assert not list_result.get("error", False)
        assert "items" in list_result
        assert len(list_result.get("items", [])) == 1
        assert list_result.get("items", [])[0].get("id") == evalset_id
        
        # 6. Update the EvalSet
        update_result = manage_evalset(
            action="update",
            evalset_id=evalset_id,
            name="Updated Test EvalSet"
        )
        
        # Verify update
        assert update_result is not None
        assert update_result.get("status") == "success"
        assert "Updated" in update_result.get("message", "")
        
        # 7. Delete the EvalSet
        delete_result = manage_evalset(
            action="delete",
            evalset_id=evalset_id
        )
        
        # Verify deletion
        assert delete_result is not None
        assert delete_result.get("status") == "success"
        assert "deleted" in delete_result.get("message", "")
        
        # Check that it's really gone
        list_after_delete = manage_evalset(action="list")
        assert len(list_after_delete.get("items", [])) == 0


class TestParallelProcessing:
    """Test parallel processing for EvalSet evaluations."""
    
    @pytest.mark.asyncio
    async def test_parallel_evaluation(self, temp_data_dir):
        """Test that evaluations can be executed in parallel."""
        
        # Create an EvalSet with multiple questions
        evalset_result = manage_evalset(
            action="create",
            name="Parallel Test EvalSet",
            template="""
            Given this conversation:
            {{ conversation }}
            
            Please answer the following yes/no question about the final assistant response:
            {{ eval_question }}
            
            Return a JSON object with the following format:
            {"judgment": 1} for yes or {"judgment": 0} for no.
            """,
            questions=[f"Test question {i}?" for i in range(10)],  # 10 questions
            description="EvalSet for testing parallel processing"
        )
        
        evalset_id = evalset_result.get("evalset", {}).get("id")
        
        # Define a conversation to evaluate
        conversation = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Can you explain quantum computing?"},
            {"role": "assistant", "content": "Quantum computing uses quantum mechanics to perform computations..."}
        ]
        
        # Mock the LLM API call to return a successful response
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
        
        # Set up a counter to verify parallel execution
        call_count = 0
        
        def mock_call_llm_api(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return mock_api_response
        
        # Apply patch for the LLM API call
        with patch('agentoptim.runner.call_llm_api', side_effect=mock_call_llm_api):
            # Run the evaluation with parallel processing
            result = await run_evalset(
                evalset_id=evalset_id,
                conversation=conversation,
                model="mock-model",
                max_parallel=4  # Run up to 4 evaluations in parallel
            )
            
            # Verify evaluation results
            assert result is not None
            assert result.get("status") == "success"
            
            # Check results array - should have one result per question
            results = result.get("results", [])
            assert len(results) == 10
            
            # Verify all calls were made
            assert call_count == 10
            
            # Summary should show all successful evaluations
            summary = result.get("summary", {})
            assert summary.get("total_questions") == 10
            assert summary.get("yes_count") == 10  # All are "yes" from our mock