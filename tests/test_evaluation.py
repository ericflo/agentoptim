"""Tests for the evaluation module."""

import os
import shutil
import tempfile
import unittest
from unittest import mock

from agentoptim.evaluation import (
    Evaluation,
    create_evaluation,
    get_evaluation,
    update_evaluation,
    delete_evaluation,
    list_evaluations,
    manage_evaluation,
)


class TestEvaluation(unittest.TestCase):
    """Test suite for evaluation functionality."""
    
    def setUp(self):
        """Set up the test environment."""
        # Create a temporary directory for test data
        self.temp_dir = tempfile.mkdtemp()
        
        # Mock the EVALUATIONS_DIR to use our temporary directory
        self.patcher = mock.patch("agentoptim.evaluation.EVALUATIONS_DIR", self.temp_dir)
        self.patcher.start()
        
        # Sample data for tests
        self.test_name = "Test Evaluation"
        self.test_template = "{{ history }}\n{{ question }}"
        self.test_questions = ["Question 1", "Question 2"]
        self.test_description = "Test description"
    
    def tearDown(self):
        """Clean up after tests."""
        # Stop the patcher
        self.patcher.stop()
        
        # Remove the temporary directory
        shutil.rmtree(self.temp_dir)
    
    def test_create_evaluation(self):
        """Test creating an evaluation."""
        evaluation = create_evaluation(
            self.test_name, self.test_template, self.test_questions, self.test_description
        )
        
        self.assertEqual(evaluation.name, self.test_name)
        self.assertEqual(evaluation.template, self.test_template)
        self.assertEqual(evaluation.questions, self.test_questions)
        self.assertEqual(evaluation.description, self.test_description)
        
        # Check that the file was created
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, f"{evaluation.id}.json")))
    
    def test_get_evaluation(self):
        """Test retrieving an evaluation."""
        created = create_evaluation(
            self.test_name, self.test_template, self.test_questions, self.test_description
        )
        
        retrieved = get_evaluation(created.id)
        
        self.assertEqual(retrieved.id, created.id)
        self.assertEqual(retrieved.name, self.test_name)
        self.assertEqual(retrieved.template, self.test_template)
        self.assertEqual(retrieved.questions, self.test_questions)
        self.assertEqual(retrieved.description, self.test_description)
        
        # Test non-existent evaluation
        self.assertIsNone(get_evaluation("nonexistent-id"))
    
    def test_update_evaluation(self):
        """Test updating an evaluation."""
        evaluation = create_evaluation(
            self.test_name, self.test_template, self.test_questions, self.test_description
        )
        
        # Update some fields
        new_name = "Updated Name"
        new_questions = ["New Question 1", "New Question 2", "New Question 3"]
        
        updated = update_evaluation(
            evaluation.id, name=new_name, questions=new_questions
        )
        
        self.assertEqual(updated.id, evaluation.id)
        self.assertEqual(updated.name, new_name)
        self.assertEqual(updated.template, self.test_template)  # Unchanged
        self.assertEqual(updated.questions, new_questions)
        self.assertEqual(updated.description, self.test_description)  # Unchanged
        
        # Test non-existent evaluation
        self.assertIsNone(update_evaluation("nonexistent-id", name="New Name"))
    
    def test_delete_evaluation(self):
        """Test deleting an evaluation."""
        evaluation = create_evaluation(
            self.test_name, self.test_template, self.test_questions, self.test_description
        )
        
        # Check that the file exists
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, f"{evaluation.id}.json")))
        
        # Delete and check that it returns True
        self.assertTrue(delete_evaluation(evaluation.id))
        
        # Check that the file is gone
        self.assertFalse(os.path.exists(os.path.join(self.temp_dir, f"{evaluation.id}.json")))
        
        # Test deleting non-existent evaluation
        self.assertFalse(delete_evaluation("nonexistent-id"))
    
    def test_list_evaluations(self):
        """Test listing evaluations."""
        # Create a few evaluations
        eval1 = create_evaluation("Eval 1", "Template 1", ["Q1", "Q2"])
        eval2 = create_evaluation("Eval 2", "Template 2", ["Q3", "Q4"])
        
        evaluations = list_evaluations()
        
        self.assertEqual(len(evaluations), 2)
        self.assertIn(eval1.id, [e.id for e in evaluations])
        self.assertIn(eval2.id, [e.id for e in evaluations])
    
    def test_manage_evaluation_create(self):
        """Test the manage_evaluation tool function for creating evaluations."""
        result = manage_evaluation(
            action="create",
            name=self.test_name,
            template=self.test_template,
            questions=self.test_questions,
            description=self.test_description,
        )
        
        self.assertFalse(result.get("error", True))
        self.assertIn(self.test_name, result.get("message", ""))
        
        # Check that one evaluation now exists
        evaluations = list_evaluations()
        self.assertEqual(len(evaluations), 1)
    
    def test_manage_evaluation_list(self):
        """Test the manage_evaluation tool function for listing evaluations."""
        # Create a few evaluations
        create_evaluation("Eval 1", "Template 1", ["Q1", "Q2"])
        create_evaluation("Eval 2", "Template 2", ["Q3", "Q4"])
        
        result = manage_evaluation(action="list")
        
        self.assertFalse(result.get("error", True))
        
        # Check if items are in the list
        items = result.get("items", [])
        names = [item.get("name") for item in items]
        self.assertIn("Eval 1", names)
        self.assertIn("Eval 2", names)
    
    def test_manage_evaluation_get(self):
        """Test the manage_evaluation tool function for getting an evaluation."""
        evaluation = create_evaluation(
            self.test_name, self.test_template, self.test_questions, self.test_description
        )
        
        # The get action now returns a dict but includes the full evaluation string in the response
        try:
            result = manage_evaluation(action="get", evaluation_id=evaluation.id)
            if isinstance(result, str):
                # If it's a string, just check contents directly
                self.assertIn(self.test_name, result)
                self.assertIn(self.test_description, result)
                self.assertIn(self.test_template, result)
                self.assertIn("Question 1", result)
                self.assertIn("Question 2", result)
            else:
                # If it's a dict with a string message field, check that instead
                result_msg = result.get("message", "")
                self.assertIn(self.test_name, result_msg)
                self.assertIn(self.test_description, result_msg)
                self.assertIn(self.test_template, result_msg)
                self.assertIn("Question 1", result_msg)
                self.assertIn("Question 2", result_msg)
        except Exception as e:
            self.fail(f"manage_evaluation raised {type(e)} unexpectedly: {str(e)}")
        
        # Test non-existent evaluation
        result = manage_evaluation(action="get", evaluation_id="nonexistent-id")
        self.assertTrue(result.get("error", False))
    
    def test_manage_evaluation_update(self):
        """Test the manage_evaluation tool function for updating evaluations."""
        evaluation = create_evaluation(
            self.test_name, self.test_template, self.test_questions, self.test_description
        )
        
        new_name = "Updated Name"
        result = manage_evaluation(
            action="update", evaluation_id=evaluation.id, name=new_name
        )
        
        self.assertFalse(result.get("error", True))
        self.assertIn(new_name, result.get("message", ""))
        
        # Check that the name was updated
        updated = get_evaluation(evaluation.id)
        self.assertEqual(updated.name, new_name)
        
        # Test non-existent evaluation
        result = manage_evaluation(
            action="update", evaluation_id="nonexistent-id", name=new_name
        )
        self.assertTrue(result.get("error", False))
    
    def test_manage_evaluation_delete(self):
        """Test the manage_evaluation tool function for deleting evaluations."""
        evaluation = create_evaluation(
            self.test_name, self.test_template, self.test_questions, self.test_description
        )
        
        result = manage_evaluation(action="delete", evaluation_id=evaluation.id)
        
        self.assertFalse(result.get("error", True))
        
        # Check that the evaluation is gone
        self.assertIsNone(get_evaluation(evaluation.id))
        
        # Test non-existent evaluation
        result = manage_evaluation(action="delete", evaluation_id="nonexistent-id")
        self.assertTrue(result.get("error", False))
    
    def test_manage_evaluation_invalid_action(self):
        """Test the manage_evaluation tool function with an invalid action."""
        result = manage_evaluation(action="invalid")
        self.assertTrue(result.get("error", False))
        self.assertIn("Invalid action", result.get("message", ""))
    
    def test_manage_evaluation_missing_params(self):
        """Test the manage_evaluation tool function with missing parameters."""
        # Missing evaluation_id for get action
        result = manage_evaluation(action="get")
        self.assertTrue(result.get("error", False))
        self.assertIn("Missing required parameters", result.get("message", ""))
        
        # Try to create with invalid parameters (missing name)
        result = manage_evaluation(action="create")
        self.assertTrue(result.get("error", False)) 
        self.assertIn("Missing required parameters", result.get("message", ""))
    
    def test_evaluation_model(self):
        """Test the Evaluation model functionality."""
        evaluation = Evaluation(
            name=self.test_name,
            template=self.test_template,
            questions=self.test_questions,
            description=self.test_description,
        )
        
        # Test to_dict method
        eval_dict = evaluation.to_dict()
        self.assertEqual(eval_dict["name"], self.test_name)
        self.assertEqual(eval_dict["template"], self.test_template)
        self.assertEqual(eval_dict["questions"], self.test_questions)
        self.assertEqual(eval_dict["description"], self.test_description)
        
        # Test from_dict method
        eval2 = Evaluation.from_dict(eval_dict)
        self.assertEqual(eval2.id, evaluation.id)
        self.assertEqual(eval2.name, self.test_name)
        self.assertEqual(eval2.template, self.test_template)
        self.assertEqual(eval2.questions, self.test_questions)
        self.assertEqual(eval2.description, self.test_description)


if __name__ == "__main__":
    unittest.main()