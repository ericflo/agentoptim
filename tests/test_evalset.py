"""Tests for the evalset module."""

import os
import shutil
import tempfile
import unittest
from unittest import mock

from agentoptim.evalset import (
    EvalSet,
    create_evalset,
    get_evalset,
    update_evalset,
    delete_evalset,
    list_evalsets,
    manage_evalset,
)


class TestEvalSet(unittest.TestCase):
    """Test suite for EvalSet functionality."""
    
    def setUp(self):
        """Set up the test environment."""
        # Create a temporary directory for test data
        self.temp_dir = tempfile.mkdtemp()
        
        # Mock the EVALSETS_DIR to use our temporary directory
        self.patcher = mock.patch("agentoptim.evalset.EVALSETS_DIR", self.temp_dir)
        self.patcher.start()
        
        # Sample data for tests
        self.test_name = "Test EvalSet"
        self.test_template = """
        Given this conversation:
        {{ conversation }}
        
        Please answer the following yes/no question about the final assistant response:
        {{ eval_question }}
        
        Return a JSON object with the following format:
        {"judgment": 1} for yes or {"judgment": 0} for no.
        """
        self.test_questions = ["Question 1", "Question 2"]
        self.test_short_description = "Concise summary of test EvalSet for unit testing"
        self.test_long_description = "This is a detailed explanation of the test EvalSet that is used for unit testing purposes. It provides a comprehensive description of how the EvalSet works, what criteria it evaluates, and how results should be interpreted. This long description is meant to provide all the necessary context for anyone who wants to use this EvalSet effectively." + " " * 50  # Padding to exceed 256 chars
        self.test_description = "Test description"  # For backward compatibility
    
    def tearDown(self):
        """Clean up after tests."""
        # Stop the patcher
        self.patcher.stop()
        
        # Remove the temporary directory
        shutil.rmtree(self.temp_dir)
    
    def test_create_evalset(self):
        """Test creating an EvalSet."""
        evalset = create_evalset(
            self.test_name, 
            self.test_questions, 
            self.test_short_description,
            self.test_long_description,
            self.test_description
        )
        
        self.assertEqual(evalset.name, self.test_name)
        # Templates are now system-defined, so we don't test for that
        self.assertEqual(evalset.questions, self.test_questions)
        self.assertEqual(evalset.short_description, self.test_short_description)
        self.assertEqual(evalset.long_description, self.test_long_description)
        self.assertEqual(evalset.description, self.test_description)
        
        # Check that the file was created
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, f"{evalset.id}.json")))
    
    def test_get_evalset(self):
        """Test retrieving an EvalSet."""
        created = create_evalset(
            self.test_name, 
            self.test_questions, 
            self.test_short_description,
            self.test_long_description,
            self.test_description
        )
        
        retrieved = get_evalset(created.id)
        
        self.assertEqual(retrieved.id, created.id)
        self.assertEqual(retrieved.name, self.test_name)
        # Templates are now system-defined, so we don't test for exact template
        self.assertEqual(retrieved.questions, self.test_questions)
        self.assertEqual(retrieved.short_description, self.test_short_description)
        self.assertEqual(retrieved.long_description, self.test_long_description)
        self.assertEqual(retrieved.description, self.test_description)
        
        # Test non-existent EvalSet
        self.assertIsNone(get_evalset("nonexistent-id"))
    
    def test_update_evalset(self):
        """Test updating an EvalSet."""
        evalset = create_evalset(
            self.test_name, 
            self.test_questions,
            self.test_short_description,
            self.test_long_description,
            self.test_description
        )
        
        # Update some fields
        new_name = "Updated Name"
        new_questions = ["New Question 1", "New Question 2", "New Question 3"]
        new_short_description = "Updated short description"
        new_long_description = "This is an updated detailed explanation of the test EvalSet. It provides even more comprehensive information about how the EvalSet works, what criteria it evaluates, and how results should be interpreted. The long description should give users all the context they need." + " " * 50  # Padding to exceed 256 chars
        
        updated = update_evalset(
            evalset.id, 
            name=new_name, 
            questions=new_questions,
            short_description=new_short_description,
            long_description=new_long_description
        )
        
        self.assertEqual(updated.id, evalset.id)
        self.assertEqual(updated.name, new_name)
        # Templates are now system-defined, so we don't test for template
        self.assertEqual(updated.questions, new_questions)
        self.assertEqual(updated.short_description, new_short_description)
        self.assertEqual(updated.long_description, new_long_description)
        self.assertEqual(updated.description, self.test_description)  # Unchanged
        
        # Test non-existent EvalSet
        self.assertIsNone(update_evalset("nonexistent-id", name="New Name"))
    
    def test_delete_evalset(self):
        """Test deleting an EvalSet."""
        evalset = create_evalset(
            self.test_name, 
            self.test_questions,
            self.test_short_description,
            self.test_long_description,
            self.test_description
        )
        
        # Check that the file exists
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, f"{evalset.id}.json")))
        
        # Delete and check that it returns True
        self.assertTrue(delete_evalset(evalset.id))
        
        # Check that the file is gone
        self.assertFalse(os.path.exists(os.path.join(self.temp_dir, f"{evalset.id}.json")))
        
        # Test deleting non-existent EvalSet
        self.assertFalse(delete_evalset("nonexistent-id"))
    
    def test_list_evalsets(self):
        """Test listing EvalSets."""
        # Create a few EvalSets
        eval1 = create_evalset(
            "Eval 1", 
            ["Q1", "Q2"], 
            "Short description 1",
            "Long description for EvalSet 1. This is a comprehensive explanation of what this EvalSet measures and how to interpret the results. It provides detailed context for users." + " " * 100
        )
        eval2 = create_evalset(
            "Eval 2", 
            ["Q3", "Q4"],
            "Short description 2",
            "Long description for EvalSet 2. This is a comprehensive explanation of what this EvalSet measures and how to interpret the results. It provides detailed context for users." + " " * 100
        )
        
        evalsets = list_evalsets()
        
        self.assertEqual(len(evalsets), 2)
        self.assertIn(eval1.id, [e.id for e in evalsets])
        self.assertIn(eval2.id, [e.id for e in evalsets])
    
    def test_manage_evalset_create(self):
        """Test the manage_evalset function for creating EvalSets."""
        result = manage_evalset(
            action="create",
            name=self.test_name,
            template=self.test_template,  # Will be ignored
            questions=self.test_questions,
            short_description=self.test_short_description,
            long_description=self.test_long_description,
            description=self.test_description,
        )
        
        self.assertEqual(result.get("status"), "success")
        self.assertIn(self.test_name, result.get("message", ""))
        
        # Check that one EvalSet now exists
        evalsets = list_evalsets()
        self.assertEqual(len(evalsets), 1)
    
    def test_manage_evalset_list(self):
        """Test the manage_evalset function for listing EvalSets."""
        # Create a few EvalSets
        create_evalset(
            "Eval 1", 
            ["Q1", "Q2"], 
            "Short description 1",
            "Long description for EvalSet 1. This is a comprehensive explanation of what this EvalSet measures and how to interpret the results. It provides detailed context for users." + " " * 100
        )
        create_evalset(
            "Eval 2", 
            ["Q3", "Q4"],
            "Short description 2",
            "Long description for EvalSet 2. This is a comprehensive explanation of what this EvalSet measures and how to interpret the results. It provides detailed context for users." + " " * 100
        )
        
        result = manage_evalset(action="list")
        
        self.assertFalse(result.get("error", False))
        
        # Check if items are in the list
        items = result.get("items", [])
        names = [item.get("name") for item in items]
        self.assertIn("Eval 1", names)
        self.assertIn("Eval 2", names)
    
    def test_manage_evalset_get(self):
        """Test the manage_evalset function for getting an EvalSet."""
        evalset = create_evalset(
            self.test_name, 
            self.test_questions,
            self.test_short_description,
            self.test_long_description,
            self.test_description
        )
        
        result = manage_evalset(action="get", evalset_id=evalset.id)
        
        self.assertEqual(result.get("status"), "success")
        self.assertIn("evalset", result)
        
        evalset_data = result.get("evalset", {})
        self.assertEqual(evalset_data.get("id"), evalset.id)
        self.assertEqual(evalset_data.get("name"), self.test_name)
        self.assertEqual(evalset_data.get("description"), self.test_description)
        self.assertEqual(evalset_data.get("questions"), self.test_questions)
        
        # Test non-existent EvalSet
        result = manage_evalset(action="get", evalset_id="nonexistent-id")
        self.assertTrue(result.get("error", False))
    
    def test_manage_evalset_update(self):
        """Test the manage_evalset function for updating EvalSets."""
        evalset = create_evalset(
            self.test_name, 
            self.test_questions,
            self.test_short_description,
            self.test_long_description,
            self.test_description
        )
        
        new_name = "Updated Name"
        result = manage_evalset(
            action="update", evalset_id=evalset.id, name=new_name
        )
        
        self.assertEqual(result.get("status"), "success")
        self.assertIn(new_name, result.get("message", ""))
        
        # Check that the name was updated
        updated = get_evalset(evalset.id)
        self.assertEqual(updated.name, new_name)
        
        # Test non-existent EvalSet
        result = manage_evalset(
            action="update", evalset_id="nonexistent-id", name=new_name
        )
        self.assertTrue(result.get("error", False))
    
    def test_manage_evalset_delete(self):
        """Test the manage_evalset function for deleting EvalSets."""
        evalset = create_evalset(
            self.test_name, 
            self.test_questions,
            self.test_short_description,
            self.test_long_description,
            self.test_description
        )
        
        result = manage_evalset(action="delete", evalset_id=evalset.id)
        
        self.assertEqual(result.get("status"), "success")
        
        # Check that the EvalSet is gone
        self.assertIsNone(get_evalset(evalset.id))
        
        # Test non-existent EvalSet
        result = manage_evalset(action="delete", evalset_id="nonexistent-id")
        self.assertTrue(result.get("error", False))
    
    def test_manage_evalset_invalid_action(self):
        """Test the manage_evalset function with an invalid action."""
        result = manage_evalset(action="invalid")
        self.assertTrue(result.get("error", False))
        self.assertIn("Invalid action", result.get("message", ""))
    
    def test_manage_evalset_missing_params(self):
        """Test the manage_evalset function with missing parameters."""
        # Missing evalset_id for get action
        result = manage_evalset(action="get")
        self.assertTrue(result.get("error", False))
        self.assertIn("Missing required parameters", result.get("message", ""))
        
        # Try to create with invalid parameters (missing name)
        result = manage_evalset(action="create")
        self.assertTrue(result.get("error", False)) 
        self.assertIn("Missing required parameters", result.get("message", ""))
    
    def test_evalset_model(self):
        """Test the EvalSet model functionality."""
        evalset = EvalSet(
            name=self.test_name,
            template=self.test_template,
            questions=self.test_questions,
            description=self.test_description,
        )
        
        # Test to_dict method
        eval_dict = evalset.to_dict()
        self.assertEqual(eval_dict["name"], self.test_name)
        self.assertEqual(eval_dict["template"], self.test_template)
        self.assertEqual(eval_dict["questions"], self.test_questions)
        self.assertEqual(eval_dict["description"], self.test_description)
        
        # Test from_dict method
        eval2 = EvalSet.from_dict(eval_dict)
        self.assertEqual(eval2.id, evalset.id)
        self.assertEqual(eval2.name, self.test_name)
        self.assertEqual(eval2.template, self.test_template)
        self.assertEqual(eval2.questions, self.test_questions)
        self.assertEqual(eval2.description, self.test_description)
    
    def test_template_validation(self):
        """Test that template validation works correctly."""
        # Templates are now system-defined, so we skip direct validation tests
        # Instead, let's just check that the default template has the required placeholders
        evalset = create_evalset(
            self.test_name, 
            self.test_questions,
            self.test_short_description,
            self.test_long_description,
            self.test_description
        )
        self.assertIn("{{ conversation }}", evalset.template)
        self.assertIn("{{ eval_question }}", evalset.template)
    
    def test_questions_count_validation(self):
        """Test that the maximum questions count validation works."""
        too_many_questions = ["Question"] * 101  # 101 questions exceeds the limit
        
        with self.assertRaises(ValueError) as context:
            EvalSet(
                name=self.test_name,
                questions=too_many_questions,
                short_description=self.test_short_description,
                long_description=self.test_long_description
            )
        self.assertIn("Maximum of 100 questions", str(context.exception))
        
        # Test with the maximum allowed
        max_questions = ["Question"] * 100  # 100 questions is the maximum allowed
        evalset = EvalSet(
            name=self.test_name,
            questions=max_questions,
            short_description=self.test_short_description,
            long_description=self.test_long_description
        )
        self.assertEqual(len(evalset.questions), 100)
        
    def test_short_description_validation(self):
        """Test short_description length validation."""
        # Test too short
        with self.assertRaises(ValueError) as context:
            EvalSet(
                name=self.test_name,
                questions=self.test_questions,
                short_description="Short",  # Too short (< 6 chars)
                long_description=self.test_long_description
            )
        self.assertIn("short_description must be at least 6 characters", str(context.exception))
        
        # Test too long
        with self.assertRaises(ValueError) as context:
            EvalSet(
                name=self.test_name,
                questions=self.test_questions,
                short_description="X" * 129,  # Too long (> 128 chars)
                long_description=self.test_long_description
            )
        self.assertIn("short_description must not exceed 128 characters", str(context.exception))
    
    def test_long_description_validation(self):
        """Test long_description length validation."""
        # Test too short
        with self.assertRaises(ValueError) as context:
            EvalSet(
                name=self.test_name,
                questions=self.test_questions,
                short_description=self.test_short_description,
                long_description="This is too short for a long description."  # Too short (< 256 chars)
            )
        self.assertIn("long_description must be at least 256 characters", str(context.exception))
        
        # Test too long
        with self.assertRaises(ValueError) as context:
            EvalSet(
                name=self.test_name,
                questions=self.test_questions,
                short_description=self.test_short_description,
                long_description="X" * 1025  # Too long (> 1024 chars)
            )
        self.assertIn("long_description must not exceed 1024 characters", str(context.exception))


if __name__ == "__main__":
    unittest.main()