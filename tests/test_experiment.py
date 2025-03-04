"""Tests for the experiment module."""

import os
import shutil
import tempfile
import unittest
from unittest import mock

from agentoptim.experiment import (
    Experiment,
    PromptVariant,
    PromptVariable,
    PromptVariantType,
    create_experiment,
    get_experiment,
    update_experiment,
    delete_experiment,
    list_experiments,
    duplicate_experiment,
    manage_experiment,
)


class TestExperiment(unittest.TestCase):
    """Test suite for experiment functionality."""
    
    def setUp(self):
        """Set up the test environment."""
        # Create a temporary directory for test data
        self.temp_dir = tempfile.mkdtemp()
        
        # Mock the EXPERIMENTS_DIR to use our temporary directory
        self.patcher = mock.patch("agentoptim.experiment.EXPERIMENTS_DIR", self.temp_dir)
        self.patcher.start()
        
        # Sample data for tests
        self.test_name = "Test Experiment"
        self.test_description = "Test description"
        self.test_dataset_id = "dataset-123"
        self.test_evaluation_id = "eval-456"
        self.test_model_name = "test-model"
        
        # Create a simple prompt variant
        self.test_prompt_variants = [
            {
                "name": "Basic Variant",
                "type": PromptVariantType.SYSTEM,
                "template": "You are a helpful assistant. {{input}}",
                "description": "Basic system prompt"
            },
            {
                "name": "Advanced Variant",
                "type": PromptVariantType.COMBINED,
                "template": "System: Be {{tone}} and {{style}}.\nUser: {{input}}",
                "description": "Variant with variables",
                "variables": [
                    {
                        "name": "tone",
                        "description": "Tone of the response",
                        "options": ["friendly", "professional", "casual"]
                    },
                    {
                        "name": "style",
                        "description": "Style of the response",
                        "options": ["detailed", "concise", "creative"]
                    }
                ]
            }
        ]
    
    def tearDown(self):
        """Clean up after tests."""
        # Stop the patcher
        self.patcher.stop()
        
        # Remove the temporary directory
        shutil.rmtree(self.temp_dir)
    
    def test_create_experiment(self):
        """Test creating an experiment."""
        experiment = create_experiment(
            name=self.test_name,
            description=self.test_description,
            dataset_id=self.test_dataset_id,
            evaluation_id=self.test_evaluation_id,
            prompt_variants=self.test_prompt_variants,
            model_name=self.test_model_name,
            temperature=0.7,
            max_tokens=100,
        )
        
        self.assertEqual(experiment.name, self.test_name)
        self.assertEqual(experiment.description, self.test_description)
        self.assertEqual(experiment.dataset_id, self.test_dataset_id)
        self.assertEqual(experiment.evaluation_id, self.test_evaluation_id)
        self.assertEqual(experiment.model_name, self.test_model_name)
        self.assertEqual(experiment.temperature, 0.7)
        self.assertEqual(experiment.max_tokens, 100)
        self.assertEqual(experiment.status, "created")
        self.assertEqual(len(experiment.prompt_variants), 2)
        
        # Check that prompt variants were properly converted
        self.assertIsInstance(experiment.prompt_variants[0], PromptVariant)
        self.assertEqual(experiment.prompt_variants[0].name, "Basic Variant")
        self.assertEqual(experiment.prompt_variants[0].type, PromptVariantType.SYSTEM)
        
        # Check variables in the second variant
        self.assertIsInstance(experiment.prompt_variants[1], PromptVariant)
        self.assertEqual(len(experiment.prompt_variants[1].variables), 2)
        self.assertIsInstance(experiment.prompt_variants[1].variables[0], PromptVariable)
        self.assertEqual(experiment.prompt_variants[1].variables[0].name, "tone")
        self.assertEqual(len(experiment.prompt_variants[1].variables[0].options), 3)
        
        # Check that the file was created
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, f"{experiment.id}.json")))
    
    def test_get_experiment(self):
        """Test retrieving an experiment."""
        created = create_experiment(
            name=self.test_name,
            description=self.test_description,
            dataset_id=self.test_dataset_id,
            evaluation_id=self.test_evaluation_id,
            prompt_variants=self.test_prompt_variants,
            model_name=self.test_model_name,
        )
        
        retrieved = get_experiment(created.id)
        
        self.assertEqual(retrieved.id, created.id)
        self.assertEqual(retrieved.name, self.test_name)
        self.assertEqual(retrieved.description, self.test_description)
        self.assertEqual(len(retrieved.prompt_variants), 2)
        
        # Test non-existent experiment
        self.assertIsNone(get_experiment("nonexistent-id"))
    
    def test_update_experiment(self):
        """Test updating an experiment."""
        experiment = create_experiment(
            name=self.test_name,
            description=self.test_description,
            dataset_id=self.test_dataset_id,
            evaluation_id=self.test_evaluation_id,
            prompt_variants=self.test_prompt_variants,
            model_name=self.test_model_name,
        )
        
        # Update some fields
        new_name = "Updated Experiment"
        new_status = "running"
        new_results = {"variant-1": {"average_score": 0.85}}
        
        updated = update_experiment(
            experiment.id,
            name=new_name,
            status=new_status,
            results=new_results,
        )
        
        self.assertEqual(updated.id, experiment.id)
        self.assertEqual(updated.name, new_name)
        self.assertEqual(updated.status, new_status)
        self.assertEqual(updated.results, new_results)
        self.assertEqual(updated.description, self.test_description)  # Unchanged
        
        # Test non-existent experiment
        self.assertIsNone(update_experiment("nonexistent-id", name="New Name"))
    
    def test_delete_experiment(self):
        """Test deleting an experiment."""
        experiment = create_experiment(
            name=self.test_name,
            dataset_id=self.test_dataset_id,
            evaluation_id=self.test_evaluation_id,
            prompt_variants=self.test_prompt_variants,
            model_name=self.test_model_name,
        )
        
        # Check that the file exists
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, f"{experiment.id}.json")))
        
        # Delete and check that it returns True
        self.assertTrue(delete_experiment(experiment.id))
        
        # Check that the file is gone
        self.assertFalse(os.path.exists(os.path.join(self.temp_dir, f"{experiment.id}.json")))
        
        # Test deleting non-existent experiment
        self.assertFalse(delete_experiment("nonexistent-id"))
    
    def test_list_experiments(self):
        """Test listing experiments."""
        # Create a few experiments
        exp1 = create_experiment(
            name="Experiment 1",
            dataset_id=self.test_dataset_id,
            evaluation_id=self.test_evaluation_id,
            prompt_variants=self.test_prompt_variants[:1],
            model_name=self.test_model_name,
        )
        
        exp2 = create_experiment(
            name="Experiment 2",
            dataset_id=self.test_dataset_id,
            evaluation_id=self.test_evaluation_id,
            prompt_variants=self.test_prompt_variants,
            model_name=self.test_model_name,
        )
        
        experiments = list_experiments()
        
        self.assertEqual(len(experiments), 2)
        self.assertIn(exp1.id, [e["id"] for e in experiments])
        self.assertIn(exp2.id, [e["id"] for e in experiments])
    
    def test_duplicate_experiment(self):
        """Test duplicating an experiment."""
        experiment = create_experiment(
            name=self.test_name,
            description=self.test_description,
            dataset_id=self.test_dataset_id,
            evaluation_id=self.test_evaluation_id,
            prompt_variants=self.test_prompt_variants,
            model_name=self.test_model_name,
            temperature=0.7,
        )
        
        # Add some results to the original (these should not be copied)
        update_experiment(
            experiment.id,
            status="completed",
            results={"variant-1": {"average_score": 0.9}}
        )
        
        # Duplicate with custom name
        duplicate = duplicate_experiment(
            experiment.id,
            new_name="Duplicated Experiment",
        )
        
        self.assertNotEqual(duplicate.id, experiment.id)  # New ID
        self.assertEqual(duplicate.name, "Duplicated Experiment")
        self.assertEqual(duplicate.description, self.test_description)
        self.assertEqual(duplicate.dataset_id, self.test_dataset_id)
        self.assertEqual(duplicate.evaluation_id, self.test_evaluation_id)
        self.assertEqual(duplicate.model_name, self.test_model_name)
        self.assertEqual(duplicate.temperature, 0.7)
        self.assertEqual(duplicate.status, "created")  # Reset status
        self.assertIsNone(duplicate.results)  # No results copied
        
        # Duplicate with default name
        duplicate2 = duplicate_experiment(experiment.id)
        self.assertEqual(duplicate2.name, f"Copy of {self.test_name}")
        
        # Test with non-existent experiment
        self.assertIsNone(duplicate_experiment("nonexistent-id"))
    
    def test_manage_experiment_create(self):
        """Test the manage_experiment function for creating experiments."""
        result = manage_experiment(
            action="create",
            name=self.test_name,
            description=self.test_description,
            dataset_id=self.test_dataset_id,
            evaluation_id=self.test_evaluation_id,
            prompt_variants=self.test_prompt_variants,
            model_name=self.test_model_name,
            temperature=0.5,
        )
        
        self.assertEqual(result["error"], False)
        self.assertIn(self.test_name, result["message"])
        
        # Check that one experiment now exists
        experiments = list_experiments()
        self.assertEqual(len(experiments), 1)
    
    def test_manage_experiment_list(self):
        """Test the manage_experiment function for listing experiments."""
        # Create a few experiments
        create_experiment(
            name="Experiment 1",
            dataset_id=self.test_dataset_id,
            evaluation_id=self.test_evaluation_id,
            prompt_variants=self.test_prompt_variants[:1],
            model_name=self.test_model_name,
        )
        
        create_experiment(
            name="Experiment 2",
            dataset_id=self.test_dataset_id,
            evaluation_id=self.test_evaluation_id,
            prompt_variants=self.test_prompt_variants,
            model_name=self.test_model_name,
        )
        
        result = manage_experiment(action="list")
        
        self.assertEqual(result["error"], False)
        self.assertEqual(len(result["items"]), 2)
        experiment_names = [item["name"] for item in result["items"]]
        self.assertIn("Experiment 1", experiment_names)
        self.assertIn("Experiment 2", experiment_names)
    
    def test_manage_experiment_get(self):
        """Test the manage_experiment function for getting an experiment."""
        experiment = create_experiment(
            name=self.test_name,
            description=self.test_description,
            dataset_id=self.test_dataset_id,
            evaluation_id=self.test_evaluation_id,
            prompt_variants=self.test_prompt_variants,
            model_name=self.test_model_name,
        )
        
        result = manage_experiment(action="get", experiment_id=experiment.id)
        
        self.assertEqual(result["error"], False)
        self.assertIn(self.test_name, result["message"])
        self.assertIn(self.test_description, result["message"])
        self.assertIn(self.test_dataset_id, result["message"])
        self.assertIn(self.test_evaluation_id, result["message"])
        self.assertIn(self.test_model_name, result["message"])
        self.assertIn("Basic Variant", result["message"])
        self.assertIn("Advanced Variant", result["message"])
        
        # Test non-existent experiment
        result = manage_experiment(action="get", experiment_id="nonexistent-id")
        self.assertEqual(result["error"], True)
    
    def test_manage_experiment_update(self):
        """Test the manage_experiment function for updating experiments."""
        experiment = create_experiment(
            name=self.test_name,
            description=self.test_description,
            dataset_id=self.test_dataset_id,
            evaluation_id=self.test_evaluation_id,
            prompt_variants=self.test_prompt_variants,
            model_name=self.test_model_name,
        )
        
        new_name = "Updated Experiment"
        result = manage_experiment(
            action="update",
            experiment_id=experiment.id,
            name=new_name,
            status="running",
        )
        
        self.assertEqual(result["error"], False)
        self.assertIn(new_name, result["message"])
        
        # Check that the name was updated
        updated = get_experiment(experiment.id)
        self.assertEqual(updated.name, new_name)
        self.assertEqual(updated.status, "running")
        
        # Test non-existent experiment
        result = manage_experiment(
            action="update",
            experiment_id="nonexistent-id",
            name=new_name,
        )
        self.assertEqual(result["error"], True)
    
    def test_manage_experiment_delete(self):
        """Test the manage_experiment function for deleting experiments."""
        experiment = create_experiment(
            name=self.test_name,
            description=self.test_description,
            dataset_id=self.test_dataset_id,
            evaluation_id=self.test_evaluation_id,
            prompt_variants=self.test_prompt_variants,
            model_name=self.test_model_name,
        )
        
        result = manage_experiment(action="delete", experiment_id=experiment.id)
        
        self.assertEqual(result["error"], False)
        
        # Check that the experiment is gone
        self.assertIsNone(get_experiment(experiment.id))
        
        # Test non-existent experiment
        result = manage_experiment(action="delete", experiment_id="nonexistent-id")
        self.assertEqual(result["error"], True)
    
    def test_manage_experiment_duplicate(self):
        """Test the manage_experiment function for duplicating experiments."""
        experiment = create_experiment(
            name=self.test_name,
            description=self.test_description,
            dataset_id=self.test_dataset_id,
            evaluation_id=self.test_evaluation_id,
            prompt_variants=self.test_prompt_variants,
            model_name=self.test_model_name,
        )
        
        result = manage_experiment(
            action="duplicate",
            experiment_id=experiment.id,
            new_name="Duplicated Test",
        )
        
        self.assertEqual(result["error"], False)
        self.assertIn("Duplicated Test", result["message"])
        
        # Check that we now have two experiments
        experiments = list_experiments()
        self.assertEqual(len(experiments), 2)
        
        # Test non-existent experiment
        result = manage_experiment(
            action="duplicate",
            experiment_id="nonexistent-id",
        )
        self.assertEqual(result["error"], True)
    
    def test_manage_experiment_invalid_action(self):
        """Test the manage_experiment function with an invalid action."""
        result = manage_experiment(action="invalid")
        self.assertEqual(result["error"], True)
        self.assertIn("Invalid action", result["message"])
    
    def test_manage_experiment_missing_params(self):
        """Test the manage_experiment function with missing parameters."""
        # Missing experiment_id for get action
        result = manage_experiment(action="get")
        self.assertEqual(result["error"], True)
        self.assertIn("Missing required parameters", result["message"])
        
        # Missing required params for create action
        result = manage_experiment(action="create", name=self.test_name)
        self.assertEqual(result["error"], True)
        self.assertIn("Missing required parameters", result["message"])
    
    def test_experiment_model(self):
        """Test the Experiment model functionality."""
        variant = PromptVariant(
            name="Test Variant",
            type=PromptVariantType.SYSTEM,
            template="Test template",
        )
        
        experiment = Experiment(
            name=self.test_name,
            description=self.test_description,
            dataset_id=self.test_dataset_id,
            evaluation_id=self.test_evaluation_id,
            prompt_variants=[variant],
            model_name=self.test_model_name,
        )
        
        # Test to_dict method
        exp_dict = experiment.to_dict()
        self.assertEqual(exp_dict["name"], self.test_name)
        self.assertEqual(exp_dict["description"], self.test_description)
        self.assertEqual(exp_dict["dataset_id"], self.test_dataset_id)
        self.assertEqual(exp_dict["evaluation_id"], self.test_evaluation_id)
        self.assertEqual(exp_dict["model_name"], self.test_model_name)
        self.assertEqual(len(exp_dict["prompt_variants"]), 1)
        
        # Test from_dict method
        exp2 = Experiment.from_dict(exp_dict)
        self.assertEqual(exp2.id, experiment.id)
        self.assertEqual(exp2.name, self.test_name)
        self.assertEqual(exp2.description, self.test_description)
        self.assertEqual(exp2.dataset_id, self.test_dataset_id)
        self.assertEqual(exp2.evaluation_id, self.test_evaluation_id)
        self.assertEqual(exp2.model_name, self.test_model_name)
        self.assertEqual(len(exp2.prompt_variants), 1)


if __name__ == "__main__":
    unittest.main()