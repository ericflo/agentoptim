"""Tests for the dataset module."""

import os
import json
import shutil
import tempfile
import unittest
from unittest import mock

from agentoptim.dataset import (
    Dataset,
    DataItem,
    create_dataset,
    get_dataset,
    update_dataset,
    delete_dataset,
    list_datasets,
    split_dataset,
    sample_dataset,
    import_from_jsonl,
    manage_dataset,
)


class TestDataset(unittest.TestCase):
    """Test suite for dataset functionality."""
    
    def setUp(self):
        """Set up the test environment."""
        # Create a temporary directory for test data
        self.temp_dir = tempfile.mkdtemp()
        
        # Mock the DATASETS_DIR to use our temporary directory
        self.patcher = mock.patch("agentoptim.dataset.DATASETS_DIR", self.temp_dir)
        self.patcher.start()
        
        # Sample data for tests
        self.test_name = "Test Dataset"
        self.test_items = [
            {"input": "Test input 1", "expected_output": "Test output 1"},
            {"input": "Test input 2", "expected_output": "Test output 2"},
            {"input": "Test input 3", "expected_output": "Test output 3"},
        ]
        self.test_description = "Test description"
        self.test_source = "Test source"
        self.test_tags = ["test", "sample"]
    
    def tearDown(self):
        """Clean up after tests."""
        # Stop the patcher
        self.patcher.stop()
        
        # Remove the temporary directory
        shutil.rmtree(self.temp_dir)
    
    def test_create_dataset(self):
        """Test creating a dataset."""
        dataset = create_dataset(
            self.test_name, self.test_items, self.test_description, 
            self.test_source, self.test_tags
        )
        
        self.assertEqual(dataset.name, self.test_name)
        self.assertEqual(len(dataset.items), len(self.test_items))
        self.assertEqual(dataset.description, self.test_description)
        self.assertEqual(dataset.source, self.test_source)
        self.assertEqual(dataset.tags, self.test_tags)
        
        # Check that all items were converted to DataItem objects
        for i, item in enumerate(dataset.items):
            self.assertIsInstance(item, DataItem)
            self.assertEqual(item.input, self.test_items[i]["input"])
            self.assertEqual(item.expected_output, self.test_items[i]["expected_output"])
        
        # Check that the file was created
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, f"{dataset.id}.json")))
    
    def test_get_dataset(self):
        """Test retrieving a dataset."""
        created = create_dataset(
            self.test_name, self.test_items, self.test_description, 
            self.test_source, self.test_tags
        )
        
        retrieved = get_dataset(created.id)
        
        self.assertEqual(retrieved.id, created.id)
        self.assertEqual(retrieved.name, self.test_name)
        self.assertEqual(len(retrieved.items), len(self.test_items))
        self.assertEqual(retrieved.description, self.test_description)
        self.assertEqual(retrieved.source, self.test_source)
        self.assertEqual(retrieved.tags, self.test_tags)
        
        # Test non-existent dataset
        self.assertIsNone(get_dataset("nonexistent-id"))
    
    def test_update_dataset(self):
        """Test updating a dataset."""
        dataset = create_dataset(
            self.test_name, self.test_items, self.test_description, 
            self.test_source, self.test_tags
        )
        
        # Update some fields
        new_name = "Updated Name"
        new_items = [{"input": "New input", "expected_output": "New output"}]
        new_tags = ["updated", "test"]
        
        updated = update_dataset(
            dataset.id, name=new_name, items=new_items, tags=new_tags
        )
        
        self.assertEqual(updated.id, dataset.id)
        self.assertEqual(updated.name, new_name)
        self.assertEqual(len(updated.items), 1)
        self.assertEqual(updated.items[0].input, "New input")
        self.assertEqual(updated.description, self.test_description)  # Unchanged
        self.assertEqual(updated.source, self.test_source)  # Unchanged
        self.assertEqual(updated.tags, new_tags)
        
        # Test non-existent dataset
        self.assertIsNone(update_dataset("nonexistent-id", name="New Name"))
    
    def test_delete_dataset(self):
        """Test deleting a dataset."""
        dataset = create_dataset(
            self.test_name, self.test_items, self.test_description
        )
        
        # Check that the file exists
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, f"{dataset.id}.json")))
        
        # Delete and check that it returns True
        self.assertTrue(delete_dataset(dataset.id))
        
        # Check that the file is gone
        self.assertFalse(os.path.exists(os.path.join(self.temp_dir, f"{dataset.id}.json")))
        
        # Test deleting non-existent dataset
        self.assertFalse(delete_dataset("nonexistent-id"))
    
    def test_list_datasets(self):
        """Test listing datasets."""
        # Create a few datasets
        dataset1 = create_dataset("Dataset 1", [{"input": "Input 1"}])
        dataset2 = create_dataset("Dataset 2", [{"input": "Input 2"}])
        
        datasets = list_datasets()
        
        self.assertEqual(len(datasets), 2)
        self.assertIn(dataset1.id, [d.id for d in datasets])
        self.assertIn(dataset2.id, [d.id for d in datasets])
    
    def test_split_dataset(self):
        """Test splitting a dataset."""
        # Create a dataset with 10 items
        items = [{"input": f"Input {i}", "expected_output": f"Output {i}"} for i in range(10)]
        dataset = create_dataset("Split Test", items)
        
        # Split with 20% test ratio
        result = split_dataset(dataset.id, 0.2, 42)
        
        self.assertIn("train_dataset_id", result)
        self.assertIn("test_dataset_id", result)
        
        # Get the split datasets
        train_dataset = get_dataset(result["train_dataset_id"])
        test_dataset = get_dataset(result["test_dataset_id"])
        
        # Check proportions (20% of 10 = 2 items in test set, 8 in train set)
        self.assertEqual(len(train_dataset.items), 8)
        self.assertEqual(len(test_dataset.items), 2)
        
        # Check properties
        self.assertIn("train", train_dataset.tags)
        self.assertIn("test", test_dataset.tags)
        self.assertIn("Split Test", train_dataset.name)
        self.assertIn("Split Test", test_dataset.name)
        
        # Test with non-existent dataset
        error_result = split_dataset("nonexistent-id")
        self.assertIn("error", error_result)
    
    def test_sample_dataset(self):
        """Test sampling from a dataset."""
        # Create a dataset with 10 items
        items = [{"input": f"Input {i}", "expected_output": f"Output {i}"} for i in range(10)]
        dataset = create_dataset("Sample Test", items)
        
        # Sample 3 items
        result = sample_dataset(dataset.id, 3, 42)
        
        self.assertIn("sample_dataset_id", result)
        
        # Get the sampled dataset
        sampled_dataset = get_dataset(result["sample_dataset_id"])
        
        # Check sample size
        self.assertEqual(len(sampled_dataset.items), 3)
        
        # Check properties
        self.assertIn("sample", sampled_dataset.tags)
        self.assertIn("Sample Test", sampled_dataset.name)
        
        # Test with invalid sample size
        error_result = sample_dataset(dataset.id, 20)  # Larger than dataset
        self.assertIn("error", error_result)
        
        error_result = sample_dataset(dataset.id, 0)  # Zero size
        self.assertIn("error", error_result)
        
        # Test with non-existent dataset
        error_result = sample_dataset("nonexistent-id", 3)
        self.assertIn("error", error_result)
    
    def test_import_from_jsonl(self):
        """Test importing a dataset from a JSONL file."""
        # Create a temporary JSONL file
        jsonl_file = os.path.join(self.temp_dir, "test.jsonl")
        
        with open(jsonl_file, "w") as f:
            for i in range(5):
                json.dump({
                    "question": f"Question {i}",
                    "answer": f"Answer {i}",
                    "metadata": {"difficulty": i % 3}
                }, f)
                f.write("\n")
        
        # Import the file
        result = import_from_jsonl(
            jsonl_file,
            "Imported Dataset",
            input_field="question",
            output_field="answer",
            description="Imported test dataset",
            tags=["imported", "test"],
        )
        
        self.assertIn("dataset_id", result)
        
        # Get the imported dataset
        dataset = get_dataset(result["dataset_id"])
        
        # Check import results
        self.assertEqual(dataset.name, "Imported Dataset")
        self.assertEqual(len(dataset.items), 5)
        self.assertEqual(dataset.description, "Imported test dataset")
        self.assertIn("imported", dataset.tags)
        self.assertIn("test", dataset.tags)
        
        # Check the first item
        self.assertEqual(dataset.items[0].input, "Question 0")
        self.assertEqual(dataset.items[0].expected_output, "Answer 0")
        # Metadata is accessible
        self.assertIsNotNone(dataset.items[0].metadata)
        self.assertEqual(dataset.items[0].metadata.get("difficulty"), 0)
        
        # Test with non-existent file
        error_result = import_from_jsonl(
            "nonexistent-file.jsonl",
            "Test Dataset",
        )
        self.assertIn("error", error_result)
    
    def test_manage_dataset_create(self):
        """Test the manage_dataset function for creating datasets."""
        result = manage_dataset(
            action="create",
            name=self.test_name,
            items=self.test_items,
            description=self.test_description,
            source=self.test_source,
            tags=self.test_tags,
        )
        
        self.assertEqual(result["error"], False)
        self.assertIn(self.test_name, result["message"])
        
        # Check that one dataset now exists
        datasets = list_datasets()
        self.assertEqual(len(datasets), 1)
    
    def test_manage_dataset_list(self):
        """Test the manage_dataset function for listing datasets."""
        # Create a few datasets
        create_dataset("Dataset 1", [{"input": "Input 1"}])
        create_dataset("Dataset 2", [{"input": "Input 2"}])
        
        result = manage_dataset(action="list")
        
        self.assertEqual(result["error"], False)
        self.assertEqual(len(result["items"]), 2)
        dataset_names = [item["name"] for item in result["items"]]
        self.assertIn("Dataset 1", dataset_names)
        self.assertIn("Dataset 2", dataset_names)
    
    def test_manage_dataset_get(self):
        """Test the manage_dataset function for getting a dataset."""
        dataset = create_dataset(
            self.test_name, self.test_items, self.test_description, 
            self.test_source, self.test_tags
        )
        
        result = manage_dataset(action="get", dataset_id=dataset.id)
        
        self.assertEqual(result["error"], False)
        self.assertIn(self.test_name, result["message"])
        self.assertIn(self.test_description, result["message"])
        self.assertIn(self.test_source, result["message"])
        self.assertIn("test", result["message"])  # Tag
        self.assertIn("Test input 1", result["message"])
        
        # Test non-existent dataset
        result = manage_dataset(action="get", dataset_id="nonexistent-id")
        self.assertEqual(result["error"], True)
    
    def test_manage_dataset_update(self):
        """Test the manage_dataset function for updating datasets."""
        dataset = create_dataset(
            self.test_name, self.test_items, self.test_description
        )
        
        new_name = "Updated Name"
        result = manage_dataset(
            action="update", dataset_id=dataset.id, name=new_name
        )
        
        self.assertEqual(result["error"], False)
        self.assertIn(new_name, result["message"])
        
        # Check that the name was updated
        updated = get_dataset(dataset.id)
        self.assertEqual(updated.name, new_name)
        
        # Test non-existent dataset
        result = manage_dataset(
            action="update", dataset_id="nonexistent-id", name=new_name
        )
        self.assertEqual(result["error"], True)
    
    def test_manage_dataset_delete(self):
        """Test the manage_dataset function for deleting datasets."""
        dataset = create_dataset(
            self.test_name, self.test_items, self.test_description
        )
        
        result = manage_dataset(action="delete", dataset_id=dataset.id)
        
        self.assertEqual(result["error"], False)
        
        # Check that the dataset is gone
        self.assertIsNone(get_dataset(dataset.id))
        
        # Test non-existent dataset
        result = manage_dataset(action="delete", dataset_id="nonexistent-id")
        self.assertEqual(result["error"], True)
    
    def test_manage_dataset_split(self):
        """Test the manage_dataset function for splitting datasets."""
        # Create a dataset with 10 items
        items = [{"input": f"Input {i}", "expected_output": f"Output {i}"} for i in range(10)]
        dataset = create_dataset("Split Test", items)
        
        result = manage_dataset(
            action="split", dataset_id=dataset.id, test_ratio=0.3, seed=42
        )
        
        self.assertEqual(result["error"], False)
        self.assertIn("Train dataset ID", result["message"])
        self.assertIn("Test dataset ID", result["message"])
        
        # Test with invalid ratio
        result = manage_dataset(
            action="split", dataset_id=dataset.id, test_ratio=1.5
        )
        self.assertEqual(result["error"], True)
    
    def test_manage_dataset_sample(self):
        """Test the manage_dataset function for sampling datasets."""
        # Create a dataset with 10 items
        items = [{"input": f"Input {i}", "expected_output": f"Output {i}"} for i in range(10)]
        dataset = create_dataset("Sample Test", items)
        
        result = manage_dataset(
            action="sample", dataset_id=dataset.id, sample_size=3, seed=42
        )
        
        self.assertEqual(result["error"], False)
        self.assertIn("Sample dataset ID", result["message"])
        
        # Test with invalid sample size
        result = manage_dataset(
            action="sample", dataset_id=dataset.id, sample_size=0
        )
        self.assertEqual(result["error"], True)
    
    def test_manage_dataset_import(self):
        """Test the manage_dataset function for importing datasets."""
        # Create a temporary JSONL file
        jsonl_file = os.path.join(self.temp_dir, "test.jsonl")
        
        with open(jsonl_file, "w") as f:
            for i in range(5):
                json.dump({
                    "question": f"Question {i}",
                    "answer": f"Answer {i}",
                }, f)
                f.write("\n")
        
        result = manage_dataset(
            action="import",
            name="Imported Dataset",
            filepath=jsonl_file,
            input_field="question",
            output_field="answer",
            description="Imported test dataset",
        )
        
        self.assertEqual(result["error"], False)
        self.assertIn("Imported Dataset", result["message"])
        
        # Test with non-existent file
        result = manage_dataset(
            action="import",
            name="Invalid Import",
            filepath="nonexistent-file.jsonl",
        )
        self.assertEqual(result["error"], True)
    
    def test_manage_dataset_invalid_action(self):
        """Test the manage_dataset function with an invalid action."""
        result = manage_dataset(action="invalid")
        self.assertEqual(result["error"], True)
        self.assertIn("Invalid action", result["message"])
    
    def test_manage_dataset_missing_params(self):
        """Test the manage_dataset function with missing parameters."""
        # Missing dataset_id for get action
        result = manage_dataset(action="get")
        self.assertEqual(result["error"], True)
        self.assertIn("Missing required parameters", result["message"])
        
        # Missing required params for create action
        result = manage_dataset(action="create", name=self.test_name)
        self.assertEqual(result["error"], True)
        self.assertIn("Missing required parameters", result["message"])
    
    def test_dataset_model(self):
        """Test the Dataset model functionality."""
        data_items = [DataItem(input=f"Input {i}", expected_output=f"Output {i}") for i in range(3)]
        
        dataset = Dataset(
            name=self.test_name,
            description=self.test_description,
            items=data_items,
            source=self.test_source,
            tags=self.test_tags,
        )
        
        # Test to_dict method
        dataset_dict = dataset.to_dict()
        self.assertEqual(dataset_dict["name"], self.test_name)
        self.assertEqual(dataset_dict["description"], self.test_description)
        self.assertEqual(len(dataset_dict["items"]), 3)
        self.assertEqual(dataset_dict["source"], self.test_source)
        self.assertEqual(dataset_dict["tags"], self.test_tags)
        
        # Test from_dict method
        dataset2 = Dataset.from_dict(dataset_dict)
        self.assertEqual(dataset2.id, dataset.id)
        self.assertEqual(dataset2.name, self.test_name)
        self.assertEqual(len(dataset2.items), 3)
        self.assertEqual(dataset2.description, self.test_description)
        self.assertEqual(dataset2.source, self.test_source)
        self.assertEqual(dataset2.tags, self.test_tags)


if __name__ == "__main__":
    unittest.main()