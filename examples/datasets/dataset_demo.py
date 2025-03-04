#!/usr/bin/env python
"""
Demonstration of using the AgentOptim dataset functionality.

This script shows how to:
1. Import a dataset from a JSONL file
2. Split a dataset into training and testing sets
3. View dataset information
4. Sample a subset from a dataset
"""

import os
import sys
import json

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from agentoptim.dataset import (
    manage_dataset,
    get_dataset,
    list_datasets,
)


def print_section(title):
    """Print a section title with dividers."""
    print("\n" + "=" * 80)
    print(f" {title} ".center(80, "="))
    print("=" * 80 + "\n")


def main():
    """Run the dataset demonstration."""
    # Get the path to the example JSONL file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    jsonl_file = os.path.join(current_dir, "simple_math_questions.jsonl")
    
    # Step 1: Import the dataset
    print_section("IMPORTING DATASET")
    result = manage_dataset(
        action="import",
        name="Math Questions",
        filepath=jsonl_file,
        input_field="question",
        output_field="answer",
        description="A collection of simple math questions for evaluation",
        tags=["math", "education", "example"],
    )
    print(result)
    
    # Get the ID of the created dataset
    datasets = list_datasets()
    dataset_id = next((d.id for d in datasets if d.name == "Math Questions"), None)
    
    if not dataset_id:
        print("Failed to find the imported dataset.")
        return
    
    # Step 2: View the dataset
    print_section("VIEWING DATASET")
    result = manage_dataset(action="get", dataset_id=dataset_id)
    print(result)
    
    # Step 3: Split the dataset
    print_section("SPLITTING DATASET")
    result = manage_dataset(
        action="split",
        dataset_id=dataset_id,
        test_ratio=0.25,  # 25% for testing, 75% for training
        seed=42,  # For reproducibility
    )
    print(result)
    
    # Get IDs of the split datasets from the output
    # (In a real application, you'd parse the output or use the return value directly)
    
    # List all datasets to find the split datasets
    all_datasets = list_datasets()
    train_id = next((d.id for d in all_datasets if d.name == "Math Questions_train"), None)
    test_id = next((d.id for d in all_datasets if d.name == "Math Questions_test"), None)
    
    # Step 4: Check the split datasets
    if train_id and test_id:
        print_section("TRAINING DATASET")
        train_dataset = get_dataset(train_id)
        print(f"Name: {train_dataset.name}")
        print(f"Number of items: {len(train_dataset.items)}")
        print(f"Tags: {', '.join(train_dataset.tags)}")
        print("\nSample items:")
        for i, item in enumerate(train_dataset.items[:3]):
            print(f"  {i+1}. Input: {item.input}")
            print(f"     Expected output: {item.expected_output}")
            print(f"     Metadata: {item.metadata}")
        
        print_section("TESTING DATASET")
        test_dataset = get_dataset(test_id)
        print(f"Name: {test_dataset.name}")
        print(f"Number of items: {len(test_dataset.items)}")
        print(f"Tags: {', '.join(test_dataset.tags)}")
        print("\nSample items:")
        for i, item in enumerate(test_dataset.items[:3]):
            print(f"  {i+1}. Input: {item.input}")
            print(f"     Expected output: {item.expected_output}")
            print(f"     Metadata: {item.metadata}")
    
    # Step 5: Create a sample dataset with specific difficulty
    print_section("CREATING A SAMPLE DATASET")
    # Get the full dataset
    full_dataset = get_dataset(dataset_id)
    
    # Filter for hard questions only
    hard_items = [
        {
            "input": item.input,
            "expected_output": item.expected_output,
            "metadata": item.metadata,
        }
        for item in full_dataset.items
        if item.metadata.get("difficulty") == "hard"
    ]
    
    # Create a new dataset with just the hard questions
    result = manage_dataset(
        action="create",
        name="Hard Math Questions",
        items=hard_items,
        description="Only the hard math questions from the original dataset",
        source=f"Filtered from {dataset_id}",
        tags=["math", "hard", "filtered"],
    )
    print(result)
    
    # Step 6: Sample a random subset
    print_section("SAMPLING FROM DATASET")
    result = manage_dataset(
        action="sample",
        dataset_id=dataset_id,
        sample_size=5,  # Get 5 random questions
        seed=123,  # For reproducibility
    )
    print(result)
    
    print_section("DEMONSTRATION COMPLETE")


if __name__ == "__main__":
    main()