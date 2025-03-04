"""Dataset management functionality for AgentOptim."""

import os
import json
from typing import Any, Dict, List, Optional, Union, Tuple
from pathlib import Path

from pydantic import BaseModel, Field

from agentoptim.utils import (
    DATASETS_DIR,
    generate_id,
    save_json,
    load_json,
    list_json_files,
    validate_action,
    validate_required_params,
    format_error,
    format_success,
    format_list,
    ValidationError,
)


class DataItem(BaseModel):
    """Model for a single data item in a dataset."""
    
    input: str
    expected_output: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Dataset(BaseModel):
    """Model for a dataset."""
    
    id: str = Field(default_factory=generate_id)
    name: str
    description: Optional[str] = None
    items: List[DataItem]
    source: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return self.model_dump()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Dataset":
        """Create from dictionary."""
        return cls(**data)
    
    def split(self, test_ratio: float = 0.2, seed: Optional[int] = None) -> Tuple["Dataset", "Dataset"]:
        """
        Split a dataset into training and testing sets.
        
        Args:
            test_ratio: Ratio of test set size to total dataset size
            seed: Random seed for reproducibility
            
        Returns:
            A tuple of (train_dataset, test_dataset)
        """
        import random
        import math
        
        if seed is not None:
            random.seed(seed)
        
        # Shuffle the data items
        shuffled_items = list(self.items)
        random.shuffle(shuffled_items)
        
        # Calculate the split index
        test_size = math.ceil(len(shuffled_items) * test_ratio)
        
        # Split the items
        test_items = shuffled_items[:test_size]
        train_items = shuffled_items[test_size:]
        
        # Create new datasets
        train_dataset = Dataset(
            name=f"{self.name}_train",
            description=f"Training split from {self.name}",
            items=train_items,
            source=f"Split from {self.id}",
            tags=self.tags + ["train", "split"],
        )
        
        test_dataset = Dataset(
            name=f"{self.name}_test",
            description=f"Testing split from {self.name}",
            items=test_items,
            source=f"Split from {self.id}",
            tags=self.tags + ["test", "split"],
        )
        
        return train_dataset, test_dataset
    
    def sample(self, n: int, seed: Optional[int] = None) -> "Dataset":
        """
        Create a new dataset by sampling n items from this dataset.
        
        Args:
            n: Number of items to sample
            seed: Random seed for reproducibility
            
        Returns:
            A new dataset with sampled items
        """
        import random
        
        if seed is not None:
            random.seed(seed)
        
        # Ensure n is not larger than the dataset size
        n = min(n, len(self.items))
        
        # Sample the items
        sampled_items = random.sample(self.items, n)
        
        # Create a new dataset
        sampled_dataset = Dataset(
            name=f"{self.name}_sample_{n}",
            description=f"Sample of {n} items from {self.name}",
            items=sampled_items,
            source=f"Sampled from {self.id}",
            tags=self.tags + ["sample"],
        )
        
        return sampled_dataset


def get_dataset_path(dataset_id: str) -> str:
    """Get the file path for a dataset."""
    return os.path.join(DATASETS_DIR, f"{dataset_id}.json")


def list_datasets() -> List[Dataset]:
    """List all available datasets."""
    datasets = []
    for dataset_id in list_json_files(DATASETS_DIR):
        dataset_data = load_json(get_dataset_path(dataset_id))
        if dataset_data:
            datasets.append(Dataset.from_dict(dataset_data))
    return datasets


def get_dataset(dataset_id: str) -> Optional[Dataset]:
    """Get a specific dataset by ID."""
    dataset_data = load_json(get_dataset_path(dataset_id))
    if dataset_data:
        return Dataset.from_dict(dataset_data)
    return None


def create_dataset(
    name: str, 
    items: List[Dict[str, Any]],
    description: Optional[str] = None,
    source: Optional[str] = None,
    tags: Optional[List[str]] = None,
) -> Dataset:
    """
    Create a new dataset.
    
    Args:
        name: The name of the dataset
        items: List of data items (each with 'input' and optional 'expected_output')
        description: Optional description
        source: Optional source information
        tags: Optional list of tags
        
    Returns:
        The created dataset
    """
    # Convert dict items to DataItem objects
    data_items = [DataItem(**item) for item in items]
    
    dataset = Dataset(
        name=name,
        description=description,
        items=data_items,
        source=source,
        tags=tags or [],
    )
    
    save_json(dataset.to_dict(), get_dataset_path(dataset.id))
    return dataset


def update_dataset(
    dataset_id: str,
    name: Optional[str] = None,
    items: Optional[List[Dict[str, Any]]] = None,
    description: Optional[str] = None,
    source: Optional[str] = None,
    tags: Optional[List[str]] = None,
) -> Optional[Dataset]:
    """
    Update an existing dataset.
    
    Args:
        dataset_id: ID of the dataset to update
        name: Optional new name
        items: Optional new list of data items
        description: Optional new description
        source: Optional new source information
        tags: Optional new list of tags
        
    Returns:
        The updated dataset or None if not found
    """
    dataset = get_dataset(dataset_id)
    if not dataset:
        return None
    
    if name is not None:
        dataset.name = name
    if items is not None:
        dataset.items = [DataItem(**item) for item in items]
    if description is not None:
        dataset.description = description
    if source is not None:
        dataset.source = source
    if tags is not None:
        dataset.tags = tags
    
    save_json(dataset.to_dict(), get_dataset_path(dataset.id))
    return dataset


def delete_dataset(dataset_id: str) -> bool:
    """Delete a dataset."""
    path = get_dataset_path(dataset_id)
    if os.path.exists(path):
        os.remove(path)
        return True
    return False


def split_dataset(
    dataset_id: str, 
    test_ratio: float = 0.2, 
    seed: Optional[int] = None
) -> Dict[str, str]:
    """
    Split a dataset into training and testing sets.
    
    Args:
        dataset_id: ID of the dataset to split
        test_ratio: Ratio of test set size to total dataset size
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary with the IDs of the created train and test datasets
    """
    dataset = get_dataset(dataset_id)
    if not dataset:
        return {"error": f"Dataset with ID '{dataset_id}' not found"}
    
    train_dataset, test_dataset = dataset.split(test_ratio, seed)
    
    # Save the new datasets
    save_json(train_dataset.to_dict(), get_dataset_path(train_dataset.id))
    save_json(test_dataset.to_dict(), get_dataset_path(test_dataset.id))
    
    return {
        "train_dataset_id": train_dataset.id,
        "test_dataset_id": test_dataset.id,
    }


def sample_dataset(
    dataset_id: str, 
    n: int, 
    seed: Optional[int] = None
) -> Dict[str, str]:
    """
    Create a new dataset by sampling n items from a dataset.
    
    Args:
        dataset_id: ID of the dataset to sample from
        n: Number of items to sample
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary with the ID of the created sample dataset
    """
    dataset = get_dataset(dataset_id)
    if not dataset:
        return {"error": f"Dataset with ID '{dataset_id}' not found"}
    
    if n <= 0:
        return {"error": "Sample size must be positive"}
    
    if n > len(dataset.items):
        return {"error": f"Sample size {n} exceeds dataset size {len(dataset.items)}"}
    
    sampled_dataset = dataset.sample(n, seed)
    
    # Save the new dataset
    save_json(sampled_dataset.to_dict(), get_dataset_path(sampled_dataset.id))
    
    return {"sample_dataset_id": sampled_dataset.id}


def import_from_jsonl(
    filepath: str,
    name: str,
    input_field: str = "input",
    output_field: Optional[str] = "output",
    description: Optional[str] = None,
    source: Optional[str] = None,
    tags: Optional[List[str]] = None,
) -> Dict[str, str]:
    """
    Import a dataset from a JSONL file.
    
    Args:
        filepath: Path to the JSONL file
        name: Name for the new dataset
        input_field: Field name for the input in each JSON line
        output_field: Field name for the expected output (if any)
        description: Optional description
        source: Optional source information
        tags: Optional list of tags
        
    Returns:
        Dictionary with the ID of the created dataset
    """
    if not os.path.exists(filepath):
        return {"error": f"File not found: {filepath}"}
    
    try:
        items = []
        with open(filepath, 'r') as f:
            for line in f:
                if line.strip():  # Skip empty lines
                    data = json.loads(line)
                    item = {"input": data.get(input_field, "")}
                    
                    if output_field and output_field in data:
                        item["expected_output"] = data[output_field]
                    
                    # Include any other fields as metadata
                    metadata = {
                        k: v for k, v in data.items() 
                        if k != input_field and k != output_field
                    }
                    if metadata:
                        item["metadata"] = metadata
                    
                    items.append(item)
        
        if not items:
            return {"error": "No valid items found in the file"}
        
        source_info = source or f"Imported from {os.path.basename(filepath)}"
        dataset = create_dataset(
            name=name,
            items=items,
            description=description,
            source=source_info,
            tags=tags or ["imported"]
        )
        
        return {"dataset_id": dataset.id}
    
    except json.JSONDecodeError:
        return {"error": "Invalid JSONL format"}
    except Exception as e:
        return {"error": f"Error importing file: {str(e)}"}


def manage_dataset(
    action: str,
    dataset_id: Optional[str] = None,
    name: Optional[str] = None,
    items: Optional[List[Dict[str, Any]]] = None,
    description: Optional[str] = None,
    source: Optional[str] = None,
    tags: Optional[List[str]] = None,
    filepath: Optional[str] = None,
    input_field: Optional[str] = None,
    output_field: Optional[str] = None,
    test_ratio: Optional[float] = None,
    sample_size: Optional[int] = None,
    seed: Optional[int] = None,
) -> str:
    """
    Manage datasets for experiments and evaluations.
    
    Args:
        action: One of "create", "list", "get", "update", "delete", "split", "sample", "import"
        dataset_id: Required for get, update, delete, split, sample
        name: Required for create and import
        items: Required for create, optional for update
        description: Optional description
        source: Optional source information
        tags: Optional list of tags
        filepath: Required for import
        input_field: Field name for input when importing
        output_field: Field name for expected output when importing
        test_ratio: Ratio for splitting dataset (default: 0.2)
        sample_size: Number of items to sample
        seed: Random seed for reproducibility
    
    Returns:
        A string response describing the result
    """
    valid_actions = ["create", "list", "get", "update", "delete", "split", "sample", "import"]
    
    try:
        validate_action(action, valid_actions)
        
        # Handle each action
        if action == "list":
            datasets = list_datasets()
            return format_list([d.to_dict() for d in datasets])
        
        elif action == "get":
            validate_required_params({"dataset_id": dataset_id}, ["dataset_id"])
            dataset = get_dataset(dataset_id)
            if not dataset:
                return format_error(f"Dataset with ID '{dataset_id}' not found")
            
            dataset_dict = dataset.to_dict()
            item_count = len(dataset_dict["items"])
            
            result = [
                f"Dataset: {dataset_dict['name']} (ID: {dataset_dict['id']})",
                f"Description: {dataset_dict.get('description', 'None')}",
                f"Source: {dataset_dict.get('source', 'None')}",
                f"Tags: {', '.join(dataset_dict.get('tags', []))}",
                f"Items: {item_count}",
            ]
            
            # Show sample items (up to 5)
            sample_count = min(5, item_count)
            if sample_count > 0:
                result.append("\nSample items:")
                for i, item in enumerate(dataset_dict["items"][:sample_count]):
                    result.append(f"\nItem {i+1}:")
                    result.append(f"Input: {item['input']}")
                    if item.get("expected_output"):
                        result.append(f"Expected Output: {item['expected_output']}")
                
                if item_count > sample_count:
                    result.append(f"\n... and {item_count - sample_count} more items")
            
            return "\n".join(result)
        
        elif action == "create":
            validate_required_params(
                {"name": name, "items": items},
                ["name", "items"],
            )
            
            if not isinstance(items, list) or not items:
                return format_error("Items must be a non-empty list")
            
            for i, item in enumerate(items):
                if "input" not in item:
                    return format_error(f"Item at index {i} is missing required 'input' field")
            
            dataset = create_dataset(name, items, description, source, tags)
            return format_success(
                f"Dataset '{name}' created with ID: {dataset.id} ({len(dataset.items)} items)"
            )
        
        elif action == "update":
            validate_required_params({"dataset_id": dataset_id}, ["dataset_id"])
            
            if items is not None:
                if not isinstance(items, list) or not items:
                    return format_error("Items must be a non-empty list")
                
                for i, item in enumerate(items):
                    if "input" not in item:
                        return format_error(f"Item at index {i} is missing required 'input' field")
            
            dataset = update_dataset(
                dataset_id, name, items, description, source, tags
            )
            if not dataset:
                return format_error(f"Dataset with ID '{dataset_id}' not found")
            
            return format_success(f"Dataset '{dataset.name}' updated")
        
        elif action == "delete":
            validate_required_params({"dataset_id": dataset_id}, ["dataset_id"])
            
            if delete_dataset(dataset_id):
                return format_success(f"Dataset with ID '{dataset_id}' deleted")
            else:
                return format_error(f"Dataset with ID '{dataset_id}' not found")
        
        elif action == "split":
            validate_required_params({"dataset_id": dataset_id}, ["dataset_id"])
            
            ratio = test_ratio if test_ratio is not None else 0.2
            if ratio <= 0 or ratio >= 1:
                return format_error("Test ratio must be between 0 and 1")
            
            result = split_dataset(dataset_id, ratio, seed)
            if "error" in result:
                return format_error(result["error"])
            
            return format_success(
                f"Dataset split successfully. Train dataset ID: {result['train_dataset_id']}, "
                f"Test dataset ID: {result['test_dataset_id']}"
            )
        
        elif action == "sample":
            validate_required_params(
                {"dataset_id": dataset_id, "sample_size": sample_size},
                ["dataset_id", "sample_size"]
            )
            
            if sample_size <= 0:
                return format_error("Sample size must be positive")
            
            result = sample_dataset(dataset_id, sample_size, seed)
            if "error" in result:
                return format_error(result["error"])
            
            return format_success(
                f"Dataset sampled successfully. Sample dataset ID: {result['sample_dataset_id']}"
            )
        
        elif action == "import":
            validate_required_params(
                {"name": name, "filepath": filepath},
                ["name", "filepath"]
            )
            
            input_field_name = input_field or "input"
            
            result = import_from_jsonl(
                filepath=filepath,
                name=name,
                input_field=input_field_name,
                output_field=output_field,
                description=description,
                source=source,
                tags=tags,
            )
            
            if "error" in result:
                return format_error(result["error"])
            
            return format_success(
                f"Dataset '{name}' imported with ID: {result['dataset_id']}"
            )
    
    except ValidationError as e:
        return format_error(str(e))
    except Exception as e:
        return format_error(f"Unexpected error: {str(e)}")