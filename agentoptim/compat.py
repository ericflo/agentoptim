"""Compatibility layer for transitioning from old API to new EvalSet architecture."""

import logging
import json
from typing import Any, Dict, List, Optional, Union

from agentoptim.evalset import (
    manage_evalset,
    EvalSet,
    get_evalset,
    list_evalsets
)

from agentoptim.runner import run_evalset

logger = logging.getLogger(__name__)


async def convert_evaluation_to_evalset(
    name: str,
    template: str,
    questions: List[str],
    description: Optional[str] = None
) -> Dict[str, Any]:
    """
    Convert an old-style evaluation to a new EvalSet format.
    
    Args:
        name: The name of the evaluation
        template: The template string
        questions: List of evaluation questions
        description: Optional description
        
    Returns:
        The created EvalSet in dictionary format
    """
    # We need to adapt the old template format to the new one
    # Old: typically used {input}, {response}, {question}
    # New: uses {{ conversation }} and {{ eval_question }}
    
    # First, try to detect what variables are used in the template
    has_input = "{input}" in template
    has_response = "{response}" in template
    has_question = "{question}" in template
    
    # Create a new template that uses the new format
    new_template = template
    
    # If the template uses both input and response, we need to adapt it
    if has_input and has_response and has_question:
        # Replace {question} with {{ eval_question }}
        new_template = new_template.replace("{question}", "{{ eval_question }}")
        
        # Create a wrapper for {{ conversation }} that formats it appropriately
        new_template = f"""
        Given this conversation:
        {{{{ conversation }}}}
        
        Please evaluate the final response using this question:
        {{{{ eval_question }}}}
        
        Respond with JSON: {{"judgment": 1}} for yes or {{"judgment": 0}} for no.
        """
    
    elif has_question:
        # Simple case - just replace {question} with {{ eval_question }}
        new_template = new_template.replace("{question}", "{{ eval_question }}")
        
        # Add {{ conversation }} placeholder if not present
        if "{{ conversation }}" not in new_template:
            new_template = f"""
            Given this conversation:
            {{{{ conversation }}}}
            
            {new_template}
            """
    
    else:
        # If we can't determine how to adapt it, create a basic template
        new_template = f"""
        Given this conversation:
        {{{{ conversation }}}}
        
        Please answer the following yes/no question about the final assistant response:
        {{{{ eval_question }}}}
        
        Return a JSON object with the following format:
        {{"judgment": 1}} for yes or {{"judgment": 0}} for no.
        """
    
    # Create the EvalSet
    result = manage_evalset(
        action="create",
        name=name,
        template=new_template,
        questions=questions,
        description=description
    )
    
    return result


async def evaluation_to_evalset_id(evaluation_id: str) -> Optional[str]:
    """
    Find an EvalSet corresponding to an old evaluation ID.
    If not found, create a compatibility EvalSet.
    
    Args:
        evaluation_id: Old evaluation ID
        
    Returns:
        New EvalSet ID or None if conversion failed
    """
    # For now, let's implement a simple mapping 
    # In a future version, we can actually convert the evaluation content
    
    # Check if we already have a compatible EvalSet for this evaluation
    evalsets = list_evalsets()
    for evalset in evalsets:
        # Look for EvalSets that have a description mentioning the evaluation ID
        if evalset.description and f"Converted from evaluation {evaluation_id}" in evalset.description:
            return evalset.id
    
    # If we don't have a corresponding EvalSet, we would need to create one
    # This would require accessing the old evaluation data and converting it
    # For now, return None (future implementation needed)
    return None


async def dataset_to_conversations(dataset_id: str, items: Optional[List[Dict[str, Any]]] = None) -> List[List[Dict[str, str]]]:
    """
    Convert a dataset to a list of conversations that can be used with run_evalset.
    
    Args:
        dataset_id: Dataset ID
        items: Optional list of dataset items to use instead of loading from storage
        
    Returns:
        List of conversations formatted for run_evalset
    """
    # For a real implementation, we would need to load dataset items
    # and convert each item to a conversation
    
    # This is a placeholder implementation
    if items is None:
        # In a real implementation, load items from storage
        return []
    
    conversations = []
    
    for item in items:
        # Try to extract input and expected_output
        input_text = item.get("input", "")
        output_text = item.get("expected_output", "")
        
        # Create a simple conversation
        conversation = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": input_text}
        ]
        
        # Add assistant response if we have expected_output
        if output_text:
            conversation.append({"role": "assistant", "content": output_text})
        
        conversations.append(conversation)
    
    return conversations


async def experiment_results_to_evalset_results(
    experiment_id: str,
    evaluation_id: str,
    evalset_id: str,
    job_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Convert experiment results to EvalSet results format.
    
    Args:
        experiment_id: ID of the experiment
        evaluation_id: ID of the evaluation
        evalset_id: ID of the EvalSet
        job_id: Optional job ID to filter results
        
    Returns:
        Results in the EvalSet format
    """
    # This is a placeholder implementation
    # In a real implementation, we would load experiment results and convert them
    
    return {
        "status": "success",
        "message": f"Converted results from experiment {experiment_id}",
        "evalset_id": evalset_id,
        "evalset_name": "Converted EvalSet",
        "model": "unknown",
        "summary": {
            "total_questions": 0,
            "successful_evaluations": 0,
            "yes_count": 0,
            "no_count": 0,
            "error_count": 0,
            "yes_percentage": 0
        },
        "results": []
    }