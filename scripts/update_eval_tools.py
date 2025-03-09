#!/usr/bin/env python
"""
Script to automate updating all example files to replace run_evalset_tool with manage_eval_runs_tool.

This script:
1. Finds all Python files in the examples directory that use run_evalset_tool
2. Updates import statements
3. Replaces function calls with the new format
4. Maintains proper indentation
5. Saves changes to the files
"""

import os
import re
import sys
import glob

def update_file(file_path, dry_run=False):
    """
    Update a single file with the required changes.
    
    Args:
        file_path: Path to the file to update
        dry_run: If True, don't write changes to file
        
    Returns:
        bool: True if file was updated (or would be in dry run), False otherwise
    """
    # Read the file content
    with open(file_path, 'r') as file:
        content = file.read()
    
    # Check if the file actually uses run_evalset_tool
    if 'run_evalset_tool' not in content:
        print(f"Skipping {file_path} - does not use run_evalset_tool")
        return False
    
    # Update the import statement
    updated_content = re.sub(
        r'from agentoptim(?:\.server)? import (.*?)run_evalset_tool(.*?)',
        r'from agentoptim.server import \1manage_eval_runs_tool\2',
        content
    )
    
    # Also handle 'import agentoptim.server' style imports
    updated_content = re.sub(
        r'from agentoptim import run_evalset_tool',
        r'from agentoptim.server import manage_eval_runs_tool',
        updated_content
    )
    
    # Replace function calls for single-line calls
    updated_content = re.sub(
        r'([ \t]*)await run_evalset_tool\(([^)]*?)(\))',
        r'\1await manage_eval_runs_tool(action="run", \2\3',
        updated_content
    )
    
    # Handle multi-line function calls
    updated_content = re.sub(
        r'([ \t]*)await run_evalset_tool\((.*?\n(?:.*?\n)*?.*?\))',
        lambda match: re.sub(
            r'([ \t]*)await run_evalset_tool\(', 
            r'\1await manage_eval_runs_tool(action="run", ', 
            match.group(0)
        ),
        updated_content,
        flags=re.DOTALL
    )
    
    # Check if there are any changes
    if content == updated_content:
        print(f"No changes needed for {file_path}")
        return False
    
    # Write the updated content back to the file
    if not dry_run:
        with open(file_path, 'w') as file:
            file.write(updated_content)
        print(f"Updated {file_path}")
    else:
        print(f"Would update {file_path} (dry run)")
        
        # Print a diff-like output to show changes
        if dry_run:
            print("\nChanges that would be made:")
            for i, (old_line, new_line) in enumerate(zip(content.splitlines(), updated_content.splitlines())):
                if old_line != new_line:
                    print(f"Line {i+1}:")
                    print(f"- {old_line}")
                    print(f"+ {new_line}")
                    print()
    
    return True

def main():
    # Parse command line arguments
    dry_run = '--dry-run' in sys.argv
    test_single = '--test' in sys.argv
    
    # Get the base directory of the project
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)
    examples_dir = os.path.join(base_dir, 'examples')
    
    if test_single:
        # Test on evalset_example.py
        test_file = os.path.join(examples_dir, 'evalset_example.py')
        print(f"Testing on file: {test_file}")
        update_file(test_file, dry_run=True)
        return
    
    # Find all Python files in the examples directory
    python_files = glob.glob(os.path.join(examples_dir, '**', '*.py'), recursive=True)
    
    # Variables to track updates
    total_files = len(python_files)
    updated_files = 0
    
    print(f"Found {total_files} Python files in the examples directory")
    print(f"Dry run: {dry_run}")
    
    # Process each file
    for file_path in python_files:
        if update_file(file_path, dry_run=dry_run):
            updated_files += 1
    
    print(f"\nUpdate complete: {updated_files} out of {total_files} files {'would be' if dry_run else 'were'} updated")
    
    if dry_run and updated_files > 0:
        print("\nThis was a dry run. To apply changes, run without the --dry-run flag")

if __name__ == "__main__":
    main()