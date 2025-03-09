#!/usr/bin/env python
"""
Script to update documentation files to replace run_evalset_tool with manage_eval_runs_tool.

This script:
1. Finds all Markdown files in the docs directory that use run_evalset_tool
2. Replaces run_evalset_tool with manage_eval_runs_tool
3. Updates function calls to include action="run" where appropriate
4. Preserves formatting and indentation
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
    
    # Check if the file actually contains run_evalset_tool
    if 'run_evalset_tool' not in content:
        print(f"Skipping {file_path} - does not contain run_evalset_tool")
        return False
    
    # Replace all occurrences of run_evalset_tool with manage_eval_runs_tool
    updated_content = content.replace('run_evalset_tool', 'manage_eval_runs_tool')
    
    # Update function calls in code blocks
    # Pattern matches code blocks with run_evalset_tool(
    updated_content = re.sub(
        r'(```.*?\n.*?)manage_eval_runs_tool\((.*?)\)',
        lambda match: handle_code_block(match),
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
    
    return True

def handle_code_block(match):
    """
    Handle updating function calls in code blocks to add action="run"
    """
    code_block = match.group(0)
    
    # If this already has action= parameter, don't modify it
    if 'action=' in code_block:
        return code_block
    
    # Insert action="run" parameter if this appears to be a function call
    return code_block.replace('manage_eval_runs_tool(', 'manage_eval_runs_tool(action="run", ')

def main():
    # Parse command line arguments
    dry_run = '--dry-run' in sys.argv
    
    # Get the base directory of the project
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)
    docs_dir = os.path.join(base_dir, 'docs')
    
    # Find all Markdown files
    md_files = glob.glob(os.path.join(docs_dir, '**', '*.md'), recursive=True)
    
    # Also include the README and other top-level md files
    md_files.extend(glob.glob(os.path.join(base_dir, '*.md')))
    
    # Variables to track updates
    total_files = len(md_files)
    updated_files = 0
    
    print(f"Found {total_files} Markdown files to check")
    print(f"Dry run: {dry_run}")
    
    # Process each file
    for file_path in md_files:
        if update_file(file_path, dry_run=dry_run):
            updated_files += 1
    
    print(f"\nUpdate complete: {updated_files} out of {total_files} files {'would be' if dry_run else 'were'} updated")
    
    if dry_run and updated_files > 0:
        print("\nThis was a dry run. To apply changes, run without the --dry-run flag")

if __name__ == "__main__":
    main()