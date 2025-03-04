#!/usr/bin/env python
"""
Documentation generator for AgentOptim.

This script generates HTML documentation for the AgentOptim package
using pdoc. It creates documentation pages for all modules and classes
in the package with cross-references and type annotations.
"""

import os
import sys
import subprocess
import argparse
import shutil
from pathlib import Path

# Add the project root to the Python path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

# Import AgentOptim to ensure all modules are importable
import agentoptim


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Generate documentation for AgentOptim")
    
    parser.add_argument(
        "--output-dir",
        default=os.path.join(ROOT_DIR, "docs"),
        help="Directory where documentation will be generated (default: ./docs)",
    )
    
    parser.add_argument(
        "--format",
        choices=["html", "markdown"],
        default="html",
        help="Output format for documentation (default: html)",
    )
    
    parser.add_argument(
        "--serve", 
        action="store_true", 
        help="Start a local server to view documentation"
    )
    
    parser.add_argument(
        "--port", 
        type=int, 
        default=8080, 
        help="Port for documentation server (default: 8080)"
    )
    
    parser.add_argument(
        "--clean", 
        action="store_true", 
        help="Clean output directory before generating documentation"
    )
    
    return parser.parse_args()


def main():
    """Generate documentation for AgentOptim."""
    args = parse_args()
    
    # Ensure output directory exists
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Clean output directory if requested
    if args.clean and os.path.exists(output_dir):
        print(f"Cleaning output directory: {output_dir}")
        for item in os.listdir(output_dir):
            item_path = os.path.join(output_dir, item)
            if os.path.isdir(item_path):
                shutil.rmtree(item_path)
            else:
                os.remove(item_path)
    
    # Build pdoc command
    cmd = [
        "pdoc",
        "agentoptim",
        "--docformat",
        "google",
    ]
    
    if args.format == "html":
        cmd.extend(["--output-dir", output_dir, "--html"])
    else:
        cmd.extend(["--output-dir", output_dir, "--output-format", "markdown"])
    
    # Generate documentation
    print(f"Generating {args.format} documentation in {output_dir}")
    result = subprocess.run(cmd, cwd=ROOT_DIR)
    
    if result.returncode != 0:
        print("Error generating documentation")
        sys.exit(1)
        
    print(f"Documentation generated successfully in {output_dir}")
    
    # Serve documentation if requested
    if args.serve:
        print(f"Starting documentation server at http://localhost:{args.port}")
        serve_cmd = [
            "python",
            "-m",
            "http.server",
            str(args.port),
            "-d",
            output_dir,
        ]
        subprocess.run(serve_cmd)


if __name__ == "__main__":
    main()