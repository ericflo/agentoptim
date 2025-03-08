#!/bin/bash
# Installation script for AgentOptim

set -e  # Exit on error

# Define colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== AgentOptim Installation ===${NC}"
echo "Starting installation at $(date)"
echo ""

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2)
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 10 ]); then
    echo -e "${RED}Error: Python 3.10 or higher is required (found $PYTHON_VERSION)${NC}"
    echo "Please upgrade your Python version and try again."
    exit 1
fi

echo -e "${GREEN}✅ Python version $PYTHON_VERSION detected${NC}"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    python3 -m venv venv
    echo -e "${GREEN}✅ Virtual environment created${NC}"
else
    echo -e "${GREEN}✅ Virtual environment already exists${NC}"
fi

# Activate virtual environment
echo -e "${YELLOW}Activating virtual environment...${NC}"
source venv/bin/activate
echo -e "${GREEN}✅ Virtual environment activated${NC}"

# Upgrade pip
echo -e "${YELLOW}Upgrading pip...${NC}"
pip install --upgrade pip
echo -e "${GREEN}✅ Pip upgraded${NC}"

# Install requirements
echo -e "${YELLOW}Installing dependencies...${NC}"
pip install -r requirements.txt
echo -e "${GREEN}✅ Dependencies installed${NC}"

# Install package in development mode
echo -e "${YELLOW}Installing AgentOptim in development mode...${NC}"
pip install -e .
echo -e "${GREEN}✅ AgentOptim installed${NC}"

echo ""
echo -e "${BLUE}=== Installation Complete! ===${NC}"
echo ""
echo -e "To use AgentOptim, activate the virtual environment with:"
echo -e "${YELLOW}source venv/bin/activate${NC}"
echo ""
echo -e "To start the MCP server:"
echo -e "${YELLOW}python -m agentoptim.server${NC}"
echo ""
echo -e "For more information, see the documentation in the docs/ directory."
echo -e "Happy optimizing! 🚀"