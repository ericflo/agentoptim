#!/bin/bash
# Script to run tests for the AgentOptim project

set -e  # Exit on error

# Define colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print header
echo -e "${BLUE}=== AgentOptim Test Runner ===${NC}"
echo "Running tests at $(date)"
echo ""

# Check for arguments
RUN_TYPE=${1:-"all"}
VERBOSE=${2:-""}

if [ "$VERBOSE" == "-v" ] || [ "$VERBOSE" == "--verbose" ]; then
    VERBOSE="-v"
else
    VERBOSE=""
fi

# Function to run tests with proper output
run_tests() {
    TEST_TYPE=$1
    MARKER=$2
    EXTRA_ARGS=$3
    
    echo -e "${YELLOW}Running $TEST_TYPE tests...${NC}"
    
    if [ -n "$MARKER" ]; then
        python -m pytest -m "$MARKER" $VERBOSE $EXTRA_ARGS
    else
        python -m pytest $VERBOSE $EXTRA_ARGS
    fi
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ $TEST_TYPE tests passed!${NC}"
    else
        echo -e "${RED}❌ $TEST_TYPE tests failed!${NC}"
        exit 1
    fi
    echo ""
}

# Check if we're in a virtual environment
if [ -z "$VIRTUAL_ENV" ]; then
    echo -e "${YELLOW}Warning: Not running in a virtual environment. It's recommended to activate one first.${NC}"
    echo "You can create one with: python -m venv venv && source venv/bin/activate"
    echo ""
fi

# Activate the correct environment if running in CI
if [ -n "$CI" ] && [ -d "venv" ]; then
    echo "Running in CI environment, activating virtual environment..."
    source venv/bin/activate
fi

# Make sure dependencies are installed
if ! command -v pytest &> /dev/null; then
    echo -e "${YELLOW}Installing test dependencies...${NC}"
    pip install -r requirements.txt
    pip install pytest pytest-cov pytest-asyncio
    echo ""
fi

# Run different test suites based on argument
case $RUN_TYPE in
    "unit")
        run_tests "Unit" "unit" ""
        ;;
    "integration")
        run_tests "Integration" "integration" ""
        ;;
    "evalset")
        run_tests "EvalSet" "evalset" ""
        ;;
    "all")
        # Run unit tests first, then integration tests
        run_tests "Unit" "unit" ""
        run_tests "Integration" "integration" ""
        run_tests "EvalSet" "evalset" ""
        echo -e "${GREEN}All tests passed!${NC}"
        ;;
    "coverage")
        echo -e "${YELLOW}Running tests with coverage report...${NC}"
        python -m pytest --cov=agentoptim --cov-report=term-missing --cov-report=html
        echo -e "${GREEN}Coverage report generated!${NC}"
        echo "HTML report available in htmlcov/index.html"
        ;;
    *)
        echo -e "${RED}Unknown test type: $RUN_TYPE${NC}"
        echo "Available options: unit, integration, evalset, all, coverage"
        exit 1
        ;;
esac

echo -e "${BLUE}=== Test run completed! ===${NC}"