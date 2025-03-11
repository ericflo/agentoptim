#!/bin/bash
# Shell script to run a system message optimization interactively

# Ensure virtual environment is activated
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
    source venv/bin/activate
    pip install -e .
else
    source venv/bin/activate
fi

# Start server in background
echo "Starting AgentOptim server..."
python -m agentoptim.server &
SERVER_PID=$!

# Give server time to start
echo "Waiting for server to start..."
sleep 2

# Check if server is running
if ! ps -p $SERVER_PID > /dev/null; then
    echo "Failed to start server"
    exit 1
fi

# Clean up function to kill server when script exits
cleanup() {
    echo "Stopping server..."
    kill $SERVER_PID
    exit 0
}

# Trap ctrl-c and call cleanup
trap cleanup INT

# Function to create a new EvalSet
create_evalset() {
    python -c "
import asyncio
from agentoptim import manage_evalset_tool

async def create():
    evalset = await manage_evalset_tool(
        action='create',
        name='System Message Optimization',
        questions=[
            'Does the response directly address the user query?',
            'Is the response clear and easy to understand?',
            'Is the response comprehensive in addressing all aspects of the query?',
            'Is the response concise without unnecessary information?',
            'Is the tone of the response appropriate for the query?',
            'Does the response provide accurate information?',
            'Does the response provide concrete examples or actionable steps when appropriate?',
            'Is the response well-structured with a logical flow?',
            'Does the response avoid introducing biases or opinions not warranted by the query?',
            'Would the response likely satisfy the user\\'s information needs?'
        ],
        short_description='Optimization criteria for system messages',
        long_description='This EvalSet is designed to evaluate system messages by judging the quality of resulting responses. It measures clarity, completeness, helpfulness, accuracy, and appropriateness of the responses generated using the system message. Use it to optimize system messages for different user queries.'
    )
    return evalset['evalset']['id']

evalset_id = asyncio.run(create())
print(evalset_id)
"
}

# Function to optimize system messages
optimize_system_messages() {
    local user_query="$1"
    local evalset_id="$2"
    local num_candidates="${3:-5}"
    
    python -c "
import asyncio
from agentoptim import optimize_system_messages_tool

async def optimize():
    result = await optimize_system_messages_tool(
        action='optimize',
        user_message='$user_query',
        evalset_id='$evalset_id',
        num_candidates=$num_candidates,
        diversity_level='high'
    )
    return result

result = asyncio.run(optimize())
print('Optimization run ID:', result['id'])
print('\nBest System Message (Score: {}%):\n'.format(round(result['best_score'], 1)))
print('-' * 80)
print(result['best_system_message'])
print('-' * 80)
"
}

# Main interactive loop
echo "=== System Message Optimizer ==="
echo "This tool helps you optimize system messages for any user query."

# Create or use existing EvalSet
echo "Do you want to create a new EvalSet or use an existing one? (new/existing)"
read -r choice

if [ "$choice" == "new" ]; then
    echo "Creating new EvalSet..."
    evalset_id=$(create_evalset)
    echo "Created EvalSet with ID: $evalset_id"
else
    echo "Enter existing EvalSet ID:"
    read -r evalset_id
fi

while true; do
    echo -e "\nEnter user query (or 'exit' to quit):"
    read -r user_query
    
    if [ "$user_query" == "exit" ]; then
        break
    fi
    
    echo "How many candidate system messages to generate? (default: 5)"
    read -r num_candidates
    
    if [ -z "$num_candidates" ]; then
        num_candidates=5
    fi
    
    echo "Optimizing system messages for: '$user_query'..."
    optimize_system_messages "$user_query" "$evalset_id" "$num_candidates"
    
    echo -e "\nPress Enter to continue or type 'exit' to quit:"
    read -r continue_choice
    
    if [ "$continue_choice" == "exit" ]; then
        break
    fi
done

# Cleanup
cleanup