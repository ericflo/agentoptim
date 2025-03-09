# AgentOptim v2.1.0 CLI Usage Examples

This guide demonstrates how to use the AgentOptim v2.1.0 command-line interface for common evaluation and optimization tasks. The CLI makes it easy to create, manage, and execute evaluations without writing any code.

**Note about v2.1.0**: The CLI and API in AgentOptim v2.1.0 includes several important changes from previous versions:

1. Templates are now system-defined (the `--template` parameter has been removed)
2. All EvalSets require `--short-desc` and `--long-desc` parameters when creating
3. The compatibility layer has been removed
4. Error handling has been improved with more detailed error messages
5. Caching has been enhanced for better performance

## Getting Started

First, ensure you have AgentOptim installed:

```bash
pip install agentoptim
```

Run `agentoptim --help` to see all available commands:

```bash
agentoptim --help
```

## Basic Operations

### Starting the MCP Server

```bash
# Start the server with default settings
agentoptim server

# Start with a specific port and debug mode
agentoptim server --port 5000 --debug
```

### Listing Evaluation Sets

```bash
# List all available evaluation sets in table format (default)
agentoptim list

# List in JSON format
agentoptim list --format json

# List in YAML format
agentoptim list --format yaml

# Save the list to a file
agentoptim list --output evalsets.json --format json
```

### Viewing Evaluation Set Details

```bash
# View detailed information about a specific evaluation set
agentoptim get 6f8d9e2a-5b4c-4a3f-8d1e-7f9a6b5c4d3e

# Get details in JSON format
agentoptim get 6f8d9e2a-5b4c-4a3f-8d1e-7f9a6b5c4d3e --format json

# Save to a file
agentoptim get 6f8d9e2a-5b4c-4a3f-8d1e-7f9a6b5c4d3e --output evalset_details.json --format json
```

## Creating and Managing Evaluation Sets

### Creating an Evaluation Set from a File

Create a file named `questions.txt` with evaluation criteria:

```
Is the response helpful for the user's needs?
Does the response directly address the user's question?
Is the response clear and easy to understand?
Is the response accurate?
Does the response provide complete information?
```

Then create the evaluation set:

```bash
agentoptim create \
  --name "Response Quality Evaluation" \
  --questions questions.txt \
  --short-desc "Basic response quality evaluation" \
  --long-desc "This evaluation set measures key aspects of response quality including helpfulness, clarity, accuracy, and completeness. Use it to evaluate the quality of AI responses to general queries."
```

### Creating an Evaluation Set from the Command Line

```bash
agentoptim create \
  --name "Support Response Quality" \
  --questions "Is the response professional?,Does the response solve the user's problem?,Is the tone empathetic?" \
  --short-desc "Support response evaluation" \
  --long-desc "This evaluation set focuses on support-specific quality criteria, emphasizing problem resolution, professionalism, and empathy."
```

### Updating an Evaluation Set

```bash
# Update the name of an evaluation set
agentoptim update 6f8d9e2a-5b4c-4a3f-8d1e-7f9a6b5c4d3e --name "Enhanced Response Quality Evaluation"

# Update questions from a file
agentoptim update 6f8d9e2a-5b4c-4a3f-8d1e-7f9a6b5c4d3e --questions new_questions.txt

# Update the short description
agentoptim update 6f8d9e2a-5b4c-4a3f-8d1e-7f9a6b5c4d3e --short-desc "Improved quality criteria"
```

### Deleting an Evaluation Set

```bash
agentoptim delete 6f8d9e2a-5b4c-4a3f-8d1e-7f9a6b5c4d3e
```

## Evaluating Conversations

### Evaluating a JSON Conversation File

Create a file named `conversation.json` with this content:

```json
[
  {
    "role": "system", 
    "content": "You are a helpful assistant."
  },
  {
    "role": "user", 
    "content": "How do I reset my password?"
  },
  {
    "role": "assistant", 
    "content": "To reset your password, please follow these steps:\n\n1. Go to the login page\n2. Click on the 'Forgot Password' link below the login form\n3. Enter the email address associated with your account\n4. Check your email for a password reset link\n5. Click the link and follow the instructions to create a new password\n\nIf you don't receive the email within a few minutes, please check your spam folder."
  }
]
```

Then evaluate it:

```bash
# Basic evaluation
agentoptim eval 6f8d9e2a-5b4c-4a3f-8d1e-7f9a6b5c4d3e conversation.json

# Evaluate with a specific judge model (using environment variable)
AGENTOPTIM_JUDGE_MODEL=gpt-4o-mini agentoptim eval 6f8d9e2a-5b4c-4a3f-8d1e-7f9a6b5c4d3e conversation.json

# Evaluate with more parallel processes (faster, but uses more resources)
agentoptim eval 6f8d9e2a-5b4c-4a3f-8d1e-7f9a6b5c4d3e conversation.json --parallel 5

# Skip reasoning in results (faster evaluation)
AGENTOPTIM_OMIT_REASONING=1 agentoptim eval 6f8d9e2a-5b4c-4a3f-8d1e-7f9a6b5c4d3e conversation.json
```

### Evaluating a Simple Text Response

Create a file named `response.txt` with a simple assistant response:

```
To reset your password, go to the login page and click on "Forgot Password". Follow the instructions sent to your email.
```

Then evaluate it as a standalone response:

```bash
# This creates a minimal conversation with just one user message and this response
agentoptim eval 6f8d9e2a-5b4c-4a3f-8d1e-7f9a6b5c4d3e --text response.txt
```

### Output Formats

```bash
# Default text format
agentoptim eval 6f8d9e2a-5b4c-4a3f-8d1e-7f9a6b5c4d3e conversation.json

# JSON output
agentoptim eval 6f8d9e2a-5b4c-4a3f-8d1e-7f9a6b5c4d3e conversation.json --format json

# YAML output
agentoptim eval 6f8d9e2a-5b4c-4a3f-8d1e-7f9a6b5c4d3e conversation.json --format yaml

# CSV output (great for spreadsheets and data analysis)
agentoptim eval 6f8d9e2a-5b4c-4a3f-8d1e-7f9a6b5c4d3e conversation.json --format csv --output results.csv
```

## Advanced Operations

### Viewing Cache Statistics

```bash
# View cache performance statistics
agentoptim stats

# Get statistics in JSON format
agentoptim stats --format json
```

## Practical Use Cases

### Creating an Evaluation Set for Technical Support

```bash
# Create a file with technical support evaluation criteria
cat > tech_support_questions.txt << EOF
Does the response directly address the user's specific technical issue?
Is the response technically accurate?
Does the response provide step-by-step instructions that are easy to follow?
Does the response explain the underlying technical concept if relevant?
Does the response suggest troubleshooting steps if the initial solution doesn't work?
Is the tone of the response professional and helpful?
EOF

# Create the evaluation set
agentoptim create \
  --name "Technical Support Quality" \
  --questions tech_support_questions.txt \
  --short-desc "Technical support evaluation criteria" \
  --long-desc "This evaluation set provides comprehensive evaluation criteria for technical support responses. It measures clarity, accuracy, completeness, and helpfulness of technical instructions. Use it to evaluate support responses to technical questions or troubleshooting scenarios."
```

### Comparing Multiple Responses

```bash
# Create different response files
cat > response1.txt << EOF
Go to Settings > Account > Reset Password.
EOF

cat > response2.txt << EOF
To reset your password, go to the login page and click 'Forgot Password'.
EOF

cat > response3.txt << EOF
To reset your password, please follow these steps:
1. Go to the login page
2. Click on the 'Forgot Password' link
3. Enter your email address
4. Check your email for the reset link
5. Click the link and create a new password
EOF

# Evaluate each response and compare
agentoptim eval 6f8d9e2a-5b4c-4a3f-8d1e-7f9a6b5c4d3e --text response1.txt > response1_results.txt
agentoptim eval 6f8d9e2a-5b4c-4a3f-8d1e-7f9a6b5c4d3e --text response2.txt > response2_results.txt
agentoptim eval 6f8d9e2a-5b4c-4a3f-8d1e-7f9a6b5c4d3e --text response3.txt > response3_results.txt

# Compare the results
echo "Response 1 score:" && grep "yes_percentage" response1_results.txt
echo "Response 2 score:" && grep "yes_percentage" response2_results.txt
echo "Response 3 score:" && grep "yes_percentage" response3_results.txt
```

### Batch Processing Multiple Conversations

```bash
# Create a simple batch processing script
cat > batch_eval.sh << EOF
#!/bin/bash
# Process all JSON conversation files in the current directory

mkdir -p results

for file in *.json; do
  if [[ "\$file" != "results"* ]]; then
    echo "Evaluating \$file..."
    agentoptim eval 6f8d9e2a-5b4c-4a3f-8d1e-7f9a6b5c4d3e "\$file" --format json > "results/\${file%.json}_result.json"
  fi
done

echo "All evaluations complete. Results saved to the 'results' directory."
EOF

chmod +x batch_eval.sh
./batch_eval.sh
```

## Integrating with Other Tools

### Combining with jq for Analysis

```bash
# Extract just the yes percentage from all results
for file in results/*.json; do
  echo -n "$file: "
  jq '.summary.yes_percentage' "$file"
done

# Find all questions that got "No" answers
for file in results/*.json; do
  echo "File: $file"
  jq '.results[] | select(.judgment==false) | .question' "$file"
  echo ""
done
```

### Creating a Simple Dashboard with pandas

```python
#!/usr/bin/env python3
# save as analyze_results.py

import json
import glob
import pandas as pd
import matplotlib.pyplot as plt

# Load all result files
results = []
for filename in glob.glob('results/*.json'):
    with open(filename, 'r') as f:
        data = json.load(f)
        # Extract model name from filename
        model_name = filename.split('/')[-1].split('_')[0]
        results.append({
            'model': model_name,
            'yes_percentage': data['summary']['yes_percentage'],
            'mean_confidence': data['summary']['mean_confidence']
        })

# Create a DataFrame
df = pd.DataFrame(results)

# Plot results
fig, ax = plt.subplots(figsize=(10, 6))
df.plot(x='model', y='yes_percentage', kind='bar', ax=ax)
plt.title('Model Comparison - Yes Percentage')
plt.ylabel('Yes Percentage (%)')
plt.tight_layout()
plt.savefig('model_comparison.png')
print(f"Average yes percentage: {df['yes_percentage'].mean():.2f}%")
print(f"Best model: {df.loc[df['yes_percentage'].idxmax()]['model']}")
```

Run with:
```bash
python analyze_results.py
```

## Best Practices

1. **Create specific evaluation sets** for different use cases (support, marketing, technical writing)
2. **Use descriptive names** for evaluation sets and include clear descriptions
3. **Start with broader criteria** and then refine with more specific evaluation sets
4. **Save evaluation results** to files for tracking and analysis over time
5. **Compare multiple judge models** to reduce evaluation bias
6. **Use the `AGENTOPTIM_OMIT_REASONING=1` environment variable** for faster evaluations when detailed reasoning isn't needed
7. **Create dedicated question files** for reuse across similar evaluation sets
8. **Set the appropriate parallel level** based on your machine's capabilities
9. **Use JSON output format** for programmatic processing in other tools
10. **Check cache statistics** periodically to monitor performance