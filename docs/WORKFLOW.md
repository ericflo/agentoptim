# AgentOptim Workflow Guide

This guide provides a comprehensive overview of the AgentOptim workflow, from dataset creation to result analysis, with practical examples.

## Streamlined Workflow

AgentOptim provides a streamlined workflow for prompt optimization:

1. **Create Evaluation Criteria** - Define how responses will be judged
2. **Create Dataset** - Prepare examples for testing prompt variations  
3. **Create Experiment** - Define different prompt variants to test
4. **Create & Run Job** - Execute the experiment and evaluate results (jobs auto-start by default)
5. **Analyze Results** - Determine the best performing prompts

## Quick Start Example

Here's a complete example of the AgentOptim workflow:

```python
# 1. Create an evaluation with specific criteria
evaluation = manage_evaluation_tool(
    action="create",
    name="Response Quality",
    template="Input: {input}\nResponse: {response}\nQuestion: {question}",
    questions=[
        "Is the response clear and concise?",
        "Does the response directly answer the question?",
        "Is the tone of the response appropriate?"
    ]
)
evaluation_id = evaluation["evaluation"]["evaluation_id"]

# 2. Create a dataset of example inputs
dataset = manage_dataset_tool(
    action="create",
    name="Customer Questions",
    items=[
        {"input": "How do I reset my password?"},
        {"input": "What are your business hours?"},
        {"input": "Can I return an item after 30 days?"}
    ]
)
dataset_id = dataset["dataset"]["dataset_id"]

# 3. Create an experiment with different prompt variants
experiment = manage_experiment_tool(
    action="create",
    name="Customer Service Tone Test",
    dataset_id=dataset_id,
    evaluation_id=evaluation_id,
    model_name="claude-3-sonnet-20240229",
    prompt_variants=[
        {
            "name": "formal",
            "content": "You are a customer service representative for a premium retailer. Use formal, professional language."
        },
        {
            "name": "friendly",
            "content": "You are a helpful customer service agent. Use a friendly, conversational tone."
        },
        {
            "name": "concise",
            "content": "You are a customer support agent. Provide brief, direct answers without unnecessary details."
        }
    ]
)
experiment_id = experiment["experiment"]["experiment_id"]

# 4. Create and run a job (waits for completion by default)
job = run_job_tool(
    action="create",
    experiment_id=experiment_id,
    dataset_id=dataset_id,
    evaluation_id=evaluation_id,
    judge_model="llama-3-8b-instruct",
    poll_interval=5  # Check status every 5 seconds
)
# No need to poll for completion - job is already complete!
job_id = job["job"]["job_id"]
print(f"Job completed with status: {job['job']['status']}")
print(f"Job took {job['job']['elapsed_time']:.2f} seconds")

# 5. Analyze results to find the best prompt variant
analysis = analyze_results_tool(
    action="analyze",
    experiment_id=experiment_id,
    name="Customer Service Tone Analysis"
)

# View the results
print(f"Best variant: {analysis['best_variant']}")
print(f"Variant scores: {analysis['variant_scores']}")
```

## Key Workflow Improvements

### 1. Auto-Starting and Waiting for Jobs

Jobs now have two major workflow improvements:

#### Auto-Starting

Jobs automatically start when created. This eliminates the need for a separate "run" step:

```python
# Old multi-step workflow
job = run_job_tool(action="create", experiment_id="123", dataset_id="456", evaluation_id="789")
job_id = job["job"]["job_id"]
run_job_tool(action="run", job_id=job_id)  # No longer necessary!

# New streamlined workflow (auto_start=True by default)
job = run_job_tool(action="create", experiment_id="123", dataset_id="456", evaluation_id="789")
job_id = job["job"]["job_id"]  # Job is already running!
```

You can opt out of auto-starting by setting `auto_start=False`:

```python
# Create job without starting it
job = run_job_tool(
    action="create",
    experiment_id="123",
    dataset_id="456",
    evaluation_id="789",
    auto_start=False
)
```

#### Wait for Completion

Jobs wait for completion by default, but you can opt out by setting `wait=False`:

```python
# Create job and wait for it to complete (default behavior)
job_result = run_job_tool(
    action="create",
    experiment_id="123",
    dataset_id="456",
    evaluation_id="789"
    # wait=True is the default, so job will complete before returning
)

# Job is already complete! Results available immediately
print(f"Job completed with status: {job_result['job']['status']}")
print(f"Job took {job_result['job']['elapsed_time']:.2f} seconds")

# For async mode, use wait=False to return immediately
async_job_result = run_job_tool(
    action="create",
    experiment_id="123",
    dataset_id="456",
    evaluation_id="789",
    wait=False  # Return immediately without waiting
)

# You can also customize the polling interval (in seconds)
job_result = run_job_tool(
    action="create",
    experiment_id="123",
    dataset_id="456",
    evaluation_id="789",
    poll_interval=10  # Check status every 10 seconds (wait=True is default)
)
```

This creates two different execution modes to choose from:

1. **Synchronous mode (default, wait=True):**
   - Job starts and the tool blocks until job completes
   - Returns complete results when finished
   - Provides the simplest workflow - create job and get results in one step
   - Specify poll_interval to control how often status is checked

2. **Asynchronous mode (wait=False):** 
   - Job starts and the tool returns immediately
   - Use the "get" action later to check progress
   - Better for larger jobs that may take a long time to complete

### 2. Enhanced Example Documentation

All MCP tools now include:

- **QUICKSTART EXAMPLES** at the top of tool descriptions
- **WORKFLOW EXAMPLES** showing complete end-to-end processes
- **Improved error messages** with formatted code examples
- **Better parameter descriptions** with clear examples

### 3. Simplified Tool Interface

The AgentOptim interface has been simplified:

- Removed unnecessary specialized tools like `check_job_status`
- Consolidated functionality into a core set of 5 powerful tools
- Added more comprehensive default values for optional parameters
- Improved response formatting for better readability

## Key Tool Examples

### Creating an Evaluation

```python
manage_evaluation_tool(
    action="create",
    name="Code Quality Evaluation",
    template="Code: {input}\nResponse: {response}\nQuestion: {question}",
    questions=[
        "Does the code solution correctly solve the problem?",
        "Is the code well-documented with comments?",
        "Is the code efficiently implemented?",
        "Does the code follow best practices for the language?"
    ],
    description="Evaluation for assessing code solution quality"
)
```

### Creating and Running a Job

```python
# Create a job and wait for completion (default behavior)
job_result = run_job_tool(
    action="create",
    experiment_id="exp_id_123",
    dataset_id="dataset_id_456",
    evaluation_id="eval_id_789",
    judge_model="claude-3-haiku-20240307"
)

# Job is already complete when it returns!
job_id = job_result["job"]["job_id"]
status = job_result["job"]["status"]
elapsed_time = job_result["job"]["elapsed_time"]
print(f"Job completed with ID: {job_id}")
print(f"Final status: {status}")
print(f"Completed in {elapsed_time:.2f} seconds")
print(f"Tasks processed: {job_result['job']['progress']['completed']}/{job_result['job']['progress']['total']}")

# For async mode (no waiting), use wait=False:
async_job = run_job_tool(
    action="create",
    experiment_id="exp_id_123",
    dataset_id="dataset_id_456",
    evaluation_id="eval_id_789",
    wait=False  # Return immediately without waiting
)
async_job_id = async_job["job"]["job_id"]
print(f"Async job started with ID: {async_job_id}")
```

## Full Workflow Example: Optimizing a Math Problem Solver

This example demonstrates a complete workflow for optimizing a prompt that solves math problems:

```python
# 1. Create a dataset of math problems
dataset = manage_dataset_tool(
    action="create",
    name="Algebra Problems",
    items=[
        {"input": "Solve for x: 2x + 5 = 13"},
        {"input": "Solve the equation: 3(x - 2) = 15"},
        {"input": "If y = 3x - 7 and y = 11, what is x?"},
        {"input": "Simplify the expression: 2(3x + 4) - 5x"}
    ]
)
dataset_id = dataset["dataset"]["dataset_id"]

# 2. Create an evaluation for math solutions
evaluation = manage_evaluation_tool(
    action="create",
    name="Math Solution Evaluation",
    template="Problem: {input}\n\nSolution: {response}\n\nQuestion: {question}",
    questions=[
        "Is the final answer correct?",
        "Are the solution steps clearly shown?",
        "Is the mathematical reasoning sound?",
        "Would this solution be helpful to a student learning algebra?"
    ]
)
evaluation_id = evaluation["evaluation"]["evaluation_id"]

# 3. Create an experiment with different approaches
experiment = manage_experiment_tool(
    action="create",
    name="Math Problem Solving Approaches",
    dataset_id=dataset_id,
    evaluation_id=evaluation_id,
    model_name="claude-3-opus-20240229",
    prompt_variants=[
        {
            "name": "step_by_step",
            "content": "You are a math tutor. Solve the math problem step by step, explaining each step clearly. Show all your work and highlight the final answer."
        },
        {
            "name": "concise_solution",
            "content": "You are a math expert. Solve the math problem efficiently with minimal steps. Focus on the most direct approach to the solution."
        },
        {
            "name": "visual_approach",
            "content": "You are a math teacher. Solve the math problem by explaining the visual or intuitive way to think about it. Use analogies or visual representations where helpful."
        }
    ]
)
experiment_id = experiment["experiment"]["experiment_id"]

# 4. Create and run a job (waits for completion by default)
import time
start_time = time.time()
print("Starting job and waiting for completion...")
job = run_job_tool(
    action="create",
    experiment_id=experiment_id,
    dataset_id=dataset_id,
    evaluation_id=evaluation_id,
    poll_interval=5  # Check status every 5 seconds
)
elapsed_time = time.time() - start_time
job_id = job["job"]["job_id"]

# Job is already complete!
print(f"Job completed with status: {job['job']['status']}")
print(f"Job completed in {elapsed_time:.2f} seconds")
print(f"Tasks processed: {job['job']['progress']['completed']}/{job['job']['progress']['total']}")
print(f"Success rate: {job['job']['results'].get('succeeded', 0)}/{job['job']['progress']['completed']} tasks")

# 6. Analyze the results
analysis = analyze_results_tool(
    action="analyze",
    experiment_id=experiment_id,
    name="Math Teaching Approach Analysis"
)

print(f"Best approach: {analysis['best_variant']}")
print(f"All scores: {analysis['variant_scores']}")
```

## Tips for Effective Optimization

1. **Start with diverse prompt variants** - Test significantly different approaches
2. **Use realistic examples** in your dataset
3. **Define clear evaluation criteria** - Questions should be specific and objective
4. **Monitor jobs during execution** - Check progress and status
5. **Run multiple experiments** - Refine based on initial findings
6. **Compare performances** across different models when possible
7. **Use local models for testing** - AgentOptim supports both cloud and local models