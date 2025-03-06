# AgentOptim

AgentOptim is a focused-but-powerful set of MCP tools that allows an MCP-aware agent to optimize a prompt in a data-driven way. Think of it as DSPy, but for agents - a toolkit that enables autonomous experimentation, evaluation, and optimization of prompts and interactions.

## Release v1.0 Complete! 🎉

We're excited to announce the completion of the AgentOptim v1.0 release, implementing all planned phases of the project. AgentOptim is now ready for use in production environments, with comprehensive features for prompt optimization, robust error handling, extensive documentation, and excellent performance.

## Overview

AgentOptim provides a streamlined toolkit for agents to:
1. Define evaluation criteria for responses
2. Create and manage datasets for testing
3. Run experiments with different prompt variations
4. Analyze results and optimize prompts based on data

With just 5 powerful tools, an agent can conduct sophisticated prompt optimization experiments autonomously, discovering optimal prompts that would be difficult or time-consuming to develop manually.

## Key Concepts

### Evaluation

An evaluation consists of yes/no questions paired with a template that formats these questions for a judge model. This provides a structured way to assess the quality of generated responses.

Example template:

```jinja2
Given the following conversation history:
<conversation_history>
{{ history }}
</conversation_history>

Please answer the following question about the final assistant response:
<question>
{{ question }}
</question>

Return a JSON object with the following format:
{"judgment": 1} for yes or {"judgment": 0} for no.
```

Example yes/no questions:

```python
QUESTIONS = [
    "Does the response define or clarify key terms or concepts if needed?",
    "Is the response concise, avoiding unnecessary filler or repetition?",
    "Does the response align with common sense or generally accepted reasoning?",
]
```

## Usage examples for MCP tools

### Creating an evaluation

```python
# Example: Creating a support response quality evaluation
result = manage_evaluation_tool(
    action="create",
    name="Support Quality Evaluation",
    template="""
        Input: {input}
        Response: {response}
        
        Question: {question}
        
        Answer yes (1) or no (0) in JSON format: {"judgment": 1 or 0}
    """,
    questions=[
        "Does the response directly address the customer's question?",
        "Is the response polite and professional?",
        "Does the response provide a complete solution?",
        "Is the response clear and easy to understand?"
    ],
    description="Evaluation criteria for customer support responses"
)
```

### Creating a dataset

```python
# Example: Creating a dataset of customer questions
result = manage_dataset_tool(
    action="create",
    name="Customer Support Questions",
    items=[
        {"input": "How do I reset my password?", "expected_output": "To reset your password..."},
        {"input": "Where can I update my shipping address?", "expected_output": "You can update your shipping address..."},
        {"input": "My order hasn't arrived yet. What should I do?", "expected_output": "If your order hasn't arrived..."}
    ],
    description="Common customer support questions"
)
```

### Creating an experiment

```python
# Example: Creating an experiment to test different customer service tones
result = manage_experiment_tool(
    action="create",
    name="Support Tone Experiment",
    description="Testing formal vs casual tone in support responses",
    dataset_id="ee7d8c9b-6f5e-4d3c-b2a1-0f9e8d7c6b5a",  # ID from manage_dataset_tool
    evaluation_id="a1b2c3d4-e5f6-g7h8-i9j0-k1l2m3n4o5p6",  # ID from manage_evaluation_tool
    prompt_variants=[
        {
            "name": "formal_tone",
            "content": "You are a customer service representative. Use formal, professional language. Address the customer respectfully and provide clear, thorough solutions to their problems."
        },
        {
            "name": "casual_tone",
            "content": "You're a friendly support agent. Use a casual, conversational tone. Be warm and approachable while still being helpful and solving the customer's problem efficiently."
        }
    ],
    model_name="claude-3-opus-20240229",
    temperature=0.7,
    max_tokens=500
)
```

### Running a job

```python
# Example: Creating and automatically running a job (auto_start=True by default)
job_result = run_job_tool(
    action="create",
    experiment_id="9c8d7e6f-5g4h-3i2j-1k0l-9m8n7o6p5q4r",  # ID from manage_experiment_tool
    dataset_id="ee7d8c9b-6f5e-4d3c-b2a1-0f9e8d7c6b5a",
    evaluation_id="a1b2c3d4-e5f6-g7h8-i9j0-k1l2m3n4o5p6",
    judge_model="claude-3-haiku-20240307",
    max_parallel=3
)

# Extract job_id from the result
job_id = job_result["job"]["job_id"]  # Jobs now start automatically

# Check job status (may need to wait for completion)
status_result = run_job_tool(
    action="get",
    job_id=job_id
)
```

### Analyzing results

```python
# Example: Analyzing job results
analysis_result = analyze_results_tool(
    action="analyze",
    experiment_id="9c8d7e6f-5g4h-3i2j-1k0l-9m8n7o6p5q4r",
    job_id="7r6q5p4o-3n2m-1l0k-9j8i-7h6g5f4e3d2c",
    name="Support Tone Analysis"
)
```

### Dataset

A dataset is a collection of examples for training and testing. Datasets can be created manually, imported from external sources, or generated by the agent.

### Experiment

An experiment tests different prompt variations against a dataset using the defined evaluations. Each experiment includes:
- Prompt variations to test (system prompts, templates)
- Test inputs from a dataset
- Evaluations to run on the results
- Configuration settings and results

## Simplified Tool Design

AgentOptim provides just 5 powerful, flexible tools:

### 1. `manage_evaluation`

A unified tool for creating, updating, listing, and deleting evaluations.

```python
def manage_evaluation(
    action: str,               # "create", "list", "get", "update", "delete"
    evaluation_id: str = None, # Required for get, update, delete
    name: str = None,          # Required for create
    template: str = None,      # Required for create, optional for update
    questions: list = None,    # Required for create, optional for update
) -> str:
    """Manage evaluation definitions for assessing response quality."""
```

### 2. `manage_dataset`

A unified tool for creating, updating, listing, and managing datasets.

```python
def manage_dataset(
    action: str,               # "create", "list", "get", "update", "delete", "split"
    dataset_id: str = None,    # Required for get, update, delete, split
    name: str = None,          # Required for create
    examples: list = None,     # Required for create, optional for update
    label_field: str = None,   # Optional
    test_split: float = 0.2,   # Optional, used for split action
) -> str:
    """Manage datasets of examples for testing prompt variations."""
```

### 3. `manage_experiment`

A unified tool for creating, updating, listing, and managing experiments.

```python
def manage_experiment(
    action: str,               # "create", "list", "get", "update", "delete"
    experiment_id: str = None, # Required for get, update, delete
    name: str = None,          # Required for create
    dataset_id: str = None,    # Required for create
    evaluation_ids: list = None, # Required for create
    prompt_variants: dict = None, # Optional for create/update
) -> str:
    """Manage experiments for testing prompt variations."""
```

### 4. `run_job`

A unified tool for executing evaluations or experiments.

```python
def run_job(
    job_type: str,            # "evaluation" or "experiment"
    job_id: str,              # Evaluation ID or experiment ID
    inputs: dict = None,      # Required for evaluation, optional for experiment
    model: str = None,        # Optional, defaults to system configuration
    endpoint: str = None,     # Optional, defaults to system configuration
) -> str:
    """Run an evaluation or experiment job."""
```

### 5. `analyze_results`

A unified tool for analyzing and optimizing based on experiment results.

```python
def analyze_results(
    experiment_id: str,        # Required
    action: str,               # "summarize", "compare", "optimize", "export"
    variants: list = None,     # Optional, for comparing specific variants
    detail_level: str = "medium", # Optional
    optimization_strategy: str = None, # Optional, for optimize action
) -> str:
    """Analyze experiment results and generate optimized prompts."""
```

## Real-World Use Cases with Tool Calls

### Use Case 1: GitHub PR Risk Classifier

**User prompt to Claude:**
```
I need to build a classifier that can determine if a GitHub pull request is risky. Here's some training data:

[
    {"pr_data": {"title": "Refactor auth module", "files_changed": 35, "additions": 1200, "deletions": 800, "comments": 3}, "is_risky": true},
    {"pr_data": {"title": "Fix typo in README", "files_changed": 1, "additions": 2, "deletions": 2, "comments": 0}, "is_risky": false},
    {"pr_data": {"title": "Update dependencies", "files_changed": 1, "additions": 15, "deletions": 10, "comments": 0}, "is_risky": false},
    {"pr_data": {"title": "Implement new payment processor", "files_changed": 25, "additions": 800, "deletions": 200, "comments": 8}, "is_risky": true},
    ...20 more examples...
]

Help me create an optimal system prompt that will accurately classify PRs as risky or not.
```

**Tool Calls Sequence (simplified):**

1. `manage_dataset(action="create", name="pr_risk", examples=[...], label_field="is_risky")`
2. `manage_evaluation(action="create", name="pr_risk_eval", template="Given a PR with these characteristics: {{ pr_data }}\nIs this PR risky?\n\nReturn JSON: {\"judgment\": 1 for risky, 0 for not risky}", questions=["Is the prediction correct compared to the ground truth?"])`
3. `manage_experiment(action="create", name="pr_risk_classifier", dataset_id="pr_risk", evaluation_ids=["pr_risk_eval"], prompt_variants={
    "base_variant": "You are an expert GitHub reviewer who evaluates pull request risk based on characteristics like size, files changed, and complexity.",
    "detailed_variant": "You are a senior software engineer specializing in code review safety. Your task is to classify pull requests as risky or not risky based on quantitative metrics including number of files changed, lines added/removed, and comments. PRs with many files changed (>20), many lines modified (>1000), or critical subsystems affected should be flagged as risky.",
    "concise_variant": "Classify GitHub PRs as risky (large changes, many files, critical areas) or not risky (small changes, few files, non-critical areas)."
   })`
4. `run_job(job_type="experiment", job_id="pr_risk_classifier")`
5. `analyze_results(experiment_id="pr_risk_classifier", action="summarize")`

**Claude's Response:**
```
I've tested different system prompts to classify PR risk, and here are the results:

- "detailed_variant" performed best with 92% accuracy on the test set
- "base_variant" achieved 78% accuracy
- "concise_variant" achieved 65% accuracy

Based on these results, I recommend using this system prompt:

"You are a senior software engineer specializing in code review safety. Your task is to classify pull requests as risky or not risky based on quantitative metrics including number of files changed, lines added/removed, and comments. PRs with many files changed (>20), many lines modified (>1000), or critical subsystems affected should be flagged as risky."

The detailed version performed better because it provided specific thresholds for what constitutes a risky PR, giving the model clear decision boundaries.
```

### Use Case 2: Customer Service Tone Optimization

**User prompt to Claude:**
```
I'm trying to improve our customer service responses for technical support. I have these examples of customer queries and I want to find the best tone and approach for our responses.

Here are some example customer messages:
1. "Your app keeps crashing when I try to upload photos. Fix this ASAP!"
2. "I'm confused about how to change my subscription plan. Can you help?"
3. "I've been charged twice this month. This is unacceptable."
4. "Is there a way to export my data in CSV format?"

Help me find the optimal tone for our responses - should we be formal, casual, empathetic, or technical?
```

**Tool Calls Sequence (simplified):**

1. `manage_dataset(action="create", name="customer_support", examples=[...])`
2. `manage_evaluation(action="create", name="support_tone_eval", template="...", questions=["Is the response empathetic?", "Does the response provide a clear solution?", "Is the tone appropriate?", "Would this response likely satisfy the customer?"])`
3. `manage_experiment(action="create", name="support_tone_test", dataset_id="customer_support", evaluation_ids=["support_tone_eval"], prompt_variants={
    "formal_tone": "You are a customer support representative. Maintain a formal, professional tone. Use proper grammar and avoid contractions. Address the customer's concerns directly and provide clear solutions.",
    "casual_tone": "You're a friendly customer support agent. Keep your tone casual and approachable. Use contractions and simple language. Be helpful while maintaining a conversational style.",
    "empathetic_tone": "You are an empathetic customer support specialist. Begin by acknowledging the customer's feelings. Show that you understand their frustration or confusion. Then provide a clear solution to their problem.",
    "technical_tone": "You are a technical support expert. Focus on accurate, detailed technical solutions. Be precise and comprehensive in your explanation while maintaining professionalism."
   })`
4. `run_job(job_type="experiment", job_id="support_tone_test")`
5. `analyze_results(experiment_id="support_tone_test", action="summarize")`
6. `analyze_results(experiment_id="support_tone_test", action="optimize", variants=["empathetic_tone", "technical_tone"])`

**Claude's Response:**
```
I've tested different tones for customer support responses and found that the empathetic tone performed best overall with a score of 86%.

Based on these results, I've created an optimized prompt that combines the strengths of both the empathetic and technical approaches:

"You are a supportive customer service representative. Begin by acknowledging the customer's feelings and validating their concerns. Then provide a clear, technically accurate solution with step-by-step instructions when appropriate. Maintain a warm tone while being precise and thorough in your explanation."

This optimized prompt scored 92% in follow-up testing, showing that combining empathy with technical precision leads to the best customer service responses.
```

### Use Case 3: Research Paper Summarization Improvement

**User prompt to Claude:**
```
I need to summarize scientific research papers for our company blog. I have several examples of papers and current summaries, but I think we can improve the quality. The summaries need to be accurate, concise, and accessible to a non-expert audience while preserving the key findings.

Can you help me develop and test different approaches to paper summarization to find what works best?

[Example papers and existing summaries provided]
```

**Tool Calls Sequence (simplified):**

1. `manage_dataset(action="create", name="paper_summaries", examples=[...])`
2. `manage_evaluation(action="create", name="summary_quality", template="...", questions=["Does the summary accurately represent the paper's main findings?", "Does the summary avoid technical jargon or explain it when used?", "Does the summary highlight practical implications of the research?", "Is the summary concise (under 250 words)?"])`
3. `manage_experiment(action="create", name="summary_approaches", dataset_id="paper_summaries", evaluation_ids=["summary_quality"], prompt_variants={
    "structure_first": "Summarize research papers using this structure: 1) Problem addressed, 2) Methodology, 3) Key findings, 4) Implications. Make each section concise and accessible to non-experts.",
    "eli5_approach": "Summarize research papers as if explaining to a smart high school student. Focus on why the research matters, what was discovered, and how it might be used in the real world. Avoid jargon or define it simply when needed.",
    "journalist_style": "Summarize research papers in journalistic style with an inverted pyramid approach: start with the most important findings, then add context and details. Use engaging language and relate findings to real-world impact."
   })`
4. `run_job(job_type="experiment", job_id="summary_approaches")`
5. `analyze_results(experiment_id="summary_approaches", action="summarize", detail_level="high")`
6. `analyze_results(experiment_id="summary_approaches", action="optimize", variants=["eli5_approach", "journalist_style"])`

**Claude's Response:**
```
After testing different approaches to research paper summarization, I've found an optimal approach that scored 88% on our evaluation metrics.

The hybrid approach combines the accessibility of the ELI5 method with the engaging structure of journalistic style:

"Summarize research papers by beginning with the most important finding and its real-world implications (the headline). Then provide context about the problem being solved and why it matters. Explain the methodology in simple terms a high school student could understand. Conclude with broader implications and next steps for the research. Define technical terms when they cannot be avoided, and use analogies when helpful."

This approach significantly outperformed the structure-first method, which scored 65%. The hybrid approach was particularly strong on "accessibility to non-experts" while maintaining accuracy.
```

### Use Case 4: Multilingual Code Documentation Generator

**User prompt to Claude:**
```
Our team is building a global developer tool that needs to generate code documentation in multiple languages. We have a dataset of English code documentation examples that we've manually translated to Spanish, French, German, and Japanese. The problem is that our current approach of translating everything is expensive and slow.

I want to develop a prompt that can generate high-quality multilingual documentation directly. Can you help me experiment with different approaches that maintain technical accuracy while being natural in each target language?

Here's an example of our code and documentation in multiple languages:
[Examples of code snippets with documentation in 5 languages]
```

**Tool Calls Sequence (simplified):**

1. `manage_dataset(action="create", name="code_docs_multilingual", examples=[...], label_field="language")`
2. `manage_dataset(action="split", dataset_id="code_docs_multilingual", test_split=0.3)`
3. `manage_evaluation(action="create", name="doc_quality", template="Evaluate this code documentation:\n\nCode:\n{{ code }}\n\nDocumentation ({{ language }}):\n{{ documentation }}\n\nQuestion: {{ question }}", questions=[
    "Is the documentation technically accurate for the code?", 
    "Does the documentation sound natural in the target language?", 
    "Does the documentation follow common conventions for the target language?", 
    "Does the documentation maintain consistent terminology?",
    "Is the documentation complete (covers parameters, return values, exceptions)?"
   ])`
4. `manage_experiment(action="create", name="multilingual_docs", dataset_id="code_docs_multilingual", evaluation_ids=["doc_quality"], prompt_variants={
    "translate_approach": "You are a multilingual technical documentation expert. First, create comprehensive documentation for the code in English, following standard conventions. Then, translate the documentation into the requested target language ({{ language }}), ensuring it sounds natural to native speakers while maintaining technical precision.",
    "direct_approach": "You are a multilingual programming expert who writes native documentation in {{ language }}. Create documentation directly in {{ language }} for the provided code, following the documentation conventions typical for {{ language }} technical content.",
    "hybrid_approach": "You are a specialized documentation system for the {{ language }} market. Analyze the code carefully, noting key functions, parameters, return values, and exceptions. Create documentation in {{ language }} that follows local conventions while ensuring all technical details are accurate. Include equivalent English technical terms in parentheses where this would help clarity."
   })`
5. `run_job(job_type="experiment", job_id="multilingual_docs")`
6. `analyze_results(experiment_id="multilingual_docs", action="summarize", detail_level="high")`
7. `analyze_results(experiment_id="multilingual_docs", action="compare", variants=["translate_approach", "direct_approach", "hybrid_approach"])`

**Claude's Response:**
```
I've analyzed the results of our multilingual documentation experiment across 5 languages, and here are the findings:

Performance by approach (average across all languages):
- translate_approach: 72% (strongest for technical accuracy)
- direct_approach: 68% (strongest for natural language)
- hybrid_approach: 84% (balanced strengths)

The hybrid approach performed best overall, but with interesting variations by language:
- For Japanese: hybrid_approach scored 89% vs. 65% for translate_approach
- For Germanic languages (German): all approaches performed similarly (within 5%)
- For Romance languages (French, Spanish): hybrid_approach outperformed by 12-15%

The optimal prompt combines elements from the hybrid approach with language-specific adjustments:

"You are creating documentation for code in {{ language }}. Follow this process:
1. Analyze the code to identify functions, parameters, return values, and exceptions
2. Create documentation that follows the conventional style for {{ language }} technical writing
3. For Asian languages, include key technical terms in English parenthetically
4. For European languages, use localized technical terminology with English equivalents only when the localized term is uncommon
5. Ensure all examples use locale-appropriate formats for numbers, dates, and code comments"

This language-aware approach achieved 91% on our evaluation metrics, representing a significant improvement over our current translation pipeline while maintaining consistency across languages.
```
