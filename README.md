# agentoptim

AgentOptim is a focused-but-powerful set of MCP tools that allows an MCP-aware agent to optimize a prompt in a data-driven way. Think about it as DSPy, but for agents.

## Concepts

### Evaluation

Evaluation, which is a list of up to 100 yes/no questions paired with a template. Example template:

```jinja2
Given the following conversation history:
<conversation_history>
{{ history }}
</conversation_history>

Please answer the following question about the final assistant response:
<question>
{{ question }}
</question>
```

Example yes/no questions:

```python
QUESTIONS = [
    "Does the response define or clarify key terms or concepts if needed?",
    "Is the response concise, avoiding unnecessary filler or repetition?",
    "Does the response align with common sense or generally accepted reasoning?",
]
```

Tools to be implemented:

- Listing evaluations
- Creating an evaluation
- Deleting an evaluation

- Listing questions of an evaluation
- Adding a question to an evaluation
- Removing a question from an evaluation

- Running an evaluation on a conversation

### Experiment
