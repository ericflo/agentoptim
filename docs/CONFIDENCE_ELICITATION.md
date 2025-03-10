# Confidence Elicitation Guide

This guide explains how to elicit confidence scores from language models using AgentOptim's confidence elicitation methods.

## Introduction to Confidence Elicitation

When evaluating AI responses, it's valuable to know not just whether a model thinks a response is correct or not, but also how confident it is in that assessment. Confidence elicitation is the process of extracting a model's uncertainty about its own judgments, providing deeper insights into model behavior.

AgentOptim implements multiple confidence elicitation methods based on the latest research in verbalized uncertainty quantification to help you understand:

1. Where models are uncertain in their evaluations
2. How well-calibrated different models are
3. Which evaluation criteria have the highest uncertainty
4. How confidence correlates with accuracy

## Confidence Elicitation Methods

AgentOptim supports the following confidence elicitation methods:

| Method | Description | Format | Example |
|--------|-------------|--------|---------|
| `basic_percentage` | Simple percentage format | 0-100% | "I'm 85% confident" |
| `basic_float` | Simple float format | 0.0-1.0 | "Confidence: 0.85" |
| `basic_letter` | Letter grades for confidence | A-E | "Confidence: B+" |
| `basic_text` | Text descriptions of confidence | Text | "High confidence" |
| `advanced_probability` | Probability with explanation | 0.0-1.0 | "0.9 confidence because..." |
| `combo_exemplars` | Few-shot examples with confidence | 0.0-1.0 | [See template below] |
| `multi_guess` | Multiple judgments with probabilities | List | "Yes (0.8), No (0.2)" |

### Default Method: combo_exemplars

Based on research findings, the `combo_exemplars` method generally yields the best calibration. This method provides few-shot examples of confidence scores across the full range (0.0-1.0) and asks the model to similarly express its confidence.

## Adding Confidence Elicitation to Evaluations

### When Creating an EvalSet

```python
from agentoptim import manage_evalset_tool

evalset_result = await manage_evalset_tool(
    action="create",
    name="Response Quality",
    questions=[
        "Is the response helpful?",
        "Is the response accurate?",
        "Is the response clear?"
    ],
    short_description="Quality assessment",
    # Add confidence elicitation config
    confidence_config={
        "method": "combo_exemplars",
        "instructions": "Also provide your confidence in your judgment on a scale from 0 to 1."
    }
)
```

### Confidence Config Options

The `confidence_config` parameter accepts the following options:

| Field | Type | Description | Default |
|-------|------|-------------|---------|
| `method` | string | Elicitation method to use | `"combo_exemplars"` |
| `instructions` | string | Custom instructions for confidence elicitation | Method-specific default |
| `extract_regex` | string | Custom regex for extracting confidence scores | Method-specific default |
| `default_value` | float | Default confidence if extraction fails | `None` |

## Methods In Detail

### basic_percentage

Elicits confidence as a percentage between 0% and 100%.

```python
confidence_config={
    "method": "basic_percentage",
    "instructions": "Express your confidence in this judgment as a percentage between 0% and 100%."
}
```

### basic_float

Elicits confidence as a float between 0.0 and 1.0.

```python
confidence_config={
    "method": "basic_float",
    "instructions": "Provide your confidence in this judgment as a number between 0.0 and 1.0."
}
```

### basic_letter

Elicits confidence as a letter grade (A, B, C, D, E) which is then converted to a numerical score.

```python
confidence_config={
    "method": "basic_letter",
    "instructions": "Rate your confidence in this judgment with a letter grade (A, B, C, D, or E)."
}
```

### basic_text

Elicits confidence as a text description (very low, low, medium, high, very high) which is then converted to a numerical score.

```python
confidence_config={
    "method": "basic_text",
    "instructions": "Rate your confidence as: very low, low, medium, high, or very high."
}
```

### advanced_probability

Elicits a probability estimate with explanation of the reasoning.

```python
confidence_config={
    "method": "advanced_probability",
    "instructions": "Express your confidence as a probability between 0 and 1, and explain your reasoning."
}
```

### combo_exemplars (Recommended)

Provides few-shot examples across confidence levels. This method typically gives the best calibration based on research.

```python
confidence_config={
    "method": "combo_exemplars",
    "instructions": "Also provide your confidence in your judgment on a scale from 0 to 1."
}
```

### multi_guess

Asks the model to provide multiple judgments with associated probabilities.

```python
confidence_config={
    "method": "multi_guess",
    "instructions": "Give your judgment (Yes/No) along with alternate judgments and their relative probabilities."
}
```

## Using Custom Confidence Instructions

You can customize the instructions for any method:

```python
confidence_config={
    "method": "basic_float",
    "instructions": "How certain are you in this assessment? Express as a decimal between 0.0 (completely uncertain) and 1.0 (completely certain)."
}
```

## Advanced Customization

### Custom Extraction Regex

For advanced users, you can provide a custom regex pattern to extract confidence scores:

```python
confidence_config={
    "method": "basic_float",
    "extract_regex": r"Confidence(?:\s+level)?(?:\s*):?\s*(\d+\.\d+)",
    "instructions": "Confidence level: [VALUE]"
}
```

### Fallback Value

You can specify a default confidence value to use if extraction fails:

```python
confidence_config={
    "method": "basic_float",
    "default_value": 0.5,
    "instructions": "Express your confidence from 0.0 to 1.0"
}
```

## Best Practices

1. **Use the recommended method**: Start with `combo_exemplars` as it generally provides the best calibration
2. **Test multiple methods**: Different models might respond better to different elicitation techniques
3. **Be consistent**: Use the same elicitation method across evaluations you want to compare
4. **Analyze calibration**: Use the [visualization tools](CONFIDENCE_VISUALIZATION.md) to check how well-calibrated the confidence scores are
5. **Include in templates**: If using custom templates, make sure to include the confidence instructions

## Understanding Evaluation Results

When you run an evaluation with confidence elicitation enabled, the results will include confidence scores:

```python
evaluation_result = await manage_eval_runs_tool(
    action="run",
    evalset_id=evalset_id,
    conversation=conversation
)

# Access confidence scores in results
for result in evaluation_result["results"]:
    judgment = result["judgment"]  # True/False
    confidence = result["confidence"]  # 0.0-1.0
    question = result["question"]
    print(f"Question: {question}")
    print(f"Judgment: {'Yes' if judgment else 'No'}")
    print(f"Confidence: {confidence:.2f}")
    print()
```

## Summary Statistics

The evaluation summary will also include confidence statistics:

```python
summary = evaluation_result["summary"]
mean_confidence = summary["mean_confidence"]
mean_yes_confidence = summary["mean_yes_confidence"]  # Average confidence for "Yes" judgments
mean_no_confidence = summary["mean_no_confidence"]  # Average confidence for "No" judgments
```

## Complete Example

For a complete working example, see the [Confidence Example](../examples/confidence_example.py) in the examples directory.

```python
# Run the example
python examples/confidence_example.py
```

For visualization of confidence scores, see the [Confidence Visualization Guide](CONFIDENCE_VISUALIZATION.md).

## Research Background

The confidence elicitation methods in AgentOptim are based on research in verbalized uncertainty quantification, particularly:

- Lin et al. (2023), "Practically Calibrated Uncertainty in Large Language Models" ([arXiv:2307.02056](https://arxiv.org/abs/2307.02056))
- Zhou et al. (2023), "SCOTT: Self-Consistent Chain-of-Thought Distillation" ([arXiv:2305.01879](https://arxiv.org/abs/2305.01879))
- Mielke et al. (2022), "Reducing Overconfidence in Large Language Models" ([arXiv:2207.05221](https://arxiv.org/abs/2207.05221))

The implementation and default settings are informed by these papers, with the `combo_exemplars` method showing the best balance of calibration and ease of use across different models.