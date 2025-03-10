# Confidence Visualization Guide

This guide explains how to use AgentOptim's confidence visualization tools to analyze and understand the confidence scores produced by LLMs during evaluation.

## Introduction to Confidence Visualization

When LLMs provide confidence scores with their judgments, these scores can be analyzed to understand:

1. How well-calibrated the model is (i.e., do its confidence scores match actual accuracy?)
2. How uncertain the model is in different scenarios 
3. The distribution of confidence across different questions
4. How different models' confidence patterns compare

AgentOptim provides tools for both programmatic and CLI-based visualization of confidence scores to help you analyze these aspects.

## Visualization Types

AgentOptim supports four main types of confidence visualizations:

### 1. Histogram

Histograms show the distribution of confidence scores across bins, helping you understand the spread and frequency of different confidence levels.

![Confidence Histogram](images/confidence_histogram.png)

### 2. Calibration Curve

Calibration curves plot predicted confidence against actual accuracy, showing how well-calibrated a model is. A perfectly calibrated model's curve would follow the diagonal line.

![Calibration Curve](images/calibration_curve.png)

### 3. Distribution Table

The distribution table shows confidence scores for each individual question, allowing you to see which questions the model was more or less confident about.

### 4. Summary Statistics

Summary statistics provide an overview of key confidence metrics including:
- Mean and median confidence
- Confidence variance and range
- Expected Calibration Error (ECE)
- Confidence-accuracy gap
- Distribution across confidence buckets

## Using the CLI for Visualization

AgentOptim's CLI provides easy access to confidence visualizations through the `run visualize` command:

```bash
# Basic summary of confidence metrics
agentoptim run visualize <eval_run_id> --type summary

# Generate a histogram (with optional colorization)
agentoptim run visualize <eval_run_id> --type histogram --color

# Create a calibration curve and save as image
agentoptim run visualize <eval_run_id> --type calibration --save-image calibration.png

# View confidence distribution by question with color coding
agentoptim run visualize <eval_run_id> --type distribution --color

# Compare calibration between two evaluation runs
agentoptim run visualize <eval_run_id> --type calibration --compare <another_eval_run_id>
```

### Command Options

- `--type`: Choose visualization type (`histogram`, `calibration`, `distribution`, `summary`)
- `--color`: Use colors in terminal output for better readability
- `--format`: Choose output format (`ascii`, `unicode`, `emoji`)
- `--output`: Save results to a file
- `--save-image`: Save visualization as an image file (for histogram and calibration)
- `--compare`: Compare with another evaluation run (for calibration)
- `--bins`: Number of bins for histogram/calibration (default: 10)

## Programmatic Visualization

You can also use AgentOptim's visualization functions directly in your Python code:

```python
from agentoptim.visualization import (
    generate_confidence_histogram,
    plot_calibration_curve,
    plot_multi_model_calibration,
    export_confidence_data,
    format_confidence_cli
)
from agentoptim.confidence import calculate_ece

# Generate a histogram
fig = generate_confidence_histogram(
    confidence_scores,
    output_path="histogram.png",
    title="Confidence Distribution",
    bins=10,
    show_stats=True
)

# Create a calibration curve
fig = plot_calibration_curve(
    confidence_scores,
    judgments,
    output_path="calibration.png",
    title="Calibration Curve",
    num_bins=10,
    show_ece=True
)

# Compare multiple models
model_data = {
    "Model A": (confidence_scores_a, judgments_a),
    "Model B": (confidence_scores_b, judgments_b)
}
fig = plot_multi_model_calibration(
    model_data,
    output_path="comparison.png",
    title="Model Comparison",
    num_bins=10
)

# Export confidence data in various formats
data = export_confidence_data(
    eval_results,
    output_format="json",  # or "csv", "html", "markdown"
    output_path="confidence_data.json"
)
```

## Understanding Key Metrics

### Expected Calibration Error (ECE)

ECE measures how well calibrated a model's confidence scores are by calculating the weighted average of the differences between confidence and accuracy across bins. Lower ECE means better calibration.

### Confidence-Accuracy Gap

The difference between average accuracy and average confidence. A positive gap means the model is underconfident (accuracy > confidence), while a negative gap means it's overconfident.

### Confidence Variance

How much the confidence scores vary across questions. High variance may indicate the model is able to distinguish between easy and hard questions.

## Example Interpretation

Here's how to interpret different visualization patterns:

### Histograms

- **Peaked at high confidence**: Model is very confident in most judgments
- **Uniform distribution**: Model has variable confidence across questions
- **Bimodal distribution**: Model has two distinct confidence levels (very sure or very unsure)

### Calibration Curves

- **Above diagonal**: Model is underconfident (accuracy higher than confidence)
- **Below diagonal**: Model is overconfident (confidence higher than accuracy)
- **On diagonal**: Well-calibrated model
- **S-shaped**: Model is well-calibrated in middle ranges but not at extremes

## Best Practices

1. **Always compare multiple models**: Calibration patterns can vary significantly between models
2. **Use larger evaluation sets**: More data points provide more reliable calibration estimates
3. **Separate by category**: Consider analyzing calibration separately for different question types
4. **Relate to qualitative analysis**: Connect confidence patterns to specific model behaviors
5. **Track over time**: Monitor calibration as you make changes to your models or prompts

## Complete Example

For a complete working example of confidence visualization, see the [Confidence Visualization Example](../examples/confidence_visualization.py) in the examples directory.

```python
# Run the example
python examples/confidence_visualization.py
```

This example will:
1. Create a sample evaluation set with confidence elicitation
2. Run evaluations with two different confidence patterns
3. Generate various visualizations including histograms and calibration curves
4. Export confidence data in multiple formats
5. Display summary statistics

## Additional Resources

- [Confidence Elicitation Documentation](CONFIDENCE_ELICITATION.md): Learn how to elicit confidence scores from LLMs
- [Example: Confidence Implementation](../examples/confidence_example.py): How to use different confidence elicitation methods
- [Calibration Paper](https://arxiv.org/abs/1706.04599): "On Calibration of Modern Neural Networks" (Guo et al., 2017)
- [Verbalized Uncertainty Quantification](https://arxiv.org/abs/2307.02056): "Practically Calibrated Uncertainty in LLMs" (Lin et al., 2023)