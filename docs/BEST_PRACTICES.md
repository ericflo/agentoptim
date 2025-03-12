# AgentOptim v2.2.0 Best Practices

This guide provides best practices for using AgentOptim v2.2.0, with a particular focus on system message optimization. Following these recommendations will help you achieve better results and more efficient workflows.

## Table of Contents

- [System Message Optimization](#system-message-optimization)
- [EvalSet Design](#evalset-design)
- [Performance Optimization](#performance-optimization)
- [Integration and Automation](#integration-and-automation)
- [Quality Assurance](#quality-assurance)
- [Collaboration and Sharing](#collaboration-and-sharing)

## System Message Optimization

### Creating Effective EvalSets for System Messages

1. **Cover all quality dimensions**: Include questions about clarity, specificity, tone, safety, and context.

2. **Balance question types**: Include both functional questions (e.g., "Does it specify the role?") and quality questions (e.g., "Is it concise while complete?").

3. **Target your domain**: Add domain-specific criteria (e.g., for customer support, include questions about empathy and problem resolution).

4. **Weight critical criteria**: If using a weighted evaluation, give higher weight to the most important criteria for your use case.

5. **Test your EvalSet**: Before using an EvalSet for optimization, test it on a few known system messages to ensure it correctly identifies good and bad examples.

### Optimizing for Specific Use Cases

1. **Provide a good starting point**: Use `base_system_message` to provide a foundation that already includes domain knowledge.

2. **Use representative queries**: Choose user queries that are representative of real user interactions in your application.

3. **Optimize for query groups**: Create different optimized system messages for different types of queries (e.g., informational vs. procedural).

4. **Balance specificity and flexibility**: Overly specific system messages may perform well for the target query but fail for variations.

5. **Consider iterative optimization**: Take the best system message and use it as a base for another round of optimization with a different query.

### Tuning System Message Generation

1. **Adjust diversity levels**: Use "high" diversity when exploring different approaches, "low" when refining an already good system message.

2. **Increase candidate count**: For important optimizations, generate more candidates (5-10) to explore a wider solution space.

3. **Provide detailed instructions**: Use the `additional_instructions` parameter to guide the generator toward specific styles or requirements.

4. **Consider concurrency**: For faster results, increase `max_parallel` based on your available API rate limits and compute resources.

5. **Enable self-optimization**: For long-running projects, periodically trigger self-optimization to improve the generator over time.

## EvalSet Design

### General EvalSet Best Practices

1. **Use binary questions**: Keep evaluation questions as yes/no questions for objective assessment.

2. **Avoid compound questions**: Split "Does it do X and Y?" into separate questions about X and Y.

3. **Be specific**: Replace vague wording like "good" or "appropriate" with specific, measurable criteria.

4. **Provide context**: Include enough information for the judge model to make consistent evaluations.

5. **Categorize questions**: Organize questions into logical groups (e.g., clarity, safety, helpfulness).

### Naming and Organization

1. **Use descriptive names**: Name EvalSets clearly to indicate their purpose (e.g., "Customer Support Quality Assessment").

2. **Include version info**: Add version numbers to EvalSet names when making significant updates.

3. **Document intended use**: Add comprehensive descriptions explaining when and how the EvalSet should be used.

4. **Reuse where possible**: Prefer modifying existing EvalSets rather than creating duplicates with minor changes.

5. **Archive unused EvalSets**: Rather than deleting, mark old EvalSets as deprecated in their description.

## Performance Optimization

### Speed and Efficiency

1. **Use caching**: Enable caching to avoid re-evaluating the same system message-query combinations.

2. **Optimize evaluation order**: Run quick, high-signal evaluations first before more detailed ones.

3. **Schedule heavy optimizations**: Run large optimization jobs during off-peak hours.

4. **Consider lighter models**: For initial evaluations, use lighter, faster models before final validation with larger models.

5. **Use batch processing**: For large-scale optimization, use batch processing to handle multiple optimizations in sequence.

### Resource Management

1. **Monitor API usage**: Keep track of token consumption, especially when using commercial APIs.

2. **Clean up old results**: Periodically archive or delete old optimization results that are no longer needed.

3. **Manage storage**: Export important results to external formats and clean up the local storage.

4. **Rate limit requests**: Implement rate limiting for concurrent requests to prevent API rate limit errors.

5. **Use appropriate timeouts**: Set realistic timeouts based on the complexity of the evaluation.

## Integration and Automation

### CI/CD Integration

1. **Automated testing**: Add system message optimization to your CI/CD pipeline to catch regressions.

2. **Continuous improvement**: Schedule periodic optimizations to keep system messages updated.

3. **Version control**: Store optimized system messages in version control alongside code.

4. **Approval workflows**: Implement approval workflows for system message updates in production.

5. **A/B testing**: Use the API to conduct A/B tests of different system messages on real user traffic.

### Scripting and Automation

1. **Create automation scripts**: Build scripts for common optimization workflows.

2. **Schedule recurring optimization**: Use cron jobs or similar to schedule regular optimizations.

3. **Notify on significant changes**: Set up notifications when optimization produces significantly better results.

4. **Chain optimizations**: Create pipelines that feed the results of one optimization into another.

5. **Integrate with monitoring**: Connect optimization results to your application monitoring.

## Quality Assurance

### Testing and Validation

1. **Maintain test sets**: Keep a set of standard queries for consistent benchmarking.

2. **Compare against baselines**: Always compare new system messages against established baselines.

3. **Evaluate in real scenarios**: Test optimized system messages on a diverse set of real user queries.

4. **Cross-validate**: Test system messages optimized for one query type on different query types.

5. **Perform human validation**: Have human reviewers verify the quality of optimized system messages.

### Safety and Ethics

1. **Add safety criteria**: Include explicit safety questions in your EvalSets.

2. **Test for bias**: Evaluate system messages for potential biases in different contexts.

3. **Check for overpromising**: Ensure system messages don't set unrealistic expectations.

4. **Maintain boundaries**: Verify that optimization doesn't remove important model limitations or guardrails.

5. **Respect privacy**: Avoid including sensitive or personally identifiable information in system messages.

## Collaboration and Sharing

### Team Workflows

1. **Establish naming conventions**: Use consistent naming for optimization runs and EvalSets.

2. **Document optimization history**: Keep logs of system message changes and their impact.

3. **Share results**: Use the export options to share optimization results with team members.

4. **Maintain a library**: Create a shared library of effective system messages for different use cases.

5. **Create templates**: Develop templates for common system message patterns.

### Knowledge Management

1. **Document rationale**: Record why specific system messages were chosen for production.

2. **Track performance metrics**: Maintain dashboards showing system message performance over time.

3. **Link to examples**: Connect system messages with example conversations showing their effectiveness.

4. **Regular reviews**: Schedule periodic reviews of active system messages to identify improvement opportunities.

5. **Knowledge sharing**: Hold review sessions to discuss system message optimization learnings.

---

By following these best practices, you'll maximize the value of AgentOptim's system message optimization capabilities and build more effective, consistent AI experiences.

For more technical details, refer to:
- [ADVANCED_TUTORIAL.md](ADVANCED_TUTORIAL.md) - Advanced examples and techniques
- [API_REFERENCE.md](API_REFERENCE.md) - Detailed API documentation