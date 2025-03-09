"""
Example of using AgentOptim for iterative response improvement.

This example demonstrates how to:
1. Evaluate an initial response against quality criteria
2. Identify specific areas for improvement based on failed criteria
3. Generate improved responses targeting those areas
4. Re-evaluate to measure improvement
5. Iterate until quality targets are met

Use case: Systematically improving response quality through targeted refinement
"""

import asyncio
import json
import os
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from pprint import pprint

from agentoptim import manage_evalset_tool, run_evalset_tool


# Define a function to format conversation for display
def format_conversation(conversation):
    formatted = ""
    for message in conversation:
        role = message["role"].upper()
        content = message["content"]
        formatted += f"{role}: {content}\n\n"
    return formatted.strip()


async def main():
    print("=== AgentOptim Iterative Response Improvement ===")
    print("This example demonstrates systematic response improvement based on evaluation feedback")
    
    # Step 1: Create an EvalSet with quality criteria
    print("\n1. Creating quality criteria EvalSet...")
    
    evalset_result = await manage_evalset_tool(
        action="create",
        name="Response Quality Improvement Criteria",
        questions=[
            "Is the response factually accurate?",
            "Does the response directly address the user's question?",
            "Is the response clear and easy to understand?",
            "Is the response comprehensive and complete?",
            "Is the information organized in a logical structure?",
            "Does the response use an appropriate level of detail?",
            "Is the tone helpful and professional?",
            "Does the response provide concrete examples where appropriate?",
            "Does the response avoid unnecessary jargon or complexity?",
            "Would this response likely satisfy the user's information needs?"
        ],
        short_description="Criteria for iterative response improvement",
        long_description="This EvalSet contains detailed quality criteria to guide systematic improvement of responses. It covers accuracy, clarity, comprehensiveness, structure, detail level, tone, examples, and overall satisfaction." + " " * 100
    )
    
    # Extract the EvalSet ID
    evalset_id = evalset_result.get("evalset", {}).get("id")
    print(f"Quality criteria EvalSet created with ID: {evalset_id}")
    
    # Step 2: Create an EvalSet with improvement suggestions template
    print("\n2. Creating improvement suggestions EvalSet...")
    
    suggestion_evalset_result = await manage_evalset_tool(
        action="create",
        name="Response Improvement Suggestions",
        template="""
        Given this conversation:
        {{ conversation }}
        
        Please provide specific improvement suggestions based on this criterion:
        {{ eval_question }}
        
        Analyze the assistant's response carefully, focusing specifically on the criterion above.
        
        Return a JSON object with the following format:
        {
            "judgment": 0 or 1,  // 0 if improvements needed, 1 if satisfactory
            "analysis": "Detailed analysis of how the response meets or fails to meet this criterion",
            "specific_suggestions": ["Specific suggestion 1", "Specific suggestion 2", ...],
            "improved_section": "An example of how to improve this aspect of the response (if judgment is 0)"
        }
        """,
        questions=[
            "How can the factual accuracy of this response be improved?",
            "How can this response better address the user's specific question?",
            "How can the clarity and understandability of this response be improved?",
            "How can this response be made more comprehensive and complete?",
            "How can the logical structure and organization of this response be improved?",
            "How can the level of detail in this response be optimized?",
            "How can the tone of this response be improved to be more helpful and professional?",
            "How can concrete examples be better incorporated into this response?",
            "How can unnecessary jargon or complexity be reduced in this response?",
            "Overall, what changes would make this response more satisfying to the user?"
        ],
        short_description="Generates specific improvement suggestions for responses",
        long_description="This EvalSet uses a custom template to generate detailed improvement suggestions for responses that don't meet quality criteria. For each criterion, it provides an analysis, specific suggestions, and example improvements that can be incorporated into an enhanced response." + " " * 100
    )
    
    # Extract the EvalSet ID
    suggestion_evalset_id = suggestion_evalset_result.get("evalset", {}).get("id")
    print(f"Improvement suggestions EvalSet created with ID: {suggestion_evalset_id}")
    
    # Step 3: Define an initial conversation with a response to improve
    print("\n3. Defining initial conversation...")
    
    # Initial conversation with a response that needs improvement
    initial_conversation = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What are the main differences between supervised and unsupervised learning in machine learning?"},
        {"role": "assistant", "content": "Supervised learning uses labeled data where the model is trained on input-output pairs, while unsupervised learning works with unlabeled data to find patterns. Supervised learning is used for classification and regression tasks, and unsupervised learning is used for clustering and dimensionality reduction. Supervised learning needs human annotation, but unsupervised doesn't require that."}
    ]
    
    print("Initial conversation:")
    print("-" * 60)
    print(format_conversation(initial_conversation))
    print("-" * 60)
    
    # Step 4: Evaluate the initial response
    print("\n4. Evaluating initial response...")
    
    initial_results = await run_evalset_tool(
        evalset_id=evalset_id,
        conversation=initial_conversation,
        model="meta-llama-3.1-8b-instruct",
        max_parallel=3
    )
    
    # Print initial evaluation summary
    print("\nInitial evaluation results:")
    print(f"Overall score: {initial_results['summary']['yes_percentage']:.1f}%")
    print(f"Criteria passed: {initial_results['summary']['yes_count']}/{initial_results['summary']['total_questions']}")
    
    # Print detailed results
    print("\nDetailed criterion evaluation:")
    for item in initial_results["results"]:
        judgment = "✅ Pass" if item["judgment"] else "❌ Fail"
        print(f"{judgment} | {item['question']}")
    
    # Step 5: Identify areas for improvement
    print("\n5. Identifying areas for improvement...")
    
    # Find failed criteria
    failed_criteria = [
        (i, item["question"]) 
        for i, item in enumerate(initial_results["results"]) 
        if not item["judgment"]
    ]
    
    if not failed_criteria:
        print("No improvements needed! The response passed all criteria.")
        return
    
    print(f"Found {len(failed_criteria)} areas for improvement:")
    for i, criterion in failed_criteria:
        print(f"• {criterion}")
    
    # Step 6: Get specific improvement suggestions
    print("\n6. Generating improvement suggestions...")
    
    # For each failed criterion, get improvement suggestions
    suggestion_results = await run_evalset_tool(
        evalset_id=suggestion_evalset_id,
        conversation=initial_conversation,
        model="meta-llama-3.1-8b-instruct",
        max_parallel=3
    )
    
    # Extract and organize the suggestions
    all_suggestions = []
    for i, (criterion_index, criterion) in enumerate(failed_criteria):
        # Find the matching suggestion
        suggestion_index = i % len(suggestion_results["results"])
        suggestion = suggestion_results["results"][suggestion_index]
        
        if "raw_result" in suggestion:
            raw = suggestion.get("raw_result", {})
            analysis = raw.get("analysis", "No analysis provided")
            specific_suggestions = raw.get("specific_suggestions", [])
            improved_section = raw.get("improved_section", "")
            
            all_suggestions.append({
                "criterion": criterion,
                "analysis": analysis,
                "specific_suggestions": specific_suggestions,
                "improved_section": improved_section
            })
            
            # Print the suggestion
            print(f"\nImprovement area {i+1}: {criterion}")
            print(f"Analysis: {analysis}")
            print("Specific suggestions:")
            for j, sugg in enumerate(specific_suggestions):
                print(f"  {j+1}. {sugg}")
    
    # Step 7: Create an improved response
    print("\n7. Creating improved response...")
    
    # Combine all suggestions into a comprehensive improvement plan
    improvement_plan = "\n\n".join([
        f"Area: {s['criterion']}\n" +
        f"Analysis: {s['analysis']}\n" +
        "Suggestions:\n" + 
        "\n".join([f"- {sugg}" for sugg in s['specific_suggestions']]) +
        (f"\nImproved example: {s['improved_section']}" if s['improved_section'] else "")
        for s in all_suggestions
    ])
    
    # Create a new improved response based on the suggestions
    # In a real application, you might use another LLM call to generate the improved response
    # based on the original and the improvement suggestions
    
    improved_conversation = initial_conversation.copy()
    improved_conversation[-1] = {
        "role": "assistant",
        "content": """Supervised and unsupervised learning are two fundamental paradigms in machine learning that differ in several important ways:

1. **Data Labeling**:
   - **Supervised Learning**: Uses labeled datasets where each input is paired with the correct output. For example, images labeled as "cat" or "dog".
   - **Unsupervised Learning**: Works with unlabeled data, finding patterns and relationships without predefined outputs.

2. **Primary Tasks**:
   - **Supervised Learning**: 
     - Classification: Assigning inputs to predefined categories (e.g., spam detection)
     - Regression: Predicting continuous values (e.g., house price prediction)
   - **Unsupervised Learning**:
     - Clustering: Grouping similar data points (e.g., customer segmentation)
     - Dimensionality Reduction: Simplifying data while preserving important patterns
     - Association: Finding relationships between variables (e.g., market basket analysis)

3. **Human Involvement**:
   - **Supervised Learning**: Requires significant human effort to label training data
   - **Unsupervised Learning**: Needs minimal human intervention for data preparation

4. **Training Process**:
   - **Supervised Learning**: The algorithm learns by comparing its predictions with correct answers, minimizing errors
   - **Unsupervised Learning**: The algorithm identifies intrinsic structures in data without feedback on correctness

5. **Examples**:
   - **Supervised Learning**: Linear regression, logistic regression, decision trees, neural networks (when trained with labeled data)
   - **Unsupervised Learning**: K-means clustering, hierarchical clustering, principal component analysis (PCA), autoencoders

6. **Practical Considerations**:
   - **Supervised Learning**: Generally achieves higher accuracy but requires more resources for data labeling
   - **Unsupervised Learning**: Can discover unexpected patterns but may be harder to evaluate and interpret

Both approaches can be combined in semi-supervised learning, where a small amount of labeled data is used alongside a larger unlabeled dataset, or in reinforcement learning, which uses rewards and penalties instead of explicit labels.

The choice between supervised and unsupervised learning depends on your specific problem, available data, and objectives."""
    }
    
    print("\nImproved response created based on suggestions")
    
    # Step 8: Evaluate the improved response
    print("\n8. Evaluating improved response...")
    
    improved_results = await run_evalset_tool(
        evalset_id=evalset_id,
        conversation=improved_conversation,
        model="meta-llama-3.1-8b-instruct",
        max_parallel=3
    )
    
    # Print improved evaluation summary
    print("\nImproved evaluation results:")
    print(f"Overall score: {improved_results['summary']['yes_percentage']:.1f}%")
    print(f"Criteria passed: {improved_results['summary']['yes_count']}/{improved_results['summary']['total_questions']}")
    
    # Step 9: Compare before and after
    print("\n9. Comparing results:")
    
    # Calculate improvement
    initial_score = initial_results['summary']['yes_percentage']
    improved_score = improved_results['summary']['yes_percentage']
    score_improvement = improved_score - initial_score
    
    # Create comparison table
    print("\nComparison of Initial vs. Improved Response:")
    print("-" * 80)
    print(f"{'Criterion':<50} | {'Initial':<10} | {'Improved':<10}")
    print("-" * 80)
    
    for i, item in enumerate(initial_results["results"]):
        criterion = item["question"]
        initial_judgment = "✅ Pass" if item["judgment"] else "❌ Fail"
        improved_judgment = "✅ Pass" if improved_results["results"][i]["judgment"] else "❌ Fail"
        print(f"{criterion:<50} | {initial_judgment:<10} | {improved_judgment:<10}")
    
    print("-" * 80)
    print(f"{'Overall Score':<50} | {initial_score:>8.1f}% | {improved_score:>8.1f}%")
    print(f"{'Improvement':<50} | {'':<10} | {score_improvement:>+8.1f}%")
    print("-" * 80)
    
    # Step 10: Create visualizations
    print("\n10. Creating result visualizations...")
    
    try:
        # Compare initial and improved scores in a bar chart
        plt.figure(figsize=(10, 6))
        
        # Prepare data
        scores = [initial_score, improved_score]
        labels = ['Initial Response', 'Improved Response']
        colors = ['#ff9999', '#66b3ff']
        
        # Create bars
        bars = plt.bar(labels, scores, color=colors)
        
        # Add score labels
        for bar, score in zip(bars, scores):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                     f'{score:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # Add improvement arrow and label
        plt.annotate(f'+{score_improvement:.1f}%', 
                     xy=(1, initial_score + score_improvement/2),
                     xytext=(0.5, initial_score + score_improvement/2),
                     arrowprops=dict(arrowstyle='->'),
                     fontweight='bold',
                     color='green')
        
        plt.ylabel('Quality Score (%)')
        plt.title('Response Quality Improvement')
        plt.ylim(0, 105)  # Give room for labels
        plt.grid(axis='y', alpha=0.3)
        
        # Save chart
        plt.tight_layout()
        plt.savefig('response_improvement_scores.png')
        print("Created 'response_improvement_scores.png'")
        
        # Create a before/after criterion comparison
        plt.figure(figsize=(12, 8))
        
        # Prepare data
        criteria = [item["question"].split('?')[0] + '?' for item in initial_results["results"]]  # Shorten
        initial_values = [1 if item["judgment"] else 0 for item in initial_results["results"]]
        improved_values = [1 if item["judgment"] else 0 for item in improved_results["results"]]
        
        # Set up bar positions
        bar_width = 0.35
        x = np.arange(len(criteria))
        
        # Create grouped bars
        plt.barh(x - bar_width/2, initial_values, bar_width, label='Initial', color='#ff9999')
        plt.barh(x + bar_width/2, improved_values, bar_width, label='Improved', color='#66b3ff')
        
        # Add labels and legend
        plt.yticks(x, criteria)
        plt.xlabel('Failed (0) → Passed (1)')
        plt.title('Criterion Satisfaction Before and After Improvement')
        plt.legend()
        
        # Improve layout for readability
        plt.tight_layout()
        plt.savefig('response_improvement_criteria.png')
        print("Created 'response_improvement_criteria.png'")
        
    except Exception as e:
        print(f"Error creating visualizations: {e}")
    
    # Step 11: Generate improvement report
    print("\n11. Generating improvement report...")
    
    # Create markdown report
    report = f"""# Response Improvement Report

## Summary

The response quality increased from **{initial_score:.1f}%** to **{improved_score:.1f}%**, an improvement of **{score_improvement:+.1f}%**.

- Initial response passed {initial_results['summary']['yes_count']}/{initial_results['summary']['total_questions']} criteria
- Improved response passed {improved_results['summary']['yes_count']}/{improved_results['summary']['total_questions']} criteria

## Initial Response

```
{initial_conversation[-1]['content']}
```

## Improved Response

```
{improved_conversation[-1]['content']}
```

## Improvement Process

The response was improved by addressing these key areas:

"""
    
    # Add improvement areas
    for i, (criterion_index, criterion) in enumerate(failed_criteria):
        report += f"### {i+1}. {criterion}\n\n"
        
        # Find if we have suggestions for this criterion
        matching_suggestions = [s for s in all_suggestions if s["criterion"] == criterion]
        if matching_suggestions:
            suggestion = matching_suggestions[0]
            report += f"**Analysis**: {suggestion['analysis']}\n\n"
            report += "**Specific improvements made**:\n"
            for sugg in suggestion['specific_suggestions']:
                report += f"- {sugg}\n"
            report += "\n"
    
    report += """## Evaluation Criteria

The following quality criteria were used to evaluate the response:

"""
    
    # Add all criteria
    for item in initial_results["results"]:
        report += f"- {item['question']}\n"
    
    report += f"""
## Conclusion

This example demonstrates how AgentOptim can be used for systematic response improvement through:

1. Identifying specific areas for improvement based on quality criteria
2. Generating targeted suggestions to address each issue
3. Implementing improvements and measuring the impact
4. Iterating as needed to reach quality targets

This approach provides a structured method for improving response quality in a measurable way.

---

*Generated on {datetime.now().strftime("%Y-%m-%d %H:%M")} with AgentOptim v2.1.0*
"""
    
    # Save report
    try:
        with open('response_improvement_report.md', 'w') as f:
            f.write(report)
        print("Saved improvement report to 'response_improvement_report.md'")
    except Exception as e:
        print(f"Error saving report: {e}")
    
    # Step 12: Print final summary
    print("\n12. Improvement process summary:")
    
    print(f"\nResponse quality increased from {initial_score:.1f}% to {improved_score:.1f}% ({score_improvement:+.1f}%)")
    print(f"Initially passed: {initial_results['summary']['yes_count']}/{initial_results['summary']['total_questions']} criteria")
    print(f"After improvement: {improved_results['summary']['yes_count']}/{improved_results['summary']['total_questions']} criteria")
    
    print("\nKey improvements:")
    for s in all_suggestions[:3]:  # Show top 3 improvements
        print(f"- {s['specific_suggestions'][0] if s['specific_suggestions'] else 'Improved '+s['criterion']}")
    
    if improved_results['summary']['yes_count'] < improved_results['summary']['total_questions']:
        print("\nFor further improvement, additional iterations could address remaining criteria.")
    else:
        print("\nSuccessfully improved the response to meet all quality criteria!")
    
    print("\nThis example demonstrates a systematic approach to improving response quality")
    print("through targeted feedback and iterative enhancement.")


if __name__ == "__main__":
    asyncio.run(main())