"""
Example of using AgentOptim with custom evaluation templates.

This example demonstrates how to:
1. Create EvalSets with custom templates for different use cases
2. Implement specialized evaluation patterns
3. Define templates with different output formats
4. Use templates for detailed, multi-faceted evaluation

Use case: Creating specialized evaluation templates for specific domains
"""

import asyncio
import json
from pprint import pprint

from agentoptim import manage_evalset_tool, run_evalset_tool


async def main():
    print("=== AgentOptim Custom Template Examples ===")
    print("This example demonstrates creating and using custom templates for specialized evaluations")
    
    # Step 1: Create an EvalSet with a basic template
    print("\n1. Creating standard template EvalSet...")
    
    standard_evalset_result = await manage_evalset_tool(
        action="create",
        name="Standard Template Example",
        template="""
        Given this conversation:
        {{ conversation }}
        
        Please answer the following yes/no question about the final assistant response:
        {{ eval_question }}
        
        Return a JSON object with the following format:
        {"judgment": 1} for yes or {"judgment": 0} for no.
        """,
        questions=[
            "Is the response helpful for the user's needs?",
            "Does the response directly address the user's question?",
            "Is the response clear and easy to understand?"
        ],
        short_description="Basic template with yes/no judgments",
        long_description="This EvalSet uses the standard template format that asks simple yes/no questions about assistant responses. It produces binary judgments with 1 for yes and 0 for no." + " " * 100
    )
    
    # Extract the EvalSet ID
    standard_evalset_id = standard_evalset_result.get("evalset", {}).get("id")
    print(f"Standard EvalSet created with ID: {standard_evalset_id}")
    
    # Step 2: Create an EvalSet with a Likert scale template
    print("\n2. Creating Likert scale template EvalSet...")
    
    likert_evalset_result = await manage_evalset_tool(
        action="create",
        name="Likert Scale Template",
        template="""
        Given this conversation:
        {{ conversation }}
        
        Please evaluate the following aspect of the final assistant response:
        {{ eval_question }}
        
        Rate your agreement on a 5-point Likert scale:
        1 - Strongly Disagree
        2 - Disagree
        3 - Neutral
        4 - Agree
        5 - Strongly Agree
        
        Provide your rating and a brief explanation in JSON format:
        {
            "rating": [1-5 integer],
            "explanation": "Brief justification for your rating",
            "judgment": [0 or 1 - convert to binary where 4-5 = 1, 1-3 = 0]
        }
        """,
        questions=[
            "The response is comprehensive and covers all aspects of the query",
            "The response is well-structured and organized logically",
            "The response provides accurate information without errors",
            "The response maintains an appropriate tone for the context"
        ],
        short_description="Template using 5-point Likert scale ratings",
        long_description="This EvalSet uses a 5-point Likert scale template to provide more nuanced evaluation beyond binary yes/no judgments. It captures the degree of agreement while still providing a binary judgment for compatibility with overall metrics." + " " * 100
    )
    
    # Extract the EvalSet ID
    likert_evalset_id = likert_evalset_result.get("evalset", {}).get("id")
    print(f"Likert scale EvalSet created with ID: {likert_evalset_id}")
    
    # Step 3: Create an EvalSet with a multi-criteria template
    print("\n3. Creating multi-criteria template EvalSet...")
    
    multi_criteria_evalset_result = await manage_evalset_tool(
        action="create",
        name="Multi-Criteria Template",
        template="""
        Given this conversation:
        {{ conversation }}
        
        Please evaluate the response in terms of the following criteria:
        {{ eval_question }}
        
        For each of these aspects, rate the response on a scale of 1-10 and provide justification.
        
        Return your evaluation in the following JSON format:
        {
            "accuracy": {
                "score": [1-10 integer],
                "justification": "Brief explanation for accuracy score"
            },
            "completeness": {
                "score": [1-10 integer],
                "justification": "Brief explanation for completeness score"
            },
            "clarity": {
                "score": [1-10 integer],
                "justification": "Brief explanation for clarity score"
            },
            "overall": {
                "score": [1-10 integer],
                "justification": "Brief explanation for overall score"
            },
            "judgment": [0 or 1 - where overall score 7+ = 1, 6- = 0]
        }
        """,
        questions=[
            "Evaluate this technical response on accuracy, completeness, clarity, and overall quality",
            "Evaluate this customer service response on accuracy, completeness, clarity, and overall quality",
            "Evaluate this educational response on accuracy, completeness, clarity, and overall quality"
        ],
        short_description="Template with multiple evaluation criteria",
        long_description="This EvalSet uses a multi-criteria template that evaluates responses across several dimensions simultaneously: accuracy, completeness, clarity, and overall quality. Each dimension is rated on a scale of 1-10, with detailed justifications for each score." + " " * 100
    )
    
    # Extract the EvalSet ID
    multi_criteria_evalset_id = multi_criteria_evalset_result.get("evalset", {}).get("id")
    print(f"Multi-criteria EvalSet created with ID: {multi_criteria_evalset_id}")
    
    # Step 4: Create an EvalSet with a domain-specific template (e.g., code review)
    print("\n4. Creating domain-specific template EvalSet...")
    
    code_review_evalset_result = await manage_evalset_tool(
        action="create",
        name="Code Review Template",
        template="""
        Given this conversation about code:
        {{ conversation }}
        
        Please perform a code review evaluation based on the following question:
        {{ eval_question }}
        
        Evaluate the code-related response using these specific software engineering criteria.
        Consider aspects like correctness, efficiency, security, readability, and best practices.
        
        Return your evaluation in the following JSON format:
        {
            "correctness": {
                "rating": [0-2, where 0=poor, 1=acceptable, 2=excellent],
                "issues": ["List any correctness issues"] or []
            },
            "security": {
                "rating": [0-2, where 0=poor, 1=acceptable, 2=excellent],
                "issues": ["List any security vulnerabilities"] or []
            },
            "performance": {
                "rating": [0-2, where 0=poor, 1=acceptable, 2=excellent],
                "issues": ["List any performance concerns"] or []
            },
            "readability": {
                "rating": [0-2, where 0=poor, 1=acceptable, 2=excellent],
                "issues": ["List any readability issues"] or []
            },
            "judgment": [0 or 1 - based on whether the code meets minimum quality standards]
        }
        """,
        questions=[
            "Evaluate the quality of the Python code provided in the assistant's response",
            "Evaluate the SQL query implementation in the assistant's response",
            "Evaluate the API design pattern described in the assistant's response",
            "Evaluate the error handling approach in the assistant's response"
        ],
        short_description="Specialized template for code review evaluations",
        long_description="This EvalSet uses a domain-specific template designed for evaluating code-related responses. It applies software engineering criteria including correctness, security, performance, and readability. Each dimension is rated on a scale with specific issues identified, making it ideal for evaluating coding assistance." + " " * 100
    )
    
    # Extract the EvalSet ID
    code_review_evalset_id = code_review_evalset_result.get("evalset", {}).get("id")
    print(f"Code review EvalSet created with ID: {code_review_evalset_id}")
    
    # Step 5: Define a conversation to evaluate
    print("\n5. Defining a test conversation...")
    
    # Sample conversation about a SQL query
    conversation = [
        {"role": "system", "content": "You are a helpful coding assistant."},
        {"role": "user", "content": "I need to write a SQL query that joins three tables: users, orders, and products. I want to find all users who have ordered a specific product category in the last 30 days, along with the total amount they've spent. Can you help me write this query?"},
        {"role": "assistant", "content": "I'd be happy to help you write that SQL query. Here's a solution that joins the three tables and filters for orders in the last 30 days for a specific product category:\n\n```sql\nSELECT \n    u.user_id,\n    u.username,\n    u.email,\n    SUM(o.amount) AS total_spent\nFROM \n    users u\nJOIN \n    orders o ON u.user_id = o.user_id\nJOIN \n    products p ON o.product_id = p.product_id\nWHERE \n    p.category = 'your_category_here'\n    AND o.order_date >= DATE_SUB(CURRENT_DATE, INTERVAL 30 DAY)\nGROUP BY \n    u.user_id, u.username, u.email\nORDER BY \n    total_spent DESC;\n```\n\nThis query:\n\n1. Joins the users, orders, and products tables\n2. Filters for orders with products in the specific category you want\n3. Filters for orders placed in the last 30 days\n4. Calculates the total amount spent by each user\n5. Groups the results by user\n6. Orders the results by the total amount spent (highest first)\n\nYou'll need to replace 'your_category_here' with the actual category you're looking for. Also, depending on your database system, the date function might be different - I used MySQL's DATE_SUB function, but Oracle, SQL Server, and PostgreSQL have their own equivalents.\n\nLet me know if you need any modifications to this query!"}
    ]
    
    print("Created a test conversation about SQL query implementation")
    
    # Step 6: Evaluate using each template type
    print("\n6. Evaluating the conversation with each template type...")
    
    # Evaluate with standard template
    print("\na) Using standard yes/no template...")
    standard_results = await run_evalset_tool(
        evalset_id=standard_evalset_id,
        conversation=conversation,
        model="meta-llama-3.1-8b-instruct",
        max_parallel=3
    )
    
    # Evaluate with Likert scale template
    print("\nb) Using Likert scale template...")
    likert_results = await run_evalset_tool(
        evalset_id=likert_evalset_id,
        conversation=conversation,
        model="meta-llama-3.1-8b-instruct",
        max_parallel=3
    )
    
    # Evaluate with multi-criteria template
    print("\nc) Using multi-criteria template...")
    multi_criteria_results = await run_evalset_tool(
        evalset_id=multi_criteria_evalset_id,
        conversation=conversation,
        model="meta-llama-3.1-8b-instruct",
        max_parallel=3
    )
    
    # Evaluate with code review template
    print("\nd) Using code review template...")
    code_review_results = await run_evalset_tool(
        evalset_id=code_review_evalset_id,
        conversation=conversation,
        model="meta-llama-3.1-8b-instruct",
        max_parallel=3
    )
    
    print("\nAll evaluations completed!")
    
    # Step 7: Compare and analyze results
    print("\n7. Comparing results from different templates:")
    
    # Standard template results
    print("\nA. Standard Template Results:")
    print(f"Overall score: {standard_results['summary']['yes_percentage']:.1f}%")
    print("Individual judgments:")
    for item in standard_results["results"]:
        judgment = "Yes" if item["judgment"] else "No"
        print(f"- {item['question']}: {judgment}")
    
    # Likert scale template results
    print("\nB. Likert Scale Template Results:")
    print(f"Overall score: {likert_results['summary']['yes_percentage']:.1f}%")
    print("Detailed ratings:")
    for item in likert_results["results"]:
        judgment = "Yes" if item["judgment"] else "No"
        rating = item.get("raw_result", {}).get("rating", "N/A")
        explanation = item.get("raw_result", {}).get("explanation", "No explanation provided")
        print(f"- {item['question']}:")
        print(f"  Rating: {rating}/5")
        print(f"  Binary judgment: {judgment}")
        print(f"  Explanation: {explanation}")
    
    # Multi-criteria template results
    print("\nC. Multi-Criteria Template Results:")
    print(f"Overall score: {multi_criteria_results['summary']['yes_percentage']:.1f}%")
    print("Multidimensional evaluation:")
    for item in multi_criteria_results["results"]:
        raw_result = item.get("raw_result", {})
        overall = raw_result.get("overall", {})
        overall_score = overall.get("score", "N/A")
        judgment = "Yes" if item["judgment"] else "No"
        
        print(f"- {item['question']}:")
        print(f"  Overall score: {overall_score}/10 (Binary judgment: {judgment})")
        
        if "accuracy" in raw_result:
            print(f"  Accuracy: {raw_result['accuracy'].get('score', 'N/A')}/10")
        if "completeness" in raw_result:
            print(f"  Completeness: {raw_result['completeness'].get('score', 'N/A')}/10")
        if "clarity" in raw_result:
            print(f"  Clarity: {raw_result['clarity'].get('score', 'N/A')}/10")
    
    # Code review template results
    print("\nD. Code Review Template Results:")
    print(f"Overall score: {code_review_results['summary']['yes_percentage']:.1f}%")
    print("Code quality assessment:")
    for item in code_review_results["results"]:
        raw_result = item.get("raw_result", {})
        judgment = "Pass" if item["judgment"] else "Fail"
        
        print(f"- {item['question']}:")
        print(f"  Overall judgment: {judgment}")
        
        if "correctness" in raw_result:
            correctness = raw_result["correctness"]
            rating = ["Poor", "Acceptable", "Excellent"][correctness.get("rating", 0)]
            print(f"  Correctness: {rating}")
            if correctness.get("issues"):
                print(f"    Issues: {', '.join(correctness['issues'])}")
        
        if "security" in raw_result:
            security = raw_result["security"]
            rating = ["Poor", "Acceptable", "Excellent"][security.get("rating", 0)]
            print(f"  Security: {rating}")
            if security.get("issues"):
                print(f"    Issues: {', '.join(security['issues'])}")
        
        if "performance" in raw_result:
            performance = raw_result["performance"]
            rating = ["Poor", "Acceptable", "Excellent"][performance.get("rating", 0)]
            print(f"  Performance: {rating}")
            if performance.get("issues"):
                print(f"    Issues: {', '.join(performance['issues'])}")
        
        if "readability" in raw_result:
            readability = raw_result["readability"]
            rating = ["Poor", "Acceptable", "Excellent"][readability.get("rating", 0)]
            print(f"  Readability: {rating}")
            if readability.get("issues"):
                print(f"    Issues: {', '.join(readability['issues'])}")
    
    # Step 8: Provide template customization guidelines
    print("\n8. Template Customization Guidelines:")
    
    print("\nWhen designing custom templates for your evaluations, consider these best practices:")
    
    print("\n1. Template Structure:")
    print("   - Always include {{ conversation }} and {{ eval_question }} variables")
    print("   - Provide clear instructions for the evaluator model")
    print("   - Specify the exact output format expected")
    
    print("\n2. Output Format:")
    print("   - Always include a 'judgment' field with 0 or 1 value")
    print("   - Structure additional data in a consistent JSON format")
    print("   - Consider adding explanation/justification fields")
    
    print("\n3. Evaluation Dimensions:")
    print("   - Tailor dimensions to your specific domain")
    print("   - Balance granularity vs. simplicity")
    print("   - Ensure dimensions are clearly defined")
    
    print("\n4. Domain-Specific Considerations:")
    print("   - For code: include security, correctness, efficiency dimensions")
    print("   - For customer service: include empathy, solution quality dimensions")
    print("   - For content: include accuracy, clarity, completeness dimensions")
    
    print("\n5. Testing Templates:")
    print("   - Always test new templates with representative conversations")
    print("   - Verify that raw_result fields contain expected data")
    print("   - Check that judgment values align with overall quality")
    
    print("\nThese examples demonstrate the flexibility of AgentOptim's template system.")
    print("You can create specialized evaluation frameworks for any domain or use case!")


if __name__ == "__main__":
    asyncio.run(main())