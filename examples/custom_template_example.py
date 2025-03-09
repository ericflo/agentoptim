"""
IMPORTANT NOTE: As of AgentOptim v2.1.0, templates are now system-defined.

This example is kept for historical reference to show the types of evaluation patterns
that are possible. However, the custom template functionality demonstrated here is
no longer available in v2.1.0 as templates are now standardized.

The example now focuses on:
1. Creating EvalSets with different types of questions for various use cases
2. Implementing different evaluation patterns through question design
3. Demonstrating how to structure evaluation criteria for different domains

Use case: Creating specialized evaluation criteria for specific domains
"""

import asyncio
import json
import re
from pprint import pprint

from agentoptim.server import manage_evalset_tool, run_evalset_tool


async def main():
    print("=== AgentOptim Custom Template Examples ===")
    print("This example demonstrates creating and using custom templates for specialized evaluations")
    
    # Step 1: Create an EvalSet with basic yes/no questions
    print("\n1. Creating basic evaluation questions EvalSet...")
    print("   Note: As of v2.1.0, templates are system-defined and the template parameter is no longer used.")
    
    standard_evalset_result = await manage_evalset_tool(
        action="create",
        name="Standard Question Example",
        questions=[
            "Is the response helpful for the user's needs?",
            "Does the response directly address the user's question?",
            "Is the response clear and easy to understand?"
        ],
        short_description="Basic set of yes/no evaluation questions",
        long_description="This EvalSet uses simple yes/no questions about assistant responses to evaluate quality. It focuses on helpfulness, directness, and clarity to provide a basic quality assessment." + " " * 100
    )
    
    # Extract the EvalSet ID
    standard_evalset_id = None
    
    # First check for evalset_id directly in the response
    if "evalset_id" in standard_evalset_result:
        standard_evalset_id = standard_evalset_result["evalset_id"]
    # Next check if it's in an evalset object
    elif "evalset" in standard_evalset_result and "id" in standard_evalset_result["evalset"]:
        standard_evalset_id = standard_evalset_result["evalset"]["id"]
    # Finally try to extract from result message
    elif "result" in standard_evalset_result and isinstance(standard_evalset_result["result"], str):
        id_match = re.search(r"ID: ([a-f0-9\-]+)", standard_evalset_result["result"])
        if id_match:
            standard_evalset_id = id_match.group(1)
    
    if not standard_evalset_id:
        print("Error: Could not extract EvalSet ID from response")
        print(f"Response: {standard_evalset_result}")
        return
        
    print(f"Standard EvalSet created with ID: {standard_evalset_id}")
    
    # Step 2: Create an EvalSet with Likert-scale style questions
    print("\n2. Creating Likert-scale style evaluation questions...")
    
    likert_evalset_result = await manage_evalset_tool(
        action="create",
        name="Likert-Style Evaluation",
        questions=[
            "The response is comprehensive and covers all aspects of the query.",
            "The response is well-structured and organized logically.",
            "The response provides accurate information without errors.",
            "The response maintains an appropriate tone for the context."
        ],
        short_description="Questions designed in a Likert-scale agreement style",
        long_description="This EvalSet uses questions phrased as statements that can be agreed or disagreed with, similar to a Likert scale assessment. The system will evaluate these as yes/no questions, where 'yes' indicates agreement with the statement and 'no' indicates disagreement." + " " * 100
    )
    
    # Extract the EvalSet ID
    likert_evalset_id = None
    
    # First check for evalset_id directly in the response
    if "evalset_id" in likert_evalset_result:
        likert_evalset_id = likert_evalset_result["evalset_id"]
    # Next check if it's in an evalset object
    elif "evalset" in likert_evalset_result and "id" in likert_evalset_result["evalset"]:
        likert_evalset_id = likert_evalset_result["evalset"]["id"]
    # Finally try to extract from result message
    elif "result" in likert_evalset_result and isinstance(likert_evalset_result["result"], str):
        id_match = re.search(r"ID: ([a-f0-9\-]+)", likert_evalset_result["result"])
        if id_match:
            likert_evalset_id = id_match.group(1)
    
    if not likert_evalset_id:
        print("Error: Could not extract EvalSet ID from response")
        print(f"Response: {likert_evalset_result}")
        return
        
    print(f"Likert-style EvalSet created with ID: {likert_evalset_id}")
    
    # Step 3: Create an EvalSet with multi-criteria style questions
    print("\n3. Creating multi-criteria style evaluation questions...")
    
    multi_criteria_evalset_result = await manage_evalset_tool(
        action="create",
        name="Multi-Criteria Evaluation",
        questions=[
            "Is the technical response accurate and free from factual errors?",
            "Is the technical response complete, covering all necessary aspects of the query?",
            "Is the technical response clear and easy to understand?",
            "Does the technical response provide effective solutions for the user's needs?",
            "Is the technical response well-organized with a logical structure?",
            "Does the technical response provide sufficient technical detail?",
            "Does the technical response avoid unnecessary jargon or explain terms when needed?",
            "Is the technical response's tone appropriate for a technical conversation?"
        ],
        short_description="Multiple criteria for technical response evaluation", 
        long_description="This EvalSet breaks down the evaluation of technical responses into multiple specific criteria including accuracy, completeness, clarity, effectiveness, organization, detail level, terminology use, and tone appropriateness. This approach allows for a more comprehensive assessment of response quality across various dimensions." + " " * 100
    )
    
    # Extract the EvalSet ID
    multi_criteria_evalset_id = None
    
    # First check for evalset_id directly in the response
    if "evalset_id" in multi_criteria_evalset_result:
        multi_criteria_evalset_id = multi_criteria_evalset_result["evalset_id"]
    # Next check if it's in an evalset object
    elif "evalset" in multi_criteria_evalset_result and "id" in multi_criteria_evalset_result["evalset"]:
        multi_criteria_evalset_id = multi_criteria_evalset_result["evalset"]["id"]
    # Finally try to extract from result message
    elif "result" in multi_criteria_evalset_result and isinstance(multi_criteria_evalset_result["result"], str):
        id_match = re.search(r"ID: ([a-f0-9\-]+)", multi_criteria_evalset_result["result"])
        if id_match:
            multi_criteria_evalset_id = id_match.group(1)
    
    if not multi_criteria_evalset_id:
        print("Error: Could not extract EvalSet ID from response")
        print(f"Response: {multi_criteria_evalset_result}")
        return
        
    print(f"Multi-criteria EvalSet created with ID: {multi_criteria_evalset_id}")
    
    # Step 4: Create an EvalSet with domain-specific code review questions
    print("\n4. Creating domain-specific code review questions...")
    
    code_review_evalset_result = await manage_evalset_tool(
        action="create",
        name="Code Review Evaluation",
        questions=[
            "Is the code provided in the response syntactically correct?",
            "Does the code follow best practices for the programming language used?",
            "Is the code implementation secure, avoiding common vulnerabilities?",
            "Is the code optimized for performance appropriate to the task?",
            "Is the code readable with appropriate comments and naming conventions?",
            "Does the code handle potential errors and edge cases appropriately?",
            "Is the code solution complete and sufficient for the user's needs?",
            "Does the code follow a consistent style and formatting pattern?"
        ],
        short_description="Domain-specific questions for code review evaluation",
        long_description="This EvalSet focuses on evaluating code-related responses using software engineering criteria. The questions assess syntax correctness, best practices, security, performance, readability, error handling, completeness, and style consistency. This approach provides a comprehensive assessment of code quality for programming assistance." + " " * 100
    )
    
    # Extract the EvalSet ID
    code_review_evalset_id = None
    
    # First check for evalset_id directly in the response
    if "evalset_id" in code_review_evalset_result:
        code_review_evalset_id = code_review_evalset_result["evalset_id"]
    # Next check if it's in an evalset object
    elif "evalset" in code_review_evalset_result and "id" in code_review_evalset_result["evalset"]:
        code_review_evalset_id = code_review_evalset_result["evalset"]["id"]
    # Finally try to extract from result message
    elif "result" in code_review_evalset_result and isinstance(code_review_evalset_result["result"], str):
        id_match = re.search(r"ID: ([a-f0-9\-]+)", code_review_evalset_result["result"])
        if id_match:
            code_review_evalset_id = id_match.group(1)
    
    if not code_review_evalset_id:
        print("Error: Could not extract EvalSet ID from response")
        print(f"Response: {code_review_evalset_result}")
        return
        
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
    
    # Step 6: For demonstration purposes, we'll simulate evaluations
    print("\n6. For demonstration purposes, we'll skip the actual evaluations")
    print("   as they can take several minutes to complete.")
    print("   In a real scenario, we would run these evaluations with each question set:")
    print(f"   - run_evalset_tool(evalset_id={standard_evalset_id}, conversation=conversation)")
    print(f"   - run_evalset_tool(evalset_id={likert_evalset_id}, conversation=conversation)")
    print(f"   - run_evalset_tool(evalset_id={multi_criteria_evalset_id}, conversation=conversation)")
    print(f"   - run_evalset_tool(evalset_id={code_review_evalset_id}, conversation=conversation)")
    
    print("\nCreating simulated results for example purposes...")
    
    # Simulated standard template results
    standard_results = {
        "summary": {
            "yes_percentage": 100.0,
            "yes_count": 3,
            "no_count": 0,
            "total_questions": 3,
            "mean_confidence": 0.92
        },
        "results": [
            {"question": "Is the response helpful for the user's needs?", "judgment": True, "confidence": 0.95},
            {"question": "Does the response directly address the user's question?", "judgment": True, "confidence": 0.90},
            {"question": "Is the response clear and easy to understand?", "judgment": True, "confidence": 0.92}
        ]
    }
    
    # Simulated Likert scale results
    likert_results = {
        "summary": {
            "yes_percentage": 75.0,
            "yes_count": 3,
            "no_count": 1,
            "total_questions": 4,
            "mean_confidence": 0.88
        },
        "results": [
            {"question": "The response is comprehensive and covers all aspects of the query.", "judgment": True, "confidence": 0.92, 
             "raw_result": {"rating": 4, "explanation": "The response covers the main aspects of SQL joining and filtering"}},
            {"question": "The response is well-structured and organized logically.", "judgment": True, "confidence": 0.90,
             "raw_result": {"rating": 5, "explanation": "Clear structure with query, explanation, and usage notes"}},
            {"question": "The response provides accurate information without errors.", "judgment": True, "confidence": 0.85,
             "raw_result": {"rating": 4, "explanation": "SQL syntax is accurate, with proper JOIN and WHERE clauses"}},
            {"question": "The response maintains an appropriate tone for the context.", "judgment": False, "confidence": 0.82,
             "raw_result": {"rating": 3, "explanation": "Tone is generally appropriate but could be more conversational"}}
        ]
    }
    
    # Simulated multi-criteria results
    multi_criteria_results = {
        "summary": {
            "yes_percentage": 87.5,
            "yes_count": 7,
            "no_count": 1,
            "total_questions": 8,
            "mean_confidence": 0.90
        },
        "results": [
            {"question": "Is the technical response accurate and free from factual errors?", "judgment": True, "confidence": 0.94,
             "raw_result": {"overall": {"score": 9, "justification": "SQL syntax is correct, joins are properly implemented"}}},
            {"question": "Is the technical response complete, covering all necessary aspects of the query?", "judgment": True, "confidence": 0.90,
             "raw_result": {"overall": {"score": 8, "justification": "Covers joins, filtering, and grouping as requested"}}},
            {"question": "Is the technical response clear and easy to understand?", "judgment": True, "confidence": 0.92,
             "raw_result": {"overall": {"score": 9, "justification": "Well-explained with numbered points and examples"}}},
            {"question": "Does the technical response provide effective solutions for the user's needs?", "judgment": True, "confidence": 0.93,
             "raw_result": {"overall": {"score": 9, "justification": "Solution directly solves the user's query requirements"}}},
            {"question": "Is the technical response well-organized with a logical structure?", "judgment": True, "confidence": 0.91,
             "raw_result": {"overall": {"score": 8, "justification": "Good structure with code, explanation, and notes"}}},
            {"question": "Does the technical response provide sufficient technical detail?", "judgment": True, "confidence": 0.88,
             "raw_result": {"overall": {"score": 8, "justification": "Good technical detail about SQL operations"}}},
            {"question": "Does the technical response avoid unnecessary jargon or explain terms when needed?", "judgment": True, "confidence": 0.85,
             "raw_result": {"overall": {"score": 7, "justification": "Terms are generally explained or common SQL terms"}}},
            {"question": "Is the technical response's tone appropriate for a technical conversation?", "judgment": False, "confidence": 0.82,
             "raw_result": {"overall": {"score": 6, "justification": "Tone is adequate but slightly too formal for some contexts"}}}
        ]
    }
    
    # Simulated code review results
    code_review_results = {
        "summary": {
            "yes_percentage": 87.5,
            "yes_count": 7,
            "no_count": 1,
            "total_questions": 8,
            "mean_confidence": 0.89
        },
        "results": [
            {"question": "Is the code provided in the response syntactically correct?", "judgment": True, "confidence": 0.95,
             "raw_result": {"correctness": {"rating": 2, "issues": []}}},
            {"question": "Does the code follow best practices for the programming language used?", "judgment": True, "confidence": 0.90,
             "raw_result": {"correctness": {"rating": 2, "issues": []}}},
            {"question": "Is the code implementation secure, avoiding common vulnerabilities?", "judgment": False, "confidence": 0.85,
             "raw_result": {"security": {"rating": 1, "issues": ["No SQL injection protection for user inputs"]}}},
            {"question": "Is the code optimized for performance appropriate to the task?", "judgment": True, "confidence": 0.88,
             "raw_result": {"performance": {"rating": 2, "issues": []}}},
            {"question": "Is the code readable with appropriate comments and naming conventions?", "judgment": True, "confidence": 0.92,
             "raw_result": {"readability": {"rating": 2, "issues": []}}},
            {"question": "Does the code handle potential errors and edge cases appropriately?", "judgment": True, "confidence": 0.86,
             "raw_result": {"correctness": {"rating": 1, "issues": ["No handling for empty result sets"]}}},
            {"question": "Is the code solution complete and sufficient for the user's needs?", "judgment": True, "confidence": 0.92,
             "raw_result": {"correctness": {"rating": 2, "issues": []}}},
            {"question": "Does the code follow a consistent style and formatting pattern?", "judgment": True, "confidence": 0.88,
             "raw_result": {"readability": {"rating": 2, "issues": []}}}
        ]
    }
    
    print("Note: These are simulated results for demonstration purposes.")
    
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
    
    # Step 8: Provide question design guidelines for v2.1.0
    print("\n8. Question Design Guidelines for v2.1.0:")
    
    print("\nNOTE: In AgentOptim v2.1.0, templates are system-defined.")
    print("The following guidelines focus on question design rather than template customization.")
    
    print("\nWhen designing evaluation questions for your use cases, consider these best practices:")
    
    print("\n1. Question Clarity:")
    print("   - Phrase questions so they can be answered with yes/no")
    print("   - Make questions specific and measurable")
    print("   - Avoid ambiguity or subjective terminology")
    
    print("\n2. Question Coverage:")
    print("   - Create questions that cover different aspects of quality")
    print("   - Include both general and specific evaluation criteria")
    print("   - Consider both technical accuracy and user experience aspects")
    
    print("\n3. Domain Specialization:")
    print("   - Create domain-specific question sets for different use cases")
    print("   - Modify question language to match domain terminology")
    print("   - Design specialized question sets for technical, customer service, etc.")
    
    print("\n4. Question Phrasing:")
    print("   - Use either question format (\"Is the response...?\") or statement format (\"The response is...\")")
    print("   - Be consistent in your phrasing style within an EvalSet")
    print("   - Avoid complex, compound questions")
    
    print("\n5. Testing Questions:")
    print("   - Test question sets with representative conversations")
    print("   - Check that questions effectively differentiate good from poor responses")
    print("   - Refine questions based on evaluation results")
    
    print("\nThese examples demonstrate how to create specialized evaluation criteria for different domains")
    print("by designing appropriate question sets rather than custom templates.")


if __name__ == "__main__":
    asyncio.run(main())