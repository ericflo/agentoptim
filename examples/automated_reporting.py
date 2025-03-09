"""
Example of using AgentOptim to generate evaluation reports automatically.

This example demonstrates how to:
1. Create an EvalSet for a specific domain
2. Evaluate conversations and capture results
3. Generate comprehensive reports in multiple formats
4. Create visualizations to highlight key findings

Use case: Generating automated quality reports for stakeholder review
"""

import asyncio
import json
import os
import datetime
import markdown
from pprint import pprint
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import base64

from agentoptim import manage_evalset_tool, run_evalset_tool


async def main():
    print("=== AgentOptim Automated Reporting ===")
    print("This example demonstrates generating quality evaluation reports")
    
    # Step 1: Create an EvalSet for evaluation
    print("\n1. Creating detailed evaluation EvalSet...")
    
    evalset_result = await manage_evalset_tool(
        action="create",
        name="Knowledge Base Response Quality",
        questions=[
            "Is the response factually accurate?",
            "Does the response directly answer the user's question?",
            "Does the response provide sufficient detail?",
            "Is the response well-structured and organized?",
            "Does the response cite relevant sources when needed?",
            "Is the information up-to-date and current?",
            "Does the response avoid unnecessary technical jargon?",
            "Is the tone appropriate for a knowledge base article?",
            "Does the response anticipate related questions?"
        ],
        short_description="Evaluates knowledge base response quality",
        long_description="This EvalSet measures the quality of knowledge base responses across factual accuracy, completeness, structure, citations, and appropriateness. It helps ensure knowledge base content meets high standards of quality and usefulness for users seeking information." + " " * 100
    )
    
    # Extract the EvalSet ID
    evalset_id = evalset_result.get("evalset", {}).get("id")
    print(f"EvalSet created with ID: {evalset_id}")
    
    # Step 2: Define sample conversations to evaluate
    print("\n2. Defining test conversations...")
    
    # Sample questions about programming languages
    conversations = {
        "Python": [
            {"role": "system", "content": "You are a helpful and accurate knowledge base assistant."},
            {"role": "user", "content": "What are Python generators and how do they work?"},
            {"role": "assistant", "content": "Python generators are special functions that allow you to create iterators in an elegant way. Unlike regular functions that return a single value, generators yield a sequence of values one at a time, suspending their execution between yields.\n\nHere's how generators work:\n\n1. When you call a generator function, it doesn't execute the function body immediately. Instead, it returns a generator object, which is an iterator.\n\n2. The generator's execution starts when you call `next()` on the generator object (or implicitly in a for loop).\n\n3. When a `yield` statement is reached, the function's state is frozen, and the yielded value is returned.\n\n4. When `next()` is called again, execution continues from where it left off, with all local variables preserved.\n\n5. This continues until the function exits or reaches a `return` statement, which raises a StopIteration exception.\n\nGenerators are memory-efficient because they generate values on-the-fly instead of creating the entire sequence in memory. This makes them ideal for working with large datasets or infinite sequences.\n\nExample of a generator function:\n```python\ndef count_up_to(max):\n    count = 1\n    while count <= max:\n        yield count\n        count += 1\n\n# Using the generator\nfor number in count_up_to(5):\n    print(number)  # Prints 1, 2, 3, 4, 5\n```\n\nPython also supports generator expressions, which are similar to list comprehensions but use parentheses instead of square brackets:\n```python\ngen = (x**2 for x in range(10))  # Generator expression\n```\n\nGenerators were introduced in Python 2.2 (PEP 255) and have become a fundamental feature for efficient data processing in Python."}
        ],
        "JavaScript": [
            {"role": "system", "content": "You are a helpful and accurate knowledge base assistant."},
            {"role": "user", "content": "Explain JavaScript promises and async/await."},
            {"role": "assistant", "content": "# JavaScript Promises and Async/Await\n\nPromises and async/await are JavaScript features for handling asynchronous operations more elegantly.\n\n## Promises\n\nA Promise is an object representing the eventual completion (or failure) of an asynchronous operation and its resulting value. A Promise is in one of these states:\n\n- **Pending**: Initial state, neither fulfilled nor rejected\n- **Fulfilled**: The operation completed successfully\n- **Rejected**: The operation failed\n\n### Creating a Promise\n\n```javascript\nconst myPromise = new Promise((resolve, reject) => {\n  // Asynchronous operation\n  if (/* operation successful */) {\n    resolve(value); // Fulfilled with a value\n  } else {\n    reject(error);  // Rejected with an error\n  }\n});\n```\n\n### Using Promises\n\n```javascript\nmyPromise\n  .then(value => {\n    // Handle success\n    return transformedValue; // Chain promises\n  })\n  .catch(error => {\n    // Handle errors\n  })\n  .finally(() => {\n    // Executes regardless of success or failure\n  });\n```\n\n## Async/Await\n\nAsync/await is syntactic sugar over Promises, making asynchronous code look and behave more like synchronous code.\n\n### Basic Usage\n\n```javascript\nasync function fetchData() {\n  try {\n    // The await keyword pauses execution until the Promise resolves\n    const response = await fetch('https://api.example.com/data');\n    const data = await response.json();\n    return data;\n  } catch (error) {\n    console.error('Error fetching data:', error);\n  }\n}\n```\n\n### Key Points\n\n1. Functions with the `async` keyword always return a Promise\n2. The `await` keyword can only be used inside `async` functions\n3. `await` pauses the execution until the Promise settles\n4. Use try/catch blocks for error handling\n\n## Comparison\n\n**Promise Chains**:\n```javascript\nfetchUser(userId)\n  .then(user => fetchUserPosts(user))\n  .then(posts => displayPosts(posts))\n  .catch(error => handleError(error));\n```\n\n**Async/Await**:\n```javascript\nasync function showUserPosts(userId) {\n  try {\n    const user = await fetchUser(userId);\n    const posts = await fetchUserPosts(user);\n    displayPosts(posts);\n  } catch (error) {\n    handleError(error);\n  }\n}\n```\n\nAsync/await was introduced in ES2017 (ES8) and is now widely supported in modern browsers and Node.js."}
        ],
        "SQL": [
            {"role": "system", "content": "You are a helpful and accurate knowledge base assistant."},
            {"role": "user", "content": "What's the difference between INNER JOIN and LEFT JOIN in SQL?"},
            {"role": "assistant", "content": "INNER JOIN and LEFT JOIN are SQL join operations that combine rows from two tables, but they behave differently when handling records that don't have matches.\n\nHere's a simple comparison table:\n\n| Feature | INNER JOIN | LEFT JOIN |\n|---------|------------|----------|\n| Matching records | Included | Included |\n| Non-matching records from left table | Excluded | Included |\n| Non-matching records from right table | Excluded | Excluded |\n| Use case | When you need only records with matches in both tables | When you need all records from the left table regardless of matches |\n\n## INNER JOIN\n\nAn INNER JOIN returns only the rows where there is a match in both tables based on the join condition.\n\n### Example\n\nConsider two tables: `Customers` and `Orders`.\n\n```sql\nSELECT Customers.CustomerName, Orders.OrderID\nFROM Customers\nINNER JOIN Orders ON Customers.CustomerID = Orders.CustomerID;\n```\n\nThis query will return only customers who have placed at least one order. Customers without orders are excluded from the results.\n\n## LEFT JOIN\n\nA LEFT JOIN (also called LEFT OUTER JOIN) returns all rows from the left table and the matching rows from the right table. If there's no match in the right table, NULL values are returned for right table columns.\n\n### Example\n\nUsing the same tables:\n\n```sql\nSELECT Customers.CustomerName, Orders.OrderID\nFROM Customers\nLEFT JOIN Orders ON Customers.CustomerID = Orders.CustomerID;\n```\n\nThis query will return all customers, regardless of whether they've placed orders. For customers without orders, the OrderID will be NULL.\n\n## Visual Representation\n\nImagine two tables as circles in a Venn diagram:\n- INNER JOIN gives you the intersection (only the overlapping part)\n- LEFT JOIN gives you the entire left circle plus the intersection\n\n## When to Use Each\n\n- Use INNER JOIN when you only want results with matching values in both tables\n- Use LEFT JOIN when you want all records from the left table, even if there are no matches in the right table\n\nFor example:\n- INNER JOIN: \"Show me customers and their orders\" (only customers who have orders)\n- LEFT JOIN: \"Show me all customers and their orders, if any\" (all customers, even those without orders)"}
        ],
        "Rust": [
            {"role": "system", "content": "You are a helpful and accurate knowledge base assistant."},
            {"role": "user", "content": "How does Rust's ownership system work?"},
            {"role": "assistant", "content": "# Rust's Ownership System\n\nRust's ownership system is a unique memory management approach that ensures memory safety without a garbage collector. This is one of Rust's most distinctive features and enables it to prevent common programming errors like null pointer dereferencing, dangling pointers, and data races at compile time.\n\n## Key Concepts\n\n### 1. Ownership Rules\n\nThese form the foundation of Rust's memory management:\n\n- Each value in Rust has a variable that's its **owner**\n- There can only be **one owner** at a time\n- When the owner goes out of scope, the value is **dropped** (memory is freed)\n\n```rust\n{                      // s is not valid here, it's not yet declared
    let s = String::from(\"hello\");   // s is valid from this point forward
    // do stuff with s
}                      // this scope is now over, and s is no longer valid
```\n\n### 2. Move Semantics\n\nWhen you assign a variable to another variable, Rust **moves** the value instead of copying it (for heap-allocated types):\n\n```rust\nlet s1 = String::from(\"hello\");\nlet s2 = s1; // s1 is moved to s2, s1 is no longer valid\n\n// This would cause a compile error:\n// println!(\"{}\", s1);\n```\n\n### 3. Borrowing\n\nInstead of transferring ownership, you can **borrow** a value using references:\n\n- **Immutable references** (`&T`) - Multiple allowed at the same time\n- **Mutable references** (`&mut T`) - Only one allowed at a time\n\n```rust\nfn calculate_length(s: &String) -> usize { // s is a reference to a String\n    s.len()\n} // s goes out of scope, but it doesn't have ownership, so nothing happens\n\nlet s1 = String::from(\"hello\");\nlet len = calculate_length(&s1); // passing a reference to s1\n// s1 is still valid here\n```\n\n### 4. Borrowing Rules\n\n- At any given time, you can have **either**:\n  - One mutable reference (`&mut T`)\n  - Any number of immutable references (`&T`)\n- References must always be valid (no dangling references)\n\n```rust\nlet mut s = String::from(\"hello\");\n\nlet r1 = &s; // no problem\nlet r2 = &s; // no problem\n// let r3 = &mut s; // BIG PROBLEM - can't have mutable and immutable references together\n\nprintln!(\"{} and {}\", r1, r2);\n// r1 and r2 are no longer used after this point\n\nlet r3 = &mut s; // no problem - r1 and r2 are no longer being used\nprintln!(\"{}\", r3);\n```\n\n### 5. The Slice Type\n\nSlices are references to a contiguous sequence of elements, allowing you to reference a part of a collection without taking ownership:\n\n```rust\nlet s = String::from(\"hello world\");\nlet hello = &s[0..5]; // slice referencing the first 5 bytes of s\nlet world = &s[6..11]; // slice referencing bytes 6 through 10 of s\n```\n\n## Benefits of Ownership\n\n1. **Memory safety without garbage collection**: No runtime performance cost\n2. **No data races**: The compiler prevents concurrent access to the same data\n3. **No null/dangling pointers**: References are always valid\n4. **No double-free errors**: Memory is automatically freed exactly once\n5. **No memory leaks** (in safe Rust): All memory is accounted for\n\n## Advanced Ownership Patterns\n\n- **Lifetimes**: Advanced feature that relates to how long references are valid\n- **Smart pointers** (`Box<T>`, `Rc<T>`, `Arc<T>`): Types that extend ownership capabilities\n- **Interior mutability** (`RefCell<T>`, `Cell<T>`): Allow mutation through immutable references in controlled ways\n\nRust's ownership system may seem strict at first, but it prevents entire categories of bugs at compile time, resulting in more reliable, secure, and concurrent code."}
        ]
    }
    
    print(f"Defined {len(conversations)} knowledge base articles to evaluate")
    
    # Step 3: Evaluate each conversation
    print("\n3. Evaluating knowledge base articles...")
    
    results = {}
    for topic, conversation in conversations.items():
        print(f"\nEvaluating {topic} article...")
        eval_result = await run_evalset_tool(
            evalset_id=evalset_id,
            conversation=conversation,
            model="meta-llama-3.1-8b-instruct",
            max_parallel=3
        )
        results[topic] = eval_result
    
    print("\nAll evaluations completed!")
    
    # Step 4: Generate report data
    print("\n4. Generating report data...")
    
    # Build report structure
    report_data = {
        "title": "Knowledge Base Quality Report",
        "date": datetime.datetime.now().strftime("%Y-%m-%d"),
        "summary": {
            "articles_evaluated": len(results),
            "overall_score": sum(r["summary"]["yes_percentage"] for r in results.values()) / len(results),
            "top_performer": max(results.keys(), key=lambda k: results[k]["summary"]["yes_percentage"]),
            "improvement_needed": min(results.keys(), key=lambda k: results[k]["summary"]["yes_percentage"])
        },
        "article_scores": {
            topic: {
                "score": result["summary"]["yes_percentage"],
                "yes_count": result["summary"]["yes_count"],
                "no_count": result["summary"]["no_count"],
                "total": result["summary"]["total_questions"]
            } for topic, result in results.items()
        },
        "criteria_analysis": {}
    }
    
    # Analyze performance by criterion
    criteria = [item["question"] for item in list(results.values())[0]["results"]]
    for criterion in criteria:
        criterion_results = {}
        for topic, result in results.items():
            for item in result["results"]:
                if item["question"] == criterion:
                    criterion_results[topic] = item["judgment"]
                    break
        
        pass_rate = sum(criterion_results.values()) / len(criterion_results) * 100
        report_data["criteria_analysis"][criterion] = {
            "pass_rate": pass_rate,
            "results": criterion_results
        }
    
    # Identify strengths and weaknesses
    report_data["strengths"] = []
    report_data["areas_for_improvement"] = []
    
    for criterion, analysis in report_data["criteria_analysis"].items():
        if analysis["pass_rate"] >= 75:
            report_data["strengths"].append(criterion)
        elif analysis["pass_rate"] <= 50:
            report_data["areas_for_improvement"].append(criterion)
    
    # Add recommendations
    report_data["recommendations"] = [
        "Focus on citing relevant sources in knowledge base articles",
        "Ensure all articles follow a consistent structure",
        "Add sections for related topics to anticipate user questions",
        "Review articles for technical jargon and simplify where possible",
        "Implement regular quality checks for all knowledge base content"
    ]
    
    print("Report data generated successfully")
    
    # Step 5: Create visualizations
    print("\n5. Creating visualizations...")
    
    # Function to create and save charts
    def create_charts():
        charts = {}
        
        # Article scores chart
        plt.figure(figsize=(10, 6))
        topics = list(report_data["article_scores"].keys())
        scores = [report_data["article_scores"][topic]["score"] for topic in topics]
        
        bars = plt.bar(topics, scores, color='skyblue')
        plt.axhline(y=report_data["summary"]["overall_score"], color='r', linestyle='-', label=f'Average ({report_data["summary"]["overall_score"]:.1f}%)')
        
        # Add score labels on top of bars
        for bar, score in zip(bars, scores):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f"{score:.1f}%", 
                    ha='center', va='bottom', fontweight='bold')
        
        plt.xlabel('Topics')
        plt.ylabel('Quality Score (%)')
        plt.title('Knowledge Base Article Quality by Topic')
        plt.ylim(0, 105)  # Give room for labels
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.legend()
        
        # Save chart
        plt.tight_layout()
        chart_path = "kb_article_scores.png"
        plt.savefig(chart_path)
        charts["article_scores"] = chart_path
        
        # Criteria performance chart
        plt.figure(figsize=(12, 8))
        criteria_labels = list(report_data["criteria_analysis"].keys())
        criteria_scores = [report_data["criteria_analysis"][c]["pass_rate"] for c in criteria_labels]
        
        # Sort criteria by score for better visualization
        sorted_data = sorted(zip(criteria_labels, criteria_scores), key=lambda x: x[1])
        sorted_labels = [item[0] for item in sorted_data]
        sorted_scores = [item[1] for item in sorted_data]
        
        # Shorten labels for better display
        short_labels = [label.replace("Does the response ", "").replace("Is the response ", "") for label in sorted_labels]
        
        # Create horizontal bar chart
        bars = plt.barh(short_labels, sorted_scores, color='lightgreen')
        
        # Add score labels
        for bar, score in zip(bars, sorted_scores):
            plt.text(min(score + 3, 95), bar.get_y() + bar.get_height()/2, f"{score:.1f}%", 
                    va='center', fontweight='bold')
        
        plt.xlabel('Pass Rate (%)')
        plt.title('Performance by Evaluation Criterion')
        plt.xlim(0, 105)  # Give room for labels
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        
        # Save chart
        plt.tight_layout()
        chart_path = "kb_criteria_performance.png"
        plt.savefig(chart_path)
        charts["criteria_performance"] = chart_path
        
        return charts
    
    # Try to create charts if matplotlib is available
    try:
        charts = create_charts()
        print(f"Created visualization charts: {', '.join(charts.values())}")
    except:
        print("Skipping visualization (matplotlib may not be available)")
        charts = {}
    
    # Step 6: Generate HTML report
    print("\n6. Generating HTML report...")
    
    # Function to encode images as base64 for embedding in HTML
    def image_to_base64(image_path):
        try:
            with open(image_path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                return f"data:image/png;base64,{encoded_string}"
        except:
            return ""
    
    # HTML Report Template
    html_report = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>{report_data["title"]}</title>
        <style>
            body {{ font-family: Arial, sans-serif; line-height: 1.6; max-width: 1200px; margin: 0 auto; padding: 20px; }}
            h1, h2, h3 {{ color: #2c3e50; }}
            .report-header {{ background-color: #f8f9fa; padding: 20px; border-radius: 5px; margin-bottom: 20px; }}
            .summary-box {{ display: flex; justify-content: space-between; margin: 20px 0; }}
            .summary-item {{ background-color: #e9ecef; padding: 15px; border-radius: 5px; width: 23%; text-align: center; }}
            .summary-item h3 {{ margin-top: 0; }}
            .chart-container {{ margin: 30px 0; }}
            table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
            th, td {{ padding: 12px 15px; text-align: left; border-bottom: 1px solid #ddd; }}
            th {{ background-color: #f8f9fa; }}
            tr:hover {{ background-color: #f1f1f1; }}
            .progress-bar {{ height: 20px; background-color: #e9ecef; border-radius: 5px; margin-top: 5px; }}
            .progress {{ height: 100%; background-color: #4CAF50; border-radius: 5px; }}
            .recommendations {{ background-color: #f8f9fa; padding: 20px; border-radius: 5px; margin: 20px 0; }}
            .good {{ color: #28a745; }}
            .bad {{ color: #dc3545; }}
        </style>
    </head>
    <body>
        <div class="report-header">
            <h1>{report_data["title"]}</h1>
            <p>Generated on: {report_data["date"]}</p>
        </div>
        
        <h2>Executive Summary</h2>
        <div class="summary-box">
            <div class="summary-item">
                <h3>Overall Score</h3>
                <p style="font-size: 24px; font-weight: bold;">{report_data["summary"]["overall_score"]:.1f}%</p>
            </div>
            <div class="summary-item">
                <h3>Articles Evaluated</h3>
                <p style="font-size: 24px; font-weight: bold;">{report_data["summary"]["articles_evaluated"]}</p>
            </div>
            <div class="summary-item">
                <h3>Top Performer</h3>
                <p style="font-size: 24px; font-weight: bold;">{report_data["summary"]["top_performer"]}</p>
                <p>{report_data["article_scores"][report_data["summary"]["top_performer"]]["score"]:.1f}%</p>
            </div>
            <div class="summary-item">
                <h3>Needs Improvement</h3>
                <p style="font-size: 24px; font-weight: bold;">{report_data["summary"]["improvement_needed"]}</p>
                <p>{report_data["article_scores"][report_data["summary"]["improvement_needed"]]["score"]:.1f}%</p>
            </div>
        </div>
        
        <div class="chart-container">
            <h2>Article Quality Scores</h2>
            {f'<img src="{image_to_base64(charts.get("article_scores", ""))}" alt="Article Scores Chart" style="max-width:100%;">' if "article_scores" in charts else '<p>Chart not available</p>'}
            
            <table>
                <tr>
                    <th>Topic</th>
                    <th>Score</th>
                    <th>Criteria Passed</th>
                    <th>Performance</th>
                </tr>
    """
    
    # Add article score rows
    for topic, data in sorted(report_data["article_scores"].items(), key=lambda x: x[1]["score"], reverse=True):
        score = data["score"]
        passed = data["yes_count"]
        total = data["total"]
        html_report += f"""
                <tr>
                    <td>{topic}</td>
                    <td>{score:.1f}%</td>
                    <td>{passed}/{total}</td>
                    <td>
                        <div class="progress-bar">
                            <div class="progress" style="width: {score}%;"></div>
                        </div>
                    </td>
                </tr>
        """
    
    html_report += """
            </table>
        </div>
        
        <div class="chart-container">
            <h2>Performance by Criterion</h2>
    """
    
    # Add criteria performance chart if available
    if "criteria_performance" in charts:
        html_report += f'<img src="{image_to_base64(charts.get("criteria_performance", ""))}" alt="Criteria Performance Chart" style="max-width:100%;">'
    
    html_report += """
            <table>
                <tr>
                    <th>Criterion</th>
                    <th>Pass Rate</th>
                    <th>Performance</th>
                    <th>Article Performance</th>
                </tr>
    """
    
    # Add criterion rows
    for criterion, data in sorted(report_data["criteria_analysis"].items(), key=lambda x: x[1]["pass_rate"], reverse=True):
        pass_rate = data["pass_rate"]
        html_report += f"""
                <tr>
                    <td>{criterion}</td>
                    <td>{pass_rate:.1f}%</td>
                    <td>
                        <div class="progress-bar">
                            <div class="progress" style="width: {pass_rate}%;"></div>
                        </div>
                    </td>
                    <td>
        """
        
        # Add article performance for this criterion
        for topic, passed in data["results"].items():
            status_class = "good" if passed else "bad"
            status_symbol = "✓" if passed else "✗"
            html_report += f'<span class="{status_class}">{topic}: {status_symbol}</span><br>'
        
        html_report += """
                    </td>
                </tr>
        """
    
    html_report += """
            </table>
        </div>
        
        <div class="recommendations">
            <h2>Key Findings</h2>
            
            <h3>Strengths</h3>
            <ul>
    """
    
    # Add strengths
    for strength in report_data["strengths"]:
        html_report += f"<li>{strength}</li>\n"
    
    html_report += """
            </ul>
            
            <h3>Areas for Improvement</h3>
            <ul>
    """
    
    # Add areas for improvement
    for area in report_data["areas_for_improvement"]:
        html_report += f"<li>{area}</li>\n"
    
    html_report += """
            </ul>
            
            <h3>Recommendations</h3>
            <ul>
    """
    
    # Add recommendations
    for recommendation in report_data["recommendations"]:
        html_report += f"<li>{recommendation}</li>\n"
    
    html_report += """
            </ul>
        </div>
        
        <div class="report-footer">
            <p><em>This report was automatically generated using AgentOptim v2.1.0.</em></p>
        </div>
    </body>
    </html>
    """
    
    # Save HTML report
    html_report_filename = "knowledge_base_quality_report.html"
    try:
        with open(html_report_filename, "w") as f:
            f.write(html_report)
        print(f"HTML report saved as: {html_report_filename}")
    except:
        print("Warning: Could not save HTML report")
    
    # Step 7: Generate Markdown report for simpler viewing
    print("\n7. Generating Markdown report...")
    
    # Markdown Report Template
    md_report = f"""# {report_data["title"]}

Generated on: {report_data["date"]}

## Executive Summary

- **Overall Score**: {report_data["summary"]["overall_score"]:.1f}%
- **Articles Evaluated**: {report_data["summary"]["articles_evaluated"]}
- **Top Performer**: {report_data["summary"]["top_performer"]} ({report_data["article_scores"][report_data["summary"]["top_performer"]]["score"]:.1f}%)
- **Needs Improvement**: {report_data["summary"]["improvement_needed"]} ({report_data["article_scores"][report_data["summary"]["improvement_needed"]]["score"]:.1f}%)

## Article Quality Scores

| Topic | Score | Criteria Passed |
|-------|-------|----------------|
"""
    
    # Add article score rows
    for topic, data in sorted(report_data["article_scores"].items(), key=lambda x: x[1]["score"], reverse=True):
        score = data["score"]
        passed = data["yes_count"]
        total = data["total"]
        md_report += f"| {topic} | {score:.1f}% | {passed}/{total} |\n"
    
    md_report += """
## Performance by Criterion

| Criterion | Pass Rate | Article Performance |
|-----------|-----------|---------------------|
"""
    
    # Add criterion rows
    for criterion, data in sorted(report_data["criteria_analysis"].items(), key=lambda x: x[1]["pass_rate"], reverse=True):
        pass_rate = data["pass_rate"]
        
        # Format article performance for this criterion
        article_results = []
        for topic, passed in data["results"].items():
            status_symbol = "✓" if passed else "✗"
            article_results.append(f"{topic}: {status_symbol}")
        
        article_performance = ", ".join(article_results)
        md_report += f"| {criterion} | {pass_rate:.1f}% | {article_performance} |\n"
    
    md_report += """
## Key Findings

### Strengths

"""
    
    # Add strengths
    for strength in report_data["strengths"]:
        md_report += f"- {strength}\n"
    
    md_report += """
### Areas for Improvement

"""
    
    # Add areas for improvement
    for area in report_data["areas_for_improvement"]:
        md_report += f"- {area}\n"
    
    md_report += """
### Recommendations

"""
    
    # Add recommendations
    for recommendation in report_data["recommendations"]:
        md_report += f"- {recommendation}\n"
    
    md_report += """

---

*This report was automatically generated using AgentOptim v2.1.0.*
"""
    
    # Save Markdown report
    md_report_filename = "knowledge_base_quality_report.md"
    try:
        with open(md_report_filename, "w") as f:
            f.write(md_report)
        print(f"Markdown report saved as: {md_report_filename}")
    except:
        print("Warning: Could not save Markdown report")
    
    # Step 8: Print report summary
    print("\n8. Report generation complete!")
    print("\nReport Summary:")
    print(f"- Overall quality score: {report_data['summary']['overall_score']:.1f}%")
    print(f"- Top performing topic: {report_data['summary']['top_performer']} ({report_data['article_scores'][report_data['summary']['top_performer']]['score']:.1f}%)")
    print(f"- Topic needing improvement: {report_data['summary']['improvement_needed']} ({report_data['article_scores'][report_data['summary']['improvement_needed']]['score']:.1f}%)")
    
    print("\nKey strengths:")
    for strength in report_data["strengths"][:3]:  # Top 3 strengths
        print(f"- {strength}")
    
    print("\nKey areas for improvement:")
    for area in report_data["areas_for_improvement"][:3]:  # Top 3 areas
        print(f"- {area}")
    
    print(f"\nFull reports available in {html_report_filename} and {md_report_filename}")


if __name__ == "__main__":
    asyncio.run(main())