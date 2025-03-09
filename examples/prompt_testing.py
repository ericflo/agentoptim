"""
Example of using AgentOptim to test different system prompts.

This example demonstrates how to:
1. Create an EvalSet focused on specific performance criteria
2. Test multiple system prompts with the same user query
3. Compare the effectiveness of different prompt strategies
4. Analyze the results to identify the most effective prompt design

Use case: Testing different system prompts for a customer service assistant
"""

import asyncio
import json
from pprint import pprint

from agentoptim import manage_evalset_tool, run_evalset_tool


async def main():
    print("=== AgentOptim System Prompt Testing ===")
    print("This example compares different system prompts for a customer service assistant")
    
    # Step 1: Create an EvalSet focused on customer service quality
    print("\n1. Creating customer service EvalSet...")
    
    evalset_result = await manage_evalset_tool(
        action="create",
        name="Customer Service Quality",
        questions=[
            "Does the response address the customer's concern directly?",
            "Is the response professional and polite?",
            "Does the response provide a clear and actionable solution?",
            "Does the response show empathy for the customer's situation?",
            "Is the response concise without omitting important information?",
            "Does the response offer additional help or follow-up?",
            "Is the response free of unnecessary jargon or complexity?",
            "Does the response follow company policies while still being helpful?"
        ],
        short_description="Evaluates customer service response quality",
        long_description="This EvalSet measures the effectiveness of customer service responses based on key criteria including clarity, professionalism, solution orientation, empathy, and adherence to service standards. It can be used to compare different response approaches and identify the most effective system prompts for customer service situations requiring a balance between policy adherence and customer satisfaction." + " " * 100
    )
    
    # Extract the EvalSet ID
    evalset_id = evalset_result.get("evalset", {}).get("id")
    print(f"EvalSet created with ID: {evalset_id}")
    
    # Step 2: Define user query and different system prompts
    print("\n2. Defining test prompts...")
    
    user_query = "I ordered a jacket two weeks ago and it still hasn't arrived. The tracking hasn't updated in 5 days. I need this for an event this weekend!"
    
    # Prompt 1: Basic professional prompt
    basic_prompt = "You are a helpful customer service assistant."
    
    # Prompt 2: Detailed role instruction prompt
    detailed_prompt = """You are a customer service assistant for a clothing retailer. 
You should be professional, helpful, and empathetic. 
Your goal is to solve customer problems efficiently while ensuring customer satisfaction.
Always provide clear next steps and resolutions when possible."""
    
    # Prompt 3: Persona-based prompt
    persona_prompt = """You are Jamie, a senior customer service specialist with 10 years of experience.
You're known for your empathy, problem-solving skills, and ability to turn frustrated customers into loyal ones.
You pride yourself on finding solutions even in difficult situations.
You speak professionally but warmly, and always look for ways to exceed customer expectations."""
    
    # Prompt 4: Principles-based prompt
    principles_prompt = """You are a customer service assistant guided by these principles:
1. Customer First: The customer's needs are your top priority
2. Ownership: Take responsibility for resolving issues completely  
3. Empathy: Acknowledge feelings and frustrations genuinely
4. Clarity: Provide clear, actionable information
5. Efficiency: Resolve issues with minimal steps for the customer
6. Above & Beyond: Find opportunities to exceed expectations

Apply these principles to every customer interaction."""
    
    # Prompt 5: Contextual constraint prompt
    constraints_prompt = """You are a customer service assistant for a clothing company.
Company policies:
- Shipping normally takes 7-10 business days
- Expedited shipping is available for issues with delays
- Refunds are processed within 3-5 business days
- Customers can cancel orders only if not yet shipped
- Exchanges are allowed within 30 days of delivery

Balance policy adherence with customer satisfaction, using your judgment to make exceptions when appropriate to maintain customer satisfaction."""
    
    # Create conversations with different system prompts
    basic_conversation = [
        {"role": "system", "content": basic_prompt},
        {"role": "user", "content": user_query},
        {"role": "assistant", "content": "I apologize for the delay with your jacket order. I understand you need this for an event this weekend, which adds urgency to the situation. Let me help you with this.\n\nThe lack of tracking updates for 5 days is concerning. I'd like to look into this immediately for you. To assist you better, I'll need your order number so I can check the status in our system.\n\nIn the meantime, I can offer a few options:\n\n1. We can expedite your shipping to ensure you receive it before your event\n2. We can locate a similar item at a store near you for pickup\n3. If we determine your package won't arrive in time, we can discuss compensation options\n\nPlease provide your order number, and I'll investigate this right away."}
    ]
    
    detailed_conversation = [
        {"role": "system", "content": detailed_prompt},
        {"role": "user", "content": user_query},
        {"role": "assistant", "content": "I'm very sorry to hear about the delay with your jacket order. I understand this is particularly stressful since you need it for an upcoming event this weekend.\n\nThe 5-day gap in tracking updates is definitely a concern. Let me help resolve this for you right away.\n\nHere's what I'll do:\n\n1. I'd like to access your order details to check its current status. Could you please provide your order number?\n\n2. Once I have this information, I can:\n   - Contact our shipping department for an urgent status update\n   - Arrange for expedited shipping to ensure delivery before your event\n   - If needed, locate a similar jacket at a store near you for immediate pickup\n\n3. If we determine there's no way to get your original order to you in time, I can also look into alternative solutions like finding a replacement at a local store.\n\nPlease share your order number, and I'll start working on this immediately to make sure you have what you need for your event."}
    ]
    
    persona_conversation = [
        {"role": "system", "content": persona_prompt},
        {"role": "user", "content": user_query},
        {"role": "assistant", "content": "I completely understand your frustration - waiting two weeks without updates would stress anyone out, especially when you have an important event coming up! Don't worry, I've helped many customers in similar situations, and we'll make sure you're taken care of.\n\nThat 5-day gap in tracking updates definitely needs immediate attention. Here's my plan to help you:\n\nFirst, I'd like to get your order number so I can look at exactly what's happening behind the scenes with your jacket. Sometimes packages are moving but the tracking isn't updating properly.\n\nMore importantly, since you need this for an event this weekend, we have several options to ensure you're not left without your jacket:\n\n1. I can flag this for our expedited resolution team who can potentially arrange priority shipping\n2. We might be able to locate the same jacket at a store near you for immediate pickup\n3. If neither option works, I can check if we can ship an identical item from a different warehouse with overnight delivery\n\nMay I have your order number to get started? I'll personally keep track of this situation until it's resolved for you."}
    ]
    
    principles_conversation = [
        {"role": "system", "content": principles_prompt},
        {"role": "user", "content": user_query},
        {"role": "assistant", "content": "I'm truly sorry about the situation with your jacket order. I understand the urgency given your upcoming event this weekend, and I'm committed to resolving this for you.\n\nThe lack of tracking updates for 5 days is concerning, and I take full ownership of investigating this for you right away. Your frustration is completely valid, and I appreciate you bringing this to our attention.\n\nHere's how I'll help you:\n\n1. First, I need to access the details of your order. Could you please provide your order number?\n\n2. Once I have that information, I'll immediately check the actual status of your package and why the tracking hasn't updated.\n\n3. Given your time constraint, I'll explore multiple solutions to ensure you have your jacket for the weekend, including:\n   - Expediting shipping on your current order if it can arrive in time\n   - Arranging for a replacement to be sent with priority shipping\n   - Checking if the same jacket is available at a store near you for immediate pickup\n\n4. As a gesture of goodwill for the inconvenience, I'd also like to offer you a discount on your next purchase.\n\nPlease share your order number, and I'll act on this immediately. I'm dedicated to finding you a solution that works for your timeline."}
    ]
    
    constraints_conversation = [
        {"role": "system", "content": constraints_prompt},
        {"role": "user", "content": user_query},
        {"role": "assistant", "content": "I'm sorry to hear about the delay with your jacket order. I understand your concern, especially since you need it for an event this weekend.\n\nWhile our standard shipping typically takes 7-10 business days, the fact that there's been no tracking update for 5 days is unusual and warrants immediate attention. Given your situation, I'd like to help find a resolution that works for you.\n\nCould you please provide your order number so I can look into this for you? Once I have that information, I can:\n\n1. Check the exact status of your order in our system\n2. Arrange for expedited shipping at no additional cost to you, as this falls under our policy for shipping delays\n3. If the jacket cannot arrive in time for your event, I can help explore alternatives such as:\n   - Locating the same jacket at a store near you for immediate pickup if available\n   - Offering a temporary solution until your order arrives\n\nOur goal is to ensure you have what you need for your event this weekend. Please provide your order number, and I'll start working on this right away."}
    ]
    
    print("Defined 5 different system prompts to test:")
    print("1. Basic professional prompt")
    print("2. Detailed role instruction prompt")
    print("3. Persona-based prompt")
    print("4. Principles-based prompt")
    print("5. Contextual constraint prompt")
    
    # Step 3: Evaluate each prompt
    print("\n3. Evaluating prompt effectiveness...")
    
    print("\na) Evaluating basic prompt...")
    basic_results = await run_evalset_tool(
        evalset_id=evalset_id,
        conversation=basic_conversation,
        model="meta-llama-3.1-8b-instruct",
        max_parallel=3
    )
    
    print("\nb) Evaluating detailed role prompt...")
    detailed_results = await run_evalset_tool(
        evalset_id=evalset_id,
        conversation=detailed_conversation,
        model="meta-llama-3.1-8b-instruct",
        max_parallel=3
    )
    
    print("\nc) Evaluating persona prompt...")
    persona_results = await run_evalset_tool(
        evalset_id=evalset_id,
        conversation=persona_conversation,
        model="meta-llama-3.1-8b-instruct",
        max_parallel=3
    )
    
    print("\nd) Evaluating principles prompt...")
    principles_results = await run_evalset_tool(
        evalset_id=evalset_id,
        conversation=principles_conversation,
        model="meta-llama-3.1-8b-instruct",
        max_parallel=3
    )
    
    print("\ne) Evaluating constraints prompt...")
    constraints_results = await run_evalset_tool(
        evalset_id=evalset_id,
        conversation=constraints_conversation,
        model="meta-llama-3.1-8b-instruct",
        max_parallel=3
    )
    
    print("\nAll evaluations completed!")
    
    # Step 4: Compare results
    print("\n4. Comparing results across different system prompts:")
    
    prompts = ["Basic", "Detailed", "Persona", "Principles", "Constraints"]
    scores = [
        basic_results["summary"]["yes_percentage"],
        detailed_results["summary"]["yes_percentage"],
        persona_results["summary"]["yes_percentage"],
        principles_results["summary"]["yes_percentage"],
        constraints_results["summary"]["yes_percentage"]
    ]
    
    # Print comparison table
    print("\nPrompt Comparison Results:")
    print("-" * 63)
    print(f"{'Prompt Type':<15} | {'Score':<8} | {'Yes':<4} | {'No':<4} | {'Questions':<4} | {'Description':<20}")
    print("-" * 63)
    
    for i, prompt in enumerate(prompts):
        if prompt == "Basic":
            results = basic_results
            desc = "Simple instruction"
        elif prompt == "Detailed":
            results = detailed_results
            desc = "Detailed role guide"
        elif prompt == "Persona":
            results = persona_results
            desc = "Character-based"
        elif prompt == "Principles":
            results = principles_results
            desc = "Values-oriented"
        else:
            results = constraints_results
            desc = "Policy constraints"
            
        yes = results["summary"]["yes_count"]
        no = results["summary"]["no_count"]
        total = results["summary"]["total_questions"]
        score = results["summary"]["yes_percentage"]
        
        print(f"{prompt:<15} | {score:>6.1f}% | {yes:>4} | {no:>4} | {total:>4} | {desc:<20}")
    
    print("-" * 63)
    
    # Identify best performing prompt
    best_index = scores.index(max(scores))
    best_prompt = prompts[best_index]
    
    print(f"\nBest performing prompt: {best_prompt} ({scores[best_index]:.1f}%)")
    
    # Step 5: Detailed analysis by criterion
    print("\n5. Detailed analysis by evaluation criterion:")
    
    # Get all questions
    questions = [q["question"] for q in basic_results["results"]]
    
    # Create dictionary to track performance by question
    question_results = {q: [] for q in questions}
    
    # Gather results by question for each prompt
    all_results = [basic_results, detailed_results, persona_results, principles_results, constraints_results]
    for i, results in enumerate(all_results):
        for item in results["results"]:
            question = item["question"]
            judgment = 1 if item["judgment"] else 0
            question_results[question].append((prompts[i], judgment))
    
    # Print detailed criterion analysis
    for question, prompt_results in question_results.items():
        print(f"\nCriterion: {question}")
        for prompt, judgment in prompt_results:
            status = "✅ Yes" if judgment else "❌ No"
            print(f"  {prompt:<15}: {status}")
    
    # Step 6: Generate recommendations
    print("\n6. Recommendations based on analysis:")
    
    # Strengths of each prompt type
    print("\nStrengths observed by prompt type:")
    
    print("\nBasic prompt:")
    print("- Simplicity allows the model to apply general helpfulness")
    print("- Less restrictive, giving the model flexibility to respond")
    
    print("\nDetailed role prompt:")
    print("- Provides clear context about the assistant's role")
    print("- Outlines specific objectives that guide response quality")
    
    print("\nPersona-based prompt:")
    print("- Creates consistent character voice and perspective")
    print("- Embeds experience and expertise signals")
    
    print("\nPrinciples-based prompt:")
    print("- Structures responses around core values")
    print("- Helps ensure consistent application of service standards")
    
    print("\nConstraints-based prompt:")
    print("- Incorporates policy knowledge directly in the system prompt")
    print("- Helps balance policy adherence with customer satisfaction")
    
    # Recommendation for this specific use case
    print(f"\nFor customer service in this specific scenario, the {best_prompt} prompt performed best.")
    
    # General prompt design recommendations
    print("\nRecommendations for effective system prompt design:")
    print("1. Incorporate elements of multiple prompt types for best results")
    print("2. Include clear role definition to establish context")
    print("3. Provide specific guiding principles that reflect service values")
    print("4. Add relevant policy constraints when they impact responses")
    print("5. Test and iterate on prompts with representative user queries")
    
    # Specific recommendations
    print("\nBased on analysis, an optimal system prompt might look like:")
    print("```")
    print("You are a customer service specialist for a clothing retailer with years of experience.")
    print("Core principles:")
    print("- Put customer needs first while working within company policies")
    print("- Take ownership of issues and see them through to resolution")
    print("- Show genuine empathy for customer concerns")
    print("- Provide clear, actionable solutions")
    print("")
    print("Key policies to know:")
    print("- Standard shipping: 7-10 business days")
    print("- Expedited shipping available for delays")
    print("- Return window: 30 days from delivery")
    print("```")


if __name__ == "__main__":
    asyncio.run(main())