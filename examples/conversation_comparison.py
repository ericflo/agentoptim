"""
Example of using AgentOptim to compare different conversation approaches.

This example demonstrates how to:
1. Create an EvalSet focused on conversational style/tone
2. Define multiple conversation approaches for the same user query
3. Compare the effectiveness of different response styles
4. Analyze the results to identify the most effective approach

Use case: Comparing formal vs. casual vs. technical response styles
"""

import asyncio
import json
from pprint import pprint
import matplotlib.pyplot as plt
import numpy as np

from agentoptim import manage_evalset_tool, run_evalset_tool


async def main():
    print("=== AgentOptim Conversation Style Comparison ===")
    print("This example compares different conversation styles for technical assistance")
    
    # Step 1: Create an EvalSet focused on conversation style
    print("\n1. Creating style-focused EvalSet...")
    
    evalset_result = await manage_evalset_tool(
        action="create",
        name="Conversation Style Evaluation",
        questions=[
            "Does the response directly address the user's question?",
            "Is the response clear and easy to understand?",
            "Does the response use appropriate tone for the context?",
            "Does the response feel natural and conversational?",
            "Does the response avoid unnecessary jargon or complexity?",
            "Would this response build rapport with the user?",
            "Is the response concise without omitting important information?",
            "Would this response make the user feel respected and valued?"
        ],
        short_description="Evaluates response style and tone effectiveness",
        long_description="This EvalSet focuses on the stylistic and tonal aspects of conversation responses, evaluating how different approaches might impact user experience and satisfaction. It measures clarity, tone appropriateness, conversational quality, and rapport-building potential. Use it to compare different response styles and identify the approach that resonates best with users in specific contexts." + " " * 100
    )
    
    # Extract the EvalSet ID
    evalset_id = evalset_result.get("evalset", {}).get("id")
    print(f"EvalSet created with ID: {evalset_id}")
    
    # Step 2: Define user query and different response approaches
    print("\n2. Defining conversation approaches...")
    
    user_query = "Can you explain how to configure a VPN connection on my computer? I'm not very technical."
    
    # Style 1: Formal/Professional
    formal_conversation = [
        {"role": "system", "content": "You are a professional technical support assistant who provides clear, accurate, and formal guidance."},
        {"role": "user", "content": user_query},
        {"role": "assistant", "content": "I'd be pleased to guide you through configuring a VPN connection on your computer. A Virtual Private Network (VPN) creates a secure connection between your device and another network.\n\nPlease follow these instructions:\n\n1. Navigate to your computer's Settings or System Preferences\n2. Select 'Network' or 'Network & Internet'\n3. Locate the VPN section (often labeled 'VPN' or 'Add a connection')\n4. Click 'Add a VPN connection' or similar option\n5. Enter the details provided by your VPN service provider, including:\n   - Connection name\n   - Server address\n   - VPN type (often L2TP/IPsec, PPTP, or OpenVPN)\n   - Username and password\n6. Click 'Save' or 'Connect'\n\nShould you encounter any difficulties during this process, please don't hesitate to provide more specific information about your operating system for more tailored instructions."}
    ]
    
    # Style 2: Casual/Friendly
    casual_conversation = [
        {"role": "system", "content": "You are a friendly, approachable technical helper who uses casual language and explains things simply."},
        {"role": "user", "content": user_query},
        {"role": "assistant", "content": "Hey there! No worries if you're not super tech-savvy - setting up a VPN is actually pretty simple! 😊\n\nHere's how to do it in easy steps:\n\n1. First, go to your computer's Settings (look for the gear icon)\n2. Click on 'Network' or 'Internet' (the name might be a bit different depending on your computer)\n3. Look for 'VPN' - it's usually easy to spot\n4. Click on 'Add a VPN' or 'Set up a new connection'\n5. Now you'll need to put in some info from your VPN provider:\n   - Give it a name (like 'My VPN')\n   - The server address (it'll look something like vpn.company.com)\n   - Your username and password\n6. Click Save or Connect and you're good to go!\n\nIf you get stuck, just let me know what kind of computer you're using (Mac, Windows, etc.) and I can give you more specific steps! 👍"}
    ]
    
    # Style 3: Technical/Detailed
    technical_conversation = [
        {"role": "system", "content": "You are a technical expert who provides detailed, precise information with technical terminology."},
        {"role": "user", "content": user_query},
        {"role": "assistant", "content": "To configure a VPN connection on your system, follow this protocol-specific implementation procedure:\n\n1. Access your system's network configuration interface via Control Panel (Windows), System Preferences (macOS), or Settings (Linux/mobile OS)\n\n2. Locate the network connection manager and select 'Add VPN Connection' or equivalent option\n\n3. Select the appropriate protocol from the available options:\n   - OpenVPN (UDP/TCP): Most secure, uses ports 1194/443\n   - IKEv2/IPSec: Good for mobile connections, automatic reconnection\n   - WireGuard: Modern protocol with enhanced performance\n   - L2TP/IPSec: Legacy protocol with PSK or certificate authentication\n\n4. Input the server parameters:\n   - Server hostname/IP (e.g., vpn.domain.tld or 203.0.113.1)\n   - Authentication credentials: username/password or certificate\n   - Encryption parameters: cipher suite, key length, hashing algorithm\n   - Split tunneling configuration (if applicable)\n\n5. Configure advanced options as needed:\n   - DNS settings (specify DNS servers or use provider's)\n   - MTU optimization (typically 1400-1500)\n   - NAT traversal settings\n\n6. Save configuration and initiate connection\n\nThe connection will establish an encrypted tunnel over port 1194 (OpenVPN UDP) or alternate ports depending on protocol, encapsulating all traffic according to the specified routing table."}
    ]
    
    # Style 4: Empathetic/Supportive
    empathetic_conversation = [
        {"role": "system", "content": "You are a supportive assistant who emphasizes with users' challenges and provides encouragement."},
        {"role": "user", "content": user_query},
        {"role": "assistant", "content": "I completely understand how VPN setup can feel intimidating if you're not comfortable with technical tasks. Many people find this challenging at first, so you're definitely not alone in feeling that way.\n\nI'll walk you through this step-by-step, and we'll take it nice and slow:\n\n1. Start by opening your computer's Settings - that's the gear or cog icon you can find in your start menu or system tray\n\n2. Look for the Network section - it might be called 'Network & Internet' or something similar\n\n3. Find the VPN option - it's usually in the left sidebar or in a list of connection types\n\n4. Click on 'Add a VPN connection' - don't worry, we'll go through each field together\n\n5. Now you'll need to enter some information from your VPN provider:\n   - A name for the connection (something you'll recognize)\n   - The server address (this comes from your VPN provider)\n   - Your username and password\n\n6. Click Save, and you've done it!\n\nIt's perfectly okay if you need to try this a couple of times - technology can be tricky! If you get stuck at any point or see anything confusing, just let me know exactly what you're seeing, and I'll help you figure it out. You're doing great just by taking this step to set up your VPN!"}
    ]
    
    print("Defined 4 different conversation styles for the same user query:")
    print("1. Formal/Professional")
    print("2. Casual/Friendly")
    print("3. Technical/Detailed")
    print("4. Empathetic/Supportive")
    
    # Step 3: Evaluate each conversation style
    print("\n3. Evaluating conversation styles...")
    
    print("\na) Evaluating formal style...")
    formal_results = await run_evalset_tool(
        evalset_id=evalset_id,
        conversation=formal_conversation,
        # Note: Model is set via environment variable
        # AGENTOPTIM_JUDGE_MODEL can be set before starting the server
        max_parallel=3
    )
    
    print("\nb) Evaluating casual style...")
    casual_results = await run_evalset_tool(
        evalset_id=evalset_id,
        conversation=casual_conversation,
        # Note: Model is set via environment variable
        # AGENTOPTIM_JUDGE_MODEL can be set before starting the server
        max_parallel=3
    )
    
    print("\nc) Evaluating technical style...")
    technical_results = await run_evalset_tool(
        evalset_id=evalset_id,
        conversation=technical_conversation,
        # Note: Model is set via environment variable
        # AGENTOPTIM_JUDGE_MODEL can be set before starting the server
        max_parallel=3
    )
    
    print("\nd) Evaluating empathetic style...")
    empathetic_results = await run_evalset_tool(
        evalset_id=evalset_id,
        conversation=empathetic_conversation,
        # Note: Model is set via environment variable
        # AGENTOPTIM_JUDGE_MODEL can be set before starting the server
        max_parallel=3
    )
    
    print("\nAll evaluations completed!")
    
    # Step 4: Compare results
    print("\n4. Comparing results across conversation styles:")
    
    styles = ["Formal", "Casual", "Technical", "Empathetic"]
    scores = [
        formal_results["summary"]["yes_percentage"],
        casual_results["summary"]["yes_percentage"],
        technical_results["summary"]["yes_percentage"],
        empathetic_results["summary"]["yes_percentage"]
    ]
    
    # Print detailed comparison table
    print("\nStyle Comparison Results:")
    print("-" * 50)
    print(f"{'Style':<12} | {'Score':<8} | {'Yes':<4} | {'No':<4} | {'Questions':<4}")
    print("-" * 50)
    
    for i, style in enumerate(styles):
        if style == "Formal":
            results = formal_results
        elif style == "Casual":
            results = casual_results
        elif style == "Technical":
            results = technical_results
        else:
            results = empathetic_results
            
        yes = results["summary"]["yes_count"]
        no = results["summary"]["no_count"]
        total = results["summary"]["total_questions"]
        score = results["summary"]["yes_percentage"]
        
        print(f"{style:<12} | {score:>6.1f}% | {yes:>4} | {no:>4} | {total:>4}")
    
    print("-" * 50)
    
    # Identify best performing style
    best_index = scores.index(max(scores))
    best_style = styles[best_index]
    
    print(f"\nBest performing style: {best_style} ({scores[best_index]:.1f}%)")
    
    # Step 5: Detailed analysis by criterion
    print("\n5. Detailed analysis by evaluation criterion:")
    
    # Get all questions
    questions = [q["question"] for q in formal_results["results"]]
    
    # Create dictionary to track performance by question
    question_results = {q: [] for q in questions}
    
    # Gather results by question for each style
    all_results = [formal_results, casual_results, technical_results, empathetic_results]
    for i, results in enumerate(all_results):
        for item in results["results"]:
            question = item["question"]
            judgment = 100 if item["judgment"] else 0  # Convert to percentage for visualization
            question_results[question].append((styles[i], judgment))
    
    # Print detailed criterion analysis
    for question, style_results in question_results.items():
        print(f"\nCriterion: {question}")
        for style, judgment in style_results:
            status = "✅ Yes" if judgment == 100 else "❌ No"
            print(f"  {style:<12}: {status}")
    
    # Step 6: Generate recommendations
    print("\n6. Recommendations based on analysis:")
    
    # General recommendations based on highest scoring style
    print(f"\nBased on the evaluation, the {best_style} style performed best overall.")
    
    # Specific recommendations by use case
    print("\nRecommendations for this specific user query (VPN setup for non-technical user):")
    
    if best_style == "Formal":
        print("- Maintain professional tone while avoiding overly technical language")
        print("- Include clear step-by-step instructions")
        print("- Provide just enough technical context without overwhelming")
        print("- Offer additional assistance for follow-up questions")
    elif best_style == "Casual":
        print("- Use approachable, friendly language to put the user at ease")
        print("- Include emoji and conversational elements to reduce anxiety")
        print("- Simplify technical concepts but don't omit important steps")
        print("- End with an offer to provide more specific help if needed")
    elif best_style == "Technical":
        print("- Focus on detailed technical explanations for advanced users")
        print("- Use precise terminology and protocol information")
        print("- Include technical context and underlying concepts")
        print("- Recognize this approach may overwhelm non-technical users")
    else:  # Empathetic
        print("- Acknowledge the user's discomfort with technical tasks")
        print("- Provide reassurance and encouragement throughout")
        print("- Focus on clear instructions with minimal technical jargon")
        print("- Normalize challenges and emphasize availability for follow-up")
    
    print("\nFor best results, consider combining elements from multiple styles:")
    print("- Clear step-by-step instructions (from formal style)")
    print("- Approachable language (from casual style)")
    print("- Reassurance for non-technical users (from empathetic style)")
    print("- Essential technical details only when necessary (from technical style)")


if __name__ == "__main__":
    asyncio.run(main())