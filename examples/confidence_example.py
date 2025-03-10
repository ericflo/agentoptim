"""
Example demonstrating the use of AgentOptim's verbalized confidence score elicitation.

This example shows how to use different confidence elicitation methods to
get confidence scores from language models, and how to evaluate the quality
of these scores.

Based on research, the most effective methods are:
1. combo_exemplars (default): Uses few-shot examples and uncertainty factors
2. advanced_probability: Instructs model to consider specific uncertainty sources
3. basic_float: Simpler approach that works well with more capable models
"""

from agentoptim.confidence import (
    BasicPercentageMethod,
    BasicFloatMethod,
    BasicLetterMethod,
    BasicTextMethod,
    AdvancedProbabilityMethod,
    ComboExemplarsMethod,
    MultiGuessMethod,
    expected_calibration_error,
    CONFIDENCE_METHODS,
)

# Sample data: a list of questions to ask the model
SAMPLE_QUESTIONS = [
    "What is the capital of France?",
    "Who wrote 'Pride and Prejudice'?",
    "What is the square root of 144?",
    "What is the boiling point of water in Celsius?",
    "Who was the first person to step on the moon?",
]

def simulate_model_response(question, method):
    """
    Simulate an LLM's response with a confidence score.
    In a real application, you would call your actual LLM API here.
    """
    # This is just for demo purposes - real models would give the actual answers
    model_answers = {
        "What is the capital of France?": "Paris",
        "Who wrote 'Pride and Prejudice'?": "Jane Austen",
        "What is the square root of 144?": "12",
        "What is the boiling point of water in Celsius?": "100",
        "Who was the first person to step on the moon?": "Neil Armstrong",
    }
    
    # Simulate confidences (again, just for demo)
    model_confidences = {
        "What is the capital of France?": 0.95,
        "Who wrote 'Pride and Prejudice'?": 0.87,
        "What is the square root of 144?": 0.99,
        "What is the boiling point of water in Celsius?": 0.93,
        "Who was the first person to step on the moon?": 0.82,
    }
    
    answer = model_answers.get(question, "I don't know")
    confidence = model_confidences.get(question, 0.5)
    
    # Format the response according to the method's expected format
    if isinstance(method, BasicPercentageMethod):
        return f"Answer: {answer}\nConfidence: {int(confidence * 100)}%"
    elif isinstance(method, BasicFloatMethod):
        return f"Answer: {answer}\nConfidence: {confidence:.2f}"
    elif isinstance(method, BasicLetterMethod):
        # Map confidence to letter grade
        letter = 'A' if confidence > 0.8 else 'B' if confidence > 0.6 else 'C'
        return f"Answer: {answer}\nConfidence: {letter}"
    elif isinstance(method, BasicTextMethod):
        # Map confidence to text description
        text = 'very high' if confidence > 0.9 else 'high' if confidence > 0.7 else 'medium'
        return f"Answer: {answer}\nConfidence: {text}"
    elif isinstance(method, AdvancedProbabilityMethod):
        return f"Answer: {answer}\nProbability: {confidence:.2f}"
    elif isinstance(method, ComboExemplarsMethod):
        return f"Guess: {answer}\nProbability: {confidence:.2f}"
    elif isinstance(method, MultiGuessMethod):
        # For simplicity, just provide one guess in the multi-guess format
        return f"G1: {answer}\nP1: {confidence:.2f}"
    else:
        return f"Answer: {answer}\nConfidence: {confidence:.2f}"

def main():
    """
    Main function demonstrating the confidence elicitation methods.
    """
    print("AgentOptim Verbalized Confidence Score Elicitation Example\n")
    
    # Display information about recommended methods
    print("RECOMMENDED METHODS BASED ON RESEARCH:")
    print("1. combo_exemplars (Default) - Uses few-shot examples with probabilities")
    print("2. advanced_probability - Explicitly considers uncertainty sources")
    print("3. basic_float - Simple approach for capable models")
    
    # Try each confidence elicitation method
    for method_name, method in CONFIDENCE_METHODS.items():
        print(f"\n{'-'*50}")
        is_recommended = ""
        if method_name in ["combo_exemplars", "advanced_probability", "basic_float"]:
            is_recommended = " [RECOMMENDED]"
        print(f"Demonstrating: {method_name}{is_recommended}")
        print(f"{'-'*50}")
        
        results = []
        
        for question in SAMPLE_QUESTIONS:
            # 1. Generate prompt with confidence elicitation instructions
            prompt = method.generate_prompt({"question": question})
            
            # 2. In a real application, you would send this prompt to your LLM
            # Here we just simulate a response
            response = simulate_model_response(question, method)
            
            # 3. Extract answer and confidence
            answer, confidence = method.extract_answer_and_confidence(response)
            
            # 4. Classify the response
            classification = method.classify_response(response, answer, confidence)
            
            print(f"\nQuestion: {question}")
            print(f"Response: {response}")
            print(f"Extracted: Answer='{answer}', Confidence={confidence}")
            print(f"Classification: {classification}")
            
            # In a real application, you would compare with ground truth
            # Here we just assume all our demo answers are correct
            results.append((1, confidence))  # (is_correct, confidence)
        
        # In a real application, you could evaluate the calibration of confidences
        if results:
            y_true = [r[0] for r in results]
            y_pred = [r[1] for r in results if r[1] is not None]
            
            if y_pred:
                ece = expected_calibration_error(y_true, y_pred)
                print(f"\nExpected Calibration Error: {ece:.4f}")
                print("(Lower is better, 0 means perfectly calibrated)")

if __name__ == "__main__":
    main()