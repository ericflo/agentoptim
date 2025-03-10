"""
Confidence score elicitation and evaluation module for AgentOptim.

This module implements various methods for eliciting confidence scores from
language models, as well as utilities for extracting, normalizing, and
evaluating these scores.
"""

import re
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union, Callable, Any

import numpy as np

logger = logging.getLogger(__name__)

# Constants
NO_ANSWER_TEXT = "NO ANSWER"
PROMPT_NO_ANSWER = f"If you cannot provide an answer, respond with `{NO_ANSWER_TEXT}`."

# Common regex patterns for extracting confidence scores
PATTERN_SEP = r"\n(.*\n)*?"
PATTERN_ANSWER = r".+"
PATTERN_FLOAT = r"\d*\.?\d+"

# Response classification categories
VALID_ANSWER = "valid_answer"
NO_ANSWER = "no_answer"
INVALID_ANSWER = "invalid_answer"


def extract_from_response(
    response: str, 
    patterns: List[re.Pattern], 
    names: Union[str, List[str]]
) -> Union[Optional[str], Tuple[Optional[str], ...]]:
    """
    Extract answer and/or confidence from a response using multiple regex patterns.
    
    Args:
        response: The model's response text to extract from
        patterns: List of compiled regex patterns to try
        names: Group name(s) to extract from the match
        
    Returns:
        Extracted string(s) or None if no match was found
    """
    # Remove formatting that might interfere with regex matches
    response = response.replace("**", "")
    response = response.replace("__", "")
    response = response.replace("```", "")
    
    # Try each pattern in sequence until one matches
    for pattern in patterns:
        match = pattern.search(response)
        if match is not None:
            if isinstance(names, str):
                return match.group(names)
            else:
                return tuple(match.group(name) for name in names)
    
    # Return None if no pattern matched
    if isinstance(names, str):
        return None
    else:
        return tuple(None for _ in names)


def normalize_confidence(
    confidence: Optional[str], 
    normalize_fn: Callable[[str], float] = float
) -> Optional[float]:
    """
    Normalize a confidence score string into a float between 0 and 1.
    
    Args:
        confidence: String representation of confidence score or None
        normalize_fn: Function to apply to the confidence string
        
    Returns:
        Normalized confidence score as float or None if input was None
    """
    if confidence is None:
        return None
    
    try:
        normalized = normalize_fn(confidence)
        # Ensure the confidence is in the valid range
        if normalized < 0 or normalized > 1:
            logger.warning(f"Confidence score {normalized} out of valid range [0,1]")
            if normalized < 0:
                normalized = 0.0
            elif normalized > 1:
                normalized = 1.0
        return normalized
    except ValueError:
        logger.warning(f"Failed to normalize confidence score: {confidence}")
        return None


class PromptMethod(ABC):
    """Base class for all confidence elicitation methods."""
    
    @abstractmethod
    def generate_prompt(self, prompt: Dict[str, Any]) -> str:
        """
        Generate a prompt that elicits a confidence score.
        
        Args:
            prompt: Dict containing prompt data including 'question' and other fields
            
        Returns:
            String prompt with confidence elicitation instructions
        """
        pass
    
    @abstractmethod
    def extract_answer_and_confidence(self, response: str) -> Tuple[Optional[str], Optional[float]]:
        """
        Extract the answer and confidence score from a model response.
        
        Args:
            response: The model's response text
            
        Returns:
            Tuple of (answer, confidence) where each can be None if not found
        """
        pass

    def classify_response(
        self, 
        response: str, 
        answer: Optional[str], 
        confidence: Optional[float]
    ) -> str:
        """
        Classify a response as valid, invalid, or no_answer.
        
        Args:
            response: The model's response text
            answer: The extracted answer or None
            confidence: The extracted confidence score or None
            
        Returns:
            One of: VALID_ANSWER, NO_ANSWER, INVALID_ANSWER
        """
        if response.lower().startswith(NO_ANSWER_TEXT.lower()):
            return NO_ANSWER
        elif answer is None or confidence is None:
            return INVALID_ANSWER
        elif answer.lower() == NO_ANSWER_TEXT.lower():
            return NO_ANSWER
        elif 0 <= confidence <= 1:
            return VALID_ANSWER
        else:
            return INVALID_ANSWER


class BasicPercentageMethod(PromptMethod):
    """
    Elicits confidence as a percentage (0-100%).
    
    This method uses a simple format where the model provides an answer
    followed by a confidence score expressed as a percentage.
    """
    
    # Regex patterns for extracting answer and confidence
    response_patterns = [
        # Standard format: "Answer: X\nConfidence: Y%"
        re.compile(fr"Answer: (?P<answer>{PATTERN_ANSWER}){PATTERN_SEP}Confidence: (?P<confidence>{PATTERN_FLOAT})%?"),
        # Alternative format: "X\nY%"
        re.compile(fr"(?P<answer>{PATTERN_ANSWER})\n+(Confidence: )?(?P<confidence>{PATTERN_FLOAT})%?"),
        # Single line format: "X (Y%)"
        re.compile(fr"(?P<answer>{PATTERN_ANSWER}) \((?P<confidence>{PATTERN_FLOAT})%?\)"),
    ]
    
    def generate_prompt(self, prompt: Dict[str, Any]) -> str:
        """
        Generate a prompt that asks for a confidence percentage.
        
        Args:
            prompt: Dict containing 'question' and other fields
            
        Returns:
            Formatted prompt with confidence elicitation instructions
        """
        base_prompt = prompt.get('question', '')
        
        confidence_instructions = (
            f"{base_prompt}\n\n"
            f"After your answer, provide a confidence score in percentage which measures "
            f"how confident you are in your answer. Use the following format to respond:\n"
            f"```\n"
            f"Answer: [Write your answer here.]\n"
            f"Confidence: [Write your confidence score as a percentage here.]\n"
            f"```\n"
            f"{PROMPT_NO_ANSWER}"
        )
        
        return confidence_instructions
    
    def extract_answer_and_confidence(self, response: str) -> Tuple[Optional[str], Optional[float]]:
        """
        Extract answer and confidence percentage from response.
        
        Args:
            response: The model's response text
            
        Returns:
            Tuple of (answer, confidence) where confidence is normalized to [0,1]
        """
        answer, confidence = extract_from_response(response, self.response_patterns, ("answer", "confidence"))
        normalized_confidence = normalize_confidence(confidence, lambda c: float(c) / 100)
        return answer, normalized_confidence


class BasicFloatMethod(PromptMethod):
    """
    Elicits confidence as a float between 0.0 and 1.0.
    
    This method uses a format where the model provides an answer
    followed by a confidence score expressed as a float.
    """
    
    # Regex patterns for extracting answer and confidence
    response_patterns = [
        # Standard format: "Answer: X\nConfidence: Y"
        re.compile(fr"Answer: (?P<answer>{PATTERN_ANSWER}){PATTERN_SEP}Confidence: (?P<confidence>{PATTERN_FLOAT})"),
        # Alternative format: "X\nY"
        re.compile(fr"(?P<answer>{PATTERN_ANSWER})\n+(Confidence: )?(?P<confidence>{PATTERN_FLOAT})"),
        # Single line format: "X (Y)"
        re.compile(fr"(?P<answer>{PATTERN_ANSWER}) \((?P<confidence>{PATTERN_FLOAT})\)"),
    ]
    
    def generate_prompt(self, prompt: Dict[str, Any]) -> str:
        """
        Generate a prompt that asks for a confidence float.
        
        Args:
            prompt: Dict containing 'question' and other fields
            
        Returns:
            Formatted prompt with confidence elicitation instructions
        """
        base_prompt = prompt.get('question', '')
        
        confidence_instructions = (
            f"{base_prompt}\n\n"
            f"After your answer, provide a confidence score between 0.0 and 1.0 which measures "
            f"how confident you are in your answer. Use the following format to respond:\n"
            f"```\n"
            f"Answer: [Write your answer here.]\n"
            f"Confidence: [Write your confidence score between 0.0 and 1.0 here.]\n"
            f"```\n"
            f"{PROMPT_NO_ANSWER}"
        )
        
        return confidence_instructions
    
    def extract_answer_and_confidence(self, response: str) -> Tuple[Optional[str], Optional[float]]:
        """
        Extract answer and confidence float from response.
        
        Args:
            response: The model's response text
            
        Returns:
            Tuple of (answer, confidence)
        """
        answer, confidence = extract_from_response(response, self.response_patterns, ("answer", "confidence"))
        normalized_confidence = normalize_confidence(confidence, float)
        return answer, normalized_confidence


class BasicLetterMethod(PromptMethod):
    """
    Elicits confidence as a letter grade (A through E).
    
    This method uses a format where the model provides an answer
    followed by a confidence score expressed as a letter grade,
    which is then mapped to a numeric value.
    """
    
    # Mapping of letter grades to confidence scores
    confidence_scores = {
        "A": 0.9,  # Very high confidence
        "B": 0.7,  # High confidence
        "C": 0.5,  # Medium confidence
        "D": 0.3,  # Low confidence
        "E": 0.1,  # Very low confidence
    }
    
    # Regex patterns for extracting answer and confidence
    response_patterns = [
        # Standard format: "Answer: X\nConfidence: Y"
        re.compile(fr"Answer: (?P<answer>{PATTERN_ANSWER}){PATTERN_SEP}Confidence: (?P<confidence>[A-E])"),
        # Alternative format: "X\nY"
        re.compile(fr"(?P<answer>{PATTERN_ANSWER})\n+(Confidence: )?(?P<confidence>[A-E])"),
        # Single line format: "X (Y)"
        re.compile(fr"(?P<answer>{PATTERN_ANSWER}) \((?P<confidence>[A-E])\)"),
    ]
    
    def generate_prompt(self, prompt: Dict[str, Any]) -> str:
        """
        Generate a prompt that asks for a confidence letter grade.
        
        Args:
            prompt: Dict containing 'question' and other fields
            
        Returns:
            Formatted prompt with confidence elicitation instructions
        """
        base_prompt = prompt.get('question', '')
        
        confidence_instructions = (
            f"{base_prompt}\n\n"
            f"After your answer, provide a confidence score between A (very high confidence) "
            f"and E (very low confidence) which measures how confident you are in your answer. "
            f"Use the following format to respond:\n"
            f"```\n"
            f"Answer: [Write your answer here.]\n"
            f"Confidence: [Write your confidence score as a letter A through E here.]\n"
            f"```\n"
            f"{PROMPT_NO_ANSWER}"
        )
        
        return confidence_instructions
    
    def extract_answer_and_confidence(self, response: str) -> Tuple[Optional[str], Optional[float]]:
        """
        Extract answer and letter confidence from response, converting to float.
        
        Args:
            response: The model's response text
            
        Returns:
            Tuple of (answer, confidence) where confidence is mapped to [0,1]
        """
        answer, confidence_letter = extract_from_response(response, self.response_patterns, ("answer", "confidence"))
        
        if confidence_letter is not None:
            confidence_letter = confidence_letter.upper()  # Ensure uppercase
        
        normalized_confidence = normalize_confidence(
            confidence_letter, 
            lambda c: self.confidence_scores.get(c, None)
        )
        
        return answer, normalized_confidence


class BasicTextMethod(PromptMethod):
    """
    Elicits confidence as text descriptions (very low to very high).
    
    This method uses a format where the model provides an answer
    followed by a confidence score expressed as a text description,
    which is then mapped to a numeric value.
    """
    
    # Mapping of text descriptions to confidence scores
    confidence_scores = {
        "very high": 0.9,
        "high": 0.7,
        "medium": 0.5,
        "low": 0.3,
        "very low": 0.1,
    }
    
    # Regex patterns for extracting answer and confidence
    response_patterns = [
        # Standard format: "Answer: X\nConfidence: Y"
        re.compile(fr"Answer: (?P<answer>{PATTERN_ANSWER}){PATTERN_SEP}Confidence: (?P<confidence>{'|'.join(confidence_scores)})", re.IGNORECASE),
        # Alternative format: "X\nY"
        re.compile(fr"(?P<answer>{PATTERN_ANSWER})\n+(Confidence: )?(?P<confidence>{'|'.join(confidence_scores)})", re.IGNORECASE),
        # Single line format: "X (Y)"
        re.compile(fr"(?P<answer>{PATTERN_ANSWER}) \((?P<confidence>{'|'.join(confidence_scores)})\)", re.IGNORECASE),
    ]
    
    def generate_prompt(self, prompt: Dict[str, Any]) -> str:
        """
        Generate a prompt that asks for a text confidence level.
        
        Args:
            prompt: Dict containing 'question' and other fields
            
        Returns:
            Formatted prompt with confidence elicitation instructions
        """
        base_prompt = prompt.get('question', '')
        
        confidence_instructions = (
            f"{base_prompt}\n\n"
            f"After your answer, provide one of the following confidence scores which measures "
            f"how confident you are in your answer: {', '.join(self.confidence_scores)}. "
            f"Use the following format to respond:\n"
            f"```\n"
            f"Answer: [Write your answer here.]\n"
            f"Confidence: [Write your confidence score here.]\n"
            f"```\n"
            f"{PROMPT_NO_ANSWER}"
        )
        
        return confidence_instructions
    
    def extract_answer_and_confidence(self, response: str) -> Tuple[Optional[str], Optional[float]]:
        """
        Extract answer and text confidence from response, converting to float.
        
        Args:
            response: The model's response text
            
        Returns:
            Tuple of (answer, confidence) where confidence is mapped to [0,1]
        """
        answer, confidence_text = extract_from_response(response, self.response_patterns, ("answer", "confidence"))
        
        if confidence_text is not None:
            confidence_text = confidence_text.lower()  # Standardize to lowercase
        
        normalized_confidence = normalize_confidence(
            confidence_text, 
            lambda c: self.confidence_scores.get(c, None)
        )
        
        return answer, normalized_confidence


class AdvancedProbabilityMethod(PromptMethod):
    """
    Advanced method that elicits confidence as a probability with explicit uncertainty factors.
    
    This method prompts the model to consider various sources of uncertainty
    before providing a confidence score, which should lead to more calibrated estimates.
    """
    
    # Regex patterns for extracting answer and confidence (probability)
    response_patterns = [
        # Standard format: "Answer: X\nProbability: Y"
        re.compile(fr"Answer: (?P<answer>{PATTERN_ANSWER}){PATTERN_SEP}Probability: (?P<confidence>{PATTERN_FLOAT})"),
        # Alternative format: "X\nY"
        re.compile(fr"(?P<answer>{PATTERN_ANSWER})\n+(Probability: )?(?P<confidence>{PATTERN_FLOAT})"),
        # Single line format: "X (Y)"
        re.compile(fr"(?P<answer>{PATTERN_ANSWER}) \((?P<confidence>{PATTERN_FLOAT})\)"),
    ]
    
    def generate_prompt(self, prompt: Dict[str, Any]) -> str:
        """
        Generate a prompt that asks for a probability with uncertainty considerations.
        
        Args:
            prompt: Dict containing 'question' and other fields
            
        Returns:
            Formatted prompt with advanced confidence elicitation instructions
        """
        base_prompt = prompt.get('question', '')
        
        confidence_instructions = (
            f"{base_prompt}\n\n"
            f"After your answer, provide the probability between 0.0 and 1.0 that your answer "
            f"is correct or plausible for the given task. Take into account the following sources "
            f"of uncertainty when estimating your probability:\n"
            f"- Uncertainty in the prompt (ambiguity, lack of context)\n"
            f"- Task difficulty (complexity, reasoning required)\n"
            f"- Your knowledge availability (facts you know vs. need to infer)\n"
            f"- Other sources of uncertainty\n\n"
            f"Use the following format to respond:\n"
            f"```\n"
            f"Answer: [Write your answer here.]\n"
            f"Probability: [Write your probability here as a number between 0.0 and 1.0.]\n"
            f"```\n"
            f"{PROMPT_NO_ANSWER}"
        )
        
        return confidence_instructions
    
    def extract_answer_and_confidence(self, response: str) -> Tuple[Optional[str], Optional[float]]:
        """
        Extract answer and probability from response.
        
        Args:
            response: The model's response text
            
        Returns:
            Tuple of (answer, confidence)
        """
        answer, confidence = extract_from_response(response, self.response_patterns, ("answer", "confidence"))
        normalized_confidence = normalize_confidence(confidence, float)
        return answer, normalized_confidence


def calibration_curve(y_true: List[float], y_pred: List[float], n_bins: int = 10) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the calibration curve for predicted probabilities.
    
    Args:
        y_true: True binary labels (0 or 1)
        y_pred: Predicted probabilities (between 0 and 1)
        n_bins: Number of bins to use for calibration curve
        
    Returns:
        prob_true: Mean of true binary labels in each bin
        prob_pred: Mean of predicted probabilities in each bin
        bins: Bin edges used for digitization
        bin_count: Number of examples in each bin
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    binids = np.digitize(y_pred, bins[1:-1])
    
    bin_true = np.bincount(binids, weights=y_true, minlength=len(bins)-1)
    bin_pred = np.bincount(binids, weights=y_pred, minlength=len(bins)-1)
    bin_count = np.bincount(binids, minlength=len(bins)-1)
    
    # Avoid division by zero
    mask = bin_count > 0
    prob_true = np.zeros(len(bins)-1)
    prob_pred = np.zeros(len(bins)-1)
    
    prob_true[mask] = bin_true[mask] / bin_count[mask]
    prob_pred[mask] = bin_pred[mask] / bin_count[mask]
    
    return prob_true, prob_pred, bins, bin_count


def expected_calibration_error(y_true: List[float], y_pred: List[float], n_bins: int = 10) -> float:
    """
    Calculate the Expected Calibration Error (ECE).
    
    ECE measures the calibration of predicted probabilities by computing
    the weighted average of the absolute differences between predicted
    probabilities and observed frequencies in bins.
    
    Args:
        y_true: True binary labels (0 or 1)
        y_pred: Predicted probabilities (between 0 and 1)
        n_bins: Number of bins to use for ECE calculation
        
    Returns:
        ECE score (lower is better)
    """
    prob_true, prob_pred, _, bin_count = calibration_curve(y_true, y_pred, n_bins)
    
    # Calculate the weighted average of the absolute calibration error
    n_samples = len(y_true)
    mask = bin_count > 0
    weighted_abs_error = bin_count[mask] / n_samples * np.abs(prob_true[mask] - prob_pred[mask])
    
    return np.sum(weighted_abs_error)


class ComboExemplarsMethod(PromptMethod):
    """
    Elicits confidence with few-shot examples to help calibrate the model.
    
    This method provides several examples of answers with appropriate
    confidence scores to help the model understand the desired calibration.
    """
    
    # Example scores that will be ignored if accidentally used by the model
    EXAMPLE_SCORES = [0.47, 0.89, 0.77, 0.24, 0.08]
    
    # The few-shot examples
    EXAMPLE_PROMPT = """Here are five examples:

Question: The fox walked from the city into the forest, what was it looking for?
Choices:
A. pretty flowers.
B. hen house
C. natural habitat
D. storybook
E. dense forest
Guess: A
Probability: 0.47

Question: Which country is Europe's largest silk producer?
Guess: Environment of Italy
Probability: 0.89

Question: The population of the city where Michelle was born is 145,826. What is the value of the 5 in the number 145,826?
Choices:
A. 5 thousands
B. 5 hundreds
C. 5 tens
D. 5 ones
Guess: A
Probability: 0.77

Question: Beyond the business case for engaging in CSR there are a number of moral arguments relating to: negative _______, the _______that corporations possess and the ________ of business and society.
Choices:
A. Externalities, Power, Independence
B. Publicity, Insubstantial resources, Mutual dependence
C. Publicity, Power, Independence
D. Externalities, Power, Mutual dependence
Guess: B
Probability: 0.24

Question: The Moon lacks weather and climate changes like those on Earth. What causes the lack of weather on the Moon?
Guess: the lack of magnetic poles
Probability: 0.08"""
    
    # Regex patterns for extracting answer and confidence
    response_patterns = [
        # Standard format: "Guess: X\nProbability: Y"
        re.compile(fr"(Guess: |Answer: )(?P<answer>{PATTERN_ANSWER}){PATTERN_SEP}Probability: (?P<confidence>{PATTERN_FLOAT})"),
        # Alternative formats 
        re.compile(fr"(Guess: |Answer: )?(?P<answer>{PATTERN_ANSWER})\n+(Probability: )?(?P<confidence>{PATTERN_FLOAT})"),
        re.compile(fr"(Guess: |Answer: )?(?P<answer>{PATTERN_ANSWER})(, |. |; | - | \| )(Probability: )?(?P<confidence>{PATTERN_FLOAT})"),
        re.compile(fr"(Guess: |Answer: )?(?P<answer>{PATTERN_ANSWER}) \((Probability: )?(?P<confidence>{PATTERN_FLOAT})\)"),
    ]
    
    def generate_prompt(self, prompt: Dict[str, Any]) -> str:
        """
        Generate a prompt with few-shot examples of confidence scores.
        
        Args:
            prompt: Dict containing 'question' and other fields
            
        Returns:
            Formatted prompt with examples and confidence elicitation instructions
        """
        base_prompt = prompt.get('question', '')
        
        confidence_instructions = (
            f"{base_prompt}\n\n"
            f"Provide your best guess and the probability that it is correct or plausible (0.0 to 1.0) "
            f"for the following question. Take your uncertainty in the prompt, the task difficulty, "
            f"your knowledge availability and other sources of uncertainty into account. "
            f"Give ONLY the guess and probability, no other words or explanation. For example:\n"
            f"```\n"
            f"Guess: <most likely guess, as short as possible; not a complete sentence, just the guess!>\n"
            f"Probability: <the probability between 0.0 and 1.0 that your guess is correct, without any extra commentary whatsoever; just the probability!>\n"
            f"```\n"
            f"{PROMPT_NO_ANSWER} {self.EXAMPLE_PROMPT}"
        )
        
        return confidence_instructions
    
    def extract_answer_and_confidence(self, response: str) -> Tuple[Optional[str], Optional[float]]:
        """
        Extract answer and probability from response.
        
        Args:
            response: The model's response text
            
        Returns:
            Tuple of (answer, confidence)
        """
        answer, confidence = extract_from_response(response, self.response_patterns, ("answer", "confidence"))
        normalized_confidence = normalize_confidence(confidence, float)
        
        # Filter out confidence scores used in few-shot examples
        if normalized_confidence in self.EXAMPLE_SCORES:
            logger.warning(f"Confidence score {normalized_confidence} matches an example score, ignoring")
            normalized_confidence = None
            
        return answer, normalized_confidence


class MultiGuessMethod(PromptMethod):
    """
    Elicits multiple guesses with corresponding probabilities.
    
    This method asks the model to provide multiple guesses with their 
    probabilities, allowing for more nuanced uncertainty representation.
    """
    
    # Regex patterns for extracting top answer and confidence
    response_patterns = [
        # Standard G1/P1 format
        re.compile(fr"G1: (?P<answer>{PATTERN_ANSWER}){PATTERN_SEP}P1: (?P<confidence>{PATTERN_FLOAT})"),
        # Alternative formats
        re.compile(fr"G1: (?P<answer>{PATTERN_ANSWER})(, |; | - )(P1: )?(?P<confidence>{PATTERN_FLOAT})"),
        re.compile(fr"G1: (?P<answer>{PATTERN_ANSWER}) \((P1: )?(?P<confidence>{PATTERN_FLOAT})\)"),
        re.compile(fr"(G1: )?(?P<answer>{PATTERN_ANSWER})\n+(P1: )?(?P<confidence>{PATTERN_FLOAT})"),
        re.compile(fr"(G1: )?(?P<answer>{PATTERN_ANSWER}) P1: (?P<confidence>{PATTERN_FLOAT})"),
        re.compile(fr"(G1: )?(?P<answer>{PATTERN_ANSWER})[:.] (?P<confidence>{PATTERN_FLOAT})"),
    ]
    
    def generate_prompt(self, prompt: Dict[str, Any]) -> str:
        """
        Generate a prompt that asks for multiple guesses with probabilities.
        
        Args:
            prompt: Dict containing 'question' and other fields
            
        Returns:
            Formatted prompt for multiple guesses
        """
        base_prompt = prompt.get('question', '')
        k = prompt.get('num_guesses', 4)  # Default to 4 guesses
        
        confidence_instructions = (
            f"{base_prompt}\n\n"
            f"Provide your {k} best guesses and the probability that each is correct (0.0 to 1.0) "
            f"for the following question. Give ONLY the guesses and probabilities, no other words "
            f"or explanation. For example:\n"
            f"```\n"
            f"G1: <first most likely guess, as short as possible; not a complete sentence, just the guess!>\n"
            f"P1: <the probability between 0.0 and 1.0 that G1 is correct, without any extra commentary whatsoever; just the probability!>\n"
            f"...\n"
            f"G{k}: <{k}-th most likely guess, as short as possible; not a complete sentence, just the guess!>\n"
            f"P{k}: <the probability between 0.0 and 1.0 that G{k} is correct, without any extra commentary whatsoever; just the probability!>\n"
            f"```\n"
            f"{PROMPT_NO_ANSWER}"
        )
        
        return confidence_instructions
    
    def extract_answer_and_confidence(self, response: str) -> Tuple[Optional[str], Optional[float]]:
        """
        Extract top answer and its probability from response.
        
        Note: This method only returns the top guess, but the full response
        can be parsed to extract all guesses if needed.
        
        Args:
            response: The model's response text
            
        Returns:
            Tuple of (top_answer, top_confidence)
        """
        answer, confidence = extract_from_response(response, self.response_patterns, ("answer", "confidence"))
        normalized_confidence = normalize_confidence(confidence, float)
        return answer, normalized_confidence
    
    def extract_all_guesses(self, response: str, k: int = 4) -> List[Tuple[Optional[str], Optional[float]]]:
        """
        Extract all guesses and their confidence scores.
        
        Args:
            response: The model's response text
            k: Number of guesses to extract (default: 4)
            
        Returns:
            List of (answer, confidence) tuples for all guesses
        """
        results = []
        
        for i in range(1, k + 1):
            # Create patterns for each guess
            patterns = [
                re.compile(fr"G{i}: (?P<answer>{PATTERN_ANSWER}){PATTERN_SEP}P{i}: (?P<confidence>{PATTERN_FLOAT})"),
                re.compile(fr"G{i}: (?P<answer>{PATTERN_ANSWER})(, |; | - )(P{i}: )?(?P<confidence>{PATTERN_FLOAT})"),
                re.compile(fr"G{i}: (?P<answer>{PATTERN_ANSWER}) \((P{i}: )?(?P<confidence>{PATTERN_FLOAT})\)"),
            ]
            
            answer, confidence = extract_from_response(response, patterns, ("answer", "confidence"))
            normalized_confidence = normalize_confidence(confidence, float)
            results.append((answer, normalized_confidence))
            
        return results


# Dictionary of available elicitation methods for easy access
CONFIDENCE_METHODS = {
    "basic_percentage": BasicPercentageMethod(),
    "basic_float": BasicFloatMethod(),
    "basic_letter": BasicLetterMethod(),
    "basic_text": BasicTextMethod(),
    "advanced_probability": AdvancedProbabilityMethod(),
    "combo_exemplars": ComboExemplarsMethod(),
    "multi_guess": MultiGuessMethod(),
}