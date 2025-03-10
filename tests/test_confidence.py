"""Tests for the confidence module."""

import unittest
import re
from unittest.mock import patch
import numpy as np

from agentoptim.confidence import (
    extract_from_response,
    normalize_confidence,
    calibration_curve,
    expected_calibration_error,
    BasicPercentageMethod,
    BasicFloatMethod,
    BasicLetterMethod,
    BasicTextMethod,
    AdvancedProbabilityMethod,
    ComboExemplarsMethod,
    MultiGuessMethod,
    VALID_ANSWER,
    NO_ANSWER,
    INVALID_ANSWER,
)


class TestConfidenceExtraction(unittest.TestCase):
    """Test the confidence extraction utilities."""
    
    def test_extract_from_response_single(self):
        """Test extracting a single value from a response."""
        test_response = "The answer is 42"
        pattern = re.compile(r"The answer is (?P<answer>\d+)")
        result = extract_from_response(test_response, [pattern], "answer")
        self.assertEqual(result, "42")
    
    def test_extract_from_response_multiple(self):
        """Test extracting multiple values from a response."""
        test_response = "Answer: 42\nConfidence: 75%"
        pattern = re.compile(r"Answer: (?P<answer>\d+)[\s\S]*Confidence: (?P<confidence>\d+)%?")
        answer, confidence = extract_from_response(test_response, [pattern], ["answer", "confidence"])
        self.assertEqual(answer, "42")
        self.assertEqual(confidence, "75")
    
    def test_extract_from_response_no_match(self):
        """Test behavior when no pattern matches."""
        test_response = "I don't know the answer"
        pattern = re.compile(r"Answer: (?P<answer>\d+)")
        result = extract_from_response(test_response, [pattern], "answer")
        self.assertIsNone(result)
    
    def test_extract_from_response_multiple_patterns(self):
        """Test trying multiple patterns until one matches."""
        test_response = "My guess: 42. Confidence: 75%"
        pattern1 = re.compile(r"Answer: (?P<answer>\d+).*Confidence: (?P<confidence>\d+)%")
        pattern2 = re.compile(r"My guess: (?P<answer>\d+)\. Confidence: (?P<confidence>\d+)%")
        answer, confidence = extract_from_response(test_response, [pattern1, pattern2], ["answer", "confidence"])
        self.assertEqual(answer, "42")
        self.assertEqual(confidence, "75")
    
    def test_normalize_confidence_percentage(self):
        """Test normalizing a percentage to [0,1]."""
        result = normalize_confidence("75", lambda c: float(c) / 100)
        self.assertEqual(result, 0.75)
    
    def test_normalize_confidence_float(self):
        """Test normalizing a float."""
        result = normalize_confidence("0.75")
        self.assertEqual(result, 0.75)
    
    def test_normalize_confidence_out_of_range(self):
        """Test normalizing values outside [0,1]."""
        with patch('agentoptim.confidence.logger.warning') as mock_warning:
            result = normalize_confidence("1.5")
            self.assertEqual(result, 1.0)
            mock_warning.assert_called_once()

        with patch('agentoptim.confidence.logger.warning') as mock_warning:
            result = normalize_confidence("-0.5")
            self.assertEqual(result, 0.0)
            mock_warning.assert_called_once()
    
    def test_normalize_confidence_none(self):
        """Test normalizing None."""
        result = normalize_confidence(None)
        self.assertIsNone(result)
    
    def test_normalize_confidence_invalid(self):
        """Test normalizing an invalid string."""
        with patch('agentoptim.confidence.logger.warning') as mock_warning:
            result = normalize_confidence("not a number")
            self.assertIsNone(result)
            mock_warning.assert_called_once()


class TestCalibrationMetrics(unittest.TestCase):
    """Test the calibration metrics."""
    
    def test_calibration_curve(self):
        """Test calibration curve calculation."""
        # Perfect calibration
        y_true = [0, 1, 0, 1, 0]
        y_pred = [0.0, 1.0, 0.0, 1.0, 0.0]
        prob_true, prob_pred, bins, bin_count = calibration_curve(y_true, y_pred, n_bins=2)
        # Check bins have correct values
        np.testing.assert_array_almost_equal(prob_true[bin_count > 0], [0.0, 1.0])
        np.testing.assert_array_almost_equal(prob_pred[bin_count > 0], [0.0, 1.0])
    
    def test_expected_calibration_error(self):
        """Test ECE calculation."""
        # Perfect calibration
        y_true = [0, 1, 0, 1, 0]
        y_pred = [0.0, 1.0, 0.0, 1.0, 0.0]
        ece = expected_calibration_error(y_true, y_pred, n_bins=2)
        self.assertAlmostEqual(ece, 0.0)
        
        # Poor calibration
        y_true = [0, 0, 0, 0, 0]
        y_pred = [0.5, 0.5, 0.5, 0.5, 0.5]
        ece = expected_calibration_error(y_true, y_pred, n_bins=2)
        self.assertAlmostEqual(ece, 0.5)


class TestPromptMethods(unittest.TestCase):
    """Test the prompt methods."""
    
    def test_basic_percentage_method(self):
        """Test the basic percentage method."""
        method = BasicPercentageMethod()
        
        # Test prompt generation
        prompt = method.generate_prompt({"question": "What is 2+2?"})
        self.assertIn("What is 2+2?", prompt)
        self.assertIn("confidence score in percentage", prompt)
        
        # Test extraction
        response = "Answer: 4\nConfidence: 90%"
        answer, confidence = method.extract_answer_and_confidence(response)
        self.assertEqual(answer, "4")
        self.assertEqual(confidence, 0.9)
        
        # Test classification
        classification = method.classify_response(response, "4", 0.9)
        self.assertEqual(classification, VALID_ANSWER)
        
        classification = method.classify_response("NO ANSWER", None, None)
        self.assertEqual(classification, NO_ANSWER)
        
        classification = method.classify_response("I'm not sure", "I'm not sure", None)
        self.assertEqual(classification, INVALID_ANSWER)
    
    def test_basic_float_method(self):
        """Test the basic float method."""
        method = BasicFloatMethod()
        
        # Test prompt generation
        prompt = method.generate_prompt({"question": "What is 2+2?"})
        self.assertIn("What is 2+2?", prompt)
        self.assertIn("between 0.0 and 1.0", prompt)
        
        # Test extraction
        response = "Answer: 4\nConfidence: 0.9"
        answer, confidence = method.extract_answer_and_confidence(response)
        self.assertEqual(answer, "4")
        self.assertEqual(confidence, 0.9)
    
    def test_basic_letter_method(self):
        """Test the basic letter method."""
        method = BasicLetterMethod()
        
        # Test prompt generation
        prompt = method.generate_prompt({"question": "What is 2+2?"})
        self.assertIn("What is 2+2?", prompt)
        self.assertIn("between A", prompt)
        
        # Test extraction
        response = "Answer: 4\nConfidence: A"
        answer, confidence = method.extract_answer_and_confidence(response)
        self.assertEqual(answer, "4")
        self.assertEqual(confidence, 0.9)
        
        response = "Answer: 4\nConfidence: C"
        answer, confidence = method.extract_answer_and_confidence(response)
        self.assertEqual(answer, "4")
        self.assertEqual(confidence, 0.5)
    
    def test_basic_text_method(self):
        """Test the basic text method."""
        method = BasicTextMethod()
        
        # Test prompt generation
        prompt = method.generate_prompt({"question": "What is 2+2?"})
        self.assertIn("What is 2+2?", prompt)
        self.assertIn("very high", prompt)
        
        # Test extraction
        response = "Answer: 4\nConfidence: high"
        answer, confidence = method.extract_answer_and_confidence(response)
        self.assertEqual(answer, "4")
        self.assertEqual(confidence, 0.7)
        
        response = "Answer: 4\nConfidence: very low"
        answer, confidence = method.extract_answer_and_confidence(response)
        self.assertEqual(answer, "4")
        self.assertEqual(confidence, 0.1)
    
    def test_advanced_probability_method(self):
        """Test the advanced probability method."""
        method = AdvancedProbabilityMethod()
        
        # Test prompt generation
        prompt = method.generate_prompt({"question": "What is 2+2?"})
        self.assertIn("What is 2+2?", prompt)
        self.assertIn("uncertainty in the prompt", prompt.lower())
        self.assertIn("task difficulty", prompt.lower())
        
        # Test extraction
        response = "Answer: 4\nProbability: 0.9"
        answer, confidence = method.extract_answer_and_confidence(response)
        self.assertEqual(answer, "4")
        self.assertEqual(confidence, 0.9)
    
    def test_combo_exemplars_method(self):
        """Test the combo exemplars method."""
        method = ComboExemplarsMethod()
        
        # Test prompt generation
        prompt = method.generate_prompt({"question": "What is 2+2?"})
        self.assertIn("What is 2+2?", prompt)
        self.assertIn("Here are five examples", prompt)
        
        # Test extraction
        response = "Guess: 4\nProbability: 0.9"
        answer, confidence = method.extract_answer_and_confidence(response)
        self.assertEqual(answer, "4")
        self.assertEqual(confidence, 0.9)
        
        # Test filtering of example scores
        response = "Guess: 4\nProbability: 0.47"  # Matches an example score
        with patch('agentoptim.confidence.logger.warning') as mock_warning:
            answer, confidence = method.extract_answer_and_confidence(response)
            self.assertEqual(answer, "4")
            self.assertIsNone(confidence)
            mock_warning.assert_called_once()
    
    def test_multi_guess_method(self):
        """Test the multi-guess method."""
        method = MultiGuessMethod()
        
        # Test prompt generation
        prompt = method.generate_prompt({"question": "What is 2+2?", "num_guesses": 3})
        self.assertIn("What is 2+2?", prompt)
        self.assertIn("3 best guesses", prompt)
        
        # Test extraction of top answer
        response = (
            "G1: 4\nP1: 0.9\n"
            "G2: 5\nP2: 0.1\n"
            "G3: 3\nP3: 0.0"
        )
        answer, confidence = method.extract_answer_and_confidence(response)
        self.assertEqual(answer, "4")
        self.assertEqual(confidence, 0.9)
        
        # Test extraction of all guesses
        all_guesses = method.extract_all_guesses(response, k=3)
        self.assertEqual(len(all_guesses), 3)
        self.assertEqual(all_guesses[0], ("4", 0.9))
        self.assertEqual(all_guesses[1], ("5", 0.1))
        self.assertEqual(all_guesses[2][0], "3")
        self.assertEqual(all_guesses[2][1], 0.0)


if __name__ == '__main__':
    unittest.main()