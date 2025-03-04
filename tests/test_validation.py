"""
Tests for the validation module.
"""

import unittest
import pytest
from agentoptim.validation import (
    validate_required,
    validate_string,
    validate_number,
    validate_bool,
    validate_list,
    validate_dict,
    validate_one_of,
    validate_uuid,
    validate_email,
    validate_url,
    validate_id_format,
    validate_template,
)
from agentoptim.errors import ValidationError


class TestValidation(unittest.TestCase):
    """Test the validation utilities."""
    
    def test_validate_required(self):
        """Test the validate_required function."""
        # Should not raise for non-None values
        validate_required("value", "field")
        validate_required(0, "field")
        validate_required(False, "field")
        
        # Should raise for None values
        with pytest.raises(ValidationError) as e:
            validate_required(None, "field")
        assert "required" in str(e.value)
    
    def test_validate_string(self):
        """Test the validate_string function."""
        # Valid strings
        self.assertEqual("hello", validate_string("hello", "field"))
        self.assertEqual("a" * 10, validate_string("a" * 10, "field", max_length=10))
        
        # None handling
        self.assertIsNone(validate_string(None, "field", required=False))
        with pytest.raises(ValidationError) as e:
            validate_string(None, "field")
        assert "required" in str(e.value)
        
        # Invalid types
        with pytest.raises(ValidationError) as e:
            validate_string(123, "field")
        assert "must be a string" in str(e.value)
        
        # Length validation
        with pytest.raises(ValidationError) as e:
            validate_string("", "field", min_length=1)
        assert "at least" in str(e.value)
        
        with pytest.raises(ValidationError) as e:
            validate_string("too long", "field", max_length=5)
        assert "at most" in str(e.value)
    
    def test_validate_number(self):
        """Test the validate_number function."""
        # Valid numbers
        self.assertEqual(123.0, validate_number(123, "field"))
        self.assertEqual(123.5, validate_number(123.5, "field"))
        self.assertEqual(123.0, validate_number("123", "field"))
        
        # Range validation
        self.assertEqual(5.0, validate_number(5, "field", min_value=0, max_value=10))
        
        # None handling
        self.assertIsNone(validate_number(None, "field", required=False))
        with pytest.raises(ValidationError) as e:
            validate_number(None, "field")
        assert "required" in str(e.value)
        
        # Invalid types
        with pytest.raises(ValidationError) as e:
            validate_number("not a number", "field")
        assert "must be a number" in str(e.value)
        
        # Range validation errors
        with pytest.raises(ValidationError) as e:
            validate_number(-1, "field", min_value=0)
        assert "at least" in str(e.value)
        
        with pytest.raises(ValidationError) as e:
            validate_number(11, "field", max_value=10)
        assert "at most" in str(e.value)
    
    def test_validate_bool(self):
        """Test the validate_bool function."""
        # Valid booleans
        self.assertTrue(validate_bool(True, "field"))
        self.assertFalse(validate_bool(False, "field"))
        
        # String conversions
        self.assertTrue(validate_bool("true", "field"))
        self.assertTrue(validate_bool("True", "field"))
        self.assertTrue(validate_bool("YES", "field"))
        self.assertTrue(validate_bool("y", "field"))
        self.assertTrue(validate_bool("1", "field"))
        
        self.assertFalse(validate_bool("false", "field"))
        self.assertFalse(validate_bool("False", "field"))
        self.assertFalse(validate_bool("NO", "field"))
        self.assertFalse(validate_bool("n", "field"))
        self.assertFalse(validate_bool("0", "field"))
        
        # Numeric conversions
        self.assertTrue(validate_bool(1, "field"))
        self.assertFalse(validate_bool(0, "field"))
        
        # None handling
        self.assertIsNone(validate_bool(None, "field", required=False))
        with pytest.raises(ValidationError) as e:
            validate_bool(None, "field")
        assert "required" in str(e.value)
        
        # Invalid values
        with pytest.raises(ValidationError) as e:
            validate_bool("invalid", "field")
        assert "must be a boolean" in str(e.value)
        
        with pytest.raises(ValidationError) as e:
            validate_bool(2, "field")
        assert "must be a boolean" in str(e.value)
    
    def test_validate_list(self):
        """Test the validate_list function."""
        # Valid lists
        self.assertEqual([], validate_list([], "field"))
        self.assertEqual([1, 2, 3], validate_list([1, 2, 3], "field"))
        
        # None handling
        self.assertIsNone(validate_list(None, "field", required=False))
        with pytest.raises(ValidationError) as e:
            validate_list(None, "field")
        assert "required" in str(e.value)
        
        # Invalid types
        with pytest.raises(ValidationError) as e:
            validate_list("not a list", "field")
        assert "must be a list" in str(e.value)
        
        # Length validation
        with pytest.raises(ValidationError) as e:
            validate_list([], "field", min_items=1)
        assert "at least" in str(e.value)
        
        with pytest.raises(ValidationError) as e:
            validate_list([1, 2, 3], "field", max_items=2)
        assert "at most" in str(e.value)
        
        # Item validation
        def validate_positive(val, name):
            if val <= 0:
                raise ValidationError(f"{name} must be positive", field=name, value=val)
            return val
        
        self.assertEqual([1, 2, 3], validate_list([1, 2, 3], "field", item_validator=validate_positive))
        
        with pytest.raises(ValidationError) as e:
            validate_list([1, 0, 3], "field", item_validator=validate_positive)
        assert "Invalid item" in str(e.value)
        assert "field[1]" in str(e.value)
    
    def test_validate_dict(self):
        """Test the validate_dict function."""
        # Valid dictionaries
        self.assertEqual({}, validate_dict({}, "field"))
        self.assertEqual({"a": 1, "b": 2}, validate_dict({"a": 1, "b": 2}, "field"))
        
        # Required keys
        self.assertEqual({"a": 1, "b": 2}, validate_dict(
            {"a": 1, "b": 2}, "field", required_keys={"a", "b"}
        ))
        
        with pytest.raises(ValidationError) as e:
            validate_dict({"a": 1}, "field", required_keys={"a", "b"})
        assert "missing required keys" in str(e.value)
        assert "b" in str(e.value)
        
        # Optional keys
        self.assertEqual({"a": 1}, validate_dict(
            {"a": 1}, "field", optional_keys={"a", "b"}
        ))
        
        # Extra keys
        with pytest.raises(ValidationError) as e:
            validate_dict(
                {"a": 1, "c": 3}, "field", 
                required_keys={"a"}, optional_keys={"b"}, 
                allow_extra_keys=False
            )
        assert "disallowed keys" in str(e.value)
        assert "c" in str(e.value)
        
        # None handling
        self.assertIsNone(validate_dict(None, "field", required=False))
        with pytest.raises(ValidationError) as e:
            validate_dict(None, "field")
        assert "required" in str(e.value)
        
        # Invalid types
        with pytest.raises(ValidationError) as e:
            validate_dict("not a dict", "field")
        assert "must be a dictionary" in str(e.value)
        
        # Value validation
        def validate_positive(val, name):
            if val <= 0:
                raise ValidationError(f"{name} must be positive", field=name, value=val)
            return val
        
        self.assertEqual(
            {"a": 1, "b": 2}, 
            validate_dict(
                {"a": 1, "b": 2}, "field", 
                key_validators={"a": validate_positive, "b": validate_positive}
            )
        )
        
        with pytest.raises(ValidationError) as e:
            validate_dict(
                {"a": 1, "b": 0}, "field", 
                key_validators={"a": validate_positive, "b": validate_positive}
            )
        assert "Invalid value" in str(e.value)
        assert "field.b" in str(e.value)
    
    def test_validate_one_of(self):
        """Test the validate_one_of function."""
        # Valid values
        self.assertEqual("a", validate_one_of("a", "field", ["a", "b", "c"]))
        self.assertEqual(1, validate_one_of(1, "field", [1, 2, 3]))
        
        # None handling
        self.assertIsNone(validate_one_of(None, "field", ["a", "b"], required=False))
        with pytest.raises(ValidationError) as e:
            validate_one_of(None, "field", ["a", "b"])
        assert "required" in str(e.value)
        
        # Invalid values
        with pytest.raises(ValidationError) as e:
            validate_one_of("d", "field", ["a", "b", "c"])
        assert "must be one of" in str(e.value)
        assert "a" in str(e.value)
        assert "b" in str(e.value)
        assert "c" in str(e.value)
    
    def test_validate_uuid(self):
        """Test the validate_uuid function."""
        # Valid UUIDs
        valid_uuid = "123e4567-e89b-12d3-a456-426614174000"
        self.assertEqual(valid_uuid, validate_uuid(valid_uuid, "field"))
        
        # None handling
        self.assertIsNone(validate_uuid(None, "field", required=False))
        with pytest.raises(ValidationError) as e:
            validate_uuid(None, "field")
        assert "required" in str(e.value)
        
        # Invalid types
        with pytest.raises(ValidationError) as e:
            validate_uuid(123, "field")
        assert "must be a string" in str(e.value)
        
        # Invalid UUIDs
        with pytest.raises(ValidationError) as e:
            validate_uuid("not-a-uuid", "field")
        assert "must be a valid UUID" in str(e.value)
    
    def test_validate_email(self):
        """Test the validate_email function."""
        # Valid emails
        self.assertEqual("user@example.com", validate_email("user@example.com", "field"))
        self.assertEqual("user.name+tag@example.co.uk", validate_email("user.name+tag@example.co.uk", "field"))
        
        # None handling
        self.assertIsNone(validate_email(None, "field", required=False))
        with pytest.raises(ValidationError) as e:
            validate_email(None, "field")
        assert "required" in str(e.value)
        
        # Invalid types
        with pytest.raises(ValidationError) as e:
            validate_email(123, "field")
        assert "must be a string" in str(e.value)
        
        # Invalid emails
        with pytest.raises(ValidationError) as e:
            validate_email("not-an-email", "field")
        assert "must be a valid email address" in str(e.value)
        
        with pytest.raises(ValidationError) as e:
            validate_email("user@", "field")
        assert "must be a valid email address" in str(e.value)
    
    def test_validate_url(self):
        """Test the validate_url function."""
        # Valid URLs
        self.assertEqual("https://example.com", validate_url("https://example.com", "field"))
        self.assertEqual("http://example.com/path?query=1", validate_url("http://example.com/path?query=1", "field"))
        
        # None handling
        self.assertIsNone(validate_url(None, "field", required=False))
        with pytest.raises(ValidationError) as e:
            validate_url(None, "field")
        assert "required" in str(e.value)
        
        # Invalid types
        with pytest.raises(ValidationError) as e:
            validate_url(123, "field")
        assert "must be a string" in str(e.value)
        
        # Invalid URLs
        with pytest.raises(ValidationError) as e:
            validate_url("not-a-url", "field")
        assert "must be a valid URL" in str(e.value)
        
        with pytest.raises(ValidationError) as e:
            validate_url("example.com", "field")  # Missing protocol
        assert "must be a valid URL" in str(e.value)
    
    def test_validate_id_format(self):
        """Test the validate_id_format function."""
        # Valid IDs
        self.assertEqual("eval_123", validate_id_format("eval_123", "field", "eval"))
        self.assertEqual("job_abc123", validate_id_format("job_abc123", "field", "job"))
        
        # None handling
        self.assertIsNone(validate_id_format(None, "field", "eval", required=False))
        with pytest.raises(ValidationError) as e:
            validate_id_format(None, "field", "eval")
        assert "required" in str(e.value)
        
        # Invalid types
        with pytest.raises(ValidationError) as e:
            validate_id_format(123, "field", "eval")
        assert "must be a string" in str(e.value)
        
        # Invalid formats
        with pytest.raises(ValidationError) as e:
            validate_id_format("123_eval", "field", "eval")
        assert "must start with" in str(e.value)
        
        with pytest.raises(ValidationError) as e:
            validate_id_format("job_123", "field", "eval")
        assert "must start with" in str(e.value)
    
    def test_validate_template(self):
        """Test the validate_template function."""
        # Valid templates
        self.assertEqual("Hello {name}", validate_template("Hello {name}", "field"))
        
        # Required variables
        self.assertEqual(
            "Hello {name}, your score is {{ score }}",
            validate_template(
                "Hello {name}, your score is {{ score }}", 
                "field", 
                required_variables={"name", "score"}
            )
        )
        
        # None handling
        self.assertIsNone(validate_template(None, "field", required=False))
        with pytest.raises(ValidationError) as e:
            validate_template(None, "field")
        assert "required" in str(e.value)
        
        # Invalid types
        with pytest.raises(ValidationError) as e:
            validate_template(123, "field")
        assert "must be a string" in str(e.value)
        
        # Missing variables
        with pytest.raises(ValidationError) as e:
            validate_template(
                "Hello {name}", 
                "field", 
                required_variables={"name", "score"}
            )
        assert "missing required variables" in str(e.value)
        assert "score" in str(e.value)


if __name__ == "__main__":
    unittest.main()