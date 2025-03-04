"""Tests for the analysis module."""

import os
import pytest
from unittest.mock import patch, MagicMock

from agentoptim.analysis import (
    analyze_results,
    analyze_experiment_results,
    compare_analyses,
    compute_variant_statistics,
    ResultsAnalysis,
    VariantAnalysis,
    get_analysis,
    list_analyses,
    delete_analysis,
)
from agentoptim.experiment import Experiment, PromptVariant
from agentoptim.jobs import Job, JobResult, JobStatus
from agentoptim.evaluation import Evaluation


# Setup mock data for testing
@pytest.fixture
def mock_experiment():
    """Create a mock experiment for testing."""
    return Experiment(
        id="exp123",
        name="Test Experiment",
        dataset_id="ds123",
        evaluation_id="eval123",
        prompt_variants=[
            PromptVariant(id="var1", name="Variant 1", type="system", template="Template 1"),
            PromptVariant(id="var2", name="Variant 2", type="system", template="Template 2"),
        ],
        model_name="test-model",
        results={
            "var1": {
                "scores": {
                    "clarity": [4.0, 4.5, 4.2, 4.8],
                    "accuracy": [3.5, 3.8, 4.0, 3.9],
                }
            },
            "var2": {
                "scores": {
                    "clarity": [3.8, 3.9, 4.0, 4.1],
                    "accuracy": [4.2, 4.3, 4.5, 4.6],
                }
            }
        }
    )


@pytest.fixture
def mock_job():
    """Create a mock job for testing."""
    return Job(
        job_id="job123",
        experiment_id="exp123",
        dataset_id="ds123",
        evaluation_id="eval123",
        status=JobStatus.COMPLETED,
        results=[
            JobResult(
                variant_id="var1",
                data_item_id="item1",
                input_text="Input 1",
                output_text="Output 1",
                scores={"clarity": 4.0, "accuracy": 3.5}
            ),
            JobResult(
                variant_id="var1",
                data_item_id="item2",
                input_text="Input 2",
                output_text="Output 2",
                scores={"clarity": 4.5, "accuracy": 3.8}
            ),
            JobResult(
                variant_id="var2",
                data_item_id="item1",
                input_text="Input 3",
                output_text="Output 3",
                scores={"clarity": 3.8, "accuracy": 4.2}
            ),
            JobResult(
                variant_id="var2",
                data_item_id="item2",
                input_text="Input 4",
                output_text="Output 4",
                scores={"clarity": 3.9, "accuracy": 4.3}
            ),
        ]
    )


@pytest.fixture
def mock_evaluation():
    """Create a mock evaluation for testing."""
    return Evaluation(
        id="eval123",
        name="Test Evaluation",
        template="Evaluate the response for {criterion}",
        criteria=[
            {"name": "clarity", "description": "How clear is the response?"},
            {"name": "accuracy", "description": "How accurate is the response?"},
        ]
    )


@pytest.fixture
def analysis_id():
    """Return a mock analysis ID."""
    return "analysis123"


def test_compute_variant_statistics():
    """Test computing statistics for a variant."""
    scores = {
        "clarity": [4.0, 4.5, 4.2, 4.8],
        "accuracy": [3.5, 3.8, 4.0, 3.9],
    }
    
    analysis = compute_variant_statistics("var1", "Variant 1", scores)
    
    # Check that basic stats are computed correctly
    assert analysis.variant_id == "var1"
    assert analysis.variant_name == "Variant 1"
    assert analysis.sample_size == 4
    
    # Check average scores (with some tolerance for floating point)
    assert abs(analysis.average_scores["clarity"] - 4.375) < 0.001
    assert abs(analysis.average_scores["accuracy"] - 3.8) < 0.001
    
    # Check min/max
    assert analysis.min_scores["clarity"] == 4.0
    assert analysis.max_scores["clarity"] == 4.8
    
    # Ensure confidence intervals are computed
    assert "clarity" in analysis.confidence_interval
    assert "accuracy" in analysis.confidence_interval
    
    # Check that percentiles are computed
    assert "clarity" in analysis.percentiles
    assert "25th" in analysis.percentiles["clarity"]
    assert "75th" in analysis.percentiles["clarity"]


@patch("agentoptim.analysis.get_experiment")
@patch("agentoptim.analysis.save_json")
def test_analyze_experiment_results(mock_save_json, mock_get_experiment, mock_experiment):
    """Test analyzing experiment results."""
    mock_get_experiment.return_value = mock_experiment
    
    analysis = analyze_experiment_results("exp123", name="Test Analysis")
    
    # Check that the analysis was created correctly
    assert analysis.experiment_id == "exp123"
    assert analysis.name == "Test Analysis"
    assert len(analysis.variant_results) == 2
    assert "var1" in analysis.variant_results
    assert "var2" in analysis.variant_results
    
    # Check that best variants were identified
    assert len(analysis.best_variants) == 2
    assert analysis.best_variants["clarity"] == "var1"  # Variant 1 has better clarity
    assert analysis.best_variants["accuracy"] == "var2"  # Variant 2 has better accuracy
    
    # Check that an overall best variant was identified
    assert analysis.overall_best_variant is not None
    
    # Check that recommendations were generated
    assert len(analysis.recommendations) > 0
    
    # Check that the analysis was saved
    mock_save_json.assert_called_once()


@patch("agentoptim.analysis.get_experiment")
@patch("agentoptim.analysis.get_job")
@patch("agentoptim.analysis.save_json")
def test_analyze_job_results(mock_save_json, mock_get_job, mock_get_experiment, 
                             mock_experiment, mock_job):
    """Test analyzing job results."""
    mock_get_experiment.return_value = mock_experiment
    mock_get_job.return_value = mock_job
    
    analysis = analyze_experiment_results("exp123", job_id="job123", name="Job Analysis")
    
    # Check that the analysis was created correctly
    assert analysis.experiment_id == "exp123"
    assert analysis.job_id == "job123"
    assert analysis.name == "Job Analysis"
    assert len(analysis.variant_results) == 2
    
    # Check that the analysis was saved
    mock_save_json.assert_called_once()


@patch("agentoptim.analysis.get_analysis")
@patch("agentoptim.analysis.get_experiment")
def test_compare_analyses(mock_get_experiment, mock_get_analysis, mock_experiment):
    """Test comparing multiple analyses."""
    # Create two mock analyses
    analysis1 = ResultsAnalysis(
        id="a1",
        experiment_id="exp123",
        name="Analysis 1",
        variant_results={
            "var1": VariantAnalysis(
                variant_id="var1",
                variant_name="Variant 1",
                sample_size=4,
                average_scores={"clarity": 4.375, "accuracy": 3.8}
            ),
            "var2": VariantAnalysis(
                variant_id="var2",
                variant_name="Variant 2",
                sample_size=4,
                average_scores={"clarity": 3.95, "accuracy": 4.4}
            )
        },
        best_variants={"clarity": "var1", "accuracy": "var2"},
        overall_best_variant="var2"
    )
    
    analysis2 = ResultsAnalysis(
        id="a2",
        experiment_id="exp456",
        name="Analysis 2",
        variant_results={
            "var3": VariantAnalysis(
                variant_id="var3",
                variant_name="Variant 3",
                sample_size=4,
                average_scores={"clarity": 4.5, "accuracy": 4.0}
            ),
            "var4": VariantAnalysis(
                variant_id="var4",
                variant_name="Variant 4",
                sample_size=4,
                average_scores={"clarity": 4.1, "accuracy": 4.2}
            )
        },
        best_variants={"clarity": "var3", "accuracy": "var4"},
        overall_best_variant="var3"
    )
    
    mock_get_analysis.side_effect = lambda id: {
        "a1": analysis1,
        "a2": analysis2
    }.get(id)
    
    mock_get_experiment.return_value = mock_experiment
    
    # Compare the analyses
    comparison = compare_analyses(["a1", "a2"])
    
    # Check that the comparison was created correctly
    assert len(comparison["analyses"]) == 2
    assert len(comparison["overall_best_variants"]) == 2
    assert len(comparison["performance_by_criterion"]) == 2
    assert "clarity" in comparison["performance_by_criterion"]
    assert "accuracy" in comparison["performance_by_criterion"]


@patch("agentoptim.analysis.list_analyses")
def test_analyze_results_list(mock_list_analyses):
    """Test listing analyses."""
    mock_list_analyses.return_value = [
        {"id": "a1", "name": "Analysis 1"},
        {"id": "a2", "name": "Analysis 2"}
    ]
    
    result = analyze_results(action="list")
    
    assert result["items"][0]["name"] == "Analysis 1"
    assert result["items"][1]["name"] == "Analysis 2"


@patch("agentoptim.analysis.get_analysis")
def test_analyze_results_get(mock_get_analysis):
    """Test getting a specific analysis."""
    mock_analysis = ResultsAnalysis(
        id="a1",
        experiment_id="exp123",
        name="Analysis 1",
        variant_results={
            "var1": VariantAnalysis(
                variant_id="var1",
                variant_name="Variant 1",
                sample_size=4,
                average_scores={"clarity": 4.375, "accuracy": 3.8}
            )
        },
        best_variants={"clarity": "var1", "accuracy": "var1"},
        overall_best_variant="var1",
        recommendations=["Variant 1 is best overall"]
    )
    
    mock_get_analysis.return_value = mock_analysis
    
    result = analyze_results(action="get", analysis_id="a1")
    
    assert "Analysis 1" in result["message"]
    assert "Variant 1" in result["message"]
    assert "Best Variants" in result["message"]
    assert "Recommendations" in result["message"]


@patch("agentoptim.analysis.analyze_experiment_results")
def test_analyze_results_analyze(mock_analyze_experiment):
    """Test analyzing experiment results."""
    mock_analysis = ResultsAnalysis(
        id="a1",
        experiment_id="exp123",
        name="New Analysis",
        variant_results={
            "var1": VariantAnalysis(
                variant_id="var1",
                variant_name="Variant 1",
                sample_size=4,
                average_scores={"clarity": 4.375}
            )
        },
        overall_best_variant="var1"
    )
    
    mock_analyze_experiment.return_value = mock_analysis
    
    result = analyze_results(
        action="analyze",
        experiment_id="exp123",
        name="New Analysis"
    )
    
    assert result["error"] == False
    assert "New Analysis" in result["message"]
    assert "Variant 1" in result["message"]


@patch("agentoptim.analysis.delete_analysis")
def test_analyze_results_delete(mock_delete_analysis):
    """Test deleting an analysis."""
    mock_delete_analysis.return_value = True
    
    result = analyze_results(action="delete", analysis_id="a1")
    
    assert result["error"] == False
    assert "deleted" in result["message"]


@patch("agentoptim.analysis.compare_analyses")
def test_analyze_results_compare(mock_compare_analyses):
    """Test comparing analyses."""
    # Create a mock comparison result
    mock_comparison = {
        "analyses": [
            {"id": "a1", "name": "Analysis 1", "experiment_name": "Exp 1", "variant_count": 2},
            {"id": "a2", "name": "Analysis 2", "experiment_name": "Exp 2", "variant_count": 2}
        ],
        "overall_best_variants": [
            {"analysis_id": "a1", "variant_id": "var1", "variant_name": "Variant 1"},
            {"analysis_id": "a2", "variant_id": "var3", "variant_name": "Variant 3"}
        ],
        "performance_by_criterion": {
            "clarity": [
                {"analysis_id": "a1", "variant_id": "var1", "variant_name": "Variant 1", "score": 4.375},
                {"analysis_id": "a2", "variant_id": "var3", "variant_name": "Variant 3", "score": 4.5}
            ]
        }
    }
    
    mock_compare_analyses.return_value = mock_comparison
    
    result = analyze_results(action="compare", analysis_ids=["a1", "a2"])
    
    assert result["error"] == False
    assert "Analysis Comparison" in result["message"]
    assert "Analysis 1" in result["message"]
    assert "Analysis 2" in result["message"]
    assert "Best Overall Variants" in result["message"]
    assert "Performance by Criterion" in result["message"]


# Additional tests to improve coverage

@patch("agentoptim.analysis.get_analysis")
def test_analyze_results_get_not_found(mock_get_analysis):
    """Test getting a non-existent analysis."""
    mock_get_analysis.return_value = None
    
    result = analyze_results(action="get", analysis_id="nonexistent")
    
    assert result["error"] == True
    assert "not found" in result["message"]


@patch("agentoptim.analysis.delete_analysis")
def test_analyze_results_delete_not_found(mock_delete_analysis):
    """Test deleting a non-existent analysis."""
    mock_delete_analysis.return_value = False
    
    result = analyze_results(action="delete", analysis_id="nonexistent")
    
    assert result["error"] == True
    assert "not found" in result["message"]


def test_analyze_results_compare_validation():
    """Test compare validation."""
    # Test with missing analysis_ids
    result = analyze_results(action="compare")
    assert result["error"] == True
    assert "required" in result["message"].lower()
    
    # Test with invalid analysis_ids (not a list)
    result = analyze_results(action="compare", analysis_ids="not-a-list")
    assert result["error"] == True
    assert "must be a list" in result["message"]
    
    # Test with too few analysis_ids
    result = analyze_results(action="compare", analysis_ids=["single-id"])
    assert result["error"] == True
    assert "at least two" in result["message"]


@patch("agentoptim.analysis.get_experiment")
def test_analyze_experiment_results_missing_experiment(mock_get_experiment):
    """Test analyzing with a missing experiment."""
    mock_get_experiment.return_value = None
    
    with pytest.raises(ValueError) as exc_info:
        analyze_experiment_results("nonexistent")
    
    assert "not found" in str(exc_info.value)


@patch("agentoptim.analysis.get_experiment")
def test_analyze_experiment_results_no_results(mock_get_experiment, mock_experiment):
    """Test analyzing an experiment with no results."""
    # Create experiment with no results
    experiment = mock_experiment
    experiment.results = None
    mock_get_experiment.return_value = experiment
    
    with pytest.raises(ValueError) as exc_info:
        analyze_experiment_results("exp123")
    
    assert "no results" in str(exc_info.value)


@patch("agentoptim.analysis.get_experiment")
@patch("agentoptim.analysis.get_job")
def test_analyze_experiment_results_job_not_found(mock_get_job, mock_get_experiment, mock_experiment):
    """Test analyzing with a missing job."""
    mock_get_experiment.return_value = mock_experiment
    mock_get_job.return_value = None
    
    with pytest.raises(ValueError) as exc_info:
        analyze_experiment_results("exp123", job_id="nonexistent")
    
    assert "not found" in str(exc_info.value)


@patch("agentoptim.analysis.get_experiment")
@patch("agentoptim.analysis.get_job")
def test_analyze_experiment_results_job_wrong_experiment(mock_get_job, mock_get_experiment, mock_experiment, mock_job):
    """Test analyzing with a job from a different experiment."""
    mock_get_experiment.return_value = mock_experiment
    
    # Modify job to have different experiment_id
    job = mock_job
    job.experiment_id = "wrong-exp"
    mock_get_job.return_value = job
    
    with pytest.raises(ValueError) as exc_info:
        analyze_experiment_results("exp123", job_id="job123")
    
    assert "not associated" in str(exc_info.value)


@patch("agentoptim.analysis.get_experiment")
@patch("agentoptim.analysis.get_job")
def test_analyze_experiment_results_job_no_results(mock_get_job, mock_get_experiment, mock_experiment, mock_job):
    """Test analyzing a job with no results."""
    mock_get_experiment.return_value = mock_experiment
    
    # Modify job to have no results
    job = mock_job
    job.results = []
    mock_get_job.return_value = job
    
    with pytest.raises(ValueError) as exc_info:
        analyze_experiment_results("exp123", job_id="job123")
    
    assert "no results" in str(exc_info.value)


def test_compute_variant_statistics_empty():
    """Test computing statistics with empty scores."""
    # Test with empty dictionary
    analysis = compute_variant_statistics("var1", "Variant 1", {})
    assert analysis.variant_id == "var1"
    assert analysis.variant_name == "Variant 1"
    assert analysis.sample_size == 0
    assert not analysis.average_scores
    
    # Test with dictionary with empty lists
    analysis = compute_variant_statistics("var1", "Variant 1", {"criterion1": []})
    assert analysis.variant_id == "var1"
    assert analysis.variant_name == "Variant 1"
    assert analysis.sample_size == 0
    assert not analysis.average_scores


@patch("agentoptim.analysis.os.path.exists")
@patch("agentoptim.analysis.os.remove")
def test_delete_analysis_success(mock_remove, mock_exists):
    """Test successful deletion of an analysis."""
    mock_exists.return_value = True
    mock_remove.return_value = None
    
    result = delete_analysis("test123")
    
    assert result == True
    mock_remove.assert_called_once()


@patch("agentoptim.analysis.os.path.exists")
def test_delete_analysis_not_found(mock_exists):
    """Test deletion of a non-existent analysis."""
    mock_exists.return_value = False
    
    result = delete_analysis("nonexistent")
    
    assert result == False


def test_analyze_results_invalid_action():
    """Test analyze_results with an invalid action."""
    result = analyze_results(action="invalid")
    
    assert result["error"] == True
    assert "Invalid action" in result["message"]


def test_analyze_results_unexpected_error():
    """Test analyze_results handles unexpected errors."""
    with patch("agentoptim.analysis.validate_action", side_effect=Exception("Test exception")):
        result = analyze_results(action="list")
        
        assert result["error"] == True
        assert "Unexpected error" in result["message"]
        assert "Test exception" in result["message"]


def test_analyze_results_validation_error():
    """Test analyze_results handles validation errors."""
    # Missing required parameters for analyze
    result = analyze_results(action="analyze")
    
    assert result["error"] == True
    assert "required" in result["message"].lower()
    
    # Missing required parameters for get
    result = analyze_results(action="get")
    
    assert result["error"] == True
    assert "required" in result["message"].lower()
    
    # Missing required parameters for delete
    result = analyze_results(action="delete")
    
    assert result["error"] == True
    assert "required" in result["message"].lower()