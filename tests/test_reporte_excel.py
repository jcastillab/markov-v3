from src.reporte_excel import weekly_status
from src.evaluation.metrics import metrics


def test_weekly_status_uses_requested_semaphore_thresholds():
    assert weekly_status(0.93) == "ACIERTO"
    assert weekly_status(1.07) == "ACIERTO"
    assert weekly_status(0.929) == "CERCA"
    assert weekly_status(1.099) == "CERCA"
    assert weekly_status(0.899) == "NO ACIERTO"


def test_metrics_includes_r2_for_regression_models():
    result = metrics(__import__("pandas").Series([1, 2, 3]), __import__("pandas").Series([1, 2, 4]))
    assert "r2" in result
    assert result["r2"] < 1
