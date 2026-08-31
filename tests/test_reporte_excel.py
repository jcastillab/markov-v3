from src.reporte_excel import weekly_status


def test_weekly_status_uses_requested_semaphore_thresholds():
    assert weekly_status(0.93) == "ACIERTO"
    assert weekly_status(1.07) == "ACIERTO"
    assert weekly_status(0.929) == "CERCA"
    assert weekly_status(1.099) == "CERCA"
    assert weekly_status(0.899) == "NO ACIERTO"
