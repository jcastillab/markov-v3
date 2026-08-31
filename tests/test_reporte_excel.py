from src.reporte_excel import weekly_status


def test_weekly_status_uses_requested_semaphore_thresholds():
    assert weekly_status(93) == "ACIERTO"
    assert weekly_status(107) == "ACIERTO"
    assert weekly_status(92.9) == "CERCA"
    assert weekly_status(109.9) == "CERCA"
    assert weekly_status(89.9) == "NO ACIERTO"
