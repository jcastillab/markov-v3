from src.reporte_excel import weekly_status


def test_rolling_semaphore_boundaries_remain_consistent():
    assert weekly_status(0.93) == "ACIERTO"
    assert weekly_status(1.10) == "CERCA"
