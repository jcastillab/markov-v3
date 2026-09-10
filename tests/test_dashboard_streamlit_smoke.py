from pathlib import Path

from streamlit.testing.v1 import AppTest


ROOT = Path(__file__).resolve().parents[1]


def test_dashboard_starts_without_uncaught_exceptions():
    app = AppTest.from_file(str(ROOT / "src" / "dashboard_streamlit.py"))
    app.run(timeout=30)
    assert not app.exception
    assert app.title[0].value == "Markov Freedom"
