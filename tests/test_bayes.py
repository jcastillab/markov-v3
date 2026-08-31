import numpy as np
import pandas as pd

from src.models.bayes import HierarchicalNB


def test_hierarchical_nb_predictions_are_nonnegative_and_finite():
    frame = pd.DataFrame({"finca": ["A", "A"], "bloque": ["1", "1"],
                          "horizonte_dia": [1, 2], "target": [10., 20.]})
    model = HierarchicalNB(10).fit(frame)
    pred = model.predict(frame)
    assert np.isfinite(pred).all() and (pred >= 0).all()


def test_dirichlet_draws_are_probability_simplex():
    from src.models.bayes import DirichletM3
    from src.models.m3 import M3Matrix
    matrix = M3Matrix("A", "JULIO", np.array([[.8, 0, 0], [.1, .8, 0], [0, .1, .8]]),
                      np.array([.05, .05, .1]), np.array([.05, .05, .1]), pd.DataFrame())
    intervals = pd.DataFrame(columns=["estado_origen", "evento"])
    posterior = DirichletM3(intervals, matrix, 5)
    q, r, loss = posterior.draw_matrix()
    assert np.allclose(q.sum(axis=0) + r + loss, 1)


def test_dirichlet_ap_uses_cut_once():
    from src.models.bayes import DirichletM3
    from src.models.m3 import M3Matrix
    matrix = M3Matrix("A", "JULIO", np.eye(3) * .8,
                      np.array([.1, .1, .1]), np.array([.1, .1, .1]), pd.DataFrame())
    posterior = DirichletM3(pd.DataFrame({"estado_origen": ["AP"] * 3,
                                          "evento": ["STAY", "CUT", "LOSS"]}), matrix, 5)
    q, r, loss = posterior.draw_matrix()
    assert np.isclose(q[2, 2] + r[2] + loss[2], 1)
