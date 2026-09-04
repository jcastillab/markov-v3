import numpy as np
import pandas as pd

from src.models.bayes import HierarchicalNB


def test_hierarchical_nb_predictions_are_nonnegative_and_finite():
    frame = pd.DataFrame({"finca": ["A", "A"], "bloque": ["1", "1"],
                          "horizonte_dia": [1, 2], "target": [10., 20.]})
    model = HierarchicalNB(10).fit(frame)
    pred = model.predict(frame)
    assert np.isfinite(pred).all() and (pred >= 0).all()


def test_hierarchical_nb_uses_group_specific_gamma_posterior():
    frame = pd.DataFrame({"finca": ["A", "A"], "bloque": ["1", "1"],
                          "horizonte_dia": [1, 1], "target": [10., 20.]})
    model = HierarchicalNB(10).fit(frame)
    # a0 = k * global_mean = 150, b0 = k = 10; group posterior adds sum and n.
    assert model.group_posterior[("A", "1", 1)] == (180.0, 12.0)
    assert np.isclose(model.predict(frame)[0], 15.0)


def test_hierarchical_nb_intervals_follow_posterior_parameters():
    frame = pd.DataFrame({"finca": ["A"], "bloque": ["1"],
                          "horizonte_dia": [1], "target": [10.]})
    model = HierarchicalNB(10).fit(frame)
    intervals = model.predictive_interval(frame, draws=2000, seed=7)
    assert intervals.shape == (4, 1)
    assert intervals[0, 0] <= intervals[1, 0] <= intervals[3, 0]


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


def test_dirichlet_posterior_adds_event_counts_to_m3_prior():
    from src.models.bayes import DirichletM3
    from src.models.m3 import M3Matrix
    matrix = M3Matrix("A", "JULIO", np.eye(3) * .8,
                      np.array([.1, .1, .1]), np.array([.1, .1, .1]), pd.DataFrame())
    intervals = pd.DataFrame({"estado_origen": ["AP", "AP"],
                              "evento": ["CUT", "LOSS"]})
    posterior = DirichletM3(intervals, matrix, prior_strength=5)
    assert np.allclose(posterior.alpha["AP"], [4.0, 1.5, 1.5])
