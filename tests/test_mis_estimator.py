"""Deterministic unit tests independent of Mitsuba and Torch."""
import unittest
import numpy as np
from neural_path_guiding.core.mis_estimator import combine_mis_samples
from neural_path_guiding.core.mis import power_heuristic_weights
import itertools


class MultiTechniqueEstimatorTests(unittest.TestCase):
    def test_identical_techniques_constant_integrand(self):
        rgb = [np.ones((2, 3)), np.ones((3, 3)), np.ones((4, 3))]
        pdf = [np.ones((n, 3)) for n in (2, 3, 4)]
        np.testing.assert_allclose(combine_mis_samples(rgb, pdf), [1, 1, 1])

    def test_three_techniques_exact_discrete_expectation(self):
        # Discrete 3-point domain; sum over every possible outcome analytically.
        strategies = np.array([[.6, .3, .1], [.2, .5, .3], [.1, .2, .7]])
        target = np.array([[1., 2., 3.], [.5, .4, .3], [2., 1., 4.]])
        n = [1, 2, 3]
        expected = np.zeros(3)
        for i in range(3):
            for x in range(3):
                # One sample's term, multiplied by n_i to account for all draws.
                rgb = [np.empty((0, 3)) for _ in range(3)]
                pdf = [np.empty((0, 3)) for _ in range(3)]
                rgb[i] = target[x][None, :]
                pdf[i] = strategies[:, x][None, :]
                # Actual sample counts for weights must reflect intended n, not
                # sparse arrays, so compute weights explicitly here.
                w = power_heuristic_weights(strategies[:, x].tolist(), sample_counts=n)[i]
                expected += strategies[i, x] * w * target[x] / strategies[i, x]
        np.testing.assert_allclose(expected, target.sum(axis=0), rtol=1e-12)

    def test_unbiased_via_complete_outcomes(self):
        # One sample per technique. Enumerate all 3^3 possible independent draws.
        ps = np.array([[.6, .3, .1], [.2, .5, .3], [.1, .2, .7]])
        f = np.array([[1., 0., 3.], [2., 4., 1.], [0.5, 2., 2.]])
        avg = np.zeros(3)
        for xs in itertools.product(range(3), repeat=3):
            weight = np.prod([ps[i, x] for i, x in enumerate(xs)])
            cs = [f[x][None, :] for x in xs]
            qs = [ps[:, x][None, :] for x in xs]
            avg += weight * combine_mis_samples(cs, qs)
        np.testing.assert_allclose(avg, f.sum(axis=0), rtol=1e-12)

    def test_invalid_own_pdf(self):
        with self.assertRaisesRegex(ValueError, 'own PDF'):
            combine_mis_samples([np.ones((1, 3)), np.ones((1, 3))],
                                [np.array([[0., 1.]]), np.array([[1., 1.]])])

    def test_nonfinite_pdf(self):
        with self.assertRaisesRegex(ValueError, 'finite'):
            combine_mis_samples([np.ones((1, 3)), np.ones((1, 3))],
                                [np.array([[np.nan, 1.]]), np.array([[1., 1.]])])

    def test_empty_technique(self):
        got = combine_mis_samples(
            [np.ones((1, 3)), np.empty((0, 3)), np.ones((1, 3))],
            [np.ones((1, 3)), np.empty((0, 3)), np.ones((1, 3))]
        )
        np.testing.assert_allclose(got, np.ones(3))


if __name__ == '__main__':
    unittest.main()
