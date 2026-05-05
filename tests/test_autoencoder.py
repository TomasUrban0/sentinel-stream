"""Smoke tests for the dense autoencoder detector.

Kept intentionally small (few epochs, low dimensionality) so the test runs in
seconds on CPU and does not require a GPU in CI.
"""

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from sentinel_stream.models.autoencoder import AutoencoderDetector  # noqa: E402


def test_autoencoder_fit_score_save_load(tmp_path):
    rng = np.random.default_rng(0)
    x_train = rng.normal(size=(400, 6)).astype(np.float32)

    detector = AutoencoderDetector(
        hidden_layers=[8, 4, 8],
        epochs=3,
        batch_size=64,
        threshold_percentile=95.0,
    )
    detector.fit(x_train)

    assert detector.threshold > 0.0
    train_scores = detector.score(x_train)
    assert train_scores.shape == (400,)

    x_eval = np.vstack(
        [rng.normal(size=(40, 6)), rng.normal(loc=8.0, size=(10, 6))]
    ).astype(np.float32)
    eval_scores = detector.score(x_eval)
    assert eval_scores.shape == (50,)
    assert eval_scores[40:].mean() > eval_scores[:40].mean()

    detector.save(str(tmp_path))
    loaded = AutoencoderDetector.load(str(tmp_path))
    np.testing.assert_allclose(loaded.score(x_eval), eval_scores, rtol=1e-4, atol=1e-5)
    assert loaded.threshold == pytest.approx(detector.threshold)
