"""Integration tests against the real DreamSim model.

Skipped unless the package is installed AND weights are already cached, so a
normal test run never triggers a ~1 GB download. Populate the cache with:

    .venv/bin/python -c "import sys; sys.path.insert(0,'src'); \
import metrics; from dreamsim_metric import DREAMSIM; import torch; \
m = DREAMSIM.slice_mode.factory(); print(m(torch.rand(1,3,96,96), torch.rand(1,3,96,96)))"

`metrics` has to be imported first: it and dreamsim_metric form the same
deliberate import cycle the segmentation_metrics modules use.
"""
import pytest
import torch

pytest.importorskip("dreamsim")

from constants import DREAMSIM_CACHE
import metrics  # noqa: F401 — must precede dreamsim_metric; see the cycle note there
from dreamsim_metric import dreamsim_spec

pytestmark = pytest.mark.skipif(
    not (DREAMSIM_CACHE.exists() and any(DREAMSIM_CACHE.iterdir())),
    reason=f"no cached DreamSim weights in {DREAMSIM_CACHE}",
)


def _phantom(n: int = 1, *, noise: float = 0.0, seed: int = 0) -> torch.Tensor:
    """A crude brain-like phantom: a bright disc on dark background, plus noise."""
    generator = torch.Generator().manual_seed(seed)
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, 96),
                            torch.linspace(-1, 1, 96), indexing="ij")
    disc = ((xx ** 2 + yy ** 2) < 0.5).float() * 0.8 + 0.1
    batch = disc.expand(n, 3, 96, 96).clone()
    if noise:
        batch = (batch + torch.randn(batch.shape, generator=generator) * noise).clamp(0, 1)
    return batch


@pytest.fixture(scope="module")
def ensemble():
    """One loaded ensemble model, shared by every test in this module."""
    return dreamsim_spec().slice_mode.factory()


class TestRealModelScores:
    def test_identical_images_score_near_zero(self, ensemble):
        reference = _phantom()
        assert ensemble(reference, reference)[0] == pytest.approx(0.0, abs=0.02)

    def test_more_noise_scores_worse(self, ensemble):
        reference = _phantom()
        mild  = ensemble(_phantom(noise=0.05, seed=1), reference)[0]
        heavy = ensemble(_phantom(noise=0.40, seed=1), reference)[0]
        assert heavy > mild

    def test_batch_returns_one_score_per_pair(self, ensemble):
        reference = _phantom(3)
        distorted = torch.cat([_phantom(1, noise=n, seed=2)
                               for n in (0.0, 0.15, 0.45)])
        scores = ensemble(distorted, reference)
        assert len(scores) == 3
        assert scores[0] < scores[1] < scores[2]

    def test_single_backbone_type_also_works(self):
        metric = dreamsim_spec(dreamsim_type="dino_vitb16").slice_mode.factory()
        reference = _phantom()
        assert metric(reference, reference)[0] == pytest.approx(0.0, abs=0.02)
