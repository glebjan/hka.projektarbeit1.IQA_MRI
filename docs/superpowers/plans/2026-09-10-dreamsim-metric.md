# DreamSim Metric Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add DreamSim as a configurable full-reference metric through a standalone `Metric` adapter, with all six upstream parameters exposed, documented, and validated.

**Architecture:** A new module `src/dreamsim_metric.py` holds `DreamSimMetric` (implements the `metrics.Metric` protocol directly — no pyiqa) and the spec factory `dreamsim_spec()`, which returns a configured `MetricSpec`. Configuration lives in the factory closure, exactly as `_pyiqa_factory("radimagenet_lpips", backbone_path=...)` already does, so the shared `MetricSpec` dataclass is not touched. The metric name is derived deterministically from the configuration so two variants can coexist in one registry. `metrics.py` imports the module at the bottom of the file (the same deliberate cycle `segmentation_metrics` uses). `DREAMSIM` stays out of `BUILTIN_METRICS` — opt-in only. `IQAEvaluator`, `VolumeEvaluator`, `EvaluationResult`, and `MaskWriter` are not touched at all.

**Tech Stack:** Python 3.14.6, torch 2.12.0, dreamsim 0.2.1 (+ peft, open-clip-torch), pytest. venv at `.venv/`.

**Spec:** `docs/superpowers/specs/2026-09-09-dreamsim-metric-design.md` — read it before starting; every decision below is argued there.

## Global Constraints

- Python must be run from the venv: `source .venv/bin/activate`, or invoke `.venv/bin/python` / `.venv/bin/pytest` directly.
- Tests import src modules with bare names (`from metrics import ...`); `tests/conftest.py` puts `src/` on `sys.path`. Run pytest **from the repository root**, never from `src/`.
- Code, comments, and docstrings are written in **English**, matching the whole codebase.
- Commit style: Conventional Commits. Every commit message ends with the trailer `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>`.
- Metric name of the default configuration, verbatim and lowercase: `dreamsim`. Variant names: `dreamsim_<type>`, with suffixes appended in the fixed order type → `_patch` → `_scratch` → `_rawembeds`.
- `direction="lower_is_better"`, `reference=True`, `channels="rgb"`, `builtin=(name == "dreamsim")`, `volume_mode=ModeUnsupported(REASON_DEEP_2D)`. Do **not** introduce a new reason constant.
- Do **not** pass `description=` or `domain=` — no pyiqa-backed built-in sets them; all explanation lives in docstrings.
- Do **not** add `DREAMSIM` to `BUILTIN_METRICS`. It stays 18 entries, so `TestBuiltinCapabilities` and `TestBuiltinMetrics` in `tests/test_metrics.py` must keep passing unchanged.
- Do **not** touch `src/iqa_evaluator.py`, `src/volume_evaluator.py`, `src/evaluator_factory.py`, `src/evaluation_result.py`, or `src/records.py` beyond the single new field in Task 4.
- Do **not** change the global `DEVICE` in `src/metrics.py`. MPS is reached only through `dreamsim_spec(device="mps")`.
- Do **not** add a `THRESHOLDS` entry for `dreamsim` in the notebook. `THRESHOLDS.get(col, [])` degrades to "histogram without threshold lines", and no MRI literature values exist for DreamSim.
- Upstream input contract: exactly `(N, 3, 224, 224)`, float in `[0, 1]`. Higher score = more different.
- The working tree currently carries an unrelated `M src/metrics.py` (one blank line at line 30) and an untracked `src/gl.py`. Leave both alone; never `git add -A`, always add named paths.

---

## File Structure

| File | Change | Responsibility |
|---|---|---|
| `src/dreamsim_metric.py` | Create | `DreamSimMetric` adapter, `dreamsim_spec()` factory, `DREAMSIM` constant, `VALID_TYPES`, `PATCH_CAPABLE`, name derivation. All DreamSim knowledge lives here and nowhere else. |
| `src/constants.py` | Modify | `DREAMSIM_CACHE` — the weight cache path, single source of truth. |
| `src/metrics.py` | Modify | One late import so `DREAMSIM` / `dreamsim_spec` are reachable from `metrics`; one docstring line. |
| `src/records.py` | Modify | One `dreamsim: Optional[float]` field, so `setattr(record, "dreamsim", value)` lands on a real column. |
| `src/main.py` | Modify | Two names added to the convenience re-export block. |
| `.gitignore` | Modify | `models/dreamsim/` — keeps ~1 GB of weights out of git. |
| `requirements.txt` | Modify | `dreamsim`, `peft`, `open-clip-torch` pins from the post-install freeze. |
| `tests/test_dreamsim_metric.py` | Create | Offline unit tests: adapter behaviour and spec factory, against an injected fake loader. |
| `tests/test_dreamsim_integration.py` | Create | Gated tests against the real model (skipped without package or cached weights). |
| `tests/test_records.py` | Modify | The `dreamsim` field exists, defaults to `None`, survives `to_dict()`. |
| `tests/test_metrics.py` | Modify | `DREAMSIM` is reachable from `metrics` and is **not** in `BUILTIN_METRICS`. |
| `CLAUDE.md`, `README.md`, `src/evaluation.ipynb` | Modify | Documentation. |

---

## Task 1: Install the dependency, add the weight-cache path, verify the baseline

The install is the riskiest step in this plan: `dreamsim` shares `timm` and `transformers` with pyiqa's `maniqa`, `musiq`, and `clipiqa`. Verify the suite **before** writing any code, so a dependency regression cannot be mistaken for a bug in new code.

**Files:**
- Modify: `src/constants.py`
- Modify: `.gitignore`
- Modify: `requirements.txt`

**Interfaces:**
- Consumes: nothing.
- Produces: `constants.DREAMSIM_CACHE: Path` = `Path("models/dreamsim")`. Tasks 2, 3, and 5 import it.

- [ ] **Step 1: Record the pre-install baseline**

```bash
.venv/bin/pip freeze > /tmp/freeze-before.txt && .venv/bin/pytest -q 2>&1 | tail -5
```

Expected: the suite passes (462 tests as of 2026-09-09). Write the exact number down — Task 6 compares against it. If it already fails before you install anything, **stop and report**; do not continue.

- [ ] **Step 2: Install dreamsim**

```bash
.venv/bin/pip install dreamsim
```

Expected: `dreamsim 0.2.1` plus `peft` and `open-clip-torch`. PyPI metadata for `dreamsim 0.2.1` carries no version pins, so torch should stay at 2.12.0.

- [ ] **Step 3: Inspect what the install changed**

```bash
.venv/bin/pip freeze > /tmp/freeze-after.txt && diff /tmp/freeze-before.txt /tmp/freeze-after.txt
```

Read the diff. If `torch`, `torchvision`, `timm`, or `transformers` changed version, note the old and new values in the commit message — those four are shared with pyiqa.

- [ ] **Step 4: Re-run the full suite on the new dependency set**

```bash
.venv/bin/pytest -q 2>&1 | tail -15
```

Expected: same pass count as Step 1, no new failures. **If anything now fails, stop and report** with the failing test names and the freeze diff. Do not "fix" a pyiqa test to accommodate the install — that decision belongs to the user.

- [ ] **Step 5: Add the weight-cache path to constants.py**

`src/constants.py` in full after the edit:

```python
from pathlib import Path

RESNET50 = Path("models/RadImageNet_pytorch/ResNet50.pt")

# DreamSim downloads its ViT/LoRA weights here on first use (~1 GB for the
# default "ensemble" type). Excluded from git via .gitignore — unlike
# RESNET50, these weights are fetched automatically and need not be tracked.
DREAMSIM_CACHE = Path("models/dreamsim")

REPORT = Path("report") / "IXI661_report.csv"
```

- [ ] **Step 6: Keep the weights out of git**

`.gitignore` currently unignores everything under `models/`:

```
!models/
!models/**
```

Add one line immediately after that pair:

```
# DreamSim weights are downloaded on demand (~1 GB) — never commit them.
models/dreamsim/
```

- [ ] **Step 7: Verify the ignore rule actually works**

```bash
mkdir -p models/dreamsim && touch models/dreamsim/probe.bin && git status --short models/ && rm models/dreamsim/probe.bin
```

Expected: `git status --short models/` prints **nothing** for `models/dreamsim/probe.bin`. If the file shows up as untracked, the rule is in the wrong place — it must come after the `!models/**` unignore.

- [ ] **Step 8: Pin the new dependencies**

Read the three versions out of the freeze:

```bash
grep -iE "^(dreamsim|peft|open.clip.torch)==" /tmp/freeze-after.txt
```

Add those three lines to `requirements.txt` in alphabetical position (the file is alphabetically sorted, lowercase-insensitive: `dreamsim` after `distlib`, `open-clip-torch` after `numpy`/before `openai-clip`, `peft` after `pandas`). Use the exact strings the grep printed. Do not touch the pre-existing stale `torch==2.8.0` pin — that is a documented follow-up, not part of this change.

- [ ] **Step 9: Commit**

```bash
git add requirements.txt src/constants.py .gitignore
git commit -m "$(cat <<'EOF'
build(deps): add dreamsim and its weight cache path

DreamSim ships its own loader, so the framework needs the package plus a
place to keep the downloaded ViT/LoRA weights. The cache lives at
models/dreamsim and is git-ignored: unlike the RadImageNet weights it is
fetched on demand and runs to about a gigabyte for the ensemble type.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: The DreamSimMetric adapter

**Files:**
- Create: `src/dreamsim_metric.py`
- Test: `tests/test_dreamsim_metric.py`

**Interfaces:**
- Consumes: `constants.DREAMSIM_CACHE` (Task 1); `metrics.DEVICE`.
- Produces:
  - `INPUT_SIZE: int = 224`
  - `VALID_TYPES: tuple[str, ...]`, `PATCH_CAPABLE: tuple[str, ...]`
  - `_import_dreamsim() -> Callable` — module-level, monkeypatchable
  - `class DreamSimMetric` with keyword-only `__init__(*, dreamsim_type="ensemble", pretrained=True, device=None, cache_dir=DREAMSIM_CACHE, normalize_embeds=True, use_patch_model=False, loader=None)` and `__call__(input: torch.Tensor, target: Optional[torch.Tensor] = None) -> list[float]`
  - Task 3 calls this constructor from inside the spec factory.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_dreamsim_metric.py`:

```python
"""Tests for src/dreamsim_metric.py — the DreamSim adapter and its spec factory.

Everything here runs offline against a fake loader: no dreamsim import, no
weight download. The real model is exercised in test_dreamsim_integration.py.
"""
import sys

import pytest
import torch

from constants import DREAMSIM_CACHE
from dreamsim_metric import (
    INPUT_SIZE,
    PATCH_CAPABLE,
    VALID_TYPES,
    DreamSimMetric,
    _import_dreamsim,
)
from metrics import DEVICE


class FakeModel:
    """Stands in for the DreamSim model: records its inputs, returns distances."""

    def __init__(self, *, scalar: bool = False):
        self.scalar = scalar
        self.calls: list[tuple[torch.Size, torch.Size]] = []

    def __call__(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        self.calls.append((img1.shape, img2.shape))
        if self.scalar:
            return torch.tensor(0.5)
        return torch.arange(img1.shape[0], dtype=torch.float32) / 10.0


def make_loader(*, scalar: bool = False):
    """Return a callable mimicking dreamsim.dreamsim(), with call recording."""
    model = FakeModel(scalar=scalar)
    kwargs_seen: list[dict] = []

    def loader(**kwargs):
        kwargs_seen.append(kwargs)
        return model, (lambda img: img)

    loader.model = model
    loader.kwargs_seen = kwargs_seen
    return loader


def pair(n: int = 2, c: int = 3, h: int = 96, w: int = 96):
    """An input/target batch pair in the framework's (N, C, H, W) [0,1] format."""
    generator = torch.Generator().manual_seed(0)
    return (torch.rand(n, c, h, w, generator=generator),
            torch.rand(n, c, h, w, generator=generator))


class TestLazyLoading:
    def test_constructor_does_not_load(self):
        loader = make_loader()
        DreamSimMetric(loader=loader)
        assert loader.kwargs_seen == []

    def test_first_call_loads_once(self):
        loader = make_loader()
        metric = DreamSimMetric(loader=loader)
        inp, ref = pair()
        metric(inp, ref)
        metric(inp, ref)
        assert len(loader.kwargs_seen) == 1


class TestLoaderArguments:
    def test_all_six_parameters_reach_the_loader(self):
        loader = make_loader()
        metric = DreamSimMetric(
            dreamsim_type="dino_vitb16",
            pretrained=False,
            device="cpu",
            cache_dir=DREAMSIM_CACHE,
            normalize_embeds=False,
            use_patch_model=True,
            loader=loader,
        )
        inp, ref = pair()
        metric(inp, ref)
        kwargs = loader.kwargs_seen[0]
        assert kwargs["dreamsim_type"]    == "dino_vitb16"
        assert kwargs["pretrained"]       is False
        assert kwargs["device"]           == "cpu"
        assert kwargs["cache_dir"]        == str(DREAMSIM_CACHE)
        assert kwargs["normalize_embeds"] is False
        assert kwargs["use_patch_model"]  is True

    def test_device_defaults_to_framework_device(self):
        metric = DreamSimMetric(loader=make_loader())
        assert metric.device == DEVICE

    def test_explicit_device_overrides(self):
        metric = DreamSimMetric(device="cpu", loader=make_loader())
        assert metric.device == torch.device("cpu")


class TestPreparation:
    def test_resizes_both_tensors_to_224(self):
        loader = make_loader()
        metric = DreamSimMetric(loader=loader)
        inp, ref = pair(n=2, h=96, w=64)
        metric(inp, ref)
        shape_in, shape_ref = loader.model.calls[0]
        assert tuple(shape_in)  == (2, 3, INPUT_SIZE, INPUT_SIZE)
        assert tuple(shape_ref) == (2, 3, INPUT_SIZE, INPUT_SIZE)

    def test_expands_single_channel_to_rgb(self):
        loader = make_loader()
        metric = DreamSimMetric(loader=loader)
        inp, ref = pair(n=3, c=1)
        metric(inp, ref)
        shape_in, _ = loader.model.calls[0]
        assert tuple(shape_in) == (3, 3, INPUT_SIZE, INPUT_SIZE)

    def test_already_correct_size_is_untouched(self):
        loader = make_loader()
        metric = DreamSimMetric(loader=loader)
        inp, ref = pair(n=1, h=INPUT_SIZE, w=INPUT_SIZE)
        metric(inp, ref)
        shape_in, _ = loader.model.calls[0]
        assert tuple(shape_in) == (1, 3, INPUT_SIZE, INPUT_SIZE)


class TestScores:
    def test_returns_one_float_per_slice(self):
        metric = DreamSimMetric(loader=make_loader())
        inp, ref = pair(n=4)
        scores = metric(inp, ref)
        assert len(scores) == 4
        assert all(isinstance(s, float) for s in scores)

    def test_scalar_return_falls_back_to_per_pair_scoring(self):
        loader = make_loader(scalar=True)
        metric = DreamSimMetric(loader=loader)
        inp, ref = pair(n=3)
        scores = metric(inp, ref)
        assert scores == [0.5, 0.5, 0.5]
        # One batched attempt, then one call per pair.
        assert len(loader.model.calls) == 4
        assert all(shape[0] == 1 for shape, _ in loader.model.calls[1:])

    def test_single_slice_batch_needs_no_fallback(self):
        loader = make_loader(scalar=True)
        metric = DreamSimMetric(loader=loader)
        inp, ref = pair(n=1)
        assert metric(inp, ref) == [0.5]
        assert len(loader.model.calls) == 1


class TestErrors:
    def test_missing_target_raises(self):
        metric = DreamSimMetric(loader=make_loader())
        inp, _ = pair()
        with pytest.raises(ValueError, match="full-reference"):
            metric(inp)

    def test_absent_package_message_is_actionable(self, monkeypatch):
        # A None entry in sys.modules makes `from dreamsim import ...` raise,
        # which is how the real "package not installed" path behaves.
        monkeypatch.setitem(sys.modules, "dreamsim", None)
        with pytest.raises(ImportError, match="pip install dreamsim"):
            _import_dreamsim()

    def test_loader_failure_names_the_cache_dir(self):
        def broken_loader(**kwargs):
            raise RuntimeError("connection reset")

        metric = DreamSimMetric(loader=broken_loader)
        inp, ref = pair()
        with pytest.raises(RuntimeError, match=str(DREAMSIM_CACHE)):
            metric(inp, ref)
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
.venv/bin/pytest tests/test_dreamsim_metric.py -q
```

Expected: collection error — `ModuleNotFoundError: No module named 'dreamsim_metric'`.

- [ ] **Step 3: Write the adapter**

Create `src/dreamsim_metric.py` (the `dreamsim_spec` factory follows in Task 3 — this file ends after `DreamSimMetric` for now):

```python
"""DreamSim as a full-reference metric.

DreamSim (Fu et al., NeurIPS 2023) is a LoRA-finetuned ViT feature extractor
whose distance is the cosine distance between image embeddings. It was trained
on human two-alternative similarity judgements (the NIGHTS dataset), so it
weighs mid-level structure — layout, shape, arrangement — more than pixel-local
texture. That makes it useful for ranking reconstructions of the same anatomy.

This module is the framework's first non-pyiqa `Metric` implementation.
DreamSim brings its own loader, weight management and device handling, so
wrapping it in pyiqa would only add indirection: `DreamSimMetric` implements the
`metrics.Metric` protocol directly, and `IQAEvaluator` never notices the
difference.

Domain caveat, stated once and meant: both the backbones' pretraining and the
human-judgement finetuning happened on natural photographs. MRI contrast, noise
characteristics and the meaning of a grey value are all different. Treat a
dreamsim score as a *relative ranking within one run*, never as an absolute
quality figure. `radimagenet_lpips` is the medically pretrained perceptual
alternative already in the framework.
"""

from pathlib import Path
from typing import Callable, Optional

import torch
import torch.nn.functional as F

from constants import DREAMSIM_CACHE
from metrics import DEVICE

# DreamSim's ViTs have fixed positional embeddings — 224x224 is not a
# suggestion, it is the only accepted input size.
INPUT_SIZE = 224

VALID_TYPES = (
    "ensemble",
    "dino_vitb16",
    "clip_vitb32",
    "open_clip_vitb32",
    "dinov2_vitb14",
    "synclr_vitb16",
)

# Only these two upstream types ship a patch-feature variant.
PATCH_CAPABLE = ("dino_vitb16", "dinov2_vitb14")


def _import_dreamsim() -> Callable:
    """Return `dreamsim.dreamsim`, or raise with an actionable message.

    Imported lazily and in one place: `metrics.py` must stay importable on a
    machine without the package, and `IQAEvaluator` swallows metric exceptions
    into a printed line, so this message is what the user actually sees.
    """
    try:
        from dreamsim import dreamsim
    except ImportError as exc:
        raise ImportError(
            "dreamsim is not installed — run: pip install dreamsim"
        ) from exc
    return dreamsim


class DreamSimMetric:
    """Adapter that makes DreamSim satisfy the `metrics.Metric` protocol.

    The model is built on the first `__call__`, not in `__init__`, so
    registering the spec costs nothing — no import, no weight download.

    Args:
        dreamsim_type: which backbones produce the embedding. See
            `dreamsim_spec` for what each choice costs and measures.
        pretrained: True loads DreamSim's LoRA weights (finetuned on human
            judgements over natural images); False leaves the backbones raw.
        device: None resolves to the framework's global `DEVICE`. The adapter
            moves its own tensors, because `IQAEvaluator` has already moved them
            to `DEVICE` and an override would otherwise mismatch.
        cache_dir: where weights are downloaded on first use.
        normalize_embeds: L2-normalize embeddings before taking the distance.
        use_patch_model: use dense patch features alongside the CLS token.
            Only available for the types in `PATCH_CAPABLE`.
        loader: injection seam for tests — a callable with
            `dreamsim.dreamsim`'s signature. None means import the real one.
    """

    def __init__(
        self,
        *,
        dreamsim_type:    str = "ensemble",
        pretrained:       bool = True,
        device:           Optional[str | torch.device] = None,
        cache_dir:        Path = DREAMSIM_CACHE,
        normalize_embeds: bool = True,
        use_patch_model:  bool = False,
        loader:           Optional[Callable] = None,
    ):
        self.device    = torch.device(device) if device is not None else DEVICE
        self._cache_dir = Path(cache_dir)
        self._loader   = loader
        self._kwargs   = {
            "dreamsim_type":    dreamsim_type,
            "pretrained":       pretrained,
            "device":           str(self.device),
            "cache_dir":        str(self._cache_dir),
            "normalize_embeds": normalize_embeds,
            "use_patch_model":  use_patch_model,
        }
        self._impl: Optional[Callable] = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load(self) -> Callable:
        """Build the model, annotating any failure with the cache location."""
        loader = self._loader if self._loader is not None else _import_dreamsim()
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        try:
            model, _preprocess = loader(**self._kwargs)
        except ImportError:
            raise
        except Exception as exc:
            raise RuntimeError(
                f"dreamsim failed to load (weight cache: {self._cache_dir}): {exc}"
            ) from exc
        return model

    def _prepare(self, batch: torch.Tensor) -> torch.Tensor:
        """Move to the metric's device, replicate to RGB, resize to 224x224.

        The resize is the same squashing `Resize((224, 224))` upstream's own
        `preprocess` performs, so scores stay comparable to published DreamSim
        numbers and no anatomy is cropped away. The cost is real: high-frequency
        MRI detail — noise texture, thin edges — does not survive the
        downsample, and that detail is often what separates two
        reconstructions.
        """
        batch = batch.to(self.device)
        if batch.shape[1] == 1:
            batch = batch.expand(-1, 3, -1, -1)
        if batch.shape[-2:] != (INPUT_SIZE, INPUT_SIZE):
            batch = F.interpolate(
                batch, size=(INPUT_SIZE, INPUT_SIZE),
                mode="bilinear", align_corners=False, antialias=True,
            )
        return batch

    @staticmethod
    def _as_scores(raw, expected: int) -> Optional[list[float]]:
        """Return `expected` floats, or None if the model did not produce them."""
        flat = torch.as_tensor(raw).detach().reshape(-1)
        return [float(v) for v in flat] if flat.numel() == expected else None

    # ------------------------------------------------------------------
    # Metric protocol
    # ------------------------------------------------------------------

    @torch.no_grad()
    def __call__(
        self,
        input:  torch.Tensor,
        target: Optional[torch.Tensor] = None,
    ) -> list[float]:
        if target is None:
            raise ValueError(
                "dreamsim is a full-reference metric — it needs a target image"
            )
        if self._impl is None:
            self._impl = self._load()

        inp = self._prepare(input)
        ref = self._prepare(target)
        n   = inp.shape[0]

        # Upstream documents model(img1, img2) without committing to batch
        # behaviour, so trust the shape rather than the docs: if the call does
        # not return one score per pair, score the pairs one at a time.
        scores = self._as_scores(self._impl(inp, ref), n)
        if scores is None:
            scores = [
                self._as_scores(self._impl(inp[i:i + 1], ref[i:i + 1]), 1)[0]
                for i in range(n)
            ]
        return scores
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
.venv/bin/pytest tests/test_dreamsim_metric.py -q
```

Expected: all pass. If `test_absent_package_message_is_actionable` fails with a different error class, check that `_import_dreamsim` re-raises `ImportError` and not a bare `Exception`.

- [ ] **Step 5: Confirm nothing else moved**

```bash
.venv/bin/pytest -q 2>&1 | tail -5
```

Expected: the Task 1 pass count plus the new tests, zero failures.

- [ ] **Step 6: Commit**

```bash
git add src/dreamsim_metric.py tests/test_dreamsim_metric.py
git commit -m "$(cat <<'EOF'
feat(metrics): add the DreamSim adapter

DreamSimMetric implements the Metric protocol directly rather than going
through pyiqa: DreamSim owns its loader, weights and device placement, so
a pyiqa arch registration would wrap a wrapper. The model is built on the
first call, both tensors are squashed to the 224x224 its ViTs require, and
a scalar return falls back to per-pair scoring because upstream does not
commit to batch behaviour.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: The `dreamsim_spec()` factory

**Files:**
- Modify: `src/dreamsim_metric.py` (append)
- Test: `tests/test_dreamsim_metric.py` (append)

**Interfaces:**
- Consumes: `DreamSimMetric`, `VALID_TYPES`, `PATCH_CAPABLE` (Task 2); `metrics.MetricSpec`, `metrics.ModeSupport`, `metrics.ModeUnsupported`, `metrics.REASON_DEEP_2D`.
- Produces:
  - `dreamsim_spec(*, dreamsim_type="ensemble", pretrained=True, device=None, cache_dir=DREAMSIM_CACHE, normalize_embeds=True, use_patch_model=False, name=None) -> MetricSpec`
  - `DREAMSIM: MetricSpec` — the default-ensemble spec, `name == "dreamsim"`
  - `_derive_name(dreamsim_type, pretrained, normalize_embeds, use_patch_model) -> str`
  - Task 4 imports `DREAMSIM` and `dreamsim_spec` into `metrics.py`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_dreamsim_metric.py`:

```python
from dreamsim_metric import DREAMSIM, dreamsim_spec
from metrics import (
    REASON_DEEP_2D,
    MetricRegistry,
    ModeSupport,
    ModeUnsupported,
)


class TestSpecNaming:
    CASES = [
        ({},                                                    "dreamsim"),
        ({"dreamsim_type": "dino_vitb16"},                       "dreamsim_dino_vitb16"),
        ({"dreamsim_type": "clip_vitb32"},                       "dreamsim_clip_vitb32"),
        ({"dreamsim_type": "dino_vitb16", "use_patch_model": True},
                                                                 "dreamsim_dino_vitb16_patch"),
        ({"pretrained": False},                                  "dreamsim_scratch"),
        ({"normalize_embeds": False},                            "dreamsim_rawembeds"),
        ({"dreamsim_type": "dinov2_vitb14", "pretrained": False},
                                                                 "dreamsim_dinov2_vitb14_scratch"),
    ]

    @pytest.mark.parametrize("kwargs,expected", CASES)
    def test_name_is_derived(self, kwargs, expected):
        assert dreamsim_spec(**kwargs).name == expected

    def test_device_does_not_change_the_name(self):
        assert dreamsim_spec(device="cpu").name == "dreamsim"

    def test_cache_dir_does_not_change_the_name(self, tmp_path):
        assert dreamsim_spec(cache_dir=tmp_path).name == "dreamsim"

    def test_explicit_name_wins(self):
        assert dreamsim_spec(name="dreamsim_mps").name == "dreamsim_mps"

    def test_distinct_configurations_get_distinct_names(self):
        names = {dreamsim_spec(**kwargs).name for kwargs, _ in self.CASES}
        assert len(names) == len(self.CASES)


class TestSpecAttributes:
    def test_default_spec_attributes(self):
        spec = dreamsim_spec()
        assert spec.direction == "lower_is_better"
        assert spec.reference is True
        assert spec.channels  == "rgb"
        assert spec.builtin   is True
        assert spec.domain    == ""

    def test_variants_are_not_builtin(self):
        assert dreamsim_spec(dreamsim_type="dino_vitb16").builtin is False

    def test_slice_mode_is_supported(self):
        assert isinstance(dreamsim_spec().slice_mode, ModeSupport)

    def test_volume_mode_is_unsupported_with_the_shared_reason(self):
        volume_mode = dreamsim_spec().volume_mode
        assert isinstance(volume_mode, ModeUnsupported)
        assert volume_mode.reason == REASON_DEEP_2D

    def test_module_constant_is_the_default_spec(self):
        assert DREAMSIM.name == "dreamsim"
        assert DREAMSIM.builtin is True


class TestSpecValidation:
    def test_unknown_type_lists_the_valid_ones(self):
        with pytest.raises(ValueError, match="ensemble"):
            dreamsim_spec(dreamsim_type="vitb99")

    def test_every_valid_type_is_accepted(self):
        for name in VALID_TYPES:
            assert dreamsim_spec(dreamsim_type=name).name.startswith("dreamsim")

    def test_patch_model_rejected_for_incapable_type(self):
        with pytest.raises(ValueError, match="use_patch_model"):
            dreamsim_spec(dreamsim_type="clip_vitb32", use_patch_model=True)

    def test_patch_model_accepted_for_capable_types(self):
        for name in PATCH_CAPABLE:
            assert dreamsim_spec(dreamsim_type=name, use_patch_model=True).builtin is False


class TestSpecFactoryBuildsTheMetric:
    def test_factory_returns_a_dreamsim_metric(self):
        metric = dreamsim_spec(device="cpu").slice_mode.factory()
        assert isinstance(metric, DreamSimMetric)
        assert metric.device == torch.device("cpu")

    def test_configuration_reaches_the_metric(self):
        metric = dreamsim_spec(dreamsim_type="dino_vitb16",
                               use_patch_model=True).slice_mode.factory()
        assert metric._kwargs["dreamsim_type"]   == "dino_vitb16"
        assert metric._kwargs["use_patch_model"] is True

    def test_registry_can_hold_two_variants_at_once(self):
        registry = MetricRegistry(DREAMSIM,
                                  dreamsim_spec(dreamsim_type="dino_vitb16"))
        assert sorted(s.name for s in registry.specs) == [
            "dreamsim", "dreamsim_dino_vitb16",
        ]

    def test_registry_skips_it_in_volume_mode(self):
        applicable, skipped = MetricRegistry(DREAMSIM).select("volume")
        assert applicable == []
        assert [(s.name, s.reason) for s in skipped] == [("dreamsim", REASON_DEEP_2D)]

    def test_registry_keeps_it_in_slice_mode(self):
        applicable, skipped = MetricRegistry(DREAMSIM).select("slice")
        assert [s.name for s in applicable] == ["dreamsim"]
        assert skipped == []

    def test_registry_caches_the_instance(self):
        registry = MetricRegistry(dreamsim_spec(device="cpu"))
        assert registry.get_metric("dreamsim") is registry.get_metric("dreamsim")
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
.venv/bin/pytest tests/test_dreamsim_metric.py -q
```

Expected: `ImportError: cannot import name 'DREAMSIM' from 'dreamsim_metric'`.

- [ ] **Step 3: Append the factory to src/dreamsim_metric.py**

Add to the imports at the top of the file:

```python
from metrics import DEVICE, MetricSpec, ModeSupport, ModeUnsupported, REASON_DEEP_2D
```

(replacing the existing `from metrics import DEVICE`), then append at the end of the file:

```python
# ---------------------------------------------------------------------------
# Spec factory
# ---------------------------------------------------------------------------

def _derive_name(
    dreamsim_type:    str,
    pretrained:       bool,
    normalize_embeds: bool,
    use_patch_model:  bool,
) -> str:
    """Build a metric name that is unique per scoring-relevant configuration.

    Only what changes the score enters the name; `device` and `cache_dir` do
    not. Suffix order is fixed — type, patch, scratch, rawembeds — so the same
    configuration always yields the same column in the report.
    """
    parts = ["dreamsim"]
    if dreamsim_type != "ensemble":
        parts.append(dreamsim_type)
    if use_patch_model:
        parts.append("patch")
    if not pretrained:
        parts.append("scratch")
    if not normalize_embeds:
        parts.append("rawembeds")
    return "_".join(parts)


def dreamsim_spec(
    *,
    dreamsim_type:    str = "ensemble",
    pretrained:       bool = True,
    device:           Optional[str | torch.device] = None,
    cache_dir:        Path = DREAMSIM_CACHE,
    normalize_embeds: bool = True,
    use_patch_model:  bool = False,
    name:             Optional[str] = None,
) -> MetricSpec:
    """Build a configured DreamSim `MetricSpec`.

    `MetricSpec` is a frozen, static description; configuration lives in the
    factory closure, the same seam `_pyiqa_factory("radimagenet_lpips",
    backbone_path=...)` uses in metrics.py.

        registry = MetricRegistry(PSNR, SSIM, DREAMSIM)
        registry = MetricRegistry(dreamsim_spec(dreamsim_type="dino_vitb16",
                                               device="mps"))

    The scores are a distance: lower is more similar, roughly within [0, 1]
    when `normalize_embeds` is True.

    Args:
        dreamsim_type: which backbones produce the embedding. "ensemble"
            (default) concatenates DINO ViT-B/16 + CLIP ViT-B/32 + OpenCLIP
            ViT-B/32, each LoRA-finetuned; it agrees best with human judgement
            and costs three forward passes per image — six per slice pair —
            plus roughly three times the weights and memory. A single backbone
            ("dino_vitb16", "clip_vitb32", "open_clip_vitb32", "dinov2_vitb14",
            "synclr_vitb16") is about a third of the cost and slightly weaker.
            They differ in kind, not just in size: DINO/DINOv2 are
            self-supervised and carry more structural and textural
            information, CLIP/OpenCLIP are language-supervised and carry more
            semantics. For MRI, structure is usually the relevant axis.
            Worth changing: iterate on a single backbone, produce final numbers
            with "ensemble", and never mix the two within one comparison —
            they are different scales.
        pretrained: True loads DreamSim's LoRA weights, trained on human
            similarity judgements over natural images (NIGHTS). False yields
            the raw, unfinetuned backbones. Both the finetuning and the
            backbones' own pretraining are generic-domain, which is why a
            dreamsim score ranks MRI reconstructions but does not measure
            diagnostic quality. Worth changing: running the same evaluation
            with False is the cheapest available ablation for how much the
            natural-image prior contributes here — if the ranking barely
            moves, the finetuning is not transferring. For a medically
            pretrained perceptual metric, see `radimagenet_lpips`.
        device: None resolves to the framework's global `DEVICE` (cuda if
            available, else cpu). Worth changing: "mps" on Apple Silicon,
            "cpu" when a backend lacks an op, "cuda:N" to pick a GPU, or when
            memory is tight — the ensemble holds three ViTs at once.
        cache_dir: where the weights are downloaded on first use. Defaults to
            `constants.DREAMSIM_CACHE`. Worth changing to share one cache
            across projects.
        normalize_embeds: L2-normalizes embeddings before the distance is
            taken. True is the reference configuration and keeps scores in
            roughly [0, 1]. False lets embedding magnitude enter the distance,
            so values stop being comparable across images — experiments only.
        use_patch_model: use dense patch features alongside the CLS token,
            making the metric more sensitive to local fine-structure
            differences (plausibly better for MRI artefacts) at a higher
            compute cost. Only available for the types in `PATCH_CAPABLE`.
        name: overrides the derived metric name. Use it when two specs differ
            only in a parameter the name ignores (`device`, `cache_dir`) and
            both should appear in one report.

    Returns:
        A `MetricSpec` whose `volume_mode` is unsupported: DreamSim is a 2D
        ViT and has no way to see a slice stack as one body.

    Raises:
        ValueError: on an unknown `dreamsim_type`, or `use_patch_model=True`
            for a type without a patch variant.
    """
    if dreamsim_type not in VALID_TYPES:
        raise ValueError(
            f"unknown dreamsim_type '{dreamsim_type}' — valid types are: "
            + ", ".join(VALID_TYPES)
        )
    if use_patch_model and dreamsim_type not in PATCH_CAPABLE:
        raise ValueError(
            f"use_patch_model is only available for {' and '.join(PATCH_CAPABLE)}, "
            f"not for '{dreamsim_type}'"
        )

    metric_name = name or _derive_name(
        dreamsim_type, pretrained, normalize_embeds, use_patch_model
    )

    def build() -> DreamSimMetric:
        return DreamSimMetric(
            dreamsim_type=dreamsim_type,
            pretrained=pretrained,
            device=device,
            cache_dir=cache_dir,
            normalize_embeds=normalize_embeds,
            use_patch_model=use_patch_model,
        )

    return MetricSpec(
        metric_name,
        "lower_is_better",
        True,
        "rgb",
        ModeSupport(build),
        ModeUnsupported(REASON_DEEP_2D),
        # Only the default configuration owns a dedicated
        # ImageEvaluatorRecord field; variants land in record.extra, which the
        # report flattens automatically.
        builtin=(metric_name == "dreamsim"),
    )


# The default ensemble configuration. Deliberately NOT part of
# BUILTIN_METRICS: registering it downloads about a gigabyte of weights and
# spends six ViT forward passes per slice pair, which no default run should do
# behind the user's back. Opt in explicitly:
#
#     registry = MetricRegistry(*BUILTIN_METRICS, DREAMSIM)
DREAMSIM = dreamsim_spec()
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
.venv/bin/pytest tests/test_dreamsim_metric.py -q
```

Expected: all pass, including the two `TestRegistry*` cases that build a real `MetricRegistry`.

- [ ] **Step 5: Commit**

```bash
git add src/dreamsim_metric.py tests/test_dreamsim_metric.py
git commit -m "$(cat <<'EOF'
feat(metrics): add the dreamsim_spec configuration factory

All six upstream parameters are exposed with defaults and documented in
place: what each one does and when changing it is worth the cost. Names
are derived from the scoring-relevant configuration, so two variants can
sit in one registry and one report; device and cache_dir stay out of the
name because they do not change what is measured. Invalid types and
patch-model combinations fail at registration rather than mid-batch.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Wire it into the framework

**Files:**
- Modify: `src/metrics.py:265-276` (module docstring line and the bottom import block)
- Modify: `src/records.py:31` (after the `vsi` field)
- Modify: `src/main.py:10-17` (the re-export block)
- Test: `tests/test_records.py`, `tests/test_metrics.py`

**Interfaces:**
- Consumes: `DREAMSIM`, `dreamsim_spec` (Task 3).
- Produces: `metrics.DREAMSIM`, `metrics.dreamsim_spec`, `main.DREAMSIM`, `main.dreamsim_spec`, and the `ImageEvaluatorRecord.dreamsim` field. Nothing depends on these afterwards.

- [ ] **Step 1: Write the failing record-field tests**

Append to `tests/test_records.py`:

```python
class TestDreamSimField:
    """dreamsim's default spec is builtin=True, so it needs a dedicated field.

    Builtin scores are written with setattr(record, spec.name, value); on a
    dataclass without the field, that write is silently dropped by asdict().
    """

    def test_field_defaults_to_none(self):
        assert ImageEvaluatorRecord(image_id="img").dreamsim is None

    def test_field_is_declared_not_ad_hoc(self):
        assert "dreamsim" in ImageEvaluatorRecord.__annotations__

    def test_value_survives_to_dict(self):
        record = ImageEvaluatorRecord(image_id="img", dreamsim=0.12)
        assert record.to_dict()["dreamsim"] == 0.12

    def test_not_stored_in_extra(self):
        record = ImageEvaluatorRecord(image_id="img", dreamsim=0.12)
        assert "dreamsim" not in record.extra
```

- [ ] **Step 2: Write the failing re-export and bundle tests**

Append to `tests/test_metrics.py`:

```python
class TestDreamSimIsReachableButOptIn:
    def test_metrics_module_reexports_the_spec(self):
        from dreamsim_metric import DREAMSIM as _DREAMSIM
        from metrics import DREAMSIM
        assert DREAMSIM is _DREAMSIM

    def test_metrics_module_reexports_the_factory(self):
        from metrics import dreamsim_spec
        assert dreamsim_spec().name == "dreamsim"

    def test_kept_out_of_builtin_metrics(self):
        assert "dreamsim" not in {s.name for s in BUILTIN_METRICS}

    def test_builtin_bundle_still_has_eighteen_entries(self):
        assert len(BUILTIN_METRICS) == 18

    def test_main_reexports_it_too(self):
        import main
        assert main.DREAMSIM.name == "dreamsim"
        assert main.dreamsim_spec is not None
```

- [ ] **Step 3: Run both test files to verify they fail**

```bash
.venv/bin/pytest tests/test_records.py::TestDreamSimField tests/test_metrics.py::TestDreamSimIsReachableButOptIn -q
```

Expected: `TypeError: ImageEvaluatorRecord.__init__() got an unexpected keyword argument 'dreamsim'` and `ImportError: cannot import name 'DREAMSIM' from 'metrics'`.

- [ ] **Step 4: Add the record field**

In `src/records.py`, in the full-reference block, immediately after the `vsi` line:

```python
    vsi:                 Optional[float] = None
    dreamsim:            Optional[float] = None
```

- [ ] **Step 5: Add the late import to metrics.py**

In `src/metrics.py`, extend the existing bottom import block (currently lines 267-276). Add after the `segmentation_metrics.volume_metrics` import:

```python
# Same cycle, same reason: dreamsim_metric.py does `from metrics import
# MetricSpec`, so this import has to come after MetricSpec is defined.
from dreamsim_metric import DREAMSIM, dreamsim_spec
```

Then add one line to the module docstring, after the paragraph ending
`e.g. \`MetricRegistry(PSNR, SSIM)\`.`:

```
DreamSim is the one built-in that is not pyiqa-backed and not part of
`BUILTIN_METRICS`: build its spec with `dreamsim_spec(...)` (or use the
`DREAMSIM` default) and register it explicitly.
```

- [ ] **Step 6: Add the re-exports to main.py**

In `src/metrics`'s import block in `src/main.py`, extend the existing list. After the line holding `MUSIQ, MANIQA, PAQ2PIQ, PIQE, ILNIQE, BUILTIN_METRICS,` add:

```python
    DREAMSIM, dreamsim_spec,
```

- [ ] **Step 7: Run the tests to verify they pass**

```bash
.venv/bin/pytest tests/test_records.py tests/test_metrics.py -q
```

Expected: all pass. `TestBuiltinCapabilities::test_sixteen_builtins_cannot_do_volume` must still pass untouched — if it now lists `dreamsim`, `DREAMSIM` was wrongly added to `BUILTIN_METRICS`.

- [ ] **Step 8: Verify a registry round-trip end to end**

```bash
.venv/bin/python -c "
import sys; sys.path.insert(0, 'src')
import main
from metrics import MetricRegistry, BUILTIN_METRICS
reg = MetricRegistry(*BUILTIN_METRICS, main.DREAMSIM)
print('specs:', len(reg.specs))
applicable, skipped = reg.select('volume')
print('skipped includes dreamsim:', 'dreamsim' in {s.name for s in skipped})
print('records field:', 'dreamsim' in __import__('records').ImageEvaluatorRecord.__annotations__)
"
```

Expected: `specs: 19`, `skipped includes dreamsim: True`, `records field: True`.

- [ ] **Step 9: Run the whole suite**

```bash
.venv/bin/pytest -q 2>&1 | tail -5
```

Expected: zero failures.

- [ ] **Step 10: Commit**

```bash
git add src/metrics.py src/records.py src/main.py tests/test_records.py tests/test_metrics.py
git commit -m "$(cat <<'EOF'
feat(metrics): wire dreamsim into the registry and the report

metrics.py reaches dreamsim_metric through the same late import the
segmentation specs use, ImageEvaluatorRecord gains the dedicated column
the builtin flag promises, and main.py re-exports both names. DREAMSIM
stays out of BUILTIN_METRICS: a default run should not download a
gigabyte of weights unasked. No evaluator changed — the Metric protocol
carried a second backend with no edits at all.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Verify against the real model

Everything so far ran against a fake. This task finds out what DreamSim actually does — including the batching question the offline tests can only guess at — and records the answer.

**Files:**
- Create: `tests/test_dreamsim_integration.py`
- Modify: `src/dreamsim_metric.py` (one measured-cost note in the class docstring)

**Interfaces:**
- Consumes: `DreamSimMetric`, `dreamsim_spec` (Tasks 2-3), `constants.DREAMSIM_CACHE` (Task 1).
- Produces: nothing further tasks depend on.

- [ ] **Step 1: Write the gated integration tests**

Create `tests/test_dreamsim_integration.py`:

```python
"""Integration tests against the real DreamSim model.

Skipped unless the package is installed AND weights are already cached, so a
normal test run never triggers a ~1 GB download. Populate the cache with:

    .venv/bin/python -c "import sys; sys.path.insert(0,'src'); \
from dreamsim_metric import DREAMSIM; import torch; \
m = DREAMSIM.slice_mode.factory(); print(m(torch.rand(1,3,96,96), torch.rand(1,3,96,96)))"
"""
import pytest
import torch

pytest.importorskip("dreamsim")

from constants import DREAMSIM_CACHE
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
```

- [ ] **Step 2: Confirm the tests skip cleanly on an empty cache**

```bash
.venv/bin/pytest tests/test_dreamsim_integration.py -q -rs
```

Expected: `4 skipped` (or 5, counting the module-level skip line) with the reason naming `models/dreamsim`. A **failure** here means the guard is wrong.

- [ ] **Step 3: Populate the cache and run the smoke check**

This downloads about a gigabyte. Write the script to the scratchpad, not the repo:

```bash
cat > /tmp/dreamsim_smoke.py <<'EOF'
import sys, time
sys.path.insert(0, "src")
import torch
from dreamsim_metric import dreamsim_spec

pair = (torch.rand(8, 3, 96, 96), torch.rand(8, 3, 96, 96))
for device in ("cpu", "mps"):
    for kind in ("ensemble", "dino_vitb16"):
        try:
            metric = dreamsim_spec(dreamsim_type=kind, device=device).slice_mode.factory()
            t0 = time.perf_counter()
            scores = metric(*pair)
            dt = time.perf_counter() - t0
            print(f"{kind:12s} {device:4s} ok  n={len(scores)}  "
                  f"{dt / len(scores) * 1000:7.1f} ms/slice-pair  first={scores[0]:.4f}")
        except Exception as exc:
            print(f"{kind:12s} {device:4s} FAILED  {type(exc).__name__}: {exc}")
EOF
.venv/bin/python /tmp/dreamsim_smoke.py
```

Expected: four lines. Record the ms/slice-pair numbers — Step 5 puts them in the docstring. If an `mps` line fails, that is a finding, not a bug to fix: note the exception type and move on (the documented fallback is `device="cpu"`).

- [ ] **Step 4: Run the integration tests for real**

```bash
.venv/bin/pytest tests/test_dreamsim_integration.py -q
```

Expected: all pass. If `test_batch_returns_one_score_per_pair` fails with three identical scores, the model collapsed the batch and the fallback did not trigger — check `_as_scores` against the actual return shape and report before changing anything.

- [ ] **Step 5: Record the measured cost in the docstring**

In `src/dreamsim_metric.py`, append to the `DreamSimMetric` class docstring, right before the `Args:` block, using the numbers from Step 3 (example shape shown — substitute the real values):

```
    Measured cost (2026-09-10, Apple M-series, 8 slice pairs of 96x96):
    ensemble ~XXX ms per slice pair on cpu, ~YYY ms on mps; dino_vitb16
    ~ZZZ ms on cpu. The ensemble is the most expensive metric in the
    framework — a 100-slice volume is minutes, not seconds.
```

- [ ] **Step 6: Run the whole suite once more**

```bash
.venv/bin/pytest -q 2>&1 | tail -5
```

Expected: zero failures; the integration tests now run instead of skipping.

- [ ] **Step 7: Commit**

```bash
git add tests/test_dreamsim_integration.py src/dreamsim_metric.py
git commit -m "$(cat <<'EOF'
test(metrics): pin down real DreamSim behaviour

Integration tests run only when the package is installed and weights are
already cached, so a normal run never pulls a gigabyte. They settle the
question the offline tests could not: the model does return one score per
pair for a batched call. Measured per-slice costs go into the adapter
docstring, where the choice between ensemble and a single backbone is
actually made.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Documentation

**Files:**
- Modify: `CLAUDE.md`
- Modify: `README.md`
- Modify: `src/evaluation.ipynb`

**Interfaces:**
- Consumes: everything above.
- Produces: nothing.

- [ ] **Step 1: Update CLAUDE.md — architecture section**

Add a new component paragraph after the `radimagenet_lpips.py` entry:

```markdown
**`dreamsim_metric.py`** — `DreamSimMetric` (the framework's only non-pyiqa `Metric` implementation) plus `dreamsim_spec()`, which returns a configured `MetricSpec`. Six documented parameters: `dreamsim_type` (which of the LoRA-finetuned ViT backbones produce the embedding — `ensemble` is best and three times the cost), `pretrained`, `device`, `cache_dir`, `normalize_embeds`, `use_patch_model`. Names are derived from the configuration (`dreamsim`, `dreamsim_dino_vitb16`, `dreamsim_scratch`, …) so several variants can be registered at once. Opt-in: `DREAMSIM` is **not** in `BUILTIN_METRICS` — registering it downloads ~1 GB into `models/dreamsim`.
```

- [ ] **Step 2: Update CLAUDE.md — constants, layout, metric table**

Three edits:

In the `constants.py` paragraph, extend the path list to name the new constant — replace `` `RESNET50` (model weights) `` with `` `RESNET50`, `DREAMSIM_CACHE` (model weights) ``.

In the data layout block, under `models/`:

```
models/
  RadImageNet_pytorch/
    ResNet50.pt
  dreamsim/            # downloaded on first use, git-ignored
```

In the metric summary table, after the `fsim, vsi` row:

```markdown
| dreamsim (opt-in) | FR | lower |
```

And in the FR/NR legend line below the table, append: `dreamsim is registered explicitly, not via BUILTIN_METRICS.`

- [ ] **Step 3: Update README.md**

```bash
grep -n "ilniqe\|fsim\|BUILTIN_METRICS\|models/" README.md
```

Apply the same three changes as in CLAUDE.md at the matching places: the metric table row, the `models/dreamsim/` line in the data layout, and one sentence naming `dreamsim_spec()` where the other metrics are introduced. Match the file's existing wording and heading style; do not restructure it.

- [ ] **Step 4: Update the notebook intro count**

In `src/evaluation.ipynb`, replace this exact line:

```
    "This notebook evaluates reconstruction quality of MRI super-resolution outputs on the **IXI dataset** using 18 IQA metrics implemented in `main.py`.\n",
```

with:

```
    "This notebook evaluates reconstruction quality of MRI super-resolution outputs on the **IXI dataset** using the 18 built-in IQA metrics implemented in `main.py`, plus DreamSim as an opt-in nineteenth.\n",
```

- [ ] **Step 5: Add the notebook reference-table row**

After the `VSI` row of the Metric Reference Thresholds table:

```
    "| DreamSim | ↓ lower | 0 | — | — | — | — |\n",
```

Do **not** add a `THRESHOLDS` dict entry — the plotting code already handles a missing key.

- [ ] **Step 6: Mention the opt-in in the notebook registry cell**

Replace this exact source string:

```
"# Pick which metrics this run computes — swap BUILTIN_METRICS for\n# SEGMENTATION_METRICS, or list individual constants\n# (e.g. MetricRegistry(PSNR, SSIM)), to run a different set.\nregistry = MetricRegistry(*BUILTIN_METRICS)"
```

with:

```
"# Pick which metrics this run computes — swap BUILTIN_METRICS for\n# SEGMENTATION_METRICS, or list individual constants\n# (e.g. MetricRegistry(PSNR, SSIM)), to run a different set.\n#\n# DreamSim is opt-in: add DREAMSIM (or dreamsim_spec(dreamsim_type=...))\n# to the call below. First use downloads ~1 GB into models/dreamsim and it\n# is by far the slowest metric here.\nregistry = MetricRegistry(*BUILTIN_METRICS)"
```

Also extend the Experiment Configuration markdown cell's FR list: replace `FSIM, GMSD, VSI)` with `FSIM, GMSD, VSI, DreamSim)`.

- [ ] **Step 7: Verify the notebook is still valid JSON**

```bash
.venv/bin/python -c "import json; nb = json.load(open('src/evaluation.ipynb')); print('cells:', len(nb['cells']))"
```

Expected: a cell count, no exception.

- [ ] **Step 8: Final full-suite run**

```bash
.venv/bin/pytest -q 2>&1 | tail -5
```

Expected: zero failures, and the count is the Task 1 baseline plus the new tests.

- [ ] **Step 9: Commit**

```bash
git add CLAUDE.md README.md src/evaluation.ipynb
git commit -m "$(cat <<'EOF'
docs: document dreamsim and its configuration surface

CLAUDE.md gains the module, the cache path and the metric-table row;
the notebook says DreamSim is the opt-in nineteenth metric and where to
switch it on. No THRESHOLDS entry: there are no MRI literature values for
a DreamSim distance, and the plot code already degrades to a plain
histogram.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

## Self-Review

**Spec coverage** — every section of `2026-09-09-dreamsim-metric-design.md` maps to a task:

| Spec section | Task |
|---|---|
| Approach: standalone adapter, lazy import | 2 |
| Configuration surface, all six parameters documented | 3 |
| Validation (`VALID_TYPES`, `PATCH_CAPABLE`) | 3 |
| Naming and record placement | 3 (naming), 4 (record field) |
| Adapter behaviour: lazy, seam, resize, batching, cost | 2 (behaviour), 5 (measured cost) |
| Errors and skips table | 2 (all four adapter cases), 3 (validation), 4 (volume skip round-trip) |
| Opt-in rather than default | 4 |
| Module layout and integration table | 1 (constants, gitignore, requirements), 4 (metrics/records/main) |
| Testing: offline unit + gated integration | 2, 3 (offline), 5 (gated) |
| Implementation order (install first, baseline suite) | 1 |

**Placeholder scan** — no TBD/TODO, no "add error handling", no "similar to Task N". The one deliberate blank is the measured timing in Task 5 Step 5, which cannot exist before Step 3 runs; the step states exactly where the numbers come from and what shape the sentence takes.

**Type consistency** — `DreamSimMetric.__init__` is keyword-only with the same six names in Tasks 2, 3, and the tests; `device` is stored as the public attribute `metric.device` (asserted in both task's tests) while the loader kwargs live in `metric._kwargs` (asserted in Task 3); `dreamsim_spec` returns `MetricSpec` and is called as `spec.slice_mode.factory()` in Tasks 3 and 5, matching `ModeSupport.factory`'s zero-argument slice-mode contract in `metrics.py`; `_derive_name`'s parameter order matches its single call site; `INPUT_SIZE`, `VALID_TYPES`, `PATCH_CAPABLE`, `_import_dreamsim` are defined in Task 2 and imported by name in the Task 2 and Task 3 tests.
