# DreamSim as a Configurable Full-Reference Metric — Design

Date: 2026-09-09
Branch: `volumetric_similarity`

## Purpose

The framework's perceptual full-reference metrics are all calibrated on natural
images: `lpips` and `dists` on BAPPS/KADID, `fsim`/`gmsd`/`vsi` on hand-designed
low-level features. `radimagenet_lpips` is the one medically pretrained
alternative, and it is uncalibrated — it averages feature differences without
learned linear layers.

DreamSim (Fu et al., NeurIPS 2023) sits between those two poles. It is a
LoRA-finetuned ViT feature extractor whose distance is the cosine distance
between embeddings, trained on the NIGHTS dataset of human two-alternative
similarity judgements. It agrees with human judgement better than LPIPS/DISTS on
natural images because it weighs mid-level structure — layout, pose, shape —
rather than pixel-local texture. That is the property this framework wants when
ranking reconstructions of the same anatomy.

It is also the metric where the domain-gap caveat is sharpest, and the design
takes that seriously rather than hiding it: both the backbones' pretraining and
the human-judgement finetuning happened on natural photographs. MRI contrast,
noise characteristics, and the semantics of a grey value are all different. So
DreamSim is a *relative ranking* instrument within one run, never an absolute
quality figure — and the configuration surface exists precisely so a user can
measure how much of the score comes from the natural-image prior.

A second motivation is architectural. Every metric shipped so far is
pyiqa-backed, so `metrics.Metric` — the adapter protocol the framework is built
around — has never had a second implementation. DreamSim brings its own loader,
its own weight management, and its own device handling; wrapping it in pyiqa
would add indirection and buy nothing. It is the first real test of whether the
protocol holds. If the design is right, DreamSim costs zero lines in
`IQAEvaluator`, `VolumeEvaluator`, `EvaluationResult`, and `MaskWriter`.

## Scope

**In scope**

- A new adapter module `src/dreamsim_metric.py`: `DreamSimMetric` (implements
  `metrics.Metric`) and the spec factory `dreamsim_spec()`.
- All six upstream parameters exposed and documented in code:
  `dreamsim_type`, `pretrained`, `device`, `cache_dir`, `normalize_embeds`,
  `use_patch_model`.
- Deterministic, collision-free metric naming across configurations.
- A dedicated `dreamsim` field in `ImageEvaluatorRecord`.
- Weight cache location as a `constants.py` path, excluded from git.
- Offline unit tests against an injected fake loader, plus gated integration
  tests against the real model.
- Documentation: `CLAUDE.md`, `README.md`, `evaluation.ipynb`.

**Out of scope**

- Volume mode. DreamSim is a 2D ViT; `volume_mode` is
  `ModeUnsupported(REASON_DEEP_2D)`, the same reason string the other deep 2D
  metrics use.
- Finetuning DreamSim on medical data. The design makes the domain gap
  measurable (`pretrained=False` as ablation) but does not close it.
- Adding `DREAMSIM` to `BUILTIN_METRICS`. It stays opt-in — see
  "Opt-in rather than default".
- Repairing the stale `requirements.txt` (it pins `torch==2.8.0` while the venv
  runs 2.12.0). Pre-existing, unrelated.
- Adding `mps` to the global `DEVICE`. A framework-wide change affecting all 18
  pyiqa metrics, several of which use ops with incomplete MPS coverage. DreamSim
  reaches MPS through its own `device` parameter instead.

## Approach

Three routes were considered.

**A standalone adapter module (chosen).** `DreamSimMetric` implements the
`Metric` protocol directly and calls `dreamsim.dreamsim()` itself. The `dreamsim`
import happens inside the method, so `metrics.py` stays importable when the
package is absent and the existing 462 tests are untouched.

**A pyiqa arch registration**, following `radimagenet_lpips.py`
(`ARCH_REGISTRY` + `DEFAULT_CONFIGS`), reusing `PyIQAMetric`. Rejected:
DreamSim's own loader already owns weight download, LoRA assembly, and device
placement, so pyiqa's `create_metric` would wrap a wrapper.
`DEFAULT_CONFIGS` is keyed one entry per metric name, which fights the
`dreamsim_<type>` naming scheme, and pyiqa's own preprocessing and `as_loss`
paths add failure surface for no gain.

**Inline in `metrics.py`.** Rejected: that file already carries the registry
plus 29 specs, and the import would become eager.

## Configuration surface

`MetricSpec` is a frozen, static description; per-metric configuration already
lives in the factory closure (`_pyiqa_factory("radimagenet_lpips",
backbone_path=...)`). DreamSim follows that established seam rather than growing
the shared dataclass:

```python
def dreamsim_spec(
    *,
    dreamsim_type:    str = "ensemble",
    pretrained:       bool = True,
    device:           Optional[str | torch.device] = None,
    cache_dir:        Path = DREAMSIM_CACHE,
    normalize_embeds: bool = True,
    use_patch_model:  bool = False,
    name:             Optional[str] = None,
) -> MetricSpec
```

`DREAMSIM = dreamsim_spec()` is the default-ensemble constant. Registration:

```python
registry = MetricRegistry(PSNR, SSIM, DREAMSIM)
registry = MetricRegistry(dreamsim_spec(dreamsim_type="dino_vitb16", device="mps"))
```

Rejected alternatives: a generic `config: Mapping[str, Any]` field on
`MetricSpec` (all 29 specs inherit a field only one uses, untyped, redundant
with the closure) and a `DreamSimSpec` subclass (introduces spec inheritance as
a new pattern the registry would have to tolerate).

### What each parameter does, and when to change it

This is the substance of the feature; the text below is what the factory
docstring must convey.

**`dreamsim_type`** — DreamSim is not one network. The distance is a cosine
distance over concatenated ViT embeddings, and this selects which backbones
contribute. `ensemble` (default) concatenates **DINO ViT-B/16 + CLIP ViT-B/32 +
OpenCLIP ViT-B/32**, each LoRA-finetuned; it agrees best with human judgement
and costs **three forward passes per image** — six per slice pair — plus roughly
three times the weights and memory. A single backbone (`dino_vitb16`,
`clip_vitb32`, `open_clip_vitb32`, `dinov2_vitb14`, `synclr_vitb16`) is about a
third of the cost and slightly weaker. The backbones also differ in kind:
DINO/DINOv2 are self-supervised and carry more structural and textural
information; CLIP/OpenCLIP are language-supervised and carry more semantics.
For MRI, structure is usually the relevant axis. Practical guidance: sweep
parameters and iterate on a single backbone, produce final numbers with
`ensemble`, and never mix the two in one comparison — they are different scales.

**`pretrained`** — `True` loads DreamSim's LoRA weights, trained on human
similarity judgements over **natural images** (NIGHTS). `False` yields the raw,
unfinetuned backbones. Both the finetuning and the backbones' own pretraining
are generic-domain, which is why a DreamSim score on MRI ranks reconstructions
but does not measure diagnostic quality. Running the same evaluation with
`pretrained=False` is the cheapest available ablation for how much the
natural-image prior contributes in this domain; if the ranking barely changes,
the finetuning is not transferring. For a medically pretrained perceptual
alternative, `radimagenet_lpips` is already in the framework.

**`device`** — `None` resolves to the framework's global `DEVICE`
(`cuda` if available, else `cpu`). Set it explicitly for `"mps"` on Apple
Silicon, for `"cpu"` when a backend lacks an op, or for a specific
`"cuda:N"`. The adapter moves its own tensors, because `IQAEvaluator` has
already moved them to the global `DEVICE` and an override would otherwise
produce a device mismatch. Worth changing when GPU memory is tight — the
ensemble holds three ViTs at once — or when the ensemble is too slow on CPU.

**`cache_dir`** — where weights are downloaded on first use; defaults to
`constants.DREAMSIM_CACHE` (`models/dreamsim`). Change it to share one cache
across projects.

**`normalize_embeds`** — L2-normalizes embeddings before the distance is taken.
`True` is the reference configuration and keeps scores in roughly `[0, 1]`.
`False` lets embedding magnitude enter the distance, so values stop being
comparable across images; experimental only.

**`use_patch_model`** — a variant that uses dense patch features alongside the
CLS token, making the metric more sensitive to local fine-structure differences.
Plausibly better for MRI artefacts, at a higher compute cost. Available only for
`dino_vitb16` and `dinov2_vitb14`.

### Validation

`dreamsim_spec()` validates eagerly, so a typo fails at registration rather than
inside the first batch:

- `dreamsim_type` outside `VALID_TYPES` → `ValueError` listing the valid names.
- `use_patch_model=True` with a type outside `PATCH_CAPABLE`
  (`dino_vitb16`, `dinov2_vitb14`) → `ValueError` naming the two that support it.

## Naming and record placement

`MetricRegistry` is keyed by name, so two configurations must not share one.
The name is derived deterministically: base `dreamsim`, then a suffix for every
deviation from the default, in fixed order.

| Configuration | Name | Report column |
|---|---|---|
| all defaults | `dreamsim` | `ImageEvaluatorRecord.dreamsim` |
| `dreamsim_type="dino_vitb16"` | `dreamsim_dino_vitb16` | `extra` |
| `+ use_patch_model=True` | `dreamsim_dino_vitb16_patch` | `extra` |
| `pretrained=False` | `dreamsim_scratch` | `extra` |
| `normalize_embeds=False` | `dreamsim_rawembeds` | `extra` |

Suffixes combine in that fixed order — type, then `_patch`, then `_scratch`,
then `_rawembeds` — so `dreamsim_spec(dreamsim_type="dinov2_vitb14",
pretrained=False)` is `dreamsim_dinov2_vitb14_scratch`. Distinct configurations
therefore always produce distinct names.

`device` and `cache_dir` are excluded: they change where and how fast the
computation runs, not what it measures. Two configurations that differ only in
device would collide, which is what the explicit `name=` override is for.
As everywhere else in the registry, registering the same name twice means the
last one wins.

`builtin = (name == "dreamsim")`. The default spec therefore gets a typed,
fixed-position CSV column (`None` in runs where it was not scored); variants are
flattened out of `record.extra`. Note the two meanings of "builtin" this
introduces — `MetricSpec.builtin` means "has its own record field", while
`BUILTIN_METRICS` means "the bundle `main.py` opts into by default". One
sentence in the `MetricSpec` docstring records the distinction.

Remaining spec values: `direction="lower_is_better"` (it is a distance),
`reference=True`, `channels="rgb"`, `domain=""` (generic — deliberately not
"medical"), `slice_mode=ModeSupport(...)`,
`volume_mode=ModeUnsupported(REASON_DEEP_2D)`.

## Adapter behaviour

```python
class DreamSimMetric:
    def __init__(self, *, dreamsim_type, pretrained, device, cache_dir,
                 normalize_embeds, use_patch_model, loader=None): ...
    def __call__(self, input, target=None) -> list[float]: ...
```

**Lazy construction.** The model is built on the first `__call__`, mirroring
`PyIQAMetric._impl`. Registering a spec and asking the registry for the instance
cost nothing — no download, no import.

**Injection seam.** `loader=None` means "import `dreamsim.dreamsim` on first
use", with the import inside the method. Tests pass a fake, which is what makes
the adapter testable offline.

**Resizing.** DreamSim requires exactly `(N, 3, 224, 224)`; ViT positional
embeddings are fixed. Both tensors are resized with
`F.interpolate(..., size=(224, 224), mode="bilinear", antialias=True)` after the
device move — the same squashing resize upstream's `preprocess` performs, so
scores stay comparable to published DreamSim numbers and no anatomy is cropped
away. The cost is that high-frequency MRI detail (noise texture, thin edges)
does not survive the downsample; the docstring says so explicitly, because that
detail is often exactly what distinguishes two reconstructions.

**Batching.** Upstream documents `model(img1, img2)` without confirming batch
behaviour. The adapter calls it once per batch and inspects the result shape:
`N` values are passed through; a scalar for `N > 1` falls back to a per-pair
loop. `IQAEvaluator` uses `BATCH_SIZE = 32`, so both paths are exercised in
practice, and the integration test pins down which one the real model takes.

**Cost.** `ensemble` is six ViT-B forward passes per slice pair. On CPU with
~100 slices per volume this is the most expensive metric in the framework by a
wide margin. No automatic downgrade — the docstring points at single-backbone
types instead, and the smoke test supplies measured per-slice timings.

## Errors and skips

`IQAEvaluator._compute_batch` catches every exception, prints
`metric '<name>' batch failed: <exc>`, and fills the column with `None`. A
missing package therefore does not abort a run; it prints once per batch. That
makes the message text part of the design, not an afterthought.

| Condition | Behaviour |
|---|---|
| `dreamsim` not installed | `ImportError` from the lazy load is re-raised as `"dreamsim is not installed — run: pip install dreamsim"`; column stays empty, run continues |
| `target is None` | `ValueError` — DreamSim is full-reference only |
| invalid `dreamsim_type` / `use_patch_model` combination | `ValueError` from `dreamsim_spec()`, at registration time |
| download failure / no network | upstream error propagated, message annotated with the `cache_dir` in use |
| scalar return for `N > 1` | per-pair fallback loop |
| `mode="volume"` | skipped via `ModeUnsupported(REASON_DEEP_2D)`, grouped into the existing message by `report_skipped_metrics`; a volume run with DreamSim as the only metric hits the existing `ValueError` in `evaluate()` |

Empty slices never reach the metric — `IQAEvaluator` filters them before
batching.

Score semantics in the report: a distance `>= 0`, in practice within `[0, 1]`
with `normalize_embeds=True`. `lower_is_better` means
`best_slice_per_metric()` selects the minimum, and `MaskWriter` writes
`_slice/_mask/_overlay` PNGs for that slice. The intensity-normalization caveat
recorded as `TODO(norm-4/norm-5)` applies unchanged.

## Opt-in rather than default

`DREAMSIM` is *not* added to `BUILTIN_METRICS`. A default `main.py` run would
otherwise download roughly a gigabyte of ensemble weights and spend six ViT
forwards per slice pair. Users opt in explicitly:

```python
registry = MetricRegistry(*BUILTIN_METRICS, DREAMSIM)
```

`BUILTIN_METRICS` stays at 18 entries, so no existing test changes count.

## Module layout and integration

| File | Change |
|---|---|
| `src/dreamsim_metric.py` | new — `DreamSimMetric`, `dreamsim_spec()`, `DREAMSIM`, `VALID_TYPES`, `PATCH_CAPABLE` |
| `src/constants.py` | `DREAMSIM_CACHE = Path("models/dreamsim")` |
| `src/metrics.py` | late import of `DREAMSIM`/`dreamsim_spec` in the existing bottom import block |
| `src/records.py` | `dreamsim: Optional[float] = None` in the full-reference block |
| `src/main.py` | `DREAMSIM`, `dreamsim_spec` added to the re-export block |
| `.gitignore` | `models/dreamsim/` after the `!models/**` unignore |
| `requirements.txt` | `dreamsim`, `peft`, `open-clip-torch`, pinned from the post-install freeze |

`dreamsim_metric.py` imports `DEVICE`, `MetricSpec`, `ModeSupport`,
`ModeUnsupported`, and `REASON_DEEP_2D` from `metrics`, while `metrics` imports
`dreamsim_metric` at the bottom of the file. This is the same deliberate cycle
`segmentation_metrics.monai_metrics` already uses: at bottom-import time
`metrics` is partially initialized, but all five names are defined above. The
existing comment pattern is repeated so the arrangement reads as intentional.

Unchanged, and that is the point: `iqa_evaluator.py`, `volume_evaluator.py`,
`evaluator_factory.py`, `evaluation_result.py`, `mask_writer.py`, and
`records.best_slice_per_metric()`. They speak only `Metric`, `MetricSpec`, and
`direction`.

## Testing

**`tests/test_dreamsim_metric.py`** — offline, always runs, fake loader.

*Spec factory:* every row of the naming table; `builtin` true only for the
default name; `direction`/`reference`/`channels`; `volume_mode` is
`ModeUnsupported` with `REASON_DEEP_2D`; `slice_mode` is `ModeSupport`;
`ValueError` for an unknown `dreamsim_type` (message lists valid values);
`ValueError` for `use_patch_model=True` with `clip_vitb32`.

*Adapter:* the fake records its inputs, so both tensors arrive as
`(N, 3, 224, 224)`; an explicit `device` is honoured; the return is a
`list[float]` of length N; a fake returning a scalar for `N > 1` still yields N
values through the fallback; `target=None` raises `ValueError`; a loader raising
`ImportError` produces a message containing `pip install dreamsim`; building the
spec and calling `registry.get_metric()` never invoke the loader, while the
first `__call__` does.

*Registry integration:* `select("slice")` includes it, `select("volume")` skips
it with the shared reason, and `get_metric()` twice returns the same cached
instance.

**`tests/test_dreamsim_integration.py`** — `pytest.importorskip("dreamsim")`
plus a skip when `DREAMSIM_CACHE` is empty, so a test run never triggers a
gigabyte download. Against the real model: identical images score ≈ 0; heavy
noise scores above mild noise; a batch of three returns three distinct values
(this is what settles the batching question); `dreamsim_type="dino_vitb16"`
loads and scores.

**`tests/test_records.py`** gains the new field. Nothing else in the existing
suite changes.

## Implementation order

The dependency install is the risk, not the code.

1. Snapshot `pip freeze`, run `pip install dreamsim`, diff the freeze. PyPI
   metadata for `dreamsim 0.2.1` carries no version pins, so torch should not
   move, but `timm` and `transformers` are shared with pyiqa's `maniqa`,
   `musiq`, and `clipiqa`.
2. Run the full 462-test suite **before touching any source file**. If it
   breaks, stop and report rather than building on a broken base.
3. TDD: the unit tests above red first, then `dreamsim_metric.py`, then the
   integration hooks.
4. Smoke test against the real model on a synthetic phantom pair, on both `mps`
   and `cpu`: confirms batching, confirms peft/LoRA works on MPS, and yields the
   per-slice timings quoted in the docstring.
5. Documentation: `CLAUDE.md` (new module, `dreamsim_spec` API, metric table row,
   `DREAMSIM_CACHE`, `models/dreamsim/` in the data layout, the opt-in note),
   `README.md`, and the `evaluation.ipynb` threshold/documentation cell.

## Known compromises

- **224×224 squash.** Aspect ratio is not preserved and fine detail is lost.
  Accepted to stay faithful to upstream preprocessing; the alternatives crop
  anatomy or introduce out-of-distribution padding.
- **Natural-image prior.** Unfixable within this scope. Mitigated by
  documentation, by `pretrained=False` as an ablation, and by pointing at
  `radimagenet_lpips`.
- **Two meanings of "builtin".** `MetricSpec.builtin` (own record field) versus
  `BUILTIN_METRICS` (default bundle). Documented rather than renamed; renaming
  would touch all 29 specs.
- **Name collisions on device-only differences.** Deliberate — device does not
  change what is measured — with `name=` as the escape hatch.
- **MPS unverified until step 4.** If peft/LoRA fails there, the documented
  fallback is `device="cpu"`.

## Follow-ups

- Reconcile `requirements.txt` with the venv (`torch` 2.8.0 vs 2.12.0).
- Consider `mps` in the global `DEVICE`, gated per metric, once someone audits
  MPS op coverage for the pyiqa metrics.
- If DreamSim proves useful, a medical LoRA finetune on paired MRI judgements is
  the obvious next step — and the reason `pretrained` is exposed.
