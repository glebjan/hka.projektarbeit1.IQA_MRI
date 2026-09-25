# iqaevaluator

`iqaevaluator` is a Python framework for the quality assessment of synthetically generated or reconstructed images and volumes. It scores an input either against a reference image (the target) or on its own, and it compares predicted segmentation masks with reference masks. Medical volumes from magnetic resonance imaging (MRI) and computed tomography (CT) are the starting point of its development, while ordinary two-dimensional images take the same path through loading, evaluation and result. Instead of a collection of individual scripts, the framework provides one reusable tool that returns a table of scores per slice or per volume.

Every run uses a metric registry that starts empty and holds only the metrics registered for that run, so computational cost and weight downloads stay visible in the call. Every result row records the normalization strategy and the intensity range used for scaling, which keeps scores traceable. The evaluation core knows metric libraries only through a narrow protocol, and each backend is connected through an exchangeable adapter.

The [project wiki](https://github.com/glebjan/hka.projektarbeit1.IQA_MRI/wiki) documents installation, concepts, every metric and complete worked examples in detail. This README gives a brief overview.

## Metrics

The framework bundles 30 metrics in four families. Full-Reference (FR) metrics compare the input with a target, No-Reference (NR) metrics assess the input alone, and segmentation metrics compare a predicted mask with a reference mask. The 18 image quality metrics are collected in `BUILTIN_METRICS`, DreamSim is registered separately because it downloads about one gigabyte of weights, and the 11 segmentation metrics are collected in `SEGMENTATION_METRICS`. The column "Better" states whether higher or lower values indicate better quality. The [Metric Catalog](https://github.com/glebjan/hka.projektarbeit1.IQA_MRI/wiki/Metric-Catalog) explains the interpretation, the channel layout and the supported scoring modes of each metric.

### Signal-Based Full-Reference Metrics

These metrics compare intensities or hand-crafted features of input and target without a trained network. PSNR and SSIM are also available in volume mode, where MONAI computes them on the whole volume.

| Name | Metric | Better | Backend | Source |
|---|---|---|---|---|
| `psnr` | Peak Signal-to-Noise Ratio | higher | [pyiqa](https://github.com/chaofengc/IQA-PyTorch) and [MONAI](https://github.com/Project-MONAI/MONAI) | [Wang and Bovik 2009](https://doi.org/10.1109/MSP.2008.930649) |
| `ssim` | Structural Similarity Index Measure | higher | [pyiqa](https://github.com/chaofengc/IQA-PyTorch) and [MONAI](https://github.com/Project-MONAI/MONAI) | [Wang et al. 2004](https://doi.org/10.1109/TIP.2003.819861) |
| `fsim` | Feature Similarity Index Measure | higher | [pyiqa](https://github.com/chaofengc/IQA-PyTorch) | [Zhang et al. 2011](https://doi.org/10.1109/TIP.2011.2109730) |
| `gmsd` | Gradient Magnitude Similarity Deviation | lower | [pyiqa](https://github.com/chaofengc/IQA-PyTorch) | [Xue et al. 2014](https://doi.org/10.1109/TIP.2013.2293423) |
| `vsi` | Visual Saliency Induced Index | higher | [pyiqa](https://github.com/chaofengc/IQA-PyTorch) | [Zhang, Shen and Li 2014](https://doi.org/10.1109/TIP.2014.2346028) |

### Learning-Based Full-Reference Metrics

These metrics measure the distance between input and target in the feature space of a pretrained network. The metric `radimagenet_lpips` is a pyiqa plugin of this framework that replaces the LPIPS backbone by a ResNet50 pretrained on RadImageNet, a dataset of CT, MRI and ultrasound images.

| Name | Metric | Better | Backend | Source |
|---|---|---|---|---|
| `lpips` | Learned Perceptual Image Patch Similarity | lower | [pyiqa](https://github.com/chaofengc/IQA-PyTorch) | [Zhang et al. 2018](https://doi.org/10.1109/CVPR.2018.00068) |
| `dists` | Deep Image Structure and Texture Similarity | lower | [pyiqa](https://github.com/chaofengc/IQA-PyTorch) | [Ding et al. 2022](https://doi.org/10.1109/TPAMI.2020.3045810) |
| `radimagenet_lpips` | LPIPS with RadImageNet ResNet50 backbone | lower | pyiqa plugin with [RadImageNet](https://github.com/BMEII-AI/RadImageNet) weights | [Mei et al. 2022](https://doi.org/10.1148/ryai.210315) |
| `dreamsim` | DreamSim perceptual distance | lower | [DreamSim](https://github.com/ssundaram21/dreamsim) | [Fu et al. 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/9f09f316a3eaf59d9ced5ffaefe97e0f-Abstract-Conference.html) |

### No-Reference Metrics

These metrics assess the input without a target. The metrics `clip_iqa_lung` and `clip_iqa_brain` are pyiqa plugins of this framework that apply the CLIP-IQA principle with five organ-specific prompt pairs each. Most NR metrics were trained on natural photographs, so their scores serve for a relative comparison within one study rather than as absolute quality figures.

| Name | Metric | Better | Backend | Source |
|---|---|---|---|---|
| `clipiqa` | CLIP Image Quality Assessment | higher | [pyiqa](https://github.com/chaofengc/IQA-PyTorch) with [CLIP](https://github.com/openai/CLIP) | [Wang, Chan and Loy 2023](https://doi.org/10.1609/aaai.v37i2.25353) |
| `clip_iqa_lung` | CLIP-IQA with lung prompts | higher | pyiqa plugin with [CLIP](https://github.com/openai/CLIP) | [Wang, Chan and Loy 2023](https://doi.org/10.1609/aaai.v37i2.25353) |
| `clip_iqa_brain` | CLIP-IQA with brain prompts | higher | pyiqa plugin with [CLIP](https://github.com/openai/CLIP) | [Wang, Chan and Loy 2023](https://doi.org/10.1609/aaai.v37i2.25353) |
| `brisque` | Blind/Referenceless Image Spatial Quality Evaluator | lower | [pyiqa](https://github.com/chaofengc/IQA-PyTorch) | [Mittal, Moorthy and Bovik 2012](https://doi.org/10.1109/TIP.2012.2214050) |
| `niqe` | Natural Image Quality Evaluator | lower | [pyiqa](https://github.com/chaofengc/IQA-PyTorch) | [Mittal, Soundararajan and Bovik 2013](https://doi.org/10.1109/LSP.2012.2227726) |
| `ilniqe` | Integrated Local NIQE | lower | [pyiqa](https://github.com/chaofengc/IQA-PyTorch) | [Zhang, Zhang and Bovik 2015](https://doi.org/10.1109/TIP.2015.2426416) |
| `piqe` | Perception-based Image Quality Evaluator | lower | [pyiqa](https://github.com/chaofengc/IQA-PyTorch) | [Venkatanath et al. 2015](https://doi.org/10.1109/NCC.2015.7084843) |
| `musiq` | Multi-scale Image Quality Transformer | higher | [pyiqa](https://github.com/chaofengc/IQA-PyTorch) | [Ke et al. 2021](https://doi.org/10.1109/ICCV48922.2021.00510) |
| `maniqa` | Multi-dimension Attention Network for IQA | higher | [pyiqa](https://github.com/chaofengc/IQA-PyTorch) | [Yang et al. 2022](https://doi.org/10.1109/CVPRW56347.2022.00126) |
| `paq2piq` | PaQ-2-PiQ | higher | [pyiqa](https://github.com/chaofengc/IQA-PyTorch) | [Ying et al. 2020](https://doi.org/10.1109/CVPR42600.2020.00363) |

### Segmentation Metrics

These metrics compare a predicted mask with a reference mask and thereby measure the usefulness of an image for a downstream task. All of them support slice mode and volume mode. In volume mode, distances are measured in millimetres using the voxel spacing of the file. The counts `v_pred`, `v_gt` and `tp` are not quality scores. They allow per-slice results to be summed into correct per-volume values of `dice` and `vs`.

| Name | Metric | Better | Backend | Source |
|---|---|---|---|---|
| `dice` | Dice Similarity Coefficient | higher | [MONAI](https://github.com/Project-MONAI/MONAI) | [Maier-Hein et al. 2024](https://doi.org/10.1038/s41592-023-02151-z) |
| `hausdorff95` | 95th Percentile Hausdorff Distance | lower | [MONAI](https://github.com/Project-MONAI/MONAI) | [Maier-Hein et al. 2024](https://doi.org/10.1038/s41592-023-02151-z) |
| `nsd` | Normalized Surface Dice | higher | [MONAI](https://github.com/Project-MONAI/MONAI) | [Nikolov et al. 2021](https://doi.org/10.2196/26151) |
| `assd` | Average Symmetric Surface Distance | lower | [MONAI](https://github.com/Project-MONAI/MONAI) | [Maier-Hein et al. 2024](https://doi.org/10.1038/s41592-023-02151-z) |
| `panoptic_quality` | Panoptic Quality | higher | [MONAI](https://github.com/Project-MONAI/MONAI) | [Kirillov et al. 2019](https://doi.org/10.1109/CVPR.2019.00963) |
| `boundary_iou` | Boundary Intersection over Union | higher | Own implementation | [Cheng et al. 2021](https://doi.org/10.1109/CVPR46437.2021.01508) |
| `vs` | Volumetric Similarity | higher | Own implementation | [Taha and Hanbury 2015](https://doi.org/10.1186/s12880-015-0068-x) |
| `vs_signed` | Signed Volumetric Similarity | not ranked | Own implementation | [Taha and Hanbury 2015](https://doi.org/10.1186/s12880-015-0068-x) |
| `v_pred` | Voxel count of the prediction | not ranked | Own implementation | |
| `v_gt` | Voxel count of the reference | not ranked | Own implementation | |
| `tp` | Voxel count of the overlap | not ranked | Own implementation | |

Further metrics can be added through `MetricRegistry.register_metric()` without changes to the framework, as described in [Extending the Framework](https://github.com/glebjan/hka.projektarbeit1.IQA_MRI/wiki/Extending-the-Framework).

## Getting Started

The framework requires Python 3.12 or newer as well as the environment manager [Hatch](https://hatch.pypa.io) and the package installer [uv](https://docs.astral.sh/uv/). All commands run in the repository root, because the model paths resolve against the current working directory. Learned metrics download their weights on first use, with the exception of the RadImageNet backbone, which is part of the repository.

```bash
git clone https://github.com/glebjan/hka.projektarbeit1.IQA_MRI.git
cd hka.projektarbeit1.IQA_MRI
hatch env create
source .venv/bin/activate
```

The following example scores a generated volume against its reference volume with PSNR and SSIM and prints one row per slice.

```python
from pathlib import Path

from iqaevaluator.evaluation_result import EvaluationResult
from iqaevaluator.evaluator_factory import build_evaluator
from iqaevaluator.image_loader import load_pair
from iqaevaluator.metrics import PSNR, SSIM, MetricRegistry

registry = MetricRegistry(PSNR, SSIM)
image, target = load_pair(Path("data/generated/subject01_t1.nii.gz"),
                          Path("data/reference/subject01_t1.nii.gz"))
records = build_evaluator(image, target, registry, mode="slice").run_evaluation()
result = EvaluationResult.from_records(records, registry)
print(result.to_frame()[["image_id", "psnr", "ssim"]])
```

A command line entry point evaluates a file or a directory with all metrics of `BUILTIN_METRICS` and writes a report as comma-separated values (CSV).

```bash
hatch run evaluate data/generated data/reference --mode slice --normalization minmax
```

The wiki continues with the [Quick Start](https://github.com/glebjan/hka.projektarbeit1.IQA_MRI/wiki/Quick-Start), the [Scoring Modes](https://github.com/glebjan/hka.projektarbeit1.IQA_MRI/wiki/Scoring-Modes), the [Normalization](https://github.com/glebjan/hka.projektarbeit1.IQA_MRI/wiki/Normalization) strategies, the [Segmentation Evaluation](https://github.com/glebjan/hka.projektarbeit1.IQA_MRI/wiki/Segmentation-Evaluation) and the [Worked Examples](https://github.com/glebjan/hka.projektarbeit1.IQA_MRI/wiki/Worked-Examples). Development tools and the comparison against reference implementations are covered in [Testing and Calibration](https://github.com/glebjan/hka.projektarbeit1.IQA_MRI/wiki/Testing-and-Calibration).

## Acknowledgements

The framework builds on several open source projects. The library [pyiqa (IQA-PyTorch)](https://github.com/chaofengc/IQA-PyTorch) computes the image quality metrics in slice mode, and [MONAI](https://github.com/Project-MONAI/MONAI) computes the volumetric PSNR and SSIM as well as most segmentation metrics. The package [DreamSim](https://github.com/ssundaram21/dreamsim) provides the metric of the same name, [RadImageNet](https://github.com/BMEII-AI/RadImageNet) provides the medically pretrained ResNet50 backbone, and [CLIP](https://github.com/openai/CLIP) provides the backbone of the CLIP-based metrics. The calibration tests compare the segmentation metrics against [surface-distance](https://github.com/google-deepmind/surface-distance), [MedPy](https://github.com/loli/medpy) and [panopticapi](https://github.com/cocodataset/panopticapi). Scientific use of a metric should cite its original publication as listed in the tables above.

## License

The project is released under the MIT License, as declared in `pyproject.toml`.
