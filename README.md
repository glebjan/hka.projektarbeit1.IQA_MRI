# iqaevaluator

A Python framework for assessing the quality of generated or reconstructed images. It works on 3D medical volumes (MRI, CT) and on ordinary 2D images. It scores an image against a reference, scores it without a reference, or compares segmentation masks. The result is one table of scores per slice or per volume.

Full documentation lives in the [project wiki](https://github.com/glebjan/hka.projektarbeit1.IQA_MRI/wiki).

## Metrics

The framework bundles 30 metrics in four families. Expand a family to see its metrics, backends and papers. The [Metric Catalog](https://github.com/glebjan/hka.projektarbeit1.IQA_MRI/wiki/Metric-Catalog) explains each metric in detail.

| Family | Needs reference | Metrics |
|---|---|---|
| Signal-based full-reference | Yes | `psnr`, `ssim`, `fsim`, `gmsd`, `vsi` |
| Learning-based full-reference | Yes | `lpips`, `dists`, `radimagenet_lpips`, `dreamsim` |
| No-reference | No | `clipiqa`, `clip_iqa_lung`, `clip_iqa_brain`, `brisque`, `niqe`, `ilniqe`, `piqe`, `musiq`, `maniqa`, `paq2piq` |
| Segmentation | Yes | `dice`, `hausdorff95`, `nsd`, `assd`, `panoptic_quality`, `boundary_iou`, `vs`, `vs_signed`, `v_pred`, `v_gt`, `tp` |

<details>
<summary>Signal-based full-reference metrics</summary>

| Name | Metric | Better | Backend | Paper |
|---|---|---|---|---|
| `psnr` | Peak Signal-to-Noise Ratio | higher | [pyiqa](https://github.com/chaofengc/IQA-PyTorch), [MONAI](https://github.com/Project-MONAI/MONAI) | [Wang and Bovik 2009](https://doi.org/10.1109/MSP.2008.930649) |
| `ssim` | Structural Similarity Index Measure | higher | [pyiqa](https://github.com/chaofengc/IQA-PyTorch), [MONAI](https://github.com/Project-MONAI/MONAI) | [Wang et al. 2004](https://doi.org/10.1109/TIP.2003.819861) |
| `fsim` | Feature Similarity Index Measure | higher | [pyiqa](https://github.com/chaofengc/IQA-PyTorch) | [Zhang et al. 2011](https://doi.org/10.1109/TIP.2011.2109730) |
| `gmsd` | Gradient Magnitude Similarity Deviation | lower | [pyiqa](https://github.com/chaofengc/IQA-PyTorch) | [Xue et al. 2014](https://doi.org/10.1109/TIP.2013.2293423) |
| `vsi` | Visual Saliency Induced Index | higher | [pyiqa](https://github.com/chaofengc/IQA-PyTorch) | [Zhang, Shen and Li 2014](https://doi.org/10.1109/TIP.2014.2346028) |

</details>

<details>
<summary>Learning-based full-reference metrics</summary>

| Name | Metric | Better | Backend | Paper |
|---|---|---|---|---|
| `lpips` | Learned Perceptual Image Patch Similarity | lower | [pyiqa](https://github.com/chaofengc/IQA-PyTorch) | [Zhang et al. 2018](https://doi.org/10.1109/CVPR.2018.00068) |
| `dists` | Deep Image Structure and Texture Similarity | lower | [pyiqa](https://github.com/chaofengc/IQA-PyTorch) | [Ding et al. 2022](https://doi.org/10.1109/TPAMI.2020.3045810) |
| `radimagenet_lpips` | LPIPS with a medical ResNet50 backbone | lower | pyiqa plugin, [RadImageNet](https://github.com/BMEII-AI/RadImageNet) weights | [Mei et al. 2022](https://doi.org/10.1148/ryai.210315) |
| `dreamsim` | DreamSim perceptual distance | lower | [DreamSim](https://github.com/ssundaram21/dreamsim) | [Fu et al. 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/9f09f316a3eaf59d9ced5ffaefe97e0f-Abstract-Conference.html) |

</details>

<details>
<summary>No-reference metrics</summary>

| Name | Metric | Better | Backend | Paper |
|---|---|---|---|---|
| `clipiqa` | CLIP Image Quality Assessment | higher | [pyiqa](https://github.com/chaofengc/IQA-PyTorch), [CLIP](https://github.com/openai/CLIP) | [Wang, Chan and Loy 2023](https://doi.org/10.1609/aaai.v37i2.25353) |
| `clip_iqa_lung` | CLIP-IQA with lung prompts | higher | pyiqa plugin, [CLIP](https://github.com/openai/CLIP) | [Wang, Chan and Loy 2023](https://doi.org/10.1609/aaai.v37i2.25353) |
| `clip_iqa_brain` | CLIP-IQA with brain prompts | higher | pyiqa plugin, [CLIP](https://github.com/openai/CLIP) | [Wang, Chan and Loy 2023](https://doi.org/10.1609/aaai.v37i2.25353) |
| `brisque` | Blind/Referenceless Image Spatial Quality Evaluator | lower | [pyiqa](https://github.com/chaofengc/IQA-PyTorch) | [Mittal, Moorthy and Bovik 2012](https://doi.org/10.1109/TIP.2012.2214050) |
| `niqe` | Natural Image Quality Evaluator | lower | [pyiqa](https://github.com/chaofengc/IQA-PyTorch) | [Mittal, Soundararajan and Bovik 2013](https://doi.org/10.1109/LSP.2012.2227726) |
| `ilniqe` | Integrated Local NIQE | lower | [pyiqa](https://github.com/chaofengc/IQA-PyTorch) | [Zhang, Zhang and Bovik 2015](https://doi.org/10.1109/TIP.2015.2426416) |
| `piqe` | Perception-based Image Quality Evaluator | lower | [pyiqa](https://github.com/chaofengc/IQA-PyTorch) | [Venkatanath et al. 2015](https://doi.org/10.1109/NCC.2015.7084843) |
| `musiq` | Multi-scale Image Quality Transformer | higher | [pyiqa](https://github.com/chaofengc/IQA-PyTorch) | [Ke et al. 2021](https://doi.org/10.1109/ICCV48922.2021.00510) |
| `maniqa` | Multi-dimension Attention Network for IQA | higher | [pyiqa](https://github.com/chaofengc/IQA-PyTorch) | [Yang et al. 2022](https://doi.org/10.1109/CVPRW56347.2022.00126) |
| `paq2piq` | PaQ-2-PiQ | higher | [pyiqa](https://github.com/chaofengc/IQA-PyTorch) | [Ying et al. 2020](https://doi.org/10.1109/CVPR42600.2020.00363) |

</details>

<details>
<summary>Segmentation metrics</summary>

| Name | Metric | Better | Backend | Paper |
|---|---|---|---|---|
| `dice` | Dice Similarity Coefficient | higher | [MONAI](https://github.com/Project-MONAI/MONAI) | [Maier-Hein et al. 2024](https://doi.org/10.1038/s41592-023-02151-z) |
| `hausdorff95` | 95th Percentile Hausdorff Distance | lower | [MONAI](https://github.com/Project-MONAI/MONAI) | [Maier-Hein et al. 2024](https://doi.org/10.1038/s41592-023-02151-z) |
| `nsd` | Normalized Surface Dice | higher | [MONAI](https://github.com/Project-MONAI/MONAI) | [Nikolov et al. 2021](https://doi.org/10.2196/26151) |
| `assd` | Average Symmetric Surface Distance | lower | [MONAI](https://github.com/Project-MONAI/MONAI) | [Maier-Hein et al. 2024](https://doi.org/10.1038/s41592-023-02151-z) |
| `panoptic_quality` | Panoptic Quality | higher | [MONAI](https://github.com/Project-MONAI/MONAI) | [Kirillov et al. 2019](https://doi.org/10.1109/CVPR.2019.00963) |
| `boundary_iou` | Boundary Intersection over Union | higher | Own implementation | [Cheng et al. 2021](https://doi.org/10.1109/CVPR46437.2021.01508) |
| `vs` | Volumetric Similarity | higher | Own implementation | [Taha and Hanbury 2015](https://doi.org/10.1186/s12880-015-0068-x) |
| `vs_signed` | Signed Volumetric Similarity | not ranked | Own implementation | [Taha and Hanbury 2015](https://doi.org/10.1186/s12880-015-0068-x) |
| `v_pred`, `v_gt`, `tp` | Voxel counts of prediction, reference and overlap | not ranked | Own implementation | |

</details>

## Getting Started

You need Python 3.12 or newer, [Hatch](https://hatch.pypa.io) and [uv](https://docs.astral.sh/uv/). Run all commands from the repository root.

```bash
git clone https://github.com/glebjan/hka.projektarbeit1.IQA_MRI.git
cd hka.projektarbeit1.IQA_MRI
hatch env create
source .venv/bin/activate
```

Score a generated volume against its reference with PSNR and SSIM.

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
print(EvaluationResult.from_records(records, registry).to_frame())
```

The [Quick Start](https://github.com/glebjan/hka.projektarbeit1.IQA_MRI/wiki/Quick-Start) in the wiki goes further.

## Acknowledgements

The metrics come from [pyiqa](https://github.com/chaofengc/IQA-PyTorch), [MONAI](https://github.com/Project-MONAI/MONAI), [DreamSim](https://github.com/ssundaram21/dreamsim), [RadImageNet](https://github.com/BMEII-AI/RadImageNet) and [CLIP](https://github.com/openai/CLIP). Please cite the original paper of each metric you use.

## License

MIT
