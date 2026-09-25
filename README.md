# iqaevaluator

A Python framework for assessing the quality of generated or reconstructed images. It works on 3D medical volumes (MRI, CT) and on ordinary 2D images. It scores an image against a reference, scores it without a reference, or compares segmentation masks. The result is one table of scores per slice or per volume.

Full documentation lives in the [project wiki](https://github.com/glebjan/hka.projektarbeit1.IQA_MRI/wiki).

## Metrics

The framework bundles 30 metrics in four families. Pick the family by what you have and what you want to know.

| Family | Needs reference | Use it when | Metrics |
|---|---|---|---|
| Signal-based full-reference | Yes | You have a ground truth and want to check pixel fidelity and local structure. Fast and easy to interpret. | `psnr`, `ssim`, `fsim`, `gmsd`, `vsi` |
| Learning-based full-reference | Yes | You have a ground truth and care about perceived similarity rather than exact pixels. | `lpips`, `dists`, `radimagenet_lpips`, `dreamsim` |
| No-reference | No | There is no ground truth, for example for purely generated images. | `clipiqa`, `clip_iqa_lung`, `clip_iqa_brain`, `brisque`, `niqe`, `ilniqe`, `piqe`, `musiq`, `maniqa`, `paq2piq` |
| Segmentation | Yes (reference mask) | You want to know if an image still works for a downstream task such as organ segmentation. | `dice`, `hausdorff95`, `nsd`, `assd`, `panoptic_quality`, `boundary_iou`, `vs`, `vs_signed`, `v_pred`, `v_gt`, `tp` |

A few rules of thumb help with the choice.

1. With a reference, combine one signal-based metric (e.g. `ssim`) with one learned metric (e.g. `lpips`). They catch different errors.
2. Most learned and no-reference metrics were trained on natural photos. Use them to rank models within one study, not as absolute quality values.
3. For medical images, `radimagenet_lpips` uses features learned on CT, MRI and ultrasound. `clip_iqa_lung` and `clip_iqa_brain` use organ-specific prompts.
4. Only `psnr`, `ssim` and the segmentation metrics can score a whole 3D volume at once. All others score slice by slice.
5. `maniqa` and `ilniqe` are much slower than the rest. Leave them out for quick runs.

Expand a family to see what each metric captures. The [Metric Catalog](https://github.com/glebjan/hka.projektarbeit1.IQA_MRI/wiki/Metric-Catalog) and [Selecting Metrics](https://github.com/glebjan/hka.projektarbeit1.IQA_MRI/wiki/Selecting-Metrics) in the wiki go into detail.

<details>
<summary>Signal-based full-reference metrics</summary>

| Name | Captures | Better | Backend | Paper |
|---|---|---|---|---|
| `psnr` | Pixel-wise error. A simple baseline that ignores structure. | higher | [pyiqa](https://github.com/chaofengc/IQA-PyTorch), [MONAI](https://github.com/Project-MONAI/MONAI) | [Wang and Bovik 2009](https://doi.org/10.1109/MSP.2008.930649) |
| `ssim` | Local luminance, contrast and structure. | higher | [pyiqa](https://github.com/chaofengc/IQA-PyTorch), [MONAI](https://github.com/Project-MONAI/MONAI) | [Wang et al. 2004](https://doi.org/10.1109/TIP.2003.819861) |
| `fsim` | Edges and other salient structures. | higher | [pyiqa](https://github.com/chaofengc/IQA-PyTorch) | [Zhang et al. 2011](https://doi.org/10.1109/TIP.2011.2109730) |
| `gmsd` | Distortions of image gradients. | lower | [pyiqa](https://github.com/chaofengc/IQA-PyTorch) | [Xue et al. 2014](https://doi.org/10.1109/TIP.2013.2293423) |
| `vsi` | Errors in visually salient regions. | higher | [pyiqa](https://github.com/chaofengc/IQA-PyTorch) | [Zhang, Shen and Li 2014](https://doi.org/10.1109/TIP.2014.2346028) |

</details>

<details>
<summary>Learning-based full-reference metrics</summary>

| Name | Captures | Better | Backend | Paper |
|---|---|---|---|---|
| `lpips` | Perceived difference in the features of a photo-trained network. | lower | [pyiqa](https://github.com/chaofengc/IQA-PyTorch) | [Zhang et al. 2018](https://doi.org/10.1109/CVPR.2018.00068) |
| `dists` | Structure and texture. Tolerates slightly shifted texture. | lower | [pyiqa](https://github.com/chaofengc/IQA-PyTorch) | [Ding et al. 2022](https://doi.org/10.1109/TPAMI.2020.3045810) |
| `radimagenet_lpips` | Like `lpips`, but with features learned on medical images. | lower | pyiqa plugin, [RadImageNet](https://github.com/BMEII-AI/RadImageNet) weights | [Mei et al. 2022](https://doi.org/10.1148/ryai.210315) |
| `dreamsim` | Layout and shape. Ignores fine noise and texture. | lower | [DreamSim](https://github.com/ssundaram21/dreamsim) | [Fu et al. 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/9f09f316a3eaf59d9ced5ffaefe97e0f-Abstract-Conference.html) |

</details>

<details>
<summary>No-reference metrics</summary>

| Name | Captures | Better | Backend | Paper |
|---|---|---|---|---|
| `clipiqa` | A generic "good or bad image" judgement by CLIP. | higher | [pyiqa](https://github.com/chaofengc/IQA-PyTorch), [CLIP](https://github.com/openai/CLIP) | [Wang, Chan and Loy 2023](https://doi.org/10.1609/aaai.v37i2.25353) |
| `clip_iqa_lung` | Sharpness, noise, contrast and artefacts in lung scans. | higher | pyiqa plugin, [CLIP](https://github.com/openai/CLIP) | [Wang, Chan and Loy 2023](https://doi.org/10.1609/aaai.v37i2.25353) |
| `clip_iqa_brain` | Sharpness, noise, contrast and artefacts in brain scans. | higher | pyiqa plugin, [CLIP](https://github.com/openai/CLIP) | [Wang, Chan and Loy 2023](https://doi.org/10.1609/aaai.v37i2.25353) |
| `brisque` | Deviation from natural image statistics. | lower | [pyiqa](https://github.com/chaofengc/IQA-PyTorch) | [Mittal, Moorthy and Bovik 2012](https://doi.org/10.1109/TIP.2012.2214050) |
| `niqe` | Distance to a model of undistorted images. Needs no training ratings. | lower | [pyiqa](https://github.com/chaofengc/IQA-PyTorch) | [Mittal, Soundararajan and Bovik 2013](https://doi.org/10.1109/LSP.2012.2227726) |
| `ilniqe` | Like `niqe`, but patch by patch. Slow. | lower | [pyiqa](https://github.com/chaofengc/IQA-PyTorch) | [Zhang, Zhang and Bovik 2015](https://doi.org/10.1109/TIP.2015.2426416) |
| `piqe` | Distortion in detailed image blocks. | lower | [pyiqa](https://github.com/chaofengc/IQA-PyTorch) | [Venkatanath et al. 2015](https://doi.org/10.1109/NCC.2015.7084843) |
| `musiq` | Overall perceived quality at native resolution. | higher | [pyiqa](https://github.com/chaofengc/IQA-PyTorch) | [Ke et al. 2021](https://doi.org/10.1109/ICCV48922.2021.00510) |
| `maniqa` | Artefacts typical of GAN-restored images. Slowest metric. | higher | [pyiqa](https://github.com/chaofengc/IQA-PyTorch) | [Yang et al. 2022](https://doi.org/10.1109/CVPRW56347.2022.00126) |
| `paq2piq` | Global and local quality of real photos. | higher | [pyiqa](https://github.com/chaofengc/IQA-PyTorch) | [Ying et al. 2020](https://doi.org/10.1109/CVPR42600.2020.00363) |

</details>

<details>
<summary>Segmentation metrics</summary>

| Name | Captures | Better | Backend | Paper |
|---|---|---|---|---|
| `dice` | Overlap of the two masks. The standard choice. | higher | [MONAI](https://github.com/Project-MONAI/MONAI) | [Maier-Hein et al. 2024](https://doi.org/10.1038/s41592-023-02151-z) |
| `hausdorff95` | Near worst-case boundary distance, robust to single outliers. | lower | [MONAI](https://github.com/Project-MONAI/MONAI) | [Maier-Hein et al. 2024](https://doi.org/10.1038/s41592-023-02151-z) |
| `nsd` | Share of the boundary within a tolerance. | higher | [MONAI](https://github.com/Project-MONAI/MONAI) | [Nikolov et al. 2021](https://doi.org/10.2196/26151) |
| `assd` | Average boundary distance. | lower | [MONAI](https://github.com/Project-MONAI/MONAI) | [Maier-Hein et al. 2024](https://doi.org/10.1038/s41592-023-02151-z) |
| `panoptic_quality` | Detection and overlap of individual instances. | higher | [MONAI](https://github.com/Project-MONAI/MONAI) | [Kirillov et al. 2019](https://doi.org/10.1109/CVPR.2019.00963) |
| `boundary_iou` | Overlap along the contour. Fair to small and large objects. | higher | Own implementation | [Cheng et al. 2021](https://doi.org/10.1109/CVPR46437.2021.01508) |
| `vs` | Size agreement only. Read together with `dice`. | higher | Own implementation | [Taha and Hanbury 2015](https://doi.org/10.1186/s12880-015-0068-x) |
| `vs_signed` | Whether the prediction is too small or too large. | not ranked | Own implementation | [Taha and Hanbury 2015](https://doi.org/10.1186/s12880-015-0068-x) |
| `v_pred`, `v_gt`, `tp` | Voxel counts for summing slice results into volume scores. | not ranked | Own implementation | |

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

MIT, as declared in `pyproject.toml`.
