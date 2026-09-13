# Ultrareview 2026-09-13: Paketstruktur, Metrik-Kalibrierung, Testabdeckung

> **Kontextdatei für die lokale Umsetzung.** Dieses Dokument ist das vollständige, verifizierte
> Ergebnis eines Code-Reviews auf maximaler Aufwandsstufe (8 unabhängige Finder, 4 Verifier,
> jede Behauptung gegen Quellcode und die entpackten Upstream-Bibliotheken pyiqa 0.1.15.post2,
> MONAI 1.6.0 und dreamsim 0.2.1 geprüft). Es ist so geschrieben, dass eine lokale Claude-Code-
> Session oder ein Mensch die Punkte ohne weiteren Kontext abarbeiten kann.
>
> **Review-Scope:** `git diff volumetric_similarity...HEAD` (HEAD = 0047d24, Merge von PR #4
> „refactor: reorganize module structure“) plus framework-weit alle Metrik-Implementierungen und
> ihre Tests, weil die Leitfrage lautete: *Werden die Metriken gegen einen Goldstandard getestet,
> ist das Framework also für wissenschaftliche Nutzung kalibriert?*
>
> **Zeilenangaben** beziehen sich auf den Stand von HEAD 0047d24 in `src/iqaevaluator/`.
> Nach Phase 1 (Import-Umbau) können sich Zeilen verschieben — immer per Symbolname suchen.

---

## 0. Kurzantwort auf die Leitfrage

**Nein, das Framework ist in diesem Zustand nicht kalibriert.**

1. **Es wird derzeit gar nichts getestet.** Der Paket-Umzug hat `tests/conftest.py`, die
   paketinternen Imports und `src/main.py` nicht nachgezogen. Alle 20 Testdateien brechen beim
   Sammeln mit `ModuleNotFoundError` ab. Der einzige Cross-Implementierungs-Check (Volume-Dice ==
   aggregierter Slice-Dice) importiert das gelöschte `main.evaluate`.
2. **Auch vor dem Umzug fehlten Referenzwerte.** Echte Goldstandard-Pins gibt es nur für
   `boundary_iou` (2D, gegen das Erosions-Rezept des Papers, Anker 0.3822) und für
   `vs`/`v_pred`/`tp`/`aggregate_patient`. Die fünf MONAI-Metriken (Dice, HD95, NSD, ASSD, PQ) und
   PSNR/SSIM (pyiqa-Slice und MONAI-3D) werden nur auf Identität (=1.0/=0.0) und Monotonie
   („verrauscht ist schlechter“) geprüft. Ein weggefallenes `percentile=95`, ein Tausch Dice↔IoU
   oder ein falsches `data_range` bliebe unbemerkt.
3. **Mehrere Metriken liefern nicht das, was Docstrings/Spec behaupten** (Details in Abschnitt 3):
   Slice-SSIM wird auf 8 Bit quantisiert, PSNR weicht von der beanspruchten fastMRI-Konvention ab,
   3D-SSIM mischt Fenstergrößen, DreamSim resized bilinear statt bicubic, `threshold=None` lässt
   MONAI nur Label 1 werten, `load_pair` binarisiert Masken falsch, `aggregate_volumes` nullt
   Slices mit echten Voxeln.

**Empfohlene Reihenfolge:** Phase 1 (Suite lauffähig machen) → Phase 2 (Referenzwert-Tests)
→ Phase 3 (Semantik-Fixes) → Phase 4 (Aufräumen). Phase 2 vor Phase 3, damit die Fixes gegen
gepinnte Werte laufen.

---

## 1. Phase 1 — Paket wieder importierbar machen (Blocker)

Alle Punkte in dieser Phase sind CONFIRMED. Ohne sie läuft kein einziger Test.

### 1.1 Flat-Imports im Paket → absolute Paketimporte
**Datei:** `src/iqaevaluator/metrics.py:30-33` und alle anderen Module.

Betroffen (flat): `metrics.py:30` `import radimagenet_lpips`, `:33` `from constants import RESNET50`,
`:272-285` Bottom-Imports; `dreamsim_metric.py:29-30`; `image_loader.py:21`; `iqa_evaluator.py:7-9`;
`volume_evaluator.py:31-34`; `evaluator_factory.py:15-18`; `evaluation_result.py:10-12`;
`segmentation_metrics/monai_metrics.py:54`, `volume_metrics.py:31`, `boundary_iou.py:49`.
Nur `data.py:10` ist bereits auf `from iqaevaluator.image_loader import ...` umgestellt — beide
Varianten können nicht gleichzeitig auflösen.

**Fix:** Überall `from iqaevaluator.<modul> import ...` bzw.
`from iqaevaluator.segmentation_metrics.<modul> import ...`. Paket editable installieren
(`pip install -e .`), `sys.path`-Hack in `tests/conftest.py:15` entfernen, alle Tests auf
Paketimporte umstellen (`from iqaevaluator.metrics import ...`).

**Nachweis:** `python -c "import iqaevaluator.metrics"` und `pytest --collect-only` ohne Fehler.

### 1.2 Import-Zyklus auflösen (gleichzeitig mit 1.1)
**Datei:** `src/iqaevaluator/metrics.py:272-285`.

`metrics.py` importiert am Dateiende `dreamsim_metric` und `segmentation_metrics.*`, die oben
`from metrics import MetricSpec, ModeSupport, ...` ziehen. Wer zuerst `dreamsim_metric` importiert,
bekommt „cannot import name … from partially initialized module“. `boundary_iou.py:34-39` und
`tests/test_dreamsim_metric.py:11-16` dokumentieren die Reihenfolge statt sie zu beheben.

**Fix:** `MetricSpec`, `ModeSupport`, `ModeUnsupported`, `ModeCapability`, `SkippedMetric`,
`Spacing`, `REASON_*` und das `Metric`-Protokoll in ein Leaf-Modul `iqaevaluator/metric_spec.py`
verschieben. `metrics.py` und alle Metrik-Module importieren daraus; Bottom-Imports werden
normale Top-Imports; die Reihenfolge-Kommentare entfallen.

### 1.3 `evaluate()`/CLI wiederherstellen, `main.py.bak` löschen
**Datei:** `src/main.py:1` (nur `import iqaevaluator`), `src/main.py.bak` (225 Zeilen, getrackt,
byteidentisch mit dem alten `main.py`), `src/iqaevaluator/__init__.py` (0 Bytes).

Stale Referenzen: `tests/test_main.py:9,134,138,146,151,213,390,393`, `tests/test_metrics.py:592-593`
(`main.DREAMSIM`), `tests/test_boundary_iou.py:289-290` (`main.BOUNDARY_IOU`),
`tests/test_volume_metrics.py:104` (`from main import evaluate`), `src/evaluation.ipynb:162`
(`from main import evaluate, MinMax`).

**Fix:**
- `evaluate()`, `report_skipped_metrics()`, `main()` aus `main.py.bak` nach
  `src/iqaevaluator/cli.py` (oder `evaluate.py`).
- `src/iqaevaluator/__init__.py`: öffentliche API re-exportieren (alles, was `main.py.bak:9-22`
  re-exportierte: Metrik-Specs, `MetricRegistry`, Normalizer `MinMax`/`Percentile`/`Raw`/
  `FixedRange`, `evaluate`).
- `pyproject.toml`: `[project.scripts] iqaevaluator = "iqaevaluator.cli:main"`.
- `src/main.py` und `src/main.py.bak` löschen (Git-History behält die alte Version).
- Tests und Notebook auf `from iqaevaluator import evaluate, MinMax, ...` umstellen.

### 1.4 `pyproject.toml` reparieren
- **Zeile 9** `readme = "README.md"`: Datei existiert nicht und würde von `.gitignore:4` (`*`)
  ignoriert. hatchling bricht mit `OSError: Readme file does not exist` ab — `pip install -e .`
  ist damit unmöglich. **Fix:** README.md anlegen und in `.gitignore` freischalten
  (`!README.md`), oder den `readme`-Key entfernen.
- **Zeile 10** `requires-python = ">=3.8"` + Classifier 3.8/3.9: `metrics.py:70`
  `ModeCapability = ModeSupport | ModeUnsupported` und `dreamsim_metric.py:99,227`
  `Optional[str | torch.device]` sind Laufzeit-PEP-604 (kein `from __future__ import
  annotations`) → TypeError unter 3.9. **Fix:** `requires-python = ">=3.10"`, Classifier
  3.8/3.9 streichen.
- **Zeile 27-148** `dependencies`: 120 exakte `==`-Pins (Kopie von requirements.txt) inkl.
  Dev-Tools (pytest, ruff, pre_commit, virtualenv, tensorboard, yapf, nodeenv, identify, cfgv),
  `opencv-python` UND `opencv-python-headless` (konfliktierend), `torch==2.14.0`, `dreamsim`
  hart gepinnt obwohl `dreamsim_metric.py:49-62` es lazy mit Install-Hinweis importiert.
  **Fix:** ~12 direkte Laufzeit-Deps mit Untergrenzen (torch, torchvision, pyiqa, monai, numpy,
  pandas, nibabel, pydicom, SimpleITK, pillow, scipy, scikit-image); `dreamsim`/`open_clip_torch`
  unter `[project.optional-dependencies] dreamsim = [...]`; pytest/ruff unter `dev`;
  `requirements.txt` bleibt als Lock.

### 1.5 Debug-Rest in `data.py`
**Datei:** `src/iqaevaluator/data.py:52-54`: `# analyze(DATA_PATH, REPORT_PATH)` und
`print("test")`. **Fix:** Aufruf wiederherstellen, print entfernen.

---

## 2. Phase 2 — Goldstandard-Tests ergänzen (Kern der Leitfrage)

Grundsatz: Jede Metrik bekommt mindestens **einen `pytest.approx`-Pin auf einen analytisch
hergeleiteten Wert** und, wo eine unabhängige Referenz existiert, einen Vergleich gegen diese
(scipy/MedPy/skimage/offizieller Code). Identität und Monotonie bleiben als Zusatztests.

### 2.1 MONAI-Segmentierungsmetriken — `tests/test_segmentation_metrics.py:223`
Fixture `_volume_pair` (Z.204-210): GT = Block D 2:4, H 3:9, W 3:9 (2×6×6 = 72 Voxel),
Pred = gleicher Block um 1 Voxel in W verschoben (W 4:10). Schnitt = 60. Spacing ist (D, H, W).

Verifizierte geschlossene Werte (mit reinem Python nachgerechnet, MONAI-Kantenextraktion aus
`monai/metrics/utils.py:234-235` berücksichtigt — bei 2 Voxel Dicke ist jeder Voxel Kante):

| Metrik | isotrop (1,1,1) | Spacing (1,1,3) | Anmerkung |
|---|---|---|---|
| Dice | 120/144 = **0.8333** | 0.8333 | IoU wäre 60/84 = 0.714 |
| HD95 | **1.0** | **3.0** | 12 von 72 Kantenvoxeln je Seite bei Distanz 1 |
| ASSD (symmetrisch) | 24/144 = **0.1667** | **0.5** | directed ASD in beide Richtungen ebenfalls 1/6 |
| NSD @ class_thresholds=[1.0] | **1.0** | 120/144 = **0.8333** | |

Aktuell asserted: `0.0 < dice < 1.0` (Z.223), `coarse > isotropic` (Z.234), `> 0.0` (Z.239),
Identität 1.0/0.0 (Z.101/122/150/168). **Konsequenz:** `setdefault('percentile', 95)`
(`monai_metrics.py:204`) oder `setdefault('symmetric', True)` (`:315`) zu streichen, oder Dice
durch IoU zu ersetzen, lässt die gesamte Suite grün, weil auf diesem Fixture percentile=95 und
plain max beide 1.0 ergeben und directed == symmetric.

**Zu ergänzende Tests:**
- Die vier Werte oben je isotrop und anisotrop mit `pytest.approx(..., abs=1e-6)` pinnen.
- Ein **zweites Fixture**, auf dem percentile=95 ≠ max und directed ≠ symmetric ist (z. B.
  Pred = GT plus ein einzelner Ausreißer-Voxel in Distanz 5: HD = 5.0, HD95 < 5.0; ASSD
  asymmetrisch).
- Cross-Check gegen eine unabhängige Implementierung (MedPy `hd95`/`assd`, oder
  `scipy.ndimage.distance_transform_edt` von Hand) auf beiden Fixtures.
- PQ: aktuell nur identische Masken (Z.246). Fixture mit 2 GT-Instanzen, 1 matched (IoU > 0.5),
  1 FN, 1 FP → PQ = SQ·RQ analytisch pinnen.
- Semantik-Tests (siehe 3.7): leeres GT + FP → welcher Wert? leere Pred → inf oder None?

### 2.2 PSNR/SSIM — `tests/test_metrics.py:199`, `tests/test_volumetric_iqa.py:24-32`
Aktuell: identisch → `> 40.0`, verrauscht → niedriger. Kein Test pinnt einen Wert; für pyiqa-SSIM
existiert **gar kein** numerischer Test (nur `record.ssim is not None`).

**Zu ergänzende Tests:**
- PSNR: zwei konstante Bilder 0.5 vs 0.6 → **20.0 dB** (mit pyiqa-eps 1e-8: 19.999996);
  identisch → pyiqa liefert exakt **80.0 dB** (eps-bedingt). Für MONAI-`PSNRMetric(max_val=1.0)`
  denselben Wert pinnen; pyiqa-Slice und MONAI-Volume müssen auf demselben Volumen bis 1e-6
  übereinstimmen.
- SSIM: bekanntes Paar gegen `skimage.metrics.structural_similarity(gaussian_weights=True,
  sigma=1.5, use_sample_covariance=False, data_range=...)` pinnen — **Achtung**: wegen 3.2
  (8-Bit-Rundung in pyiqa) muss das Testbild vorher auf `round(x*255)/255` quantisiert werden,
  sonst weicht der Vergleich ab. Genau dieser Test macht die Quantisierung sichtbar.
- Ein Test, der ein absichtlich falsches `data_range` (255 statt 1.0) erkennen würde: identische
  Bilder mit einem Pixel Differenz 1/255 → PSNR = 20·log10(255·√N) − …; der Wert unterscheidet
  sich um 48 dB je nach data_range.

### 2.3 Boundary-IoU 3D/physisch — `tests/test_boundary_iou.py:337-353`
2D-Pfad ist korrekt gepinnt (Anker 0.3822, iteriertes Erosions-Rezept — verifiziert). Der
3D-/physische Pfad hat keinen geschlossenen Wert: Z.337-341 asserted nur
`coarse.sum() < isotropic.sum()` — besteht auch mit **umgekehrtem** Spacing-Tupel
(iso = 528, coarse (3,1,1) = 384, reversed (1,1,3) = 480). Z.350-353 rechnet die
Implementierungsformel nach (Tautologie).

**Zu ergänzende Tests:** Für den 1-Voxel-verschobenen Würfel den EDT-Bandwert analytisch
herleiten und pinnen; ein Test, der das Spacing-Tupel als (D,H,W) festnagelt (Verschiebung
entlang D vs. W muss unterschiedliche Scores geben); siehe 3.9 für den dabei sichtbar werdenden Bug.

### 2.4 Normalisierung — `tests/test_normalization.py`
Kein Test vergleicht die neue `MinMax` gegen die alte Formel `(arr-lo)/(hi-lo+1e-8)` auf den
PNG/NIfTI/DICOM-Fixtures, obwohl die Spec
(`docs/superpowers/specs/2026-09-11-intensity-normalization-design.md:246-248, 277-278`)
„numerically unchanged … bit-for-bit“ verspricht. Siehe 3.11 — die Behauptung ist für kleine
Float-Ranges falsch. **Test:** alt vs. neu auf uint8-, int16- und float-[0,0.05]-Arrays; für die
Float-Range entweder Abweichung akzeptieren und die Spec korrigieren, oder Epsilon behalten.

### 2.5 Cross-Implementierungs-Check reaktivieren
`tests/test_volume_metrics.py:100-133` `TestCrossCheck.test_volume_dice_equals_aggregated_slice_dice`
— nach Phase 1 wieder lauffähig. Zusätzlich auf eine **Multi-Label-Raw-Map** ausweiten (deckt 3.6 auf).

---

## 3. Phase 3 — Semantik-Fehler in Metriken und Loader (alle CONFIRMED, außer markiert)

### 3.1 `load_pair` skaliert den Input immer mit der Target-Range → Masken falsch
**`src/iqaevaluator/image_loader.py:336-337`**: `inp = ImageLoader(input_path,
FixedRange(target.intensity_range))`. Die Baseline (`volumetric_similarity`) normalisierte beide
Seiten unabhängig.

**Fehlerfall:** GT als 0/255-uint8-PNG, Prediction als 0/1-uint8-NIfTI, `evaluate(...,
MetricRegistry(DICE, VS))` mit Default-MinMax: Input-Vordergrund wird 1/255 ≈ 0.0039 →
`as_mask(>=0.5)` und MONAI-Binarisierung alles False → **vs = 0.0, dice = 0.0 für eine perfekte
Prediction**, ohne Fehler. Vor dem Branch: 1.0.

**Fix:** Masken gehören nicht auf die Target-Range. Entweder Masken-Runs explizit mit `Raw()`
fahren und `load_pair` die Kopplung nur für Intensitätsbilder anwenden, oder `load_pair` prüft
die Kodierung (Wertemenge ⊆ {0, max}) und lehnt gemischte Kodierungen ab. `scale_source='target'`
im Report vermerken statt eine `FixedRange` als „minmax“ zu etikettieren.

### 3.2 Konstantes Target lässt `scale()` den Input nur clippen
**`src/iqaevaluator/normalization.py:68-76`**: Im `else`-Zweig (`hi == lo`) wird `arr` nicht
skaliert, nur `np.clip(arr, 0, 1)` (Z.76); die Warnung nennt `label` = die **Input**-Datei als
„constant image“. `FixedRange.__post_init__` (Z.129) lehnt nur `hi < lo` ab.

**Fehlerfall:** leeres/konstantes Target (blanke Referenz, zero-padded Volume) oder unter
`Percentile()` ein Target mit > 99.5 % Hintergrund (lo = hi = 0): eine 0..255-Prediction wird zu
lauter Einsen, Report schreibt `scale_lo == scale_hi`, PSNR/SSIM/LPIPS laufen auf binarisierten
Daten — **eine Garbage-Prediction über blanker Referenz scort perfekt.**

**Fix:** Eine degenerierte Referenz-Range ist keine Skala: `load_pair` muss
`target.intensity_range.hi == lo` ablehnen (Exception mit klarer Meldung, der Fall landet als
„evaluation failed“ im Report). Den Masken-Sonderfall aus `scale()` entfernen — Masken gehen laut
Spec über `Raw()`.

### 3.3 `threshold=None` → MONAI wertet nur Label == 1
**`src/iqaevaluator/segmentation_metrics/monai_metrics.py:126-127`**: `t = t.float(); return
(t > thr).float() if thr is not None else t`. DICE/HAUSDORFF95/NSD/ASSD (Z.177/228/291/339)
haben Default `threshold=None`. MONAI-Single-Channel-Pfad: `meandice.py:424-426` `for c in [1]:
x = y[b,0] == c`; `utils.py:211-214` `seg == label_idx` (1) — auch für Float-Input.

**Fehlerfall:** Raw-geladene int16-Labelmap {0,1,2,3}, Organ = Label 2 → dice/hd95 nur für
Label 1, `y_o == 0` → NaN → None ohne Meldung. Raw-geladene 0/255-PNG-Maske → `== 1` beidseitig
False → dice None, hd95 None. Die Spec (`...-intensity-normalization-design.md:75-78`) behauptet
„every non-zero label is foreground under Raw“ — falsch für den Default. Das Modul-Docstring
(`monai_metrics.py:10-16`) ist dagegen korrekt (empfiehlt `threshold=0.0`).

**Fix:** siehe 3.6 (Vordergrund-Policy einmal am Loader entscheiden). Kurzfristig:
Default `threshold=0.0` für alle MONAI-Specs und Spec-Text korrigieren.

### 3.4 Slice-SSIM wird auf 8 Bit quantisiert, PSNR nicht
**`src/iqaevaluator/metrics.py:290/294`** bauen `psnr`/`ssim` über `_pyiqa_factory` ohne kwargs.
pyiqa `ssim`: `archs/ssim_arch.py:103` `self.data_range = 255`; `archs/func_util.py:25-33`
1-Kanal-Zweig `x = x * 255` dann `x.round()`. pyiqa `psnr`: `data_range=1.0`, keine Rundung.
`normalization.py:4-5` nennt ×255 nur für brisque/niqe/piqe/vsi.

**Fehlerfall:** 12-Bit-MRI auf [0,1]: Rekonstruktion mit Fehler < 0.5/255 (≈ 8 Graustufen von
4095) pro Pixel rundet auf das identische uint8-Bild → **ssim == 1.0 exakt**, psnr aus demselben
Run ~55 dB endlich. psnr und ssim vergleichen zwei Quantisierungen.

**Fix:** Entweder SSIM explizit mit `data_range=1.0` und ohne Rundung instanziieren
(`pyiqa.create_metric('ssim', data_range=1.0)` bzw. eigene skimage/MONAI-2D-SSIM), oder das
Verhalten im Docstring und im Report dokumentieren (`ssim_quantized_8bit=True`). Danach 2.2 pinnen.

### 3.5 PSNR entspricht nicht der beanspruchten fastMRI-Konvention
**`src/iqaevaluator/image_loader.py:328`** Docstring: „fastMRI's convention: data_range comes
from the reference“. MinMax subtrahiert `lo` (`normalization.py:69`): Framework-PSNR =
10·log10((hi−lo)²/MSE); fastMRI = 10·log10(gt.max()²/MSE) — gleich nur wenn `gt.min() == 0`.
Zusätzlich clippt Z.76 Überschwinger des Inputs; fastMRI nicht.

**Fehlerfall:** Target-Range [500, 1500] (DICOM mit RescaleIntercept, Noise-Floor) →
Framework-PSNR liegt 20·log10(1500/1000) = **3.52 dB** unter dem fastMRI-Wert; bei lo = hi/2
sind es 6.0 dB. Die Spec (`design.md:17-18`) beansprucht dieselbe Konvention.

**Fix:** Entweder wirklich fastMRI nachbilden (kein Shift, `data_range = gt.max()`, kein Clip)
oder Docstring/Spec auf „min-max-normalisierter PSNR, nicht fastMRI-identisch“ ändern. So oder so
Test aus 2.2 pinnen.

### 3.6 Vier Binarisierer, zwei Semantiken; `label=` von keinem Spec erreichbar
- MONAI-Adapter: `t > threshold` (`monai_metrics.py:127, 361`).
- `VolumeFunctionMetric` (`volume_metrics.py:68`) und `BoundaryIoUMetric` (`boundary_iou.py:184`)
  über `volume.as_mask`: `x >= threshold` für Float, `x == label` (label = 1) für Integer,
  threshold dabei ignoriert.
- `label` wird von keinem registrierten Spec durchgereicht (`volume_metrics.py:79-141`,
  `boundary_iou.py:244-259`), obwohl `monai_metrics.py:13-15` das verspricht.
- `boundary_iou.py:204-207` („Tensoren sind immer float32 in [0,1]“) ist seit `Raw()` veraltet.

**Fehlerfall:** Raw-Labelmaps {0,1,2}: GT 50 Voxel Label 1 + 100 Voxel Label 2, Pred trifft
Label 1 exakt, Label 2 disjunkt. `mode='volume'` mit `dice_metric(threshold=0.0)`: dice =
100/300 = 0.33; `mode='slice'` mit V_PRED/V_GT/TP + `aggregate_volumes()`: nur Label 1 → dice =
1.0. Beide ohne Warnung.

**Fix (Root Cause):** Vordergrund-Policy **einmal** am Loader entscheiden, wo bekannt ist, dass
die Datei eine Maske ist — z. B. `Raw(label=None | int)` oder ein `MaskLoader`, der einen
Bool-Tensor liefert (alle Nicht-Null-Labels oder genau ein Label). Alle Adapter erhalten dieselbe
Binärmaske; `threshold`/`label`-Knöpfe an den einzelnen Metriken entfallen. Die gewählte Policy
wird wie `normalization` im Report vermerkt. Übergangsweise: ein gemeinsamer `binarize()`-Helper
für alle vier Pfade mit einer dokumentierten Vergleichsregel.

### 3.7 HD95/ASSD schreiben `inf` in die CSV; leeres GT → Dice `None`
**`monai_metrics.py:133-134`**: `per_sample = scores.mean(dim=1); return [None if
torch.isnan(s) else float(s)]` — filtert nur NaN.
- Einseitig leere Maske: MONAI `utils.py:286-292` füllt `inf` → `float('inf')` im Report;
  pandas-Mittelwerte werden inf. Nirgends `isinf`-Behandlung (evaluation_result.py, records.py).
- Leeres GT mit False-Positives: `compute_dice(ignore_empty=True)` → NaN → **None statt 0.0** —
  die schlechtesten Fälle fallen aus der Statistik.
- Multi-Channel-One-Hot mit fehlender Klasse: `mean` (nicht `nanmean`) → ganze Probe None.
- NSD liefert bei einseitig leer 0.0 (kein inf) — korrekt.

**Fix:** Explizite Policy pro Metrik definieren und testen: leeres GT + FP → dice 0.0
(`ignore_empty=False`), HD95/ASSD bei einseitig leer → `None` mit Skip-Grund (oder ein
dokumentierter Sentinel), `nanmean` über Kanäle. Tests aus 2.1.

### 3.8 3D-SSIM schrumpft das Fenster auf allen Achsen und mischt Definitionen
**`src/iqaevaluator/volumetric_iqa.py:62-76`**: `_window_for` liefert einen Skalar (größte
ungerade Größe ≤ 11, die in die dünnste Achse passt); MONAI `regression.py:325-331` broadcastet
auf alle drei Achsen, `kernel_sigma` bleibt 1.5. Das Fenster wird pro Aufruf aus `input.shape`
bestimmt, steht weder im Registry-Key (`volume_evaluator.py:78`) noch im Record.

**Fehlerfall:** 6-Slice-Stack → 5×5×5-Gauß bei ±1.33σ auch in-plane; 4 Slices → 3×3×3 (≈ Box).
8- und 20-Slice-Stacks in einem Run liefern zwei verschiedene SSIM-Schätzer in derselben Spalte.

**Fix:** per-Achsen-Sequenz `win_size=(w_d, 11, 11)` (MONAI akzeptiert `Sequence[int]`), sodass
nur die Through-Plane-Achse schrumpft; `ssim_win_size` pro Zeile in den Record schreiben und bei
Schrumpfung warnen. Test: zwei Volumina verschiedener Tiefe, gleicher In-Plane-Inhalt → gleiche
In-Plane-SSIM-Komponente.

### 3.9 Boundary-IoU (physisch): 1-mm-Floor löscht Through-Plane-Band
**`src/iqaevaluator/segmentation_metrics/boundary_iou.py:77-81`**: `band_width = max(1.0,
0.02·‖shape·spacing‖)` — Floor ist 1 **mm**, nicht 1 Voxel. Mit `sampling=spacing` (Z.134-140)
liegt der nächste Hintergrund-/Pad-Voxel `spacing[axis]` mm entfernt: auf Achsen mit
`spacing > band` gehört **keine** dazu senkrechte Fläche je zum Band. Docstring Z.121-122 ist
dann falsch. (Pre-existing; verifiziert mit scipy-EDT.)

**Fehlerfall:** Volume (10,20,20) bei Spacing (3,1,1) → band = 1.0 mm, EDT an einer D-Kappe =
3.0 > 1.0 → Kappe ausgeschlossen. Ein 1-Slice-Through-Plane-Shift scort **0.714** im physischen
Modus vs. **0.354** im Voxel-Modus. Große Volumina ((130,256,256)@(1.2,1,1) → 7.9 mm) sind sicher.

**Fix:** Floor auf `max(spacing)` (mindestens ein Voxel in jeder Richtung) statt 1.0 mm; Test
aus 2.3.

### 3.10 `aggregate_volumes` nullt Slices, die echte Voxel enthalten
**`src/iqaevaluator/evaluation_result.py:129-132`** füllt `v_pred/v_gt/tp` für `is_empty`-Slices
mit 0; Docstring Z.94-96 behauptet „blank on both sides“. Aber `empty_slice_mask`
(`image_loader.py:296-310`) ist eine ≤ 0.1 %-Heuristik (Lift-Test Z.308-309), und
`iqa_evaluator.py:92-109` überspringt einen Slice, wenn Input UND Target geflaggt sind. Baseline
ließ `v_gt` NaN und warnte; Tests `test_blank_prediction_slice_zeroes_only_the_prediction_counts`
/ `..._is_warned_about_by_name` wurden ersetzt.

**Fehlerfall:** Binäre Maske 256×256, Gesamt-Vordergrund ≥ 0.5 % (Percentile-Regime): jeder Slice
mit ≤ 65 Vordergrundvoxeln pro Seite gilt als leer. Pred mit 30 FP vs. GT mit 40-Voxel-
Läsionsspitze (disjunkt) → beide geflaggt → nie gescort → **30 FP und 40 FN verschwinden** aus
Volume-Dice/VS ohne Warnung. Der Volume-Modus zählt dieselben Voxel.

**Fix (Root Cause):** Leerheits-Filterung ist eine Eigenschaft des Runs, nicht der Daten:
`empty_slice_mask` liefert all-False unter `Raw()` (Masken), oder `IQAEvaluator(skip_empty=False)`
für Masken-Runs. Dann können der Extremes-Fallback und der bedingte Lift-Test entfallen.

### 3.11 `is_empty` bedeutet je Modus etwas anderes
**`src/iqaevaluator/volume_evaluator.py:108`** setzt `is_empty` allein aus dem Input;
Slice-Modus (`iqa_evaluator.py:92-94`) aus Input UND Target. Notebook `evaluation.ipynb:405`
filtert `df[~df.is_empty]` und verliert im Volume-Modus genau den Fall „leere Prediction,
belegtes GT“. **Fix:** eine Definition für beide Modi (folgt aus 3.10).

### 3.12 MinMax ohne 1e-8 ändert kleine Float-Ranges (Spec-Widerspruch)
**`src/iqaevaluator/normalization.py:69`**: alt `/(hi-lo+1e-8)`, neu `/np.float32(hi-lo)`.
Float32-Emulation: Spannen < 0.25 weichen systematisch ab (Range 0.05: 1000/1001 Werte anders),
sporadisch darüber (0.26). Integer-Ranges (uint8, HU) unverändert. Spec `design.md:246-248,
277-278` („bit-for-bit … below float32 resolution for any real range“) ist für Float-Karten in
[0, 0.05] falsch. **Fix:** Spec korrigieren + Test 2.4.

### 3.13 DreamSim-Resize bilinear statt upstream BICUBIC (Docstring falsch)
**`src/iqaevaluator/dreamsim_metric.py:139-153`**: Docstring behauptet Upstream-identische
Vorverarbeitung („scores stay comparable to published DreamSim numbers“); tatsächlich
`F.interpolate(mode='bilinear', antialias=True)` auf Float, upstream `PIL Resize((224,224),
BICUBIC)` auf uint8 (`dreamsim/model.py:298-306`). Tests prüfen nur Shapes/Identität/Ordnung.
**Fix:** `mode='bicubic', antialias=False` (näher an upstream) **und** Behauptung abschwächen
(publizierte Werte sind ohnehin auf Naturfotos). Test: `_prepare` vs. upstream `preprocess()` auf
einem Fixture bis Toleranz.

### 3.14 `dreamsim_spec(name=...)` kann Backbones vertauschen und Spalten überschreiben
**`dreamsim_metric.py:333`** `builtin=(metric_name == 'dreamsim')`; `metrics.py:164` ersetzt
gleichnamige Specs still; `records.py:57` `d.update(d.pop('extra'))` überschreibt Builtin-Felder;
`evaluation_result.py:73-75` listet Spalten doppelt. **Fix:** `register()` wirft bei Namens-
Kollision; `name` darf kein Builtin-Feldname sein; `builtin` nur bei Default-Konfiguration.

### 3.15 Exakter Float-Vergleich der Spacings (PLAUSIBLE, pre-existing)
**`volume_evaluator.py:57`** `!=` auf Tupeln; NIfTI-Zooms sind float32-pixdim
(`image_loader.py:116/121` → 1.2000000476837158), DICOM/SimpleITK float64 (1.2). Registry-Key
`metrics.py:210` ebenfalls. **Fix:** Spacing beim Dekodieren auf 1e-4 mm runden oder
`math.isclose(rel_tol=1e-5)` in einem gemeinsamen `spacing_equal()`.

### 3.16 `Raw()` scheitert an Big-Endian-NIfTI (PLAUSIBLE)
**`normalization.py:65-66`** `torch.from_numpy(np.ascontiguousarray(raw))` ohne dtype-Cast;
nibabel liefert bei fehlender Skalierung den On-Disk-dtype (`>i2`), den `torch.from_numpy`
ablehnt. **Fix:** `raw.astype(raw.dtype.newbyteorder('='), copy=False)` bzw. `np.ascontiguousarray(
raw, dtype=raw.dtype.newbyteorder('='))`. Test mit `arr.astype('>i2')`-Fixture.

---

## 4. Phase 4 — Aufräumen (Reuse / Vereinfachung / Effizienz)

| # | Ort | Befund | Fix |
|---|---|---|---|
| 4.1 | `monai_metrics.py:137-336` | Vier ~40-Zeilen-Copy-Paste-Builder (dice/hd95/nsd/assd); `.float()`-Cast aus 4f90160 fehlt im PQ-Adapter Z.361 (kosmetisch) | Ein `_monai_spec(name, compute_fn, *, direction, defaults, uses_spacing, description, threshold, **kw)`-Helper mit einer `build(spacing)`-Closure |
| 4.2 | `segmentation_metrics/volume.py:114-118` | Docstring „aggregate_volumes() wraps this function“ ist falsch; `aggregate_patient` hat keinen src-Aufrufer, summiert `skipna=True` (aggregate_volumes bewusst `skipna=False`), emittiert `avd_voxels` | Entweder `aggregate_volumes` auf `aggregate_patient` aufsetzen (Suffix strippen, is_empty nullen, `skipna`-Parameter) oder `aggregate_patient` entfernen |
| 4.3 | `image_loader.py:247, 297` | `raw_range` ist zeilengleich mit `MinMax().range_of`; `empty_slice_mask` hart-kodiert `[0.5, 99.5]` statt `Percentile().range_of` | `return MinMax().range_of(self.raw)`; `Percentile().range_of(raw)` |
| 4.4 | `image_loader.py:228-262` | Handgerollte Caches (`_loaded/_tensor/_intensity_range/_intensity_range_known`) leaken in Test-Fixtures (`tests/test_iqa_evaluator.py:28-35, 237-244` via `object.__new__`) | `functools.cached_property`; Tests bauen Loader über den echten Konstruktor |
| 4.5 | `image_loader.py:296` + `normalization.py:67-76, 113` | `empty_slice_mask` ungecacht, kopiert/partitioniert das Volumen; `Percentile.range_of` partitioniert dasselbe nochmals; `scale()` legt 4 volle Temporaries an; Loader hält `raw` UND `tensor` (Baseline nur Tensor) → 3-4 Extra-Durchläufe, ~2× Speicher bei 512×512×300 | Maske/Ranges einmal aus einer float32-Kopie ableiten und cachen; `scale()` in-place auf eigener Kopie; `raw` nur unter `Raw()` behalten |
| 4.6 | `dreamsim_metric.py:327, 147-153` | `channels='rgb'` materialisiert 3 identische Kanäle und schiebt 3× Bytes auf DEVICE; `_prepare` expandiert VOR `F.interpolate` | `channels='gray'`, 1-Kanal interpolieren, danach `.expand(-1,3,-1,-1)` |
| 4.7 | `dreamsim_metric.py:186-191` | `_as_scores`-Fallback für skalare Modellrückgabe bei N>1 — dreamsim 0.2.1 kann das nicht erzeugen (`model.py:112-119`); nur `FakeModel(scalar=True)` in Tests erreicht ihn | `scores = self._impl(inp, ref).reshape(-1)`; `assert len(scores) == n`; die zwei Fallback-Tests entfernen |
| 4.8 | `tests/test_image_loader.py:29, 437` | `IMG_SIZE = 96` dreifach kopiert (auch `test_main.py:19`, `test_data.py:11`); `_save_nifti` nach neun Inline-`nib.save` definiert | `from conftest import IMG_SIZE`; dtype-erhaltendes `save_nifti` in conftest |

---

## 5. Abnahmekriterien

- [ ] `pip install -e .` läuft auf frischem Clone (Python ≥ 3.10) durch.
- [ ] `pytest tests/` sammelt alle 20 Dateien ohne Import-Fehler; `iqaevaluator` CLI startet.
- [ ] Für jede Metrik existiert mindestens ein `pytest.approx`-Pin auf einen analytisch
      hergeleiteten Wert (Tabelle 2.1, Werte 2.2, EDT-Wert 2.3).
- [ ] Mindestens Dice, HD95, ASSD, PSNR, SSIM werden zusätzlich gegen eine unabhängige
      Referenzimplementierung (MedPy/scipy/skimage) geprüft.
- [ ] Mutations-Check: `setdefault('percentile', 95)` entfernen → mindestens ein Test rot;
      Dice → IoU tauschen → rot; SSIM `data_range` auf 255 setzen → rot.
- [ ] Docstrings/Spec enthalten keine Vergleichbarkeitsbehauptung mehr, die 3.4/3.5/3.8/3.13
      widerlegen, oder der Code wurde entsprechend angepasst.
- [ ] Cross-Check Volume-Dice == aggregierter Slice-Dice läuft auch auf Multi-Label-Raw-Maps.

---

## 6. Verifikationsnotizen (für Nachprüfung)

- Geschlossene Werte 2.1 wurden mit reinem Python nachgerechnet; MONAI-Verhalten aus
  `monai/metrics/{meandice,utils,hausdorff_distance,surface_distance,surface_dice,panoptic_quality}.py`
  (1.6.0) gelesen, nicht aus dem Gedächtnis.
- pyiqa-Verhalten (3.4) aus `pyiqa/archs/{ssim_arch,psnr_arch,func_util}.py` (0.1.15.post2).
- dreamsim-Verhalten (3.13, 4.7) aus `dreamsim/model.py` (0.2.1).
- 3.9 mit `scipy.ndimage.distance_transform_edt` in einer Scratch-Venv reproduziert
  (band_width((10,20,20),(3,1,1)) = 1.0; EDT an D-Kappe = 3.0; Shift-Scores 0.714 vs. 0.354).
- 3.12 per Float32-Emulation in reinem Python (Spannen 0.05/0.1/0.17/0.24/0.26 abweichend;
  0.2/0.3/1.0/255/4095 identisch).
- Baseline für „alt vs. neu“ ist Branch `volumetric_similarity` (ab113e1), **nicht** HEAD~1
  (HEAD~1 enthält bereits die gesamte Normalisierungsarbeit; der letzte Commit ist nur der Umzug).
