# Wiki-Follow-up: Kalibrierung (Sub-Projekt A — Segmentierung)

> Notiz für den Agenten, der das Wiki nach der Kalibrierungsarbeit aktualisiert.
> Wird während der Design-/Umsetzungsphase fortgeschrieben. Jeder Punkt ist eine
> Entscheidung, die im Wiki dokumentiert werden **muss** — nicht abgeleitet aus dem
> Code, sondern hier festgehalten, weil sie das bisherige Wiki widerlegt.

## Entscheidungen, die ins Wiki müssen

1. **`Mask()` ist der Masken-Weg, nicht `Raw()`.**
   Neue Normalisierungsstrategie `Mask(label=None, threshold=0.5)` in
   `iqaevaluator.normalization`. CLI: `--normalization mask`. Report-Spalte
   `normalization` = `mask` bzw. `mask_label_<k>`.
   - bool/int-Array: `label=None` → jeder Nicht-Null-Wert ist Vordergrund
     (0/255-PNG und 0/1-NIfTI werden damit gleich behandelt); `label=k` → nur
     `== k` ist Vordergrund.
   - float-Array (Probability-Map): `>= threshold`; Werte außerhalb [0,1] → Fehler.
   - Ausgabe: float32-Tensor mit exakt `{0.0, 1.0}`.
   - Alle bisherigen Wiki-Stellen „Masken mit `Raw()` laden" / „PNG-Masken mit
     `MinMax()` laden" sind damit überholt.

2. **`Raw()` bleibt nur als Escape-Hatch** (dtype erhalten, unskaliert). Für
   Segmentierungsmetriken ist `Raw()` **kein** gültiger Input mehr — die Adapter
   werfen, wenn der Tensor nicht binär ist, mit Hinweis auf `Mask()`.

3. **`load_pair` koppelt unter `Mask()` nicht auf die Target-Range.** Jede Seite
   wird eigenständig binarisiert. (Vorher: Input wurde auf die Target-Range
   skaliert → 0/1-Prediction gegen 0/255-GT ergab Dice 0.0 für eine perfekte
   Prediction.)

4. **Ein Label pro Run.** Multi-Class-Masken werden mit einem
   `Mask(label=k)` pro Klasse in getrennten Runs ausgewertet. Es gibt keinen
   One-Hot-/Multi-Channel-Pfad durch den Loader.

5. **`threshold`/`label`-Parameter der Spec-Builder sind entfernt:**
   `dice_metric()`, `hausdorff95_metric()`, `normalized_surface_dice_metric()`,
   `average_surface_distance_metric()`, `panoptic_quality_metric()`,
   `boundary_iou_metric()` und `VolumeFunctionMetric` nehmen keinen
   `threshold`/`label` mehr. Wiki-Beispiele mit `dice_metric(threshold=0.0)`
   müssen weg. Domänen-Parameter bleiben und laufen weiter über `**monai_kwargs`:
   NSD `class_thresholds`/`include_background`, HD95 `percentile`/`directed`,
   ASSD `symmetric`, PQ `match_threshold`/`metric_name`, Boundary-IoU
   `dilation_ratio`.
   Die Numpy-Funktionen (`boundary_iou(...)`, `vs`/`v_pred`/`v_gt`/`tp`,
   `as_mask`) behalten `label`/`threshold` als Standalone-Utilities; `as_mask`
   verwendet dieselbe Regel wie `Mask()` (Nicht-Null = Vordergrund bei
   `label=None`).
   `match_threshold` heißt korrekt `match_iou_threshold`. Ein `threshold=`/
   `label=`-Argument an irgendeinen Builder wirft `TypeError`.

6. **Leere Slices bei Masken-Runs:** unter `Mask()` gilt ein Slice als leer
   genau dann, wenn er null Vordergrund-Voxel hat (exakte Zählung, keine
   Heuristik). Übersprungen wird ein Slice nur, wenn Input **und** Target leer
   sind. Volume-Modus setzt `is_empty` nicht mehr (bleibt `False`).

7. **Leer-Masken-Policy (Werte im Report):**

   | Fall | Dice | VS | HD95 / ASSD | NSD | Boundary-IoU |
   |---|---|---|---|---|---|
   | GT leer, Pred hat Voxel | 0.0 | 0.0 | `None` | 0.0 | 0.0 |
   | Pred leer, GT hat Voxel | 0.0 | 0.0 | `None` | 0.0 | 0.0 |
   | beide leer | `None` | `None` | `None` | `None` | `None` |

   `None` = undefiniert. `inf`/`NaN` erscheinen nie in der CSV.

8. **Boundary-IoU (physischer Modus):** Band-Floor ist `max(spacing)` (mindestens
   ein Voxel in jede Richtung), nicht mehr 1 mm. Spacing-Tupel ist `(D, H, W)`.

9. **Panoptic Quality und `Raw()`:** Multi-Instanz-PQ braucht ganzzahlige
   Instanz-IDs — dafür bleibt `Raw()` der richtige Loader. Unter `Mask()` ist PQ
   ein Ein-Instanz-PQ (Vordergrund = eine Instanz).
   Der `panopticapi`-Vergleich kodiert den Hintergrund als Stuff-Segment
   (eigene Kategorie, `isthing=0`) und mittelt nur über Things — als VOID (0)
   ignoriert panopticapi FP-Instanzen auf dem Hintergrund und liefert 0.5
   statt 0.3 auf F4.

10. **Kalibrierung — drei Schichten** (neue Wiki-Seite „Calibration"):
    1. Handwert nach Paper-Definition auf kleinem Fixture, Herleitung im Test
       (`tests/calibration/cases.py`) — vom Menschen geprüft.
    2. Offizielle Implementierung derselben Definitionsvariante (MedPy für
       Dice/ASSD, `panopticapi` für PQ, `boundary-iou-api` für 2D-Boundary-IoU).
       `boundary-iou-api` ist nicht als Paket installierbar (leeres
       Top-Level-Paket, ungepinnte panopticapi-URL); `mask_to_boundary` ist in
       `tests/calibration/official.py` vendored (BSD-2, SHA 37d2558).
    3. MONAI-eigene Testfälle (1.6.0) durch Loader + Adapter — beweist, dass die
       Adapter `percentile`, `symmetric`, `class_thresholds`, `spacing` (D,H,W)
       unverfälscht durchreichen.
    Bericht: `docs/calibration/segmentation.md` (generiert via
    `hatch run calibration:report`), verlinken.

11. **Definitionsvarianten benennen** (Wiki muss das sagen, weil publizierte
    Zahlen davon abhängen):
    - HD95 = `max(P95(pred→gt), P95(gt→pred))` über Oberflächen-Voxel
      (Erosionskante) — Taha & Hanbury 2015 / MONAI. MedPy nimmt P95 über die
      vereinigten Distanzen; `surface-distance` (DeepMind) wichtet nach Fläche.
    - ASSD = Mittel über vereinigte gerichtete Oberflächen-Voxel-Distanzen
      (MONAI = MedPy). `surface-distance` flächengewichtet.
    - NSD = MONAI-voxelbasiert; die Autoren-Implementierung
      (`surface-distance`) ist flächengewichtet und liefert andere Werte.
    - Boundary-IoU 3D/physisch ist eine Framework-Erweiterung ohne offizielle
      Referenz (nur Handwert).

12. **Testbefehl und Umgebung:** Tests laufen nur noch im Calibration-Env:
    `hatch env create calibration && hatch run calibration:test`. Das
    Laufzeit-Venv `.venv` enthält kein pytest mehr; das sdist enthält kein
    `tests/`. Optionale Extras `[calibration]` (surface-distance==0.1,
    medpy==0.5.2, panopticapi@7bb4655).
    pyiqa deklariert pytest/ruff/pre-commit/yapf/tensorboard selbst als
    Laufzeit-Abhängigkeiten — sie bleiben in `.venv` transitiv installiert;
    entfernt sind nur unsere direkten Pins und `tests/` aus dem sdist.

## Noch offen (wird ergänzt)

- Sub-Projekte B (Intensitätsmetriken) und C (Loader/Normalisierung).
- `docs/calibration/segmentation.md` ist generiert und committed — im Wiki
  verlinken, nicht kopieren.
