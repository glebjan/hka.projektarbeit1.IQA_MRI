# Wiki Rewrite — Design

Date: 2026-09-21. Status: approved in brainstorming (grill session, 2026-09-21).

## Goal

Replace the GitHub wiki of the framework (`framework.wiki`, remote
`glebjan/hka.projektarbeit1.IQA_MRI.wiki`) with a new, English, usage-oriented
documentation that enables researchers and students to operate the framework
without reading its source. The previous wiki content and its git history are
discarded.

## Sources of truth

1. **Code** of `framework/` at the commit being documented (branch
   `volumetric_similarity`, HEAD `4506b9c` or later). The code decides every
   behavioural statement.
2. **Project report** `framework.wiki/main.pdf` ("Framework für die
   Bildqualitätsbewertung anhand referenzbasierter und referenzloser Metriken",
   Gleb Janickij, HKA, SoSe 2026). The wiki must not contradict it. The wiki may
   and should be more detailed than the report.
3. When code and report disagree, the code wins in the wiki, and the
   disagreement is recorded in a discrepancy list for the author (not in the
   wiki). Where the report is internally inconsistent (e.g. "29 metrics" versus
   30 rows in Tabelle A.4), the wiki states the verifiable breakdown and avoids
   the disputed figure until the author resolves it.
4. The old wiki, `framework/docs/wiki/` and the outdated `framework/README.md`
   are ignored as sources.

## Audience

Researchers and students who evaluate their own generative models for medical
images, know Python and pandas, and have not read the framework code.

## Language and form rules

Derived from the local, author-only rule file `framework.wiki/AGENTS.md`
(written for the Typst report). `AGENTS.md` itself is never published.

- English, American spelling (matches identifiers such as `normalization`).
  Scientific register, impersonal, present tense, complete sentences. No first
  person, no direct address of the reader ("you", "your", "we", "our"), no
  colloquialisms, no filler words. Technical terms exact and identical to the
  report (Full-Reference, No-Reference, slice mode, volume mode, registry,
  record, normalization strategy).
- In prose, headings and table cells the characters `–`, `—`, `:` and `;` are
  not used, and there are no bullet or numbered lists. Enumerations are written
  as prose ("first ... second ...", "as well as") or as a Markdown table.
- Exempt are fenced code blocks, inline code, URLs, link syntax, HTML comments
  and Markdown table separator rows.
- Markdown tables are allowed. Each table is introduced by one sentence that
  states what it shows.
- Abbreviations are written out at their first occurrence on each page, in the
  form used by the report ("Peak Signal-to-Noise Ratio (PSNR)").
- `_Sidebar.md` uses no list markers. Links are grouped under bold group titles
  and separated by line breaks.
- No `_Footer.md`.

## Content

Theory stays minimal. Each metric receives one or two sentences on its
principle and interpretation plus a link to the original publication (DOI or
URL, taken from the report's bibliography where present). No formulas, no
numbered citations. `Home` names the project report once as the source for
background and design rationale (title, author, HKA, SoSe 2026, no link).

| File | Content |
|---|---|
| `Home.md` | Purpose, four metric families, scope limits (no GUI, no statistics, no metric training, no resampling), page guide |
| `Installation.md` | Clone, hatch and uv, the two environments, model weights, device, working directory |
| `Quick-Start.md` | First run by CLI and by Python |
| `Core-Concepts.md` | Input and target, FR and NR, registry per run, MetricSpec, record, data flow |
| `Loading-Images.md` | Formats and decoder details, pairing by file name, spacing, `ImageLoader` and `load_pair` |
| `Normalization.md` | Strategies, coupling to the target range, logged columns, custom strategy |
| `Selecting-Metrics.md` | Filling a registry, subsets, configurable builders, skip messages |
| `Metric-Catalog.md` | All registrable metrics with constant, family, reference, channels, direction, modes, backend, weights, interpretation, source |
| `Scoring-Modes.md` | Slice versus volume, units, comparability, aggregation |
| `Segmentation-Evaluation.md` | `Mask()`, label maps, classes, instance maps with `Raw()`, empty-mask policy, `aggregate_volumes` |
| `Results-and-Reports.md` | `to_frame`, columns, CSV, analysis with pandas |
| `Extending-the-Framework.md` | `register_metric`, hand-built `MetricSpec`, backend adapter, custom normalizer |
| `Worked-Examples.md` | End-to-end scenarios |
| `Testing-and-Calibration.md` | Test suite, calibration report, reference implementations |
| `Troubleshooting.md` | Real framework messages with cause and remedy |
| `_Sidebar.md` | Navigation |

## Examples

Every page except `Home` carries runnable examples. All Python and shell
examples are executed against a deterministic mock dataset, and every output
shown in the wiki comes from those runs. Examples assume the framework
repository root as working directory, the runtime environment `.venv`
activated, and `src` on `PYTHONPATH` (because `evaluate` lives in the reference
script `src/main.py`). A block that must not be executed (installation
commands, signatures) is preceded by the invisible marker
`<!-- example: norun -->`. A fenced `text` block that directly follows a code
block is its expected output and is checked line by line against the run.
The mock-data generator, the example runner and the style linter live in the
session scratchpad and are not published.

## Verification

- Example runner passes for every page.
- Style linter reports zero findings (forbidden characters, list markers,
  direct address, broken internal links and anchors).
- Every number and behavioural claim is checked against code and report.
  Discrepancies go into `discrepancies.md` in the scratchpad and are summarized
  to the author.

## Repository operations

- Wiki repository: abort the pending rebase, create an orphan branch that
  replaces `master` with exactly one commit containing the 15 pages and
  `_Sidebar.md`. `AGENTS.md`, `main.pdf` and `.DS_Store` are listed in
  `.git/info/exclude`. No push.
- Framework repository: this spec and its plan are committed on the current
  branch. No code changes. No push.

## Out of scope

Changes to framework code, README or docstrings (inconsistencies found there go
into the discrepancy list). Publishing or pushing anything.
