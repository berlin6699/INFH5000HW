# Multimodal Multi-Agent Healthcare Assistant

**Intelligent Triage & Continuous Health Monitoring**

> ### ⚠️ Educational Research Prototype
> This project is a course prototype for **medical decision-support research**.
> It is **not** a medical device, it does **not** produce diagnoses, and its
> output **never** replaces a qualified clinician. All patient data is
> **synthetic**. No real protected health information is stored or processed.

The brief this was built against, and what is and is not finished, is in
[REQUIREMENTS.md](REQUIREMENTS.md).

A multi-agent system that reasons over **historical patient state + current
symptoms + medical imaging + continuous physiological monitoring + retrieved
clinical knowledge**, and produces an explainable, *longitudinal* assessment —
not just "what is the patient's state now", but **"what changed compared to
this patient's own history"**.

---

## Core idea

Most medical AI demos are `symptoms → LLM → answer`. This system's contribution
is fusing five evidence streams in one coordinator and computing the *delta*
against the patient's own baseline:

```
        ┌── History Agent ────┐
        ├── Triage Agent ─────┤   parallel fan-out
Patient ├── Monitoring Agent ─┤
        └── Imaging Agent ────┘
                  │
                  ▼
          Knowledge Agent (RAG)      ← queries are constructed FROM the
                  │                     structured findings above
                  ▼
          Coordinator Agent
            ├─ LongitudinalComparator   baseline vs. current, per dimension
            ├─ RiskScorer               auditable weighted rules + overrides
            ├─ DepartmentRouter
            ├─ EvidenceAssembler
            └─ SummaryWriter            prose only, never structured fields
                  │
                  ▼
           FinalAssessment
```

The first offline MVP uses a deliberately small sequential Python workflow over
Pydantic contracts. LangGraph and external LLM providers are deferred until the
single-port mock demo is stable.

### Deterministic core, optional LLM

**Every structured field** (`risk_level`, `recommended_department`,
`historical_changes`, `score_breakdown`) is computed by an explicit rule engine.
An LLM, when configured, may only rewrite prose and **cannot** alter a
structured field. Consequences:

- The full demo runs with **no API key and no network access** (`MEDAI_LLM_PROVIDER=null`).
- Risk scoring is **reproducible and auditable** — every fired rule carries its
  own weight contribution and is rendered in the UI.
- Phase 10 gets a free ablation study (± LLM, ± RAG, ± longitudinal module).

---

## Quick start

### Windows — first offline MVP

Double-click `run.bat`, or run:

```powershell
.\start.ps1
```

The first launch installs local dependencies and builds the dashboard. Open
**http://127.0.0.1:8000**. Both the page and API are served from this one port.
This version is completely offline: the Triage, Imaging, Knowledge and
Coordinator agents use clearly labelled deterministic mock logic, while the
existing History and Monitoring agents run their tested rule engines. No model
weights are downloaded and no external API is called.

### Existing Makefile workflow

Requires **conda** (the system Python on macOS is 3.9.6, too old for this stack).

```bash
make setup     # create conda env "medai" (Python 3.11), install all deps, write .env
make dev       # backend :8000 + frontend :5173 (Ctrl-C stops both)
```

Then open **http://localhost:5173**.

| Command | Purpose |
|---|---|
| `make setup` | conda env + backend/frontend deps + `.env` |
| `make dev` | run backend and frontend together |
| `make backend` / `make frontend` | run either alone |
| `make seed` | recreate SQLite and load the synthetic demo patient |
| `make test` | pytest |
| `make types` | regenerate TS types from the pydantic schemas |
| `make check` | print resolved toolchain versions |
| `make demo` | one command: reset → seed → run analysis (completed in Phase 9) |
| `make clean` / `make distclean` | remove artifacts / also drop the DB |

Optional: set `MEDAI_LLM_PROVIDER=openai` or `deepseek` plus its key in `.env` to
enable prose synthesis. See `.env.example` — every value has an offline default.

---

## Scope

**In scope (MVP):** respiratory presentations (fever, cough, chest pain,
dyspnoea, SpO₂ drop, tachycardia) and **chest X-ray** only.

**Explicitly out of scope:** hospital HIS integration, whole-disease-coverage,
CT/MRI, medication prescribing or dose adjustment, production deployment, real
wearable hardware, and anything that replaces clinical judgement.

## Imaging output provenance

`ImagingFinding.source_mode` is a **required** field:

| Mode | Meaning |
|---|---|
| `mock_preset` | Predefined demo findings. **Always** badged `DEMO / MOCK OUTPUT` in the UI. |
| `uploaded_report` | Findings parsed from a radiology report supplied by the user. |

No real CXR convolutional model is integrated — that would require torch plus
~1 GB of weights. Badges are driven by `/api/system/info`, so they cannot drift
out of sync with what the backend actually did.

---

## Layout

```
backend/app/
  schemas/     Pydantic contracts — the single source of truth for all agent I/O
  db/          SQLAlchemy models, SQLite-specific column types, synthetic-data seed
  agents/      six agents + the LangGraph workflow
  reasoning/   everything deterministic: clinical threshold tables, the named rule
               registry, time-series trend maths, longitudinal comparator, risk
               scorer, department router
  llm/         LLMProvider abstraction (null | openai | deepseek)
  rag/         Retriever abstraction (pure-Python BM25) + guideline corpus
  safety.py    disclaimer + lexical guard against diagnostic wording
  tools/       export_types.py — regenerates the frontend contract from pydantic
frontend/src/
  components/  six dashboard panels + persistent disclaimer banner
  types/       generated.ts (auto-generated — do not edit)
```

One package for all deterministic computation rather than splitting "rules" from
"maths": the trend maths and the thresholds are read by the same rule engines,
and two packages with a fuzzy boundary between them is how a threshold ends up
defined twice.

## Development phases

| # | Phase | Status |
|---|---|---|
| 0 | Scaffold & toolchain | ✅ |
| 1 | Data contracts, DB schema, synthetic patient, type sync | ✅ |
| 2 | History Agent + timeline | ✅ |
| 3 | Monitoring Agent + trend maths | ✅ |
| 4 | Triage Agent + warning signs | Mock MVP ✓ |
| 5 | Knowledge Agent + BM25 RAG | Mock placeholder ✓ |
| 6 | Coordinator + longitudinal reasoning | Mock MVP ✓ |
| 7 | Web dashboard | Single-port MVP ✓ |
| 8 | Imaging Agent | Upload preview + labelled preset ✓ |
| 9 | End-to-end demo | Offline workflow ✓ |
| 10 | Evaluation & ablations | — |

Test suite: **620 passing** (`make test`).

### What Phase 2 delivered

The History Agent turns the raw record into a `PatientProfile` plus a
`TimelineEvent` list, and derives three things rather than asserting them:

- **`risk_factors`** — each one naming the rule that produced it (`R-HIS-AGE-02`)
  and carrying the threshold and citation behind its weight.
- **`history_multiplier`** — the product of those weights, clamped to
  `[1.0, 1.4]` so a long comorbidity list cannot dominate the score. The demo
  patient derives **×1.3698**.
- **`data_gaps`** — what the record does *not* contain. An assessment that
  silently ignored an absent prior study would be indistinguishable from one
  that had a prior study and found it unchanged, and those mean very different
  things clinically.

Two invariants worth knowing before reading the code:

**No natural-language inference.** Prior findings come from a coded `normalised`
column on the record that produced them. The agent never parses radiology prose;
when Phase 8 parses an uploaded report, its output lands in that same column.

**No wall clock.** `reference_date` is derived from the patient's own latest
observation. `datetime.now()` would age the demo patient every year the project
sits unused and start firing the stale-record rules for a reason that has nothing
to do with the patient. It also means records dated *after* an earlier anchor are
excluded — otherwise a Phase 10 replay would reason from data that had not yet
happened.

### What Phase 3 delivered

The Monitoring Agent's brief is *current value + temporal trend*, and the failure
mode it exists to avoid is concluding from a single time point. Three independent
components are therefore computed for every vital:

- **The published instrument.** A NEWS2 parameter score for the current reading
  (Royal College of Physicians, 2017). This is the only component with external
  validation behind it, and the cutoff it crossed is rendered next to the rule
  that fired.
- **An adverse temporal trend.** An ordinary least-squares slope per day, the
  length of the monotonic run ending at the latest sample, and the goodness of
  fit. Below three samples no slope is reported at all, because a line through
  two points has r² = 1.0 by construction and reads on a chart as a confident
  trend.
- **Deviation from the patient's own baseline** — the mean of the earliest two
  observations in the window, never including the current value, since a reading
  that is part of its own baseline cannot show deterioration. This is what makes
  a still-normal 95% alarming in a patient whose baseline was 98%.

`rapid_deterioration` needs all four of an adverse slope, a sustained run, enough
samples and an adequate fit. Each alone has a failure mode, and the reverse cases
are tested as hard as the demo: an unchanging 91% fires the absolute-threshold
rule and is *not* deterioration.

Three things are stated in the output rather than left to the reader:

**The NEWS2 total is partial.** Four of its seven parameters are recorded — blood
pressure and level of consciousness are not in `VitalSample`. The caveat is
attached by a schema validator, so a caller cannot omit it, and the rules escalate
on the "score of 3 in any one parameter" trigger, which stays valid under partial
scoring, rather than on the total, which does not.

**No monitoring rule assigns risk points.** `contribution` is 0.0 throughout,
because the Coordinator's scorer owns the mapping onto the monitoring band and a
second set of numbers here would be a competing answer to "how much did this
add?". What the rules carry is evidence.

**An empty window is a data gap, not an error.** A patient nobody measured gets
`data_quality: insufficient` and seven named gaps. Reporting normal vitals for
them would be the dangerous version of that outcome.

`anomaly_score` renormalises its component weights over whichever components
apply, then scales wearable-derived signals back down, so percent, bpm, degrees
and steps can be ranked against each other to pick `worst_metric`. For the demo
patient: SpO₂ 98→91 and HR 72→103 over five days, partial NEWS2 total **7**,
escalation trigger set, rapid deterioration in all six metrics, worst metric
**SpO₂** at 0.9676, `data_quality: complete`. Replayed at 09-06 — three days in —
the same patient shows no rapid deterioration and a different worst metric, which
is the point of the anchor.

**Known limitation, recorded for clinical review.** The band table is published
against readings of one decimal place or coarser, so a finer observation needs a
policy. The implemented one is uniform, and it is not direction-neutral: SpO₂
91.5% scores 3 like 91%, but heart rate 130.5 scores 2 rather than 3. Making the
second case safer means inventing half-step bounds nobody published, which would
stop every rendered cutoff being checkable against the source chart. Nothing here
has been validated against patient outcomes.
