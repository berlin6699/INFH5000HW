# Project Requirements

This is the brief the implementation was built against, recorded verbatim where
the wording carries a constraint. It lives in the repository because the working
directory it was written in no longer exists.

**Title:** 面向多模态病史理解、智能分诊与连续健康监测的多智能体医疗辅助系统
*A Multimodal Multi-Agent Healthcare Assistant for Intelligent Triage and
Continuous Health Monitoring*

**Status at the time of writing:** Phases 0–3 complete (620 tests passing),
Phases 4–10 not started. See [Development phases](#development-phases).

---

## 1. Positioning

> 该系统定位为："Medical Decision-Support / Educational Research Prototype"
> 不能声称能够替代医生，也不能把系统输出描述成正式医学诊断。

A course-project prototype. It must be demonstrable, explainable, and finishable
in limited time. The instruction given at the outset was:

> 请首先不要过度设计 … 课程项目级别、可以演示、可以解释、可以在有限时间做完。

## 2. Core idea

Most medical AI demos are `symptoms → LLM → answer`. The contribution here is
fusing five evidence streams in one coordinator and computing the *delta* against
the patient's own baseline:

**Historical State + Current Symptoms + Imaging + Continuous Monitoring +
Retrieved Knowledge → multi-agent reasoning → longitudinal assessment**

The output must answer not only "what is the patient's state now" but
**"与历史记录相比，该异常发生了变化"** — what changed compared to this patient's
own history.

## 3. The six agents

| Agent | Responsibility |
|---|---|
| History | Chronic conditions, prior findings, timeline, `history_multiplier`, `data_gaps` |
| Triage | Symptom intake, warning signs, **preliminary** urgency only |
| Imaging | Chest X-ray findings |
| Monitoring | Continuous vitals: current value **plus** temporal trend |
| Knowledge | RAG over clinical guidelines; queries constructed *from* the structured findings |
| Coordinator | Multimodal fusion + longitudinal comparison + final risk, department, evidence, summary |

> 注意不要让 Triage Agent 单独产生最终医学结论。

`preliminary_urgency` from the Triage Agent is never the final answer: the
Coordinator owns `risk_level` and `urgency`, because a symptom set that looks
moderate in isolation can be HIGH once a falling SpO₂ trend and a prior
pulmonary finding are added.

The Monitoring Agent must never conclude from a single time point — its brief is
explicitly *current value + temporal trend*.

## 4. Structured data between agents

> 请优先使用结构化数据，而不是所有 Agent 都互相传自然语言。
> Agent 之间优先传：JSON / Pydantic Model 而不是：free-form text。

> 不要让其他 Agent 仅仅依赖 LLM 参数记忆进行医疗推理。

Implemented as a deterministic core with an optional LLM on top: **every
structured field** (`risk_level`, `recommended_department`, `historical_changes`,
`score_breakdown`) is computed by an explicit rule engine. An LLM, when
configured, may only rewrite prose and cannot alter a structured field.
Consequences: the demo runs with no API key and no network access
(`MEDAI_LLM_PROVIDER=null`); scoring is reproducible and auditable; Phase 10 gets
a free ablation study.

## 5. Vendor neutrality

> 不要把业务逻辑完全绑定到某一家模型 API。

`LLMProvider` is an abstraction with `null | openai | deepseek` implementations.
Orchestration is a LangGraph `StateGraph` whose nodes are plain Python functions
over Pydantic contracts — LangChain's LLM abstractions are deliberately not used.

## 6. Provenance: real vs. demo output

> 必须在 UI 和代码中明确区分：REAL MODEL OUTPUT 和 DEMO / MOCK OUTPUT
> 绝对不能伪装成真实医学模型结果。

> Demo 输出不应该伪造真实诊断。

`ImagingFinding.source_mode` is a **required** field: `mock_preset` (always
badged `DEMO / MOCK OUTPUT` in the UI) or `uploaded_report` (parsed from a
radiology report the user supplied). No real CXR convolutional model is
integrated. Badges are driven by `/api/system/info` so they cannot drift out of
sync with what the backend actually did. Every `AgentTrace` carries `llm_used`,
and the offline build asserts it is `False` — the trace must not imply inference
that did not happen.

## 7. Scope

**In scope (MVP):** respiratory presentations — fever, cough, chest pain,
dyspnoea, SpO₂ drop, tachycardia — and **chest X-ray** only.

**第一版不要做:**

1. 完整医院 HIS 系统
2. 所有疾病诊断
3. 所有 CT/MRI 医学影像
4. 自动开药
5. 自动调整药物剂量
6. 自动替代医生
7. 真正医院生产环境部署
8. 复杂 wearable hardware integration

## 8. Demo patient

A synthetic 55-year-old male: essential hypertension, a prior pulmonary
abnormality, presenting with fever / cough / dyspnoea, and a five-day monitoring
series showing SpO₂ 98 → 91 and heart rate 72 → 103. Expected outcome: risk
**HIGH**. All patient data is synthetic; no real protected health information is
stored or processed.

## 9. Frontend

React + Vite dashboard with six panels, one per agent, answering the question
**"AI 为什么做出这个判断？"** Every fired rule is rendered with the weight it
contributed, the threshold it crossed and the citation behind it. A persistent
disclaimer banner states that the output does not replace a qualified clinician.

## 10. Technology choices made up front

| Decision | Choice |
|---|---|
| LLM key | None available → `NullProvider`, everything offline |
| Orchestration | LangGraph `StateGraph` directly |
| Imaging | Mock first; report-text parsing deferred to Phase 8 |
| Backend | FastAPI, SQLAlchemy, SQLite |
| Retrieval | Pure-Python BM25, no vector DB |
| Frontend | React + Vite + TypeScript, types generated from the Pydantic schemas |
| Environment | conda env `medai`, Python 3.11 |

---

## Development phases

| # | Phase | Status |
|---|---|---|
| 0 | Scaffold & toolchain | ✅ |
| 1 | Data contracts, DB schema, synthetic patient, type sync | ✅ |
| 2 | History Agent + timeline | ✅ |
| 3 | Monitoring Agent + trend maths | ✅ |
| 4 | Triage Agent + warning signs | not started |
| 5 | Knowledge Agent + BM25 RAG | not started |
| 6 | Coordinator + longitudinal reasoning | not started |
| 7 | Web dashboard (6 panels, SSE progress) | not started |
| 8 | Imaging Agent (report-text parsing) | not started |
| 9 | End-to-end demo (`make demo`) | not started |
| 10 | Evaluation & ablations (± LLM, ± RAG, ± longitudinal) | not started |

### What exists and is tested

Phases 0–3 are complete. `backend/app/schemas/` holds the Pydantic contracts for
all six agents — including the ones not yet implemented, so the interfaces are
fixed ahead of the code. `backend/app/reasoning/` holds the deterministic core:
`thresholds.py` is the clinical threshold table, where every cutoff carries its
value, unit and source (a published citation or an explicit
`project-defined modelling choice`), plus the NEWS2 band table; `trends.py` is the
time-series maths; `history_rules.py` and `monitoring_rules.py` are the two named
rule engines that read them. Two agents run end to end against the seeded
database: History and Monitoring.

Design decisions worth carrying into Phases 4–10, all of them pinned by tests:

- **No wall clock.** `reference_date` is derived from the patient's own latest
  observation, shared by both agents. `datetime.now()` would age the demo patient
  every year the project sits unused, and would let a Phase 10 replay reason from
  data that had not yet happened. Records and samples dated after the anchor are
  excluded.
- **Errors are captured in the trace, never raised.** The workflow fans out to six
  agents; one failing must not take the others down. The error path rolls back so
  the caller does not inherit a poisoned transaction. Agents flush but do not
  commit — the caller owns the transaction.
- **`None` is not `0.0`.** A slope of `None` means "not enough data to know";
  `0.0` means "measured and flat". Collapsing them reports an unobserved trend as
  a stable one. Below three samples no slope is reported at all, because a line
  through two points has r² = 1.0 by construction.
- **A baseline never includes the current value**, or every delta shrinks toward
  zero by construction.
- **An empty window is a data gap, not an error.** A patient nobody measured gets
  `data_quality: insufficient` and seven named gaps. Reporting normal vitals for
  them would be the dangerous version of that outcome.
- **No monitoring rule assigns risk points.** `contribution` is 0.0 throughout;
  the Phase 6 Coordinator scorer owns the mapping onto risk bands. A second set of
  numbers would be a competing answer to "how much did this add?".
- **The NEWS2 total is partial** — four of seven parameters are recorded. The
  caveat is attached by a schema validator so a caller cannot omit it, and the
  rules escalate on the "score of 3 in any one parameter" trigger, which stays
  valid under partial scoring, rather than on the total, which does not.
- **A lexical safety guard** (`app/safety.py`) audits generated prose and fails
  the agent rather than publishing wording that asserts a diagnosis.

### Known limitations, recorded rather than hidden

- The NEWS2 band table is published against readings of one decimal place or
  coarser, so a finer observation needs a policy. The implemented one is uniform
  and **not direction-neutral**: SpO₂ 91.5% scores 3 like 91%, but heart rate
  130.5 scores 2 rather than 3. Making the second case safer means inventing
  half-step bounds nobody published, which would stop every rendered cutoff being
  checkable against the source chart.
- The History Agent never parses radiology prose; prior findings come from a coded
  `normalised` column. When Phase 8 parses an uploaded report, its output must land
  in that same column.
- Nothing in this repository has been validated against patient outcomes.

### The open question left for Phase 4

Whether the Triage Agent takes `SymptomInput` as a parameter or reads it from the
database. Only symptom persistence exists so far (`AnalysisRunRow.symptom_input`);
`backend/app/api/routes/` is still empty.
