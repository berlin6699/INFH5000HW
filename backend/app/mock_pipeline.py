"""Small, fully-offline multi-agent workflow used by the first demo.

History and monitoring use the project's real deterministic implementations.
The remaining agents intentionally use transparent demo rules and preset data.
When a local imaging result has already been generated, the Imaging Agent reads
that structured result; no external API or diagnosis generation is involved.
"""

from __future__ import annotations

from datetime import date
from time import perf_counter
from typing import Callable
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.agents.history import run_history
from app.agents.monitoring import run_monitoring
from app.db.models import MedicalImageRow
from app.safety import DISCLAIMER
from app.schemas import (
    Acuity,
    AgentName,
    AgentStatus,
    AgentTrace,
    AnalysisRequest,
    AnalysisResult,
    ChangeType,
    ClinicalDimension,
    ClinicalDomain,
    Department,
    FinalAssessment,
    FiredRule,
    HistoricalChange,
    ImagingFinding,
    KnowledgeResult,
    RetrievedEvidence,
    RiskLevel,
    RunConfig,
    RunStatus,
    Significance,
    SymptomAssessment,
    SymptomName,
    Urgency,
    WarningSign,
    WarningSignSeverity,
    utcnow,
)


def _trace(agent: AgentName, output: object, started: float, *, rules: list[FiredRule] | None = None) -> AgentTrace:
    return AgentTrace(
        agent_name=agent,
        status=AgentStatus.OK,
        output=output.model_dump(mode="json") if hasattr(output, "model_dump") else {},
        fired_rules=rules or [],
        notes=["Offline deterministic demo; no external model or API was called."],
        llm_used=False,
        duration_ms=max(0, int((perf_counter() - started) * 1000)),
        started_at=utcnow(),
        finished_at=utcnow(),
    )


def run_triage(request: AnalysisRequest) -> tuple[SymptomAssessment, AgentTrace]:
    started = perf_counter()
    active = [s for s in request.symptoms if s.present]
    respiratory_names = {
        SymptomName.COUGH,
        SymptomName.DYSPNEA,
        SymptomName.CHEST_PAIN,
        SymptomName.CHEST_TIGHTNESS,
        SymptomName.SPUTUM,
        SymptomName.HEMOPTYSIS,
        SymptomName.WHEEZE,
    }
    infectious_names = {SymptomName.FEVER, SymptomName.CHILLS}
    warning_signs: list[WarningSign] = []
    rules: list[FiredRule] = []

    for symptom in active:
        major = symptom.name in {SymptomName.CONFUSION, SymptomName.CYANOSIS, SymptomName.HEMOPTYSIS}
        severe_breathing = symptom.name == SymptomName.DYSPNEA and (
            symptom.severity.value == "severe" or symptom.is_progressive
        )
        if major or severe_breathing:
            rule_id = f"DEMO-TRIAGE-{symptom.name.value.upper()}"
            description = f"Reported {symptom.name.value.replace('_', ' ')} requires prompt clinical attention."
            warning_signs.append(
                WarningSign(
                    sign=symptom.name.value,
                    severity=WarningSignSeverity.MAJOR,
                    rule_id=rule_id,
                    description=description,
                )
            )
            rules.append(
                FiredRule(
                    rule_id=rule_id,
                    category="triage",
                    description=description,
                    contribution=25.0 if severe_breathing else 30.0,
                    evidence=f"severity={symptom.severity.value}; progressive={symptom.is_progressive}",
                )
            )

    affected: list[ClinicalDomain] = []
    if any(s.name in respiratory_names for s in active):
        affected.append(ClinicalDomain.RESPIRATORY)
    if any(s.name in infectious_names for s in active):
        affected.append(ClinicalDomain.INFECTIOUS)
    if not affected:
        affected.append(ClinicalDomain.OTHER)

    missing = []
    if not any(s.name == SymptomName.DYSPNEA for s in active):
        missing.append("Whether breathing difficulty is present")
    if not any(s.name == SymptomName.CHEST_PAIN for s in active):
        missing.append("Whether chest pain is present")
    if any(s.onset_days is None for s in active):
        missing.append("Duration for one or more symptoms")

    urgency = Urgency.URGENT if warning_signs else (Urgency.SEMI_URGENT if len(active) >= 3 else Urgency.ROUTINE)
    assessment = SymptomAssessment(
        symptoms=active,
        warning_signs=warning_signs,
        dominant_domain=affected[0],
        affected_domains=affected,
        respiratory_symptom_count=sum(s.name in respiratory_names for s in active),
        has_progressive_symptoms=any(s.is_progressive for s in active),
        longest_duration_days=max((s.onset_days for s in active if s.onset_days is not None), default=None),
        missing_information=missing,
        followup_questions=[f"Please clarify: {item}." for item in missing],
        unparsed_free_text=request.free_text,
        preliminary_urgency=urgency,
        summary=f"Demo triage found {len(active)} active symptoms and {len(warning_signs)} warning sign(s).",
    )
    return assessment, _trace(AgentName.TRIAGE, assessment, started, rules=rules)


def run_imaging(session: Session, patient_id: str) -> tuple[ImagingFinding, AgentTrace]:
    started = perf_counter()
    row = session.scalars(
        select(MedicalImageRow)
        .where(MedicalImageRow.patient_id == patient_id)
        .order_by(MedicalImageRow.study_date.desc(), MedicalImageRow.created_at.desc())
    ).first()
    if row is None or not row.findings:
        finding = ImagingFinding(
            source_mode="mock_preset",
            provenance="No image result was available; empty demo result.",
            findings=[],
            abnormalities=[],
            summary="No demo imaging finding is available.",
        )
    else:
        finding = ImagingFinding.model_validate(row.findings)
    rules = []
    if finding.has_acute_abnormality or any(a.acuity == Acuity.ACUTE for a in finding.abnormalities):
        rules.append(
            FiredRule(
                rule_id="DEMO-IMG-ACUTE-01",
                category="imaging",
                description="The preset demo imaging result contains an acute abnormality.",
                contribution=15.0,
                evidence="Source is DEMO / MOCK OUTPUT; no image model was executed.",
            )
        )
    return finding, _trace(AgentName.IMAGING, finding, started, rules=rules)


def run_knowledge(triage: SymptomAssessment, monitoring, imaging: ImagingFinding) -> tuple[KnowledgeResult, AgentTrace]:
    started = perf_counter()
    queries: list[str] = []
    evidence: list[RetrievedEvidence] = []
    if triage.respiratory_symptom_count:
        queries.append("respiratory symptoms and escalation")
        evidence.append(
            RetrievedEvidence(
                chunk_id="demo-kb-respiratory-01",
                text="Worsening breathing symptoms should be assessed together with objective vital-sign trends.",
                source="Local demo knowledge base — educational placeholder",
                relevance_score=0.92,
                query=queries[-1],
                retrieved_for="triage: respiratory symptoms",
                topic="triage",
            )
        )
    if monitoring and monitoring.worst_metric:
        queries.append(f"{monitoring.worst_metric.value} trend escalation")
        evidence.append(
            RetrievedEvidence(
                chunk_id="demo-kb-monitoring-01",
                text="A sustained adverse trend can add context beyond the latest measurement alone.",
                source="Local demo knowledge base — educational placeholder",
                relevance_score=0.95,
                query=queries[-1],
                retrieved_for=f"monitoring: {monitoring.worst_metric.value}",
                topic="monitoring",
            )
        )
    if imaging.abnormalities:
        queries.append("chest x-ray abnormality with respiratory symptoms")
        evidence.append(
            RetrievedEvidence(
                chunk_id="demo-kb-imaging-01",
                text="Imaging observations should be interpreted with symptoms, monitoring and prior studies.",
                source="Local demo knowledge base — educational placeholder",
                relevance_score=0.90,
                query=queries[-1],
                retrieved_for="imaging: mock abnormality",
                topic="imaging",
            )
        )
    result = KnowledgeResult(
        queries=queries,
        evidence=evidence,
        retriever="bm25",
        corpus_size=3,
        top_k=3,
        summary=f"Retrieved {len(evidence)} clearly labelled demo knowledge snippets.",
    )
    return result, _trace(AgentName.KNOWLEDGE, result, started)


def run_coordinator(run_id: str, patient_id: str, history, triage, monitoring, imaging, knowledge) -> tuple[FinalAssessment, AgentTrace]:
    started = perf_counter()
    rules: list[FiredRule] = list(triage and [] or [])
    key_findings: list[str] = []

    if triage.warning_signs:
        rules.append(FiredRule(rule_id="DEMO-COORD-TRIAGE", category="triage", description="Triage warning signs are present.", contribution=25, evidence=triage.summary))
        key_findings.append("Progressive or severe respiratory symptoms were reported.")

    spo2_feature = next((t for t in monitoring.trends if t.metric.value == "spo2"), None) if monitoring else None
    if spo2_feature and spo2_feature.current <= 92:
        rules.append(FiredRule(rule_id="DEMO-COORD-SPO2", category="monitoring", description="Latest SpO2 is at or below the demo escalation threshold.", contribution=35, evidence=f"SpO2={spo2_feature.current:g}%", value=spo2_feature.current, threshold=92))
        key_findings.append(f"SpO2 declined from a baseline near {spo2_feature.baseline:g}% to {spo2_feature.current:g}%.")
    if monitoring and monitoring.rapid_deterioration:
        rules.append(FiredRule(rule_id="DEMO-COORD-TREND", category="monitoring", description="Multiple monitoring signals show sustained deterioration.", contribution=20, evidence=monitoring.summary))
        key_findings.append("Continuous monitoring shows an adverse multi-day trend.")
    acute = [a for a in imaging.abnormalities if a.acuity == Acuity.ACUTE]
    model_signals = imaging.abnormalities if imaging.is_real_model else []
    if acute:
        rules.append(FiredRule(rule_id="DEMO-COORD-IMAGE", category="imaging", description="Preset imaging includes an acute abnormality.", contribution=15, evidence="Mock imaging output only."))
        key_findings.append("Mock chest X-ray output contains acute abnormal findings.")
    elif model_signals:
        strongest = max(model_signals, key=lambda a: a.confidence)
        rules.append(FiredRule(rule_id="DEMO-COORD-IMAGE-MODEL", category="imaging", description="The local research model produced an above-threshold imaging signal.", contribution=10, evidence=f"{strongest.label.value}: score={strongest.confidence:.3f}; not a diagnosis."))
        key_findings.append(f"Local chest X-ray model signal: {strongest.label.value} ({strongest.confidence:.0%}).")
    if history.previous_findings:
        rules.append(FiredRule(rule_id="DEMO-COORD-HISTORY", category="history", description="Relevant prior pulmonary findings are present.", contribution=5, evidence=f"{len(history.previous_findings)} prior coded finding(s)."))
        key_findings.append("The patient has relevant previous pulmonary imaging findings.")

    score = min(100.0, sum(r.contribution for r in rules) * history.history_multiplier)
    risk = RiskLevel.HIGH if score >= 60 else RiskLevel.MEDIUM if score >= 30 else RiskLevel.LOW
    department = Department.EMERGENCY if risk == RiskLevel.HIGH else Department.RESPIRATORY if risk == RiskLevel.MEDIUM else Department.GENERAL_PRACTICE
    urgency = Urgency.IMMEDIATE if risk == RiskLevel.HIGH else Urgency.SEMI_URGENT if risk == RiskLevel.MEDIUM else Urgency.ROUTINE

    changes: list[HistoricalChange] = []
    if history.previous_findings and acute:
        prior = history.previous_findings[-1]
        current = max(acute, key=lambda a: a.severity_score)
        changes.append(HistoricalChange(
            dimension=ClinicalDimension.PULMONARY_IMAGING,
            change_type=ChangeType.RECURRENT if prior.resolved else ChangeType.WORSENING,
            significance=Significance.HIGH,
            prior_state=prior.description or prior.label,
            current_state=current.description or current.label.value,
            prior_date=prior.occurred_on,
            current_date=imaging.study_date,
            prior_severity=prior.severity_score,
            current_severity=current.severity_score,
            severity_delta=current.severity_score - prior.severity_score,
            interval_days=(imaging.study_date - prior.occurred_on).days if imaging.study_date else None,
            basis="Coded prior pulmonary finding compared with the preset current imaging finding.",
        ))
    if spo2_feature and spo2_feature.baseline is not None:
        changes.append(HistoricalChange(
            dimension=ClinicalDimension.OXYGENATION,
            change_type=ChangeType.WORSENING if spo2_feature.current < spo2_feature.baseline else ChangeType.STABLE,
            significance=Significance.HIGH if spo2_feature.current <= 92 else Significance.MODERATE,
            prior_state=f"Personal baseline SpO2 {spo2_feature.baseline:g}%",
            current_state=f"Current SpO2 {spo2_feature.current:g}%",
            prior_severity=0,
            current_severity=3 if spo2_feature.current <= 92 else 1,
            severity_delta=3 if spo2_feature.current <= 92 else 1,
            basis="Current value and multi-day trend compared with the patient's own baseline.",
        ))

    final = FinalAssessment(
        run_id=run_id,
        patient_id=patient_id,
        generated_at=utcnow(),
        risk_level=risk,
        risk_score=round(score, 1),
        score_breakdown=rules,
        history_multiplier=history.history_multiplier,
        recommended_department=department,
        urgency=urgency,
        care_advice="Seek timely assessment from a qualified healthcare professional. If symptoms feel severe or rapidly worsen, use local emergency services.",
        key_findings=key_findings,
        historical_changes=changes,
        longitudinal_summary="Current respiratory symptoms, imaging observations and monitoring deterioration are more concerning than the recorded baseline.",
        reasoning_summary="The demo coordinator combined history, symptoms, imaging observations, temporal monitoring trends and local placeholder knowledge. No diagnosis was produced.",
        evidence=knowledge.evidence,
        contributing_agents=[a.value for a in AgentName if a != AgentName.COORDINATOR],
        limitations=["Synthetic patient data.", "Imaging scores are research-model signals, not diagnoses or calibrated clinical probabilities." if imaging.is_real_model else "Imaging is preset DEMO / MOCK OUTPUT.", "Knowledge passages are educational placeholders.", "No LLM was called."],
        disclaimer=DISCLAIMER,
    )
    return final, _trace(AgentName.COORDINATOR, final, started, rules=rules)


ProgressCallback = Callable[[AgentName, str, str, int, int], None]


def run_mock_analysis(
    session: Session,
    request: AnalysisRequest,
    progress: ProgressCallback | None = None,
) -> AnalysisResult:
    """Run all six agents, optionally reporting their real execution stages."""
    started_at = utcnow()
    clock = perf_counter()
    run_id = f"RUN-{uuid4().hex[:10]}"

    def report(agent: AgentName, status: str, message: str, completed: int) -> None:
        if progress is not None:
            progress(agent, status, message, completed, 6)

    report(AgentName.HISTORY, "running", "正在整理既往病史并建立患者时间线", 0)
    history, history_trace = run_history(session, request.patient_id)
    report(AgentName.HISTORY, "completed", "病史结构化完成，已生成纵向患者基线", 1)

    report(AgentName.TRIAGE, "running", "正在提取当前症状并识别预警信号", 1)
    triage, triage_trace = run_triage(request)
    report(AgentName.TRIAGE, "completed", "当前症状和分诊信息已加入共享上下文", 2)

    report(AgentName.IMAGING, "running", "正在读取胸片模型或 Mock 的结构化结果", 2)
    imaging, imaging_trace = run_imaging(session, request.patient_id)
    report(AgentName.IMAGING, "completed", "影像征象及来源信息已加入共享上下文", 3)

    report(AgentName.MONITORING, "running", "正在分析血氧、心率等连续变化趋势", 3)
    monitoring, monitoring_trace = run_monitoring(session, request.patient_id)
    if history is None or monitoring is None:
        raise ValueError("History or monitoring data could not be loaded for this patient.")
    report(AgentName.MONITORING, "completed", "生命体征趋势和个人基线差异已加入共享上下文", 4)

    report(AgentName.KNOWLEDGE, "running", "正在根据前四个智能体的发现检索相关依据", 4)
    knowledge, knowledge_trace = run_knowledge(triage, monitoring, imaging)
    report(AgentName.KNOWLEDGE, "completed", "知识依据已返回，准备进行联合融合", 5)

    report(AgentName.COORDINATOR, "running", "正在联合病史、症状、影像、趋势和知识进行纵向推理", 5)
    final, coordinator_trace = run_coordinator(run_id, request.patient_id, history, triage, monitoring, imaging, knowledge)
    report(AgentName.COORDINATOR, "completed", "联合分析完成，正在生成最终可解释评估", 6)
    traces = [history_trace, triage_trace, imaging_trace, monitoring_trace, knowledge_trace, coordinator_trace]
    return AnalysisResult(
        run_id=run_id,
        patient_id=request.patient_id,
        status=RunStatus.COMPLETED,
        config=RunConfig(llm_provider="mock", llm_enabled=False, rag_enabled=False, imaging_mode=imaging.source_mode, longitudinal_enabled=True),
        history=history,
        triage=triage,
        imaging=imaging,
        monitoring=monitoring,
        knowledge=knowledge,
        assessment=final,
        traces=traces,
        agent_statuses={t.agent_name.value: t.status for t in traces},
        started_at=started_at,
        finished_at=utcnow(),
        duration_ms=max(0, int((perf_counter() - clock) * 1000)),
    )
