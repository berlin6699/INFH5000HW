import { useEffect, useMemo, useState } from 'react'
import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { ApiError, fetchDemo, runAnalysis, type AnalysisResult, type DemoData, type Symptom } from '@/api/client'

const symptomLabels: Record<string, string> = {
  fever: '发热', cough: '咳嗽', dyspnea: '呼吸困难',
  chest_tightness: '胸闷', chest_pain: '胸痛', fatigue: '乏力',
}
const agentLabels: Record<string, string> = {
  history: '病史', triage: '分诊', imaging: '影像', monitoring: '监测',
  knowledge: '知识', coordinator: '协调',
}

const translations: Record<string, string> = {
  'Essential hypertension': '原发性高血压',
  Amlodipine: '氨氯地平',
  'Progressive or severe respiratory symptoms were reported.': '患者报告进行性或严重呼吸系统症状。',
  'Continuous monitoring shows an adverse multi-day trend.': '连续监测显示多日不良变化趋势。',
  'Mock chest X-ray output contains acute abnormal findings.': '模拟胸片结果包含急性异常表现。',
  'The patient has relevant previous pulmonary imaging findings.': '患者既往存在相关肺部影像异常。',
  'Seek timely assessment from a qualified healthcare professional. If symptoms feel severe or rapidly worsen, use local emergency services.': '建议及时寻求专业医务人员评估；如症状严重或迅速加重，请联系当地急救服务。',
  'Current respiratory symptoms, mock imaging findings and monitoring deterioration are more concerning than the recorded baseline.': '与既往记录和个人基线相比，当前呼吸系统症状、模拟影像异常及监测指标恶化更值得关注。',
  'The demo coordinator combined history, symptoms, preset imaging, temporal monitoring trends and local placeholder knowledge. No diagnosis was produced.': '协调智能体综合了病史、症状、预设影像结果、连续监测趋势和本地模拟知识；系统未生成医学诊断。',
  'Worsening breathing symptoms should be assessed together with objective vital-sign trends.': '呼吸症状加重时，应结合客观生命体征趋势进行评估。',
  'A sustained adverse trend can add context beyond the latest measurement alone.': '持续的不良趋势能够提供单次最新测量之外的重要信息。',
  'Imaging observations should be interpreted with symptoms, monitoring and prior studies.': '影像观察结果应结合当前症状、监测数据和既往检查共同解读。',
}

function zh(text: string) {
  if (translations[text]) return translations[text]
  if (text.startsWith('SpO2 declined from a baseline near')) {
    const values = text.match(/[\d.]+/g) ?? []
    return `血氧饱和度从约 ${values[1] ?? '—'}% 的个人基线下降至 ${values[2] ?? '—'}%。`
  }
  return text
}

function timelineZh(text: string) {
  if (text.includes('Essential hypertension')) return '诊断原发性高血压'
  if (text.includes('Started Amlodipine')) return '开始服用氨氯地平 5 mg'
  if (text.includes('right lower lobe focal opacity')) return '胸片发现右下肺局灶性阴影'
  if (text.includes('Inflammatory markers elevated')) return '炎症指标升高'
  if (text.includes('prior opacity resolved')) return '复查胸片：既往阴影已消退'
  if (text.includes('controlled')) return '高血压常规复查：控制良好'
  if (text.includes('multifocal opacities')) return '当前胸片：多灶性阴影及纤维化（模拟）'
  if (text.includes('Continuous monitoring')) return '连续健康监测：5 天共 5 组数据'
  return text
}

function departmentZh(value: string) {
  return ({
    'Emergency Department': '急诊科',
    'Respiratory Medicine': '呼吸内科',
    Cardiology: '心内科',
    'Internal Medicine': '内科',
    'Infectious Disease': '感染科',
    'General Practice': '全科医学科',
  } as Record<string, string>)[value] ?? value
}

function urgencyZh(value: string) {
  return ({ IMMEDIATE: '立即', URGENT: '紧急', SEMI_URGENT: '较紧急', ROUTINE: '常规' } as Record<string, string>)[value] ?? value
}

function riskZh(value: string) {
  return ({ LOW: '低', MEDIUM: '中', HIGH: '高' } as Record<string, string>)[value] ?? value
}

function severityZh(value: string) {
  return ({ mild: '轻度', moderate: '中度', severe: '重度' } as Record<string, string>)[value] ?? value
}

function findingZh(text: string) {
  if (text.includes('Patchy increased opacities')) return '双侧下肺野可见斑片状密度增高影，右侧更明显'
  if (text.includes('No pleural effusion')) return '未见胸腔积液或气胸'
  if (text.includes('Cardiac silhouette')) return '心影大小在正常范围内'
  if (text.includes('Residual fibrotic streak')) return '右肺底残留纤维条索影，与既往相比无明显变化'
  if (text.includes('No bony abnormality')) return '未见明显骨性异常'
  return text
}

function App() {
  const [demo, setDemo] = useState<DemoData | null>(null)
  const [symptoms, setSymptoms] = useState<Symptom[]>([])
  const [freeText, setFreeText] = useState('')
  const [result, setResult] = useState<AnalysisResult | null>(null)
  const [loading, setLoading] = useState(true)
  const [analysing, setAnalysing] = useState(false)
  const [error, setError] = useState('')
  const [imageUrl, setImageUrl] = useState<string | null>(null)
  const [imageName, setImageName] = useState('Demo chest X-ray preset')

  useEffect(() => {
    fetchDemo().then((data) => {
      setDemo(data)
      setSymptoms(data.symptoms.symptoms.map((item) => ({ ...item, present: true })))
      setFreeText('两天前开始发热、干咳，呼吸越来越困难，走到洗手间时尤其明显，并伴有胸闷。')
    }).catch((err: unknown) => setError(formatError(err))).finally(() => setLoading(false))
  }, [])

  const chartData = useMemo(() => (demo?.monitoring ?? []).map((sample, index) => ({
    day: `第${index + 1}天`, SpO2: sample.spo2, HR: sample.heart_rate,
  })), [demo])

  async function analyse() {
    if (!demo) return
    setAnalysing(true); setError('')
    try { setResult(await runAnalysis(demo.patient.patient_id, symptoms, freeText)) }
    catch (err: unknown) { setError(formatError(err)) }
    finally { setAnalysing(false) }
  }

  function toggleSymptom(index: number) {
    setSymptoms((current) => current.map((item, itemIndex) =>
      itemIndex === index ? { ...item, present: !(item.present ?? true) } : item))
  }

  function selectImage(file?: File) {
    if (!file) return
    if (imageUrl) URL.revokeObjectURL(imageUrl)
    setImageUrl(URL.createObjectURL(file)); setImageName(file.name)
  }

  if (loading) return <div className="loading-screen"><div className="pulse" />正在准备离线医疗辅助工作台…</div>
  if (!demo) return <div className="loading-screen error-screen"><strong>无法加载演示数据</strong><span>{error}</span></div>

  const risk = result?.assessment.risk_level ?? '—'
  const riskClass = result ? `risk-${risk.toLowerCase()}` : 'risk-pending'

  return <div className="app-shell">
    <header className="topbar">
      <div className="brand"><div className="brand-mark">P</div><div><strong>PULSELINE</strong><span>多模态医疗辅助工作台</span></div></div>
      <div className="group-credit"><strong>Group 25 · INFH5000 Project</strong><span>郝一帆 · 胡可 · 蓝嘉雪 · 孙博林 · 杨哲</span></div>
      <div className="topbar-status"><span className="status-dot" /> 本地 · 离线 <span className="badge badge-amber">模拟智能体</span><span className="badge">无 API 调用</span></div>
    </header>
    <div className="safety-strip"><span>教学研究原型</span>本系统不是医疗器械，输出不构成医学诊断，也不能替代专业医务人员。</div>
    <main>
      <section className="hero-row"><div><p className="eyebrow">呼吸系统辅助 · 案例 001</p><h1>患者纵向健康评估</h1><p className="hero-copy">综合病史、当前症状、模拟影像和五天生命体征趋势，形成一份可解释的多智能体评估。</p></div>
        <button className="analyse-button" onClick={analyse} disabled={analysing}><span>{analysing ? '正在运行 6 个智能体…' : result ? '重新运行分析' : '运行多智能体分析'}</span><b>→</b></button>
      </section>
      {error && <div className="error-banner">{error}</div>}
      <section className="assessment-grid">
        <article className={`risk-card ${riskClass}`}><div className="card-kicker">当前评估</div><div className="risk-value">{result ? riskZh(risk) : '—'}</div><div className="risk-label">风险等级</div>{result ? <div className="score-line"><span>综合评分</span><strong>{result.assessment.risk_score}/100</strong></div> : <p className="muted">运行工作流后生成风险等级</p>}</article>
        <article className="card disposition-card"><div className="card-kicker">推荐就诊路径</div><h2>{result ? departmentZh(result.assessment.recommended_department) : '等待评估'}</h2><span className="urgency-pill">{result ? urgencyZh(result.assessment.urgency) : '尚未评估'}</span><p>{result ? zh(result.assessment.care_advice) : '协调智能体将综合五类证据进行判断。'}</p></article>
        <article className="card changes-card"><div className="card-kicker">发生了哪些变化？</div><h2>{result ? `${result.assessment.historical_changes.length} 项纵向变化` : '过去与现在'}</h2><p>{result ? zh(result.assessment.longitudinal_summary) : '这里将展示当前状态与患者既往记录及个人基线之间的比较。'}</p></article>
      </section>
      <section className="workspace-grid">
        <div className="left-column"><PatientCard demo={demo}/><SymptomCard symptoms={symptoms} freeText={freeText} setFreeText={setFreeText} toggle={toggleSymptom}/><TimelineCard timeline={demo.patient.timeline}/></div>
        <div className="right-column"><MonitoringCard data={chartData} result={result}/><ImagingCard demo={demo} result={result} imageUrl={imageUrl} imageName={imageName} selectImage={selectImage}/><ReasoningCard result={result}/></div>
      </section>
    </main>
    <footer><strong>Group 25 · INFH5000 Project</strong> · 郝一帆 · 胡可 · 蓝嘉雪 · 孙博林 · 杨哲<br/>全部病例数据均为合成数据 · 分析模式：确定性模拟 · 单端口本地运行</footer>
  </div>
}

function PatientCard({ demo }: { demo: DemoData }) {
  const p = demo.patient
  return <article className="card patient-card"><div className="section-heading"><div><div className="card-kicker">患者概览</div><h2>亚历克斯·摩根（合成患者）</h2></div><span className="patient-id">{p.patient_id}</span></div>
    <div className="demographics"><div><span>年龄</span><strong>{p.age}</strong></div><div><span>性别</span><strong>{p.sex === 'male' ? '男' : p.sex}</strong></div><div><span>记录</span><strong>{p.timeline.length}</strong></div></div>
    <DetailList label="既往疾病" values={p.chronic_conditions.map((item) => translations[item.name] ?? item.name)}/><DetailList label="长期用药" values={p.active_medications.map((item) => `${translations[item.name] ?? item.name} ${item.dose ?? ''}`)}/><DetailList label="过敏史" values={p.allergies.map((item) => `${item.allergen === 'Penicillin' ? '青霉素' : item.allergen} · ${item.severity === 'moderate' ? '中度' : item.severity}`)} tone="warn"/><DetailList label="风险因素" values={['55 岁', '高血压病史', '既往肺部影像异常']}/>
  </article>
}
function DetailList({ label, values, tone }: { label: string; values: string[]; tone?: 'warn' }) { return <div className="detail-row"><span>{label}</span><div>{values.length ? values.map((value) => <em className={tone} key={value}>{value}</em>) : <em>暂无记录</em>}</div></div> }

function SymptomCard({ symptoms, freeText, setFreeText, toggle }: { symptoms: Symptom[]; freeText: string; setFreeText: (value: string) => void; toggle: (index: number) => void }) {
  return <article className="card"><div className="section-heading"><div><div className="card-kicker">当前问诊信息</div><h2>症状</h2></div><span className="edit-note">可编辑</span></div>
    <div className="symptom-list">{symptoms.map((symptom, index) => <button type="button" className={`symptom-chip ${symptom.present ?? true ? 'active' : ''}`} key={`${symptom.name}-${index}`} onClick={() => toggle(index)}><span>{symptom.present ?? true ? '✓' : '+'}</span>{symptomLabels[symptom.name] ?? symptom.name}<small>{severityZh(symptom.severity)}</small></button>)}</div>
    <label className="field-label" htmlFor="complaint">患者自述</label><textarea id="complaint" value={freeText} onChange={(event) => setFreeText(event.target.value)} rows={3}/>
  </article>
}

function TimelineCard({ timeline }: { timeline: DemoData['patient']['timeline'] }) { return <article className="card"><div className="card-kicker">纵向医疗记录</div><h2>患者时间线</h2><div className="timeline">{timeline.map((event, index) => <div className={`timeline-event ${event.is_current ? 'current' : ''}`} key={`${event.occurred_on}-${index}`}><time>{event.occurred_on}</time><div><strong>{timelineZh(event.label)}</strong></div></div>)}</div></article> }

function MonitoringCard({ data, result }: { data: Array<Record<string, string | number | null>>; result: AnalysisResult | null }) {
  return <article className="card chart-card"><div className="section-heading"><div><div className="card-kicker">连续健康监测</div><h2>五天生命体征趋势</h2></div><div className="chart-legend"><span className="blue">血氧</span><span className="coral">心率</span></div></div>
    <div className="chart-wrap"><ResponsiveContainer width="100%" height="100%"><LineChart data={data} margin={{ top: 8, right: 18, left: -18, bottom: 0 }}><CartesianGrid strokeDasharray="3 3" stroke="#e7eceb" vertical={false}/><XAxis dataKey="day" axisLine={false} tickLine={false} tick={{ fill: '#71807d', fontSize: 11 }}/><YAxis domain={[60, 110]} axisLine={false} tickLine={false} tick={{ fill: '#71807d', fontSize: 11 }}/><Tooltip contentStyle={{ borderRadius: 10, border: '1px solid #dce5e2' }}/><Line type="monotone" dataKey="SpO2" stroke="#167a74" strokeWidth={3} dot={{ r: 4, fill: '#167a74', strokeWidth: 2, stroke: '#fff' }}/><Line type="monotone" dataKey="HR" stroke="#e46f51" strokeWidth={3} dot={{ r: 4, fill: '#e46f51', strokeWidth: 2, stroke: '#fff' }}/></LineChart></ResponsiveContainer></div>
    <div className="metric-strip"><div><span>血氧饱和度</span><strong>98 → 91<small>%</small></strong><em>下降 7 个百分点</em></div><div><span>心率</span><strong>72 → 103<small>次/分</small></strong><em>上升 31 次/分</em></div><div><span>总体趋势</span><strong>{result?.monitoring.rapid_deterioration ? '持续恶化' : '等待分析'}</strong><em>{result ? '五天监测窗口' : '请运行分析'}</em></div></div>
  </article>
}

function ImagingCard({ demo, result, imageUrl, imageName, selectImage }: { demo: DemoData; result: AnalysisResult | null; imageUrl: string | null; imageName: string; selectImage: (file?: File) => void }) {
  const imaging = result?.imaging ?? demo.imaging
  return <article className="card imaging-card"><div className="section-heading"><div><div className="card-kicker">医学影像</div><h2>胸部 X 光片</h2></div><span className="badge badge-amber">演示 / 模拟输出</span></div><div className="imaging-layout"><label className="upload-zone"><input type="file" accept="image/*" onChange={(event) => selectImage(event.target.files?.[0])}/>{imageUrl ? <img src={imageUrl} alt="用户选择的胸片预览"/> : <div className="xray-placeholder"><span>XR</span><p>选择胸片进行预览</p><small>模拟模式不会分析该图像</small></div>}<b>{imageName === 'Demo chest X-ray preset' ? '胸片模拟预设' : imageName}</b></label><div className="finding-list"><p className="provenance">系统没有运行真实影像模型。以下发现来自合成病例的预设模拟结果。</p>{imaging?.findings.slice(0, 4).map((finding) => <div className="finding" key={finding}><span>•</span>{findingZh(finding)}</div>)}</div></div></article>
}

function ReasoningCard({ result }: { result: AnalysisResult | null }) {
  return <article className="card reasoning-card"><div className="section-heading"><div><div className="card-kicker">可解释性</div><h2>智能体推理与证据</h2></div>{result && <span className="run-id">{result.run_id} · {result.duration_ms} 毫秒</span>}</div>{!result ? <div className="empty-state"><div className="agent-orbit">6</div><strong>六个智能体已准备就绪</strong><p>运行分析后，可查看每个智能体的贡献以及协调智能体的最终推理。</p></div> : <><div className="agent-run">{result.traces.map((trace, index) => <div className="agent-step" key={trace.agent_name}><span>{index + 1}</span><div><strong>{agentLabels[trace.agent_name] ?? trace.agent_name}智能体</strong><small>{trace.status === 'ok' ? '完成' : trace.status} · {trace.duration_ms} 毫秒 · 离线模拟</small></div></div>)}</div><div className="reasoning-summary"><h3>协调智能体的判断依据</h3><p>{zh(result.assessment.reasoning_summary)}</p>{result.assessment.key_findings.map((finding, index) => <div className="reason" key={finding}><b>{String(index + 1).padStart(2, '0')}</b><span>{zh(finding)}</span></div>)}</div><div className="evidence-grid">{result.assessment.evidence.map((item) => <div className="evidence" key={item.chunk_id}><span>模拟知识库</span><p>{zh(item.text)}</p><small>本地演示知识库 · 教学占位内容</small></div>)}</div></>}</article>
}

function formatError(error: unknown) { if (error instanceof ApiError) return `${error.message}: ${JSON.stringify(error.detail)}`; return error instanceof Error ? error.message : String(error) }
export default App
