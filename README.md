# 多模态多智能体医疗辅助系统

INFH5000 课程项目：一个可在 Windows 本地运行的中文医疗辅助研究原型。

> 仅用于教学与研究演示，不是医疗器械，不提供医学诊断，也不能替代专业医务人员。

## 已实现

- 病史、症状、连续生命体征和患者时间线；
- History、Triage、Imaging、Monitoring、Knowledge、Coordinator 六个智能体工作流；
- 风险等级、推荐科室、纵向变化和可解释依据；
- 上传 PNG/JPEG 胸片后，由本地 TorchXRayVision DenseNet121 模型生成研究模型分数；
- FastAPI + React + SQLite，统一使用一个本地端口。

当前除胸片模型外，其余 Agent 和知识库仍使用 Mock / 本地规则。胸片模型分数不是诊断，也不是校准后的临床概率。

## 启动

需要 Python 3.11 和 Node.js。双击：

```text
run.bat
```

也可以在 PowerShell 中运行：

```powershell
.\start.ps1
```

首次启动会安装依赖并下载模型，之后访问：

```text
http://127.0.0.1:8000
```

## API Key 放在哪里

API Key 只放在项目根目录的 `.env` 文件中，不要写进前端，也不要提交到 GitHub。首次运行 `start.ps1` 会自动从 `.env.example` 创建 `.env`。

OpenAI 示例：

```env
MEDAI_LLM_PROVIDER=openai
MEDAI_OPENAI_API_KEY=你的_API_Key
MEDAI_OPENAI_MODEL=gpt-4o-mini
```

DeepSeek 示例：

```env
MEDAI_LLM_PROVIDER=deepseek
MEDAI_DEEPSEEK_API_KEY=你的_API_Key
MEDAI_DEEPSEEK_MODEL=deepseek-chat
```

目前 Agent 仍为 Mock，因此填入 Key 后不会产生费用，也不会调用外部 API；后续接入 LLM Provider 时会直接读取这些配置。`.env` 已被 Git 忽略。

## 测试

```powershell
cd backend
..\.venv\Scripts\python.exe -m pytest -q
```

## 项目成员

**Group 25 · INFH5000 Project**

郝一帆、胡可、蓝嘉雪、孙博林、杨哲
