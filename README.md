# 多模态多智能体医疗辅助系统

面向智能分诊与连续健康监测的课程项目原型。

系统综合患者病史、当前症状、胸部 X 光片、连续生命体征和模拟医学知识，生成风险等级、推荐就诊科室及可解释的判断依据。

> **重要说明**
> 本项目仅用于教学和研究演示，不是医疗器械，不提供正式医学诊断，也不能替代专业医务人员。

## 当前功能

- 患者基本信息及既往病史展示；
- 患者医疗时间线；
- 当前症状选择和文字输入；
- SpO₂、心率等五天监测趋势；
- Chest X-ray 上传预览；
- History、Triage、Imaging、Monitoring、Knowledge、Coordinator 六个智能体；
- 历史状态与当前状态的纵向比较；
- 风险等级、推荐科室和主要判断依据；
- 单端口本地网页。

## 当前版本说明

第一版完全离线运行：

- 不调用外部模型 API；
- 不下载医学影像模型权重；
- 影像结果来自预设 Mock 数据；
- Knowledge Agent 使用本地模拟知识；
- 页面会明确显示“模拟输出”，不会冒充真实模型结果；
- 当前患者和监测数据均为合成数据。

上传的胸片目前只用于页面预览，不会进行真实图像识别。

## Windows 启动方法

需要提前安装：

- Python 3.11；
- Node.js。

最简单的方法是双击：

```text
run.bat
```

也可以在 PowerShell 中运行：

```powershell
.\start.ps1
```

首次启动会自动创建 Python 环境、安装依赖并构建前端。启动完成后访问：

```text
http://127.0.0.1:8000
```

按 `Ctrl+C` 可以停止服务。

## 使用流程

1. 打开网页；
2. 查看合成患者的病史和监测数据；
3. 选择或取消当前症状；
4. 根据需要上传胸片进行预览；
5. 点击“运行多智能体分析”；
6. 查看风险等级、推荐科室、趋势变化和智能体推理依据。

## 技术栈

- 前端：React、TypeScript、Vite、Recharts；
- 后端：Python、FastAPI；
- 数据库：SQLite；
- 数据结构：Pydantic、SQLAlchemy；
- 当前模型模式：Deterministic Mock。

## 主要目录

```text
backend/       FastAPI、数据模型、智能体和测试
frontend/      React Dashboard
start.ps1      Windows 一键启动脚本
run.bat        Windows 双击启动入口
```

## 测试

```powershell
cd backend
..\.venv\Scripts\python.exe -m pytest -q
```

## 项目成员

**Group 25 · INFH5000 Project**

按姓名拼音排序：

- 郝一帆
- 胡可
- 蓝嘉雪
- 孙博林
- 杨哲
