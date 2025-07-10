# OutFinder: Industrial Sensor Anomaly Diagnostic Agent

OutFinder is a multi-stage diagnostic assistant designed for industrial environments. It uses structured sensor readings to detect anomalies, analyze root causes, and optionally search the web to recommend context-aware maintenance actions — all built using LangGraph, LLMs, and MLJAR AutoML.

## 🔧 Features

- **Sensor Input Extraction**: Parses raw user input for vibration, temperature, pressure, motor current, and flow rate.
- **Anomaly Detection**: Uses a pre-trained MLJAR AutoML model to classify readings as normal or anomaly.
- **Reasoning Engine**: An LLM provides explanations, identifies abnormal parameters, and recommends actions.
- **Web-Enhanced Suggestions**: Optionally performs live web search using Tavily and cites top sources in its response.
- **LangGraph Orchestration**: Robust control flow using LangGraph with conditional edges for dynamic tool invocation.

## 💡 Example Workflow

**Input:**  
`"Temp is very high, v=16.2, pressure around 2.5"`

**Extraction:**  
→ Extracts: temperature=88.9°C, vibration=16.2 mm/s, pressure=2.5 bar, etc.

**Prediction:**  
→ Model classifies as anomaly

**Reasoning:**  
→ Diagnoses overheating, recommends inspecting coolant system

**Optional Web Search:**  
→ Searches "recommended actions for overheating industrial motor"  
→ Cites results with links

## 📁 Project Structure
```
outfinder/
├── outfinder.py
├── AutoML_1/
├── app.py
├── requirements.txt
└── README.md
```


## 🚀 Getting Started

1. **Install dependencies**
```bash
pip install -r requirements.txt
```

Make sure you also have:

Python 3.9+

ollama running with deepseek-r1:8b model pulled

Google + Tavily API keys configured (if applicable)

2. Run the App
```
streamlit run app.py
```

## 🧠 Core Technologies
LangGraph: Conditional multi-step agent orchestration

LangChain: Prompt engineering, tools, memory (coming soon)

MLJAR AutoML: Sensor anomaly prediction

Gemini & Ollama (DeepSeek): LLMs for reasoning and tool selection

Tavily API: Web search tool for real-time recommendations