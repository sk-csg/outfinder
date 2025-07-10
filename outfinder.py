#outfinder.py
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama
from typing import TypedDict, Optional
import pandas as pd
import re
import json
from supervised.automl import AutoML
# from langchain.tools.tavily_search import TavilySearchResults

# --- Load MLJAR AutoML model ---
automl_model = AutoML(results_path="AutoML_1")

# --- LangGraph State Type ---
class AgentState(TypedDict):
    input: str
    sensors: Optional[dict]
    prediction: Optional[str]
    diagnostic: Optional[str]

# --- LLMs ---
llm_extractor = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)
llm_reasoner = ChatOllama(model="deepseek-r1:8b", temperature=0.3, streaming=True)

# --- Extraction Prompt ---
EXTRACTION_PROMPT = """
You are a manufacturing sensor assistant. Extract the following values from user input:
- vibration
- temperature
- pressure
- motor_current
- flowrate
Handle any abbreviations (e.g., v, temp, pres, mc, flow) and return this JSON format:
{"vibration": float or null, "temperature": float or null, "pressure": float or null, "motor_current": float or null, "flowrate": float or null}
Return ONLY the dictionary.
"""

# --- Step 1: Extract Inputs ---
def extract_sensor_data(state: AgentState) -> dict:
    user_input = state["input"]
    messages = [
        SystemMessage(content=EXTRACTION_PROMPT),
        HumanMessage(content=user_input)
    ]
    response = llm_extractor.invoke(messages)
    raw = response.content.strip()
    try:
        match = re.search(r"{.*}", raw, re.DOTALL)
        if match:
            extracted = json.loads(match.group())
            # Fill missing values with mean (from stats below)
            means = {
                "vibration": 5.01,
                "temperature": 70.01,
                "pressure": 2.00,
                "motor_current": 12.01,
                "flowrate": 109.98
            }
            for k in means:
                if extracted.get(k) is None:
                    extracted[k] = means[k]
            return {"sensors": extracted}
    except:
        return {"sensors": {}}
    return {"sensors": {}}

def predict_anomaly(state: AgentState) -> dict:
    sensors = state["sensors"]
    if sensors:
        df = pd.DataFrame([{
            "vibration": sensors["vibration"],
            "temperature": sensors["temperature"],
            "pressure": sensors["pressure"],
            "motor_current": sensors["motor_current"],
            "flow_rate": sensors["flowrate"],
        }])
        raw_pred = automl_model.predict(df)[0]
        label = "anomaly" if raw_pred == 1 else "normal"
        return {"prediction": label}
    return {"error": "Missing sensor values"}

def run_reasoner(state: AgentState) -> dict:
    sensors = state["sensors"]
    prediction = state["prediction"]

    formatted_sensors = json.dumps(sensors, indent=2)
    REASONER_PROMPT = f"""
    You are a diagnostic expert for industrial machines.

    Your task is to:
    1. Analyze the sensor readings.
    2. Consider the anomaly prediction from the model.
    3. Identify which readings are outside safe operating thresholds.
    4. Provide a short explanation of what is likely going wrong.
    5. Recommend next diagnostic or maintenance actions.

    Sensor reading:
    {formatted_sensors}

    Safe operating stats:
    - Vibration (mm/s RMS) - Mean: 5.01, Std Dev: 0.55, Min: 2.88, Max: 15.67
    - Temperature (°C) - Mean: 70.01, Std Dev: 2.04, Min: 61.79, Max: 88.87
    - Pressure (bar) - Mean: 2.00, Std Dev: 0.10, Min: 1.08, Max: 2.47
    - Motor Current (A) - Mean: 12.01, Std Dev: 1.02, Min: 7.75, Max: 21.58
    - Flow Rate (RPM) - Mean: 109.98, Std Dev: 3.06, Min: 80.71, Max: 123.64

    Instructions:
    - If any sensor exceeds (or falls below) the threshold, clearly mention it.
    - Use terms like "high vibration", "overheating", or "low flowrate".
    - Suggest actions like "inspect bearings", "check coolant system", or "flush line".
    - If all values are within range and prediction is normal, confirm system is operating well.

    Return your answer in this format:

    <your internal reasoning about what's happening>

    Answer:
    <whether it is anomaly or not and if yes which parameter. that's all – no explanation here>

    Explanation:
    <why the anomaly occurred or confirmation of normal state>

    Recommended Actions:
    <what to inspect, adjust, or fix>
    """

    messages = [
        SystemMessage(content=REASONER_PROMPT),
        HumanMessage(content=f"Model Output: {prediction}")
    ]
    response = llm_reasoner.invoke(messages)

    if isinstance(response, str):
        return {"diagnostic": response.strip()}
    else:
        return {"diagnostic": response.content.strip()}


# --- LangGraph Setup ---
graph = StateGraph(AgentState)
graph.add_node("extract", extract_sensor_data)
graph.add_node("predict", predict_anomaly)
graph.add_node("reason", run_reasoner)

graph.set_entry_point("extract")
graph.add_edge("extract", "predict")
graph.add_edge("predict", "reason")
graph.add_edge("reason", END)

runnable = graph.compile()
