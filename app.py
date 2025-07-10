import streamlit as st
from outfinder import runnable
from langgraph.graph import END
import re

st.set_page_config(page_title="OutFinder – Sensor Diagnostic", page_icon="🛠️")
st.title("🛠️ OutFinder – Industrial Sensor Diagnostic")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.chat_input("Enter your sensor report (e.g., temp high, v=6.2)...")

if user_input:
    st.chat_message("user").write(user_input)
    state = {"input": user_input}
    full_output = ""
    placeholder = st.chat_message("assistant").empty()

    # Phase-wise status display
    status_text = st.status("Running OutFinder pipeline...", expanded=True)

    with status_text:
        st.write("🔍 **Step 1: Extracting sensor readings**...")

    for step in runnable.stream(state):
        if END in step:
            result = step[END]
            diagnostic = result.get("diagnostic", "")
            full_output += diagnostic

            # Remove <think> block
            think_match = re.search(r"<think>(.*?)</think>", full_output, re.DOTALL)
            think_text = think_match.group(1).strip() if think_match else None
            cleaned = re.sub(r"<think>.*?</think>", "", full_output, flags=re.DOTALL).strip()

            placeholder.write(cleaned)

            if think_text:
                with st.expander("🔍 Show internal reasoning (<think>)", expanded=False):
                    st.markdown(think_text)

            st.session_state.chat_history.append({
                "user": user_input,
                "bot": cleaned,
                "think": think_text
            })

            with status_text:
                st.success("✅ OutFinder completed analysis.")
        else:
            current_step = list(step.keys())[0]
            if current_step == "predict":
                with status_text:
                    st.write("🧠 **Step 2: Running anomaly prediction model**...")
            elif current_step == "reason":
                with status_text:
                    st.write("🧾 **Step 3: Generating diagnostic reasoning**...")

            chunk = list(step.values())[0]
            partial = chunk.get("diagnostic", "")
            full_output += partial
            placeholder.write(full_output)
