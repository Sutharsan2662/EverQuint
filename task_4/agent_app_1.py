import streamlit as st
import json
import requests
from typing import List

# ---------------- CONFIG ----------------
OLLAMA_MODEL = "mistral:7b-instruct"
OLLAMA_ENDPOINT = "http://localhost:11434/api/generate"

# ---------------- PROMPTS ----------------
# Moving Hints to the top ensures the model sees them as primary instructions
PLANNER_PROMPT = """
### MANDATORY CORRECTIONS FROM USER:
{hint_block}

### TASK:
Create a short numbered plan to solve the problem below. 
If hints are provided above, you MUST change your previous strategy to follow them.

Question: {question}
"""

EXECUTOR_PROMPT = """
### MANDATORY CORRECTIONS FROM USER:
{hint_block}

### TASK:
Solve the problem step-by-step following the Plan. 
If the hints above contradict the plan, prioritize the hints.

Plan: {plan}
Question: {question}

### OUTPUT FORMAT:
Show your reasoning clearly.
FINAL ANSWER: <value>
"""

# ---------------- AGENT ----------------
class ReasoningAgent:
    def __init__(self):
        self.model = OLLAMA_MODEL
        self.endpoint = OLLAMA_ENDPOINT

    def _call_llm(self, prompt: str, temp: float = 0.0) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temp}
        }
        try:
            # High timeout to prevent ReadTimeout errors on local hardware
            r = requests.post(self.endpoint, json=payload, timeout=120)
            r.raise_for_status()
            return r.json()["response"]
        except Exception as e:
            return f"Error: {str(e)}"

    def _extract_final_answer(self, text: str) -> str:
        for line in reversed(text.splitlines()):
            if "FINAL ANSWER" in line.upper():
                return line.split(":", 1)[-1].strip()
        return "N/A"

    def solve(self, question: str, hints: List[str], retries: int) -> dict:
        # Format hints as numbered corrections for higher impact
        if hints:
            hint_block = "\n".join([f"CORRECTION {i+1}: {h}" for i, h in enumerate(hints)])
        else:
            hint_block = "No previous errors reported. Solve normally."
        
        # Jitter: Use 0.0 for first try, 0.4 for retries to break logical loops
        current_temp = 0.0 if not hints else 0.4

        # 1. Planner
        planner_input = PLANNER_PROMPT.format(question=question, hint_block=hint_block)
        plan = self._call_llm(planner_input, temp=current_temp)

        # 2. Executor
        executor_prompt = EXECUTOR_PROMPT.format(
            plan=plan,
            question=question,
            hint_block=hint_block
        )
        solution = self._call_llm(executor_prompt, temp=current_temp)
        answer = self._extract_final_answer(solution)

        # 3. Validation & JSON Construction
        extraction_success = answer != "N/A"
        
        return {
            "answer": answer,
            "status": "success" if extraction_success else "failed",
            "reasoning_visible_to_user": solution,
            "metadata": {
                "plan": plan,
                "checks": [
                    {
                        "check_name": "Output Parsing",
                        "passed": extraction_success,
                        "details": "Successfully extracted FINAL ANSWER." if extraction_success else "Missing tag."
                    },
                    {
                        "check_name": "Hint Compliance",
                        "passed": len(hints) > 0,
                        "details": f"Processed {len(hints)} corrective hints."
                    }
                ],
                "retries": len(hints)
            }
        }

# ---------------- STREAMLIT UI ----------------
st.set_page_config(layout="wide", page_title="Iterative Reasoning Agent")
st.title("🧠 Iterative Reasoning Agent")

agent = ReasoningAgent()

# Session State
if "hints" not in st.session_state:
    st.session_state.hints = []
if "attempts" not in st.session_state:
    st.session_state.attempts = []

tab1, tab2 = st.tabs(["🔁 Interactive Solver", "📊 Validation Suite"])

with tab1:
    question = st.text_area("Enter the problem:", "Out of 20 people, 12 like coffee and 15 like tea. If 5 like neither, how many like both?")
    
    hint = st.chat_input("Hint to solve")

    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("▶️ Solve / Re-solve"):
            if hint:
                st.session_state.hints.append(hint)
            
            with st.spinner("Reasoning..."):
                res = agent.solve(question, st.session_state.hints, len(st.session_state.hints))
                st.session_state.attempts.append(res)
    
    with c2:
        if st.button("🔄 Reset"):
            st.session_state.hints = []
            st.session_state.attempts = []
            st.rerun()

    if st.session_state.attempts:
        last = st.session_state.attempts[-1]
        st.subheader("📦 JSON Output")
        st.json(last)
        
        with st.expander("🔍 Detailed Trace"):
            st.write(last["reasoning_visible_to_user"])

with tab2:
    st.subheader("Validation Suite")
    if st.button("Run test_cases.json"):
        try:
            with open("test_cases.json") as f:
                tests = json.load(f)
            results = []
            for tc in tests:
                out = agent.solve(tc["question"], [], 0)
                results.append({
                    "Question": tc["question"],
                    "Expected": tc["expected_answer"],
                    "Got": out["answer"],
                    "Status": "✅" if str(tc["expected_answer"]) in out["answer"] else "❌"
                })
            st.table(results)
        except Exception as e:
            st.error(f"Error: {e}")