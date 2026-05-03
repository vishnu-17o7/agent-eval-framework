"""Streamlit dashboard for the Agent Evaluation Framework."""

import sys
from pathlib import Path

# Add parent to path so we can import agent_eval
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
import pandas as pd
from agent_eval.tracing.store import get_score_history


st.set_page_config(
    page_title="Agent Eval Dashboard",
    page_icon="",
    layout="wide",
)

st.title("Agent Evaluation Dashboard")
st.markdown("Multi-layer evaluation scores across agent versions.")

history = get_score_history()

if not history:
    st.warning("No score history found. Run `agent-eval eval` first for both v1 and v2.")
    st.stop()

df = pd.DataFrame(history)

st.header("Score Overview")

cols = st.columns(4)
cols[0].metric("Overall Score", f"{df['overall_score'].iloc[-1]:.3f}")
cols[1].metric("Pre-Tool", f"{df['pre_tool_score'].iloc[-1]:.3f}")
cols[2].metric("At-Tool", f"{df['at_tool_score'].iloc[-1]:.3f}")
cols[3].metric("End-to-End", f"{df['end_to_end_score'].iloc[-1]:.3f}")

st.header("Score Comparison by Version")

chart_data = df.melt(
    id_vars=["agent_version", "timestamp"],
    value_vars=["overall_score", "pre_tool_score", "at_tool_score", "end_to_end_score"],
    var_name="Layer",
    value_name="Score",
)
chart_data["Layer"] = chart_data["Layer"].str.replace("_score", "")

st.bar_chart(
    chart_data,
    x="agent_version",
    y="Score",
    color="Layer",
)

st.header("Score History")

st.dataframe(
    df[["agent_version", "timestamp", "overall_score",
         "pre_tool_score", "at_tool_score", "end_to_end_score"]],
    use_container_width=True,
    hide_index=True,
)

st.header("Gate Status")

gate_threshold = 0.6
for _, row in df.iterrows():
    passed = row["overall_score"] >= gate_threshold
    icon = ":white_check_mark:" if passed else ":x:"
    st.write(f"{icon} **{row['agent_version']}** — Overall: {row['overall_score']:.3f} "
              f"(threshold: {gate_threshold:.2f}) — {'PASSED' if passed else 'FAILED'}")

st.caption(f"Last updated: {df['timestamp'].iloc[-1] if len(df) > 0 else 'N/A'}")
