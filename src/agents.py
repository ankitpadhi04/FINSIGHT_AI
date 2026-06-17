# src/agents.py

import pandas as pd
from typing import TypedDict
from langgraph.graph import StateGraph, END
from src.parser import parse_pdf
from src.categorizer import categorize_batch
from src.risk import calculate_risk
from src.rag import generate_suggestions


# ── Shared state schema ──────────────────────────────────────
class FinSightState(TypedDict):
    pdf_file:        object
    monthly_income:  float
    transactions:    list
    categorized_df:  object
    risk_report:     dict
    suggestions:     str
    error:           str


# ── Agent nodes ──────────────────────────────────────────────

def parse_node(state: FinSightState) -> FinSightState:
    """Node 1: Parse PDF and extract transactions."""
    try:
        df = parse_pdf(state["pdf_file"])
        if df.empty:
            return {**state, "error": "No transactions found in PDF."}
        state["transactions"] = df.to_dict(orient="records")
        return {**state, "error": ""}
    except Exception as e:
        return {**state, "error": f"Parse error: {str(e)}"}


def categorize_node(state: FinSightState) -> FinSightState:
    """Node 2: Categorize each transaction using hybrid DistilBERT + Groq."""
    try:
        df = pd.DataFrame(state["transactions"])
        descriptions = df["description"].tolist()
        results      = categorize_batch(descriptions)

        df["category"]   = [r["category"]   for r in results]
        df["confidence"] = [r["confidence"] for r in results]
        df["method"]     = [r["method"]     for r in results]

        return {**state, "categorized_df": df.to_dict(orient="records")}
    except Exception as e:
        return {**state, "error": f"Categorize error: {str(e)}"}


def risk_node(state: FinSightState) -> FinSightState:
    """Node 3: Calculate risk score from categorized transactions."""
    try:
        df     = pd.DataFrame(state["categorized_df"])
        report = calculate_risk(df, state["monthly_income"])
        return {**state, "risk_report": report}
    except Exception as e:
        return {**state, "error": f"Risk error: {str(e)}"}


def suggestion_node(state: FinSightState) -> FinSightState:
    """Node 4: Generate RAG-powered suggestions based on risk report."""
    try:
        suggestions = generate_suggestions(state["risk_report"])
        return {**state, "suggestions": suggestions}
    except Exception as e:
        return {**state, "error": f"Suggestion error: {str(e)}"}


# ── Conditional routing ──────────────────────────────────────

def should_continue(state: FinSightState) -> str:
    """Stop the graph if any node raised an error."""
    if state.get("error"):
        return "end"
    return "continue"


# ── Build the graph ──────────────────────────────────────────

def build_graph():
    graph = StateGraph(FinSightState)

    graph.add_node("parse",      parse_node)
    graph.add_node("categorize", categorize_node)
    graph.add_node("risk",       risk_node)
    graph.add_node("suggest",    suggestion_node)

    graph.set_entry_point("parse")

    graph.add_conditional_edges(
        "parse",
        should_continue,
        {"continue": "categorize", "end": END}
    )
    graph.add_conditional_edges(
        "categorize",
        should_continue,
        {"continue": "risk", "end": END}
    )
    graph.add_conditional_edges(
        "risk",
        should_continue,
        {"continue": "suggest", "end": END}
    )
    graph.add_edge("suggest", END)

    return graph.compile()


# ── Main entry point ─────────────────────────────────────────

def run_pipeline(pdf_file, monthly_income: float) -> FinSightState:
    """
    Run the full LangGraph pipeline.
    Returns the final state with all results.
    """
    graph = build_graph()

    initial_state: FinSightState = {
        "pdf_file":       pdf_file,
        "monthly_income": monthly_income,
        "transactions":   [],
        "categorized_df": [],
        "risk_report":    {},
        "suggestions":    "",
        "error":          ""
    }

    final_state = graph.invoke(initial_state)
    return final_state