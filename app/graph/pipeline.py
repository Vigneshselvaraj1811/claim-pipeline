"""
LangGraph Pipeline
==================
Defines the exact workflow:

  START
    ↓
  [Segregator Agent]          ← classifies all pages into 9 doc types
    ↓          ↓         ↓
  [ID Agent]  [Discharge]  [Itemized Bill]   ← parallel extraction
         ↘        ↓        ↙
           [Aggregator]
               ↓
             END

Uses LangGraph's StateGraph with a typed state dict.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
from typing import Any, Dict, TypedDict

from langgraph.graph import END, START, StateGraph

from app.agents.aggregator import aggregator_node
from app.agents.discharge_summary_agent import discharge_summary_agent_node
from app.agents.id_agent import id_agent_node
from app.agents.itemized_bill_agent import itemized_bill_agent_node
from app.agents.segregator import segregator_node
from app.models import PipelineState
from app.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# LangGraph requires a plain TypedDict as state, not a Pydantic model.
# We bridge between the two via wrapper functions.
# ---------------------------------------------------------------------------

class GraphState(TypedDict):
    pipeline_state: PipelineState


def _wrap_segregator(state: GraphState) -> GraphState:
    ps = segregator_node(state["pipeline_state"])
    return {"pipeline_state": ps}


def _wrap_id_agent(state: GraphState) -> GraphState:
    ps = id_agent_node(state["pipeline_state"])
    return {"pipeline_state": ps}


def _wrap_discharge_agent(state: GraphState) -> GraphState:
    ps = discharge_summary_agent_node(state["pipeline_state"])
    return {"pipeline_state": ps}


def _wrap_itemized_bill_agent(state: GraphState) -> GraphState:
    ps = itemized_bill_agent_node(state["pipeline_state"])
    return {"pipeline_state": ps}


def _wrap_aggregator(state: GraphState) -> GraphState:
    ps = aggregator_node(state["pipeline_state"])
    return {"pipeline_state": ps}


# ---------------------------------------------------------------------------
# Fan-out: run the 3 extraction agents IN PARALLEL after segregation
# ---------------------------------------------------------------------------

def _parallel_extraction(state: GraphState) -> GraphState:
    """
    Run ID, Discharge Summary, and Itemized Bill agents concurrently
    using a thread pool (safe because each operates on its own copy of state).
    """
    ps = state["pipeline_state"]

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        future_id = executor.submit(id_agent_node, ps.model_copy(deep=True))
        future_ds = executor.submit(discharge_summary_agent_node, ps.model_copy(deep=True))
        future_ib = executor.submit(itemized_bill_agent_node, ps.model_copy(deep=True))

        result_id = future_id.result()
        result_ds = future_ds.result()
        result_ib = future_ib.result()

    # Merge results back into the main state
    ps.identity_data = result_id.identity_data
    ps.discharge_summary_data = result_ds.discharge_summary_data
    ps.itemized_bill_data = result_ib.itemized_bill_data
    ps.errors.extend(result_id.errors)
    ps.errors.extend(result_ds.errors)
    ps.errors.extend(result_ib.errors)

    return {"pipeline_state": ps}


# ---------------------------------------------------------------------------
# Build the LangGraph
# ---------------------------------------------------------------------------

def build_pipeline() -> StateGraph:
    graph = StateGraph(GraphState)

    # Nodes
    graph.add_node("segregator", _wrap_segregator)
    graph.add_node("extraction_agents", _parallel_extraction)   # fan-out inside
    graph.add_node("aggregator", _wrap_aggregator)

    # Edges: START → segregator → extraction_agents → aggregator → END
    graph.add_edge(START, "segregator")
    graph.add_edge("segregator", "extraction_agents")
    graph.add_edge("extraction_agents", "aggregator")
    graph.add_edge("aggregator", END)

    return graph.compile()


# Compiled pipeline singleton
_pipeline = None


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        _pipeline = build_pipeline()
    return _pipeline


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def run_pipeline(claim_id: str, pdf_bytes: bytes) -> Dict[str, Any]:
    """
    Entry point called by the FastAPI route.
    Returns the final_result dict.
    """
    logger.info(f"[Pipeline] Running for claim_id={claim_id}")

    initial_state: GraphState = {
        "pipeline_state": PipelineState(
            claim_id=claim_id,
            pdf_bytes=pdf_bytes,
        )
    }

    pipeline = get_pipeline()
    final_graph_state = pipeline.invoke(initial_state)
    final_ps: PipelineState = final_graph_state["pipeline_state"]

    if final_ps.final_result is None:
        logger.error("[Pipeline] Aggregator did not produce a final result")
        return {"error": "Pipeline failed to produce a result", "errors": final_ps.errors}

    logger.info(f"[Pipeline] Completed for claim_id={claim_id}")
    return final_ps.final_result
