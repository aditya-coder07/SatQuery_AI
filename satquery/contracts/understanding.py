"""The query-understanding IR: what the system decided a query asked for.

Produced by `satquery/controller/understanding.py`, carried in the routing
trace, and handed to the tools as runtime parameters. A contract rather than
a controller detail because the frontend, the trace goldens and the NL
benchmark (`evaluation/nl_understanding_eval.py`) all read it.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from .plan import TaskID

Intent = Literal[
    "describe", "ask", "locate", "classify", "fuse",
    "change_describe", "change_ask", "change_map", "unclear",
]
RequestedOutput = Literal[
    "prose", "short_answer", "boxes", "class_map", "fused_map",
    "change_mask", "measurement", "clarification",
]
Quantity = Literal["count", "fraction", "area", "comparison"]


class QueryUnderstanding(BaseModel):
    """What the system understood. Serialised into the routing trace."""

    query: str
    resolved_query: str
    config: str
    intent: Intent
    task: TaskID
    inputs: list[str]
    requested_output: RequestedOutput
    # The thing the user is asking about - "airport", "buildings". Bare noun
    # phrase, spatial words removed.
    object_filter: str | None = None
    # The phrase handed to the grounder: the object plus its spatial
    # qualifier ("the plane at the top"), the way DIOR-RSVG expressions read.
    referring_expression: str | None = None
    spatial_scope: str | None = None
    temporal_relation: Literal["before_after"] | None = None
    quantity: Quantity | None = None
    # Land-cover classes named in the query, in the matrix's vocabulary.
    classes: list[str] | None = None
    # Which input the question is about when it names one ("in the second
    # image"): 0-based index into the manifest. None = the task's default.
    image_index: int | None = None
    follow_up: bool = False
    # Human-readable note on how a follow-up was resolved.
    resolution: str | None = None
    cues: list[str] = Field(default_factory=list)
