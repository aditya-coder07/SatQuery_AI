"""Query understanding: natural language -> typed, validated, observable IR.

The router used to hand the user's sentence to the tools verbatim. That
worked for benchmark phrasings and failed for people: "Find the airport in
this image." reached the grounding adapter as the referring phrase, so the
model was asked to *"Locate the Find the airport in this image in the
image"*; "How much vegetation is present?" ran land cover with the default
classes `[built_up, water]`; and "did anything get built?" on a two-image
input was classified with no knowledge that there were two images, landed
on SINGLE_VQA, and was answered about one of them.

This module is the layer between the sentence and the plan::

    query (+ conversation history)
      -> resolve follow-ups            ("Where exactly?" -> "Where did the
                                          changes occur?")
      -> lexical cues                  (change / locate / count / ... - features
                                          for the classifier, never the verdict)
      -> parameters                    (referring phrase, land-cover classes,
                                          spatial scope, quantity, which image)
      -> QueryUnderstanding            (typed, validated, in the trace)

Everything here is deterministic and inspectable: a user can read the
`understanding` block of a trace and see exactly what the system thought
they asked. The *task* decision itself is still the classifier's, restricted
by config gating, in `router.py`; this module supplies its features and
fills in the parameters once the task is known.

Design rules:

* No lexicon decides a task on its own. Cues are features of the linear
  classifier (`intent.py`), so the bank of paraphrases and the cues are
  learned together and a single word cannot override the sentence.
* Extraction never invents. A phrase that cannot be found is `None`, and the
  tool falls back to its documented default with a warning, exactly as
  before.
* Follow-up resolution produces a *sentence* (`resolved_query`) that is
  routed like any other, so a resolved follow-up is subject to the same
  gating and the same validation as a fresh query. The rewrite is recorded.
"""

from __future__ import annotations

import re
from typing import Any

from satquery.contracts.plan import TaskID
from satquery.contracts.understanding import (  # noqa: F401 - re-exported
    Intent, QueryUnderstanding, Quantity, RequestedOutput,
)


INTENT_BY_TASK: dict[str, Intent] = {
    "SINGLE_VQA": "ask",
    "SINGLE_CAPTION": "describe",
    "SINGLE_GROUND": "locate",
    "SINGLE_LANDCOVER": "classify",
    "XMODAL_JOINT_EXTRACT": "fuse",
    "TEMPORAL_CHANGE_DESC": "change_describe",
    "TEMPORAL_CHANGE_VQA": "change_ask",
    "TEMPORAL_CHANGE_MAP": "change_map",
    "CLARIFY_OR_ABSTAIN": "unclear",
}
OUTPUT_BY_TASK: dict[str, RequestedOutput] = {
    "SINGLE_VQA": "short_answer",
    "SINGLE_CAPTION": "prose",
    "SINGLE_GROUND": "boxes",
    "SINGLE_LANDCOVER": "class_map",
    "XMODAL_JOINT_EXTRACT": "fused_map",
    "TEMPORAL_CHANGE_DESC": "prose",
    "TEMPORAL_CHANGE_VQA": "measurement",
    "TEMPORAL_CHANGE_MAP": "change_mask",
    "CLARIFY_OR_ABSTAIN": "clarification",
}
TEMPORAL_TASKS = {"TEMPORAL_CHANGE_DESC", "TEMPORAL_CHANGE_VQA", "TEMPORAL_CHANGE_MAP"}


# ---------------------------------------------------------------------------
# Lexical cues. Features, not rules: see the module docstring.
# ---------------------------------------------------------------------------

_CUE_PATTERNS: dict[str, str] = {
    # temporal / change
    "change": r"\b(chang(e|ed|es|ing)|differ(ent|ence|ences)|alter(ed|ation)|transform(ed|ation)|evolv(e|ed)|develop(ed|ment)|over time|before and after|then and now|now vs then|spot the difference|diff)\b",
    "new": r"\b(new|newly|added|appear(ed)?|built|construct(ed|ion)|erected|expand(ed)?|grew|grown|growth|sprawl|increase[ds]?|emerged|recently|since)\b",
    "gone": r"\b(demolish(ed)?|remov(ed|al)|destroy(ed)?|disappear(ed)?|lost|loss|cleared|cut down|shrink|shrank|shrunk|decreas(e|ed)|dried up|gone|vanish(ed)?|converted|replaced)\b",
    "two_images": r"\b(two|both|these|pair|first|second|earlier|later|older|newer|previous(ly)?|before|after|then|now|dates?|passes|acquisitions?)\b",
    "first_image": r"\b(first|earlier|older|initial|pre[- ]?(change|event)|before) (image|picture|photo|scene|one|date|acquisition)\b|\bin the first\b|\bbefore\b",
    "second_image": r"\b(second|later|newer|latest|last|recent|post[- ]?(change|event)|after) (image|picture|photo|scene|one|date|acquisition)\b|\bin the second\b|\bnow\b",
    # single-image intents
    "locate": r"\b(where|locat(e|ion|ed)|find|point (to|out|at)|show me (the|where|every|all)|highlight|mark|box(es)?|outline|circle|pinpoint|spot|detect|coordinates?|position|whereabouts|localis(e|ation)|refer(ring)? expression|comprehension)\b",
    "caption": r"\b(caption(ing)?|summar(y|ise|ize)|describe|description|overview|write|sum up|narrate|rundown|report|depict|analy[sz]e|tell me (everything|the important|about this))\b",
    "classify": r"\b(land[- ]?cover|land[- ]?use|classif(y|ication)|surface types?|terrain (class|type)|thematic|per[- ]class|segment(ation)?|categori[sz]e|breakdown|break (this|the scene|it) down|layer|covered with|ndvi|multi[- ]label)\b",
    "fuse": r"\b(sar|radar|optical|backscatter|both sensors|both images together|fus(e|ion|ed)|modalit(y|ies)|cross[- ]check|jointly?)\b",
    "mask": r"\b(mask|raster|geotiff|qgis|gis|layer|per[- ]pixel|binary|export|segment(ation)?|delineate|footprint|outline)\b",
    # quantity
    "count": r"\b(how many|number of|count|numbers?)\b",
    "fraction": r"\b(percent(age)?|%|proportion|fraction|share|how much|coverage|how green|how built|how dense)\b",
    "area": r"\b(hectares?|ha|km2|km²|sq(uare)? ?(km|kilomet(re|er)s?|met(re|er)s?)|area|extent|quantify|by how much)\b",
    "comparison": r"\b(more|less|fewer|bigger|larger|smaller|wider|narrower|greater|higher|lower|than|vs|versus|compare[ds]?|comparison|identical|same)\b",
    # form
    "question": r"^\s*(is|are|was|were|do|does|did|has|have|can|could|would|will|which|what|where|when|who|how|any|anything)\b|\?\s*$",
    "greeting": r"^\s*(hi|hello|hey|thanks|thank you|ok|okay|cheers|yo|sup)\b",
    "anaphor": r"\b(it|its|that|those|them|they|there|this one|that one|the same|the other|and the|what about|only|just|also|again|exactly|why)\b",
}
_CUES = {name: re.compile(pattern, re.IGNORECASE) for name, pattern in _CUE_PATTERNS.items()}
CUE_NAMES: tuple[str, ...] = tuple(_CUE_PATTERNS)


def cues(text: str) -> list[str]:
    """Names of the lexical cues present in `text`, in a fixed order."""
    return [name for name in CUE_NAMES if _CUES[name].search(text or "")]


def cue_vector(text: str) -> list[float]:
    """Binary cue indicators, one per CUE_NAMES entry - the classifier's
    hand-crafted feature block."""
    found = set(cues(text))
    return [1.0 if name in found else 0.0 for name in CUE_NAMES]


# ---------------------------------------------------------------------------
# Parameter extraction
# ---------------------------------------------------------------------------

# Matrix vocabulary for `classes` (SINGLE_LANDCOVER / XMODAL permitted_params).
CLASS_LEXICON: dict[str, str] = {
    "built_up": r"\b(built[- ]?up|urban|buildings?|built|houses|housing|settlements?|city|cities|town|residential|industrial|structures?|impervious|roofs?|rooftops?|concrete|developed)\b",
    "water": r"\b(water|waters|lakes?|rivers?|ponds?|reservoirs?|sea|ocean|coast(al|line)?|flood(ed|ing)?|wet(land)?s?|streams?|canals?|harbou?rs?)\b",
    "vegetation": r"\b(vegetation|vegetated|green(ery)?|forests?|trees?|woods?|woodland|crops?|farm(land|s)?|fields?|grass(land)?|plants?|canopy|orchards?|ndvi|agricultur(e|al))\b",
    "bare_soil": r"\b(bare|barren|soil|sand|sandy|desert|dunes?|rock(y)?|bareland|earth|dirt|unvegetated|non[- ]vegetated)\b",
}
_CLASS_RES = {k: re.compile(v, re.IGNORECASE) for k, v in CLASS_LEXICON.items()}

# Spatial scope: absolute positions in the frame. Compass words are included
# because users describe scenes that way; the grounder was trained on
# "top-right", "lower left" and "in the middle" phrasings.
_SPATIAL_RE = re.compile(
    r"\b((?:in |at |on |near |towards? |around )?(?:the )?"
    r"(?:(?:top[- ]?left|top[- ]?right|bottom[- ]?left|bottom[- ]?right|upper[- ]?left|upper[- ]?right|"
    r"lower[- ]?left|lower[- ]?right|north[- ]?east|north[- ]?west|south[- ]?east|south[- ]?west|"
    r"northern|southern|eastern|western|top|upper|bottom|lower|left|right|center|centre|middle|north|south|east|west)"
    r"(?:[- ]?(?:hand )?(?:side|half|corner|part|section|quadrant|edge|end|portion|region|area))?"
    r"(?: of the (?:image|scene|picture|frame|photo|runway|river|road))?))\b",
    re.IGNORECASE,
)
_SPATIAL_WORDS = re.compile(
    r"\b(top|upper|bottom|lower|left|right|center|centre|middle|north|south|east|west|northern|southern|eastern|western)\b",
    re.IGNORECASE,
)

_QUOTED = re.compile(r"[\"'‘’“”]([^\"'‘’“”]{2,80})[\"'‘’“”]")

# Scaffolding stripped from around a referring phrase. Order matters: the
# longest, most specific patterns first.
_LEAD = [
    r"^(?:ok(?:ay)?|so|please|hey|hi|now|also|and|then)[,\s]+",
    r"^(?:can|could|would|will|may) (?:you|u|i)(?: please| possibly)?[,\s]+",
    r"^(?:please|kindly|just|simply|quickly)\s+",
    r"^i(?:'d| would| want| need|'m looking) (?:like )?(?:to |you to )?(?:know |see |find |have )?(?:if |whether |where )?",
    r"^(?:perform|run|do) (?:a |an |the )?(?:referring expression comprehension|object localisation|object localization|grounding|detection)(?: for| of| on)?[:\s]+",
    r"^(?:object )?(?:localisation|localization|grounding|detection)[:\s]+",
    r"^(?:show|give|get|draw|put|place|render|display)(?: me)?(?: a| the| some)? (?:box(?:es)?|the location|location|coordinates|position|positions|bounding box(?:es)?)(?: of| on| around| for)?(?: every| all| each| any| the| a| an)?\s+",
    r"^(?:show me|show|find|locate|detect|identify|point (?:to|at|out)|mark|highlight|box|circle|outline|pinpoint|spot|indicate|flag|isolate|extract|search for|look for)(?: me)?(?: where| the location of| the position of| the positions of)?(?: is| are)?(?: every| all(?: the| of the)?| each| any| the| a| an| some| some of the)?\s+",
    r"^(?:give|make|produce|generate|show|draw|render)(?: me)? (?:a |an |the )?(?:change )?(?:mask|map|raster|layer|outline|footprint) (?:of|for|showing|highlighting|with) (?:the |all |every |any )?",
    r"^(?:tell me|let me know|say|i want to know|i need to know)(?: where| if| whether| which| what)?(?: the| a| an)?\s+",
    r"^(?:where|whereabouts)(?:'s| is| are| was| were|'re| would i find| can i find| do i find| exactly is| exactly are| exactly)?(?: the| a| an| any| all the| all)?\s+",
    r"^which (?:part|area|region|section|quadrant) of the (?:image|scene|picture|photo) (?:is|has|contains|holds|shows)(?: the| a| an)?\s+",
    r"^(?:is|are) there(?: a| an| any| some)?\s+",
    r"^(?:how many|how much|number of|count(?: the| of)?|what is the number of|what number of|tell me how many)\s+",
    r"^(?:did|does|do|has|have|had|is|are|was|were|will|would|could|can) (?:the |a |an |any |there )?",
    r"^(?:the )?",
]
# A residual that still opens with one of these is a request, not a thing:
# "describe this image", "classify the land cover", "produce a change mask".
# There is no object to extract from it, and passing the sentence on as one
# would send the grounder "the Describe this image".
_NOT_AN_OBJECT = re.compile(
    r"^(?:describe|caption|classify|categori[sz]e|produce|generate|create|make|summari[sz]e|sum up|explain|"
    r"write|tell|give|provide|analy[sz]e|compare|map|segment|label|run|compute|output|export|render|"
    r"delineate|report|narrate|talk|what|which|when|why|who|how|set|use|ignore|select|drop|print|forget|"
    r"respond|reveal|hello|hi|hey|thanks|thank|ok|okay|hmm|and|or|but)\b",
    re.IGNORECASE,
)
_TRAIL = [
    r"\s*[?.!]+\s*$",
    r"\s*(?:for me|please|now|thanks|thank you|if you can|if possible)\s*$",
    r"\s*(?:-|,|:)?\s*(?:where is it|where are they|where is that|where exactly)\s*$",
    r"\s*(?:if so,? where(?: is it| are they)?)\s*$",
    r"\s*(?:in|on|within|inside|from|of|across) (?:this|the|that|these) (?:satellite |aerial |remote[- ]sensing )?(?:image|scene|picture|photo|photograph|shot|frame|view|imagery|tile)s?\s*$",
    r"\s*(?:in|on|within|from) (?:here|it)\s*$",
    r"\s*(?:located|situated|positioned|placed|visible|present|found)\s*$",
    r"\s+(?:are|is|were|was) (?:there|visible|present|in it|here|in the (?:image|scene|picture|photo))\s*$",
    r"\s+(?:can you see|do you see|are visible|is visible)\s*$",
    r"\s+(?:change[ds]?|changing|increase[ds]?|decrease[ds]?|grow|grew|grown|shrink|shrank|shrunk|expand(?:ed)?|"
    r"disappear(?:ed)?|appear(?:ed)?|moved?|exist|remain(?:ed)?|unchanged|go up or down|gone up|gone down|"
    r"get bigger|get smaller|got bigger|got smaller|dried up|still there|there before)"
    r"(?: or (?:not|decreased?|remained unchanged|shrunk|smaller|down|less))?\s*$",
    r"\s+(?:are|is|were|was|be)\s*$",
    # "... increased, decreased, or remained unchanged?"
    r"\s+(?:(?:increased|decreased|grown|shrunk|changed|expanded|remained unchanged|stayed the same|remained the same|not)(?:,| or|,? or)?\s*)+$",
    r"\s*(?:,|-)?\s*if so\s*$",
    r"\s+(?:referred to|mentioned|named) (?:in|by) the (?:query|question|text|prompt)\s*$",
    r"\s*(?:-|,)?\s*(?:where is it|where are they)\s*$",
    r"\s*(?:for me|please)\s*$",
    r"\s*[?.!]+\s*$",
]
_LEAD_RES = [re.compile(p, re.IGNORECASE) for p in _LEAD]
_TRAIL_RES = [re.compile(p, re.IGNORECASE) for p in _TRAIL]
_IMAGE_REF_RE = re.compile(
    r"\s*\b(?:in|on|of|from) the (?:first|second|earlier|later|older|newer|latest|last|recent|initial|"
    r"pre[- ]?(?:change|event)|post[- ]?(?:change|event)|before|after|optical|sar|radar) "
    r"(?:image|picture|photo|scene|one|date|acquisition)\b",
    re.IGNORECASE,
)
_CHANGE_TAIL_RE = re.compile(
    r"\s+(?:that|which|who)? ?(?:have )?(?:changed|are new|were added|appeared|got built|were built|were demolished|disappeared|are different)\s*$",
    re.IGNORECASE,
)
_NEW_HEAD_RE = re.compile(r"^(?:new|newly built|added|recent|changed|demolished|removed|missing)\s+", re.IGNORECASE)


def _strip(text: str, patterns: list[re.Pattern]) -> str:
    changed = True
    while changed:
        changed = False
        for pat in patterns:
            new = pat.sub("", text, count=1).strip()
            if new != text:
                text, changed = new, True
    return text


def extract_spatial(text: str) -> str | None:
    """The absolute position the query names ("top-right", "lower half",
    "northern section"), or None."""
    m = _SPATIAL_RE.search(text or "")
    if not m:
        return None
    scope = m.group(1).strip()
    scope = re.sub(r"^(?:in|at|on|near|towards?|around) (?:the )?", "", scope, flags=re.IGNORECASE)
    scope = re.sub(r"^the ", "", scope, flags=re.IGNORECASE)
    return scope or None


def _residual(text: str) -> str:
    """The query with the intent scaffolding stripped: what is left is the
    thing being asked about, spatial and relational words included."""
    if not text:
        return ""
    quoted = _QUOTED.search(text)
    core = quoted.group(1) if quoted else text
    core = _IMAGE_REF_RE.sub("", core)
    core = _strip(core.strip(), _LEAD_RES)
    core = _strip(core, _TRAIL_RES)
    core = _CHANGE_TAIL_RE.sub("", core)
    return core.strip(" ,.-:;")


def extract_object(text: str) -> str | None:
    """The noun phrase the query is about, with the intent scaffolding and
    the spatial qualifier removed. None when nothing survives - a bare
    "where?" or "describe this" has no object."""
    core = _residual(text)
    # Split off a spatial qualifier: "airplane on the right side of the runway".
    m = _SPATIAL_RE.search(core)
    if m and m.start() > 0:
        head = core[: m.start()].strip().rstrip(",")
        head = re.sub(r"\s+(?:is|are|was|were)?\s*(?:in|at|on|near|towards?|around|located|situated|that is|which is|closest to|next to)$", "", head, flags=re.IGNORECASE)
        head = re.sub(r"\s+(?:is|are|was|were)$", "", head, flags=re.IGNORECASE)
        core = head
    core = _NEW_HEAD_RE.sub("", core)
    core = re.sub(r"^(?:the|a|an|every|all|each|any|some|those|these|that|this)\s+", "", core, flags=re.IGNORECASE)
    core = core.strip(" ,.-:;")
    if not core or len(core.split()) > 8 or _NOT_AN_OBJECT.match(core):
        return None
    if re.fullmatch(r"(?:it|them|they|that|this|those|these|one|ones|thing|things|image|scene|picture|here|there|what|everything|anything)", core, re.IGNORECASE):
        return None
    return core


def referring_expression(text: str, obj: str | None, spatial: str | None) -> str | None:
    """The phrase handed to the grounder: the object with its spatial and
    relational words kept as the user wrote them ("the tennis court on the
    left of the tennis court at the bottom"), which is how DIOR-RSVG
    expressions read. Only the article is normalised. Measured on
    2026-09-20 (evaluation/grounding_phrase_format.py): rebuilding the
    phrase from the bare object and a normalised spatial term lost the
    relational tail and changed prepositions, so the residual is used as is."""
    if not obj:
        return None
    phrase = _residual(text)
    phrase = _NEW_HEAD_RE.sub("", phrase)
    phrase = re.sub(r"^(?:the|a|an|every|all|each|any|some|those|these|that|this)\s+", "", phrase, flags=re.IGNORECASE)
    phrase = phrase.strip(" ,.-:;")
    if not phrase or len(phrase.split()) > 24 or _NOT_AN_OBJECT.match(phrase):
        phrase = obj
    return f"the {phrase}"


def extract_classes(text: str) -> list[str] | None:
    """Land-cover classes the query names, in the matrix's vocabulary and
    order; None when it names none (the tool keeps its default)."""
    found = [name for name in CLASS_LEXICON if _CLASS_RES[name].search(text or "")]
    return found or None


def extract_quantity(text: str) -> Quantity | None:
    t = (text or "").lower()
    if re.search(r"\b(than|vs|versus|which (?:one|image|date) has (?:more|less|fewer))\b", t):
        return "comparison"
    if re.search(r"\b(how many|how much) (?:hectares?|km2|km²|sq(?:uare)? ?(?:km|kilomet(?:re|er)s?|met(?:re|er)s?)|acres?)\b", t):
        return "area"
    if re.search(r"\b(how many|number of|count(?: the| of)?|numbers? of)\b", t):
        return "count"
    if re.search(r"\b(hectares?|km2|km²|sq(?:uare)? ?(?:km|kilomet(?:re|er)s?|met(?:re|er)s?)|quantify|by how much|how much (?:did|has|have|was|were) .*(?:grow|grew|shrink|shrank|chang|expand|increase|decrease|clear|lost|gain)|how much (?:land|area|ground) )\b", t):
        return "area"
    if re.search(r"\b(percent(?:age)?|%|proportion|fraction|share|coverage|how much(?: of)?|what part of|how green|how built|how dense(?:ly)?)\b", t):
        return "fraction"
    if re.search(r"\b(more|less|fewer|bigger|larger|smaller|wider|narrower|greater|higher|lower|taller|identical|the same)\b", t):
        return "comparison"
    return None


def extract_image_index(text: str, config: str) -> int | None:
    """0 or 1 when the query is about one named input of a pair; None when
    it names none or both ("more built up than the first")."""
    if config not in ("BITEMPORAL_PAIR", "CROSSMODAL_PAIR"):
        return None
    t = text or ""
    first = bool(re.search(r"\b(first|earlier|older|initial|pre[- ]?(?:change|event)|before) (?:image|picture|photo|scene|one|date|acquisition)\b|\bthe first\b|\bthe earlier\b", t, re.IGNORECASE))
    second = bool(re.search(r"\b(second|later|newer|latest|last|recent|post[- ]?(?:change|event)|after) (?:image|picture|photo|scene|one|date|acquisition)\b|\bthe second\b|\bthe later\b", t, re.IGNORECASE))
    if config == "CROSSMODAL_PAIR":
        first = first or bool(re.search(r"\b(?:in |on |of )?the optical (?:image|scene|picture|photo|one)\b", t, re.IGNORECASE))
        second = second or bool(re.search(r"\b(?:in |on |of )?the (?:sar|radar) (?:image|scene|picture|photo|one)\b", t, re.IGNORECASE))
    if first == second:
        return None
    return 0 if first else 1


# A comparison request that names no subject: "compare the scenes", "show me
# the difference". On one image there is nothing to compare; on a dated pair
# it can only mean "describe the change". The classifier cannot learn that
# split, because the same words would need two labels and the pair token alone
# would carry the difference (it then sent "hmm" on a pair to change).
_COMPARE_WORDS = frozenset({
    "compare", "comparing", "comparison", "contrast", "difference", "differences",
    "differ", "diff", "different",
})
_COMPARE_FILLER = frozenset({
    "please", "pls", "plz", "can", "could", "would", "you", "i", "want", "to", "me",
    "us", "just", "the", "a", "these", "those", "them", "they", "both", "two",
    "image", "images", "scene", "scenes", "picture", "pictures", "photo", "photos",
    "tile", "tiles", "date", "dates", "pair", "show", "tell", "give", "do", "does",
    "make", "spot", "find", "what", "whats", "what's", "is", "are", "how", "and",
    "between", "of", "for", "in", "it", "now", "there", "see", "let", "know",
})


def is_bare_comparison(text: str) -> bool:
    words = re.findall(r"[a-z']+", (text or "").lower())
    return (
        any(w in _COMPARE_WORDS for w in words)
        and all(w in _COMPARE_WORDS or w in _COMPARE_FILLER for w in words)
    )


# ---------------------------------------------------------------------------
# Follow-up resolution
# ---------------------------------------------------------------------------

def _prior(history: list[dict[str, Any]] | None) -> dict[str, Any] | None:
    """The most recent completed turn, normalised: task, object, query."""
    if not history:
        return None
    for turn in reversed(history):
        if not isinstance(turn, dict):
            continue
        task = turn.get("task") or (turn.get("understanding") or {}).get("task")
        if not task:
            continue
        u = turn.get("understanding") or {}
        obj = u.get("object_filter") or extract_object(str(turn.get("query") or ""))
        return {"task": task, "object": obj, "query": str(turn.get("query") or ""),
                "classes": u.get("classes")}
    return None


_FOLLOW_UP_MAX_WORDS = 7


def _plural(noun: str) -> bool:
    """Crude number agreement for the rewrites ("Where are the ships?")."""
    w = noun.split()[-1].lower() if noun else ""
    return w.endswith("s") and not w.endswith(("ss", "us", "is")) or w in ("aircraft", "people")


def _be(noun: str) -> str:
    return "are" if _plural(noun) else "is"


def resolve_follow_up(query: str, history: list[dict[str, Any]] | None, config: str) -> tuple[str, str | None]:
    """Rewrite an elliptical or anaphoric follow-up into a standalone query.

    Returns (resolved_query, note). The note is None when nothing was done,
    which is also the answer for a query that stands on its own: a fresh
    sentence is never rewritten, however short, unless it leans on the
    previous turn (a pronoun, "only …", "and the …", "where exactly").
    """
    q = (query or "").strip()
    prior = _prior(history)
    if not prior or not q:
        return q, None
    words = q.rstrip("?.!").split()
    short = len(words) <= _FOLLOW_UP_MAX_WORDS
    anaphoric = bool(_CUES["anaphor"].search(q))
    if not (short or anaphoric):
        return q, None
    obj = prior["object"]
    temporal = prior["task"] in TEMPORAL_TASKS
    the_obj = f"the {obj}" if obj else None
    lower = q.lower().rstrip("?.! ")

    def note(msg: str) -> str:
        return f"follow-up of {prior['task']} ({prior['query']!r}): {msg}"

    # "Where exactly?" / "show me on a map" / "show me where"
    if re.fullmatch(r"(?:where(?: exactly| is that| was that| are they| is it)?|show (?:me )?(?:on a |on the )?map|on a map|show me where|show it on a map|point (?:to|at) (?:it|them|that)|mark (?:it|them)|show (?:it|them)|point it out)", lower):
        if temporal:
            what = f"the {obj} that changed" if obj else "the changes"
            return f"Show me where {what} occurred.", note("asked where; mapped the change")
        if obj:
            return f"Where {_be(obj)} {the_obj}?", note("asked where; located the previous object")
        return q, None
    # "Only show buildings." / "just the roads" / "only buildings"
    m = re.fullmatch(r"(?:only|just)(?: show(?: me)?| the| show me the| keep)? (?:the )?([a-z][a-z \-]+)", lower)
    if m:
        new_obj = m.group(1).strip()
        if temporal:
            return f"Show me the {new_obj} that changed.", note(f"narrowed the change result to {new_obj!r}")
        if prior["task"] == "SINGLE_LANDCOVER" or extract_classes(new_obj):
            return f"Map the {new_obj} land cover.", note(f"narrowed the classification to {new_obj!r}")
        return f"Where are the {new_obj}?", note(f"narrowed to {new_obj!r}")
    # "And the dam?" / "What about the harbour?" / "what about the land cover?"
    m = re.fullmatch(r"(?:and|what about|how about|also|now)(?: the| a| an)? ([a-z][a-z \-]+)", lower)
    if m:
        new_obj = m.group(1).strip()
        if extract_classes(new_obj) and re.search(r"\b(land ?cover|land ?use|classes|classification|layer|vegetation|water|built)\b", new_obj):
            return f"Classify the {new_obj} in this image.", note(f"switched subject to {new_obj!r}")
        if re.search(r"\b(land ?cover|land ?use|classification)\b", new_obj):
            return f"Classify the {new_obj}.", note(f"switched subject to {new_obj!r}")
        if re.search(r"\b(caption|description)\b", new_obj):
            return "Caption this image.", note("switched to a caption")
        if temporal:
            return f"Did the {new_obj} change between the two images?", note(f"switched subject to {new_obj!r}")
        if prior["task"] == "SINGLE_GROUND":
            return f"Where {_be(new_obj)} the {new_obj}?", note(f"switched subject to {new_obj!r}")
        return f"Is there {new_obj} in this image?", note(f"switched subject to {new_obj!r}")
    # "how many?" / "how many are there"
    if re.fullmatch(r"how many(?: are there| of them| are they| in total)?", lower):
        if obj:
            return f"How many {obj} are there?", note("counted the previous object")
        return q, None
    # "how much?" after a change answer
    if re.fullmatch(r"(?:how much|by how much|how big(?: a change)?|how significant)(?: did it change| was it| is it)?", lower):
        if temporal:
            what = the_obj or "the changed area"
            return f"By how much did {what} change between the two images?", note("quantified the change")
        if obj:
            return f"What proportion of the scene is {obj}?", note("quantified the previous object")
        return q, None
    # "why?"
    if re.fullmatch(r"why(?: is that| did that happen)?", lower):
        if temporal:
            return "Describe what changed between the two images and what the changes look like.", note("asked why; described the change")
        return q, None
    # "the same but as a raster" / "as a mask" / "as a map"
    if re.search(r"\b(as a|as an|but as|in) (raster|mask|map|geotiff|layer|image)\b", lower) or re.fullmatch(r"(raster|mask|map)( version)?( please)?", lower):
        if temporal:
            what = f" of the {obj} that changed" if obj else ""
            return f"Produce a change mask{what}.", note("re-requested the result as a raster")
        return "Produce a land cover map.", note("re-requested the result as a map")
    # "describe that area" / "describe it"
    if re.fullmatch(r"(?:describe|tell me about|explain) (?:it|that|them|this|that area|this area|the area|that region|those)", lower):
        if obj:
            return f"Describe the {obj} and the area around it.", note("described the previous object")
        return q, None
    # Pronoun substitution: "Is it near the coast?" -> "Is the airport near the coast?"
    if obj and re.search(r"\b(it|its|that|them|they|those|there)\b", lower):
        resolved = re.sub(r"\b(it|they|them|that|those)\b", the_obj, q, count=1, flags=re.IGNORECASE)
        resolved = re.sub(r"\bof it\b", f"of {the_obj}", resolved, flags=re.IGNORECASE)
        resolved = re.sub(r"\bare the ", "are the ", resolved)
        if resolved != q:
            return resolved, note("substituted the pronoun with the previous object")
    return q, None


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def understand(
    query: str,
    config: str,
    task: TaskID,
    inputs: list[str],
    history: list[dict[str, Any]] | None = None,
    resolved: tuple[str, str | None] | None = None,
) -> QueryUnderstanding:
    """Build the IR for a query whose task the router has decided.

    `resolved` is the (resolved_query, note) pair if the router already
    resolved the follow-up (it must, before classifying); otherwise it is
    computed here.
    """
    resolved_query, note = resolved if resolved is not None else resolve_follow_up(query, history, config)
    text = resolved_query or query
    spatial = extract_spatial(text)
    obj = extract_object(text)
    prior = _prior(history)
    if obj is None and prior and prior.get("object") and note:
        obj = prior["object"]
    classes = extract_classes(text)
    temporal = "before_after" if task in TEMPORAL_TASKS else None
    return QueryUnderstanding(
        query=query,
        resolved_query=text,
        config=config,
        intent=INTENT_BY_TASK.get(task, "unclear"),
        task=task,
        inputs=list(inputs),
        requested_output=OUTPUT_BY_TASK.get(task, "clarification"),
        object_filter=obj,
        referring_expression=referring_expression(text, obj, spatial),
        spatial_scope=spatial,
        temporal_relation=temporal,
        quantity=extract_quantity(text),
        classes=classes,
        image_index=extract_image_index(text, config) if task not in TEMPORAL_TASKS and task != "XMODAL_JOINT_EXTRACT" else None,
        follow_up=note is not None,
        resolution=note,
        cues=cues(text),
    )
