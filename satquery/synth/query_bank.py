"""Synthetic query bank for the Tier-1 intent classifier (plan task 1.4).

Templates with filled slots, expanded into thousands of paraphrases. The task
boundaries here define what the classifier can learn, so each task's templates
are written to be *linguistically* distinguishable, not just semantically:
- CAPTION asks for a free description of the whole scene
- VQA asks a specific question with a specific answer
- GROUND asks for a location, expecting boxes back
- LANDCOVER asks for classification into classes
- XMODAL explicitly invokes both sensors
- CHANGE_DESC / CHANGE_VQA / CHANGE_MAP split by requested output form:
  prose, an answer to a question, or a raster mask
- CLARIFY_OR_ABSTAIN covers greetings, vagueness and impossible requests

Deterministic given a seed, so classifier metrics are reproducible.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from satquery.contracts.plan import TaskID

FEATURES = [
    "buildings", "roads", "water bodies", "farmland", "forest", "urban area",
    "vegetation", "bridges", "ships", "aircraft", "solar panels", "rooftops",
    "rivers", "lakes", "crops", "bare soil", "settlements", "runways",
    "storage tanks", "harbours",
]

SINGULAR = [
    "a building", "a road", "a river", "a bridge", "a ship", "an aircraft",
    "a lake", "a runway", "a storage tank", "a harbour",
]

LANDCOVER_CLASSES = [
    "land cover", "land use", "surface types", "terrain classes",
    "land cover classes",
]

CHANGE_NOUNS = [
    "urban growth", "deforestation", "new construction", "water extent",
    "vegetation loss", "flooding", "land clearing", "built-up expansion",
]

_VQA = [
    "How many {feature} are visible in this image?",
    "How many {feature} can you count?",
    "Is there {singular} in this scene?",
    "Are there any {feature} present?",
    "What is the approximate area covered by {feature}?",
    "Does this image contain {feature}?",
    "What proportion of the scene is {feature}?",
    "Which is more prevalent, {feature} or {feature2}?",
    "Is the {feature} area larger than the {feature2} area?",
    "What colour are the {feature} in this image?",
    "Count the {feature} in the image.",
    "Tell me how many {feature} there are.",
    "What is the total extent of {feature}?",
    "Can you confirm whether {feature} appear here?",
    "Roughly what percentage of this scene is {feature}?",
    "Any idea if there are {feature} here?",
    "What share of this is {feature}?",
    "Give me a number for the {feature}.",
    "Whats the count of {feature}?",
]

_CAPTION = [
    "Describe this image.",
    "Caption this scene.",
    "Give me a description of what this image shows.",
    "Write a caption for this satellite image.",
    "Summarise what is visible in this scene.",
    "What does this image show?",
    "Provide an overall description of the imagery.",
    "Explain what this scene contains.",
    "Give a short summary of this image.",
    "Describe the overall content of this scene.",
    "Narrate what can be seen here.",
    "Write a few sentences about this image.",
    "Give me an overview of this scene.",
    "Describe the landscape in this image.",
    "What would you say this image depicts?",
    "Give me a rundown of this picture.",
    "Whats going on in this shot?",
    "Sum up the imagery.",
    "In a sentence or two, describe this scene.",
    "Talk me through what this image shows.",
]

_GROUND = [
    "Show me where the {feature} are.",
    "Locate the {feature} in this image.",
    "Highlight all {feature}.",
    "Point to {singular}.",
    "Where is {singular} in this scene?",
    "Draw boxes around the {feature}.",
    "Mark the location of the {feature}.",
    "Find the {feature} and show their positions.",
    "Identify where {feature} appear.",
    "Give me the coordinates of the {feature}.",
    "Outline the {feature} in the image.",
    "Where exactly are the {feature} located?",
    "Detect and localise the {feature}.",
    "Circle the {feature} in this scene.",
    "Pinpoint {singular} for me.",
    "Put a box on every {singular}.",
    "Whereabouts is {singular}?",
    "I want the pixel positions of the {feature}.",
    "Box the {feature}.",
    "Which part of the image has the {feature}?",
]

_LANDCOVER = [
    "Classify the {landcover} in this image.",
    "Produce a {landcover} map.",
    "What {landcover} are present in this scene?",
    "Generate a {landcover} classification.",
    "Break this scene down by {landcover}.",
    "Map the {landcover} for this image.",
    "Which {landcover} categories occur here?",
    "Give me a {landcover} breakdown.",
    "Segment this image by {landcover}.",
    "Label the {landcover} across the scene.",
    "Provide a per-class {landcover} summary.",
    "Categorise the surface into {landcover}.",
    "Run a {landcover} classification on this image.",
    "What fraction of each {landcover} class is present?",
    "Produce a thematic {landcover} map.",
]

_XMODAL = [
    "Combine the optical and radar images to identify {feature}.",
    "Using both the optical and SAR data, find {feature}.",
    "What does the SAR image add to the optical view?",
    "Fuse the optical and radar imagery and describe {feature}.",
    "Compare what the optical and SAR sensors show for {feature}.",
    "Use the radar and optical images together to assess {feature}.",
    "Where do the optical and SAR images disagree about {feature}?",
    "Jointly analyse the optical and SAR scenes.",
    "Extract {feature} using both sensors.",
    "Cross-check the optical result against the SAR data.",
    "What can radar reveal here that optical cannot?",
    "Merge the two modalities and report on {feature}.",
    "Use optical and SAR jointly to map {feature}.",
    "Exploit both sensors to detect {feature}.",
    "Combine radar backscatter with optical reflectance for {feature}.",
]

_CHANGE_DESC = [
    "Describe what changed between the two images.",
    "What is different between these two dates?",
    "Summarise the changes between the two acquisitions.",
    "Explain how the scene changed over time.",
    "Tell me what has changed here.",
    "Describe the {change} visible between the images.",
    "Give me a narrative of the changes.",
    "What developments occurred between the two images?",
    "Characterise the differences between the two scenes.",
    "Report on how this area evolved.",
    "Describe the transformation between the two dates.",
    "In words, what changed here?",
    "Give an account of the {change}.",
    "Summarise the temporal differences.",
    "What happened between the first and second image?",
    "Whats moved since last time?",
    "Write up how the area shifted.",
    "Talk me through the differences across the two passes.",
    "Describe how things developed since the earlier image.",
]

_CHANGE_VQA = [
    "How much did the {feature} area change between the two images?",
    "Did the {feature} increase or decrease?",
    "By what percentage did {feature} change?",
    "How many new {feature} appeared?",
    "Was there any {change} between the two dates?",
    "Did {change} occur here?",
    "How much {feature} was lost?",
    "What is the net change in {feature}?",
    "Is the {feature} extent larger in the second image?",
    "Quantify the change in {feature}.",
    "How many {feature} were removed?",
    "What is the change in area of {feature} in hectares?",
    "Did the amount of {feature} grow?",
    "How significant was the {change}?",
    "Count the {feature} that changed.",
    "How many more {feature} are there now than before?",
    "Put a number on the {change}.",
    "Did the {feature} shrink since the earlier image?",
    "How much {feature} was there before compared to now?",
    "Is there more {feature} now than previously?",
]

_CHANGE_MAP = [
    "Produce a change mask for these two images.",
    "Generate a change detection map.",
    "Show me where the changes occurred.",
    "Map the areas that changed between the two dates.",
    "Create a binary change raster.",
    "Output a change mask highlighting {change}.",
    "Give me a georeferenced change layer.",
    "Render the changed pixels as a mask.",
    "Produce a change map I can open in QGIS.",
    "Segment the changed regions.",
    "Export the change detection result as a raster.",
    "Where did {change} happen? Give me a mask.",
    "Delineate the changed areas.",
    "Generate a per-pixel change map.",
    "Produce a spatial map of {change}.",
    "I want a raster showing altered pixels.",
    "Output a difference layer for GIS.",
    "Draw the footprint of what moved.",
    "Give me a mask of the altered pixels.",
    "Export a difference raster between the two dates.",
]

_CLARIFY = [
    "Hello.",
    "Hi there.",
    "What do you think?",
    "Tell me about it.",
    "Can you help?",
    "What is this?",
    "Anything interesting?",
    "Do the thing.",
    "Analyse.",
    "Go ahead.",
    "What should I do next?",
    "Is it good?",
    "Compare them.",
    "Which one is better?",
    "Show me the difference.",
    "What about the other one?",
    "Explain.",
    "And?",
    "Continue.",
    "Sort it out.",
]

TEMPLATES: dict[TaskID, list[str]] = {
    "SINGLE_VQA": _VQA,
    "SINGLE_CAPTION": _CAPTION,
    "SINGLE_GROUND": _GROUND,
    "SINGLE_LANDCOVER": _LANDCOVER,
    "XMODAL_JOINT_EXTRACT": _XMODAL,
    "TEMPORAL_CHANGE_DESC": _CHANGE_DESC,
    "TEMPORAL_CHANGE_VQA": _CHANGE_VQA,
    "TEMPORAL_CHANGE_MAP": _CHANGE_MAP,
    "CLARIFY_OR_ABSTAIN": _CLARIFY,
}

# Prefixes and suffixes that add natural variation without changing intent.
_PREFIXES = ["", "", "", "Please ", "Could you ", "I need you to ", "Can you "]
_SUFFIXES = ["", "", "", "", " for me", " please", " in this scene"]


@dataclass(frozen=True)
class QueryExample:
    text: str
    task: TaskID


def _fill(template: str, rng: random.Random) -> str:
    feature = rng.choice(FEATURES)
    feature2 = rng.choice([f for f in FEATURES if f != feature])
    return (
        template.replace("{feature2}", feature2)
        .replace("{feature}", feature)
        .replace("{singular}", rng.choice(SINGULAR))
        .replace("{landcover}", rng.choice(LANDCOVER_CLASSES))
        .replace("{change}", rng.choice(CHANGE_NOUNS))
    )


def _decorate(text: str, rng: random.Random) -> str:
    prefix = rng.choice(_PREFIXES)
    suffix = rng.choice(_SUFFIXES)
    if prefix and text:
        # Lower-case the first letter so "Please Describe" reads naturally.
        text = text[0].lower() + text[1:]
    if suffix and text.endswith((".", "?")):
        text = text[:-1] + suffix + text[-1]
    return prefix + text


def generate(
    per_task: int = 400, seed: int = 20260829, decorate: bool = True
) -> list[QueryExample]:
    """Generate a balanced, deterministic query bank."""
    rng = random.Random(seed)
    examples: list[QueryExample] = []
    for task, templates in TEMPLATES.items():
        seen: set[str] = set()
        attempts = 0
        while len(seen) < per_task and attempts < per_task * 60:
            attempts += 1
            text = _fill(rng.choice(templates), rng)
            if decorate:
                text = _decorate(text, rng)
            if text not in seen:
                seen.add(text)
        examples.extend(QueryExample(text=t, task=task) for t in sorted(seen))
    rng.shuffle(examples)
    return examples


def train_test_split(
    examples: list[QueryExample], test_fraction: float = 0.2, seed: int = 7
) -> tuple[list[QueryExample], list[QueryExample]]:
    """Stratified split so every task is represented in both halves."""
    rng = random.Random(seed)
    by_task: dict[str, list[QueryExample]] = {}
    for ex in examples:
        by_task.setdefault(ex.task, []).append(ex)

    train: list[QueryExample] = []
    test: list[QueryExample] = []
    for task_examples in by_task.values():
        shuffled = list(task_examples)
        rng.shuffle(shuffled)
        cut = int(len(shuffled) * test_fraction)
        test.extend(shuffled[:cut])
        train.extend(shuffled[cut:])
    rng.shuffle(train)
    rng.shuffle(test)
    return train, test
