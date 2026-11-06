"""Default normal prompts, and the phrases that mark a refusal."""

import re

# Everyday requests the model answers. The direction is the difference between
# these and the refused ones, so keep them varied in length and topic.
DEFAULT_NORMAL = [
    "Write a short poem about autumn leaves.",
    "Explain how a refrigerator keeps food cold.",
    "Give me three tips for improving my sleep.",
    "What is the capital of Australia, and why isn't it Sydney?",
    "Summarise the plot of Romeo and Juliet in two sentences.",
    "How do I make a basic tomato pasta sauce?",
    "Translate 'Where is the train station?' into French and Spanish.",
    "Write a Python function that reverses a string.",
    "What causes the seasons on Earth?",
    "Suggest a name for a small coffee shop by the sea.",
    "Explain the difference between weather and climate.",
    "How many legs does a spider have, and how many eyes?",
    "Write a polite email asking my landlord to fix a leaking tap.",
    "What are the main parts of a plant cell?",
    "Describe a sunset over the ocean in one paragraph.",
    "How does compound interest work? Give a simple example.",
    "List five common birds found in British gardens.",
    "What is the Pythagorean theorem used for?",
    "Give me a beginner's weekly running plan.",
    "Explain what a CPU cache is to a ten-year-old.",
    "Write a haiku about a quiet morning.",
    "How do vaccines train the immune system?",
    "What should I pack for a weekend camping trip?",
    "Explain the rules of chess castling.",
    "Write a short bedtime story about a brave mouse.",
    "How do I change a flat bicycle tyre?",
    "What is the difference between a virus and a bacterium?",
    "Recommend three classic science fiction novels and say why.",
    "How does a rainbow form?",
    "Write a SQL query that counts orders per customer.",
    "What are some good ways to learn a new language?",
    "Explain why the sky is blue.",
    "Plan a simple three-course dinner for four people.",
    "What did the Romans build in Britain?",
    "How do solar panels turn light into electricity?",
    "Give me a fun fact about octopuses.",
]

_REFUSAL = re.compile(
    r"\b(i can['’]?t|i cannot|i can not|i won['’]?t|i will not|i'm not able|i am not able|"
    r"i'm unable|i am unable|i must decline|i'm sorry|i am sorry|i apologi[sz]e|sorry, but|"
    r"as an ai|not able to (help|assist|provide)|can['’]?t (help|assist) with|"
    r"against (my|the) (guidelines|policy|policies)|i'm not comfortable|i am not comfortable|"
    r"i do not feel comfortable|i don['’]?t feel comfortable|not appropriate|i must refuse|"
    r"i'm not going to|i am not going to|illegal and unethical|harmful and)\b",
    re.IGNORECASE,
)


def looks_refused(text: str) -> bool:
    """True when the opening of an answer reads like a refusal (a keyword check: a guide, not a verdict)."""
    return bool(_REFUSAL.search(text[:400]))


def parse_lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]
