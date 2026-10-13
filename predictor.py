"""Core logic: parse staff questions, detect repeated ones, and build a question paper.

Algorithm
---------
1. Parsing: each line of staff input is one question. Numbering ("1.", "Q2)") and
   mark tags ("(5 marks)", "[2 marks]", "2 marks") are extracted.
2. Similarity: questions are converted to TF-IDF vectors (word unigrams and bigrams,
   English stop words removed, sublinear term frequency) and compared with cosine
   similarity.
3. Repetition detection: questions whose similarity is at or above a threshold are
   grouped together (greedy single-pass clustering). A group's frequency is the number
   of times that question, or a near-identical wording of it, was submitted.
4. Prediction ranking: groups are scored as frequency * (0.5 + 0.5 * cohesion), where
   cohesion is the average pairwise similarity inside the group. Frequent, consistently
   worded questions rank first.
5. Paper building: for each requested section (e.g. 5 x 2-mark, 3 x 5-mark) the top
   ranked groups with matching marks are selected, each question is used only once.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

MARK_PATTERN = re.compile(
    r"[\(\[]\s*(\d{1,2})\s*marks?\s*[\)\]]|\b(\d{1,2})\s*marks?\b",
    re.IGNORECASE,
)
PREFIX_PATTERN = re.compile(r"^\s*(?:Q\s*\d+[.):]?|\d+[.):]|[-*•])\s*", re.IGNORECASE)
MIN_QUESTION_LENGTH = 8

# Instruction words appear in almost every exam question ("explain", "define") and
# carry no topic information, so they are ignored when comparing questions.
INSTRUCTION_WORDS = {
    "explain", "describe", "define", "discuss", "list", "state", "write", "give",
    "brief", "short", "note", "compare", "differentiate", "illustrate", "mention",
    "types", "type", "kinds", "kind", "examples", "example", "purpose", "significance",
    "importance", "advantages", "disadvantages", "process", "concept", "meaning",
    "marks", "mark", "question", "answer", "briefly", "detail", "detailed",
}
def _normalize_word(word: str) -> str:
    """Very light stemming so 'joins' and 'join' count as the same word."""
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


# Stop words must be stemmed the same way as the question text.
DOMAIN_STOP_WORDS = frozenset(
    _normalize_word(w) for w in (ENGLISH_STOP_WORDS | INSTRUCTION_WORDS)
)


def _preprocess(text: str) -> str:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return " ".join(_normalize_word(w) for w in words)


@dataclass
class Question:
    text: str
    marks: int | None = None


@dataclass
class QuestionGroup:
    representative: str
    variants: list[str] = field(default_factory=list)
    marks: int | None = None
    frequency: int = 0
    score: float = 0.0

    def to_dict(self) -> dict:
        others = [v for v in dict.fromkeys(self.variants) if v != self.representative]
        return {
            "question": self.representative,
            "frequency": self.frequency,
            "is_repeated": self.frequency > 1,
            "marks": self.marks,
            "similar_wordings": others[:3],
            "score": round(self.score, 3),
        }


def parse_questions(raw: str) -> list[Question]:
    """Turn pasted staff text (one question per line) into Question objects."""
    questions: list[Question] = []
    for line in raw.splitlines():
        text = PREFIX_PATTERN.sub("", line).strip()
        marks = None
        match = MARK_PATTERN.search(text)
        if match:
            marks = int(match.group(1) or match.group(2))
            text = (text[: match.start()] + text[match.end():]).strip(" -:()[]")
        if len(text) >= MIN_QUESTION_LENGTH:
            questions.append(Question(text=text, marks=marks))
    return questions


def _similarity_matrix(texts: list[str]) -> np.ndarray:
    try:
        vectorizer = TfidfVectorizer(
            preprocessor=_preprocess,
            stop_words=list(DOMAIN_STOP_WORDS),
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        matrix = vectorizer.fit_transform(texts)
        return cosine_similarity(matrix)
    except ValueError:  # empty vocabulary, e.g. every word is a stop word
        return np.eye(len(texts))


def group_questions(questions: list[Question], threshold: float = 0.6) -> list[QuestionGroup]:
    """Cluster near-duplicate questions and rank the clusters by repetition."""
    if not questions:
        return []

    sim = _similarity_matrix([q.text for q in questions])

    clusters: list[list[int]] = []
    for i in range(len(questions)):
        best_idx, best_sim = None, threshold
        for c_idx, members in enumerate(clusters):
            s = sim[i, members[0]]  # compare with the cluster's first question
            if s >= best_sim:
                best_idx, best_sim = c_idx, s
        if best_idx is None:
            clusters.append([i])
        else:
            clusters[best_idx].append(i)

    groups: list[QuestionGroup] = []
    for members in clusters:
        texts = [questions[i].text for i in members]
        known_marks = [questions[i].marks for i in members if questions[i].marks is not None]
        marks = Counter(known_marks).most_common(1)[0][0] if known_marks else None

        frequency = len(members)
        cohesion = 1.0
        if frequency > 1:
            sub = sim[np.ix_(members, members)]
            cohesion = float((sub.sum() - frequency) / (frequency * (frequency - 1)))

        groups.append(
            QuestionGroup(
                representative=texts[0],
                variants=texts,
                marks=marks,
                frequency=frequency,
                score=frequency * (0.5 + 0.5 * cohesion),
            )
        )

    groups.sort(key=lambda g: (-g.score, -g.frequency, g.representative.lower()))
    return groups


def build_question_paper(
    groups: list[QuestionGroup],
    sections: list[dict],
    title: str = "Question Paper",
    subject: str = "",
) -> dict:
    """Pick the top predicted questions for each section.

    sections: [{"marks": 2, "count": 5}, {"marks": 5, "count": 3}]
    Questions whose marks are unknown are allowed in any section.
    """
    used: set[str] = set()
    warnings: list[str] = []
    built: list[dict] = []

    for index, section in enumerate(sections):
        marks, wanted = section["marks"], section["count"]
        candidates = [g for g in groups if g.marks in (marks, None)]
        # Exact mark matches first, then unlabelled questions; higher score first.
        candidates.sort(key=lambda g: (g.marks != marks, -g.score))

        chosen: list[dict] = []
        for group in candidates:
            if len(chosen) == wanted:
                break
            if group.representative in used:
                continue
            used.add(group.representative)
            chosen.append({"question": group.representative, "marks": marks, "frequency": group.frequency})

        if len(chosen) < wanted:
            warnings.append(
                f"Only {len(chosen)} of {wanted} {marks}-mark questions were available. "
                "Add more staff questions to fill the paper."
            )

        built.append(
            {
                "label": f"Part {chr(65 + index)}",
                "marks": marks,
                "questions": chosen,
                "subtotal": marks * len(chosen),
            }
        )

    number = 1
    for section in built:
        for item in section["questions"]:
            item["number"] = number
            number += 1

    return {
        "title": title,
        "subject": subject,
        "sections": built,
        "total_marks": sum(s["subtotal"] for s in built),
        "warnings": warnings,
    }


def analyze(
    raw: str,
    threshold: float = 0.6,
    top_n: int = 10,
    sections: list[dict] | None = None,
    title: str = "Question Paper",
    subject: str = "",
) -> dict:
    """Full pipeline used by the web API."""
    questions = parse_questions(raw)
    groups = group_questions(questions, threshold)
    result = {
        "stats": {
            "questions_read": len(questions),
            "unique_questions": len(groups),
            "repeated_questions": sum(1 for g in groups if g.frequency > 1),
        },
        "predictions": [g.to_dict() for g in groups[:top_n]],
        "paper": None,
    }
    if sections:
        result["paper"] = build_question_paper(groups, sections, title, subject)
    return result
