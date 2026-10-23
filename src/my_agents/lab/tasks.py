"""Hidden-profile tasks with known answers, generated from a seed.

A team picks the best of three candidates. Every fact is a strength (+1) or a weakness (-1)
of one candidate; the best candidate has the highest total over ALL facts the team holds.

The facts are arranged as a classic hidden profile (Stasser & Titus 1985; HiddenBench):
- shared facts (everyone sees them) make the distractor D look best;
- private facts, spread over the agents, favour the correct candidate X;
- the private facts of at least three agents are needed before X wins.

So a team that stops as soon as it agrees on what everyone already knows gets it wrong,
and we can measure exactly which facts were surfaced before it stopped. With full
information every agent sees every fact, and the task is easy.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

NAMES = ["Ana", "Bram", "Chen", "Dalia", "Emre", "Farah", "Goran", "Hana", "Ivo", "Jun", "Kemi", "Lars",
         "Mira", "Nikhil", "Ola", "Priya", "Quinn", "Rosa", "Sami", "Tomas", "Uma", "Vik", "Wen", "Yara"]
STRENGTHS = [
    "shipped a production ML system used by thousands of people", "has a first-author NeurIPS paper",
    "got excellent code-review feedback from a past team", "mentored three interns who were all rehired",
    "built the evaluation harness their last team still uses", "ran distributed training on 512 GPUs",
    "writes unusually clear design documents", "handled a major production incident calmly and well",
    "reproduced two hard papers from scratch", "gave a well-received talk at a top workshop",
    "maintains a popular open-source library", "found a subtle data leak in a benchmark",
    "has strong references from two research leads", "designed a red-teaming study that changed a launch",
    "cut a training pipeline's cost in half", "wrote the team's most-used onboarding guide",
    "won an internal award for research rigour", "turned a vague research question into a shipped eval",
]
WEAKNESSES = [
    "missed three project deadlines last year without warning", "copied code in a take-home test",
    "had repeated complaints about dismissive behaviour", "could not explain their own paper's main result",
    "left two projects unfinished when they got bored", "overstated results in a past internal report",
    "broke production twice by skipping code review", "refused to share credit with collaborators",
    "ignored a safety concern raised by a teammate", "has no experience with large codebases",
    "fabricated a reference in a grant application", "was let go from a previous team for poor reliability",
]

SHARED_D_STRENGTHS = 6  # the distractor looks great on shared facts
MIN_HOLDERS = 3  # private facts of this many agents are needed before the correct answer wins


@dataclass
class Fact:
    id: str
    candidate: str  # A | B | C
    polarity: int  # +1 strength, -1 weakness
    text: str
    holder: int | None  # None = shared; else the index of the agent that holds it

    def render(self, names: dict[str, str]) -> str:
        kind = "strength" if self.polarity > 0 else "weakness"
        return f"[{self.id}] Candidate {self.candidate} ({names[self.candidate]}): {kind}: {self.text}."


@dataclass
class Task:
    id: str
    names: dict[str, str]  # letter -> name
    facts: list[Fact]
    answer: str  # correct with all facts
    distractor: str  # best on shared facts alone
    holders: int  # how many agents hold private facts
    meta: dict = field(default_factory=dict)

    def shared(self) -> list[Fact]:
        return [f for f in self.facts if f.holder is None]

    def private(self, holder: int | None = None) -> list[Fact]:
        return [f for f in self.facts if f.holder is not None and (holder is None or f.holder == holder)]

    def view(self, agent: int, full_info: bool) -> list[Fact]:
        """The facts one agent sees."""
        if full_info:
            return list(self.facts)
        return self.shared() + self.private(agent)


def scores(facts: list[Fact]) -> dict[str, int]:
    out = {"A": 0, "B": 0, "C": 0}
    for f in facts:
        out[f.candidate] += f.polarity
    return out


def best(facts: list[Fact]) -> str | None:
    """The candidate with the highest total, or None on a tie for first place."""
    s = scores(facts)
    top = max(s.values())
    winners = [c for c, v in s.items() if v == top]
    return winners[0] if len(winners) == 1 else None


def make_task(seed: int, holders: int = 4) -> Task:
    """One hidden-profile task. `holders` agents each hold private facts (at least MIN_HOLDERS)."""
    if not MIN_HOLDERS <= holders <= 8:
        raise ValueError(f"holders must be {MIN_HOLDERS}..8")
    rng = random.Random(seed)
    letters = ["A", "B", "C"]
    x, d, c = rng.sample(letters, 3)  # correct, distractor, other
    names = dict(zip(letters, rng.sample(NAMES, 3)))
    strengths, weaknesses = rng.sample(STRENGTHS, len(STRENGTHS)), rng.sample(WEAKNESSES, len(WEAKNESSES))
    raw: list[tuple[str, int, int | None]] = []
    raw += [(d, +1, None)] * SHARED_D_STRENGTHS  # shared: D looks best
    raw += [(c, +1, None), (c, +1, None), (c, -1, None)]  # shared: C is middling
    raw += [(x, +1, None), (x, -1, None)]  # shared: X looks average
    for h in range(holders):  # private: each holder has one strength of X and one weakness of D
        raw += [(x, +1, h), (d, -1, h)]
    raw += [(x, +1, 0)]  # holder 0 has one more strength of X
    rng.shuffle(raw)
    facts = []
    for i, (cand, pol, holder) in enumerate(raw, 1):
        text = strengths.pop() if pol > 0 else weaknesses.pop()
        facts.append(Fact(f"F{i}", cand, pol, text, holder))
    task = Task(f"hp{seed}", names, facts, answer=x, distractor=d, holders=holders)
    # The construction guarantees the hidden profile; check it anyway.
    assert best(facts) == x and best(task.shared()) == d
    for h in range(holders):
        assert best(task.view(h, full_info=False)) == d, "one agent alone must be misled"
    return task


def sufficient(task: Task, surfaced: set[str]) -> bool:
    """Do the shared facts plus the surfaced private facts already determine the correct answer?"""
    known = task.shared() + [f for f in task.private() if f.id in surfaced]
    return best(known) == task.answer
