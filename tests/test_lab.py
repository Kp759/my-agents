"""Stop-signal lab, offline: task construction, controllers and metrics with scripted agents."""

import json
import re

import pytest

from my_agents.adapters import FakeAdapter
from my_agents.lab import run as lab
from my_agents.lab.tasks import best, make_task, scores, sufficient
from my_agents.lab.team import Team, consensus, debate, evidence, evidence_state, parse_turn, vote


@pytest.fixture(autouse=True)
def data(tmp_path, monkeypatch):
    monkeypatch.setenv("MY_AGENTS_DATA", str(tmp_path))


FACT = re.compile(r"\[(F\d+)\] Candidate ([ABC]) \([^)]*\): (strength|weakness)")


def thinker(share=True, stubborn=None):
    """A scripted agent: adds up every fact it can see (its own and those cited so far).

    share=False: answers correctly from what it sees but cites nothing (a team of these never pools).
    stubborn: always answer this letter (an adversary that never moves).
    """
    def reply(system, messages):
        text = messages[-1]["content"]
        facts = {fid: (c, 1 if k == "strength" else -1) for fid, c, k in FACT.findall(text)}
        total = {"A": 0, "B": 0, "C": 0}
        for c, pol in facts.values():
            total[c] += pol
        answer = stubborn or max(total, key=total.get)
        return json.dumps({"answer": answer, "confidence": 0.9, "evidence": sorted(facts) if share else [],
                           "message": "ok"})
    return FakeAdapter(reply=reply)


def test_every_task_is_a_hidden_profile():
    for seed in range(50):
        t = make_task(seed)
        assert best(t.facts) == t.answer != t.distractor == best(t.shared())
        for h in range(t.holders):
            assert best(t.view(h, full_info=False)) == t.distractor  # alone, everyone is misled
        two = {f.id for f in t.private(0) + t.private(1)}
        assert not sufficient(t, two)  # two members' facts are not enough
        assert sufficient(t, {f.id for f in t.private()})


def test_turns_are_parsed_robustly():
    assert parse_turn('```json\n{"answer": "b", "confidence": 2, "evidence": ["F3", "F9"], "message": "hi"}\n```') == \
        {"answer": "B", "confidence": 1.0, "evidence": ["F3", "F9"], "message": "hi", "parsed": True}
    assert parse_turn("I think the answer: C")["answer"] == "C"
    assert parse_turn("no idea")["answer"] is None


def test_agents_can_only_share_facts_they_know():
    t = make_task(1)
    other = t.private(3)[0].id
    team = Team(t, FakeAdapter(reply=json.dumps({"answer": "A", "evidence": [other, "F999"]})), full_info=False)
    turn = team.turn(0, 1, [])
    assert turn["evidence"] == []  # can't cite member 4's private fact or a made-up one


def test_consensus_stops_on_agreement_even_when_both_are_misled():
    t = make_task(2)
    team = Team(t, thinker(share=False), full_info=False)
    res = consensus(team)
    row = lab.score(t, res, team)
    assert res["stop"] == "agreement" and row["answer"] == t.distractor
    assert row["unsafe_stop"] and row["premature_stop"] and row["coverage"] == 0 and team.calls == 2


def test_pooling_needs_discussion_and_enough_members():
    t = make_task(3)
    team = Team(t, thinker(share=True), full_info=False)
    assert vote(4)(team)["answer"] == t.distractor  # independent votes: nobody sees the others' facts
    team = Team(t, thinker(share=True), full_info=False)
    res = debate(4, 2)(team)  # round 2 sees everything shared in round 1
    assert res["answer"] == t.answer and lab.score(t, res, team)["coverage"] == 1.0
    team = Team(t, thinker(share=True), full_info=False)
    assert debate(2, 2)(team)["answer"] == t.distractor  # two members' facts are not enough


def test_evidence_controller_adds_members_when_support_is_narrow():
    t = make_task(7)
    shared_fact = t.shared()[0].id
    team = Team(t, FakeAdapter(reply=json.dumps({"answer": t.distractor, "confidence": 0.9,
                                                 "evidence": [shared_fact]})), full_info=False)
    res = evidence(team)
    assert evidence_state(res["transcript"][:2])["distinct"] == 1
    assert len(team.used) == t.holders and res["stop"] == "escalated"  # never enough support to stop


def test_dissenting_adversary_forces_escalation():
    t = make_task(4)
    wrong = next(c for c in "ABC" if c not in (t.answer,))
    honest = thinker(share=True)
    stubborn = thinker(stubborn=wrong)
    adapter = FakeAdapter(reply=lambda s, m: (stubborn if "never agree" in s.lower() else honest).reply(s, m))
    benign = Team(t, honest, full_info=True)
    consensus(benign)
    attacked = Team(t, adapter, full_info=True, adversary="budget_exhaustion")
    res = consensus(attacked)
    assert res["stop"] == "escalated" and attacked.calls > benign.calls  # compute amplification
    assert res["answer"] == t.answer  # the vote of all members still outvotes one adversary


def test_adversary_prompts_and_injection():
    t = make_task(5)
    team = Team(t, FakeAdapter(), full_info=False, adversary="premature_stop", adv_pos=1)
    sys1 = team.system(1)
    assert t.distractor in sys1 and all(f.id in sys1 for f in t.private(1))
    assert "secret goal" not in team.system(0)
    inj = Team(t, FakeAdapter(), full_info=False, adversary="injected", adv_pos=1)
    assert "strict deadline" in inj.prompt(1, [], False) and "strict deadline" not in inj.prompt(0, [], False)
    assert "secret goal" not in inj.system(1)  # an honest agent, compromised only by what it reads
    with pytest.raises(ValueError):
        Team(t, FakeAdapter(), True, adversary="bogus")


def test_budget_stops_before_overspending():
    t = make_task(6)
    team = Team(t, thinker(share=True), full_info=False, budget=1)
    team.tokens, team.calls = 500, 1  # pretend one call already cost 500 tokens
    assert evidence(team)["stop"] == "budget"


def test_study_runs_end_to_end_and_reports_effects():
    out = lab.run_study(conditions=("E0", "E1"), names=("single", "consensus", "evidence"), n_tasks=3,
                        adapter=thinker(share=True))
    s = out["summary"]
    assert s["trials"] == 18 and s["table"]["E0|consensus"]["accuracy"] == 1.0
    assert s["table"]["E1|single"]["accuracy"] == 0.0  # one member alone is always misled
    assert s["effects"]["single"]["info_effect"] == -1.0
    assert "| E1 split info, honest | **evidence** |" in out["report"]
    assert lab.list_runs()[0]["id"] == out["id"] and lab.load_run(out["id"])["summary"]["trials"] == 18
    with pytest.raises(ValueError):
        lab.load_run("../../x")


def test_every_lab_call_is_traced_with_its_exact_prompt():
    out = lab.run_study(conditions=("E1",), names=("consensus",), n_tasks=1, adapter=thinker(share=False))
    trial = out["results"][0]["trial"]
    steps = lab.trial_steps(out["id"], trial)
    assert len(steps) == out["results"][0]["calls"] == 2
    assert "Decision rule" in steps[0]["io"]["system"] and "Your facts" in steps[0]["io"]["messages"][0]["content"]
    with pytest.raises(ValueError):
        lab.trial_steps(out["id"], "../x")
