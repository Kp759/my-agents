# Literature check: compute-efficient multi-agent scaling

*2026-10-23. arXiv search (about 15 queries) plus abstracts of the 20 closest papers. Every ID
below was retrieved from the arXiv API. This was a one-hour scan, not a systematic review:
read the five starred papers in full before committing to the gap.*

## The question we started with

> Under a fixed compute budget, when should a system add agents, which agents (models) should
> it use, and when should it stop?

## What is already done (don't redo it)

### Scaling curves under matched compute
| Paper | Finding | For us |
|---|---|---|
| ★ Kim et al., *Towards a Science of Scaling Agent Systems* ([2512.08296](https://arxiv.org/abs/2512.08296), Dec 2025) | 260 configurations, 5 architectures, 3 model families, compute standardised. Coordination gives diminishing returns once the single agent is strong ("capability saturation"); tool-heavy tasks pay a multi-agent overhead. | The scaling-curves paper exists. Cite it and build on it. |
| Fu et al., *Do More Agents Help? Accuracy and Cost in LLM Multi-Agent Workflows* ([2606.05670](https://arxiv.org/abs/2606.05670), Jun 2026) | Six MAS workflows vs one single-agent baseline on 10 benchmarks; most MAS trail the single agent, and debate uses 1.4× the tokens. | Single-agent baselines are strong. |
| Li et al., *More Agents Is All You Need* ([2402.05120](https://arxiv.org/abs/2402.05120), 2024) | Sampling-and-voting scales with agent count. | The fixed-N baseline. |
| Alavi et al., *More Agents Improve Math … but Adversarial Robustness Gap Persists* ([2511.07112](https://arxiv.org/abs/2511.07112), Nov 2025) | Gains are largest from 1 to 5 agents and saturate around 10; noisy inputs still hurt. | Expect saturation; robustness is not solved by N. |

### Debate vs voting
| Paper | Finding |
|---|---|
| ★ Choi et al., *Debate or Vote* ([2508.17536](https://arxiv.org/abs/2508.17536), Aug 2025) | Majority voting explains most of debate's gains; debate alone is a martingale (no expected improvement). |
| Zhang et al., *Stop Overvaluing Multi-Agent Debate* ([2502.08788](https://arxiv.org/abs/2502.08788), Feb 2025) | Debate often loses to chain-of-thought and self-consistency even with more compute; heterogeneous models help. |
| *When and Why Does Multi-Agent Debate Fail* ([2510.20963](https://arxiv.org/abs/2510.20963), Oct 2025) | "Debate hacking": competitive debate becomes cheap talk; consensus-seeking debate filters out informative disagreement. |

### Adaptive compute: how many agents, which models, when to stop
| Paper | What it adapts | Signal |
|---|---|---|
| ★ HCP-MAD ([2604.09679](https://arxiv.org/abs/2604.09679), Apr 2026) | Two heterogeneous agents; stop if they agree, else debate, else escalate to a larger vote | **Consensus** |
| iMAD ([2511.11306](https://arxiv.org/abs/2511.11306), Nov 2025) | Trigger debate only when it is likely to fix a wrong answer | Self-critique features + classifier |
| ATOM ([2605.26178](https://arxiv.org/abs/2605.26178), May 2026) | Agent count (budget-controllable graph, trained with RL) | Estimated query difficulty |
| Agent-REINFORCE ([2511.00086](https://arxiv.org/abs/2511.00086), Oct 2025) | Model mix and topology under a fixed budget | Search |
| AgentTTS ([2508.00890](https://arxiv.org/abs/2508.00890), Jul 2025) | Model and budget per subtask | Search |
| Zhai et al. ([2604.14853](https://arxiv.org/abs/2604.14853), Apr 2026) | Compute per input (single model), constrained optimisation | Input features |
| ★ ECMR ([2610.17869](https://arxiv.org/abs/2610.17869), Oct 2026) | Propose / challenge / verify / arbitrate / stop | Evidence diversity and independence (against "semantic false consensus") |
| Model routing and cascades: RouteLLM ([2406.18665](https://arxiv.org/abs/2406.18665)), cascades ([2207.10342](https://arxiv.org/abs/2207.10342), [2404.10136](https://arxiv.org/abs/2404.10136)) | Which single model answers | Learned router / uncertainty |

**So "which agent and how many, adaptively, under a budget" has methods already.** A new
controller of that kind would be incremental.

### Distributed information and group failures
| Paper | Finding |
|---|---|
| ★ Li et al., *HiddenBench* ([2505.11556](https://arxiv.org/abs/2505.11556), May 2025) | With evidence split across agents, MAS reach 30% vs 81% for one agent with all the evidence; premature convergence on shared evidence; **worse as groups grow**. |
| Mansuri et al., *Discussion gains less when dissent is withheld* ([2609.38324](https://arxiv.org/abs/2609.38324), Sep 2026) | Discussion only fixes a wrong majority if the dissent-withholding rate is below a critical value. |
| Consilience ([2608.20564](https://arxiv.org/abs/2608.20564), Aug 2026) | Calibrated control of *who speaks and how* in hidden-profile tasks (communication, not compute). |
| Conformity: [2501.13381](https://arxiv.org/abs/2501.13381), [2505.21588](https://arxiv.org/abs/2505.21588), [2609.33495](https://arxiv.org/abs/2609.33495) | Agents conform to peers, and more to "their own" model family. |

### Malicious agents and Byzantine robustness (aggregation, not compute)
- Detection: *Who's the Mole?* ([2507.04724](https://arxiv.org/abs/2507.04724)), SentinelNet ([2510.16219](https://arxiv.org/abs/2510.16219)),
  graph anomaly detection ([2512.18733](https://arxiv.org/abs/2512.18733)); spread: Prompt Infection ([2410.07283](https://arxiv.org/abs/2410.07283)).
- Byzantine-robust consensus: weighted BFT for multi-LLM networks ([2505.05103](https://arxiv.org/abs/2505.05103)),
  *Rethinking the Reliability of Multi-agent System: A Perspective from Byzantine Fault Tolerance* ([2511.10400](https://arxiv.org/abs/2511.10400)),
  *Resilient Consensus in Agentic AI* ([2606.15024](https://arxiv.org/abs/2606.15024)).
- Persuasion: *When Persuasion Overrides Truth in Multi-Agent LLM Debates* ([2504.00374](https://arxiv.org/abs/2504.00374)),
  *Learning to Persuade Exposes How Easily LLMs Abandon Correct Beliefs* ([2608.11624](https://arxiv.org/abs/2608.11624)).

These ask *how agents reach a robust answer despite bad peers*. We ask a different layer: *how a
budget-limited controller decides whether to spend more when the agents producing its stop signal may
mislead it*. The controller itself is the attack surface.

**Not verified** (cited by an outside summary, not found in our searches; find the source before citing):
a method called "SAC" for Byzantine-robust message filtering, and a Nature paper reporting that one
adversarial persuader reduces group accuracy and that more agents or rounds don't fix it.

## The gap

Stated carefully (ECMR already targets "semantic false consensus", so we don't claim nobody has
studied false consensus):

> Existing adaptive deliberation controllers have been evaluated mainly with cooperative,
> non-strategic agents. Their stopping and compute-allocation policies have not been systematically
> stress-tested when the consensus signal is corrupted by latent information asymmetry or by
> adversarial agents.

ECMR is the closest paper: its controller tracks concentration, support diversity and independence,
contradictions and budget, but it is evaluated on MedQA, CaseHOLD and FinQA with frozen cooperative
roles, with no distributed private information and no adversarial or injected members. Read it first.

Every adaptive controller above (HCP-MAD, iMAD, ATOM, ECMR) is evaluated with **honest agents
and complete information**. They decide how much to spend from signals the agents themselves
produce: agreement, confidence, disagreement. Those signals are exactly what fails in two
settings the rest of the literature documents:

1. **Distributed information** (HiddenBench): honest agents converge early on shared evidence.
   A consensus-based stopping rule then stops confidently and wrong, and stops sooner as groups grow.
2. **One adversarial or injected agent** (Prompt Infection, our `injection` suite): it can
   *fake agreement* (early wrong stop) or *fake persistent disagreement* (force escalation and burn
   the budget, a compute denial-of-service).

We found no paper that evaluates adaptive compute controllers under either condition at matched
compute. (The token-economics survey [2605.09104](https://arxiv.org/abs/2605.09104) names security as a
dimension of token cost but doesn't run this experiment.)

## Research questions

- **RQ1, reliability:** how robust are adaptive compute controllers to latent information asymmetry?
- **RQ2, security:** can one adversarial agent manipulate early stopping or escalation, to cause wrong
  answers or to exhaust compute?
- **RQ3, interaction:** is manipulation more effective when information is already split?
- **RQ4, mitigation:** can evidence-coverage- and trust-aware stopping keep the efficiency of adaptive
  allocation while cutting premature stops and compute amplification?

Order matters: **first show that existing controllers break (RQ1–3), then design the fix from the traces
(RQ4).** "We found a failure mode of adaptive multi-agent scaling" is the contribution; a better
controller is the follow-up.

## Proposed study

**Working title:** *Gaming the Budget: Adaptive Compute Allocation in Agent Teams under Distributed
Information and Adversarial Agents*

**Hypotheses**
- H1: Consensus-triggered controllers (HCP-MAD-style) save compute on clean tasks but lose their
  accuracy advantage over fixed-N voting under distributed information, and the loss grows with team size.
- H2: One adversarial agent can raise a consensus-triggered controller's spend by a large factor
  (forced escalation), or lower its accuracy (forced early stop), more than it can harm fixed-N voting.
- H3: An evidence-independence signal (ECMR-style) recovers part of H1's loss but stays gameable by H2.
- H4 (defence): a controller that bounds each agent's influence on spending decisions, and requires
  independent evidence before stopping, keeps most of the savings with bounded worst-case spend.

**Conditions** (synthetic tasks with known answers, so data is unlimited and difficulty controlled)
- Clean: complete information, all agents honest.
- Distributed: each agent holds part of the evidence (hidden-profile style).
- Adversarial: one agent is instructed (or injected) to fake agreement / fake disagreement.

**Controllers, all at matched dollar budgets**: 1 agent; fixed-N vote (N = 2, 4, 8); fixed-N debate;
consensus-escalation (HCP-MAD-style); evidence-aware (ECMR-lite); our robust variant.

**Models**: Claude Haiku and Sonnet (subscription) plus local Qwen3 8B (free); heterogeneous mixes as a factor.

**Design**: a 2×2 of information (full / split) × agents (honest / one adversary): E0 full-honest,
E1 split-honest, E2 full-adversary, E3 split-adversary. This separates the effect of information
asymmetry, the effect of the adversary, and their interaction.

**Adversaries**: *premature stop* (fakes confident, "independently verified" agreement with the trap),
*budget exhaustion* (never agrees, to force escalation), *withholding* (agrees and hides its decisive
private facts: looks like genuine consensus, defeats outlier detection), and *injected* (an honest
agent compromised by instructions inside a fact it reads, the realistic route via prompt injection).

**Metrics** (controller-centred, not only accuracy):
- unsafe stop rate: P(wrong | controller chose to stop);
- premature stop rate: chose to stop before the surfaced evidence determined the answer;
- coverage at stop: share of the private facts surfaced before stopping (hidden-profile tasks make this exact);
- compute amplification: tokens with the adversary / tokens without it;
- interaction: (E3 − E1) − (E2 − E0) in accuracy;
- later: controller regret against an oracle that knows whether one more round would help, and budget waste.

**Baselines still to add**: iMAD-style (self-critique trigger) and ATOM-style (difficulty-based agent
count) simplified controllers; HiddenBench's structured-communication intervention; a Byzantine-robust
aggregation baseline. Reviewers will ask "why not just make consensus Byzantine-robust?", so test it.

**Implementation**: `src/my_agents/lab/` (`my-agents lab run`). Synthetic hidden-profile tasks
(`lab/tasks.py`) put shared facts in favour of a distractor; the right answer needs the private facts
of at least three members. First Qwen turn observed: with split information the agent cited only
shared facts and none of its own private ones (spontaneous withholding), as HiddenBench reports.

## Risks and checks

- One hour of arXiv search can miss work, so read the starred papers in full and check their related-work sections.
  Re-run this scan weekly (the Sunday career job can include it).
- Reimplementing HCP-MAD and ECMR exactly is costly; we use simplified versions and say so.
- Subscription rate limits cap trial counts: plan the sweep size before running, and use Qwen for breadth.

## Next steps

1. Read in full: Kim et al., Choi et al., HCP-MAD, ECMR, HiddenBench.
2. Build: N-agent pool with a hard budget, the three task conditions, the controllers above, and a sweep command.
3. Pilot on Qwen (free) to size effects, then run the matched-budget sweep with Claude.
