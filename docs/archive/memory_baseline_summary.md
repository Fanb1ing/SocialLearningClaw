# Memory Agent Baseline Summary

**Date:** 2026-06-28; TGM added 2026-07-07  
**Model:** `anthropic/claude-opus-4.8` via OpenRouter  
**Baselines:** Reflexion, ExPeL, A-MEM, TGM  

This note summarizes the current memory-agent baseline runs on ContextMATH, IntPhys2, and ARC-AGI-3.

## Setup

### ContextMATH

- Splits: `aime_2024_sg`, `aime_2024_cs`, `aime_2025_sg`, `aime_2025_cs`
- Sample size: 10 problems per split
- Metric: exact numeric accuracy
- Retry policy: Reflexion uses up to 3 attempts per problem; ExPeL, A-MEM, and TGM use one attempt per problem
- Result files:
  - `SelectBenchmark/results/contextmath/reflexion_claude-opus-4.8.json`
  - `SelectBenchmark/results/contextmath/expel_claude-opus-4.8.json`
  - `SelectBenchmark/results/contextmath/amem_claude-opus-4.8.json`
  - `SelectBenchmark/results/contextmath/tgm_claude-opus-4.8.json`

### IntPhys2

- Data: Debug split, Solidity condition, 20 videos
- Metric: binary classification accuracy
- Result files:
  - `SelectBenchmark/results/intphys2/reflexion_claude-opus-4.8.json`
  - `SelectBenchmark/results/intphys2/expel_claude-opus-4.8.json`
  - `SelectBenchmark/results/intphys2/amem_claude-opus-4.8.json`
  - `SelectBenchmark/results/intphys2/tgm_claude-opus-4.8.json`

### ARC-AGI-3

- Games: `cd82-fb555c5d`, `sk48-d8078629`, `tu93-0768757b`
- Max steps per level: 50
- Memory update interval: 10 steps
- Retry policy: no retry after GAME_OVER/TIMEOUT
- Memory policy: online memory update inside the same ongoing level attempt; refreshed memory is injected into later steps
- Command:

```bash
.venv/bin/python -u scripts/run_arc_memory_baseline.py \
  --all-games \
  --baseline all \
  --max-steps 50 \
  --memory-update-interval 10 \
  --runs-dir runs
```

ARC run directories:

| Game | Baseline | Run directory |
|---|---|---|
| cd82 | Reflexion | `runs/arc_memory_reflexion/claude-opus-4.8/20260628_202156` |
| cd82 | ExPeL | `runs/arc_memory_expel/claude-opus-4.8/20260628_202738` |
| cd82 | A-MEM | `runs/arc_memory_amem/claude-opus-4.8/20260628_203314` |
| cd82 | TGM | `runs/arc_memory_tgm/claude-opus-4.8/20260707_141336` |
| sk48 | Reflexion | `runs/arc_memory_reflexion/claude-opus-4.8/20260628_204035` |
| sk48 | ExPeL | `runs/arc_memory_expel/claude-opus-4.8/20260628_204626` |
| sk48 | A-MEM | `runs/arc_memory_amem/claude-opus-4.8/20260628_205201` |
| sk48 | TGM | `runs/arc_memory_tgm/claude-opus-4.8/20260707_144012` |
| tu93 | Reflexion | `runs/arc_memory_reflexion/claude-opus-4.8/20260628_205925` |
| tu93 | ExPeL | `runs/arc_memory_expel/claude-opus-4.8/20260628_210500` |
| tu93 | A-MEM | `runs/arc_memory_amem/claude-opus-4.8/20260628_211003` |
| tu93 | TGM | `runs/arc_memory_tgm/claude-opus-4.8/20260707_144806` |

## Results

### ContextMATH

For any method with per-problem retry, report both:

- **First attempt:** accuracy before retry, using the first direct answer.
- **Final:** accuracy after the method's retry loop finishes.

| Baseline | Retry? | AIME 2024 SG | AIME 2024 CS | AIME 2025 SG | AIME 2025 CS | Mean |
|---|---|---:|---:|---:|---:|---:|
| Vanilla Claude Opus 4.8 | No | 80.0 | 60.0 | 70.0 | 70.0 | 70.0 |
| Reflexion first attempt | Yes | 90.0 | 100.0 | 80.0 | 80.0 | 87.5 |
| Reflexion final after retry | Yes | 100.0 | 100.0 | 90.0 | 90.0 | 95.0 |
| ExPeL | No | 90.0 | 90.0 | 70.0 | 70.0 | 80.0 |
| A-MEM | No | 90.0 | 100.0 | 90.0 | 80.0 | 90.0 |
| TGM | No | 90.0 | 90.0 | 70.0 | 70.0 | 80.0 |

Reflexion is strongest on final accuracy, but part of that gain comes from its retry budget. Its first-attempt accuracy is 87.5%, while its final-after-retry accuracy is 95.0%. A-MEM is the strongest no-retry memory baseline on this sample. TGM matches ExPeL on this 40-problem sample while preserving explicit query/path/meta-cognition graph structure. But, A-MEM notes include post-evaluation context with gold answer。

### IntPhys2

| Baseline | Accuracy | Correct / Total |
|---|---:|---:|
| Vanilla Claude Opus 4.8 | 70.0 | 14 / 20 |
| Reflexion | 65.0 | 13 / 20 |
| ExPeL | 60.0 | 12 / 20 |
| A-MEM | 55.0 | 11 / 20 |
| TGM | 65.0 | 13 / 20 |

Memory baselines did not improve IntPhys2 over vanilla in this setting. TGM matches Reflexion and learns one consolidated solidity strategy, but the dominant failure remains over-predicting `impossible` for physically plausible videos.

### ARC-AGI-3

| Game | Baseline | Outcome | Wins / Required | Steps | Main action pattern | Memory size |
|---|---|---|---:|---:|---|---:|
| cd82 | Reflexion | TIMEOUT | 0 / 6 | 50 | ACTION4-heavy, broad exploration | 6 reflections |
| cd82 | ExPeL | TIMEOUT | 0 / 6 | 50 | ACTION1/ACTION4-heavy, one click | 9 insights / 6 experiences |
| cd82 | A-MEM | TIMEOUT | 0 / 6 | 50 | ACTION1/ACTION4-heavy | 6 notes |
| cd82 | TGM | TIMEOUT | 0 / 6 | 50 | ACTION2/ACTION4-heavy, some ACTION5/ACTION6 | 1 meta node |
| sk48 | Reflexion | TIMEOUT | 0 / 8 | 50 | ACTION4-heavy, occasional ACTION6/ACTION7 | 6 reflections |
| sk48 | ExPeL | TIMEOUT | 0 / 8 | 50 | ACTION4-heavy, occasional ACTION6/ACTION7 | 10 insights / 6 experiences |
| sk48 | A-MEM | TIMEOUT | 0 / 8 | 50 | ACTION4-heavy, occasional ACTION6/ACTION7 | 6 notes |
| sk48 | TGM | TIMEOUT | 0 / 8 | 50 | ACTION4-heavy, occasional ACTION6/ACTION7 | 1 meta node |
| tu93 | Reflexion | GAME_OVER | 0 / 9 | 50 | ACTION2-heavy | 5 reflections |
| tu93 | ExPeL | GAME_OVER | 0 / 9 | 50 | ACTION2-heavy | 10 insights / 5 experiences |
| tu93 | A-MEM | GAME_OVER | 0 / 9 | 50 | ACTION2-heavy | 5 notes |
| tu93 | TGM | GAME_OVER | 0 / 9 | 50 | ACTION2-heavy | 1 meta node |

No memory baseline solved Level 1 within 50 steps on any of the three ARC-AGI-3 games. TGM also fails all three, despite preserving graph structure over query/path/meta-cognition nodes.

Additional trajectory observations:

| Game | Reflexion no-change steps | ExPeL no-change steps | A-MEM no-change steps | TGM no-change steps |
|---|---:|---:|---:|---:|
| cd82 | 13 / 50 | 11 / 50 | 11 / 50 | 12 / 50 |
| sk48 | 8 / 50 | 13 / 50 | 12 / 50 | 16 / 50 |
| tu93 | 0 / 50 | 0 / 50 | 0 / 50 | 0 / 50 |

For `tu93`, every move changes the grid, but all four memory methods still reach GAME_OVER. This suggests that detecting no-effect actions is not enough; the agent needs a structured model of goal progress and hazards.

## Takeaways

1. ContextMATH is friendly to text-memory baselines. Reflexion and A-MEM produce clear gains over vanilla Claude Opus 4.8 on the 40-problem sample.
2. IntPhys2 is not helped by these memory baselines. The problem is likely perceptual/temporal grounding rather than reusable textual memory.
3. ARC-AGI-3 remains hard for generic text/graph-memory baselines. The memories mostly become high-level reminders such as "avoid repetitive action spam" or "map action effects", but they do not discover the game-specific operational rules.
4. Compared with schema-based ARC work, these memory baselines lack explicit object/action-effect grounding. They store verbal summaries, but do not build a structured state/action model that can reliably constrain the next action.
5. For ARC-AGI-3, the online-memory update implementation is now aligned with the desired protocol: memory is updated during the same game attempt and no retry is granted after failure.

## Recommended Use In Paper/Report

- Use ContextMATH to show that memory baselines can be strong when the task is text/math reasoning and feedback is easy to verbalize.
- Use IntPhys2 to show that generic memory is not a universal fix, especially when failures depend on visual-temporal physical perception.
- Use ARC-AGI-3 as the key contrast: generic memory agents update language-level advice but still fail to form actionable, grounded schema. This supports the motivation for SocialLearningClaw's structured schema/action-effect learning.
