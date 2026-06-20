# Paper Notes

**Purpose:** a durable, timestamped record of PreferenceLayer's research results
and their publication potential, captured so a future write-up does not have to
reconstruct the state of the work from scratch.

**Created:** 2026-06-20T07:48:54Z
**Repo state at creation:** `HEAD = 6ed019bb` (2026-06-20T06:14:09Z), branch
`claude/amazing-hamilton-5od112`.

This directory is **notes for a future paper**, not a paper draft. Nothing here is
intended for submission as-is. Each file is timestamped at the top; results are
anchored to the commit that introduced them so the record stays reproducible even
as the codebase moves on.

## Contents

| File | What it records |
|------|-----------------|
| [`2026-06-20-paper-worthiness-assessment.md`](2026-06-20-paper-worthiness-assessment.md) | Honest verdict on whether the current results support a paper, and why. |
| [`2026-06-20-results-ledger.md`](2026-06-20-results-ledger.md) | Every experimental result to date, with headline numbers, landed date, commit, reproduction command, and raw-metrics path. |
| [`2026-06-20-publication-roadmap.md`](2026-06-20-publication-roadmap.md) | The two viable paper framings and the concrete work each needs before it is submittable. |

## How to keep this current

When a new result lands, append a dated entry to the results ledger (do not
rewrite history — add, with a timestamp and commit). Revisit the assessment and
roadmap when a result would change the verdict (e.g. the QIL text-extraction
pipeline closes the real-data featurization gap).

## One-line state of play (2026-06-20)

The modeling and protocol-composition results are strong **on synthetic
benchmarks with planted ground truth**; the one real-data check (Amazon Reviews
2023) shows the preference-graph advantage does **not** replicate under coarse
features, locating the binding constraint at attribute/quality extraction. The
most paper-ready contribution today is the *honest negative* ("featurization, not
ranking-model expressiveness, is the bottleneck"); the strongest future paper
requires closing that gap with real text-derived features.
</content>
</invoke>
