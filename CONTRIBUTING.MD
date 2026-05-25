# Contributing

PreferenceLayer is pre-prototype. The most valuable contributions right now are to the research foundation and protocol design, not code.

---

## Where to Start

### Phase 0 Research (Active)

The immediate open questions are:

1. **Preference graph topology** — Does a sparse DAG with context-conditional edge weights outperform flat preference vectors on cross-category recommendation tasks? If you want to run experiments against public e-commerce datasets (Amazon Reviews 2023), this is the highest-priority validation work.

2. **QIL extraction precision** — Can failure modes and use profiles be reliably extracted from unstructured public sources (Reddit, iFixit, forum threads) at >75% precision with a practical annotation budget? Contributions to the NLP pipeline and annotation schema are welcome.

3. **Protocol design** — PTP is in draft. If you have experience with W3C Verifiable Credentials, DIF Presentation Exchange, or MCP server implementation, review [`docs/protocol-spec.md`](docs/protocol-spec.md) and open an issue with feedback.

---

## Ground Rules

- Open an issue before opening a pull request for anything non-trivial. This is early-stage work; coordinate first.
- Protocol changes require discussion. Don't submit schema or API changes without prior issue thread.
- If you're running experiments, document your setup fully — dataset version, hyperparameters, evaluation metric. Results that can't be reproduced aren't useful.
- No dependencies on platform-specific APIs. The whole point is platform-agnostic infrastructure.

---

## Code Style

- Python: PEP 8, type hints required on public functions, docstrings for anything non-obvious.
- TypeScript (when applicable): strict mode, no `any`.
- Tests required for all protocol logic.

---

## Issues

Use GitHub Issues for bugs, research questions, and protocol feedback. Label accordingly:

- `research` — open empirical questions
- `protocol` — PTP or QIL API design
- `infra` — implementation and tooling
- `data` — dataset, annotation, ingestion pipeline
