"""QIL daily ingestion job -- runnable demo entrypoint (Work Stream B1).

Runs the full pipeline (connectors -> normalize -> extract -> product_signal)
against the committed offline fixtures, so the wiring is exercisable without API
keys. In production this is the cron entrypoint; swap FixtureConnector for the
live Reddit/iFixit/Notebookcheck connectors (plug keys into
``connectors._LiveConnector._fetch_pages``) and InMemorySink for PostgresSink.

    python experiments/run_qil_ingest.py
"""

from __future__ import annotations

from pathlib import Path

from preferencelayer.qil import QILExtractor, generate
from preferencelayer.qil.ingest import (
    CanonicalProduct,
    FixtureConnector,
    InMemorySink,
    ProductRegistry,
    run_daily,
    schema_sql,
)

FIXTURES = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "ingest"


def build_registry() -> ProductRegistry:
    """A tiny canonical model list. Production loads this from the model catalog."""
    return ProductRegistry(threshold=0.5).add(
        CanonicalProduct("lenovo-thinkpad-x1-carbon-gen12", "laptops",
                         "Lenovo ThinkPad X1 Carbon Gen 12",
                         aliases=("thinkpad x1 carbon gen 12", "x1c g12"))
    ).add(
        CanonicalProduct("dell-xps-15-9530", "laptops", "Dell XPS 15 9530",
                         aliases=("dell xps 15 9530", "xps 15"))
    )


def main() -> None:
    print("=== QIL ingestion (SCAFFOLD; running against offline fixtures) ===\n")
    print("product_signal DDL (first lines):")
    for line in schema_sql().splitlines()[:6]:
        print("  " + line)
    print()

    # Train the extractor on the controlled Phase 0 corpus (stand-in for the
    # production fine-tuned model -- see Work Stream B2 for real-text precision).
    corpus = generate()
    extractor = QILExtractor().fit(corpus.train)

    registry = build_registry()
    connectors = [
        FixtureConnector("laptops", FIXTURES / "reddit_laptops.json", source_type="reddit"),
    ]
    sink = InMemorySink()
    stats = run_daily(connectors, registry, extractor, sink)

    print(f"fetched={stats.fetched} matched={stats.matched} "
          f"unmatched={stats.unmatched} written={stats.written}\n")
    for row in sink.rows:
        print(f"  {row.product_id:35s} use_profile={row.use_profile:12s} "
              f"signal_type={row.signal_type:11s} conf={row.model_confidence:.2f}")
    print("\nNOTE: no raw scraped text is persisted -- only normalized product_signal rows.")


if __name__ == "__main__":
    main()
