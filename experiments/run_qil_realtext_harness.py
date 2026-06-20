"""B2: use-profile extraction precision harness on real text (runnable).

Measures macro precision of a use-profile classifier and maps it to the kickoff
decision bands (>=70% pass / 60-70% recoverable / <60% escalate).

    # SMOKE TEST (controlled corpus; result is NOT a real-text number):
    python experiments/run_qil_realtext_harness.py

    # REAL run, once an annotated corpus exists (raw text is .gitignore'd):
    python experiments/run_qil_realtext_harness.py --real-corpus /abs/path/annotated.jsonl
    # swap in the transformer fine-tune path (needs `transformers`+`torch`):
    python experiments/run_qil_realtext_harness.py --real-corpus /abs/path/annotated.jsonl --model transformer

Reproducibility: dataset version = controlled corpus seed 17 (smoke) or the named
JSONL file (real); metric = macro precision over the five use profiles; the
TF-IDF baseline hyperparameters live in `extract.py`, the transformer's in
`harness.TransformerClassifier.hyperparameters()`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from preferencelayer.qil import (
    TfidfBaselineClassifier,
    TransformerClassifier,
    load_controlled_smoke,
    load_real_corpus,
    measure,
)

_BANNER = {
    "pass": ">= 70% -> PROCEED to coverage (B4).",
    "recoverable": "60-70% -> assess whether MORE ANNOTATION recovers it.",
    "escalate": "< 60% -> automation story does NOT hold; ESCALATE before scaling.",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--real-corpus", type=str, default=None,
                    help="Absolute path to an annotated real-text JSONL corpus.")
    ap.add_argument("--model", choices=["tfidf", "transformer"], default="tfidf")
    ap.add_argument("--out", type=str, default="experiments/qil_realtext_harness_results.json")
    args = ap.parse_args()

    if args.real_corpus:
        split = load_real_corpus(args.real_corpus)
    else:
        split = load_controlled_smoke()
        print("!! SMOKE TEST on the CONTROLLED corpus -- this is NOT a real-text result.")
        print("!! Real-text precision is UNVERIFIED until --real-corpus points at an")
        print("!! annotated scraped corpus (~300 adjudicated samples).\n")

    classifier = TransformerClassifier() if args.model == "transformer" else TfidfBaselineClassifier()
    result = measure(classifier, split)

    print(f"model:            {result.model}")
    print(f"corpus:           {result.source} (real_text={result.is_real_text})")
    print(f"macro precision:  {result.macro_precision:.4f}  (baseline {result.baseline_precision:.4f})")
    print(f"micro / accuracy: {result.micro_precision:.4f}   n_test={result.n_test}")
    print(f"checkpoint band:  {result.band.upper()} -- {_BANNER[result.band]}")
    print(f"verified on REAL text: {result.verified_on_real_text}")
    if not result.is_real_text:
        print("\n(reminder: band/precision above are controlled-corpus; do not quote as real-text.)")

    out = Path(args.out)
    out.write_text(json.dumps(result.to_json(), indent=2) + "\n")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
