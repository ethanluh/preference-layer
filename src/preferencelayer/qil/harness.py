"""Real-text use-profile extraction harness (Work Stream B2).

The gate-behind-the-gate. Phase 0 reached 88.3% macro precision on a *controlled*
corpus (``docs/phase0-qil-results.md``); the real-data reality check
(``docs/phase1-amazon-realdata.md``) showed extraction quality is the binding
constraint. This module is the apparatus to measure use-profile precision on
**real scraped text** and to swap the Phase 0 TF-IDF baseline for a fine-tuned
transformer behind one interface.

What this is and is NOT
-----------------------
* IS: a runnable, model-agnostic measurement harness with the kickoff's decision
  bands (>=70% / 60-70% / <60%), reproducibility metadata, and an explicit slot
  to drop in a real annotated corpus.
* IS NOT: a verified real-text precision number. No paid annotation budget / real
  annotated corpus exists in this environment, so the harness runs as a SMOKE
  TEST against a held-out split of the controlled corpus. **70% on real text is
  UNVERIFIED until** ``load_real_corpus`` is pointed at an annotated real corpus
  (see its docstring). The smoke number is a controlled-corpus number, labeled as
  such -- do not quote it as the real-text result.

Metric: **macro precision** over the five use profiles (matches ``eval.py``'s
``GATE_PRECISION`` and the Phase 0 gate -- cannot be inflated by the majority
class).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Protocol

from .corpus import Corpus, Sample, generate
from .eval import GATE_PRECISION, ClassificationReport, evaluate
from .extract import QILExtractor
from .schema import USE_PROFILES

# Kickoff decision bands for use-profile precision on REAL text.
GATE_PASS = GATE_PRECISION          # >= 0.70  -> proceed to coverage (B4)
GATE_RECOVERABLE = 0.60             # 0.60-0.70 -> assess whether more annotation recovers it
# < 0.60 -> automation story doesn't hold; escalate before scaling coverage.


# --------------------------------------------------------------------------- #
# Model interface (TF-IDF baseline <-> transformer fine-tune, swappable)
# --------------------------------------------------------------------------- #

class UseProfileClassifier(Protocol):
    """Anything that learns use-profile labels from text and predicts them.

    Both the Phase 0 TF-IDF baseline and the transformer fine-tune path implement
    this, so the harness measures either without code changes.
    """

    name: str

    def fit(self, samples: list[Sample]) -> "UseProfileClassifier": ...

    def predict_use_profiles(self, samples: list[Sample]) -> list[str]: ...


@dataclass
class TfidfBaselineClassifier:
    """Phase 0 TF-IDF + softmax use-profile head, wrapped to the interface."""

    name: str = "tfidf_softmax_baseline"
    _extractor: QILExtractor | None = None
    clf_kwargs: dict = field(default_factory=dict)

    def fit(self, samples: list[Sample]) -> "TfidfBaselineClassifier":
        self._extractor = QILExtractor(**self.clf_kwargs).fit(samples)
        return self

    def predict_use_profiles(self, samples: list[Sample]) -> list[str]:
        assert self._extractor is not None, "call fit() first"
        return self._extractor.predict_use_profiles(samples)


@dataclass
class TransformerClassifier:
    """Fine-tuned-transformer use-profile head (HuggingFace). SCAFFOLD.

    The production headroom path: fine-tune a small encoder (e.g. ``distilbert-
    base-uncased``) for sequence classification over the five use profiles. The HF
    stack (``transformers``/``torch``) is an OPTIONAL dependency (the ``[b2]``
    extra); when it is absent ``fit`` raises a clear, actionable message and the
    harness falls back to ``TfidfBaselineClassifier``. When it is present, ``fit``
    runs a real ``Trainer`` fine-tune and ``predict_use_profiles`` serves it — the
    interface matches the baseline so the harness swaps models with one
    constructor change.

    Suggested hyperparameters (documented for reproducibility; tune on the real
    annotated set, not the controlled corpus): model=distilbert-base-uncased,
    epochs=4, lr=2e-5, batch=16, max_len=256, weight_decay=0.01, seed=17.
    """

    name: str = "distilbert_finetune"
    model_name: str = "distilbert-base-uncased"
    epochs: int = 4
    lr: float = 2e-5
    batch_size: int = 16
    max_len: int = 256
    weight_decay: float = 0.01
    seed: int = 17
    output_dir: str | None = None  # weights land here (default: a temp dir); .gitignore'd
    # Fitted state (populated by fit(); not constructor args).
    _model: object = field(default=None, repr=False, compare=False)
    _tokenizer: object = field(default=None, repr=False, compare=False)
    _labels: list[str] = field(default_factory=list, repr=False, compare=False)

    def hyperparameters(self) -> dict:
        return {
            "model_name": self.model_name, "epochs": self.epochs, "lr": self.lr,
            "batch_size": self.batch_size, "max_len": self.max_len,
            "weight_decay": self.weight_decay, "seed": self.seed,
        }

    def fit(self, samples: list[Sample]) -> "TransformerClassifier":  # pragma: no cover - needs HF
        try:
            import tempfile

            import torch
            from transformers import (
                AutoModelForSequenceClassification,
                AutoTokenizer,
                Trainer,
                TrainingArguments,
                set_seed,
            )
        except ImportError as exc:  # scaffold boundary
            raise NotImplementedError(
                "TransformerClassifier needs the optional HF stack "
                "(`pip install 'preferencelayer[b2]'` or `pip install transformers torch`) "
                "AND a real annotated corpus. This is the documented fine-tune path; "
                "until both are present, use TfidfBaselineClassifier. See harness.py docstring."
            ) from exc

        set_seed(self.seed)
        self._labels = list(USE_PROFILES)
        label2id = {lbl: i for i, lbl in enumerate(self._labels)}
        id2label = {i: lbl for lbl, i in label2id.items()}

        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        enc = tokenizer([s.text for s in samples], truncation=True,
                        padding=True, max_length=self.max_len)
        labels = [label2id[s.use_profile] for s in samples]

        class _Dataset(torch.utils.data.Dataset):
            def __len__(self) -> int:
                return len(labels)

            def __getitem__(self, i: int) -> dict:
                item = {k: torch.tensor(v[i]) for k, v in enc.items()}
                item["labels"] = torch.tensor(labels[i])
                return item

        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=len(self._labels),
            id2label=id2label, label2id=label2id,
        )
        args = TrainingArguments(
            output_dir=self.output_dir or tempfile.mkdtemp(prefix="qil_b2_"),
            num_train_epochs=self.epochs, learning_rate=self.lr,
            per_device_train_batch_size=self.batch_size, weight_decay=self.weight_decay,
            seed=self.seed, report_to=[], logging_strategy="no", save_strategy="no",
        )
        Trainer(model=model, args=args, train_dataset=_Dataset()).train()
        self._model, self._tokenizer = model, tokenizer
        return self

    def predict_use_profiles(self, samples: list[Sample]) -> list[str]:  # pragma: no cover - needs HF
        if self._model is None:
            raise NotImplementedError("call fit() first (needs the HF stack + a real corpus)")
        import torch

        enc = self._tokenizer([s.text for s in samples], truncation=True,
                              padding=True, max_length=self.max_len, return_tensors="pt")
        self._model.eval()
        with torch.no_grad():
            logits = self._model(**enc).logits
        return [self._labels[i] for i in logits.argmax(dim=-1).tolist()]


# --------------------------------------------------------------------------- #
# Corpus loading: real-text slot + controlled-corpus smoke fallback
# --------------------------------------------------------------------------- #

@dataclass
class LabeledSplit:
    train: list[Sample]
    test: list[Sample]
    source: str            # provenance label for reproducibility
    is_real_text: bool     # False => controlled corpus (smoke), True => real scraped+annotated


def load_controlled_smoke(seed: int = 17, n_train: int = 1400, n_test: int = 400) -> LabeledSplit:
    """Held-out split of the controlled Phase 0 corpus -- a SMOKE TEST only.

    Lets the harness run offline/in CI and proves the plumbing end to end. The
    precision it yields is a CONTROLLED-corpus number (~0.88), NOT a real-text
    result; ``is_real_text=False`` flags that everywhere downstream.
    """
    cp: Corpus = generate(n_train=n_train, n_test=n_test, seed=seed)
    return LabeledSplit(train=cp.train, test=cp.test,
                        source=f"controlled_corpus(seed={seed})", is_real_text=False)


def load_real_corpus(path: str | Path, test_frac: float = 0.3, seed: int = 17) -> LabeledSplit:
    """Load a REAL, human-annotated scraped corpus from a JSONL file.

    THE PLUG-IN SLOT. Per the kickoff: ~300 real samples, 2 annotators, adjudicated.
    Each JSONL line is an object with at least ``text`` and gold ``use_profile``
    (one of ``schema.USE_PROFILES``); optional ``category``/``product_id``. The
    file lives OUTSIDE version control (raw scraped data is .gitignore'd: see
    ``*.jsonl``) -- pass an absolute path produced by the B1 ingestion run plus
    annotation.

    Until such a file exists, this raises; the harness falls back to
    ``load_controlled_smoke`` and labels the result UNVERIFIED-on-real-text.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"real annotated corpus not found at {p}. This is the B2 plug-in slot; "
            "supply ~300 adjudicated real samples as JSONL (text + gold use_profile). "
            "Real-text precision is UNVERIFIED until this is provided."
        )
    import random

    rows = [json.loads(ln) for ln in p.read_text().splitlines() if ln.strip()]
    samples = [
        Sample(
            text=r["text"], category=r.get("category", "laptops"),
            product_id=r.get("product_id", "unknown"), use_profile=r["use_profile"],
            signal_type=r.get("signal_type", "performance"),
            failure_mode=r.get("failure_mode"), quality_dim=r.get("quality_dim"),
            signal_value=float(r.get("signal_value", 0.5)),
            label_confidence=float(r.get("label_confidence", 1.0)),
        )
        for r in rows
    ]
    rng = random.Random(seed)
    rng.shuffle(samples)
    n_test = max(1, int(len(samples) * test_frac))
    return LabeledSplit(train=samples[n_test:], test=samples[:n_test],
                        source=f"real_annotated:{p.name}", is_real_text=True)


# --------------------------------------------------------------------------- #
# Measurement + decision band
# --------------------------------------------------------------------------- #

def checkpoint_band(macro_precision: float) -> str:
    """Map precision to the kickoff's B2 decision bands."""
    if macro_precision >= GATE_PASS:
        return "pass"               # >= 0.70 -> proceed to coverage (B4)
    if macro_precision >= GATE_RECOVERABLE:
        return "recoverable"        # 0.60-0.70 -> assess more annotation
    return "escalate"               # < 0.60 -> automation story doesn't hold


@dataclass
class HarnessResult:
    model: str
    source: str
    is_real_text: bool
    macro_precision: float
    micro_precision: float
    baseline_precision: float
    band: str
    n_test: int
    verified_on_real_text: bool
    report: ClassificationReport

    def to_json(self) -> dict:
        d = {k: v for k, v in asdict(self).items() if k != "report"}
        d["per_class"] = {m.label: {"precision": m.precision, "recall": m.recall,
                                    "f1": m.f1, "support": m.support}
                          for m in self.report.per_class}
        return d


def measure(classifier: UseProfileClassifier, split: LabeledSplit) -> HarnessResult:
    """Fit on the train split, score macro precision on the test split, band it."""
    classifier.fit(split.train)
    y_true = [s.use_profile for s in split.test]
    y_pred = classifier.predict_use_profiles(split.test)
    report = evaluate(y_true, y_pred)
    return HarnessResult(
        model=getattr(classifier, "name", type(classifier).__name__),
        source=split.source,
        is_real_text=split.is_real_text,
        macro_precision=report.macro_precision,
        micro_precision=report.micro_precision,
        baseline_precision=report.baseline_precision,
        band=checkpoint_band(report.macro_precision),
        n_test=len(split.test),
        # The headline claim (">=70% on real text") is only VERIFIED when the
        # split is real text AND it cleared the gate.
        verified_on_real_text=bool(split.is_real_text and report.macro_precision >= GATE_PASS),
        report=report,
    )
