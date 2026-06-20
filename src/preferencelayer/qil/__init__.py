"""Quality Intelligence Layer (QIL): use-profile-conditioned product quality.

Phase 0 feasibility prototype for Claim 2 — that use-profile-conditioned quality
signals are extractable from public unstructured text at >= 70% precision. The
pipeline is: labeled corpus -> TF-IDF + softmax extraction -> Bayesian
aggregation -> ``/quality`` & ``/compare`` query service, with a QIL MCP server.
"""

from . import corpus, schema
from .aggregate import FailureRatePosterior, QualityAggregator, QualityPosterior
from .corpus import Corpus, Sample, generate
from .eval import GATE_PRECISION, ClassificationReport, evaluate
from .extract import ExtractedSignal, QILExtractor, SoftmaxClassifier, TfidfVectorizer
from .gp import GPHyperparams, GPPosterior, fit_gp_posterior
from .harness import (
    GATE_PASS,
    GATE_RECOVERABLE,
    HarnessResult,
    LabeledSplit,
    TfidfBaselineClassifier,
    TransformerClassifier,
    UseProfileClassifier,
    checkpoint_band,
    load_controlled_smoke,
    load_real_corpus,
    measure,
)
from .annotate import (
    AdjudicationReport,
    adjudicate,
    cohen_kappa,
    export_for_annotation,
    stable_id,
)
from .cli import (
    b2_main,
    build_demo_registry,
    ingest_main,
    refit_main,
    run_ingest,
    seconds_until_next_run,
)
from .http_api import build_app
from .mcp_server import QIL_TOOLS, QILToolHandler
from .query import QualityService
from .refit import (
    InMemoryPosteriorSink,
    PostgresPosteriorSink,
    PosteriorSink,
    QualityPosteriorRow,
    run_nightly_refit,
)

__all__ = [
    "schema",
    "corpus",
    "generate",
    "Corpus",
    "Sample",
    "QILExtractor",
    "TfidfVectorizer",
    "SoftmaxClassifier",
    "ExtractedSignal",
    "QualityAggregator",
    "QualityPosterior",
    "FailureRatePosterior",
    "QualityService",
    "QILToolHandler",
    "QIL_TOOLS",
    "build_app",
    "evaluate",
    "ClassificationReport",
    "GATE_PRECISION",
    "fit_gp_posterior",
    "GPHyperparams",
    "GPPosterior",
    "run_nightly_refit",
    "QualityPosteriorRow",
    "PosteriorSink",
    "InMemoryPosteriorSink",
    "PostgresPosteriorSink",
    "UseProfileClassifier",
    "TfidfBaselineClassifier",
    "TransformerClassifier",
    "LabeledSplit",
    "HarnessResult",
    "load_controlled_smoke",
    "load_real_corpus",
    "checkpoint_band",
    "measure",
    "GATE_PASS",
    "GATE_RECOVERABLE",
    "refit_main",
    "ingest_main",
    "b2_main",
    "run_ingest",
    "build_demo_registry",
    "seconds_until_next_run",
    "export_for_annotation",
    "adjudicate",
    "cohen_kappa",
    "stable_id",
    "AdjudicationReport",
]
