"""QIL extraction: TF-IDF + multinomial logistic classifiers, hand-rolled.

This is the NLP extraction step of the QIL pipeline (``docs/architecture.md``
Component 2): turn an unstructured post into structured ``(use_profile,
failure_mode, signal_type)`` labels plus a confidence. The Phase 0 go/no-go gate
is precision on the **use-profile** head.

We deliberately avoid heavy dependencies (no scikit-learn, no transformer
download): a bag-of-words TF-IDF vectorizer feeds a softmax classifier trained by
gradient descent. This reuses the same logistic-regression-on-standardized-
features approach already proven in ``models/graph.py``; here it is the
multinomial (softmax) generalization. A classical model clearing the 70% bar is
the right Phase 0 signal — it shows the task is learnable from public text; the
production pipeline can later swap in a fine-tuned transformer for headroom.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from .corpus import Sample

_TOKEN_RE = re.compile(r"[a-z]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class TfidfVectorizer:
    """Minimal unigram+bigram TF-IDF vectorizer (fit on train, applied to test)."""

    def __init__(self, min_df: int = 2, use_bigrams: bool = True):
        self.min_df = min_df
        self.use_bigrams = use_bigrams
        self.vocab: dict[str, int] = {}
        self.idf: np.ndarray | None = None

    def _terms(self, text: str) -> list[str]:
        toks = _tokenize(text)
        if not self.use_bigrams:
            return toks
        bigrams = [f"{a}_{b}" for a, b in zip(toks, toks[1:])]
        return toks + bigrams

    def fit(self, texts: list[str]) -> "TfidfVectorizer":
        df: dict[str, int] = {}
        for t in texts:
            for term in set(self._terms(t)):
                df[term] = df.get(term, 0) + 1
        self.vocab = {term: i for i, term in enumerate(sorted(t for t, c in df.items() if c >= self.min_df))}
        n_docs = len(texts)
        idf = np.zeros(len(self.vocab))
        for term, i in self.vocab.items():
            idf[i] = np.log((1.0 + n_docs) / (1.0 + df[term])) + 1.0
        self.idf = idf
        return self

    def transform(self, texts: list[str]) -> np.ndarray:
        assert self.idf is not None, "call fit() first"
        X = np.zeros((len(texts), len(self.vocab)))
        for r, t in enumerate(texts):
            for term in self._terms(t):
                j = self.vocab.get(term)
                if j is not None:
                    X[r, j] += 1.0
        X *= self.idf
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return X / norms

    def fit_transform(self, texts: list[str]) -> np.ndarray:
        return self.fit(texts).transform(texts)


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class SoftmaxClassifier:
    """Multinomial logistic regression trained with full-batch gradient descent + L2."""

    def __init__(self, epochs: int = 600, lr: float = 0.6, l2: float = 1e-3, seed: int = 0):
        self.epochs = epochs
        self.lr = lr
        self.l2 = l2
        self.seed = seed
        self.classes_: list[str] = []
        self.W: np.ndarray | None = None  # (n_features+1, n_classes), last row = bias

    def fit(self, X: np.ndarray, y: list[str]) -> "SoftmaxClassifier":
        self.classes_ = sorted(set(y))
        cls_idx = {c: i for i, c in enumerate(self.classes_)}
        Y = np.zeros((len(y), len(self.classes_)))
        for r, label in enumerate(y):
            Y[r, cls_idx[label]] = 1.0
        Xb = np.hstack([X, np.ones((X.shape[0], 1))])
        rng = np.random.default_rng(self.seed)
        self.W = rng.normal(0.0, 0.01, size=(Xb.shape[1], len(self.classes_)))
        n = Xb.shape[0]
        for _ in range(self.epochs):
            P = _softmax(Xb @ self.W)
            grad = Xb.T @ (P - Y) / n
            grad[:-1] += self.l2 * self.W[:-1]  # regularize weights, not bias
            self.W -= self.lr * grad
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        assert self.W is not None, "call fit() first"
        Xb = np.hstack([X, np.ones((X.shape[0], 1))])
        return _softmax(Xb @ self.W)

    def predict(self, X: np.ndarray) -> list[str]:
        P = self.predict_proba(X)
        return [self.classes_[i] for i in P.argmax(axis=1)]


@dataclass
class ExtractedSignal:
    """Structured output of extraction over one post (predicted, not gold)."""

    product_id: str
    category: str
    use_profile: str
    signal_type: str
    failure_mode: str | None
    quality_dim: str | None
    signal_value: float
    confidence: float


# Source-reliability weight applied to model confidence (architecture.md: "model
# confidence x source reliability weight"). Constant here; a real pipeline keys
# this off the source (Notebookcheck > random forum post).
_SOURCE_RELIABILITY = 0.9


class QILExtractor:
    """Bundles the three classification heads used by the QIL ingestion pipeline."""

    def __init__(self, **clf_kwargs):
        self.vectorizer = TfidfVectorizer()
        self.use_profile_clf = SoftmaxClassifier(**clf_kwargs)
        self.signal_type_clf = SoftmaxClassifier(**clf_kwargs)

    def fit(self, samples: list[Sample]) -> "QILExtractor":
        X = self.vectorizer.fit_transform([s.text for s in samples])
        self.use_profile_clf.fit(X, [s.use_profile for s in samples])
        self.signal_type_clf.fit(X, [s.signal_type for s in samples])
        return self

    def predict_use_profiles(self, samples: list[Sample]) -> list[str]:
        X = self.vectorizer.transform([s.text for s in samples])
        return self.use_profile_clf.predict(X)

    def extract(self, samples: list[Sample]) -> list[ExtractedSignal]:
        """Run all heads over posts, producing structured signals for aggregation.

        ``failure_mode`` / ``quality_dim`` / ``signal_value`` are taken from the
        post's structured fields (a real pipeline parses these with a span model);
        the learned heads supply ``use_profile`` and ``signal_type``, which is
        where the feasibility risk lives.
        """
        X = self.vectorizer.transform([s.text for s in samples])
        up = self.use_profile_clf.predict(X)
        up_p = self.use_profile_clf.predict_proba(X).max(axis=1)
        st = self.signal_type_clf.predict(X)
        out: list[ExtractedSignal] = []
        for i, s in enumerate(samples):
            out.append(ExtractedSignal(
                product_id=s.product_id,
                category=s.category,
                use_profile=up[i],
                signal_type=st[i],
                failure_mode=s.failure_mode if st[i] == "failure" else None,
                quality_dim=s.quality_dim,
                signal_value=s.signal_value,
                confidence=float(up_p[i] * _SOURCE_RELIABILITY),
            ))
        return out
