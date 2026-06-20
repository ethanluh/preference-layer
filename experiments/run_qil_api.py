"""Run the QIL HTTP API locally (Work Stream B4).

Builds a QualityService from the controlled Phase 0 corpus (stand-in for the
nightly-refit posteriors) and serves /quality + /compare over HTTP.

    pip install 'preferencelayer[api]'
    python experiments/run_qil_api.py            # http://127.0.0.1:8000
    # then:
    curl -s localhost:8000/quality -H 'content-type: application/json' \
      -d '{"product_id":"laptop_model_00","use_profile":"gaming"}'
"""

from __future__ import annotations

import argparse

from preferencelayer.qil import (
    QILExtractor,
    QualityAggregator,
    QualityService,
    build_app,
    generate,
)


def build_service() -> QualityService:
    corpus = generate()
    extractor = QILExtractor().fit(corpus.train)
    signals = extractor.extract(corpus.train + corpus.test)
    return QualityService(QualityAggregator().fit(signals))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    import uvicorn

    app = build_app(build_service())
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
