-- QIL ingestion target schema (PostgreSQL).
--
-- This is the authoritative DDL for the QIL database, mirroring
-- docs/architecture.md ("Component 2: QIL -> Data Model"). The ingestion
-- pipeline (src/preferencelayer/qil/ingest) lands normalized records in
-- product_signal; the nightly Bayesian refit (Work Stream B3) writes
-- quality_posterior. Both tables hold ONLY product + use-profile signals --
-- never any user identifier (invariant from architecture.md "QIL privacy").

CREATE TABLE IF NOT EXISTS product_signal (
  id               BIGSERIAL PRIMARY KEY,
  product_id       TEXT NOT NULL,         -- canonical id (see ingest.normalize)
  model_normalized TEXT NOT NULL,         -- normalized model string used to match
  category         TEXT NOT NULL,         -- 'laptops' | 'keyboards' (Phase 1 scope)
  failure_mode     TEXT,
  quality_dim      TEXT,
  use_profile      TEXT NOT NULL,         -- how the product is used (never a user id)
  signal_type      TEXT NOT NULL,         -- 'failure' | 'performance' | 'comparison'
  signal_value     FLOAT,                 -- normalized quality score if quantifiable
  source_url       TEXT,
  source_type      TEXT,                  -- 'reddit' | 'ifixit' | 'notebookcheck' | 'return_data'
  content_hash     TEXT NOT NULL,         -- dedup key (source_type + stable post id/url + body)
  extracted_at     TIMESTAMPTZ,
  model_confidence FLOAT,
  upvote_count     INT DEFAULT 0,
  -- A signal is uniquely identified by where it came from and its content. Re-runs
  -- of the daily job are idempotent: ON CONFLICT DO NOTHING on this constraint.
  CONSTRAINT uq_product_signal_dedup UNIQUE (source_type, content_hash)
);

CREATE INDEX IF NOT EXISTS ix_product_signal_lookup
  ON product_signal (product_id, use_profile, quality_dim);
CREATE INDEX IF NOT EXISTS ix_product_signal_category
  ON product_signal (category);

-- Posterior parameters (refit nightly by Work Stream B3). Parameters ONLY -- no
-- raw observations -- so the served table stays small and carries no PII.
CREATE TABLE IF NOT EXISTS quality_posterior (
  product_id      TEXT NOT NULL,
  use_profile     TEXT NOT NULL,
  quality_dim     TEXT NOT NULL,
  posterior_mean  FLOAT NOT NULL,
  posterior_std   FLOAT NOT NULL,
  credible_lo_90  FLOAT NOT NULL,
  credible_hi_90  FLOAT NOT NULL,
  evidence_count  INT NOT NULL,
  freshness_score FLOAT NOT NULL,         -- decays with signal age
  last_refit      TIMESTAMPTZ NOT NULL,
  PRIMARY KEY (product_id, use_profile, quality_dim)
);
