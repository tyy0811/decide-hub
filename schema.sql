-- decide-hub Postgres schema
-- Run on startup: psql -f schema.sql or via Docker entrypoint

CREATE TABLE IF NOT EXISTS outcomes (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL,
    action TEXT NOT NULL,
    reward REAL NOT NULL,
    policy_id TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS automation_runs (
    run_id TEXT PRIMARY KEY,
    status TEXT NOT NULL DEFAULT 'running',
    entities_processed INTEGER DEFAULT 0,
    entities_failed INTEGER DEFAULT 0,
    action_distribution JSONB DEFAULT '{}',
    started_at TIMESTAMPTZ DEFAULT NOW(),
    completed_at TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS automation_outcomes (
    id SERIAL PRIMARY KEY,
    run_id TEXT REFERENCES automation_runs(run_id),
    entity_id TEXT NOT NULL,
    enriched_fields JSONB DEFAULT '{}',
    action_taken TEXT NOT NULL,
    rule_matched TEXT,
    permission_result TEXT NOT NULL,
    processed_date DATE DEFAULT CURRENT_DATE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (entity_id, processed_date)
);

CREATE TABLE IF NOT EXISTS pending_approvals (
    id SERIAL PRIMARY KEY,
    entity_id TEXT NOT NULL,
    proposed_action TEXT NOT NULL,
    reason TEXT,
    status TEXT DEFAULT 'pending',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS failed_entities (
    id SERIAL PRIMARY KEY,
    entity_id TEXT NOT NULL,
    run_id TEXT REFERENCES automation_runs(run_id),
    error_type TEXT NOT NULL,
    error_message TEXT,
    retry_count INTEGER DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_outcomes_user_id ON outcomes(user_id);
CREATE INDEX IF NOT EXISTS idx_outcomes_policy_id ON outcomes(policy_id);
CREATE INDEX IF NOT EXISTS idx_automation_outcomes_run_id ON automation_outcomes(run_id);
CREATE INDEX IF NOT EXISTS idx_automation_outcomes_entity_id ON automation_outcomes(entity_id);
CREATE INDEX IF NOT EXISTS idx_pending_approvals_status ON pending_approvals(status);
CREATE INDEX IF NOT EXISTS idx_failed_entities_run_id ON failed_entities(run_id);

-- Shadow mode: candidate policy comparison
CREATE TABLE IF NOT EXISTS shadow_outcomes (
    id SERIAL PRIMARY KEY,
    run_id TEXT REFERENCES automation_runs(run_id),
    entity_id TEXT NOT NULL,
    production_action TEXT NOT NULL,
    shadow_action TEXT NOT NULL,
    production_rule TEXT NOT NULL,
    shadow_rule TEXT NOT NULL,
    diverged BOOLEAN NOT NULL,
    created_at TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_shadow_outcomes_run ON shadow_outcomes(run_id);

-- Audit trail: every action decision logged with reason
CREATE TABLE IF NOT EXISTS action_audit_log (
    id SERIAL PRIMARY KEY,
    entity_id TEXT,
    run_id TEXT,
    actor TEXT NOT NULL,
    action_type TEXT NOT NULL,
    action TEXT NOT NULL,
    rule_matched TEXT,
    permission_result TEXT,
    reason TEXT,
    created_at TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_audit_entity ON action_audit_log(entity_id);
CREATE INDEX IF NOT EXISTS idx_audit_run ON action_audit_log(run_id);
CREATE INDEX IF NOT EXISTS idx_audit_type ON action_audit_log(action_type);
