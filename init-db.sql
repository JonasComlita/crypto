-- PostgreSQL initialization script for blockchain database

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- Create schema
CREATE SCHEMA IF NOT EXISTS blockchain;

-- Create blocks table
CREATE TABLE IF NOT EXISTS blocks (
    height INTEGER PRIMARY KEY,
    hash TEXT NOT NULL,
    previous_hash TEXT NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    merkle_root TEXT NOT NULL,
    difficulty INTEGER NOT NULL,
    nonce INTEGER NOT NULL,
    data BYTEA NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

-- Create transactions table
CREATE TABLE IF NOT EXISTS transactions (
    tx_id TEXT PRIMARY KEY,
    block_height INTEGER REFERENCES blocks(height),
    sender TEXT NOT NULL,
    recipient TEXT NOT NULL,
    amount NUMERIC(20, 8) NOT NULL,
    fee NUMERIC(20, 8) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    tx_type TEXT NOT NULL,
    data BYTEA NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

-- Create UTXOs table
CREATE TABLE IF NOT EXISTS utxos (
    tx_id TEXT NOT NULL,
    output_index INTEGER NOT NULL,
    recipient TEXT NOT NULL,
    amount NUMERIC(20, 8) NOT NULL,
    spent BOOLEAN NOT NULL DEFAULT FALSE,
    spent_in_tx TEXT,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    PRIMARY KEY (tx_id, output_index)
);

-- Create wallets table
CREATE TABLE IF NOT EXISTS wallets (
    address TEXT PRIMARY KEY,
    public_key TEXT NOT NULL,
    encrypted_private_key TEXT NOT NULL,
    salt BYTEA NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

-- Create peer nodes table
CREATE TABLE IF NOT EXISTS peer_nodes (
    node_id TEXT PRIMARY KEY,
    host TEXT NOT NULL,
    port INTEGER NOT NULL,
    last_seen TIMESTAMP NOT NULL,
    public_key TEXT,
    certificate TEXT,
    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

-- Create nonce tracking table
CREATE TABLE IF NOT EXISTS nonces (
    address TEXT NOT NULL,
    nonce BIGINT NOT NULL,
    block_height INTEGER REFERENCES blocks(height),
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    PRIMARY KEY (address, nonce)
);

-- Create user metadata table
CREATE TABLE IF NOT EXISTS user_metadata (
    address TEXT PRIMARY KEY REFERENCES wallets(address),
    username TEXT UNIQUE,
    email TEXT,
    last_login TIMESTAMP,
    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

-- Create indexes for common queries
CREATE INDEX IF NOT EXISTS idx_transactions_block_height ON transactions(block_height);
CREATE INDEX IF NOT EXISTS idx_transactions_sender ON transactions(sender);
CREATE INDEX IF NOT EXISTS idx_transactions_recipient ON transactions(recipient);
CREATE INDEX IF NOT EXISTS idx_utxos_recipient ON utxos(recipient);
CREATE INDEX IF NOT EXISTS idx_utxos_spent ON utxos(spent);
CREATE INDEX IF NOT EXISTS idx_nonces_address ON nonces(address);
CREATE INDEX IF NOT EXISTS idx_peer_nodes_last_seen ON peer_nodes(last_seen);

-- Create transaction log table to track changes
CREATE TABLE IF NOT EXISTS transaction_log (
    id SERIAL PRIMARY KEY,
    tx_id TEXT NOT NULL,
    action TEXT NOT NULL CHECK (action IN ('create', 'update', 'delete')),
    changed_at TIMESTAMP NOT NULL DEFAULT NOW(),
    data JSONB NOT NULL
);

-- Create trigger function for logging transaction changes
CREATE OR REPLACE FUNCTION log_transaction_changes()
RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'INSERT' THEN
        INSERT INTO transaction_log (tx_id, action, data)
        VALUES (NEW.tx_id, 'create', row_to_json(NEW));
    ELSIF TG_OP = 'UPDATE' THEN
        INSERT INTO transaction_log (tx_id, action, data)
        VALUES (NEW.tx_id, 'update', row_to_json(NEW));
    ELSIF TG_OP = 'DELETE' THEN
        INSERT INTO transaction_log (tx_id, action, data)
        VALUES (OLD.tx_id, 'delete', row_to_json(OLD));
    END IF;
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

-- Create trigger for transaction changes
CREATE TRIGGER transactions_audit
AFTER INSERT OR UPDATE OR DELETE ON transactions
FOR EACH ROW EXECUTE FUNCTION log_transaction_changes();

-- Create function to clean up old UTXOs
CREATE OR REPLACE FUNCTION cleanup_old_utxos(days_old INTEGER)
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    -- Delete old spent UTXOs that are at least days_old days old
    DELETE FROM utxos
    WHERE spent = TRUE
    AND created_at < NOW() - (days_old || ' days')::INTERVAL;
    
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- Create materialized view for address balances
CREATE MATERIALIZED VIEW IF NOT EXISTS address_balances AS
SELECT
    recipient AS address,
    SUM(amount) AS balance
FROM utxos
WHERE spent = FALSE
GROUP BY recipient;

-- Create index on the materialized view
CREATE UNIQUE INDEX IF NOT EXISTS idx_address_balances_address ON address_balances(address);

-- Function to refresh the materialized view
CREATE OR REPLACE FUNCTION refresh_address_balances()
RETURNS VOID AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY address_balances;
END;
$$ LANGUAGE plpgsql;

-- Set up permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO blockchain;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO blockchain;
