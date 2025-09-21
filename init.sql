CREATE DATABASE trading_bot;
CREATE USER bot_user WITH PASSWORD 'bot_password';
GRANT ALL PRIVILEGES ON DATABASE trading_bot TO bot_user;

\c trading_bot;

CREATE TABLE IF NOT EXISTS signals (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    direction VARCHAR(10) NOT NULL,
    price DECIMAL(18, 8) NOT NULL,
    confidence DECIMAL(5, 4) NOT NULL,
    stop_loss DECIMAL(18, 8),
    take_profit DECIMAL(18, 8),
    risk_reward DECIMAL(5, 2),
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    reasons TEXT
);

CREATE TABLE IF NOT EXISTS trades (
    id SERIAL PRIMARY KEY,
    signal_id INTEGER REFERENCES signals(id),
    order_id VARCHAR(50),
    symbol VARCHAR(20) NOT NULL,
    direction VARCHAR(10) NOT NULL,
    quantity DECIMAL(18, 8) NOT NULL,
    entry_price DECIMAL(18, 8) NOT NULL,
    exit_price DECIMAL(18, 8),
    pnl DECIMAL(18, 8),
    status VARCHAR(20) DEFAULT 'OPEN',
    opened_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    closed_at TIMESTAMP,
    stop_loss DECIMAL(18, 8),
    take_profit DECIMAL(18, 8)
);

CREATE TABLE IF NOT EXISTS portfolio (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    total_balance DECIMAL(18, 8) NOT NULL,
    available_balance DECIMAL(18, 8) NOT NULL,
    equity DECIMAL(18, 8) NOT NULL
);

CREATE INDEX idx_signals_timestamp ON signals(timestamp);
CREATE INDEX idx_signals_symbol ON signals(symbol);
CREATE INDEX idx_trades_status ON trades(status);
CREATE INDEX idx_trades_symbol ON trades(symbol);
CREATE INDEX idx_portfolio_timestamp ON portfolio(timestamp);