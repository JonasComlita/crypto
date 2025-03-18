"""
PostgreSQL database adapter for blockchain storage.
Provides high-performance, scalable storage for blockchain data.
"""

import os
import asyncio
import logging
import json
import msgpack
import asyncpg
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime
from functools import wraps
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

# Database connection settings
DB_CONFIG = {
    'host': os.getenv('POSTGRES_HOST', 'localhost'),
    'port': int(os.getenv('POSTGRES_PORT', '5432')),
    'user': os.getenv('POSTGRES_USER', 'blockchain'),
    'password': os.getenv('POSTGRES_PASSWORD', 'blockchain'),
    'database': os.getenv('POSTGRES_DB', 'blockchain'),
    'min_size': int(os.getenv('POSTGRES_MIN_CONN', '5')),
    'max_size': int(os.getenv('POSTGRES_MAX_CONN', '20')),
}

# Connection pool
_pool = None

async def get_pool():
    """Get or create the database connection pool."""
    global _pool
    if _pool is None:
        try:
            _pool = await asyncpg.create_pool(**DB_CONFIG)
            logger.info(f"Connected to PostgreSQL database at {DB_CONFIG['host']}:{DB_CONFIG['port']}")
        except Exception as e:
            logger.error(f"Failed to connect to PostgreSQL: {e}")
            raise
    return _pool

@asynccontextmanager
async def get_connection():
    """Context manager for getting a database connection."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        try:
            yield conn
        except Exception as e:
            logger.error(f"Database error: {e}")
            raise

async def initialize_database():
    """Initialize database schema."""
    async with get_connection() as conn:
        # Create tables if they don't exist
        await conn.execute('''
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
            )
        ''')
        
        await conn.execute('''
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
            )
        ''')
        
        await conn.execute('''
            CREATE TABLE IF NOT EXISTS utxos (
                tx_id TEXT NOT NULL,
                output_index INTEGER NOT NULL,
                recipient TEXT NOT NULL,
                amount NUMERIC(20, 8) NOT NULL,
                spent BOOLEAN NOT NULL DEFAULT FALSE,
                spent_in_tx TEXT,
                created_at TIMESTAMP NOT NULL DEFAULT NOW(),
                PRIMARY KEY (tx_id, output_index)
            )
        ''')
        
        await conn.execute('''
            CREATE TABLE IF NOT EXISTS wallets (
                address TEXT PRIMARY KEY,
                public_key TEXT NOT NULL,
                encrypted_private_key TEXT NOT NULL,
                created_at TIMESTAMP NOT NULL DEFAULT NOW()
            )
        ''')
        
        await conn.execute('''
            CREATE TABLE IF NOT EXISTS peer_nodes (
                node_id TEXT PRIMARY KEY,
                host TEXT NOT NULL,
                port INTEGER NOT NULL,
                last_seen TIMESTAMP NOT NULL,
                public_key TEXT,
                certificate TEXT,
                created_at TIMESTAMP NOT NULL DEFAULT NOW()
            )
        ''')
        
        # Create indices for common queries
        await conn.execute('CREATE INDEX IF NOT EXISTS idx_transactions_block_height ON transactions(block_height)')
        await conn.execute('CREATE INDEX IF NOT EXISTS idx_transactions_sender ON transactions(sender)')
        await conn.execute('CREATE INDEX IF NOT EXISTS idx_transactions_recipient ON transactions(recipient)')
        await conn.execute('CREATE INDEX IF NOT EXISTS idx_utxos_recipient ON utxos(recipient)')
        await conn.execute('CREATE INDEX IF NOT EXISTS idx_utxos_spent ON utxos(spent)')
        
        logger.info("Database schema initialized")

class BlockchainStorage:
    """PostgreSQL storage for blockchain data with msgpack serialization."""
    
    @staticmethod
    async def store_block(block_data: Dict[str, Any]) -> bool:
        """
        Store a block in the database.
        
        Args:
            block_data: Dictionary with block data
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Convert timestamp to datetime if it's a string
            if isinstance(block_data['timestamp'], str):
                try:
                    timestamp = datetime.fromisoformat(block_data['timestamp'])
                except ValueError:
                    # Handle numeric timestamp
                    timestamp = datetime.fromtimestamp(float(block_data['timestamp']))
            else:
                timestamp = datetime.fromtimestamp(block_data['timestamp'])
            
            # Serialize full block data with msgpack
            serialized_data = msgpack.packb(block_data, use_bin_type=True)
            
            async with get_connection() as conn:
                async with conn.transaction():
                    # Insert block
                    await conn.execute('''
                        INSERT INTO blocks (
                            height, hash, previous_hash, timestamp, merkle_root, 
                            difficulty, nonce, data
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                        ON CONFLICT (height) DO UPDATE SET
                            hash = EXCLUDED.hash,
                            previous_hash = EXCLUDED.previous_hash,
                            timestamp = EXCLUDED.timestamp,
                            merkle_root = EXCLUDED.merkle_root,
                            difficulty = EXCLUDED.difficulty,
                            nonce = EXCLUDED.nonce,
                            data = EXCLUDED.data
                    ''', 
                    block_data['index'], 
                    block_data['hash'],
                    block_data['previous_hash'],
                    timestamp,
                    block_data.get('merkle_root', '0' * 64),
                    block_data.get('difficulty', 4),
                    block_data.get('nonce', 0),
                    serialized_data)
                    
                    # Insert each transaction
                    for tx in block_data.get('transactions', []):
                        await BlockchainStorage.store_transaction(tx, block_data['index'], conn)
                        
            logger.debug(f"Stored block {block_data['index']}")
            return True
        except Exception as e:
            logger.error(f"Failed to store block {block_data.get('index', 'unknown')}: {e}")
            return False

    @staticmethod
    async def store_transaction(tx_data: Dict[str, Any], block_height: Optional[int] = None, conn = None) -> bool:
        """
        Store a transaction in the database.
        
        Args:
            tx_data: Dictionary with transaction data
            block_height: Optional block height for transactions in blocks
            conn: Optional database connection for transactions
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Convert timestamp to datetime if it's a string
            if isinstance(tx_data.get('timestamp'), str):
                try:
                    timestamp = datetime.fromisoformat(tx_data['timestamp'])
                except ValueError:
                    # Handle numeric timestamp
                    timestamp = datetime.fromtimestamp(float(tx_data['timestamp']))
            else:
                timestamp = datetime.fromtimestamp(tx_data.get('timestamp', datetime.now().timestamp()))
            
            # Serialize full transaction data with msgpack
            serialized_data = msgpack.packb(tx_data, use_bin_type=True)
            
            # Extract transaction type
            tx_type = tx_data.get('tx_type', 'transfer')
            if isinstance(tx_type, int):
                # Handle enum values
                tx_type = ['coinbase', 'transfer'][tx_type - 1]
            
            should_close_conn = False
            if conn is None:
                conn = await (await get_pool()).acquire()
                should_close_conn = True
            
            try:
                # Insert transaction
                await conn.execute('''
                    INSERT INTO transactions (
                        tx_id, block_height, sender, recipient, amount, fee,
                        timestamp, tx_type, data
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                    ON CONFLICT (tx_id) DO UPDATE SET
                        block_height = EXCLUDED.block_height,
                        data = EXCLUDED.data
                ''', 
                tx_data['tx_id'],
                block_height,
                tx_data.get('sender', '0'),
                tx_data.get('recipient', ''),
                float(tx_data.get('amount', 0)),
                float(tx_data.get('fee', 0)),
                timestamp,
                tx_type,
                serialized_data)
                
                # Handle UTXOs
                if tx_type != 'coinbase':
                    # Mark inputs as spent
                    for tx_input in tx_data.get('inputs', []):
                        await conn.execute('''
                            UPDATE utxos SET 
                                spent = TRUE, 
                                spent_in_tx = $1
                            WHERE tx_id = $2 AND output_index = $3
                        ''', tx_data['tx_id'], tx_input['tx_id'], tx_input['output_index'])
                
                # Add new UTXOs from outputs
                for i, output in enumerate(tx_data.get('outputs', [])):
                    await conn.execute('''
                        INSERT INTO utxos (
                            tx_id, output_index, recipient, amount, spent
                        ) VALUES ($1, $2, $3, $4, FALSE)
                        ON CONFLICT (tx_id, output_index) DO NOTHING
                    ''',
                    tx_data['tx_id'],
                    i,
                    output.get('recipient', ''),
                    float(output.get('amount', 0)))
                
                # If this is a coinbase transaction with no outputs, create a single UTXO
                if tx_type == 'coinbase' and not tx_data.get('outputs') and tx_data.get('recipient'):
                    await conn.execute('''
                        INSERT INTO utxos (
                            tx_id, output_index, recipient, amount, spent
                        ) VALUES ($1, $2, $3, $4, FALSE)
                        ON CONFLICT (tx_id, output_index) DO NOTHING
                    ''',
                    tx_data['tx_id'],
                    0,
                    tx_data['recipient'],
                    float(tx_data.get('amount', 0)))
                
                return True
            finally:
                if should_close_conn:
                    await conn.close()
        except Exception as e:
            logger.error(f"Failed to store transaction {tx_data.get('tx_id', 'unknown')}: {e}")
            return False

    @staticmethod
    async def get_block(height: int) -> Optional[Dict[str, Any]]:
        """
        Retrieve a block by height.
        
        Args:
            height: Block height
            
        Returns:
            Block data dictionary or None if not found
        """
        try:
            async with get_connection() as conn:
                row = await conn.fetchrow('''
                    SELECT height, hash, previous_hash, timestamp, merkle_root,
                           difficulty, nonce, data
                    FROM blocks WHERE height = $1
                ''', height)
                
                if not row:
                    return None
                
                # Unpack the full data
                data = msgpack.unpackb(row['data'], raw=False)
                
                # Get transactions for the block
                txs = await BlockchainStorage.get_transactions_for_block(height)
                data['transactions'] = txs
                
                return data
        except Exception as e:
            logger.error(f"Failed to retrieve block {height}: {e}")
            return None

    @staticmethod
    async def get_blocks(start_height: int = 0, count: int = 100) -> List[Dict[str, Any]]:
        """
        Retrieve a range of blocks.
        
        Args:
            start_height: Starting block height
            count: Number of blocks to retrieve
            
        Returns:
            List of block data dictionaries
        """
        try:
            async with get_connection() as conn:
                rows = await conn.fetch('''
                    SELECT height, hash, previous_hash, timestamp, merkle_root,
                           difficulty, nonce, data
                    FROM blocks
                    WHERE height >= $1
                    ORDER BY height
                    LIMIT $2
                ''', start_height, count)
                
                blocks = []
                for row in rows:
                    # Unpack the full data
                    data = msgpack.unpackb(row['data'], raw=False)
                    blocks.append(data)
                
                return blocks
        except Exception as e:
            logger.error(f"Failed to retrieve blocks from {start_height}: {e}")
            return []

    @staticmethod
    async def get_latest_block() -> Optional[Dict[str, Any]]:
        """
        Retrieve the latest block.
        
        Returns:
            Latest block data dictionary or None if not found
        """
        try:
            async with get_connection() as conn:
                row = await conn.fetchrow('''
                    SELECT height, hash, previous_hash, timestamp, merkle_root,
                           difficulty, nonce, data
                    FROM blocks
                    ORDER BY height DESC
                    LIMIT 1
                ''')
                
                if not row:
                    return None
                
                # Unpack the full data
                data = msgpack.unpackb(row['data'], raw=False)
                
                # Get transactions for the block
                txs = await BlockchainStorage.get_transactions_for_block(row['height'])
                data['transactions'] = txs
                
                return data
        except Exception as e:
            logger.error(f"Failed to retrieve latest block: {e}")
            return None

    @staticmethod
    async def get_transaction(tx_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a transaction by ID.
        
        Args:
            tx_id: Transaction ID
            
        Returns:
            Transaction data dictionary or None if not found
        """
        try:
            async with get_connection() as conn:
                row = await conn.fetchrow('''
                    SELECT tx_id, block_height, sender, recipient, amount, fee,
                           timestamp, tx_type, data
                    FROM transactions WHERE tx_id = $1
                ''', tx_id)
                
                if not row:
                    return None
                
                # Unpack the full data
                data = msgpack.unpackb(row['data'], raw=False)
                return data
        except Exception as e:
            logger.error(f"Failed to retrieve transaction {tx_id}: {e}")
            return None

    @staticmethod
    async def get_transactions_for_block(block_height: int) -> List[Dict[str, Any]]:
        """
        Retrieve all transactions for a given block.
        
        Args:
            block_height: Block height
            
        Returns:
            List of transaction data dictionaries
        """
        try:
            async with get_connection() as conn:
                rows = await conn.fetch('''
                    SELECT data
                    FROM transactions
                    WHERE block_height = $1
                    ORDER BY tx_id
                ''', block_height)
                
                return [msgpack.unpackb(row['data'], raw=False) for row in rows]
        except Exception as e:
            logger.error(f"Failed to retrieve transactions for block {block_height}: {e}")
            return []

    @staticmethod
    async def get_transactions_for_address(address: str, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Retrieve transactions involving a specific address.
        
        Args:
            address: Blockchain address
            limit: Maximum number of transactions to return
            
        Returns:
            List of transaction data dictionaries
        """
        try:
            async with get_connection() as conn:
                rows = await conn.fetch('''
                    SELECT data
                    FROM transactions
                    WHERE sender = $1 OR recipient = $1
                    ORDER BY timestamp DESC
                    LIMIT $2
                ''', address, limit)
                
                return [msgpack.unpackb(row['data'], raw=False) for row in rows]
        except Exception as e:
            logger.error(f"Failed to retrieve transactions for address {address}: {e}")
            return []

    @staticmethod
    async def get_utxos_for_address(address: str) -> List[Dict[str, Any]]:
        """
        Retrieve all unspent transaction outputs for an address.
        
        Args:
            address: Blockchain address
            
        Returns:
            List of UTXO dictionaries
        """
        try:
            async with get_connection() as conn:
                rows = await conn.fetch('''
                    SELECT tx_id, output_index, amount
                    FROM utxos
                    WHERE recipient = $1 AND spent = FALSE
                    ORDER BY tx_id, output_index
                ''', address)
                
                return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"Failed to retrieve UTXOs for address {address}: {e}")
            return []

    @staticmethod
    async def get_balance(address: str) -> float:
        """
        Calculate the balance for an address from its UTXOs.
        
        Args:
            address: Blockchain address
            
        Returns:
            Balance as a float
        """
        try:
            async with get_connection() as conn:
                row = await conn.fetchrow('''
                    SELECT COALESCE(SUM(amount), 0) as balance
                    FROM utxos
                    WHERE recipient = $1 AND spent = FALSE
                ''', address)
                
                return float(row['balance'])
        except Exception as e:
            logger.error(f"Failed to calculate balance for address {address}: {e}")
            return 0.0

    @staticmethod
    async def store_wallet(address: str, public_key: str, encrypted_private_key: str) -> bool:
        """
        Store wallet information.
        
        Args:
            address: Blockchain address
            public_key: Public key (hex string)
            encrypted_private_key: Encrypted private key
            
        Returns:
            True if successful, False otherwise
        """
        try:
            async with get_connection() as conn:
                await conn.execute('''
                    INSERT INTO wallets (address, public_key, encrypted_private_key)
                    VALUES ($1, $2, $3)
                    ON CONFLICT (address) DO UPDATE SET
                        public_key = EXCLUDED.public_key,
                        encrypted_private_key = EXCLUDED.encrypted_private_key
                ''', address, public_key, encrypted_private_key)
                
                return True
        except Exception as e:
            logger.error(f"Failed to store wallet {address}: {e}")
            return False

    @staticmethod
    async def get_wallet(address: str) -> Optional[Dict[str, str]]:
        """
        Retrieve wallet information.
        
        Args:
            address: Blockchain address
            
        Returns:
            Wallet dictionary or None if not found
        """
        try:
            async with get_connection() as conn:
                row = await conn.fetchrow('''
                    SELECT address, public_key, encrypted_private_key
                    FROM wallets
                    WHERE address = $1
                ''', address)
                
                if not row:
                    return None
                
                return dict(row)
        except Exception as e:
            logger.error(f"Failed to retrieve wallet {address}: {e}")
            return None

    @staticmethod
    async def get_all_wallets() -> List[Dict[str, str]]:
        """
        Retrieve all wallets.
        
        Returns:
            List of wallet dictionaries
        """
        try:
            async with get_connection() as conn:
                rows = await conn.fetch('''
                    SELECT address, public_key, encrypted_private_key
                    FROM wallets
                ''')
                
                return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"Failed to retrieve all wallets: {e}")
            return []

    @staticmethod
    async def store_peer(node_id: str, host: str, port: int, public_key: Optional[str] = None, certificate: Optional[str] = None) -> bool:
        """
        Store or update peer node information.
        
        Args:
            node_id: Unique identifier for the peer
            host: Host address
            port: Port number
            public_key: Optional public key
            certificate: Optional certificate
            
        Returns:
            True if successful, False otherwise
        """
        try:
            async with get_connection() as conn:
                await conn.execute('''
                    INSERT INTO peer_nodes (node_id, host, port, last_seen, public_key, certificate)
                    VALUES ($1, $2, $3, NOW(), $4, $5)
                    ON CONFLICT (node_id) DO UPDATE SET
                        host = EXCLUDED.host,
                        port = EXCLUDED.port,
                        last_seen = NOW(),
                        public_key = COALESCE(EXCLUDED.public_key, peer_nodes.public_key),
                        certificate = COALESCE(EXCLUDED.certificate, peer_nodes.certificate)
                ''', node_id, host, port, public_key, certificate)
                
                return True
        except Exception as e:
            logger.error(f"Failed to store peer {node_id}: {e}")
            return False

    @staticmethod
    async def update_peer_last_seen(node_id: str) -> bool:
        """
        Update the last_seen timestamp for a peer.
        
        Args:
            node_id: Unique identifier for the peer
            
        Returns:
            True if successful, False otherwise
        """
        try:
            async with get_connection() as conn:
                result = await conn.execute('''
                    UPDATE peer_nodes 
                    SET last_seen = NOW()
                    WHERE node_id = $1
                ''', node_id)
                
                return 'UPDATE' in result
        except Exception as e:
            logger.error(f"Failed to update last_seen for peer {node_id}: {e}")
            return False

    @staticmethod
    async def get_peers(active_only: bool = True, hours: int = 24) -> List[Dict[str, Any]]:
        """
        Retrieve peer node information.
        
        Args:
            active_only: If True, only return peers seen within the last 'hours'
            hours: Number of hours to consider a peer active
            
        Returns:
            List of peer dictionaries
        """
        try:
            async with get_connection() as conn:
                query = '''
                    SELECT node_id, host, port, last_seen, public_key, certificate
                    FROM peer_nodes
                '''
                
                if active_only:
                    query += f" WHERE last_seen > NOW() - INTERVAL '{hours} hours'"
                
                query += " ORDER BY last_seen DESC"
                
                rows = await conn.fetch(query)
                
                return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"Failed to retrieve peers: {e}")
            return []

    @staticmethod
    async def delete_peer(node_id: str) -> bool:
        """
        Delete a peer node.
        
        Args:
            node_id: Unique identifier for the peer
            
        Returns:
            True if successful, False otherwise
        """
        try:
            async with get_connection() as conn:
                result = await conn.execute('''
                    DELETE FROM peer_nodes
                    WHERE node_id = $1
                ''', node_id)
                
                return 'DELETE' in result
        except Exception as e:
            logger.error(f"Failed to delete peer {node_id}: {e}")
            return False
