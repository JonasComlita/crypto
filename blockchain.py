import json
import time
import hashlib
import asyncio
import sqlite3
from typing import List, Dict, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum
import logging
import ecdsa
from logging.handlers import RotatingFileHandler
from prometheus_client import Counter, Gauge
from utils import SecurityUtils, TransactionInput, TransactionOutput, TransactionType
from collections import defaultdict
from security import MFAManager
import datetime
import aiosqlite
import threading

handler = RotatingFileHandler("originalcoin.log", maxBytes=5*1024*1024, backupCount=3)
logging.basicConfig(level=logging.INFO, handlers=[handler])
logger = logging.getLogger("Blockchain")

BLOCKS_MINED = Counter('blocks_mined_total', 'Total number of blocks mined')
PEER_COUNT = Gauge('peer_count', 'Number of connected peers')

class Transaction:
    """A transaction in the blockchain, representing a transfer of value."""
    def __init__(self, sender, recipient, amount):
        self.sender = sender
        self.recipient = recipient
        self.amount = amount
        self.timestamp = datetime.datetime.now().isoformat()
        self.signature = None
        
    def sign(self, private_key):
        """Sign the transaction with the sender's private key - synchronous version"""
        message = f"{self.sender}{self.recipient}{self.amount}{self.timestamp}".encode()
        self.signature = private_key.sign(message)
        return self.signature
        
    def verify(self, public_key):
        """Verify the transaction signature - synchronous version"""
        if not self.signature:
            return False
            
        try:
            message = f"{self.sender}{self.recipient}{self.amount}{self.timestamp}".encode()
            public_key.verify(self.signature, message)
            return True
        except:
            return False

    def to_dict(self) -> dict:
        """Convert transaction to dictionary"""
        return {
            'sender': self.sender,
            'recipient': self.recipient,
            'amount': self.amount,
            'timestamp': self.timestamp,
            'signature': self.signature
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'Transaction':
        """Create transaction from dictionary data"""
        tx = cls(
            sender=data['sender'],
            recipient=data['recipient'],
            amount=data['amount']
        )
        tx.timestamp = data.get('timestamp', tx.timestamp)
        tx.signature = data.get('signature')
        return tx

class TransactionFactory:
    @staticmethod
    async def create_coinbase_transaction(recipient: str, amount: float, block_height: int) -> Transaction:
        try:
            tx_id = hashlib.sha256(f"coinbase_{block_height}_{recipient}".encode()).hexdigest()
            inputs = [TransactionInput(tx_id=tx_id, output_index=-1)]
            outputs = [TransactionOutput(recipient=recipient, amount=amount)]
            return Transaction(tx_type=TransactionType.COINBASE, inputs=inputs, outputs=outputs)
        except Exception as e:
            logger.error(f"Failed to create coinbase transaction: {e}")
            raise

@dataclass
class BlockHeader:
    index: int
    previous_hash: str
    timestamp: float
    difficulty: int
    merkle_root: str
    nonce: int = 0
    hash: Optional[str] = None

    def calculate_hash(self) -> str:
        data = f"{self.index}{self.previous_hash}{self.timestamp}{self.difficulty}{self.merkle_root}{self.nonce}"
        return hashlib.sha256(data.encode()).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "previous_hash": self.previous_hash,
            "timestamp": self.timestamp,
            "difficulty": self.difficulty,
            "merkle_root": self.merkle_root,
            "nonce": self.nonce,
            "hash": self.hash
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BlockHeader':
        return cls(
            index=data["index"],
            previous_hash=data["previous_hash"],
            timestamp=data["timestamp"],
            difficulty=data["difficulty"],
            merkle_root=data.get("merkle_root", "0" * 64),
            nonce=data["nonce"],
            hash=data["hash"]
        )

async def calculate_merkle_root(transactions: List[Transaction]) -> str:
    try:
        tx_ids = [tx.tx_id for tx in transactions]
        if not tx_ids:
            return "0" * 64
        while len(tx_ids) > 1:
            temp_ids = []
            for i in range(0, len(tx_ids), 2):
                pair = tx_ids[i:i+2]
                if len(pair) == 1:
                    pair.append(pair[0])
                combined = hashlib.sha256((pair[0] + pair[1]).encode()).hexdigest()
                temp_ids.append(combined)
            tx_ids = temp_ids
        return tx_ids[0]
    except Exception as e:
        logger.error(f"Failed to calculate Merkle root: {e}")
        raise

class Block:
    def __init__(self, index: int, transactions: List, previous_hash: str):
        """Initialize a new block"""
        self.index = index
        self.timestamp = datetime.datetime.now().isoformat()
        self.transactions = transactions
        self.previous_hash = previous_hash
        self.nonce = 0
        self.hash = self.calculate_hash()

    def to_dict(self) -> dict:
        """Convert block to dictionary for storage"""
        return {
            'index': self.index,
            'timestamp': self.timestamp,
            'transactions': [tx.__dict__ for tx in self.transactions],
            'previous_hash': self.previous_hash,
            'nonce': self.nonce,
            'hash': self.hash
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'Block':
        """Create block from dictionary data"""
        # Create block with basic data
        block = cls(
            index=data['index'],
            transactions=[],  # Empty list initially
            previous_hash=data['previous_hash']
        )
        
        # Restore other attributes
        block.timestamp = data['timestamp']
        block.nonce = data['nonce']
        block.hash = data['hash']
        
        # Restore transactions
        block.transactions = [
            Transaction(
                sender=tx['sender'],
                recipient=tx['recipient'],
                amount=tx['amount']
            ) for tx in data['transactions']
        ]
        
        # Set transaction timestamps if they exist in the data
        for i, tx_data in enumerate(data['transactions']):
            if 'timestamp' in tx_data:
                block.transactions[i].timestamp = tx_data['timestamp']
            if 'signature' in tx_data:
                block.transactions[i].signature = tx_data['signature']
                
        return block

    def calculate_hash(self) -> str:
        """Calculate block hash"""
        block_string = (
            f"{self.index}"
            f"{self.timestamp}"
            f"{json.dumps([tx.__dict__ for tx in self.transactions])}"
            f"{self.previous_hash}"
            f"{self.nonce}"
        )
        return hashlib.sha256(block_string.encode()).hexdigest()

    def mine_block(self, difficulty: int) -> str:
        """Mine the block with proof of work"""
        target = "0" * difficulty
        while not self.hash.startswith(target):
            self.nonce += 1
            self.hash = self.calculate_hash()
        return self.hash

class UTXOSet:
    def __init__(self):
        self.utxos: Dict[str, List[TransactionOutput]] = {}
        self.used_nonces: Dict[str, set] = {}
        self._lock = asyncio.Lock()

    async def update_with_block(self, block: Block):
        async with self._lock:
            try:
                for tx in block.transactions:
                    for i, output in enumerate(tx.outputs):
                        if tx.tx_id not in self.utxos:
                            self.utxos[tx.tx_id] = []
                        while len(self.utxos[tx.tx_id]) <= i:
                            self.utxos[tx.tx_id].append(None)
                        self.utxos[tx.tx_id][i] = output
                    for input in tx.inputs:
                        if input.tx_id in self.utxos and input.output_index < len(self.utxos[input.tx_id]):
                            self.utxos[input.tx_id][input.output_index] = None
                    if tx.tx_type != TransactionType.COINBASE:
                        for input in tx.inputs:
                            if input.public_key:
                                address = SecurityUtils.public_key_to_address(input.public_key)
                                self.used_nonces.setdefault(address, set()).add(tx.nonce)
            except Exception as e:
                logger.error(f"Failed to update UTXO set with block {block.index}: {e}")
                raise

    async def get_utxos_for_address(self, address: str) -> List[tuple[str, int, TransactionOutput]]:
        async with self._lock:
            result = []
            for tx_id, outputs in self.utxos.items():
                for i, output in enumerate(outputs):
                    if output and output.recipient == address:
                        result.append((tx_id, i, output))
            return result

    async def get_utxo(self, tx_id: str, output_index: int) -> Optional[TransactionOutput]:
        async with self._lock:
            if tx_id in self.utxos and output_index < len(self.utxos[tx_id]):
                return self.utxos[tx_id][output_index]
            return None

    async def get_balance(self, address: str) -> float:
        utxos = await self.get_utxos_for_address(address)
        return sum(utxo[2].amount for utxo in utxos)

    async def is_nonce_used(self, address: str, nonce: int) -> bool:
        async with self._lock:
            return address in self.used_nonces and nonce in self.used_nonces[address]

class Mempool:
    def __init__(self):
        self.transactions: Dict[str, Transaction] = {}
        self.timestamps: Dict[str, float] = {}
        self._lock = asyncio.Lock()
        self.max_size = 1000  # Configurable via environment or config

    async def add_transaction(self, tx: Transaction) -> bool:
        async with self._lock:
            try:
                if not await tx.verify():
                    logger.warning(f"Transaction {tx.tx_id} failed verification")
                    return False
                if tx.tx_id not in self.transactions:
                    if len(self.transactions) >= self.max_size:
                        now = time.time()
                        tx_scores = {tx_id: (tx.fee / len(json.dumps(tx.to_dict())) * 1000) / (now - ts + 1)
                                    for tx_id, tx, ts in [(tid, t, self.timestamps[tid]) 
                                    for tid, t in self.transactions.items()]}
                        lowest_score_tx = min(tx_scores, key=tx_scores.get)
                        del self.transactions[lowest_score_tx]
                        del self.timestamps[lowest_score_tx]
                    self.transactions[tx.tx_id] = tx
                    self.timestamps[tx.tx_id] = time.time()
                    return True
                return False
            except Exception as e:
                logger.error(f"Failed to add transaction {tx.tx_id} to mempool: {e}")
                return False

    async def get_transactions(self, max_txs: int, max_size: int) -> List[Transaction]:
        async with self._lock:
            sorted_txs = sorted(self.transactions.values(), key=lambda tx: tx.fee, reverse=True)
            now = time.time()
            expired = [tx_id for tx_id, ts in self.timestamps.items() if now - ts > 24 * 3600]
            for tx_id in expired:
                self.transactions.pop(tx_id, None)
                self.timestamps.pop(tx_id, None)
            return sorted_txs[:max_txs]

    async def remove_transactions(self, tx_ids: List[str]) -> None:
        async with self._lock:
            for tx_id in tx_ids:
                self.transactions.pop(tx_id, None)
                self.timestamps.pop(tx_id, None)

class Miner:
    def __init__(self, blockchain: 'Blockchain', mempool: Mempool, wallet_address: Optional[str] = None):
        self.blockchain = blockchain
        self.mempool = mempool
        self.wallet_address = wallet_address
        self.current_block: Optional[Block] = None
        self._running = False
        self._lock = asyncio.Lock()

    async def start_mining(self):
        async with self._lock:
            if not self.wallet_address:
                raise ValueError("No wallet address set for mining")
            if self._running:
                logger.warning("Mining is already running")
                return
            self._running = True
            logger.info("Mining started")
            asyncio.create_task(self._mine_continuously())

    async def _mine_continuously(self) -> None:
        while self._running:
            try:
                await self._create_new_block()
                if await self._mine_current_block():
                    success = await self.blockchain.add_block(self.current_block)
                    if success:
                        tx_ids = [tx.tx_id for tx in self.current_block.transactions]
                        await self.mempool.remove_transactions(tx_ids)
                        logger.info(f"Successfully mined block {self.current_block.index}")
                        if hasattr(self.blockchain, 'network'):
                            await self.blockchain.network.broadcast_block(self.current_block)
                await asyncio.sleep(0.1)
            except Exception as e:
                logger.error(f"Mining error: {e}")
                await asyncio.sleep(1)

    async def _create_new_block(self) -> None:
        async with self._lock:
            latest_block = self.blockchain.chain[-1]
            transactions = await self.mempool.get_transactions(1000, 1000000)
            coinbase_tx = await TransactionFactory.create_coinbase_transaction(
                recipient=self.wallet_address,
                amount=self.blockchain.current_reward,
                block_height=latest_block.index + 1
            )
            transactions.insert(0, coinbase_tx)
            self.current_block = Block(
                index=latest_block.index + 1,
                transactions=transactions,
                previous_hash=latest_block.hash
            )

    async def stop_mining(self):
        async with self._lock:
            self._running = False
            logger.info("Mining stopped")

    async def _mine_current_block(self) -> bool:
        async with self._lock:
            if not self.current_block or not self._running:
                return False
            target = "0" * self.current_block.difficulty
            nonce = 0
            start_time = time.time()
            logger.info(f"Starting to mine block {self.current_block.index} with difficulty {self.current_block.difficulty}")
            while self._running:
                self.current_block.nonce = nonce
                block_hash = self.current_block.calculate_hash()
                if block_hash.startswith(target):
                    self.current_block.hash = block_hash
                    logger.info(f"Block {self.current_block.index} mined with hash {block_hash}")
                    return True
                nonce += 1
                if nonce % 10000 == 0:
                    logger.debug(f"Reached nonce {nonce} for block {self.current_block.index}")
                    elapsed = time.time() - start_time
                    await asyncio.sleep(0.001)
                    start_time = time.time()
            return False

class NonceTracker:
    def __init__(self):
        self.nonce_map = defaultdict(set)
        self.nonce_expiry = {}  # Store block height when nonce was used
        
    async def add_nonce(self, address: str, nonce: int, block_height: int):
        self.nonce_map[address].add(nonce)
        self.nonce_expiry[(address, nonce)] = block_height
        
    async def is_nonce_used(self, address: str, nonce: int) -> bool:
        return nonce in self.nonce_map[address]
    
    async def cleanup_old_nonces(self, current_height: int, retention_blocks: int = 10000):
        """Remove nonces older than retention_blocks"""
        expired = [(addr, nonce) for (addr, nonce), height 
                  in self.nonce_expiry.items() 
                  if current_height - height > retention_blocks]
        
        for addr, nonce in expired:
            self.nonce_map[addr].remove(nonce)
            del self.nonce_expiry[(addr, nonce)]

class Blockchain:
    def __init__(self, mfa_manager=None, backup_manager=None, storage_path: str = "chain.db", node_id=None):
        self.chain: List[Block] = []
        self.storage_path = storage_path
        self.difficulty = 4
        self.current_reward = 50.0
        self.halving_interval = 210000
        self.mempool = Mempool()
        self.utxo_set = UTXOSet()
        self.orphans: Dict[str, Block] = {}
        self.max_orphans = 100
        self._lock = asyncio.Lock()
        self.listeners = {"new_block": [], "new_transaction": []}
        self.network = None
        self.block_height = Gauge('blockchain_height', 'Current height of the blockchain')
        self.checkpoint_interval = 100
        self.checkpoints = [0]
        self.nonce_tracker = NonceTracker()
        self.mfa_manager = mfa_manager
        self.backup_manager = backup_manager
        self.node_id = node_id
        self.wallets = {}  # Store wallet addresses and their keys
        self.pending_transactions: List[Transaction] = []
        self.mining_thread = None
        self.mining_flag = threading.Event()
        self.create_genesis_block()

    async def initialize(self):
        """Initialize the blockchain database and load chain"""
        try:
            async with aiosqlite.connect('blockchain.db') as db:
                # Create tables if they don't exist
                await db.execute('''
                    CREATE TABLE IF NOT EXISTS blocks (
                        id INTEGER PRIMARY KEY,
                        data TEXT NOT NULL
                    )
                ''')
                await db.execute('''
                    CREATE TABLE IF NOT EXISTS wallets (
                        address TEXT PRIMARY KEY,
                        public_key TEXT NOT NULL,
                        private_key TEXT NOT NULL
                    )
                ''')
                await db.commit()
                logger.info("Database initialized")
            
            # Load existing chain
            await self.load_chain()
            await self.load_wallets()
            
        except Exception as e:
            logger.error(f"Failed to initialize blockchain: {e}")
            raise

    async def load_chain(self):
        """Load blockchain from storage"""
        try:
            async with aiosqlite.connect('blockchain.db') as db:
                async with db.execute('SELECT data FROM blocks ORDER BY id') as cursor:
                    rows = await cursor.fetchall()
                    if rows:
                        self.chain = [Block.from_dict(json.loads(row[0])) for row in rows]
                        logger.info(f"Loaded {len(self.chain)} blocks from storage")
                    else:
                        logger.info("No existing chain found in storage")
                        # If no chain exists, create genesis block
                        if not self.chain:
                            self.create_genesis_block()
                            await self.save_chain()
        except Exception as e:
            logger.error(f"Failed to load chain: {e}")
            if "no such table" in str(e):
                logger.info("Creating new blockchain")
                self.create_genesis_block()
                await self.save_chain()
            else:
                raise

    def create_genesis_block(self):
        """Create the genesis block synchronously"""
        genesis_block = Block(
            index=0,
            transactions=[],
            previous_hash="0"
        )
        self.chain.append(genesis_block)
        return genesis_block

    async def save_chain(self):
        """Save blockchain to storage"""
        try:
            async with aiosqlite.connect('blockchain.db') as db:
                # Create table if it doesn't exist
                await db.execute('''
                    CREATE TABLE IF NOT EXISTS blocks (
                        id INTEGER PRIMARY KEY,
                        data TEXT NOT NULL
                    )
                ''')
                
                # Clear existing blocks
                await db.execute('DELETE FROM blocks')
                
                # Save all blocks
                for block in self.chain:
                    await db.execute(
                        'INSERT INTO blocks (data) VALUES (?)',
                        (json.dumps(block.to_dict()),)
                    )
                await db.commit()
                logger.info(f"Saved {len(self.chain)} blocks to storage")
        except Exception as e:
            logger.error(f"Failed to save chain: {e}")
            raise

    async def load_wallets(self):
        """Load wallets from storage"""
        try:
            async with aiosqlite.connect('blockchain.db') as db:
                async with db.execute('SELECT address, public_key, private_key FROM wallets') as cursor:
                    rows = await cursor.fetchall()
                    for row in rows:
                        address, public_key, private_key = row
                        self.wallets[address] = {
                            'public_key': public_key,
                            'private_key': private_key
                        }
                    logger.info(f"Loaded {len(self.wallets)} wallets from storage")
        except Exception as e:
            logger.error(f"Failed to load wallets: {e}")
            if "no such table" not in str(e):
                raise

    async def save_wallets(self):
        """Save wallets to storage"""
        try:
            async with aiosqlite.connect('blockchain.db') as db:
                # Create table if it doesn't exist
                await db.execute('''
                    CREATE TABLE IF NOT EXISTS wallets (
                        address TEXT PRIMARY KEY,
                        public_key TEXT NOT NULL,
                        private_key TEXT NOT NULL
                    )
                ''')
                
                # Clear existing wallets
                await db.execute('DELETE FROM wallets')
                
                # Save all wallets
                for address, wallet in self.wallets.items():
                    await db.execute(
                        'INSERT INTO wallets (address, public_key, private_key) VALUES (?, ?, ?)',
                        (address, wallet['public_key'], wallet['private_key'])
                    )
                await db.commit()
                logger.info(f"Saved {len(self.wallets)} wallets to storage")
        except Exception as e:
            logger.error(f"Failed to save wallets: {e}")
            raise

    def update_metrics(self):
        self.block_height.set(len(self.chain) - 1)
        if self.network:
            PEER_COUNT.set(len(self.network.peers))

    async def validate_block(self, block: Block) -> bool:
        async with self._lock:
            try:
                if block.index > 0:
                    if block.index > len(self.chain):
                        logger.info(f"Block {block.hash[:8]} index {block.index} exceeds chain length {len(self.chain)} - potential orphan")
                        return False
                    prev_block = self.chain[block.index - 1]
                    if block.timestamp <= prev_block.timestamp:
                        return False
                    if block.previous_hash != prev_block.hash:
                        return False

                if block.timestamp > time.time() + 2 * 3600:
                    return False

                if not block.transactions or block.transactions[0].tx_type != TransactionType.COINBASE:
                    return False
                coinbase_amount = sum(o.amount for o in block.transactions[0].outputs)
                if coinbase_amount > self.current_reward:
                    return False

                spent_utxos = set()
                for tx in block.transactions[1:]:
                    if tx.tx_type == TransactionType.COINBASE:
                        return False
                    for tx_input in tx.inputs:
                        utxo_key = (tx_input.tx_id, tx_input.output_index)
                        if utxo_key in spent_utxos:
                            return False
                        spent_utxos.add(utxo_key)
                        if tx_input.public_key:
                            address = SecurityUtils.public_key_to_address(tx_input.public_key)
                            if await self.utxo_set.is_nonce_used(address, tx.nonce):
                                return False

                target = "0" * block.difficulty
                if not block.hash or not block.hash.startswith(target):
                    return False

                calculated_merkle_root = await calculate_merkle_root(block.transactions)
                if block.merkle_root != calculated_merkle_root:
                    return False

                tasks = [tx.verify() for tx in block.transactions]
                results = await asyncio.gather(*tasks)
                if not all(results):
                    return False

                return True
            except Exception as e:
                logger.error(f"Block validation failed for {block.index}: {e}")
                return False

    async def add_block(self, block: Block) -> bool:
        async with self._lock:
            try:
                if any(b.hash == block.hash for b in self.chain):
                    return False
                if block.index == len(self.chain) and block.previous_hash == self.chain[-1].hash:
                    if await self.validate_block(block):
                        self.chain.append(block)
                        await self.utxo_set.update_with_block(block)
                        if len(self.chain) % 2016 == 0:
                            self.adjust_difficulty()
                        if len(self.chain) % self.halving_interval == 0:
                            self.halve_block_reward()
                        if len(self.chain) % self.checkpoint_interval == 0:
                            self.checkpoints.append(len(self.chain) - 1)
                        self.trigger_event("new_block", block)
                        await self.save_chain()
                        self.update_metrics()
                        await self._process_orphans()
                        BLOCKS_MINED.inc()
                        logger.info(f"Added block {block.index} to chain: {block.hash[:8]}")
                        return True
                else:
                    await self.handle_potential_fork(block)
                return False
            except Exception as e:
                logger.error(f"Failed to add block {block.index}: {e}")
                return False

    async def _process_orphans(self):
        async with self._lock:
            for hash, orphan in list(self.orphans.items()):
                if orphan.previous_hash == self.chain[-1].hash and orphan.index == len(self.chain):
                    if await self.validate_block(orphan):
                        self.chain.append(orphan)
                        await self.utxo_set.update_with_block(orphan)
                        self.trigger_event("new_block", orphan)
                        await self.save_chain()
                        self.update_metrics()
                        del self.orphans[hash]
                        await self._process_orphans()
                        break

    def adjust_difficulty(self) -> int:
        if len(self.chain) % 2016 == 0 and len(self.chain) > 1:
            period_blocks = self.chain[-2016:]
            time_taken = period_blocks[-1].timestamp - period_blocks[0].timestamp
            target_time = 2016 * 60
            if time_taken > 0:
                ratio = target_time / time_taken
                self.difficulty = max(1, min(20, int(self.difficulty * ratio)))
                logger.info(f"Difficulty adjusted to {self.difficulty}")
        return self.difficulty

    def dynamic_difficulty(self):
        if self.network and len(self.network.peers) < 3:
            return max(6, self.difficulty)
        return self.difficulty

    def get_total_difficulty(self):
        return sum(block.difficulty for block in self.chain)

    async def is_valid_chain(self, chain):
        if not chain or chain[0].hash != self.create_genesis_block().hash:
            return False
        for i in range(1, len(chain)):
            if not await self._is_valid_block(chain[i], chain[i-1]):
                return False
        for checkpoint in self.checkpoints:
            if checkpoint >= len(chain) or chain[checkpoint].hash != self.chain[checkpoint].hash:
                return False
        return True

    async def _is_valid_block(self, block: Block, prev_block: Block) -> bool:
        if block.index != prev_block.index + 1 or block.previous_hash != prev_block.hash:
            return False
        target = "0" * block.difficulty
        return block.hash and block.hash.startswith(target)

    def halve_block_reward(self) -> None:
        self.current_reward /= 2

    async def handle_potential_fork(self, block: Block) -> None:
        async with self._lock:
            if block.index <= len(self.chain) - 1:
                return
            if block.index > len(self.chain):
                if len(self.orphans) >= self.max_orphans:
                    oldest = min(self.orphans.keys(), key=lambda k: self.orphans[k].timestamp)
                    del self.orphans[oldest]
                self.orphans[block.hash] = block
            if self.network:
                await self.network.request_chain()

    async def replace_chain(self, new_chain: List[Block]):
        async with self._lock:
            if len(new_chain) <= len(self.chain):
                return
            if await self.is_valid_chain(new_chain):
                self.chain = new_chain
                self.utxo_set = UTXOSet()
                for block in self.chain:
                    await self.utxo_set.update_with_block(block)
                await self.save_chain()
                self.update_metrics()

    def subscribe(self, event: str, callback: Callable) -> None:
        if event in self.listeners:
            self.listeners[event].append(callback)

    def trigger_event(self, event: str, data: Any) -> None:
        for callback in self.listeners[event]:
            asyncio.create_task(self._async_callback(callback, data, event))

    async def _async_callback(self, callback: Callable, data: Any, event: str) -> None:
        try:
            if asyncio.iscoroutinefunction(callback):
                await callback(data)
            else:
                await asyncio.get_event_loop().run_in_executor(None, callback, data)
        except Exception as e:
            logger.error(f"Callback error for event {event}: {e}")

    async def validate_transaction(self, tx: Transaction) -> bool:
        try:
            if tx.tx_type == TransactionType.COINBASE:
                return True
            if not tx.inputs or not tx.outputs:
                return False
            input_sum = 0
            for tx_input in tx.inputs:
                utxo = await self.utxo_set.get_utxo(tx_input.tx_id, tx_input.output_index)
                if not utxo or not tx_input.public_key or not tx_input.signature:
                    return False
                address = SecurityUtils.public_key_to_address(tx_input.public_key)
                if address != utxo.recipient or await self.utxo_set.is_nonce_used(address, tx.nonce):
                    return False
                if not await tx.verify():
                    return False
                input_sum += utxo.amount
            output_sum = sum(output.amount for output in tx.outputs)
            return output_sum <= input_sum and abs(input_sum - output_sum - tx.fee) < 0.0001
        except Exception as e:
            logger.error(f"Transaction validation failed for {tx.tx_id}: {e}")
            return False

    async def add_transaction_to_mempool(self, tx: Transaction) -> bool:
        if not await self.validate_transaction(tx):
            return False
        success = await self.mempool.add_transaction(tx)
        if success:
            self.trigger_event("new_transaction", tx)
            if self.network:
                await self.network.broadcast_transaction(tx)
        return success

    async def get_balance(self, address: str) -> float:
        return await self.utxo_set.get_balance(address)

    async def create_transaction(self, sender_private_key: str, sender_address: str, 
                               recipient_address: str, amount: float, fee: float = 0.001) -> Optional[Transaction]:
        try:
            sender_utxos = await self.utxo_set.get_utxos_for_address(sender_address)
            total_available = sum(utxo[2].amount for utxo in sender_utxos)
            if total_available < amount + fee:
                return None
            selected_utxos = []
            selected_amount = 0
            for tx_id, output_index, utxo in sender_utxos:
                selected_utxos.append((tx_id, output_index, utxo.amount))
                selected_amount += utxo.amount
                if selected_amount >= amount + fee:
                    break
            if selected_amount < amount + fee:
                return None
            private_key = ecdsa.SigningKey.from_string(bytes.fromhex(sender_private_key), curve=ecdsa.SECP256k1)
            public_key = private_key.get_verifying_key().to_string().hex()
            inputs = [TransactionInput(tx_id=tx_id, output_index=index, public_key=public_key) 
                      for tx_id, index, _ in selected_utxos]
            outputs = [TransactionOutput(recipient=recipient_address, amount=amount)]
            change = selected_amount - amount - fee
            if change > 0:
                outputs.append(TransactionOutput(recipient=sender_address, amount=change))
            tx = Transaction(tx_type=TransactionType.TRANSFER, inputs=inputs, outputs=outputs, fee=fee)
            await tx.sign(private_key)
            return tx
        except Exception as e:
            logger.error(f"Failed to create transaction: {e}")
            return None

    async def add_validator(self, validator_id: str, mfa_token: str = None) -> bool:
        """Add a new validator with MFA verification"""
        if not await self.mfa_manager.verify_mfa(validator_id, mfa_token):
            logger.warning(f"MFA verification failed for validator {validator_id}")
            return False
            
        # ... existing validator addition logic ...
        
    async def update_network_config(self, config: dict, admin_id: str, mfa_token: str = None) -> bool:
        """Update network configuration with MFA verification"""
        if not await self.mfa_manager.verify_mfa(admin_id, mfa_token):
            logger.warning(f"MFA verification failed for admin {admin_id}")
            return False
            
        # ... existing configuration update logic ...

    def get_all_addresses(self):
        """Return list of all wallet addresses in the blockchain"""
        # Collect unique addresses from all transactions
        addresses = set()
        
        # Add addresses from chain
        for block in self.chain:
            for tx in block.transactions:
                addresses.add(tx.sender)
                addresses.add(tx.recipient)
                
        # Add addresses from pending transactions
        for tx in self.pending_transactions:
            addresses.add(tx.sender)
            addresses.add(tx.recipient)
            
        # Remove None or empty addresses
        addresses.discard(None)
        addresses.discard("")
        
        # Convert to sorted list for consistent display
        return sorted(list(addresses))

    def get_balance(self, address):
        """Calculate balance for a given address"""
        balance = 0.0
        
        # Check all confirmed transactions in the chain
        for block in self.chain:
            for tx in block.transactions:
                if tx.sender == address:
                    balance -= tx.amount
                if tx.recipient == address:
                    balance += tx.amount
                    
        # Check pending transactions
        for tx in self.pending_transactions:
            if tx.sender == address:
                balance -= tx.amount
            if tx.recipient == address:
                balance += tx.amount
                
        return balance

    def get_transactions_for_address(self, address):
        """Get all transactions involving a specific address"""
        transactions = []
        
        # Get transactions from chain
        for block in self.chain:
            for tx in block.transactions:
                if tx.sender == address or tx.recipient == address:
                    transactions.append(tx)
                    
        # Get pending transactions
        for tx in self.pending_transactions:
            if tx.sender == address or tx.recipient == address:
                transactions.append(tx)
                
        # Sort by timestamp, newest first
        transactions.sort(key=lambda x: x.timestamp, reverse=True)
        return transactions

    def create_wallet(self):
        """Create a new wallet with public/private key pair"""
        private_key = ecdsa.SigningKey.generate()
        public_key = private_key.get_verifying_key()
        
        # Create address from public key
        address = public_key.to_string().hex()
        
        # Store wallet
        self.wallets[address] = {
            'private_key': private_key,
            'public_key': public_key
        }
        
        return address

    def get_wallet(self, address):
        """Get wallet information for an address"""
        return self.wallets.get(address)

    def create_transaction(self, sender, recipient, amount):
        """Create a new transaction"""
        if amount <= 0:
            raise ValueError("Amount must be positive")
            
        # Check sender's balance
        if self.get_balance(sender) < amount:
            raise ValueError("Insufficient balance")
            
        # Create and sign transaction
        tx = Transaction(sender, recipient, amount)
        
        # Get sender's wallet
        wallet = self.get_wallet(sender)
        if wallet:
            # Synchronous signing
            tx.sign(wallet['private_key'])
        else:
            raise ValueError("Sender wallet not found")
        
        # Add to pending transactions
        self.pending_transactions.append(tx)
        return tx

    def create_block(self) -> Block:
        """Create a new block with pending transactions"""
        if not self.chain:
            return self.create_genesis_block()

        last_block = self.chain[-1]
        new_block = Block(
            index=len(self.chain),
            transactions=self.pending_transactions,
            previous_hash=last_block.hash
        )
        
        # Add to chain and clear pending transactions
        self.chain.append(new_block)
        self.pending_transactions = []
        
        return new_block

    def get_latest_block(self) -> Optional[Block]:
        """Get the latest block in the chain"""
        return self.chain[-1] if self.chain else None

    def start_mining(self):
        """Start the mining process in a separate thread"""
        if self.mining_thread and self.mining_thread.is_alive():
            logger.warning("Mining already in progress")
            return False
            
        self.mining_flag.set()
        self.mining_thread = threading.Thread(target=self._mine_pending_transactions)
        self.mining_thread.daemon = True
        self.mining_thread.start()
        logger.info("Mining process started")
        return True

    def stop_mining(self):
        """Stop the mining process"""
        self.mining_flag.clear()
        if self.mining_thread:
            self.mining_thread.join(timeout=1)
            self.mining_thread = None
        logger.info("Mining process stopped")

    def _mine_pending_transactions(self):
        """Mining loop that runs in a separate thread"""
        while self.mining_flag.is_set():
            if not self.pending_transactions:
                time.sleep(1)  # Wait for transactions
                continue
                
            try:
                # Create new block with pending transactions
                new_block = Block(
                    index=len(self.chain),
                    transactions=self.pending_transactions.copy(),
                    previous_hash=self.chain[-1].hash
                )
                
                # Mine the block
                success = new_block.mine_block(self.difficulty)
                if success and self.mining_flag.is_set():
                    with self._lock:
                        self.chain.append(new_block)
                        self.pending_transactions = []
                        logger.info(f"Mined new block #{new_block.index} with hash {new_block.hash}")
                        
                    # Broadcast new block to network
                    if hasattr(self, 'network'):
                        asyncio.run_coroutine_threadsafe(
                            self.network.broadcast_block(new_block),
                            self.network.loop
                        )
                        
            except Exception as e:
                logger.error(f"Mining error: {e}")
                time.sleep(1)

    def validate_block(self, block: Block) -> bool:
        """Validate a block's hash and transactions"""
        # Check block hash
        if not block.hash.startswith('0' * self.difficulty):
            return False
            
        # Verify block hash matches its contents
        if block.calculate_hash() != block.hash:
            return False
            
        # Verify transactions
        for tx in block.transactions:
            if not self.validate_transaction(tx):
                return False
                
        return True

    def validate_transaction(self, transaction: Transaction) -> bool:
        """Validate a transaction's signature and sender's balance"""
        try:
            # Skip validation for mining rewards
            if transaction.sender is None:
                return True
                
            # Verify signature
            if not transaction.verify():
                return False
                
            # Check sender has sufficient balance
            sender_balance = self.get_balance(transaction.sender)
            if sender_balance < transaction.amount:
                return False
                
            return True
        except Exception as e:
            logger.error(f"Transaction validation error: {e}")
            return False

    def add_block(self, block: Block) -> bool:
        """Add a new block to the chain after validation"""
        try:
            # Validate block
            if not self.validate_block(block):
                return False
                
            # Check block links to previous block
            if block.previous_hash != self.chain[-1].hash:
                return False
                
            # Add block to chain
            with self._lock:
                self.chain.append(block)
                # Remove transactions that are now in the block
                self.pending_transactions = [
                    tx for tx in self.pending_transactions 
                    if tx not in block.transactions
                ]
                
            logger.info(f"Added block #{block.index} to chain")
            return True
            
        except Exception as e:
            logger.error(f"Error adding block: {e}")
            return False

    def get_hashrate(self) -> float:
        """Calculate current mining hashrate"""
        # Simple hashrate calculation based on recent blocks
        try:
            if len(self.chain) < 2:
                return 0
                
            recent_blocks = self.chain[-10:]  # Look at last 10 blocks
            time_diff = (
                datetime.datetime.fromisoformat(recent_blocks[-1].timestamp) -
                datetime.datetime.fromisoformat(recent_blocks[0].timestamp)
            ).total_seconds()
            
            if time_diff <= 0:
                return 0
                
            return len(recent_blocks) / time_diff * self.difficulty
            
        except Exception as e:
            logger.error(f"Error calculating hashrate: {e}")
            return 0