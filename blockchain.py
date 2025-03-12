import json
import time
import hashlib
import asyncio
import sqlite3
from typing import List, Dict, Optional, Any, Callable, TYPE_CHECKING
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
import concurrent.futures
import psutil
import tkinter as tk
from tkinter import ttk
import multiprocessing

handler = RotatingFileHandler("originalcoin.log", maxBytes=5*1024*1024, backupCount=3)
logging.basicConfig(level=logging.INFO, handlers=[handler])
logger = logging.getLogger("Blockchain")

BLOCKS_MINED = Counter('blocks_mined_total', 'Total number of blocks mined')
PEER_COUNT = Gauge('peer_count', 'Number of connected peers')

class Transaction:
    def __init__(self, 
                 sender=None, 
                 recipient=None, 
                 amount=None, 
                 tx_type: TransactionType = TransactionType.TRANSFER, 
                 inputs: List[TransactionInput] = None, 
                 outputs: List[TransactionOutput] = None, 
                 fee: float = 0.0, 
                 nonce: Optional[int] = None):
        # Set defaults for all cases
        self.nonce = nonce  # Explicitly set nonce, even if None
        
        # Coinbase transaction
        if sender == "0" and recipient is not None and amount is not None:
            self.sender = "0"
            self.recipient = recipient
            self.amount = amount
            self.timestamp = datetime.datetime.now().isoformat()
            self.signature = None
            self.tx_type = TransactionType.COINBASE
            self.fee = 0.0
            self.inputs = []
            self.outputs = []
        # Simple transfer
        elif sender is not None and recipient is not None and amount is not None:
            self.sender = sender
            self.recipient = recipient
            self.amount = amount
            self.timestamp = datetime.datetime.now().isoformat()
            self.signature = None
            self.tx_type = tx_type or TransactionType.TRANSFER
            self.fee = fee
            self.inputs = []
            self.outputs = []
        # UTXO-based
        else:
            self.inputs = inputs or []
            self.outputs = outputs or []
            self.tx_type = tx_type
            self.timestamp = time.time()
            self.fee = fee
            self.nonce = nonce or int(time.time() * 1000)
            self.sender = self.inputs[0].public_key if self.inputs else None
            self.recipient = self.outputs[0].recipient if self.outputs else None
            self.amount = self.outputs[0].amount if self.outputs else 0
        
        # Calculate tx_id after all attributes are set
        self.tx_id = None
        self.tx_id = self.calculate_tx_id()

    def calculate_tx_id(self) -> str:
        """Calculate transaction ID based on transaction details"""
        try:
            if self.inputs or self.tx_type == TransactionType.COINBASE:
                # UTXO-based or coinbase transaction
                data = {
                    "tx_type": self.tx_type.value,
                    "inputs": [i.to_dict() for i in self.inputs],
                    "outputs": [o.to_dict() for o in self.outputs],
                    "fee": self.fee,
                    "timestamp": self.timestamp
                }
                # Include nonce only if it exists
                if hasattr(self, 'nonce') and self.nonce is not None:
                    data["nonce"] = self.nonce
                return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
            else:
                # Simple transfer transaction
                return hashlib.sha256(
                    f"{self.sender}{self.recipient}{self.amount}{self.timestamp}".encode()
                ).hexdigest()
        except Exception as e:
            logger.error(f"Failed to calculate transaction ID: {e}")
            raise

    def to_dict(self, exclude_signature: bool = False) -> Dict[str, Any]:
        """Convert transaction to dictionary"""
        try:
            base_dict = {
                "sender": self.sender,
                "recipient": self.recipient,
                "amount": self.amount,
                "timestamp": self.timestamp,
                "signature": self.signature if not exclude_signature else None,
                "tx_type": self.tx_type.value,
                "tx_id": self.tx_id
            }
            if hasattr(self, 'nonce') and self.nonce is not None:
                base_dict["nonce"] = self.nonce
            return base_dict
        except Exception as e:
            logger.error(f"Failed to convert transaction to dict: {e}")
            raise

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Transaction':
        """Create transaction from dictionary data"""
        try:
            if "inputs" in data and "outputs" in data:
                # UTXO-based transaction
                tx_type = TransactionType(data["tx_type"])
                inputs = [TransactionInput.from_dict(i) for i in data["inputs"]]
                outputs = [TransactionOutput.from_dict(o) for o in data["outputs"]]
                tx = cls(
                    tx_type=tx_type, 
                    inputs=inputs, 
                    outputs=outputs, 
                    fee=data.get("fee", 0.0), 
                    nonce=data.get("nonce")
                )
                tx.tx_id = data.get("tx_id")
                tx.timestamp = data.get("timestamp", time.time())
                return tx
            else:
                # Simple transfer transaction
                tx = cls(
                    sender=data['sender'], 
                    recipient=data['recipient'], 
                    amount=data['amount']
                )
                tx.timestamp = data.get('timestamp')
                tx.signature = data.get('signature')
                return tx
        except Exception as e:
            logger.error(f"Failed to create transaction from dict: {e}")
            raise

    def sign(self, private_key):
        """Sign the transaction with the sender's private key - synchronous version"""
        try:
            if hasattr(self, 'inputs'):
                # UTXO-based transaction signing
                sk = ecdsa.SigningKey.from_string(bytes.fromhex(private_key), curve=ecdsa.SECP256k1)
                message = json.dumps(self.to_dict(exclude_signature=True), sort_keys=True).encode()
                for input_tx in self.inputs:
                    input_tx.signature = sk.sign(message)
            else:
                # Simple transfer transaction signing
                message = f"{self.sender}{self.recipient}{self.amount}{self.timestamp}".encode()
                sk = ecdsa.SigningKey.from_string(bytes.fromhex(private_key), curve=ecdsa.SECP256k1)
                self.signature = sk.sign(message)
            return self.signature
        except Exception as e:
            logger.error(f"Failed to sign transaction: {e}")
            raise

    async def sign_async(self, private_key: str):
        """Async version of sign method"""
        try:
            if hasattr(self, 'inputs'):
                # UTXO-based transaction signing
                sk = ecdsa.SigningKey.from_string(bytes.fromhex(private_key), curve=ecdsa.SECP256k1)
                message = json.dumps(self.to_dict(exclude_signature=True), sort_keys=True).encode()
                for input_tx in self.inputs:
                    input_tx.signature = await asyncio.get_event_loop().run_in_executor(
                        None, lambda: sk.sign(message)
                    )
            else:
                # Simple transfer transaction signing
                message = f"{self.sender}{self.recipient}{self.amount}{self.timestamp}".encode()
                sk = ecdsa.SigningKey.from_string(bytes.fromhex(private_key), curve=ecdsa.SECP256k1)
                self.signature = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: sk.sign(message)
                )
            return self.signature
        except Exception as e:
            logger.error(f"Failed to sign transaction asynchronously: {e}")
            raise

    async def verify(self) -> bool:
        """Verify transaction signature"""
        try:
            if hasattr(self, 'inputs'):
                # UTXO-based transaction verification
                message = json.dumps(self.to_dict(exclude_signature=True), sort_keys=True).encode()
                tasks = []
                for input_tx in self.inputs:
                    if not input_tx.signature or not input_tx.public_key:
                        return False
                    vk = ecdsa.VerifyingKey.from_string(bytes.fromhex(input_tx.public_key), curve=ecdsa.SECP256k1)
                    tasks.append(
                        asyncio.get_event_loop().run_in_executor(
                            None, lambda: vk.verify(input_tx.signature, message)
                        )
                    )
                results = await asyncio.gather(*tasks, return_exceptions=True)
                return all(isinstance(r, bool) and r for r in results)
            else:
                # Simple transfer transaction verification
                message = f"{self.sender}{self.recipient}{self.amount}{self.timestamp}".encode()
                vk = ecdsa.VerifyingKey.from_string(
                    bytes.fromhex(self.sender), 
                    curve=ecdsa.SECP256k1
                )
                return vk.verify(self.signature, message)
        except Exception as e:
            logger.error(f"Transaction verification failed: {e}")
            return False

class TransactionFactory:
    @staticmethod
    async def create_coinbase_transaction(self, recipient: str, amount: float, block_height: int) -> Transaction:
        try:
            # Create a simple coinbase transaction without unnecessary UTXO complexity
            tx = Transaction(
                sender="0",
                recipient=recipient,
                amount=amount,
                tx_type=TransactionType.COINBASE
            )
            logger.info(f"Created coinbase transaction for block {block_height} to {recipient}")
            return tx
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

class TransactionEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, TransactionType):
            return obj.value
        return super().default(obj)

class Block:
    def __init__(self, index: int, transactions: List, previous_hash: str):
        """Initialize a new block"""
        self.index = index
        self.timestamp = datetime.datetime.now().isoformat()
        self.transactions = transactions
        self.previous_hash = previous_hash
        self.nonce = 0
        self.difficulty = transactions[0].blockchain.difficulty if transactions and hasattr(transactions[0], 'blockchain') else 4
        self.merkle_root = "0" * 64  # Default merkle root
        self.hash = self.calculate_hash()

    def to_dict(self) -> dict:
        """Convert block to dictionary for storage"""
        return {
            'index': self.index,
            'timestamp': self.timestamp,
            'transactions': [tx.to_dict() for tx in self.transactions],  # Use to_dict() instead of __dict__
            'previous_hash': self.previous_hash,
            'nonce': self.nonce,
            'difficulty': self.difficulty,
            'merkle_root': self.merkle_root,
            'hash': self.hash,
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
        block.nonce = data.get('nonce', 0)
        block.hash = data['hash']
        block.difficulty = data.get('difficulty', 4)
        
        # Restore transactions
        block.transactions = [
            Transaction(
                sender=tx.get('sender'),
                recipient=tx.get('recipient'),
                amount=tx.get('amount')
            ) for tx in data.get('transactions', [])
        ]
        
        # Set transaction timestamps if they exist in the data
        for i, tx_data in enumerate(data.get('transactions', [])):
            if 'timestamp' in tx_data:
                block.transactions[i].timestamp = tx_data['timestamp']
            if 'signature' in tx_data:
                block.transactions[i].signature = tx_data['signature']
                
        return block

    def calculate_hash(self) -> str:
        block_string = (
            f"{self.index}"
            f"{self.timestamp}"
            f"{json.dumps([tx.to_dict() for tx in self.transactions], cls=TransactionEncoder)}"
            f"{self.previous_hash}"
            f"{self.nonce}"
        )
        return hashlib.sha256(block_string.encode()).hexdigest()

    def _mine_block(self, block):
        self._logger.debug(f"Starting to mine block {block.index}")
        target = "0" * self.difficulty
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

class MiningService:
    def __init__(self, blockchain):
        self.blockchain = blockchain
        self.process_pool = concurrent.futures.ProcessPoolExecutor()
        self.mining_tasks = {}

    async def start_mining(self, wallet_address):
        self._logger.info("Starting mining loop")
        self.mining = True
        return True

    async def stop_mining(self):
        self._logger.info("Stopping mining loop")
        self.mining = False  # Assuming _mining_loop checks this
        return True

import threading
import time
import logging
import queue

class AsyncMiner:
    """A robust asynchronous mining implementation"""
    def __init__(self, blockchain, wallet_address=None, loop=None):
        """
        Initialize the miner
        
        :param blockchain: The blockchain instance
        :param wallet_address: Address to receive mining rewards
        """
        self.blockchain = blockchain
        self.wallet_address = wallet_address
        
        # Mining control
        self.mining = False
        self._mining_thread = None
        self._stop_event = threading.Event()
        
        # Logging
        self._logger = logging.getLogger(__name__)
        
        # Performance tracking
        self.blocks_mined = 0
        self.total_hashes = 0
        self.start_time = 0

        self.loop = loop
        self._process_pool = concurrent.futures.ProcessPoolExecutor()

    def start_mining(self):
        if not self.wallet_address:
            self._logger.error("No wallet address set for mining")
            return False

        if self.mining:
            self._logger.warning("Mining is already running")
            return False

        self.mining = True
        self.start_time = time.time()
        self._stop_event.clear()
        try:
            asyncio.create_task(self._mining_loop())
            self._logger.info(f"Mining thread started for {self.wallet_address}")
            return True
        except Exception as e:
            self._logger.error(f"failed to start mining loop: {e}")
            self.mining = False
            return False

    async def stop_mining(self):
        """Stop the mining process cleanly"""
        self._logger.info("Stopping mining...")
        try:
            self.mining = False
            self._stop_event.set()
            
            # Wait briefly for mining loop to exit
            await asyncio.sleep(0.5)  # Give time for loop to check stop condition
            
            # Shut down the process pool
            if self._process_pool:
                self._process_pool.shutdown(wait=True)
                self._logger.info("Mining process pool shut down")
                self._process_pool = None  # Clear reference
            
            self.start_time = 0  # Reset hashrate tracking
            self.total_hashes = 0
            return True
        except Exception as e:
            self._logger.error(f"Failed to stop mining: {e}")
            return False
        finally:
            self.mining = False
            self._logger.info("Mining stopped")

    async def _mining_loop(self):
        self._logger.info("Entering mining loop")
        try:
            while self.mining and not self._stop_event.is_set():
                self._logger.debug("Mining loop iteration")
                try:
                    block = self._create_block()
                    if block:
                        self._logger.info(f"Created block {block.index}")
                        mined_block = await asyncio.get_event_loop().run_in_executor(
                            self._process_pool, self._mine_block, block
                        )
                        if mined_block:
                            if await self._add_block_to_chain(mined_block):
                                self._logger.debug(f"Block {block.index} processed successfully")
                            else:
                                self._logger.warning(f"Block {block.index} not added, continuing")
                        else:
                            self._logger.warning("Failed to mine block")
                    else:
                        self._logger.warning("Failed to create block")
                    await asyncio.sleep(0.1)
                except Exception as e:
                    self._logger.error(f"Mining iteration error: {e}")
                    await asyncio.sleep(1)
        except Exception as e:
            self._logger.error(f"Mining loop crashed: {e}")
        finally:
            self.mining = False
            self._logger.info("Mining loop exited")

    def _create_block(self):
        """Synchronous block creation"""
        try:
            latest_block = self.blockchain.chain[-1]
            transactions = list(self.blockchain.mempool.transactions.values())[:1000]
            # Synchronous coinbase creation for simplicity
            coinbase_tx = Transaction(
                sender="0",
                recipient=self.wallet_address,
                amount=self.blockchain.current_reward,
                tx_type=TransactionType.COINBASE
            )
            transactions.insert(0, coinbase_tx)
            block = Block(
                index=latest_block.index + 1,
                transactions=transactions,
                previous_hash=latest_block.hash
            )
            self._logger.info(f"Block {block.index} created with {len(transactions)} transactions")
            return block
        except Exception as e:
            self._logger.error(f"Block creation error: {e}")
            return None

    def _mine_block(self, block):
        """Synchronous block mining"""
        self._logger.debug(f"Starting to mine block {block.index}")
        try:
            target = "0" * self.blockchain.difficulty
            nonce = 0
            while self.mining and not self._stop_event.is_set():
                block.nonce = nonce
                block_hash = block.calculate_hash()
                self.total_hashes += 1
                self._logger.debug(f"Hash attempt: {self.total_hashes}, Hash: {block_hash[:10]}...")
                if block_hash.startswith(target):
                    block.hash = block_hash
                    self._logger.info(f"Block mined with nonce {nonce}")
                    return True
                nonce += 1
                if nonce % 10000 == 0:
                    self._logger.debug(f"Nonce at {nonce}, hash: {block_hash[:10]}...")
                    time.sleep(0.001)
            self._logger.info("Mining stopped")
            return False
        except Exception as e:
            self._logger.error(f"Block mining error: {e}")
            return False

    async def _add_block_to_chain(self, block):
        """Add block to blockchain asynchronously"""
        self._logger.debug(f"Attempting to add block {block.index}")
        try:
            # Assuming self.loop is passed from BlockchainGUI or Blockchain.network.loop
            if not self.loop or not self.loop.is_running():
                self._logger.error("Event loop is not available or not running")
                return False
            success = await self.blockchain.add_block(block)
            if success:
                self.blocks_mined += 1
                self._logger.info(f"Successfully added block {block.index} to chain: {block.hash[:8]}")
            else:
                self._logger.warning(f"Failed to add block {block.index} to chain")
            return success
        except Exception as e:
            self._logger.error(f"Failed to add block {block.index} to chain: {str(e)}")
            return False

    def get_hashrate(self):
        if not self.mining or self.start_time == 0:
            return 0.0
        elapsed = time.time() - self.start_time
        hashrate = self.total_hashes / elapsed if elapsed > 0 else 0.0
        self._logger.debug(f"Hashrate: {hashrate}, Total Hashes: {self.total_hashes}, Elapsed: {elapsed}")
        return hashrate
    

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

class NetworkService:
    def __init__(self, blockchain_network=None):
        self.network = blockchain_network
        self.peers = set()
        self.loop = None
        self.message_handlers = {
            'block': self._handle_block,
            'transaction': self._handle_transaction,
            'chain_request': self._handle_chain_request
        }

    async def start(self):
        """Initialize the network service"""
        self.loop = asyncio.get_event_loop()
        if self.network:
            await self.network.start()
        logger.info("Network service started")

    async def stop(self):
        """Stop the network service"""
        if self.network:
            await self.network.stop()
        logger.info("Network service stopped")

    async def broadcast_block(self, block):
        """Broadcast a new block to all peers"""
        if self.network:
            try:
                await self.network.broadcast_block(block)
                logger.info(f"Block {block.index} broadcasted to network")
            except Exception as e:
                logger.error(f"Failed to broadcast block: {e}")

    async def broadcast_transaction(self, transaction):
        """Broadcast a new transaction to all peers"""
        if self.network:
            try:
                await self.network.broadcast_transaction(transaction)
                logger.info(f"Transaction {transaction.tx_id} broadcasted to network")
            except Exception as e:
                logger.error(f"Failed to broadcast transaction: {e}")

    async def request_chain(self):
        """Request blockchain from peers"""
        if self.network:
            try:
                await self.network.request_chain()
                logger.info("Chain request sent to network")
            except Exception as e:
                logger.error(f"Failed to request chain: {e}")

    async def _handle_block(self, block_data):
        """Handle received block from network"""
        try:
            block = Block.from_dict(block_data)
            await self.blockchain.add_block(block)
        except Exception as e:
            logger.error(f"Failed to handle received block: {e}")

    async def _handle_transaction(self, tx_data):
        """Handle received transaction from network"""
        try:
            transaction = Transaction.from_dict(tx_data)
            await self.blockchain.add_transaction_to_mempool(transaction)
        except Exception as e:
            logger.error(f"Failed to handle received transaction: {e}")

    async def _handle_chain_request(self, request_data):
        """Handle chain request from peers"""
        try:
            if self.network:
                chain_data = [block.to_dict() for block in self.blockchain.chain]
                await self.network.send_chain(request_data['peer_id'], chain_data)
        except Exception as e:
            logger.error(f"Failed to handle chain request: {e}")

class Blockchain:
    def __init__(self, mfa_manager=None, backup_manager=None, storage_path: str = "chain.db", node_id=None):
        from network import BlockchainNetwork
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
        self.network = BlockchainNetwork(self, node_id or "default", "127.0.0.1", 5000)
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
        self._thread_lock = threading.Lock()  # Keep this for non-async operations
        self._mining_pool = concurrent.futures.ProcessPoolExecutor()  # For mining
        self.miner = AsyncMiner(self)

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
                    logger.info(f"Block {block.index} already exists in chain: {block.hash[:8]}")
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
                        logger.info(f"Successfully added block {block.index} to chain: {block.hash[:8]}, chain length: {len(self.chain)}")
                        return True
                    else:
                        logger.warning(f"Block {block.index} failed validation: {block.hash[:8]}")
                else:
                    logger.info(f"Block {block.index} doesn’t follow chain tip, handling as potential fork")
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

    async def start_mining(self, wallet_address=None):
        logger.info("Attempting to start mining...")
        try:
            if not wallet_address:
                addresses = self.get_all_addresses()
                if not addresses:
                    logger.error("No wallet addresses available for mining")
                    raise ValueError("No wallet addresses available")
                wallet_address = addresses[0]
            self.miner.wallet_address = wallet_address
            result = self.miner.start_mining()  # Synchronous call
            if result:
                logger.info(f"Mining started successfully with address: {wallet_address}")
                return True
            logger.warning("Mining failed to start")
            return False
        except Exception as e:
            logger.error(f"Mining error: {e}")
            return False

    async def stop_mining(self):
        """Stop the mining process and clean up resources"""
        logger.info("Stopping mining...")
        try:
            # Stop the AsyncMiner
            result = await self.miner.stop_mining()
            if not result:
                logger.warning("Miner failed to stop cleanly")
                return False
            
            # Shut down Blockchain's mining pool
            if hasattr(self, '_mining_pool') and self._mining_pool:
                self._mining_pool.shutdown(wait=True)
                logger.info("Blockchain mining process pool shut down")
                self._mining_pool = None  # Clear reference
            
            return True
        except Exception as e:
            logger.error(f"Failed to stop mining: {e}")
            return False

    async def _mining_process(self):
        """Async mining process"""
        while self.mining_flag.is_set():
            try:
                # Create new block synchronously
                block = self._create_new_block()
                
                # Do mining work in process pool
                mining_result = await asyncio.get_event_loop().run_in_executor(
                    self._mining_pool,
                    self._do_mining_work,
                    block,
                    self.difficulty
                )
                
                if mining_result and self.mining_flag.is_set():
                    # Add block to chain if mining was successful and still running
                    success = await self.add_block(mining_result)
                    if success:
                        logger.info(f"Successfully mined and added block {mining_result.index}")
                        # Clear any transactions that were included in this block
                        self.pending_transactions = []
                    
            except Exception as e:
                logger.error(f"Mining error: {e}")
                await asyncio.sleep(1)
            
            # Small delay to prevent CPU hogging
            await asyncio.sleep(0.1)

    def _do_mining_work(self, block: Block, difficulty: int) -> Optional[Block]:
        """Synchronous mining calculations - runs in process pool"""
        target = "0" * difficulty
        max_nonce = 2**32  # Maximum nonce value
        
        for nonce in range(max_nonce):
            if not self.mining_flag.is_set():
                return None
                
            block.nonce = nonce
            block_hash = block.calculate_hash()
            
            if block_hash.startswith(target):
                block.hash = block_hash
                return block
            
            if nonce % 10000 == 0:
                time.sleep(0.001)  # Prevent CPU hogging
                
        return None

    def _create_new_block(self) -> Block:
        """Create a new block with pending transactions"""
        if not self.chain:
            return self.create_genesis_block()

        last_block = self.chain[-1]
        
        # Create coinbase transaction for mining reward
        coinbase_tx = Transaction(
            sender="0",
            recipient=self.mining_address,
            amount=self.current_reward,
            tx_type=TransactionType.COINBASE
        )
        
        # Get pending transactions (limit to reasonable size)
        transactions = self.pending_transactions[:10]  # Limit to 10 transactions per block
        transactions.insert(0, coinbase_tx)  # Add coinbase transaction first
        
        # Create new block
        new_block = Block(
            index=len(self.chain),
            transactions=transactions,
            previous_hash=last_block.hash
        )
        new_block.difficulty = self.difficulty
        new_block.timestamp = time.time()
        
        return new_block

    async def get_hashrate(self) -> float:
        if hasattr(self, 'miner'):
            hashrate = self.miner.get_hashrate()  # Synchronous call is fine here since it's a simple calculation
            logger.debug(f"Blockchain get_hashrate: {hashrate}")
            return hashrate
        logger.debug("No miner available, returning 0")
        return 0.0
        
    async def create_coinbase_transaction(self, recipient: str, amount: float, block_height: int) -> Transaction:
        """
        Create a coinbase transaction for mining rewards
        
        :param recipient: Address receiving the mining reward
        :param amount: Amount of the mining reward
        :param block_height: Height of the block being mined
        :return: Coinbase Transaction
        """
        try:
            tx = Transaction(
                sender="0",
                recipient=recipient,
                amount=amount,
                tx_type=TransactionType.COINBASE
            )
            logger.info(f"Created coinbase transaction for block {block_height} to {recipient}")
            return tx
        except Exception as e:
            logger.error(f"Failed to create coinbase transaction: {e}")
            raise

class ResourceMonitor:
    async def monitor_resources(self):
        while True:
            cpu_usage = psutil.cpu_percent()
            memory_usage = psutil.virtual_memory().percent
            
            if cpu_usage > 90:  # Threshold
                await self.adjust_mining_intensity()
            
            await asyncio.sleep(1)

class BlockchainNode:
    def __init__(self):
        from network import BlockchainNetwork
        self.blockchain = Blockchain()
        self.mining_service = MiningService(self.blockchain)
        self.blockchain_network = BlockchainNetwork(self.blockchain)  # Create BlockchainNetwork instance
        self.network_service = NetworkService(self.blockchain_network)  # Pass it to NetworkService
        self.resource_monitor = ResourceMonitor()
        
        # Set up network reference in blockchain
        self.blockchain.network = self.network_service

    async def start(self):
        """Start all services"""
        try:
            # Initialize blockchain first
            await self.blockchain.initialize()
            
            # Start all services
            await asyncio.gather(
                self.mining_service.start(),
                self.network_service.start(),
                self.resource_monitor.start()
            )
            logger.info("BlockchainNode started successfully")
        except Exception as e:
            logger.error(f"Failed to start BlockchainNode: {e}")
            raise

    async def stop(self):
        """Stop all services gracefully"""
        try:
            await self.network_service.stop()
            await self.mining_service.stop_mining()
            logger.info("BlockchainNode stopped successfully")
        except Exception as e:
            logger.error(f"Error during BlockchainNode shutdown: {e}")
            raise

    async def toggle_mining(self):
        """Toggle mining state"""
        if not self.mining:
            if await self.blockchain.start_mining():  # Now properly awaited
                self.mining = True
                self.mining_btn.configure(text="Stop Mining")
                self.status_var.set("Mining started")
        else:
            await self.blockchain.stop_mining()  # Make sure stop_mining is also async
            self.mining = False
            self.mining_btn.configure(text="Start Mining")
            self.status_var.set("Mining stopped")

    def init_mining_tab(self):
        """Initialize mining control tab"""
        mining_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(mining_frame, text="Mining")
        
        # Mining controls
        controls_frame = ttk.LabelFrame(mining_frame, text="Mining Controls", padding="10")
        controls_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=5)
        
        self.mining_btn = ttk.Button(
            controls_frame,
            text="Start Mining",
            command=self._handle_mining_click,
            style='Mining.TButton'
        )
        self.mining_btn.grid(row=0, column=0, padx=5, pady=5)

    def _handle_mining_click(self):
        """Handle mining button click"""
        if not hasattr(self, 'mining_task'):
            self.mining_task = None

        if not self.mining:
            # Cancel existing task if any
            if self.mining_task:
                self.mining_task.cancel()
            
            # Create new task
            self.mining_task = asyncio.run_coroutine_threadsafe(
                self.blockchain.start_mining(),
                self.blockchain.network.loop
            )
            
            # Set up callback
            def mining_started(future):
                try:
                    if future.result():
                        self.mining = True
                        self.mining_btn.configure(text="Stop Mining")
                        self.status_var.set("Mining started")
                except Exception as e:
                    logger.error(f"Mining start error: {e}")
                    self.status_var.set(f"Mining failed: {str(e)}")
                    
            self.mining_task.add_done_callback(mining_started)
        else:
            # Stop mining
            asyncio.run_coroutine_threadsafe(
                self.blockchain.stop_mining(),
                self.blockchain.network.loop
            )
            self.mining = False
            self.mining_btn.configure(text="Start Mining")
            self.status_var.set("Mining stopped")