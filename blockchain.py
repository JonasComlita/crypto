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
from utils import (
    SecurityUtils, TransactionInput, TransactionOutput, TransactionType,
    BLOCKS_RECEIVED, TXS_BROADCAST, PEER_FAILURES, ACTIVE_REQUESTS, BLOCKS_MINED, PEER_COUNT, BLOCK_HEIGHT,
    safe_gauge, safe_counter, CONFIG
)
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

logging.basicConfig(level=logging.DEBUG)
handler = RotatingFileHandler("originalcoin.log", maxBytes=5*1024*1024, backupCount=3)
logging.basicConfig(level=logging.INFO, handlers=[handler])
logger = logging.getLogger("Blockchain")

def _do_mining_work( block_data: dict, difficulty: int) -> Optional[dict]:
    """CPU-intensive mining work (runs in a separate process)"""
    try:
        logger.debug("Starting mining work in process...")
        logger.debug(f"Received block data keys: {list(block_data.keys())}")
        
        target = "0" * difficulty
        max_nonce = 2**32
        nonce = 0
        total_hashes = 0
        
        # Create block string for hashing
        try:
            block_string_base = (
                f"{block_data['index']}"
                f"{block_data['timestamp']}"
                f"{json.dumps(block_data['transactions'], sort_keys=True)}"
                f"{block_data['previous_hash']}"
            )
            logger.debug("Block string base created successfully")
        except Exception as e:
            logger.error(f"Error creating block string: {str(e)}")
            logger.error(f"Block data content: {block_data}")
            raise
        
        while nonce < max_nonce:
            # Calculate hash with current nonce
            block_string = f"{block_string_base}{nonce}"
            block_hash = hashlib.sha256(block_string.encode()).hexdigest()
            total_hashes += 1
            
            if block_hash.startswith(target):
                logger.debug(f"Found valid hash after {total_hashes} attempts")
                return {
                    'nonce': nonce,
                    'hash': block_hash,
                    'hashes': total_hashes
                }
            
            nonce += 1
        
        logger.debug(f"Max nonce reached after {total_hashes} attempts")
        return None
        
    except Exception as e:
        logger.error(f"Mining work error: {str(e)}", exc_info=True)
        logger.error(f"Error occurred with difficulty: {difficulty}")
        return None


class Transaction:
    def __init__(self, sender: str, recipient: str, amount: float, 
             tx_type: TransactionType = TransactionType.TRANSFER,
             signature: Optional[bytes] = None,
             timestamp: Optional[str] = None,
             inputs: Optional[List[TransactionInput]] = None,
             outputs: Optional[List[TransactionOutput]] = None,
             fee: float = 0.0,
             nonce: int = 0):
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
            self.timestamp = timestamp or datetime.datetime.now().isoformat()
            self.signature = None
            self.tx_type = tx_type or TransactionType.TRANSFER
            self.fee = fee
            self.inputs = inputs or []  
            self.outputs = outputs or []
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
        
        if asyncio.isfuture(signature):
            logger.error(f"Transaction initialized with Future signature for tx_id {self.tx_id}")
            raise ValueError("Signature cannot be an asyncio.Future")
        assert isinstance(self.signature, (bytes, type(None))), f"Signature must be bytes or None, got {type(self.signature)}"
        # Calculate transaction ID
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
                logger.debug(f"Deserializing transaction with signature type: {type(data.get('signature'))}")
                signature = bytes.fromhex(data["signature"]) if data.get("signature") else None
                if signature:
                    tx.signature = signature
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

    async def sign(self, private_key):
        """Sign the transaction with the sender's private key - async version"""
        try:
            if hasattr(self, 'inputs') and self.inputs:
                # UTXO-based transaction signing
                sk = ecdsa.SigningKey.from_string(bytes.fromhex(private_key), curve=ecdsa.SECP256k1)
                message = json.dumps(self.to_dict(exclude_signature=True), sort_keys=True).encode()
                # Run CPU-intensive signing in thread pool
                for input_tx in self.inputs:
                    input_tx.signature = await asyncio.to_thread(lambda: sk.sign(message))
            else:
                # Simple transfer transaction signing
                message = f"{self.sender}{self.recipient}{self.amount}{self.timestamp}".encode()
                sk = ecdsa.SigningKey.from_string(bytes.fromhex(private_key), curve=ecdsa.SECP256k1)
                self.signature = await asyncio.to_thread(lambda: sk.sign(message))
            return self.signature
        except Exception as e:
            logger.error(f"Failed to sign transaction asynchronously: {e}")
            raise

    async def verify(self) -> bool:
        """Verify transaction signature"""
        try:
            if self.tx_type == TransactionType.COINBASE:
                return True
            
            if hasattr(self, 'inputs') and self.inputs:
                # UTXO-based transaction verification
                message = json.dumps(self.to_dict(exclude_signature=True), sort_keys=True).encode()
                tasks = []
                for input_tx in self.inputs:
                    if not input_tx.signature or not input_tx.public_key:
                        return False
                    vk = ecdsa.VerifyingKey.from_string(bytes.fromhex(input_tx.public_key), curve=ecdsa.SECP256k1)
                    tasks.append(
                        asyncio.to_thread(lambda v=vk, s=input_tx.signature: v.verify(s, message))
                    )
                results = await asyncio.gather(*tasks, return_exceptions=True)
                return all(isinstance(r, bool) and r for r in results)
            else:
                # Simple transfer transaction verification
                if not self.signature:
                    return False
                message = f"{self.sender}{self.recipient}{self.amount}{self.timestamp}".encode()
                vk = ecdsa.VerifyingKey.from_string(
                    bytes.fromhex(self.sender), 
                    curve=ecdsa.SECP256k1
                )
                return await asyncio.to_thread(lambda: vk.verify(self.signature, message))
        except Exception as e:
            logger.error(f"Transaction verification failed: {e}")
            return False

class TransactionFactory:
    @staticmethod
    async def create_coinbase_transaction(recipient: str, amount: float, block_height: int) -> Transaction:
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
        self.difficulty = 4  # Default difficulty
        self.merkle_root = "0" * 64  # Default merkle root
        self.hash = None
        # Calculate hash last after all properties are set
        self.hash = self.calculate_hash()

    def to_dict(self) -> dict:
        """Convert block to dictionary for storage"""
        return {
            'index': self.index,
            'timestamp': self.timestamp,
            'transactions': [tx.to_dict() for tx in self.transactions],
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
        block.merkle_root = data.get('merkle_root', "0" * 64)
        
        # Restore transactions
        block.transactions = [
            Transaction.from_dict(tx) for tx in data.get('transactions', [])
        ]
                
        return block

    def calculate_hash(self) -> str:
        """Calculate the hash of the block"""
        block_string = (
            f"{self.index}"
            f"{self.timestamp}"
            f"{json.dumps([tx.to_dict() for tx in self.transactions], cls=TransactionEncoder)}"
            f"{self.previous_hash}"
            f"{self.nonce}"
        )
        return hashlib.sha256(block_string.encode()).hexdigest()

    def to_mineable_dict(self) -> dict:
        """Convert block to a dictionary format safe for mining process"""
        try:
            logger.debug(f"Converting block {self.index} with {len(self.transactions)} transactions")
            
            # Safely convert transactions to basic dictionaries
            safe_transactions = []
            for tx in self.transactions:
                try:
                    if asyncio.isfuture(tx.signature):
                        logger.error(f"Found Future signature in tx {tx.tx_id}")
                        raise ValueError("Signature cannot be a Future")
                    
                    safe_tx = {
                        'tx_id': str(tx.tx_id),
                        'sender': str(tx.sender) if tx.sender else "0",
                        'recipient': str(tx.recipient) if tx.recipient else "",
                        'amount': float(tx.amount),
                        'timestamp': str(tx.timestamp),
                        'tx_type': str(tx.tx_type.value) if hasattr(tx, 'tx_type') else 'transfer',
                        'fee': float(getattr(tx, 'fee', 0.0))
                    }
                    if hasattr(tx, 'signature'):
                        if isinstance(tx.signature, bytes):
                            safe_tx['signature'] = tx.signature.hex()
                        elif tx.signature is None:
                            safe_tx['signature'] = None
                        else:
                            logger.error(f"Unpicklable signature in tx {tx.tx_id}: {type(tx.signature)}")
                            raise ValueError(f"Unpicklable signature: {type(tx.signature)}")
                    # Handle inputs and outputs explicitly
                    if hasattr(tx, 'inputs') and tx.inputs:
                        safe_tx['inputs'] = [
                            {
                                'tx_id': str(inp.tx_id),
                                'output_index': int(inp.output_index),
                                'public_key': str(inp.public_key) if inp.public_key else None,
                                'signature': inp.signature.hex() if inp.signature else None
                            }
                            for inp in tx.inputs
                        ]
                    if hasattr(tx, 'outputs') and tx.outputs:
                        safe_tx['outputs'] = [
                            {
                                'recipient': str(out.recipient),
                                'amount': float(out.amount),
                                'script': str(out.script)
                            }
                            for out in tx.outputs
                        ]
                    safe_transactions.append(safe_tx)
                except Exception as e:
                    logger.error(f"Error converting transaction {getattr(tx, 'tx_id', 'unknown')}: {str(e)}")
                    raise

            # Create the safe dictionary with primitive types only
            safe_dict = {
                'index': int(self.index),
                'timestamp': str(self.timestamp),
                'transactions': safe_transactions,
                'previous_hash': str(self.previous_hash),
                'nonce': int(self.nonce),
                'difficulty': int(self.difficulty),
                'merkle_root': str(self.merkle_root)
            }
            
            logger.debug(f"Successfully created mineable dict for block {self.index}")
            return safe_dict
            
        except Exception as e:
            logger.error(f"Error in to_mineable_dict: {str(e)}", exc_info=True)
            raise

class UTXOSet:
    def __init__(self):
        self.utxos: Dict[str, List[TransactionOutput]] = {}
        self.used_nonces: Dict[str, set] = {}
        self._lock = asyncio.Lock()

    async def update_with_block(self, block: Block):
        async with self._lock:
            try:
                for tx in block.transactions:
                    # Add transaction outputs to UTXO set
                    for i, output in enumerate(tx.outputs):
                        if tx.tx_id not in self.utxos:
                            self.utxos[tx.tx_id] = []
                        while len(self.utxos[tx.tx_id]) <= i:
                            self.utxos[tx.tx_id].append(None)
                        self.utxos[tx.tx_id][i] = output
                    
                    # Remove spent transaction inputs
                    for input in tx.inputs:
                        if input.tx_id in self.utxos and input.output_index < len(self.utxos[input.tx_id]):
                            self.utxos[input.tx_id][input.output_index] = None
                    
                    # Add nonces for replay protection
                    if tx.tx_type != TransactionType.COINBASE:
                        for input in tx.inputs:
                            if input.public_key:
                                address = SecurityUtils.public_key_to_address(input.public_key)
                                if tx.nonce is not None:  # Ensure nonce exists
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
                    # If mempool is full, remove lowest-scoring transaction
                    if len(self.transactions) >= self.max_size:
                        now = time.time()
                        # Calculate score for fee/size ratio and age
                        tx_scores = {
                            tx_id: (tx.fee / len(json.dumps(tx.to_dict())) * 1000) / (now - self.timestamps[tx_id] + 1)
                            for tx_id, tx in self.transactions.items()
                        }
                        lowest_score_tx = min(tx_scores, key=tx_scores.get)
                        del self.transactions[lowest_score_tx]
                        del self.timestamps[lowest_score_tx]
                    
                    # Add new transaction
                    self.transactions[tx.tx_id] = tx
                    self.timestamps[tx.tx_id] = time.time()
                    return True
                return False
            except Exception as e:
                logger.error(f"Failed to add transaction {tx.tx_id} to mempool: {e}")
                return False

    async def get_transactions(self, max_txs: int, max_size: int) -> List[Transaction]:
        async with self._lock:
            # Sort transactions by fee, highest first
            sorted_txs = sorted(self.transactions.values(), key=lambda tx: tx.fee, reverse=True)
            
            # Clean up expired transactions (24 hours old)
            now = time.time()
            expired = [tx_id for tx_id, ts in self.timestamps.items() if now - ts > 24 * 3600]
            for tx_id in expired:
                self.transactions.pop(tx_id, None)
                self.timestamps.pop(tx_id, None)
                
            # Return limited number of transactions
            return sorted_txs[:max_txs]

    async def remove_transactions(self, tx_ids: List[str]) -> None:
        async with self._lock:
            for tx_id in tx_ids:
                self.transactions.pop(tx_id, None)
                self.timestamps.pop(tx_id, None)

class AsyncMiner:
    """A robust asynchronous mining implementation"""
    def __init__(self, blockchain):
        """Initialize the miner"""
        self.blockchain = blockchain
        self.wallet_address = None
        
        # Mining control
        self.mining = False
        self._stop_event = asyncio.Event()
        self._mining_task = None

        # Add lock initialization
        self._lock = asyncio.Lock()
        self.hashrate_lock = asyncio.Lock()
        self.mining_lock = asyncio.Lock()
        self._stats = {
            'hashrate': 0,
            'blocks_mined': 0
        }
        
        # Logging
        self._logger = logging.getLogger(__name__)
        
        # Performance tracking
        self.blocks_mined = 0
        self.total_hashes = 0
        self.start_time = 0
        self.last_hash_count = 0
        self.last_hash_time = time.time()
        
        # Use a process pool for CPU-intensive mining
        self._process_pool = concurrent.futures.ProcessPoolExecutor(
            max_workers=max(1, multiprocessing.cpu_count() - 1)
        )

    async def start_mining(self, wallet_address: str) -> bool:
        """Start the mining process"""
        async with self.mining_lock:
            if not wallet_address:
                self._logger.error("No wallet address set for mining")
                return False

            if self.mining:
                self._logger.warning("Mining is already running")
                return False

            self._logger.info(f"Starting mining for wallet {wallet_address}")
            self.wallet_address = wallet_address
            self.mining = True
            self.start_time = time.time()
            self._stop_event.clear()
            
            # Start mining in a separate task
            self._mining_task = asyncio.create_task(self._mining_loop())
            return True

    async def stop_mining(self) -> bool:
        """Stop the mining process cleanly"""
        async with self.mining_lock:
            if not self.mining:
                self._logger.info("Mining is not running")
                return True
                
            self._logger.info("Stopping mining...")
            try:
                # Signal mining loop to stop
                self.mining = False
                self._stop_event.set()
                
                # Wait for mining loop to finish (with timeout)
                if self._mining_task and not self._mining_task.done():
                    try:
                        await asyncio.wait_for(self._mining_task, timeout=2.0)
                    except asyncio.TimeoutError:
                        self._logger.warning("Mining task didn't stop in time, cancelling")
                        self._mining_task.cancel()
                        try:
                            await self._mining_task
                        except asyncio.CancelledError:
                            pass
                
                # Shut down the process pool
                self._process_pool.shutdown(wait=False)
                self._process_pool = concurrent.futures.ProcessPoolExecutor(
                    max_workers=max(1, multiprocessing.cpu_count() - 1)
                )
                
                # Reset state
                self.start_time = 0
                self.total_hashes = 0
                self._logger.info("Mining stopped")
                return True
            except Exception as e:
                self._logger.error(f"Failed to stop mining: {e}")
                return False
            finally:
                self.mining = False

    async def _mining_loop(self):
        """Main mining loop"""
        self._logger.info("Entering mining loop")
        try:
            while self.mining and not self._stop_event.is_set():
                try:
                    # Create a new block to mine
                    self._logger.debug("Creating new block...")
                    block = await self._create_block()
                    if not block:
                        self._logger.warning("Failed to create block, waiting...")
                        await asyncio.sleep(1)
                        continue
                    
                    self._logger.info(f"Created block {block.index}, preparing for mining...")
                    
                    try:
                        # Convert block to mining-safe format
                        self._logger.debug("Converting block to mineable format...")
                        mineable_data = block.to_mineable_dict()
                        self._logger.debug(f"Block converted successfully: {list(mineable_data.keys())}")
                    except Exception as e:
                        self._logger.error(f"Error converting block to mineable format: {str(e)}")
                        raise

                    try:
                        # Mine the block in a separate process
                        self._logger.debug("Starting mining process...")
                        mining_result = await self._mine_block(mineable_data)
                        self._logger.debug(f"Mining process completed with result: {mining_result is not None}")
                    except Exception as e:
                        self._logger.error(f"Error during mining process: {str(e)}")
                        self._logger.error(f"Mineable data keys: {list(mineable_data.keys())}")
                        self._logger.error(f"Block data type: {type(mineable_data)}")
                        for key, value in mineable_data.items():
                            self._logger.error(f"Key: {key}, Type: {type(value)}")
                        raise
                    
                    if not mining_result:
                        self._logger.debug("Mining was stopped or failed")
                        continue
                    
                    # Update the original block with mining results
                    block.nonce = mining_result['nonce']
                    block.hash = mining_result['hash']
                    
                    # Add the mined block to the chain
                    if await self.blockchain.add_block(block):
                        self.blocks_mined += 1
                        self._logger.info(f"✅ Successfully mined and added block {block.index}")
                    else:
                        self._logger.warning(f"Block {block.index} was mined but not added to chain")
                        
                except asyncio.CancelledError:
                    self._logger.info("Mining loop cancelled")
                    raise
                except Exception as e:
                    self._logger.error(f"Mining error in loop: {str(e)}", exc_info=True)
                    self._logger.error(f"Error type: {type(e)}")
                    await asyncio.sleep(0.5)
                    continue
                
                await asyncio.sleep(0.1)
                
        except asyncio.CancelledError:
            self._logger.info("Mining loop cancelled")
        except Exception as e:
            self._logger.error(f"Fatal mining loop error: {str(e)}", exc_info=True)
        finally:
            self.mining = False
            self._logger.info("Mining loop exited")

    async def _create_block(self) -> Optional[Block]:
        """Create a new block to mine"""
        async with self._lock:
            try:
                # Get the latest block
                latest_block = self.blockchain.get_latest_block()
                if not latest_block:
                    self._logger.error("No latest block found")
                    return None
                    
                # Get transactions from mempool
                transactions = await self.blockchain.mempool.get_transactions(1000, 1000000)
                for transaction in transactions:
                    logger.debug(f"Transaction {transaction.tx_id}: signature type = {type(transaction.signature)}")
                # Create coinbase transaction (mining reward)
                coinbase_tx = await TransactionFactory.create_coinbase_transaction(
                    recipient=self.wallet_address,
                    amount=self.blockchain.current_reward,
                    block_height=latest_block.index + 1
                )
                
                # Add coinbase as first transaction
                transactions.insert(0, coinbase_tx)
                
                # Create the new block
                block = Block(
                    index=latest_block.index + 1,
                    transactions=transactions,
                    previous_hash=latest_block.hash,
                )

                block.difficulty = self.blockchain.difficulty  # Use blockchain's difficulty
                
                # Calculate merkle root
                block.merkle_root = await calculate_merkle_root(transactions)

                self._logger.debug(f"Created block with difficulty {block.difficulty} from blockchain difficulty {self.blockchain.difficulty}")
                self._logger.info(f"Block {block.index} created with {len(transactions)} transactions")
                return block
            except Exception as e:
                self._logger.error(f"Block creation error: {str(e)}")
                self._logger.debug("Error details:", exc_info=True)
                return None
        
    async def _mine_block(self, block_data: dict) -> Optional[dict]:
        if not self.mining or self._stop_event.is_set():
            return None
        try:
            logger.debug("Inspecting block_data before mining:")
            for key, value in block_data.items():
                logger.debug(f"Key: {key}, Type: {type(value)}")
                if isinstance(value, list):
                    for i, item in enumerate(value):
                        logger.debug(f"  Item {i}: Type: {type(item)}, Value: {item}")
            
            import pickle
            try:
                pickle.dumps(block_data)
                logger.debug("Block data successfully serialized")
            except Exception as e:
                logger.error(f"Serialization test failed: {str(e)}")
                logger.error(f"Block data content: {block_data}")
                raise
            
            mining_result = await asyncio.get_event_loop().run_in_executor(
                self._process_pool,
                _do_mining_work,
                block_data,
                self.blockchain.difficulty
            )
            if not mining_result or not self.mining or self._stop_event.is_set():
                return None
            self.total_hashes += mining_result['hashes']
            return mining_result
        except Exception as e:
            logger.error(f"Mining execution error: {str(e)}", exc_info=True)
            return None

    def get_hashrate(self) -> float:
        """Calculate current hashrate"""
        if not self.mining:
            return 0.0
        
        current_time = time.time()
        time_diff = current_time - self.last_hash_time
        hash_diff = self.total_hashes - self.last_hash_count
        
        if time_diff > 0:
            hashrate = hash_diff / time_diff
            # Update last values for next calculation
            self.last_hash_count = self.total_hashes
            self.last_hash_time = current_time
            return hashrate
        return 0.0

class NonceTracker:
    def __init__(self):
        self.nonce_map = defaultdict(set)
        self.nonce_expiry = {}  # Store block height when nonce was used
        self._lock = asyncio.Lock()
        
    async def add_nonce(self, address: str, nonce: int, block_height: int):
        async with self._lock:
            self.nonce_map[address].add(nonce)
            self.nonce_expiry[(address, nonce)] = block_height
        
    async def is_nonce_used(self, address: str, nonce: int) -> bool:
        async with self._lock:
            return nonce in self.nonce_map[address]
    
    async def cleanup_old_nonces(self, current_height: int, retention_blocks: int = 10000):
        """Remove nonces older than retention_blocks"""
        async with self._lock:
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
        self.blockchain = None  # Will be set by Blockchain class
        self.message_handlers = {
            'block': self._handle_block,
            'transaction': self._handle_transaction,
            'chain_request': self._handle_chain_request
        }

    async def start(self):
        """Initialize the network service"""
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
            if not self.blockchain:
                logger.error("Blockchain reference not set in NetworkService")
                return
                
            block = Block.from_dict(block_data)
            await self.blockchain.add_block(block)
        except Exception as e:
            logger.error(f"Failed to handle received block: {e}")

    async def _handle_transaction(self, tx_data):
        """Handle received transaction from network"""
        try:
            if not self.blockchain:
                logger.error("Blockchain reference not set in NetworkService")
                return
                
            transaction = Transaction.from_dict(tx_data)
            await self.blockchain.add_transaction_to_mempool(transaction)
        except Exception as e:
            logger.error(f"Failed to handle received transaction: {e}")

    async def _handle_chain_request(self, request_data):
        """Handle chain request from peers"""
        try:
            if not self.blockchain:
                logger.error("Blockchain reference not set in NetworkService")
                return
                
            if self.network:
                chain_data = [block.to_dict() for block in self.blockchain.chain]
                await self.network.send_chain(request_data['peer_id'], chain_data)
        except Exception as e:
            logger.error(f"Failed to handle chain request: {e}")

class Blockchain:
    def __init__(self, mfa_manager=None, backup_manager=None, storage_path: str = "chain.db", node_id=None):
        """Initialize blockchain"""
        from network import BlockchainNetwork
    
        # Initialization flag
        self.initialized = False
        self._initializing = False
        
        # Chain storage
        self.chain: List[Block] = []
        self.storage_path = storage_path
        
        # Mining parameters
        self.difficulty = CONFIG["difficulty"]
        self.current_reward = CONFIG["current_reward"]
        self.halving_interval = CONFIG["halving_interval"]
        
        # Transaction handling
        self.mempool = Mempool()
        self.utxo_set = UTXOSet()
        self.orphans: Dict[str, Block] = {}
        self.max_orphans = 100
        
        # Concurrency controls
        self._lock = asyncio.Lock()
        
        # Event system
        self.listeners = {"new_block": [], "new_transaction": []}
        
        # Networking
        self.node_id = node_id or "default"
        self.network_service = NetworkService()
        self.network = BlockchainNetwork(self, self.node_id, "127.0.0.1", 5000)
        self.network_service.network = self.network
        self.network_service.blockchain = self
        
        # Metrics
        BLOCK_HEIGHT.labels(instance=self.node_id).set(0)
        # Checkpointing
        self.checkpoint_interval = 100
        self.checkpoints = [0]
        
        # Security
        self.nonce_tracker = NonceTracker()
        self.mfa_manager = mfa_manager
        self.backup_manager = backup_manager
        
        # Wallet storage
        self.wallets = {}  # Store wallet addresses and their keys
        
        # Mining
        self.miner = AsyncMiner(self)
        
        # Create genesis block to initialize the chain
        self.create_genesis_block()

        self._chain_lock = asyncio.Lock()
        self._mempool_lock = asyncio.Lock()
        self._metric_lock = asyncio.Lock()

    async def initialize(self):
        """Initialize the blockchain database and load chain"""
        if hasattr(self, '_initializing') and self._initializing:
            return
            
        self._initializing = True
        
        try:
            async with aiosqlite.connect(self.storage_path) as db:
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
            
            # Initialize network
            await self.network_service.start()
            
            # Update metrics
            await self.update_metrics()

            # Set initialized flag
            self.initialized = True
            
            logger.info(f"Blockchain initialized with {len(self.chain)} blocks")
            
        except Exception as e:
            logger.error(f"Failed to initialize blockchain: {e}")
            raise
        finally:
            self._initializing = False

    async def shutdown(self):
        """Safely shut down the blockchain"""
        try:
            # Stop mining if active
            await self.stop_mining()
            
            # Stop network
            await self.network_service.stop()
            
            # Save chain and wallets
            await self.save_chain()
            await self.save_wallets()
            
            logger.info("Blockchain shut down successfully")
        except Exception as e:
            logger.error(f"Error during blockchain shutdown: {e}")
            raise

    async def load_chain(self):
        """Load blockchain from storage"""
        try:
            async with aiosqlite.connect(self.storage_path) as db:
                async with db.execute('SELECT data FROM blocks ORDER BY id') as cursor:
                    rows = await cursor.fetchall()
                    if rows:
                        self.chain = [Block.from_dict(json.loads(row[0])) for row in rows]
                        logger.info(f"Loaded {len(self.chain)} blocks from storage")
                    else:
                        logger.info("No existing chain found in storage")
                        # If no chain exists, create genesis block
                        if not self.chain:
                            self.create_genesis_block()  # Don't await this if it's synchronous
                            await self.save_chain()
                            
                        # Initialize UTXO set from chain
                        await self._rebuild_utxo_set()
        except Exception as e:
            logger.error(f"Failed to load chain: {e}")
            if "no such table" in str(e):
                logger.info("Creating new blockchain")
                self.create_genesis_block()
                await self.save_chain()
            else:
                raise

    async def _rebuild_utxo_set(self):
        """Rebuild the UTXO set from scratch based on the blockchain"""
        self.utxo_set = UTXOSet()
        for block in self.chain:
            await self.utxo_set.update_with_block(block)
        logger.info("UTXO set rebuilt from chain")

    def create_genesis_block(self):
        """Create the genesis block"""
        genesis_tx = Transaction(
            sender="0", 
            recipient="genesis", 
            amount=0, 
            tx_type=TransactionType.COINBASE
        )
        genesis_block = Block(
            index=0,
            transactions=[genesis_tx],
            previous_hash="0"
        )
        # Ensure empty chain before adding genesis
        if not self.chain:
            self.chain.append(genesis_block)
        return genesis_block

    async def save_chain(self):
        """Save blockchain to storage"""
        try:
            async with aiosqlite.connect(self.storage_path) as db:
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
            async with aiosqlite.connect(self.storage_path) as db:
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
            async with aiosqlite.connect(self.storage_path) as db:
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

    async def update_metrics(self):
        """Update prometheus metrics"""
        async with self._metric_lock:
            try:
                BLOCK_HEIGHT.labels(instance=self.node_id).set(len(self.chain) - 1)
                if hasattr(self.network, 'peers'):
                    PEER_COUNT.labels(instance=self.node_id).set(len(self.network.peers))
            except Exception as e:
                logger.error(f"Failed to update metrics: {e}")


    async def validate_block(self, block: Block) -> bool:
        """Validate block against consensus rules"""
        async with self._lock:
            try:
                logger.info(f"Starting validation for block {block.index}")
                # Check block is properly linked to chain
                if block.index > 0:
                    if block.index > len(self.chain):
                        logger.info(f"Block {block.hash[:8]} index {block.index} exceeds chain length {len(self.chain)} - potential orphan")
                        return False
                        
                    prev_block = self.chain[block.index - 1]
                    if block.previous_hash != prev_block.hash:
                        logger.warning(f"Block {block.index} has incorrect previous_hash")
                        return False
                
                # Validate timestamp (not in future, after previous block)
                if block.index > 0:
                    prev_block = self.chain[block.index - 1]
                    if block.timestamp <= prev_block.timestamp:
                        logger.warning(f"Block {block.index} timestamp not after previous block")
                        return False
                        
                if isinstance(block.timestamp, str):
                    # Handle ISO format timestamps
                    block_time = datetime.datetime.fromisoformat(block.timestamp)
                    now = datetime.datetime.now()
                    if block_time > now + datetime.timedelta(hours=2):
                        logger.warning(f"Block {block.index} timestamp too far in future")
                        return False
                else:
                    # Handle epoch timestamps
                    if block.timestamp > time.time() + 2 * 3600:
                        logger.warning(f"Block {block.index} timestamp too far in future")
                        return False

                # Check for coinbase transaction
                if not block.transactions or len(block.transactions) == 0:
                    logger.warning(f"Block {block.index} has no transactions")
                    return False
                    
                if block.transactions[0].tx_type != TransactionType.COINBASE:
                    logger.warning(f"Block {block.index} first transaction is not coinbase")
                    return False
                    
                # Validate coinbase reward
                coinbase_amount = block.transactions[0].amount
                if coinbase_amount > self.current_reward:
                    logger.warning(f"Block {block.index} coinbase reward {coinbase_amount} exceeds limit {self.current_reward}")
                    return False

                # Check for double-spends within block
                spent_utxos = set()
                for tx in block.transactions[1:]:  # Skip coinbase
                    if tx.tx_type == TransactionType.COINBASE:
                        logger.warning(f"Block {block.index} has multiple coinbase transactions")
                        return False
                        
                    # Check each input
                    for tx_input in tx.inputs:
                        utxo_key = (tx_input.tx_id, tx_input.output_index)
                        if utxo_key in spent_utxos:
                            logger.warning(f"Block {block.index} contains double-spend of {utxo_key}")
                            return False
                        spent_utxos.add(utxo_key)
                        
                        # Check for nonce replay
                        if tx_input.public_key and tx.nonce is not None:
                            address = SecurityUtils.public_key_to_address(tx_input.public_key)
                            if await self.utxo_set.is_nonce_used(address, tx.nonce):
                                logger.warning(f"Block {block.index} contains used nonce for {address}")
                                return False

                # Validate proof of work
                target = "0" * block.difficulty
                if not block.hash or not block.hash.startswith(target):
                    logger.warning(f"Block {block.index} hash {block.hash[:10]} doesn't meet difficulty target")
                    return False

                # Validate merkle root
                calculated_merkle_root = await calculate_merkle_root(block.transactions)
                if block.merkle_root != calculated_merkle_root:
                    logger.warning(f"Block {block.index} has invalid merkle root")
                    return False

                # Validate transaction signatures
                tasks = [tx.verify() for tx in block.transactions]
                results = await asyncio.gather(*tasks)
                if not all(results):
                    logger.info(f"Validating {len(block.transactions)} transaction signatures")
                    for i, tx in enumerate(block.transactions):
                        if not await tx.verify():
                            logger.warning(f"Transaction {i} (tx_id: {tx.tx_id[:8]}) in block {block.index} failed signature verification")
                            logger.warning(f"Transaction type: {tx.tx_type}")
                            if hasattr(tx, 'signature'):
                                logger.warning(f"Signature present: {tx.signature is not None}, Type: {type(tx.signature)}")
                            else:
                                logger.warning("Transaction has no signature attribute")
                            return False

                return True
            except Exception as e:
                logger.error(f"Block validation failed for {block.index}: {e}")
                return False

    async def add_block(self, block: Block) -> bool:
        """Add a validated block to the blockchain"""
        async with self._chain_lock:
            try:
                # Check if block already exists
                if any(b.hash == block.hash for b in self.chain):
                    logger.info(f"Block {block.index} already exists in chain: {block.hash[:8]}")
                    return False
                    
                # Check if block fits at the end of current chain
                if block.index == len(self.chain) and block.previous_hash == self.chain[-1].hash:
                    # Validate block
                    is_valid = await self.validate_block(block)
                    if not is_valid:
                        logger.warning(f"Block {block.index} failed validation: {block.hash[:8]}")
                        return False
                        
                    # More detailed logging:
                    logger.info(f"Block validation successful for block {block.index}")
                    logger.info(f"Adding block {block.index} with hash {block.hash[:8]}")
                    logger.info(f"Previous block hash: {self.chain[-1].hash[:8]}")
                    logger.info(f"Block transactions: {len(block.transactions)}")

                    self.chain.append(block)
                        
                    # Check if we need to adjust difficulty
                    if len(self.chain) % 2016 == 0:
                        self.adjust_difficulty()
                        
                    # Check if we need to halve block reward
                    if len(self.chain) % self.halving_interval == 0:
                        self.halve_block_reward()
                        
                    # Add checkpoint if needed
                    if len(self.chain) % self.checkpoint_interval == 0:
                        self.checkpoints.append(len(self.chain) - 1)
                        
                    # Notify listeners
                    self.trigger_event("new_block", block)
                    
                    # Save chain to storage
                    await self.save_chain()
                    
                    # Update metrics
                    self.update_metrics()
                    
                    # Process any orphan blocks that might now fit
                    await self._process_orphans()
                    
                    # Increment blocks mined counter
                    BLOCKS_MINED.labels(instance=self.node_id).inc()
            
                    # Remove block's transactions from mempool
                    await self._remove_block_txs_from_mempool(block)
                    
                    logger.info(f"Successfully added block {block.index} to chain: {block.hash[:8]}, chain length: {len(self.chain)}")
                    return True
                else:
                    logger.info(f"Block {block.index} doesn't follow chain tip, handling as potential fork")
                    logger.info(f"Block index: {block.index}, Chain length: {len(self.chain)}")
                    logger.info(f"Block previous_hash: {block.previous_hash[:8]}")
                    logger.info(f"Chain tip hash: {self.chain[-1].hash[:8]}")
                    await self.handle_potential_fork(block)
                return False
            except Exception as e:
                logger.error(f"Failed to add block {block.index}: {e}")
                return False

    async def _remove_block_txs_from_mempool(self, block: Block):
        """Remove block transactions from mempool"""
        tx_ids = [tx.tx_id for tx in block.transactions]
        await self.mempool.remove_transactions(tx_ids)

    async def _process_orphans(self):
        """Process orphan blocks that might now fit in the chain"""
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
                        # Recursively process any additional orphans
                        await self._process_orphans()
                        break

    def adjust_difficulty(self) -> int:
        """Adjust mining difficulty based on block generation time"""
        if len(self.chain) % 2016 == 0 and len(self.chain) > 1:
            # Get blocks from the current period
            period_blocks = self.chain[-2016:]
            
            # Calculate time taken to mine these blocks
            if isinstance(period_blocks[0].timestamp, str) and isinstance(period_blocks[-1].timestamp, str):
                # ISO format timestamps
                start_time = datetime.datetime.fromisoformat(period_blocks[0].timestamp)
                end_time = datetime.datetime.fromisoformat(period_blocks[-1].timestamp)
                time_taken = (end_time - start_time).total_seconds()
            else:
                # Epoch timestamps
                time_taken = period_blocks[-1].timestamp - period_blocks[0].timestamp
                
            # Target time for 2016 blocks (10 minutes per block)
            target_time = 2016 * 600  # seconds
            
            if time_taken > 0:
                # Calculate ratio of expected time to actual time
                ratio = target_time / time_taken
                
                # Adjust difficulty (with limits)
                self.difficulty = max(1, min(20, int(self.difficulty * ratio)))
                logger.info(f"Difficulty adjusted to {self.difficulty}")
                
        return self.difficulty

    def dynamic_difficulty(self) -> int:
        """Get dynamic difficulty based on network conditions"""
        # Increase difficulty if few peers to prevent attacks
        if hasattr(self.network, 'peers') and len(self.network.peers) < 3:
            return max(6, self.difficulty)
        return self.difficulty

    def get_total_difficulty(self) -> int:
        """Calculate total chain difficulty"""
        return sum(block.difficulty for block in self.chain)

    async def is_valid_chain(self, chain: List[Block]) -> bool:
        """Validate an entire blockchain"""
        # Check chain starts with valid genesis block
        if not chain or len(chain) == 0:
            return False
            
        genesis = self.create_genesis_block()
        if chain[0].hash != genesis.hash:
            return False
            
        # Validate each block in the chain
        for i in range(1, len(chain)):
            if not await self._is_valid_block(chain[i], chain[i-1]):
                return False
                
        # Check chain includes all known checkpoints
        for checkpoint in self.checkpoints:
            if checkpoint >= len(chain) or chain[checkpoint].hash != self.chain[checkpoint].hash:
                return False
                
        return True

    async def _is_valid_block(self, block: Block, prev_block: Block) -> bool:
        """Check if a block is valid and properly connected to previous block"""
        # Check block index increments by 1
        if block.index != prev_block.index + 1:
            return False
            
        # Check block links to previous block
        if block.previous_hash != prev_block.hash:
            return False
            
        # Validate proof of work
        target = "0" * block.difficulty
        if not block.hash or not block.hash.startswith(target):
            return False
            
        return True

    def halve_block_reward(self) -> None:
        """Halve the block mining reward"""
        self.current_reward /= 2
        logger.info(f"Block reward halved to {self.current_reward}")

    async def handle_potential_fork(self, block: Block) -> None:
        """Handle a block that might be part of a fork"""
        async with self._lock:
            # Ignore old blocks
            if block.index <= len(self.chain) - 1:
                return
                
            # Add to orphans if it's ahead of our chain
            if block.index > len(self.chain):
                # Remove oldest orphan if we're at capacity
                if len(self.orphans) >= self.max_orphans:
                    oldest = min(self.orphans.keys(), key=lambda k: self.orphans[k].timestamp)
                    del self.orphans[oldest]
                    
                # Add new orphan
                self.orphans[block.hash] = block
                logger.info(f"Added block {block.index} to orphan pool: {block.hash[:8]}")
                
            # Request missing blocks from network
            if self.network_service:
                await self.network_service.request_chain()

    async def replace_chain(self, new_chain: List[Block]) -> bool:
        """Replace the current chain with a longer, valid chain"""
        async with self._lock:
            # New chain must be longer than current chain
            if len(new_chain) <= len(self.chain):
                return False
                
            # New chain must be valid
            if await self.is_valid_chain(new_chain):
                logger.info(f"Replacing chain of length {len(self.chain)} with new chain of length {len(new_chain)}")
                
                # Replace chain
                self.chain = new_chain
                
                # Rebuild UTXO set
                await self._rebuild_utxo_set()
                
                # Save to storage
                await self.save_chain()
                
                # Update metrics
                self.update_metrics()
                
                return True
                
            return False

    def subscribe(self, event: str, callback: Callable) -> None:
        """Subscribe to blockchain events"""
        if event in self.listeners:
            self.listeners[event].append(callback)

    def trigger_event(self, event: str, data: Any) -> None:
        """Trigger event callbacks"""
        for callback in self.listeners[event]:
            asyncio.create_task(self._async_callback(callback, data, event))

    async def _async_callback(self, callback: Callable, data: Any, event: str) -> None:
        """Execute callback asynchronously"""
        try:
            if asyncio.iscoroutinefunction(callback):
                await callback(data)
            else:
                # Run synchronous callbacks in thread pool to avoid blocking
                await asyncio.to_thread(callback, data)
        except Exception as e:
            logger.error(f"Callback error for event {event}: {e}")

    async def validate_transaction(self, tx: Transaction) -> bool:
        """Validate a transaction before adding to mempool"""
        try:
            # Coinbase transactions are only valid in blocks
            if tx.tx_type == TransactionType.COINBASE:
                logger.warning("Attempted to validate standalone coinbase transaction")
                return False
                
            # Check transaction has inputs and outputs
            if not tx.inputs or not tx.outputs:
                return False
                
            # Verify signature
            if not await tx.verify():
                logger.warning(f"Transaction {tx.tx_id} failed signature verification")
                return False
                
            # Validate inputs
            input_sum = 0
            for tx_input in tx.inputs:
                utxo = await self.utxo_set.get_utxo(tx_input.tx_id, tx_input.output_index)
                
                # Check UTXO exists
                if not utxo:
                    logger.warning(f"Transaction {tx.tx_id} references non-existent UTXO")
                    return False
                    
                # Verify input has valid credentials
                if not tx_input.public_key:
                    logger.warning(f"Transaction {tx.tx_id} has input without public key")
                    return False
                    
                # Verify input can spend this UTXO
                address = SecurityUtils.public_key_to_address(tx_input.public_key)
                if address != utxo.recipient:
                    logger.warning(f"Transaction {tx.tx_id} has input with non-matching address")
                    return False
                    
                # Check for nonce replay
                if tx.nonce is not None and await self.utxo_set.is_nonce_used(address, tx.nonce):
                    logger.warning(f"Transaction {tx.tx_id} uses already spent nonce")
                    return False
                    
                input_sum += utxo.amount
                
            # Validate outputs
            output_sum = sum(output.amount for output in tx.outputs)
            
            # Check output sum + fee equals input sum
            if not (abs(input_sum - output_sum - tx.fee) < 0.0001):
                logger.warning(f"Transaction {tx.tx_id} has imbalanced inputs/outputs")
                return False
                
            return True
        except Exception as e:
            logger.error(f"Transaction validation failed for {tx.tx_id}: {e}")
            return False

    async def add_transaction_to_mempool(self, tx: Transaction) -> bool:
        """Add a validated transaction to the mempool"""
        async with self._mempool_lock:
            logger.debug(f"Adding tx {tx.tx_id} to mempool, signature type: {type(tx.signature)}")
            # Validate transaction
            if not await self.validate_transaction(tx):
                return False
                
            # Add to mempool
            success = await self.mempool.add_transaction(tx)
            
            if success:
                # Notify listeners
                self.trigger_event("new_transaction", tx)
                
                # Broadcast to network
                if self.network_service:
                    await self.network_service.broadcast_transaction(tx)
                    
            return success

    async def get_balance(self, address: str) -> float:
        """Get balance for an address from UTXO set"""
        return await self.utxo_set.get_balance(address)

    async def create_transaction(self, sender_private_key: str, sender_address: str, 
                               recipient_address: str, amount: float, fee: float = 0.001) -> Optional[Transaction]:
        """Create a new transaction"""
        try:
            # Get UTXOs for the sender
            sender_utxos = await self.utxo_set.get_utxos_for_address(sender_address)
            
            # Calculate total available funds
            total_available = sum(utxo[2].amount for utxo in sender_utxos)
            
            # Check if sender has enough funds
            if total_available < amount + fee:
                logger.warning(f"Insufficient funds: {total_available} < {amount + fee}")
                return None
                
            # Select UTXOs for this transaction
            selected_utxos = []
            selected_amount = 0
            
            for tx_id, output_index, utxo in sender_utxos:
                selected_utxos.append((tx_id, output_index, utxo.amount))
                selected_amount += utxo.amount
                if selected_amount >= amount + fee:
                    break
                    
            if selected_amount < amount + fee:
                logger.warning("Not enough UTXOs to cover amount")
                return None
                
            # Create key objects
            private_key = sender_private_key  # Already have the private key string
            
            # Get corresponding public key
            sk = ecdsa.SigningKey.from_string(bytes.fromhex(private_key), curve=ecdsa.SECP256k1)
            public_key = sk.get_verifying_key().to_string().hex()
            
            # Create transaction inputs
            inputs = [
                TransactionInput(
                    tx_id=tx_id, 
                    output_index=index, 
                    public_key=public_key
                ) 
                for tx_id, index, _ in selected_utxos
            ]
            
            # Create transaction outputs
            outputs = [TransactionOutput(recipient=recipient_address, amount=amount)]
            
            # Add change output if necessary
            change = selected_amount - amount - fee
            if change > 0:
                outputs.append(TransactionOutput(recipient=sender_address, amount=change))
                
            # Create transaction with unique nonce
            tx = Transaction(
                tx_type=TransactionType.TRANSFER, 
                inputs=inputs, 
                outputs=outputs, 
                fee=fee,
                nonce=int(time.time() * 1000)  # Millisecond timestamp as nonce
            )
            
            # Sign transaction
            await tx.sign(private_key)
            
            return tx
        except Exception as e:
            logger.error(f"Failed to create transaction: {e}")
            return None

    async def add_validator(self, validator_id: str, mfa_token: str = None) -> bool:
        """Add a new validator with MFA verification"""
        if not self.mfa_manager:
            logger.warning("MFA manager not configured")
            return False
            
        if not await self.mfa_manager.verify_mfa(validator_id, mfa_token):
            logger.warning(f"MFA verification failed for validator {validator_id}")
            return False
            
        # Add validator logic would go here
        return True
        
    async def update_network_config(self, config: dict, admin_id: str, mfa_token: str = None) -> bool:
        """Update network configuration with MFA verification"""
        if not self.mfa_manager:
            logger.warning("MFA manager not configured")
            return False
            
        if not await self.mfa_manager.verify_mfa(admin_id, mfa_token):
            logger.warning(f"MFA verification failed for admin {admin_id}")
            return False
            
        # Update configuration logic would go here
        return True

    async def get_all_addresses(self) -> List[str]:
        """Return list of all wallet addresses in the blockchain"""
        # Collect unique addresses from all transactions
        addresses = set()
        
        # Add addresses from chain
        for block in self.chain:
            for tx in block.transactions:
                if hasattr(tx, 'sender') and tx.sender:
                    addresses.add(tx.sender)
                if hasattr(tx, 'recipient') and tx.recipient:
                    addresses.add(tx.recipient)
                
                # Also check transaction outputs for UTXO model
                if hasattr(tx, 'outputs') and tx.outputs:
                    for output in tx.outputs:
                        if hasattr(output, 'recipient') and output.recipient:
                            addresses.add(output.recipient)
                
        # Remove genesis, null addresses
        addresses.discard("0")
        addresses.discard(None)
        addresses.discard("")
        addresses.discard("genesis")
        
        # Convert to sorted list for consistent display
        return sorted(list(addresses))

    async def get_transactions_for_address(self, address: str) -> List[Transaction]:
        """Get all transactions involving a specific address"""
        transactions = []
        
        # Get transactions from chain
        for block in self.chain:
            for tx in block.transactions:
                # Check simple sender/recipient model
                if hasattr(tx, 'sender') and tx.sender == address:
                    transactions.append(tx)
                elif hasattr(tx, 'recipient') and tx.recipient == address:
                    transactions.append(tx)
                # Check UTXO model
                elif hasattr(tx, 'inputs') and tx.inputs:
                    for inp in tx.inputs:
                        if hasattr(inp, 'public_key') and inp.public_key:
                            input_address = SecurityUtils.public_key_to_address(inp.public_key)
                            if input_address == address:
                                transactions.append(tx)
                                break
                elif hasattr(tx, 'outputs') and tx.outputs:
                    for out in tx.outputs:
                        if hasattr(out, 'recipient') and out.recipient == address:
                            transactions.append(tx)
                            break
                
        # Sort by timestamp, newest first
        transactions.sort(key=lambda x: x.timestamp, reverse=True)
        return transactions

    async def create_wallet(self) -> str:
        """Create a new wallet with public/private key pair"""
        # Generate new key pair
        private_key = ecdsa.SigningKey.generate(curve=ecdsa.SECP256k1)
        public_key = private_key.get_verifying_key()
        
        # Convert to hex strings
        private_key_hex = private_key.to_string().hex()
        public_key_hex = public_key.to_string().hex()
        
        # Create address from public key
        address = SecurityUtils.public_key_to_address(public_key_hex)
        
        # Store wallet
        self.wallets[address] = {
            'private_key': private_key_hex,
            'public_key': public_key_hex
        }
        
        # Save wallets to storage
        await self.save_wallets()
        
        return address

    async def get_wallet(self, address: str) -> Optional[Dict[str, str]]:
        """Get wallet information for an address"""
        return self.wallets.get(address)

    def get_latest_block(self) -> Optional[Block]:
        """Get the latest block in the chain"""
        return self.chain[-1] if self.chain else None

    async def start_mining(self, wallet_address: Optional[str] = None) -> bool:
        """Start mining blocks"""
        logger.info("Attempting to start mining...")
        try:
            # If no wallet address provided, use first available
            if not wallet_address:
                addresses = await self.get_all_addresses()
                if not addresses:
                    # Create a new wallet if none exist
                    wallet_address = await self.create_wallet()
                    logger.info(f"Created new mining wallet: {wallet_address}")
                else:
                    wallet_address = addresses[0]
                    
            # Start mining with chosen wallet
            result = await self.miner.start_mining(wallet_address)
            
            if result:
                logger.info(f"Mining started successfully with address: {wallet_address}")
            else:
                logger.warning("Mining failed to start")
                
            return result
        except Exception as e:
            logger.error(f"Mining error: {e}")
            return False

    async def stop_mining(self) -> bool:
        """Stop the mining process"""
        logger.info("Stopping mining...")
        try:
            # Stop the miner
            result = await self.miner.stop_mining()
            return result
        except Exception as e:
            logger.error(f"Failed to stop mining: {e}")
            return False

    async def get_hashrate(self) -> float:
        """Get current mining hashrate"""
        if hasattr(self, 'miner'):
            return self.miner.get_hashrate()
        return 0.0
        
    async def create_coinbase_transaction(self, recipient: str, amount: float, block_height: int) -> Transaction:
        """Create a coinbase transaction for mining rewards"""
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
    def __init__(self, blockchain):
        self.blockchain = blockchain
        self._running = False
        self._task = None
        
    async def start(self):
        """Start monitoring system resources"""
        if self._running:
            return
            
        self._running = True
        self._task = asyncio.create_task(self.monitor_resources())
        logger.info("Resource monitor started")
        
    async def stop(self):
        """Stop monitoring system resources"""
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Resource monitor stopped")
        
    async def monitor_resources(self):
        """Monitor system resources and adjust as needed"""
        try:
            while self._running:
                cpu_usage = psutil.cpu_percent()
                memory_usage = psutil.virtual_memory().percent
                
                # Adjust mining intensity based on system load
                if cpu_usage > 90:
                    await self.adjust_mining_intensity(high_load=True)
                elif cpu_usage < 50:
                    await self.adjust_mining_intensity(high_load=False)
                
                # Sleep between checks
                await asyncio.sleep(5)
        except asyncio.CancelledError:
            logger.info("Resource monitoring task cancelled")
        except Exception as e:
            logger.error(f"Error in resource monitoring: {e}")
            
    async def adjust_mining_intensity(self, high_load: bool):
        """Adjust mining settings based on system load"""
        if high_load:
            # Reduce mining thread count or pause temporarily
            if self.blockchain.miner.mining:
                logger.info("System under high load, reducing mining intensity")
                # Implementation would depend on miner capabilities
        else:
            # Increase mining intensity if appropriate
            pass

class BlockchainNode:
    def __init__(self):
        """Initialize a blockchain node with all services"""
        # Create blockchain instance
        self.blockchain = Blockchain(node_id=f"node-{int(time.time())}")
        
        # Create resource monitor
        self.resource_monitor = ResourceMonitor(self.blockchain)
        
        # Running flag
        self.is_running = False
        
        # GUI-related attributes
        self.mining = False
        self.status_var = None
        self.mining_btn = None
        self.notebook = None

    async def start(self):
        """Start all blockchain node services"""
        if self.is_running:
            logger.warning("BlockchainNode already running")
            return
            
        try:
            # Initialize blockchain first
            await self.blockchain.initialize()
            
            # Start resource monitor
            await self.resource_monitor.start()
            
            self.is_running = True
            logger.info("BlockchainNode started successfully")
        except Exception as e:
            logger.error(f"Failed to start BlockchainNode: {e}")
            raise

    async def stop(self):
        """Stop all services gracefully"""
        if not self.is_running:
            return
            
        try:
            # Stop mining if active
            if self.blockchain.miner.mining:
                await self.blockchain.stop_mining()
            
            # Stop resource monitor
            await self.resource_monitor.stop()
            
            # Shutdown blockchain
            await self.blockchain.shutdown()
            
            self.is_running = False
            logger.info("BlockchainNode stopped successfully")
        except Exception as e:
            logger.error(f"Error during BlockchainNode shutdown: {e}")
            raise

    async def toggle_mining(self):
        """Toggle mining state"""
        if not self.mining:
            if await self.blockchain.start_mining():
                self.mining = True
                self.update_ui_mining_active()
        else:
            if await self.blockchain.stop_mining():
                self.mining = False
                self.update_ui_mining_inactive()

    def update_ui_mining_active(self):
        """Update UI to reflect active mining state"""
        if hasattr(self, 'mining_btn') and self.mining_btn:
            self.mining_btn.configure(text="Stop Mining")
        if hasattr(self, 'status_var') and self.status_var:
            self.status_var.set("Mining active")

    def update_ui_mining_inactive(self):
        """Update UI to reflect inactive mining state"""
        if hasattr(self, 'mining_btn') and self.mining_btn:
            self.mining_btn.configure(text="Start Mining")
        if hasattr(self, 'status_var') and self.status_var:
            self.status_var.set("Mining inactive")

    def init_gui(self):
        """Initialize GUI components"""
        # Create main window
        self.root = tk.Tk()
        self.root.title("Blockchain Node")
        self.root.geometry("800x600")
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Initialize tabs
        self.init_mining_tab()
        self.init_wallet_tab()
        self.init_blockchain_tab()
        
        # Status bar
        status_frame = ttk.Frame(self.root)
        status_frame.pack(fill=tk.X, side=tk.BOTTOM, padx=10, pady=5)
        
        self.status_var = tk.StringVar()
        self.status_var.set("Blockchain node ready")
        status_label = ttk.Label(status_frame, textvariable=self.status_var)
        status_label.pack(side=tk.LEFT)
        
        # Event loop handler for async operations
        self.setup_async_handlers()

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
            command=self._handle_mining_click
        )
        self.mining_btn.grid(row=0, column=0, padx=5, pady=5)
        
        # Mining stats
        stats_frame = ttk.LabelFrame(mining_frame, text="Mining Statistics", padding="10")
        stats_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=5)
        
        # Wallet selection
        wallet_frame = ttk.LabelFrame(mining_frame, text="Mining Wallet", padding="10")
        wallet_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=5)

    def init_wallet_tab(self):
        """Initialize wallet management tab"""
        wallet_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(wallet_frame, text="Wallets")
        
        # Wallet controls and display would go here

    def init_blockchain_tab(self):
        """Initialize blockchain explorer tab"""
        explorer_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(explorer_frame, text="Explorer")
        
        # Blockchain explorer UI would go here

    def setup_async_handlers(self):
        """Set up handlers for async operations in GUI"""
        # The asyncio event loop for GUI integration
        self.loop = asyncio.get_event_loop()
        
        # Schedule periodic UI updates
        self.root.after(100, self._async_update)
        
    def _async_update(self):
        """Handle periodic UI updates and async task processing"""
        # Schedule the next update
        self.root.after(100, self._async_update)
        
        # Process any pending asyncio tasks
        self.loop.call_soon(self._process_asyncio)
        
    def _process_asyncio(self):
        """Process pending asyncio tasks"""
        # This method runs in the main thread but processes
        # any pending asyncio tasks without blocking the UI
        try:
            self.loop.stop()
            self.loop.run_forever()
        except Exception as e:
            logger.error(f"Error processing asyncio tasks: {e}")

    def _handle_mining_click(self):
        """Handle mining button click"""
        # Create a task for the toggle_mining async method
        task = asyncio.create_task(self.toggle_mining())
        
        # Add the task to the loop
        self.loop.create_task(task)

    def run(self):
        """Run the blockchain node with GUI"""
        # Start node services
        self.loop.create_task(self.start())
        
        # Start GUI main loop
        self.root.mainloop()
        
        # Ensure clean shutdown
        self.loop.create_task(self.stop())
        self.loop.run_until_complete(asyncio.sleep(1))  # Give shutdown tasks time to complete