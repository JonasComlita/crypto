"""
Python wrapper for the C++ acceleration library.
This module provides high-performance implementations of compute-intensive blockchain operations.
"""

import os
import sys
import logging
import importlib.util
from typing import List, Tuple, Optional, Dict, Any, Union

logger = logging.getLogger(__name__)

# Try to import the C++ module
try:
    import blockchain_cpp
    CPP_AVAILABLE = True
    logger.info("C++ acceleration library loaded successfully")
except ImportError:
    logger.warning("C++ acceleration library not available, falling back to Python implementations")
    CPP_AVAILABLE = False

# GPU support detection
try:
    if CPP_AVAILABLE and hasattr(blockchain_cpp, 'GPUMiner'):
        GPU_AVAILABLE = True
        logger.info("GPU mining support detected")
    else:
        GPU_AVAILABLE = False
except Exception as e:
    logger.warning(f"Error checking GPU support: {e}")
    GPU_AVAILABLE = False

# Create singleton instance of GPU miner if available
_gpu_miner = None
if GPU_AVAILABLE:
    try:
        _gpu_miner = blockchain_cpp.GPUMiner()
        logger.info("GPU miner initialized")
    except Exception as e:
        logger.error(f"Failed to initialize GPU miner: {e}")
        GPU_AVAILABLE = False

def sha256(data: str) -> str:
    """
    Calculate SHA-256 hash using C++ implementation if available.
    
    Args:
        data: Input string to hash
        
    Returns:
        Hexadecimal string of the hash
    """
    if CPP_AVAILABLE:
        return blockchain_cpp.sha256(data)
    else:
        # Fallback to Python implementation
        import hashlib
        return hashlib.sha256(data.encode()).hexdigest()

def calculate_merkle_root(tx_ids: List[str]) -> str:
    """
    Calculate Merkle root from transaction IDs using C++ implementation if available.
    
    Args:
        tx_ids: List of transaction IDs
        
    Returns:
        Merkle root hash as a hex string
    """
    if CPP_AVAILABLE:
        return blockchain_cpp.calculate_merkle_root(tx_ids)
    else:
        # Fallback to Python implementation
        if not tx_ids:
            return "0" * 64
        
        tree = tx_ids.copy()
        while len(tree) > 1:
            new_level = []
            for i in range(0, len(tree), 2):
                left = tree[i]
                right = tree[i + 1] if i + 1 < len(tree) else left
                combined = left + right
                new_level.append(sha256(combined))
            tree = new_level
        return tree[0]

def mine_block(block_string_base: str, difficulty: int, max_nonce: int = 2**32) -> Tuple[int, str, int]:
    """
    Mine a block using available acceleration (C++/GPU).
    
    Args:
        block_string_base: Base string for the block (without nonce)
        difficulty: Number of leading zeros required
        max_nonce: Maximum nonce to try
        
    Returns:
        Tuple of (nonce, hash, hashes_performed)
    """
    # Try GPU mining first if available
    if GPU_AVAILABLE and _gpu_miner is not None:
        try:
            result = _gpu_miner.mine(block_string_base, difficulty, max_nonce)
            nonce, block_hash, hashes = result
            if nonce != -1:  # Solution found
                return nonce, block_hash, hashes
            # If GPU mining fails or doesn't find a solution, fall back to CPU
            logger.info("GPU mining did not find a solution, falling back to CPU")
        except Exception as e:
            logger.error(f"GPU mining error: {e}, falling back to CPU")
    
    # CPU mining with C++ acceleration if available
    if CPP_AVAILABLE:
        return blockchain_cpp.mine_block(block_string_base, difficulty, max_nonce)
    else:
        # Fallback to Python implementation
        import hashlib
        target = "0" * difficulty
        total_hashes = 0
        
        for nonce in range(max_nonce):
            block_string = f"{block_string_base}{nonce}"
            block_hash = hashlib.sha256(block_string.encode()).hexdigest()
            total_hashes += 1
            
            if block_hash.startswith(target):
                return nonce, block_hash, total_hashes
        
        return -1, "", total_hashes

def public_key_to_address(public_key: str) -> str:
    """
    Convert a public key to a blockchain address using C++ if available.
    
    Args:
        public_key: Public key in hex format
        
    Returns:
        Blockchain address
    """
    if CPP_AVAILABLE:
        return blockchain_cpp.public_key_to_address(public_key)
    else:
        # Fallback to Python implementation
        import hashlib
        sha256_hash = hashlib.sha256(bytes.fromhex(public_key)).hexdigest()
        ripemd160 = hashlib.new('ripemd160')
        ripemd160.update(bytes.fromhex(sha256_hash))
        return f"1{ripemd160.hexdigest()[:20]}"
