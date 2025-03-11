import os
import yaml
import ecdsa
import hashlib
from enum import Enum
from dataclasses import dataclass
from typing import Dict, Optional, Any, Tuple
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import socket
from dotenv import load_dotenv
from key_rotation.core import KeyRotationManager
import logging
import time
import random

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Global KeyRotationManager instance
rotation_manager: Optional[KeyRotationManager] = None

def init_rotation_manager(node_id: str) -> None:
    """Initialize the KeyRotationManager for the node."""
    global rotation_manager
    start_time = time.time()
    rotation_manager = KeyRotationManager(node_id=node_id)
    logger.info(f"Initialized KeyRotationManager for {node_id} in {(time.time() - start_time):.3f} seconds")

def get_peer_auth_secret() -> str:
    """Retrieve the current peer authentication secret."""
    if not rotation_manager:
        raise ValueError("Rotation manager not initialized")
    start_time = time.time()
    secret = rotation_manager.get_current_auth_secret()
    logger.debug(f"Retrieved peer auth secret in {(time.time() - start_time) * 1e6:.2f} µs")
    return secret

PEER_AUTH_SECRET = get_peer_auth_secret

def validate_peer_auth(received_auth: str) -> bool:
    """Validate peer authentication against current and previous secrets."""
    if not rotation_manager:
        raise ValueError("Rotation manager not initialized")
    start_time = time.time()
    result = rotation_manager.authenticate_peer(received_auth)
    logger.debug(f"Validated peer auth in {(time.time() - start_time) * 1e6:.2f} µs")
    return result

SSL_CERT_PATH = os.getenv("SSL_CERT_PATH", "server.crt")
SSL_KEY_PATH = os.getenv("SSL_KEY_PATH", "server.key")

if not os.path.exists(SSL_CERT_PATH) or not os.path.exists(SSL_KEY_PATH):
    result = os.system(f'openssl req -x509 -newkey rsa:2048 -keyout "{SSL_KEY_PATH}" -out "{SSL_CERT_PATH}" -days 365 -nodes -subj "/CN=localhost"')
    if result != 0:
        raise RuntimeError("Failed to generate SSL certificate with OpenSSL")

def generate_node_keypair() -> Tuple[str, str]:
    """Generate an ECDSA key pair for node identity."""
    private_key = ecdsa.SigningKey.generate(curve=ecdsa.SECP256k1)
    public_key = private_key.get_verifying_key()
    return private_key.to_string().hex(), public_key.to_string().hex()

def load_config(config_file: str = "config.yaml") -> Dict[str, Any]:
    """Load and validate configuration from a YAML file."""
    default_config = {
        "difficulty": 4,
        "current_reward": 50.0,
        "halving_interval": 210000,
        "mempool_max_size": 1000,
        "max_retries": 3,
        "sync_interval": 300
    }
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f) or {}
        for key, value in default_config.items():
            config.setdefault(key, value)
        return config
    except FileNotFoundError:
        logger.warning(f"Config file {config_file} not found, using defaults")
        return default_config
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        return default_config

CONFIG = load_config()

class TransactionType(Enum):
    """Enum representing types of transactions."""
    COINBASE = "coinbase"
    TRANSFER = "transfer"  # Renamed from REGULAR for clarity

@dataclass
class TransactionOutput:
    """Represents an output in a transaction."""
    recipient: str
    amount: float
    script: str = "P2PKH"

    def to_dict(self) -> Dict[str, Any]:
        return {"recipient": self.recipient, "amount": self.amount, "script": self.script}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TransactionOutput':
        return cls(recipient=data["recipient"], amount=data["amount"], script=data.get("script", "P2PKH"))

@dataclass
class TransactionInput:
    """Represents an input in a transaction."""
    tx_id: str
    output_index: int
    public_key: Optional[str] = None
    signature: Optional[bytes] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tx_id": self.tx_id,
            "output_index": self.output_index,
            "public_key": self.public_key,
            "signature": self.signature.hex() if self.signature else None
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TransactionInput':
        signature = bytes.fromhex(data["signature"]) if data.get("signature") else None
        return cls(tx_id=data["tx_id"], output_index=data["output_index"], 
                  public_key=data.get("public_key"), signature=signature)

class SecurityUtils:
    """Utility class for cryptographic operations."""
    @staticmethod
    def generate_keypair() -> Tuple[str, str]:
        """Generate an ECDSA key pair."""
        try:
            private_key = ecdsa.SigningKey.generate(curve=ecdsa.SECP256k1)
            public_key = private_key.get_verifying_key()
            return private_key.to_string().hex(), public_key.to_string().hex()
        except Exception as e:
            logger.error(f"Failed to generate keypair: {e}")
            raise

    @staticmethod
    def public_key_to_address(public_key: str) -> str:
        """Convert a public key to a blockchain address."""
        try:
            pub_bytes = bytes.fromhex(public_key)
            sha256_hash = hashlib.sha256(pub_bytes).hexdigest()
            ripemd160_hash = hashlib.new('ripemd160', bytes.fromhex(sha256_hash)).hexdigest()
            return f"1{ripemd160_hash[:20]}"
        except Exception as e:
            logger.error(f"Failed to convert public key to address: {e}")
            raise

def generate_wallet() -> Dict[str, str]:
    """Generate a wallet with private key, public key, and address."""
    try:
        private_key, public_key = SecurityUtils.generate_keypair()
        address = SecurityUtils.public_key_to_address(public_key)
        return {"address": address, "private_key": private_key, "public_key": public_key}
    except Exception as e:
        logger.error(f"Failed to generate wallet: {e}")
        raise

def derive_key(password: str, salt: Optional[bytes] = None) -> Tuple[bytes, bytes]:
    """Derive an encryption key from a password using PBKDF2."""
    try:
        if not salt:
            salt = os.urandom(16)
        kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=salt, iterations=100000)
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        return key, salt
    except Exception as e:
        logger.error(f"Failed to derive key: {e}")
        raise

def is_port_available(port: int, host: str = 'localhost') -> bool:
    """Check if a port is available on the host."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex((host, port)) != 0
    except Exception as e:
        logger.warning(f"Error checking port {port} availability: {e}")
        return False

def find_available_port(start_port: int = 1024, end_port: int = 65535, host: str = 'localhost') -> int:
    """Find an available port within a range."""
    try:
        port = random.randint(start_port, end_port)
        attempts = 0
        max_attempts = 100  # Prevent infinite loops
        while not is_port_available(port, host) and attempts < max_attempts:
            port = random.randint(start_port, end_port)
            attempts += 1
        if attempts >= max_attempts:
            raise RuntimeError(f"No available port found between {start_port} and {end_port}")
        return port
    except Exception as e:
        logger.error(f"Failed to find available port: {e}")
        raise