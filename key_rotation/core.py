import sys
import os
import asyncio
import base64
import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.fernet import Fernet
from aiohttp import ClientSession, ClientTimeout
from security import KeyBackupManager
from dotenv import load_dotenv
from blockchain import Blockchain

logger = logging.getLogger(__name__)

load_dotenv()
KEY_ROTATION_INTERVAL_DAYS = int(os.getenv("KEY_ROTATION_INTERVAL_DAYS", 30))
CERT_VALIDITY_DAYS = int(os.getenv("CERT_VALIDITY_DAYS", 365))
VOTE_THRESHOLD_PERCENT = float(os.getenv("VOTE_THRESHOLD_PERCENT", 66))
VOTE_TIMEOUT_HOURS = int(os.getenv("VOTE_TIMEOUT_HOURS", 48))
BACKUP_PASSWORD = os.getenv("BACKUP_PASSWORD")  # Set in .env or prompt in production

class SecureStorage:
    def __init__(self, backup_dir: str = "data/key_storage"):
        self._data: Dict[str, Dict[str, str]] = {}
        self._lock = asyncio.Lock()
        self._fernet = Fernet(base64.urlsafe_b64encode(os.urandom(32)))
        self._backup_dir = backup_dir
        os.makedirs(backup_dir, exist_ok=True)
        asyncio.create_task(self._load_from_disk())  # Load at startup

    async def store(self, key: str, value: str, namespace: str = "default") -> None:
        async with self._lock:
            if namespace not in self._data:
                self._data[namespace] = {}
            encrypted_value = self._fernet.encrypt(value.encode()).decode()
            self._data[namespace][key] = encrypted_value
            await self._save_to_disk()

    async def retrieve(self, key: str, namespace: str = "default") -> Optional[str]:
        async with self._lock:
            if namespace in self._data and key in self._data[namespace]:
                return self._fernet.decrypt(self._data[namespace][key].encode()).decode()
            return None

    async def delete(self, key: str, namespace: str = "default") -> None:
        async with self._lock:
            if namespace in self._data and key in self._data[namespace]:
                del self._data[namespace][key]
                await self._save_to_disk()

    async def _save_to_disk(self) -> None:
        try:
            with open(os.path.join(self._backup_dir, "storage.enc"), "wb") as f:
                f.write(self._fernet.encrypt(json.dumps(self._data).encode()))
        except Exception as e:
            logger.error(f"Failed to save storage: {e}")

    async def _load_from_disk(self) -> None:
        try:
            file_path = os.path.join(self._backup_dir, "storage.enc")
            if os.path.exists(file_path):
                with open(file_path, "rb") as f:
                    self._data = json.loads(self._fernet.decrypt(f.read()).decode())
        except Exception as e:
            logger.error(f"Failed to load storage: {e}")

class PKIManager:
    """Manages PKI certificates and keys."""
    def __init__(self, node_id: str):
        self._node_id = node_id
        self._private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        self._public_key = self._private_key.public_key()
        self._certificate = self._generate_self_signed_cert()

    def _generate_self_signed_cert(self) -> x509.Certificate:
        """Generate a self-signed certificate."""
        try:
            subject = issuer = x509.Name([
                x509.NameAttribute(x509.NameOID.COMMON_NAME, f"node-{self._node_id}")
            ])
            cert = (
                x509.CertificateBuilder()
                .subject_name(subject)
                .issuer_name(issuer)
                .public_key(self._public_key)
                .serial_number(x509.random_serial_number())
                .not_valid_before(datetime.utcnow())
                .not_valid_after(datetime.utcnow() + timedelta(days=CERT_VALIDITY_DAYS))
                .sign(self._private_key, hashes.SHA256())
            )
            return cert
        except Exception as e:
            logger.error(f"Failed to generate certificate: {e}")
            raise

    async def encrypt_message(self, message: str, public_key_pem: str) -> str:
        """Encrypt a message using a public key."""
        try:
            public_key = serialization.load_pem_public_key(public_key_pem.encode())
            encrypted = public_key.encrypt(
                message.encode(),
                padding.OAEP(mgf=padding.MGF1(hashes.SHA256()), algorithm=hashes.SHA256(), label=None)
            )
            return base64.b64encode(encrypted).decode()
        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            raise

    async def decrypt_message(self, encrypted_message: str) -> str:
        """Decrypt a message using the private key."""
        try:
            decrypted = self._private_key.decrypt(
                base64.b64decode(encrypted_message),
                padding.OAEP(mgf=padding.MGF1(hashes.SHA256()), algorithm=hashes.SHA256(), label=None)
            )
            return decrypted.decode()
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            raise

    async def sign_message(self, message: str) -> str:
        """Sign a message with the private key."""
        try:
            signature = self._private_key.sign(
                message.encode(),
                padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH),
                hashes.SHA256()
            )
            return base64.b64encode(signature).decode()
        except Exception as e:
            logger.error(f"Signing failed: {e}")
            raise

    async def verify_signature(self, message: str, signature: str, public_key_pem: str) -> bool:
        """Verify a signature using a public key."""
        try:
            public_key = serialization.load_pem_public_key(public_key_pem.encode())
            public_key.verify(
                base64.b64decode(signature),
                message.encode(),
                padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH),
                hashes.SHA256()
            )
            return True
        except Exception as e:
            logger.error(f"Signature verification failed: {e}")
            return False

    def get_public_key_pem(self) -> str:
        return self._public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        ).decode()

    def get_certificate_pem(self) -> str:
        return self._certificate.public_bytes(serialization.Encoding.PEM).decode()

class NodeRegistry:
    """In-memory node registry."""
    def __init__(self):
        self._nodes: Dict[str, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()

    async def register_node(self, node_id: str, node_url: str, public_key: str, certificate: str) -> bool:
        """Register a new node."""
        async with self._lock:
            try:
                self._nodes[node_id] = {
                    "url": node_url,
                    "public_key": public_key,
                    "certificate": certificate,
                    "last_seen": datetime.utcnow().isoformat()
                }
                logger.info(f"Registered node {node_id}")
                return True
            except Exception as e:
                logger.error(f"Failed to register node {node_id}: {e}")
                return False

    async def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        async with self._lock:
            return self._nodes.get(node_id)

    async def get_node_public_key(self, node_id: str) -> Optional[str]:
        async with self._lock:
            node = self._nodes.get(node_id)
            return node.get("public_key") if node else None

    async def get_all_nodes(self) -> Dict[str, Dict[str, Any]]:
        async with self._lock:
            return self._nodes.copy()

    async def get_active_nodes(self) -> Dict[str, Dict[str, Any]]:
        async with self._lock:
            now = datetime.utcnow()
            return {
                node_id: data for node_id, data in self._nodes.items()
                if (now - datetime.fromisoformat(data["last_seen"])).total_seconds() < 86400
            }

    async def update_node_last_seen(self, node_id: str) -> None:
        async with self._lock:
            if node_id in self._nodes:
                self._nodes[node_id]["last_seen"] = datetime.utcnow().isoformat()

class ConsensusManager:
    """Manages consensus for key rotation."""
    def __init__(self, node_id: str, node_registry: 'NodeRegistry', pki: PKIManager):
        self._node_id = node_id
        self._node_registry = node_registry
        self._pki = pki
        self._active_proposals: Dict[str, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()

    async def create_proposal(self, key_hash: str) -> Optional[str]:
        """Create a new key rotation proposal."""
        async with self._lock:
            try:
                proposal_id = str(uuid.uuid4())
                signature = await self._pki.sign_message(key_hash)
                proposal = {
                    "id": proposal_id,
                    "type": "key_rotation",
                    "key_hash": key_hash,
                    "proposer": self._node_id,
                    "proposer_signature": signature,
                    "timestamp": datetime.utcnow().isoformat(),
                    "expiration": (datetime.utcnow() + timedelta(hours=VOTE_TIMEOUT_HOURS)).isoformat(),
                    "votes": {self._node_id: {"approved": True, "timestamp": datetime.utcnow().isoformat(), "signature": signature}},
                    "finalized": False
                }
                self._active_proposals[proposal_id] = proposal
                logger.info(f"Created proposal {proposal_id}")
                return proposal_id
            except Exception as e:
                logger.error(f"Failed to create proposal: {e}")
                return None

    async def vote_on_proposal(self, proposal_id: str, approve: bool) -> bool:
        """Vote on a key rotation proposal."""
        async with self._lock:
            try:
                proposal = self._active_proposals.get(proposal_id)
                if not proposal or self._node_id in proposal.get("votes", {}):
                    return False
                if datetime.utcnow() > datetime.fromisoformat(proposal["expiration"]):
                    return False
                
                signature = await self._pki.sign_message(f"{proposal_id}:{approve}")
                proposal["votes"][self._node_id] = {
                    "approved": approve,
                    "timestamp": datetime.utcnow().isoformat(),
                    "signature": signature
                }
                logger.info(f"Voted {approve} on proposal {proposal_id}")
                return True
            except Exception as e:
                logger.error(f"Failed to vote on proposal {proposal_id}: {e}")
                return False

    async def check_proposal_status(self, proposal_id: str) -> Dict[str, Any]:
        """Check the status of a proposal."""
        async with self._lock:
            try:
                proposal = self._active_proposals.get(proposal_id)
                if not proposal:
                    return {"error": "Proposal not found"}
                
                approval_count = sum(1 for v in proposal.get("votes", {}).values() if v["approved"])
                total_votes = len(proposal.get("votes", {}))
                active_nodes = len(await self._node_registry.get_active_nodes())
                approval_percentage = (approval_count / total_votes * 100) if total_votes > 0 else 0
                threshold_reached = (approval_count / active_nodes * 100 >= VOTE_THRESHOLD_PERCENT) if active_nodes > 0 else False
                expired = datetime.utcnow() > datetime.fromisoformat(proposal["expiration"])
                
                return {
                    "id": proposal_id,
                    "type": proposal.get("type"),
                    "key_hash": proposal.get("key_hash"),
                    "proposer": proposal.get("proposer"),
                    "timestamp": proposal.get("timestamp"),
                    "expiration": proposal.get("expiration"),
                    "total_votes": total_votes,
                    "approval_count": approval_count,
                    "approval_percentage": approval_percentage,
                    "active_nodes": active_nodes,
                    "threshold_reached": threshold_reached,
                    "expired": expired,
                    "finalized": proposal.get("finalized", False)
                }
            except Exception as e:
                logger.error(f"Failed to check proposal status {proposal_id}: {e}")
                return {"error": str(e)}

    async def finalize_proposal(self, proposal_id: str) -> Tuple[bool, Optional[str]]:
        """Finalize an approved proposal."""
        async with self._lock:
            try:
                status = await self.check_proposal_status(proposal_id)
                if "error" in status or status["finalized"] or not status["threshold_reached"] or status["expired"]:
                    return False, None
                
                proposal = self._active_proposals[proposal_id]
                proposal["finalized"] = True
                proposal["finalized_timestamp"] = datetime.utcnow().isoformat()
                key_hash = proposal["key_hash"]
                del self._active_proposals[proposal_id]
                logger.info(f"Finalized proposal {proposal_id}")
                return True, key_hash
            except Exception as e:
                logger.error(f"Failed to finalize proposal {proposal_id}: {e}")
                return False, None

    async def get_active_proposals(self) -> List[Dict[str, Any]]:
        """Get all active proposals."""
        async with self._lock:
            return [
                await self.check_proposal_status(pid)
                for pid in self._active_proposals
                if not self._active_proposals[pid]["finalized"] and datetime.utcnow() < datetime.fromisoformat(self._active_proposals[pid]["expiration"])
            ]

    async def cleanup_expired_proposals(self) -> None:
        """Clean up expired proposals."""
        async with self._lock:
            to_remove = [
                pid for pid, proposal in self._active_proposals.items()
                if datetime.utcnow() > datetime.fromisoformat(proposal["expiration"]) and not proposal["finalized"]
            ]
            for pid in to_remove:
                del self._active_proposals[pid]
            logger.debug(f"Cleaned up {len(to_remove)} expired proposals")

class P2PNetwork:
    """Async P2P network for key rotation."""
    def __init__(self, node_id: str, node_registry: NodeRegistry, pki: PKIManager, consensus: ConsensusManager):
        self._node_id = node_id
        self._node_registry = node_registry
        self._pki = pki
        self._consensus = consensus
        self._message_cache: set = set()
        self._lock = asyncio.Lock()
        self._session: Optional[ClientSession] = None

    async def start(self) -> None:
        """Start the P2P network."""
        try:
            self._session = ClientSession(timeout=ClientTimeout(total=5))
            logger.info("P2P network started")
        except Exception as e:
            logger.error(f"Failed to start P2P network: {e}")
            raise

    async def stop(self) -> None:
        if self._session:
            await self._session.close()
        self._message_cache.clear()
        logger.info("P2P network stopped")

    async def broadcast_proposal(self, proposal_id: str) -> bool:
        """Broadcast a new proposal."""
        try:
            status = await self._consensus.check_proposal_status(proposal_id)
            if "error" in status:
                return False
            message = {"type": "new_proposal", "proposal_id": proposal_id, "sender": self._node_id, "timestamp": datetime.utcnow().isoformat()}
            return await self._broadcast_message(message)
        except Exception as e:
            logger.error(f"Failed to broadcast proposal {proposal_id}: {e}")
            return False

    async def broadcast_vote(self, proposal_id: str, approved: bool) -> bool:
        """Broadcast a vote."""
        try:
            message = {"type": "vote", "proposal_id": proposal_id, "approved": approved, "sender": self._node_id, "timestamp": datetime.utcnow().isoformat()}
            return await self._broadcast_message(message)
        except Exception as e:
            logger.error(f"Failed to broadcast vote for {proposal_id}: {e}")
            return False

    async def broadcast_finalized_key(self, proposal_id: str, key_hash: str, encrypted_keys: Dict[str, str]) -> bool:
        """Broadcast a finalized key."""
        try:
            message = {
                "type": "finalized_key",
                "proposal_id": proposal_id,
                "key_hash": key_hash,
                "encrypted_keys": encrypted_keys,
                "sender": self._node_id,
                "timestamp": datetime.utcnow().isoformat()
            }
            return await self._broadcast_message(message)
        except Exception as e:
            logger.error(f"Failed to broadcast finalized key for {proposal_id}: {e}")
            return False

    async def process_message(self, message: Dict[str, Any], signature: str, sender_id: str) -> bool:
        """Process a received P2P message."""
        async with self._lock:
            try:
                sender_public_key = await self._node_registry.get_node_public_key(sender_id)
                if not sender_public_key or not await self._pki.verify_signature(json.dumps(message, sort_keys=True), signature, sender_public_key):
                    return False
                
                message_id = hashlib.sha256(json.dumps(message, sort_keys=True).encode()).hexdigest()
                if message_id in self._message_cache:
                    return True
                
                self._message_cache.add(message_id)
                if len(self._message_cache) > 1000:
                    self._message_cache.pop()
                
                await self._node_registry.update_node_last_seen(sender_id)
                message_type = message.get("type")
                
                if message_type == "new_proposal":
                    logger.info(f"Received new proposal {message['proposal_id']} from {sender_id}")
                    return True
                elif message_type == "vote":
                    await self._consensus.vote_on_proposal(message["proposal_id"], message["approved"])
                    logger.info(f"Processed vote from {sender_id} for {message['proposal_id']}")
                    return True
                elif message_type == "finalized_key" and sender_id in message["encrypted_keys"]:
                    logger.info(f"Received finalized key for {message['proposal_id']} from {sender_id}")
                    return True
                return False
            except Exception as e:
                logger.error(f"Failed to process message: {e}")
                return False

    async def _broadcast_message(self, message: Dict[str, Any]) -> bool:
        """Broadcast a message to all nodes."""
        async with self._lock:
            try:
                signature = await self._pki.sign_message(json.dumps(message, sort_keys=True))
                payload = {"message": message, "signature": signature}
                nodes = await self._node_registry.get_all_nodes()
                tasks = [
                    self._send_to_node(node_data["url"], payload)
                    for node_id, node_data in nodes.items()
                    if node_id != self._node_id and (datetime.utcnow() - datetime.fromisoformat(node_data["last_seen"])).total_seconds() < 86400
                ]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                success_count = sum(1 for r in results if not isinstance(r, Exception))
                logger.info(f"Broadcasted to {success_count}/{len(tasks)} nodes")
                return success_count >= len(nodes) / 2
            except Exception as e:
                logger.error(f"Failed to broadcast message: {e}")
                return False

    async def _send_to_node(self, url: str, payload: Dict[str, Any]) -> None:
        """Send a message to a specific node."""
        try:
            async with self._session.post(f"{url}/api/v1/p2p/message", json=payload) as resp:
                if resp.status != 200:
                    raise Exception(f"Failed with status {resp.status}")
        except Exception as e:
            logger.warning(f"Failed to send to {url}: {e}")
            raise

class KeyRotationManager:
    """Manages secure key rotation with PKI and distributed consensus."""
    def __init__(self, node_id: str, is_validator: bool = False, backup_manager: Optional[KeyBackupManager] = None, blockchain: Blockchain = None):
        self.node_id = node_id
        self.is_validator = is_validator
        self.backup_manager = backup_manager
        self.blockchain = blockchain
        self._secure_storage = SecureStorage()
        self._pki = PKIManager(self.node_id)
        self._node_registry = NodeRegistry()
        self._consensus = ConsensusManager(self.node_id, self._node_registry, self._pki)
        self._p2p = P2PNetwork(self.node_id, self._node_registry, self._pki, self._consensus)
        self._current_auth_secret: Optional[str] = None
        self._previous_auth_secret: Optional[str] = None
        self._pending_auth_secret: Optional[str] = None
        self._pending_proposal_id: Optional[str] = None
        self._lock = asyncio.Lock()
        self._running = False
        self._scheduler_task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        await self._load_auth_secrets()
        await self._p2p.start()
        self._running = True
        self._scheduler_task = asyncio.create_task(self._run_scheduler())
        logger.info(f"KeyRotationManager started for node {self.node_id}")

    async def stop(self) -> None:
        self._running = False
        if self._scheduler_task:
            self._scheduler_task.cancel()
            try:
                await self._scheduler_task
            except asyncio.CancelledError:
                pass
        await self._p2p.stop()
        logger.info("KeyRotationManager stopped")

    async def _load_auth_secrets(self) -> None:
        """Load or initialize authentication secrets."""
        async with self._lock:
            try:
                self._current_auth_secret = await self._secure_storage.retrieve("current_auth_secret")
                if not self._current_auth_secret:
                    self._current_auth_secret = await self.generate_secure_secret()
                    await self._secure_storage.store("current_auth_secret", self._current_auth_secret)
                self._previous_auth_secret = await self._secure_storage.retrieve("previous_auth_secret")
                self._pending_auth_secret = await self._secure_storage.retrieve("pending_auth_secret")
                self._pending_proposal_id = await self._secure_storage.retrieve("pending_proposal_id")
                logger.info("Loaded authentication secrets")
            except Exception as e:
                logger.error(f"Failed to load auth secrets: {e}")
                raise

    async def generate_secure_secret(self, length: int = 64) -> str:
        """Generate a secure random secret."""
        try:
            return base64.b64encode(os.urandom(length)).decode()
        except Exception as e:
            logger.error(f"Failed to generate secret: {e}")
            raise

    def hash_secret(self, secret: str) -> str:
        """Hash a secret for verification."""
        try:
            return hashlib.sha256(secret.encode()).hexdigest()
        except Exception as e:
            logger.error(f"Failed to hash secret: {e}")
            raise

    async def _run_scheduler(self) -> None:
        """Run periodic key rotation tasks."""
        while self._running:
            try:
                if self.is_validator:
                    await self._initiate_key_rotation()
                await self._check_proposals()
                await self._consensus.cleanup_expired_proposals()
                await asyncio.sleep(3600)  # Check hourly
            except Exception as e:
                logger.error(f"Scheduler error: {e}")
                await asyncio.sleep(60)

    async def _initiate_key_rotation(self) -> None:
        """Initiate a key rotation proposal."""
        async with self._lock:
            if self._pending_proposal_id:
                return
            try:
                new_secret = await self.generate_secure_secret()
                key_hash = self.hash_secret(new_secret)
                self._pending_auth_secret = new_secret
                await self._secure_storage.store("pending_auth_secret", new_secret)
                proposal_id = await self._consensus.create_proposal(key_hash)
                if proposal_id:
                    self._pending_proposal_id = proposal_id
                    await self._secure_storage.store("pending_proposal_id", proposal_id)
                    await self._p2p.broadcast_proposal(proposal_id)
                    logger.info(f"Initiated key rotation with proposal {proposal_id}")
            except Exception as e:
                logger.error(f"Failed to initiate key rotation: {e}")

    async def _check_proposals(self) -> None:
        """Check and act on active proposals."""
        async with self._lock:
            try:
                if self._pending_proposal_id:
                    status = await self._consensus.check_proposal_status(self._pending_proposal_id)
                    if status.get("threshold_reached") and not status.get("finalized"):
                        success, _ = await self._consensus.finalize_proposal(self._pending_proposal_id)
                        if success:
                            await self.apply_key_rotation()
                            await self.distribute_finalized_key()
                
                if self.is_validator:
                    for proposal in await self._consensus.get_active_proposals():
                        if self.node_id not in proposal.get("votes", {}):
                            await self._consensus.vote_on_proposal(proposal["id"], True)
                            await self._p2p.broadcast_vote(proposal["id"], True)
            except Exception as e:
                logger.error(f"Failed to check proposals: {e}")

    async def apply_key_rotation(self) -> None:
        async with self._lock:
            try:
                if not self._pending_auth_secret or not self._pending_proposal_id:
                    logger.error("Missing pending secret or proposal ID")
                    return
                self._previous_auth_secret = self._current_auth_secret
                self._current_auth_secret = self._pending_auth_secret
                await self._secure_storage.store("previous_auth_secret", self._previous_auth_secret)
                await self._secure_storage.store("current_auth_secret", self._current_auth_secret)
                await self._secure_storage.delete("pending_auth_secret")
                await self._secure_storage.delete("pending_proposal_id")
                self._pending_auth_secret = None
                self._pending_proposal_id = None
                if self.blockchain:
                    await self.blockchain.trigger_event("key_rotated", {"new_key_hash": self.hash_secret(self._current_auth_secret)})
                logger.info("Applied key rotation")
            except Exception as e:
                logger.error(f"Failed to apply key rotation: {e}")

    async def distribute_finalized_key(self) -> None:
        """Distribute the finalized key to all nodes."""
        async with self._lock:
            try:
                if not self._current_auth_secret or not self._pending_proposal_id:
                    return
                encrypted_keys = {}
                for node_id, node_data in (await self._node_registry.get_all_nodes()).items():
                    if node_id != self.node_id:
                        encrypted_keys[node_id] = await self._pki.encrypt_message(self._current_auth_secret, node_data["public_key"])
                key_hash = self.hash_secret(self._current_auth_secret)
                await self._p2p.broadcast_finalized_key(self._pending_proposal_id, key_hash, encrypted_keys)
                logger.info(f"Distributed finalized key for proposal {self._pending_proposal_id}")
            except Exception as e:
                logger.error(f"Failed to distribute finalized key: {e}")

    async def receive_key(self, encrypted_key: str) -> bool:
        """Receive and apply a new encrypted key."""
        async with self._lock:
            try:
                decrypted_key = await self._pki.decrypt_message(encrypted_key)
                if decrypted_key == self._current_auth_secret:
                    return True
                self._previous_auth_secret = self._current_auth_secret
                self._current_auth_secret = decrypted_key
                await self._secure_storage.store("previous_auth_secret", self._previous_auth_secret)
                await self._secure_storage.store("current_auth_secret", self._current_auth_secret)
                logger.info("Received and applied new key")
                return True
            except Exception as e:
                logger.error(f"Failed to receive key: {e}")
                return False

    async def authenticate_peer(self, provided_secret: str) -> bool:
        """Authenticate a peer."""
        async with self._lock:
            try:
                if not self._current_auth_secret:
                    return False
                if provided_secret == self._current_auth_secret:
                    return True
                if self._previous_auth_secret and (time.time() - float(await self._secure_storage.retrieve("last_rotation_time") or "0")) < 172800:
                    return provided_secret == self._previous_auth_secret
                return False
            except Exception as e:
                logger.error(f"Authentication failed: {e}")
                return False

    async def get_current_auth_secret(self) -> str:
        """Get the current auth secret."""
        async with self._lock:
            if not self._current_auth_secret:
                raise ValueError("Current auth secret not initialized")
            return self._current_auth_secret

    async def propose_key_rotation(self, new_key: str, key_hash: str) -> Optional[str]:
        """Propose a key rotation."""
        async with self._lock:
            try:
                self._pending_auth_secret = new_key
                await self._secure_storage.store("pending_auth_secret", new_key)
                proposal_id = await self._consensus.create_proposal(key_hash)
                if proposal_id:
                    self._pending_proposal_id = proposal_id
                    await self._secure_storage.store("pending_proposal_id", proposal_id)
                return proposal_id
            except Exception as e:
                logger.error(f"Failed to propose key rotation: {e}")
                return None

    async def rotate_keys(self) -> None:
        """Rotate keys with automatic backup"""
        async with self._lock:
            try:
                if not self._pending_auth_secret or not self._pending_proposal_id:
                    logger.error("No pending rotation to apply")
                    return
                self._previous_auth_secret = self._current_auth_secret
                self._current_auth_secret = self._pending_auth_secret
                await self._secure_storage.store("previous_auth_secret", self._previous_auth_secret)
                await self._secure_storage.store("current_auth_secret", self._current_auth_secret)
                await self._secure_storage.store("last_rotation_time", str(time.time()))
                await self._secure_storage.delete("pending_auth_secret")
                await self._secure_storage.delete("pending_proposal_id")
                if self.backup_manager:
                    password = BACKUP_PASSWORD or await self.get_backup_password()
                    await self.backup_manager.create_backup(
                        keys={
                            "public_key": self._pki.get_public_key_pem(),
                            "private_key": self._pki._private_key.private_bytes(
                                encoding=serialization.Encoding.PEM,
                                format=serialization.PrivateFormat.PKCS8,
                                encryption_algorithm=serialization.NoEncryption()
                            ).decode(),
                            "rotation_time": datetime.utcnow().isoformat()
                        },
                        password=password
                    )
                self._pending_auth_secret = None
                self._pending_proposal_id = None
                if self.blockchain:
                    await self.blockchain.trigger_event("key_rotated", {"new_key_hash": self.hash_secret(self._current_auth_secret)})
                logger.info("Keys rotated and backed up")
            except Exception as e:
                logger.error(f"Key rotation failed: {e}")
                raise

    async def get_backup_password(self) -> str:
        """Prompt for backup password in production if not set in env"""
        if "pytest" in sys.modules:  # For testing
            return "test_password"
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: input("Enter backup password: "))

