import aiohttp
import aiohttp.web
from aiohttp import web
import asyncio
import ssl
import ecdsa
import os
import logging
import time
import random
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING
from collections import defaultdict
from blockchain import Blockchain, Block, Transaction
from utils import PEER_AUTH_SECRET, SSL_CERT_PATH, SSL_KEY_PATH, generate_node_keypair, validate_peer_auth, SecurityUtils
from utils import BLOCKS_RECEIVED, TXS_BROADCAST, PEER_FAILURES, BLOCKS_MINED, BLOCK_HEIGHT, PEER_COUNT, ACTIVE_REQUESTS, safe_gauge, safe_counter
from security import SecurityMonitor
from security.mfa import MFAManager
import json
import threading
from dataclasses import dataclass
from blockchain import Blockchain, Block, Transaction

# Configure logging
logger = logging.getLogger("BlockchainNetwork")
logging.basicConfig(level=logging.DEBUG)

# Configuration
CONFIG = {
    "sync_interval": 10,           # Seconds between sync attempts
    "max_peers": 10,               # Maximum number of peers to maintain
    "peer_discovery_interval": 60, # Seconds between peer discovery cycles
    "max_retries": 3,              # Retry attempts for failed requests
    "isolation_timeout": 300,      # Timeout before considering node isolated (seconds)
    "tls_cert_file": "cert.pem",   # TLS certificate file
    "tls_key_file": "key.pem"      # TLS key file
}

async def rate_limit_middleware(app: aiohttp.web.Application, handler: callable) -> callable:
    """Simple rate-limiting middleware (placeholder for expansion)."""
    async def middleware(request: web.Request) -> web.Response:
        # Future: Add IP-based or token-based rate limiting
        return await handler(request)
    return middleware

class PeerReputation:
    def __init__(self):
        self.reputation_scores = defaultdict(lambda: 100)  # Start with 100 points
        self.violation_weights = {
            'invalid_transaction': -10,
            'invalid_block': -20,
            'failed_auth': -15,
            'rate_limit_exceeded': -5,
            'successful_transaction': 1,
            'successful_block': 2
        }
        
    def update_reputation(self, peer_id: str, event: str) -> int:
        """Update peer reputation based on events"""
        self.reputation_scores[peer_id] += self.violation_weights.get(event, 0)
        self.reputation_scores[peer_id] = max(0, min(100, self.reputation_scores[peer_id]))
        return self.reputation_scores[peer_id]
    
    def is_peer_trusted(self, peer_id: str, minimum_score: int = 50) -> bool:
        return self.reputation_scores[peer_id] >= minimum_score

class RateLimiter:
    def __init__(self):
        self.request_counts = defaultdict(lambda: defaultdict(int))
        self.last_reset = defaultdict(float)
        
        # Configure limits for different operations
        self.limits = {
            'transaction': {'count': 100, 'window': 60},  # 100 transactions per minute
            'block': {'count': 10, 'window': 60},        # 10 blocks per minute
            'peer_connect': {'count': 5, 'window': 60},  # 5 connection attempts per minute
        }

    async def check_rate_limit(self, peer_id: str, operation: str) -> bool:
        current_time = time.time()
        window = self.limits[operation]['window']
        
        # Reset counters if window has passed
        if current_time - self.last_reset[peer_id] > window:
            self.request_counts[peer_id] = defaultdict(int)
            self.last_reset[peer_id] = current_time
        
        # Check if limit is exceeded
        if self.request_counts[peer_id][operation] >= self.limits[operation]['count']:
            return False
        
        self.request_counts[peer_id][operation] += 1
        return True

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

class BlockchainNetwork:
    """Manages peer-to-peer networking for the blockchain with enhanced security and reliability."""
    def __init__(self, blockchain: 'Blockchain', node_id: str, host: str, port: int, 
                 bootstrap_nodes: Optional[List[Tuple[str, int]]] = None, security_monitor=None):
        self.blockchain = blockchain
        self.node_id = node_id
        self.host = host
        self.port = port
        self.bootstrap_nodes = bootstrap_nodes or []
        self.security_monitor = security_monitor
        self.shutdown_flag = False
        self.loop = None or asyncio.get_event_loop()
        self.private_key, self.public_key = generate_node_keypair()
        self.peers = {}
        self.app = web.Application(middlewares=[rate_limit_middleware])
        self.setup_routes()
        self.sync_task: Optional[asyncio.Task] = None
        self.discovery_task: Optional[asyncio.Task] = None
        self.last_announcement = 0
        self.peer_failures: Dict[str, int] = defaultdict(int)
        self.start_time = time.time()
        self.lock = threading.Lock() 
        self.active_requests = ACTIVE_REQUESTS
        self.active_requests.labels(instance=self.node_id).set(0)
        self.peer_reputation = PeerReputation()
        self.rate_limiter = RateLimiter()
        self.nonce_tracker = NonceTracker()
        self.mfa_manager = MFAManager()
        self.server = None  # Store server instance for cleanup
        self.health_server = None
        self.runner = None  # Add runner for proper web app handling
        
        # Initialize SSL contexts
        self.ssl_context = None
        self.client_ssl_context = None
        self.init_ssl()

    def init_ssl(self):
        """Initialize SSL contexts"""
        # Client SSL context
        self.client_ssl_context = ssl.create_default_context()
        self.client_ssl_context.check_hostname = False
        self.client_ssl_context.verify_mode = ssl.CERT_NONE

        # Server SSL context
        try:
            # Use port-specific certificate paths
            cert_path = f"certs/{self.node_id}_{self.port}.crt"
            key_path = f"certs/{self.node_id}_{self.port}.key"
            
            # Ensure certs directory exists
            os.makedirs("certs", exist_ok=True)
            
            # Debug file existence and paths
            logger.debug(f"Checking SSL for {self.node_id} on port {self.port}: "
                        f"cert_path={cert_path}, exists={os.path.exists(cert_path)}, "
                        f"key_path={key_path}, exists={os.path.exists(key_path)}")
            
            # Generate self-signed certificate if files are missing
            if not (os.path.exists(cert_path) and os.path.exists(key_path)):
                logger.info(f"SSL certificates not found for {self.node_id} on port {self.port}. Generating self-signed certificates...")
                cmd = (
                    f'openssl req -x509 -newkey rsa:2048 -keyout "{key_path}" '
                    f'-out "{cert_path}" -days 365 -nodes -subj "/CN={self.node_id}"'
                )
                result = os.system(cmd)
                if result != 0:
                    raise RuntimeError(f"Failed to generate SSL certificates for {self.node_id} on port {self.port} with OpenSSL")
                logger.info(f"Generated self-signed certificates: {cert_path}, {key_path}")
            else:
                logger.debug(f"Using existing SSL certificates for {self.node_id} on port {self.port}")
            
            # Load the certificates
            self.ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
            self.ssl_context.load_cert_chain(
                certfile=cert_path,
                keyfile=key_path
            )
            logger.info(f"HTTPS enabled with certificates for {self.node_id} on port {self.port}")
            
        except Exception as e:
            logger.error(f"Failed to initialize SSL for {self.node_id} on port {self.port}: {e}", exc_info=True)
            logger.warning(f"Running without HTTPS for {self.node_id} on port {self.port} due to SSL failure")
            self.ssl_context = None

    def setup_routes(self):
        """Setup API routes"""
        self.app.router.add_get('/health', self.health_check_endpoint)

    async def health_check_endpoint(self, request):
        """Health check endpoint"""
        return web.Response(text="OK", status=200)

    def _setup_routes(self) -> None:
        """Configure HTTP routes for the network."""
        self.app.add_routes([
            web.get("/health", self.health_handler),
            web.post('/receive_block', self.receive_block),
            web.post('/receive_transaction', self.receive_transaction),
            web.get('/get_chain', self.get_chain),
            web.post('/announce_peer', self.announce_peer),
            web.get('/get_peers', self.get_peers)
        ])

    async def health_handler(self, request: web.Request) -> web.Response:
        """Handle health check requests."""
        logger.debug(f"Health check from {request.remote}")
        return web.Response(status=200, text="OK")

    async def start(self):
        """Start the network and security monitoring"""
        # Initialize KeyRotationManager if not already done
        from utils import init_rotation_manager, rotation_manager
        if not rotation_manager:
            await init_rotation_manager(self.node_id)
            logger.info(f"Initialized KeyRotationManager for node {self.node_id} in network start")

        # Start security monitoring if available
        if self.security_monitor:
            asyncio.create_task(self.security_monitor.analyze_patterns())

        # Start the server and sync tasks
        await self.start_server()
        logger.info("Network service started")

    def run(self):
        logger.info(f"Starting network server on {self.host}:{self.port}")
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self.start_server())

    async def stop(self):
        """Stop the network and cancel background tasks."""
        logger.info("Stopping network...")
        if self.sync_task and not self.sync_task.done():
            self.sync_task.cancel()
        if self.discovery_task and not self.discovery_task.done():
            self.discovery_task.cancel()
        try:
            await asyncio.gather(self.sync_task, self.discovery_task, return_exceptions=True)
        except asyncio.CancelledError:
            pass
        if hasattr(self, 'runner'):
            await self.runner.cleanup()
        logger.info("Network stopped")

    async def send_with_retry(self, url: str, data: dict, method: str = "post", 
                            max_retries: int = CONFIG["max_retries"]) -> Tuple[bool, Optional[dict]]:
        """Send HTTP request with retry logic."""
        # Get auth secret asynchronously
        auth_secret = await PEER_AUTH_SECRET()
        headers = {"Authorization": f"Bearer {auth_secret}"}
        
        with ACTIVE_REQUESTS.labels(instance=self.node_id).track_inprogress():
            for attempt in range(max_retries):
                try:
                    async with aiohttp.ClientSession() as session:
                        if method == "post":
                            async with session.post(url, json=data, headers=headers, 
                                                  ssl=self.client_ssl_context, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                                return resp.status == 200, None
                        elif method == "get":
                            async with session.get(url, headers=headers, 
                                                 ssl=self.client_ssl_context, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                                return resp.status == 200, await resp.json() if resp.status == 200 else None
                except Exception as e:
                    logger.warning(f"Request to {url} failed (attempt {attempt + 1}/{max_retries}): {e}")
                    if attempt == max_retries - 1:
                        return False, None
                    await asyncio.sleep(0.5 * (2 ** attempt))
        return False, None

    async def broadcast_block(self, block: Block) -> None:
        """Broadcast a block to all connected peers."""
        with self.lock:
            tasks = []
            for peer_id, (host, port, _) in self.peers.items():
                tasks.append(self.send_block(peer_id, host, port, block))
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for peer_id, result in zip(self.peers.keys(), results):
                if isinstance(result, Exception):
                    logger.warning(f"Failed to broadcast block {block.index} to {peer_id}: {result}")
                    self._increment_failure(peer_id)
            BLOCKS_RECEIVED.labels(instance=self.node_id).inc()

    async def send_block(self, peer_id: str, host: str, port: int, block: Block) -> None:
        """Send a block to a specific peer."""
        url = f"https://{host}:{port}/receive_block"
        data = {"block": block.to_dict()}
        success, _ = await self.send_with_retry(url, data)
        if success:
            logger.info(f"Sent block {block.index} to {peer_id}")
        else:
            logger.warning(f"Failed to send block {block.index} to {peer_id}")
            self._increment_failure(peer_id)

    async def receive_block(self, request: web.Request) -> web.Response:
        """Handle incoming block from a peer."""
        if not validate_peer_auth(request.headers.get("Authorization", "").replace("Bearer ", "")):
            return web.Response(status=403, text="Invalid authentication")
        try:
            data = await request.json()
            block = Block.from_dict(data["block"])
            if await self.blockchain.add_block(block):
                logger.info(f"Received and added block {block.index} from {request.remote}")
                self._save_peers()
                return web.Response(status=200)
            return web.Response(status=400, text="Block validation failed")
        except Exception as e:
            logger.error(f"Error receiving block: {e}")
            return web.Response(status=400, text=str(e))

    async def broadcast_transaction(self, transaction: Transaction) -> None:
        """Thread-safe transaction broadcast"""
        with self.lock:
            tasks = []
            for peer_id, (host, port, _) in self.peers.items():
                tasks.append(self.send_transaction(peer_id, host, port, transaction))
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for peer_id, result in zip(self.peers.keys(), results):
                if isinstance(result, Exception):
                    logger.warning(f"Failed to broadcast transaction {transaction.tx_id[:8]} to {peer_id}: {result}")
                    self._increment_failure(peer_id)
            TXS_BROADCAST.labels(instance=self.node_id).inc()

    async def send_transaction(self, peer_id: str, host: str, port: int, tx: Transaction) -> None:
        """Send a transaction to a specific peer."""
        url = f"https://{host}:{port}/receive_transaction"
        data = {"transaction": tx.to_dict()}
        success, _ = await self.send_with_retry(url, data)
        if success:
            logger.info(f"Sent transaction {tx.tx_id[:8]} to {peer_id}")
        else:
            logger.warning(f"Failed to send transaction {tx.tx_id[:8]} to {peer_id}")

    async def receive_transaction(self, request: web.Request) -> web.Response:
        """Handle incoming transaction from a peer."""
        if not validate_peer_auth(request.headers.get("Authorization", "").replace("Bearer ", "")):
            return web.Response(status=403, text="Invalid authentication")
        try:
            data = await request.json()
            transaction = Transaction.from_dict(data["transaction"])
            if await self.blockchain.add_transaction_to_mempool(transaction):
                logger.info(f"Received transaction {transaction.tx_id[:8]} from {request.remote}")
                return web.Response(status=200)
            return web.Response(status=400, text="Transaction validation failed")
        except Exception as e:
            logger.error(f"Error receiving transaction: {e}")
            return web.Response(status=400, text=str(e))

    async def get_chain(self, request: web.Request) -> web.Response:
        """Return the current blockchain to a requesting peer."""
        if not validate_peer_auth(request.headers.get("Authorization", "").replace("Bearer ", "")):
            return web.Response(status=403, text="Invalid authentication")
        chain_data = [block.to_dict() for block in self.blockchain.chain]
        return web.json_response(chain_data)

    async def send_chain(self, peer_id: str, chain_data: List[dict]) -> None:
        """Send blockchain data to a specific peer.
        
        Args:
            peer_id: The ID of the peer to send the chain to
            chain_data: List of serialized blocks to send
        """
        try:
            if peer_id not in self.peers:
                logger.warning(f"Cannot send chain to unknown peer {peer_id}")
                return

            host, port, _ = self.peers[peer_id]
            url = f"https://{host}:{port}/receive_chain"
            data = {"chain": chain_data}
            
            success, _ = await self.send_with_retry(url, data)
            if success:
                logger.info(f"Successfully sent chain data to peer {peer_id}")
            else:
                logger.warning(f"Failed to send chain data to peer {peer_id}")
                self._increment_failure(peer_id)
                
        except Exception as e:
            logger.error(f"Error sending chain to peer {peer_id}: {e}")
            self._increment_failure(peer_id)

    def _load_peers(self) -> Dict[str, Tuple[str, int, str]]:
        """Load known peers from persistent storage."""
        peers = {}
        try:
            if os.path.exists("known_peers.txt"):
                with open("known_peers.txt", "r") as f:
                    for line in f:
                        if ":" in line:
                            parts = line.strip().split(":")
                            if len(parts) >= 3:
                                host, port, pubkey = parts[0], int(parts[1]), ":".join(parts[2:])
                                peer_id = f"node{port}"
                                peers[peer_id] = (host, port, pubkey)
        except Exception as e:
            logger.error(f"Failed to load peers: {e}")
        return peers

    def _save_peers(self) -> None:
        """Save current peers to persistent storage."""
        try:
            with open("known_peers.txt", "w") as f:
                for peer_id, (host, port, pubkey) in self.peers.items():
                    f.write(f"{host}:{port}:{pubkey}\n")
        except Exception as e:
            logger.error(f"Failed to save peers: {e}")

    async def announce_peer(self, request: web.Request) -> web.Response:
        """Handle peer announcement from another node."""
        if not validate_peer_auth(request.headers.get("Authorization", "").replace("Bearer ", "")):
            return web.Response(status=403, text="Invalid authentication")
        try:
            data = await request.json()
            peer_id = data.get("peer_id")
            host = data.get("host")
            port = int(data.get("port"))
            public_key = data.get("public_key")
            signature = bytes.fromhex(data.get("signature", ""))
            message = f"{peer_id}{host}{port}".encode()

            if public_key and signature:
                vk = ecdsa.VerifyingKey.from_string(bytes.fromhex(public_key), curve=ecdsa.SECP256k1)
                if not vk.verify(signature, message):
                    logger.warning(f"Peer {peer_id} failed signature verification")
                    return web.Response(status=403, text="Invalid signature")

            await self.add_peer(peer_id, host, port, public_key)
            self._save_peers()
            logger.info(f"Authenticated and added peer {peer_id} from {request.remote}")
            return web.Response(status=200)
        except Exception as e:
            logger.error(f"Error in peer announcement: {e}")
            return web.Response(status=400, text=str(e))

    async def broadcast_peer_announcement(self) -> None:
        """Announce this node to all peers."""
        sk = ecdsa.SigningKey.from_string(bytes.fromhex(self.private_key), curve=ecdsa.SECP256k1)
        message = f"{self.node_id}{self.host}{self.port}".encode()
        signature = sk.sign(message).hex()
        data = {
            "peer_id": self.node_id,
            "host": self.host,
            "port": self.port,
            "public_key": self.public_key,
            "signature": signature
        }
        tasks = []
        with self.lock:
            for peer_id, (host, port, _) in self.peers.items():
                url = f"https://{host}:{port}/announce_peer"
                tasks.append(self.send_with_retry(url, data))
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for peer_id, result in zip(self.peers.keys(), results):
                if isinstance(result[0], Exception) or not result[0]:
                    logger.warning(f"Failed to announce to {peer_id}")
                    self._increment_failure(peer_id)
                else:
                    logger.debug(f"Announced to {peer_id}")
                    self.peer_failures[peer_id] = 0

    async def get_peers(self, request: web.Request) -> web.Response:
        """Return a list of known peers to a requesting node."""
        if not validate_peer_auth(request.headers.get("Authorization", "").replace("Bearer ", "")):
            return web.Response(status=403, text="Invalid authentication")
        peer_list = [
            {"peer_id": pid, "host": host, "port": port}
            for pid, (host, port, _) in self.peers.items()
            if (host, port) != (self.host, self.port)
        ]
        random.shuffle(peer_list)
        limited_list = peer_list[:min(CONFIG["max_peers"], len(peer_list))]
        return web.json_response(limited_list)

    async def discover_peers(self) -> None:
        """Discover new peers from bootstrap nodes and existing peers."""
        with self.lock:
            # Bootstrap nodes
            for host, port in self.bootstrap_nodes:
                if (host, port) != (self.host, self.port):
                    peer_id = f"node{port}"
                    url = f"https://{host}:{port}/get_chain"
                    success, _ = await self.send_with_retry(url, {}, method="get")
                    if success:
                        await self.add_peer(peer_id, host, port, PEER_AUTH_SECRET())
                    else:
                        logger.debug(f"Skipping unresponsive bootstrap node {peer_id}")

            # Discover from existing peers
            if not self.peers:
                logger.info("No peers available for discovery")
                return
            peer_id, (host, port, _) = random.choice(list(self.peers.items()))
            url = f"https://{host}:{port}/get_peers"
            success, peers_data = await self.send_with_retry(url, {}, method="get")
            if success and peers_data:
                for peer in peers_data:
                    if (peer["host"], peer["port"]) != (self.host, self.port):
                        await self.add_peer(peer["peer_id"], peer["host"], peer["port"], PEER_AUTH_SECRET())
            else:
                logger.warning(f"Peer discovery failed with {peer_id}")
                self._increment_failure(peer_id)
            logger.info("Peer discovery cycle completed")

    async def add_peer(self, peer_id: str, host: str, port: int, public_key: str) -> bool:
        """Add a peer to the network."""
        with self.lock:
            if len(self.peers) >= CONFIG["max_peers"] and peer_id not in self.peers:
                return False
            peer_key = (host, port)
            if peer_id not in self.peers or self.peers[peer_id][:2] != peer_key:
                self.peers[peer_id] = (host, port, public_key)
                logger.info(f"Added peer {peer_id}: {host}:{port}")
                if time.time() - self.last_announcement > 10:
                    await self.broadcast_peer_announcement()
                    self.last_announcement = time.time()
                return True
            return False

    def _increment_failure(self, peer_id: str) -> None:
        """Track peer failures and remove unresponsive peers."""
        self.peer_failures[peer_id] += 1
        PEER_FAILURES.labels(instance=self.node_id).inc()
        if self.peer_failures[peer_id] > 3:
            if peer_id in self.peers:
                del self.peers[peer_id]
                logger.info(f"Removed unresponsive peer {peer_id} after {self.peer_failures[peer_id]} failures")
                self._save_peers()
            del self.peer_failures[peer_id]

    async def request_chain(self) -> None:
        """Request and potentially update the blockchain from peers."""
        best_chain = self.blockchain.chain
        best_difficulty = self.blockchain.get_total_difficulty()
        with self.lock:
            tasks = []
            for peer_id, (host, port, _) in self.peers.items():
                url = f"https://{host}:{port}/get_chain"
                tasks.append(self.send_with_retry(url, {}, method="get"))
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for peer_id, (success, chain_data) in zip(self.peers.keys(), results):
                if not success or isinstance(chain_data, Exception):
                    logger.warning(f"Failed to get chain from {peer_id}")
                    self._increment_failure(peer_id)
                    continue
                try:
                    new_chain = [Block.from_dict(block) for block in chain_data]
                    new_difficulty = sum(block.header.difficulty for block in new_chain)
                    if new_difficulty > best_difficulty and await self.blockchain.is_valid_chain(new_chain):
                        best_chain = new_chain
                        best_difficulty = new_difficulty
                        logger.info(f"Found better chain from {peer_id} with difficulty {new_difficulty}")
                        await self.blockchain.replace_chain(best_chain)
                    self.peer_failures[peer_id] = 0
                except Exception as e:
                    logger.warning(f"Invalid chain data from {peer_id}: {e}")
                    self._increment_failure(peer_id)

    async def sync_and_discover(self) -> None:
        """Perform a full sync and discovery cycle."""
        try:
            logger.debug("Starting sync and discovery cycle")
            self.blockchain.difficulty = self.blockchain.adjust_difficulty()
            await self.discover_peers()
            await self.request_chain()
            await self.broadcast_peer_announcement()
            logger.debug("Sync and discovery cycle completed")
        except Exception as e:
            logger.error(f"Sync and discovery error: {e}")

    async def periodic_discovery(self) -> None:
        """Run peer discovery periodically."""
        while True:
            try:
                await self.discover_peers()
            except Exception as e:
                logger.error(f"Periodic discovery error: {e}")
            await asyncio.sleep(CONFIG["peer_discovery_interval"])

    async def start_periodic_sync(self) -> asyncio.Task:
        """Start periodic sync and discovery tasks."""
        async def sync_loop():
            try:
                logger.info("Starting periodic sync loop")
                await self.discover_peers()
                if not self.discovery_task or self.discovery_task.done():
                    self.discovery_task = self.loop.create_task(self.periodic_discovery())
                    self.discovery_task.add_done_callback(self._handle_task_result)
                while True:
                    await self.sync_and_discover()
                    await asyncio.sleep(CONFIG["sync_interval"])
            except asyncio.CancelledError:
                logger.info("Sync loop cancelled")
                if self.discovery_task and not self.discovery_task.done():
                    self.discovery_task.cancel()
            except Exception as e:
                logger.critical(f"Sync loop failed: {e}")
                raise

        if self.sync_task and not self.sync_task.done():
            logger.warning("Sync task already running")
            return self.sync_task
        self.sync_task = self.loop.create_task(sync_loop())
        self.sync_task.add_done_callback(self._handle_task_result)
        return self.sync_task

    def _handle_task_result(self, task: asyncio.Task) -> None:
        """Handle task completion and log exceptions."""
        try:
            task.result()
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Task failed: {e}")

    async def handle_transaction(self, transaction, peer_id):
        # Check rate limit
        if not await self.rate_limiter.check_rate_limit(peer_id, 'transaction'):
            self.peer_reputation.update_reputation(peer_id, 'rate_limit_exceeded')
            return False
            
        # Check peer reputation
        if not self.peer_reputation.is_peer_trusted(peer_id):
            logger.warning(f"Rejecting transaction from untrusted peer {peer_id}")
            return False
            
        # Check for replay attacks
        if await self.nonce_tracker.is_nonce_used(
            SecurityUtils.public_key_to_address(transaction.inputs[0].public_key),
            transaction.nonce
        ):
            self.peer_reputation.update_reputation(peer_id, 'invalid_transaction')
            return False
            
        # ... rest of transaction handling ...

    async def handle_connection(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        """Thread-safe connection handling"""
        with self.lock:
            peer_address = writer.get_extra_info('peername')
            client_ip = peer_address[0] if peer_address else 'unknown'
            
            try:
                # Check security monitor if available
                if self.security_monitor and not await self.security_monitor.monitor_connection(client_ip):
                    logger.warning(f"Connection rejected from {client_ip} by security monitor")
                    writer.close()
                    await writer.wait_closed()
                    return

                logger.info(f"New connection from {client_ip}")
                
                while not self.shutdown_flag:
                    try:
                        # Read message length first (4 bytes)
                        length_data = await reader.read(4)
                        if not length_data:
                            break
                        
                        message_length = int.from_bytes(length_data, 'big')
                        if message_length > 1048576:  # 1MB limit
                            logger.warning(f"Message too large from {client_ip}")
                            break
                        
                        # Read the actual message
                        data = await reader.read(message_length)
                        if not data:
                            break
                        
                        try:
                            # Try to decode as JSON
                            message = json.loads(data.decode('utf-8'))
                            await self.handle_message(message, client_ip)
                        except json.JSONDecodeError:
                            # If not JSON, try to handle as binary data
                            await self.handle_binary_message(data, client_ip)
                        
                        # Send acknowledgment
                        ack = "ACK".encode('utf-8')
                        writer.write(len(ack).to_bytes(4, 'big') + ack)
                        await writer.drain()
                        
                    except Exception as e:
                        logger.error(f"Error handling connection from {client_ip}: {e}")
                        if self.security_monitor:
                            await self.security_monitor.record_failed_attempt(client_ip, 'connection_error')
                        break
                        
            except Exception as e:
                logger.error(f"Connection error from {client_ip}: {e}")
                if self.security_monitor:
                    await self.security_monitor.record_failed_attempt(client_ip, 'connection_error')
                
            finally:
                try:
                    writer.close()
                    await writer.wait_closed()
                    logger.info(f"Connection closed from {client_ip}")
                except Exception as e:
                    logger.error(f"Error closing connection from {client_ip}: {e}")

    async def start_health_server(self):
        """Start the health check HTTP server"""
        try:
            self.runner = web.AppRunner(self.app)
            await self.runner.setup()
            site = web.TCPSite(self.runner, self.host, self.port + 1)
            await site.start()
            logger.info(f"Starting health check server on {self.host}:{self.port + 1}")
        except Exception as e:
            logger.error(f"Failed to start health check server: {e}")
            raise

    async def start_server(self):
        """Start both P2P and HTTP servers"""
        try:
            # Start P2P server
            self.server = await asyncio.start_server(
                self.handle_connection,
                self.host,
                self.port,
                ssl=self.ssl_context
            )
            logger.info(f"Starting P2P server on {self.host}:{self.port}")

            # Start health check server
            await self.start_health_server()
            
            # Start serving in the background without awaiting
            asyncio.create_task(self.server_task())
            
            # Start periodic sync task
            await self.start_periodic_sync()
            
            # Return without blocking
            return self.server

        except Exception as e:
            logger.error(f"Failed to start servers: {e}")
            raise

    async def server_task(self):
        """Run the server in a background task"""
        try:
            async with self.server:
                await self.server.serve_forever()
        except asyncio.CancelledError:
            logger.info("Server task cancelled")
        except Exception as e:
            logger.error(f"Server error: {e}")

    async def handle_message(self, message: str, client_ip: str):
        """Handle incoming messages"""
        try:
            # Add message handling logic here
            # For example:
            message_data = json.loads(message)
            message_type = message_data.get('type')
            
            if message_type == 'transaction':
                await self.handle_transaction(message_data['data'], client_ip)
            elif message_type == 'block':
                await self.handle_block(message_data['data'], client_ip)
            else:
                logger.warning(f"Unknown message type from {client_ip}: {message_type}")
                
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON message from {client_ip}")
            if self.security_monitor:
                await self.security_monitor.record_failed_attempt(client_ip, 'invalid_message')
        except Exception as e:
            logger.error(f"Error processing message from {client_ip}: {e}")
            if self.security_monitor:
                await self.security_monitor.record_failed_attempt(client_ip, 'message_error')

    async def cleanup(self):
        """Cleanup network resources"""
        logger.info("Cleaning up network resources...")
        try:
            if self.server:
                self.server.close()
                await self.server.wait_closed()
            
            if self.runner:
                await self.runner.cleanup()
            
            # Close all peer connections - make sure peer.close() is async
            for peer in list(self.peers.values()):
                try:
                    if hasattr(peer, 'close') and callable(peer.close):
                        if asyncio.iscoroutinefunction(peer.close):
                            await peer.close()
                        else:
                            peer.close()
                except Exception as e:
                    logger.warning(f"Error closing peer connection: {e}")
            
            # Stop security monitor if active
            if self.security_monitor:
                await self.security_monitor.stop()
            
            logger.info("Network cleanup completed")
        except Exception as e:
            logger.error(f"Error during network cleanup: {e}")

    def run(self):
        try:
            self.loop.run_until_complete(self.start_server())
        except Exception as e:
            logger.error(f"Network error: {e}")
        finally:
            self.loop.run_until_complete(self.cleanup())
            self.loop.close()