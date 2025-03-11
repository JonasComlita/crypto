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
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from blockchain import Blockchain, Block, Transaction
from utils import PEER_AUTH_SECRET, SSL_CERT_PATH, SSL_KEY_PATH, generate_node_keypair, validate_peer_auth
from prometheus_client import Counter, Gauge

# Configure logging
logger = logging.getLogger("BlockchainNetwork")

# Metrics
BLOCKS_RECEIVED = Counter('blocks_received_total', 'Total number of blocks received from peers')
TXS_BROADCAST = Counter('transactions_broadcast_total', 'Total number of transactions broadcast')
PEER_FAILURES = Counter('peer_failures_total', 'Total number of peer connection failures')

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

class BlockchainNetwork:
    """Manages peer-to-peer networking for the blockchain with enhanced security and reliability."""
    def __init__(self, blockchain: 'Blockchain', node_id: str, host: str, port: int, 
                 bootstrap_nodes: Optional[List[Tuple[str, int]]] = None):
        self.blockchain = blockchain
        self.node_id = node_id
        self.host = host
        self.port = port
        self.bootstrap_nodes = bootstrap_nodes or []
        self.loop = None  # Will be set in run()
        self.private_key, self.public_key = generate_node_keypair()
        self.peers: Dict[str, Tuple[str, int, str]] = self._load_peers()
        self.app = aiohttp.web.Application(middlewares=[rate_limit_middleware])
        self._setup_routes()
        self._setup_ssl_contexts()
        self.blockchain.network = self
        self.sync_task: Optional[asyncio.Task] = None
        self.discovery_task: Optional[asyncio.Task] = None
        self.last_announcement = 0
        self.peer_failures: Dict[str, int] = defaultdict(int)
        self.start_time = time.time()
        self._lock = asyncio.Lock()
        self.active_requests = Gauge('active_peer_requests', 'Number of active requests to peers')

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

    def _setup_ssl_contexts(self) -> None:
        """Initialize SSL contexts for server and client connections."""
        try:
            self.server_ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            self.server_ssl_context.load_cert_chain(certfile=CONFIG["tls_cert_file"], keyfile=CONFIG["tls_key_file"])
            
            self.client_ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
            self.client_ssl_context.load_verify_locations(cafile=CONFIG["tls_cert_file"])
            self.client_ssl_context.check_hostname = False  # For self-signed certs in testing
            self.client_ssl_context.verify_mode = ssl.CERT_REQUIRED
        except Exception as e:
            logger.error(f"Failed to setup SSL contexts: {e}")
            raise

    async def health_handler(self, request: web.Request) -> web.Response:
        """Handle health check requests."""
        logger.debug(f"Health check from {request.remote}")
        return web.Response(status=200, text="OK")

    def run(self) -> None:
        """Start the network server and background tasks in its own event loop."""
        logger.info(f"Starting network server on {self.host}:{self.port}")
        self.loop = asyncio.new_event_loop()  # Create a new loop for this thread
        asyncio.set_event_loop(self.loop)
        
        runner = aiohttp.web.AppRunner(self.app)
        self.loop.run_until_complete(runner.setup())
        site = aiohttp.web.TCPSite(runner, self.host, self.port, ssl_context=self.server_ssl_context)
        self.loop.run_until_complete(site.start())
        logger.info(f"Network server running on {self.host}:{self.port}")
        
        self.loop.run_until_complete(self.start_periodic_sync())
        try:
            self.loop.run_forever()
        except KeyboardInterrupt:
            logger.info("Network loop interrupted")
        finally:
            self.loop.run_until_complete(self.stop())
            self.loop.close()

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
        headers = {"Authorization": f"Bearer {PEER_AUTH_SECRET()}"}
        async with self.active_requests.track_inprogress():
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
        async with self._lock:
            tasks = []
            for peer_id, (host, port, _) in self.peers.items():
                tasks.append(self.send_block(peer_id, host, port, block))
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for peer_id, result in zip(self.peers.keys(), results):
                if isinstance(result, Exception):
                    logger.warning(f"Failed to broadcast block {block.index} to {peer_id}: {result}")
                    self._increment_failure(peer_id)
            BLOCKS_RECEIVED.inc()

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
        """Broadcast a transaction to all connected peers."""
        async with self._lock:
            tasks = []
            for peer_id, (host, port, _) in self.peers.items():
                tasks.append(self.send_transaction(peer_id, host, port, transaction))
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for peer_id, result in zip(self.peers.keys(), results):
                if isinstance(result, Exception):
                    logger.warning(f"Failed to broadcast transaction {transaction.tx_id[:8]} to {peer_id}: {result}")
                    self._increment_failure(peer_id)
            TXS_BROADCAST.inc()

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
        async with self._lock:
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
        async with self._lock:
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
        async with self._lock:
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
        PEER_FAILURES.inc()
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
        async with self._lock:
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
            self.blockchain.adjust_difficulty()
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