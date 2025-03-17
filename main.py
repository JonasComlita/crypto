import asyncio
import ssl
import argparse
import logging
import signal
import sys
import os
import time
import concurrent.futures
from blockchain import Blockchain, Transaction, TransactionType, Block
from network import BlockchainNetwork, load_config, save_config, NodeIdentity
import getpass
from utils import init_rotation_manager, find_available_port_async, is_port_available
from gui import BlockchainGUI
import threading
from threading import Lock, Event
from security import SecurityMonitor, MFAManager, KeyBackupManager
from key_rotation.core import KeyRotationManager

# Add project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def create_ssl_context():
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = True
    ssl_context.minimum_version = ssl.TLSVersion.TLS1_2
    ssl_context.verify_mode = ssl.CERT_REQUIRED
    return ssl_context

async def health_check(host: str, port: int, client_ssl_context, retries: int = 5, delay: float = 1.0) -> bool:
    """Check if the node is healthy and responding"""
    health_port = port  # Adjusted to use network's HTTP port directly
    try:
        timeout = aiohttp.ClientTimeout(total=5)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            for attempt in range(retries):
                try:
                    async with session.get(f"https://{host}:{health_port}/health", ssl=client_ssl_context) as resp:
                        if resp.status == 200:
                            return True
                except Exception as e:
                    logger.warning(f"Health check attempt {attempt + 1}/{retries} failed: {e}")
                if attempt < retries - 1:
                    await asyncio.sleep(delay)
            return False
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return False

async def run_async_tasks(blockchain: Blockchain, network: BlockchainNetwork, rotation_manager, loop: asyncio.AbstractEventLoop):
    """Run initial async tasks with health check"""
    await asyncio.sleep(3)  # Wait for servers to start
    if not await health_check(network.host, network.port, network.client_ssl_context):
        logger.error("Health check failed after retries")
        return False
    return True

def validate_port(port: int) -> bool:
    return isinstance(port, int) and 1024 <= port <= 65535

def validate_bootstrap_nodes(nodes_str: str) -> bool:
    try:
        nodes = nodes_str.split(",")
        for node in nodes:
            host, port = node.split(":")
            if not (validate_port(int(port)) and len(host) > 0):
                return False
        return True
    except:
        return False

def set_resource_limits():
    """Set resource limits based on platform"""
    import platform
    if platform.system() == 'Windows':
        try:
            import psutil
            process = psutil.Process()
            process.nice(psutil.NORMAL_PRIORITY_CLASS)
        except Exception as e:
            logger.warning(f"Could not set Windows process priority: {e}")
    else:
        try:
            import resource
            soft, hard = resource.getrlimit(resource.RLIMIT_AS)
            resource.setrlimit(resource.RLIMIT_AS, (2 * 1024 * 1024 * 1024, hard))
            soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
            resource.setrlimit(resource.RLIMIT_NOFILE, (1024, hard))
        except Exception as e:
            logger.warning(f"Could not set resource limits: {e}")

async def initialize_security(node_id: str) -> tuple:
    """Initialize security components"""
    security_monitor = SecurityMonitor()
    mfa_manager = MFAManager()
    backup_manager = KeyBackupManager(backup_dir=os.path.join('data', 'key_backups'))
    await security_monitor.start()
    return security_monitor, mfa_manager, backup_manager

async def get_wallet_password(blockchain: Blockchain) -> str:
    """Asynchronously get or set wallet password"""
    print("\n=== Wallet Connection ===")
    choice = input("Do you want to connect to an existing wallet (1) or create a new one (2)? [1/2]: ").strip()
    
    if choice == "2":
        while True:
            password = getpass.getpass("Set a wallet encryption password: ").strip()
            if not password:
                print("Password cannot be empty. Please try again.")
                continue
            confirm = getpass.getpass("Confirm password: ").strip()
            if password != confirm:
                print("Passwords do not match. Please try again.")
                continue
            address = await blockchain.create_wallet()
            blockchain.key_manager.password = password
            await blockchain.save_wallets()
            print(f"New wallet created with address: {address}")
            return password
    
    elif choice == "1":
        while True:
            password = getpass.getpass("Enter wallet encryption password: ").strip()
            if not password:
                print("Password cannot be empty. Please try again.")
                continue
            original_password = blockchain.key_manager.password
            blockchain.key_manager.password = password
            try:
                wallets = await blockchain.key_manager.load_keys()
                if wallets:
                    print(f"Successfully connected to wallet(s). Found {len(wallets)} address(es).")
                    blockchain.wallets = wallets
                    return password
                else:
                    print("No wallets found with this password. Please try again or create a new wallet.")
                    blockchain.key_manager.password = original_password
            except ValueError as e:
                if "Incorrect password" in str(e):
                    print("Incorrect password. Please try again.")
                else:
                    raise
    else:
        print("Invalid choice. Please enter 1 or 2.")
        return await get_wallet_password(blockchain)

async def async_main(args, loop):
    """Async initialization with retry logic"""
    config = load_config(args.config)
    if args.p2p_port:
        config["p2p_port"] = args.p2p_port
    if args.api_port:
        config["api_port"] = args.api_port
    if args.data_dir:
        config["data_dir"] = args.data_dir

    # Parse bootstrap nodes
    bootstrap_nodes = []
    if args.bootstrap:
        try:
            for node in args.bootstrap.split(","):
                host, port = node.strip().split(":")
                port_num = int(port)
                if not validate_port(port_num) or not host:
                    raise ValueError(f"Invalid bootstrap node: {node}")
                bootstrap_nodes.append((host, port_num))
            logger.info(f"Using bootstrap nodes: {bootstrap_nodes}")
        except Exception as e:
            logger.error(f"Failed to parse bootstrap nodes '{args.bootstrap}': {e}")

    # Initialize node identity and blockchain
    node_identity = NodeIdentity(config["data_dir"])
    node_id, private_key, public_key = await node_identity.initialize()
    blockchain = Blockchain(node_id=node_id, wallet_password=None, port=config["p2p_port"])
    wallet_password = await get_wallet_password(blockchain)
    blockchain.key_manager.password = wallet_password
    await blockchain.initialize()

    # Initialize security components
    security_monitor, mfa_manager, backup_manager = await initialize_security(node_id)
    from utils import rotation_manager
    if not rotation_manager:
        await init_rotation_manager(node_id)

    # Start network with retry
    for attempt in range(3):
        try:
            network = BlockchainNetwork(
                blockchain, node_id, "127.0.0.1", config["p2p_port"], bootstrap_nodes,
                security_monitor=security_monitor, config_path=args.config
            )
            network.private_key = private_key
            network.public_key = public_key
            network.loop = loop
            await network.start()
            break
        except Exception as e:
            logger.warning(f"Network start attempt {attempt + 1}/3 failed: {e}")
            if attempt == 2:
                raise
            await asyncio.sleep(2)

    # Adjust key rotation port if needed
    key_rotation_port = config["key_rotation_port"]
    if not is_port_available(key_rotation_port, host="127.0.0.1"):
        new_port = await find_available_port_async(start_port=key_rotation_port + 1, end_port=key_rotation_port + 100, host="127.0.0.1")
        if new_port:
            logger.info(f"Using alternative port {new_port} for key rotation")
            key_rotation_port = new_port
            config["key_rotation_port"] = new_port
            save_config(config, args.config)

    # Start key rotation
    from key_rotation.main import main as rotation_main
    shutdown_event = asyncio.Event()
    rotation_task = asyncio.create_task(rotation_main(node_id, args.validator, key_rotation_port, "127.0.0.1", loop, shutdown_event))
    network.background_tasks.append(rotation_task)

    # Verify health
    if not await run_async_tasks(blockchain, network, rotation_manager, loop):
        raise RuntimeError("Health check failed")

    return blockchain, network, security_monitor, mfa_manager, backup_manager, rotation_manager, shutdown_event

def run_async_loop(loop):
    """Run the asyncio event loop in a separate thread"""
    asyncio.set_event_loop(loop)
    try:
        loop.run_forever()
    except Exception as e:
        logger.error(f"Async loop crashed: {e}")
    finally:
        logger.info("Async loop stopped")

async def async_shutdown(network: BlockchainNetwork, blockchain: Blockchain, shutdown_event: asyncio.Event, loop: asyncio.AbstractEventLoop):
    """Cleanly shut down all components"""
    logger.info("Initiating shutdown...")
    shutdown_event.set()
    await network.stop()
    await blockchain.shutdown()
    tasks = [t for t in asyncio.all_tasks(loop) if not t.done()]
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    loop.stop()

def main():
    set_resource_limits()
    parser = argparse.ArgumentParser(description="Run a blockchain node.")
    parser.add_argument("--config", type=str, default="network_config.json", help="Path to configuration file")
    parser.add_argument("--p2p-port", type=int, default=None, help="P2P communication port")
    parser.add_argument("--api-port", type=int, default=None, help="HTTP API port")
    parser.add_argument("--data-dir", type=str, default=None, help="Data directory")
    parser.add_argument("--bootstrap", type=str, default=None, help="Comma-separated list of bootstrap nodes (host:port)")
    parser.add_argument("--validator", action="store_true", help="Run as validator node")

    args = parser.parse_args()

    if args.p2p_port and not validate_port(args.p2p_port):
        logger.error("Invalid P2P port number")
        sys.exit(1)
    if args.api_port and not validate_port(args.api_port):
        logger.error("Invalid API port number")
        sys.exit(1)
    if args.bootstrap and not validate_bootstrap_nodes(args.bootstrap):
        logger.error("Invalid bootstrap nodes format")
        sys.exit(1)
    if args.data_dir and not os.path.isdir(args.data_dir):
        logger.error(f"Data directory {args.data_dir} is not accessible")
        sys.exit(1)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    async_thread = threading.Thread(target=run_async_loop, args=(loop,), daemon=True)
    async_thread.start()

    # Async initialization
    try:
        blockchain, network, security_monitor, mfa_manager, backup_manager, rotation_manager, shutdown_event = loop.run_until_complete(
            async_main(args, loop)
        )
    except Exception as e:
        logger.error(f"Initialization failed: {e}", exc_info=True)
        loop.run_until_complete(async_shutdown(network, blockchain, shutdown_event, loop))
        async_thread.join(timeout=5)
        sys.exit(1)

    # Setup GUI with event subscription
    gui = BlockchainGUI(blockchain, network, mfa_manager=mfa_manager, backup_manager=backup_manager)
    gui.loop = loop
    gui.loop_thread = async_thread

    def on_new_block(block):
        gui.update_chain_display()
    def on_error(error):
        gui.show_error(str(error))
    blockchain.subscribe("new_block", on_new_block)
    blockchain.subscribe("error", on_error)

    # Signal handlers
    def signal_handler(sig, frame):
        logger.info(f"Received signal {sig}, shutting down...")
        loop.call_soon_threadsafe(lambda: asyncio.create_task(async_shutdown(network, blockchain, shutdown_event, loop)))
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Run GUI in a separate thread
    gui_thread = threading.Thread(target=gui.run, daemon=True)
    gui_thread.start()

    try:
        while gui_thread.is_alive() and async_thread.is_alive():
            time.sleep(1)
            if not async_thread.is_alive():
                logger.error("Async thread crashed unexpectedly")
                raise RuntimeError("Async loop terminated")
    except Exception as e:
        logger.error(f"Application failed: {e}", exc_info=True)
    finally:
        loop.run_until_complete(async_shutdown(network, blockchain, shutdown_event, loop))
        gui_thread.join(timeout=5)
        async_thread.join(timeout=5)
        if gui_thread.is_alive() or async_thread.is_alive():
            logger.warning("Threads did not terminate cleanly")
        logger.info("Application fully shut down")
        sys.exit(0)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.critical(f"Critical error in main program: {e}", exc_info=True)
        sys.exit(1)