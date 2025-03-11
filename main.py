import asyncio
import ssl
import argparse
import logging
import signal
import sys
import os
from blockchain import Blockchain, Transaction
from network import BlockchainNetwork
from utils import find_available_port, init_rotation_manager
from gui import BlockchainGUI
import threading
import aiohttp
from threading import Lock
from security import SecurityMonitor, MFAManager, KeyBackupManager
from key_rotation.core import KeyRotationManager

# Debug prints
print("Main.py sys.path:", sys.path)
print("Main.py current directory:", os.getcwd())

# Add the project root directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_ssl_context():
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = True
    ssl_context.minimum_version = ssl.TLSVersion.TLS1_2
    ssl_context.verify_mode = ssl.CERT_REQUIRED
    return ssl_context

async def health_check(host: str, port: int, client_ssl_context, retries: int = 5, delay: float = 1.0) -> bool:
    """Check if the node is healthy and responding"""
    health_port = port + 1  # Health check runs on main port + 1
    
    try:
        timeout = aiohttp.ClientTimeout(total=5)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            for attempt in range(retries):
                try:
                    # Try without SSL first
                    async with session.get(f"http://{host}:{health_port}/health") as resp:
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

def shutdown(gui: BlockchainGUI, network: BlockchainNetwork, rotation_manager, loop: asyncio.AbstractEventLoop):
    logger.info("Initiating graceful shutdown...")
    try:
        # Set shutdown flag
        network.shutdown_flag = True
        
        # Stop GUI
        gui.exit()
        
        # Stop network operations
        shutdown_task = asyncio.run_coroutine_threadsafe(
            network.stop(), 
            network.loop
        )
        shutdown_task.result(timeout=5)  # Wait up to 5 seconds
        
        # Stop rotation manager
        rotation_task = asyncio.run_coroutine_threadsafe(
            rotation_manager.stop(), 
            loop
        )
        rotation_task.result(timeout=5)
        
        # Clean up security monitoring
        if network.security_monitor:
            asyncio.run_coroutine_threadsafe(
                network.security_monitor.cleanup(),
                loop
            ).result(timeout=5)
        
        # Cancel all pending tasks
        for task in asyncio.all_tasks(loop):
            task.cancel()
        
        loop.stop()
        logger.info("Shutdown completed successfully")
    except Exception as e:
        logger.error(f"Error during shutdown: {e}", exc_info=True)
        sys.exit(1)

async def run_async_tasks(gui: BlockchainGUI, network: BlockchainNetwork, rotation_manager, loop: asyncio.AbstractEventLoop):
    """Run initial async tasks."""
    try:
        # Give servers time to start
        await asyncio.sleep(3)  # Increased from 2 to 3 seconds
        
        if not await health_check(network.host, network.port, network.client_ssl_context):
            logger.error("Health check failed after retries, exiting...")
            try:
                await network.cleanup()
            except Exception as cleanup_error:
                logger.error(f"Error during cleanup: {cleanup_error}")
            gui.root.quit()
            sys.exit(1)
    except Exception as e:
        logger.error(f"Critical error in async tasks: {e}")
        try:
            await network.cleanup()
        except Exception as cleanup_error:
            logger.error(f"Error during cleanup: {cleanup_error}")
        gui.root.quit()
        sys.exit(1)

class ThreadSafeBlockchain:
    def __init__(self):
        self._lock = Lock()
        self._blockchain = Blockchain()
    
    async def safe_operation(self, operation):
        with self._lock:
            return await operation(self._blockchain)

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
            # Windows-specific resource management
            import psutil
            process = psutil.Process()
            process.nice(psutil.NORMAL_PRIORITY_CLASS)
        except Exception as e:
            logger.warning(f"Could not set Windows process priority: {e}")
    else:
        # Unix-specific resource management
        try:
            import resource
            # Set maximum memory usage (e.g., 2GB)
            soft, hard = resource.getrlimit(resource.RLIMIT_AS)
            resource.setrlimit(resource.RLIMIT_AS, (2 * 1024 * 1024 * 1024, hard))
            
            # Set maximum number of open files
            soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
            resource.setrlimit(resource.RLIMIT_NOFILE, (1024, hard))
        except Exception as e:
            logger.warning(f"Could not set resource limits: {e}")

async def initialize_security(node_id: str) -> tuple:
    """Initialize security components"""
    security_monitor = SecurityMonitor()
    mfa_manager = MFAManager()
    backup_manager = KeyBackupManager(
        backup_dir=os.path.join('data', 'key_backups')
    )
    
    # Start security monitoring
    await security_monitor.start()
    
    return security_monitor, mfa_manager, backup_manager

def run_async_loop(loop):
    """Run the async event loop in a separate thread"""
    asyncio.set_event_loop(loop)
    loop.run_forever()

def main():
    # Add resource limits before any other initialization
    set_resource_limits()
    
    parser = argparse.ArgumentParser(description="Run a blockchain node.")
    parser.add_argument("--port", type=int, default=None, help="Port to run the node on (1024-65535)")
    parser.add_argument("--bootstrap", type=str, default=None, 
                       help="Comma-separated list of bootstrap nodes (host:port)")
    parser.add_argument("--validator", action="store_true", 
                       help="Run as validator node for key rotation")
    args = parser.parse_args()

    if args.port and not validate_port(args.port):
        logger.error("Invalid port number")
        sys.exit(1)

    if args.bootstrap and not validate_bootstrap_nodes(args.bootstrap):
        logger.error("Invalid bootstrap nodes format")
        sys.exit(1)

    port = args.port if args.port else find_available_port()
    api_port = port + 1000
    node_id = f"node{port}"
    logger.info(f"Initializing blockchain on {port} and key rotation API on {api_port}")

    # Initialize security components FIRST
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    security_monitor, mfa_manager, backup_manager = loop.run_until_complete(
        initialize_security(node_id)
    )

    # Initialize key rotation (synchronous part)
    init_rotation_manager(node_id)
    try:
        rotation_manager = KeyRotationManager(
            node_id=node_id, 
            is_validator=args.validator,
            backup_manager=backup_manager if 'backup_manager' in locals() else None
        )
    except Exception as e:
        logger.error(f"Failed to initialize KeyRotationManager: {e}")
        sys.exit(1)

    # Start key rotation API in a separate thread
    from key_rotation.main import main as rotation_main
    rotation_thread = threading.Thread(
        target=lambda: asyncio.run(rotation_main(node_id, args.validator, api_port, "127.0.0.1")),
        daemon=True
    )
    rotation_thread.start()

    # Initialize blockchain with security components
    blockchain = Blockchain(f"node{args.port}")
    
    # Create default wallet if none exists
    if not blockchain.get_all_addresses():
        default_address = blockchain.create_wallet()
        logger.info(f"Created default wallet with address: {default_address}")
        
        # Add some initial coins to the default wallet (for testing)
        genesis_tx = Transaction(None, default_address, 100.0)
        blockchain.pending_transactions.append(genesis_tx)
        
        # Create a new block with the initial transaction
        try:
            new_block = blockchain.create_block()
            logger.info(f"Created genesis block with hash: {new_block.hash}")
        except Exception as e:
            logger.error(f"Failed to create genesis block: {e}")
            raise
    
    # Parse bootstrap nodes
    bootstrap_nodes = []
    if args.bootstrap:
        bootstrap_nodes = [(node.split(":")[0], int(node.split(":")[1])) for node in args.bootstrap.split(",")]
    elif port != 5000:
        bootstrap_nodes = [("127.0.0.1", 5000)]

    # Update network initialization with security monitor
    network = BlockchainNetwork(
        blockchain,
        node_id,
        "127.0.0.1",
        port,
        bootstrap_nodes,
        security_monitor=security_monitor
    )
    logger.info(f"Node {node_id} public key: {network.public_key}")

    network_thread = threading.Thread(target=network.run, daemon=True)
    network_thread.start()

    # Create GUI with security components
    gui = BlockchainGUI(
        blockchain,
        network,
        mfa_manager=mfa_manager,
        backup_manager=backup_manager
    )
    
    # Create new event loop for async operations
    loop = asyncio.new_event_loop()
    
    # Initialize blockchain in the async loop
    loop.run_until_complete(blockchain.initialize())
    
    # Create and start async thread
    async_thread = threading.Thread(
        target=run_async_loop,
        args=(loop,),
        daemon=True
    )
    async_thread.start()

    try:
        # Schedule network and key manager startup in the async loop
        asyncio.run_coroutine_threadsafe(
            network.start_server(),
            loop
        )
        asyncio.run_coroutine_threadsafe(
            rotation_manager.start(),
            loop
        )

        # Create and run GUI in main thread
        gui.run()  # This blocks until GUI is closed

    except Exception as e:
        logger.error(f"Error in main: {e}")
    finally:
        # Cleanup
        loop.call_soon_threadsafe(loop.stop)
        async_thread.join(timeout=5)
        
        # Force cleanup if thread doesn't exit
        if async_thread.is_alive():
            logger.warning("Async thread didn't exit cleanly")

    # Signal handlers
    signal.signal(signal.SIGINT, lambda s, f: shutdown(gui, network, rotation_manager, loop))
    signal.signal(signal.SIGTERM, lambda s, f: shutdown(gui, network, rotation_manager, loop))

if __name__ == "__main__":
    main()