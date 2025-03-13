import asyncio
import ssl
import argparse
import logging
import signal
import sys
import os
from blockchain import Blockchain, Transaction, TransactionType, Block
from network import BlockchainNetwork
from utils import find_available_port, init_rotation_manager, find_available_port_async
from gui import BlockchainGUI
import threading
import aiohttp
from threading import Lock
from security import SecurityMonitor, MFAManager, KeyBackupManager
from key_rotation.core import KeyRotationManager

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

async def shutdown_async(gui: BlockchainGUI, blockchain: Blockchain, network: BlockchainNetwork, rotation_manager):
    logger.info("Initiating graceful async shutdown...")
    try:
        # Stop mining if active
        if hasattr(blockchain, 'miner') and blockchain.miner.mining:
            await blockchain.stop_mining()
            logger.info("Mining stopped during shutdown")
        
        # Stop network operations
        await network.stop()
        logger.info("Network stopped")
        
        # Stop blockchain
        await blockchain.shutdown()
        logger.info("Blockchain shutdown completed")
        
        # Stop rotation manager
        await rotation_manager.stop()
        logger.info("Key rotation manager stopped")
        
        # Clean up security monitoring
        if network.security_monitor:
            await network.security_monitor.cleanup()
            logger.info("Security monitor cleaned up")
            
        logger.info("Async shutdown completed successfully")
    except Exception as e:
        logger.error(f"Error during async shutdown: {e}", exc_info=True)

def shutdown(gui: BlockchainGUI, blockchain: Blockchain, network: BlockchainNetwork, rotation_manager, loop: asyncio.AbstractEventLoop):
    logger.info("Initiating graceful shutdown...")
    try:
        # Set shutdown flag
        network.shutdown_flag = True
        
        # Run the async shutdown in the event loop
        asyncio.run_coroutine_threadsafe(
            shutdown_async(gui, blockchain, network, rotation_manager),
            loop
        ).result(timeout=10)
        
        # Stop GUI (this must run in the main thread)
        gui.exit()
        
        # Cancel all pending tasks
        for task in asyncio.all_tasks(loop):
            task.cancel()
        
        # Stop the event loop
        loop.call_soon_threadsafe(loop.stop)
        logger.info("Shutdown completed successfully")
    except Exception as e:
        logger.error(f"Error during shutdown: {e}", exc_info=True)
        sys.exit(1)

async def run_async_tasks(blockchain: Blockchain, network: BlockchainNetwork, rotation_manager, loop: asyncio.AbstractEventLoop):
    """Run initial async tasks."""
    try:
        # Give servers time to start
        await asyncio.sleep(3)
        
        # Check health
        if not await health_check(network.host, network.port, network.client_ssl_context):
            logger.error("Health check failed after retries, exiting...")
            await network.stop()
            return False
            
        # Return success
        return True
    except Exception as e:
        logger.error(f"Critical error in async tasks: {e}")
        try:
            await network.stop()
        except Exception as cleanup_error:
            logger.error(f"Error during cleanup: {cleanup_error}")
        return False

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

async def create_genesis_blockchain(node_id: str) -> Blockchain:
    """Initialize blockchain with genesis block"""
    # Create blockchain instance
    blockchain = Blockchain(node_id=node_id)
    
    # Initialize it
    await blockchain.initialize()
    
    # Check if we need to create a default wallet
    addresses = await blockchain.get_all_addresses()
    if not addresses:
        # Create default wallet
        default_address = await blockchain.create_wallet()
        logger.info(f"Created default wallet with address: {default_address}")
        
        # Add some initial coins to the default wallet (for testing)
        genesis_tx = Transaction(
            sender="0",
            recipient=default_address,
            amount=100.0,
            tx_type=TransactionType.COINBASE
        )
        
        # Create a genesis block with this transaction
        genesis_block = Block(
            index=0,
            transactions=[genesis_tx],
            previous_hash="0"
        )
        
        # Replace the empty genesis block
        blockchain.chain = [genesis_block]
        
        # Save the chain
        await blockchain.save_chain()
        logger.info(f"Created genesis block with initial coins for {default_address}")
    
    return blockchain

def run_async_loop(loop):
    """Run the asyncio event loop in a separate thread"""
    asyncio.set_event_loop(loop)
    try:
        loop.run_forever()
    except Exception as e:
        logger.error(f"Async loop crashed: {e}")
    finally:
        logger.info("Async loop stopped")
        # Ensure all tasks are properly cancelled and loop is closed
        pending = asyncio.all_tasks(loop)
        for task in pending:
            task.cancel()
        
        loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        loop.close()

async def async_main(args, loop):
    try:
        port = args.port if args.port else await find_available_port_async()
        api_port = port + 1000
        node_id = f"node{port}"
        logger.info(f"Initializing blockchain on {port} and key rotation API on {api_port}")
        
        security_monitor, mfa_manager, backup_manager = await initialize_security(node_id)
        await init_rotation_manager(node_id)
        rotation_manager = KeyRotationManager(node_id=node_id, is_validator=args.validator, backup_manager=backup_manager)
        blockchain = await create_genesis_blockchain(node_id)
        
        bootstrap_nodes = []
        if args.bootstrap:
            bootstrap_nodes = [(node.split(":")[0], int(node.split(":")[1])) for node in args.bootstrap.split(",")]
        
        network = BlockchainNetwork(blockchain, node_id, "127.0.0.1", port, bootstrap_nodes, security_monitor=security_monitor)
        network.loop = loop
        await network.start_server()
        
        await rotation_manager.start()
        from key_rotation.main import main as rotation_main
        shutdown_event = asyncio.Event()
        asyncio.create_task(rotation_main(node_id, args.validator, api_port, "127.0.0.1", loop, shutdown_event))
        
        return blockchain, network, security_monitor, mfa_manager, backup_manager, rotation_manager, shutdown_event
    except Exception as e:
        logger.error(f"Error in async initialization: {e}", exc_info=True)
        raise

def shutdown(gui, blockchain, network, rotation_manager, loop, shutdown_event=None):
    logger.info("Initiating graceful shutdown...")
    try:
        network.shutdown_flag = True
        asyncio.run_coroutine_threadsafe(
            shutdown_async(gui, blockchain, network, rotation_manager),
            loop
        ).result(timeout=10)
        gui.exit()
        if shutdown_event:
            loop.call_soon_threadsafe(shutdown_event.set)  # Signal key rotation to stop
        for task in asyncio.all_tasks(loop):
            task.cancel()
        loop.call_soon_threadsafe(loop.stop)
        logger.info("Shutdown completed successfully")
    except Exception as e:
        logger.error(f"Error during shutdown: {e}", exc_info=True)
        sys.exit(1)

def main():
    set_resource_limits()
    parser = argparse.ArgumentParser(description="Run a blockchain node.")
    parser.add_argument("--port", type=int, default=None, help="Port to run the node on (1024-65535)")
    parser.add_argument("--bootstrap", type=str, default=None, help="Comma-separated list of bootstrap nodes (host:port)")
    parser.add_argument("--validator", action="store_true", help="Run as validator node for key rotation")
    args = parser.parse_args()

    if args.port and not validate_port(args.port):
        logger.error("Invalid port number")
        sys.exit(1)
    if args.bootstrap and not validate_bootstrap_nodes(args.bootstrap):
        logger.error("Invalid bootstrap nodes format")
        sys.exit(1)

    loop = asyncio.new_event_loop()
    async_thread = threading.Thread(target=run_async_loop, args=(loop,), daemon=True)
    async_thread.start()
    
    try:
        init_future = asyncio.run_coroutine_threadsafe(async_main(args, loop), loop)
        blockchain, network, security_monitor, mfa_manager, backup_manager, rotation_manager, shutdown_event = init_future.result(timeout=30)
        
        gui = BlockchainGUI(blockchain, network, mfa_manager=mfa_manager, backup_manager=backup_manager)
        gui.loop = loop
        
        def signal_handler(sig, frame):
            logger.info(f"Received signal {sig}, initiating shutdown")
            shutdown(gui, blockchain, network, rotation_manager, loop, shutdown_event)
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        gui.run()
    except Exception as e:
        logger.error(f"Error in main: {e}", exc_info=True)
    finally:
        logger.info("Main thread exiting, cleaning up...")
        loop.call_soon_threadsafe(loop.stop)
        async_thread.join(timeout=5)
        if async_thread.is_alive():
            logger.warning("Event loop thread did not terminate cleanly")

if __name__ == "__main__":
    main()
