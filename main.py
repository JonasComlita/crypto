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
from network import BlockchainNetwork
from utils import find_available_port, init_rotation_manager, find_available_port_async
from gui import BlockchainGUI
import threading
import aiohttp
from threading import Lock, Event
from security import SecurityMonitor, MFAManager, KeyBackupManager
from key_rotation.core import KeyRotationManager

# Add the project root directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Global flag to prevent double shutdown
shutdown_in_progress = False
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

async def shutdown_async(gui, blockchain, network, rotation_manager, shutdown_event=None):
    """Asynchronous shutdown sequence, runs in the event loop"""
    logger.info("Initiating graceful async shutdown...")
    try:
        # Stop mining if active
        if hasattr(blockchain, 'miner') and blockchain.miner.mining:
            await blockchain.stop_mining()
            logger.info("Mining stopped during shutdown")
        
        # Stop network operations
        if network:
            await network.stop()
            logger.info("Network stopped")
        
        # Stop blockchain
        if blockchain and hasattr(blockchain, 'shutdown'):
            await blockchain.shutdown()
            logger.info("Blockchain shutdown completed")
        
        # Stop rotation manager
        if rotation_manager and hasattr(rotation_manager, 'stop'):
            await rotation_manager.stop()
            logger.info("Key rotation manager stopped")
        
        # Clean up security monitoring
        if network.security_monitor and hasattr(network.security_monitor, 'cleanup'):
            await network.security_monitor.cleanup()
            logger.info("Security monitor cleaned up")
            
        # Set shutdown event if provided
        if shutdown_event:
            shutdown_event.set()
            
        logger.info("Async shutdown completed successfully")
    except Exception as e:
        logger.error(f"Error during async shutdown: {e}", exc_info=True)

def graceful_shutdown(gui, blockchain, network, rotation_manager, loop, shutdown_event=None):
    """Improved graceful shutdown function that prevents race conditions"""
    global shutdown_in_progress
    
    # Prevent multiple shutdown attempts
    if shutdown_in_progress:
        logger.info("Shutdown already in progress, skipping")
        return
        
    shutdown_in_progress = True
    logger.info("Initiating graceful shutdown...")
    
    try:
        # Set shutdown flag on network
        if network and hasattr(network, 'shutdown_flag'):
            network.shutdown_flag = True
        
        # Create a flag to track GUI shutdown
        gui_shutdown_complete = Event()
        
        # Ensure the GUI is notified to begin shutdown
        try:
            # Assuming gui.exit() will eventually trigger gui.root.destroy()
            # Modify gui.exit() to set this event when complete
            old_exit = getattr(gui, 'exit', None)
            
            def new_exit():
                if old_exit:
                    old_exit()
                gui_shutdown_complete.set()
                
            gui.exit = new_exit
            
            # Start GUI shutdown
            if threading.current_thread() is not threading.main_thread():
                gui.root.after(0, gui.root.quit)
            else:
                gui.root.quit()
                
        except Exception as e:
            logger.error(f"Error initiating GUI shutdown: {e}")
            # Continue with other shutdown steps
        
        try:
            # Run async shutdown
            future = asyncio.run_coroutine_threadsafe(
                shutdown_async(gui, blockchain, network, rotation_manager, shutdown_event),
                loop
            )
            # Wait for async shutdown with timeout
            future.result(timeout=8)
        except concurrent.futures.TimeoutError:
            logger.warning("Async shutdown timed out")
        except Exception as e:
            logger.error(f"Error running async shutdown: {e}")
        
        # Wait for GUI to finish closing (with timeout)
        gui_shutdown_complete.wait(timeout=3)
        
        # Safe cleanup of the loop
        if not loop.is_closed():
            try:
                # Cancel remaining tasks
                remaining_tasks = [t for t in asyncio.all_tasks(loop) 
                                  if t is not asyncio.current_task()]
                
                if remaining_tasks:
                    logger.info(f"Cancelling {len(remaining_tasks)} remaining tasks")
                    for task in remaining_tasks:
                        task.cancel()
                
                # Use a safe approach to stop the loop
                try:
                    loop.call_soon_threadsafe(loop.stop)
                except RuntimeError:
                    # Loop might already be closed
                    pass
                    
            except Exception as e:
                logger.error(f"Error cleaning up loop: {e}")
        
        logger.info("Shutdown procedures completed")
        
    except Exception as e:
        logger.error(f"Critical error during shutdown: {e}", exc_info=True)
        sys.exit(1)

def add_exit_method_to_gui(gui):
    """Add or enhance the exit method on the GUI"""
    if not hasattr(gui, 'exit'):
        def exit_method():
            try:
                gui.root.quit()
                gui.root.update()  # Process any pending events
                gui.root.destroy()
            except Exception as e:
                logger.error(f"Error in GUI exit: {e}")
        
        gui.exit = exit_method
        logger.info("Added exit method to GUI")


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
        try:
            # Ensure all tasks are properly cancelled and loop is closed
            if not loop.is_closed():
                pending = asyncio.all_tasks(loop)
                for task in pending:
                    task.cancel()
                
                # Run the loop briefly to process cancellations
                loop.run_until_complete(asyncio.sleep(0.1))
                loop.close()
        except Exception as e:
            logger.error(f"Error closing loop: {e}")


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

def run_async_loop(loop):
    """Run the asyncio event loop in a separate thread"""
    asyncio.set_event_loop(loop)
    try:
        loop.run_forever()
    except Exception as e:
        logger.error(f"Async loop crashed: {e}")
    finally:
        logger.info("Async loop stopped")
        try:
            # Ensure all tasks are properly cancelled and loop is closed
            if not loop.is_closed():
                pending = asyncio.all_tasks(loop)
                for task in pending:
                    task.cancel()
                
                # Run the loop briefly to process cancellations
                loop.run_until_complete(asyncio.sleep(0.1))
                loop.close()
        except Exception as e:
            logger.error(f"Error closing loop: {e}")


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

    # Create the loop and start it in a separate thread
    loop = asyncio.new_event_loop()
    async_thread = threading.Thread(target=run_async_loop, args=(loop,), daemon=True)
    async_thread.start()
    
    # Initialize components
    blockchain = None
    network = None
    rotation_manager = None
    gui = None
    shutdown_event = None
    
    try:
        # Initialize the application
        init_future = asyncio.run_coroutine_threadsafe(async_main(args, loop), loop)
        components = init_future.result(timeout=30)
        blockchain, network, security_monitor, mfa_manager, backup_manager, rotation_manager, shutdown_event = components
        
        # Create and configure the GUI
        gui = BlockchainGUI(blockchain, network, mfa_manager=mfa_manager, backup_manager=backup_manager)
        gui.loop = loop
        
        # Add an exit method if needed
        add_exit_method_to_gui(gui)
        
        # Configure signal handlers for graceful shutdown
        def signal_handler(sig, frame):
            logger.info(f"Received signal {sig}, initiating shutdown")
            graceful_shutdown(gui, blockchain, network, rotation_manager, loop, shutdown_event)
            # Exit with a small delay to allow logging to complete
            time.sleep(0.5)
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Override the GUI's on_closing method to use our graceful shutdown
        original_on_closing = gui.on_closing
        def new_on_closing():
            logger.info("Window close requested, initiating shutdown")
            try:
                graceful_shutdown(gui, blockchain, network, rotation_manager, loop, shutdown_event)
                # Don't call the original as it might cause issues
            except Exception as e:
                logger.error(f"Error in custom on_closing: {e}")
                # Try the original as a fallback
                try:
                    original_on_closing()
                except:
                    pass
        
        gui.on_closing = new_on_closing
        gui.root.protocol("WM_DELETE_WINDOW", gui.on_closing)
        
        # Run the GUI
        gui.run()
        
    except Exception as e:
        logger.error(f"Error in main: {e}", exc_info=True)
    finally:
        # Final cleanup if we got here without a proper shutdown
        logger.info("Main thread exiting, final cleanup...")
        
        # If we haven't shut down yet, do it now
        if not shutdown_in_progress and gui and loop and not loop.is_closed():
            try:
                graceful_shutdown(gui, blockchain, network, rotation_manager, loop, shutdown_event)
            except Exception as e:
                logger.error(f"Error in final shutdown: {e}")
        
        # Wait for the async thread to end (with timeout)
        if 'async_thread' in locals():
            async_thread.join(timeout=3)
            if async_thread.is_alive():
                logger.warning("Event loop thread did not terminate cleanly")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.critical(f"Critical error in main program: {e}", exc_info=True)
        sys.exit(1)