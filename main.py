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
from network import load_config, NodeIdentity
import getpass
from utils import init_rotation_manager, find_available_port_async
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
    # In our new architecture, health check is on the API port (port + 1)
    health_port = port + 1
    
    try:
        timeout = aiohttp.ClientTimeout(total=5)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            for attempt in range(retries):
                try:
                    # Use the /health endpoint on the HTTP API server
                    async with session.get(f"https://{host}:{health_port}/health", 
                                           ssl=client_ssl_context) as resp:
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
    """Run initial async tasks."""
    try:
        # Give servers time to start
        await asyncio.sleep(3)
        
        # Check health using the network's port (HTTP API will be on port+1)
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

def get_wallet_password_synchronous(blockchain: Blockchain) -> str:
    """Synchronously get wallet password for existing wallet or create a new one."""
    print("\n=== Wallet Connection ===")
    choice = input("Do you want to connect to an existing wallet (1) or create a new one (2)? [1/2]: ").strip()
    
    if choice == "2":
        # Create a new wallet synchronously (using run_until_complete)
        loop = asyncio.new_event_loop()
        try:
            address = loop.run_until_complete(blockchain.create_wallet())
            print(f"New wallet created with address: {address}")
            while True:
                password = getpass.getpass("Set a wallet encryption password: ").strip()
                if not password:
                    print("Password cannot be empty. Please try again.")
                    continue
                confirm = getpass.getpass("Confirm password: ").strip()
                if password != confirm:
                    print("Passwords do not match. Please try again.")
                    continue
                blockchain.key_manager.password = password
                loop.run_until_complete(blockchain.save_wallets())
                print("Wallet encrypted and saved successfully.")
                return password
        finally:
            loop.close()
    
    elif choice == "1":
        while True:
            password = getpass.getpass("Enter wallet encryption password: ").strip()
            if not password:
                print("Password cannot be empty. Please try again.")
                continue
            loop = asyncio.new_event_loop()
            try:
                original_password = blockchain.key_manager.password
                blockchain.key_manager.password = password
                wallets = loop.run_until_complete(blockchain.key_manager.load_keys())
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
            except Exception as e:
                logger.error(f"Error loading wallet: {e}")
                raise
            finally:
                loop.close()
    else:
        print("Invalid choice. Please enter 1 or 2.")
        return get_wallet_password_synchronous(blockchain)
    
async def create_genesis_blockchain(node_id: str, wallet_password: str) -> Blockchain:
    """Initialize blockchain with genesis block"""
    # Create blockchain instance
    blockchain = Blockchain(node_id=node_id, wallet_password=wallet_password)
    
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

async def async_main(args, loop, wallet_password: str):
    try:
        config = load_config(args.config)

        # Override config with command line arguments if provided
        if args.p2p_port:
            config["p2p_port"] = args.p2p_port
        if args.api_port:
            config["api_port"] = args.api_port
        if args.data_dir:
            config["data_dir"] = args.data_dir

        # Initialize network with the correct config
        bootstrap_nodes = []
         # Check for bootstrap nodes from args before defaulting to empty list
        if args.bootstrap:
            try:
                for node in args.bootstrap.split(","):
                    node = node.strip()
                    host, port_str = node.split(":")
                    port_num = int(port_str)
                    if not validate_port(port_num) or not host:
                        raise ValueError(f"Invalid bootstrap node format: {node}")
                    bootstrap_nodes.append((host, port_num))
                logger.info(f"Using bootstrap nodes from args: {bootstrap_nodes}")
            except Exception as e:
                logger.error(f"Failed to parse bootstrap nodes '{args.bootstrap}': {e}")
                logger.info("Falling back to no bootstrap nodes")
                bootstrap_nodes = []

        # Initialize blockchain with node identity
        node_identity = NodeIdentity(config["data_dir"])
        node_id, private_key, public_key = await node_identity.initialize()
        
        # Initialize blockchain with default password first
        blockchain = Blockchain(node_id=node_id, wallet_password=wallet_password, port=config["p2p_port"])  # No password yet
        
        security_monitor, mfa_manager, backup_manager = await initialize_security(node_id)
        from utils import rotation_manager
        if not rotation_manager:
            await init_rotation_manager(node_id)  # Initializes global rotation_manager

        # Create network with the correct config
        network = BlockchainNetwork(
            blockchain, 
            node_id, 
            "127.0.0.1", 
            config["p2p_port"],
            bootstrap_nodes, 
            security_monitor=security_monitor,
            config_path=args.config
        )

        # Manually set the network's private and public keys from node_identity
        network.private_key = private_key
        network.public_key = public_key
        network.loop = loop
        
        # Start the network
        network_start_task = asyncio.create_task(network.start())
        
        key_rotation_port = config["key_rotation_port"]
        from utils import is_port_available, find_available_port_async

        key_rotation_port = config["key_rotation_port"]
        if not is_port_available(key_rotation_port, host="127.0.0.1"):
            logger.warning(f"Key rotation port {key_rotation_port} is already in use.")
            # Try to find an available port
            try:
                # Find the next available port starting from the current one
                new_port = await find_available_port_async(
                    start_port=key_rotation_port + 1, 
                    end_port=key_rotation_port + 100,
                    host="127.0.0.1"
                )
                logger.info(f"Using alternative port {new_port} for key rotation service")
                key_rotation_port = new_port
                config["key_rotation_port"] = new_port
            except Exception as e:
                logger.error(f"Failed to find available port for key rotation: {e}")
                # Continue with original port and let the service handle any binding errors

        # Start key rotation with the potentially updated port
        from key_rotation.main import main as rotation_main
        shutdown_event = asyncio.Event()
        rotation_task = asyncio.create_task(rotation_main(
            node_id, args.validator, key_rotation_port, "127.0.0.1", loop, shutdown_event))
        
        # Store tasks in network object for later cleanup
        network.background_tasks = [network_start_task, rotation_task]

        return blockchain, network, security_monitor, mfa_manager, backup_manager, rotation_manager, shutdown_event
    except Exception as e:
        logger.error(f"Error in async initialization: {e}", exc_info=True)
        raise

def main():
    set_resource_limits()
    parser = argparse.ArgumentParser(description="Run a blockchain node.")
    parser.add_argument("--config", type=str, default="config.json", help="Path to configuration file")
    parser.add_argument("--p2p-port", type=int, default=None, help="P2P communication port (default: 8333)")
    parser.add_argument("--api-port", type=int, default=None, help="HTTP API port (default: 8332)")
    parser.add_argument("--data-dir", type=str, default=None, help="Data directory (default: data)")
    parser.add_argument("--bootstrap", type=str, default=None, help="Comma-separated list of bootstrap nodes (host:port)")
    parser.add_argument("--validator", action="store_true", help="Run as validator node for key rotation")

    args = parser.parse_args()

    if args.p2p_port and not validate_port(args.p2p_port):
        logger.error("Invalid port number")
        sys.exit(1)
    if args.bootstrap and not validate_bootstrap_nodes(args.bootstrap):
        logger.error("Invalid bootstrap nodes format")
        sys.exit(1)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    async_thread = threading.Thread(target=run_async_loop, args=(loop,), daemon=True)
    async_thread.start()

    # Create blockchain instance for wallet setup
    blockchain = Blockchain(node_id=f"node_temp", wallet_password=None, port=args.p2p_port)
    
    # Get wallet password synchronously before starting async loop
    wallet_password = get_wallet_password_synchronous(blockchain)
    
    # Initialize components asynchronously
    try:
        blockchain, network, security_monitor, mfa_manager, backup_manager, rotation_manager, shutdown_event = asyncio.run(
            async_main(args, loop, wallet_password)
        )
    except Exception as e:
        logger.error(f"Initialization failed: {e}", exc_info=True)
        sys.exit(1)

    # Create and configure the GUI
    gui = BlockchainGUI(blockchain, network, mfa_manager=mfa_manager, backup_manager=backup_manager)
    gui.loop = loop
    gui.loop_thread = async_thread

    # Initialize the GUI after setting up the event loop
    try:
        gui.run()
    except Exception as e:
        logger.error(f"Failed to initialize GUI: {e}", exc_info=True)
        sys.exit(1)
    finally:
        logger.info("Application fully shut down")
        sys.exit(0)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.critical(f"Critical error in main program: {e}", exc_info=True)
        sys.exit(1)