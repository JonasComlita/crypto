import asyncio
import ssl
import argparse
import logging
import signal
import sys
from blockchain import Blockchain
from network import BlockchainNetwork
from utils import find_available_port, init_rotation_manager
from gui import BlockchainGUI
import threading
import aiohttp

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def health_check(host: str, port: int, client_ssl_context, retries: int = 5, delay: float = 1.0) -> bool:
    try:
        async with aiohttp.ClientSession() as session:
            for attempt in range(retries):
                try:
                    async with session.get(f"https://{host}:{port}/health", ssl=client_ssl_context) as resp:
                        if resp.status == 200:
                            return True
                except Exception as e:
                    logger.warning(f"Health check failed (attempt {attempt + 1}/{retries}): {e}")
                await asyncio.sleep(delay)
            return False
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return False

def shutdown(gui: BlockchainGUI, network: BlockchainNetwork, rotation_manager, loop: asyncio.AbstractEventLoop):
    logger.info("Shutting down...")
    gui.exit()
    asyncio.run_coroutine_threadsafe(network.stop(), network.loop)
    asyncio.run_coroutine_threadsafe(rotation_manager.stop(), loop)
    loop.stop()

async def run_async_tasks(gui: BlockchainGUI, network: BlockchainNetwork, rotation_manager, loop: asyncio.AbstractEventLoop):
    """Run initial async setup tasks."""
    port = network.port
    await asyncio.sleep(2)
    if not await health_check(network.host, network.port, network.client_ssl_context):
        logger.error("Health check failed after retries, exiting...")
        gui.root.quit()
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Run a blockchain node.")
    parser.add_argument("--port", type=int, default=None, help="Port to run the node on")
    parser.add_argument("--bootstrap", type=str, default=None, help="Comma-separated list of bootstrap nodes (host:port)")
    parser.add_argument("--validator", action="store_true", help="Run as validator node for key rotation")
    args = parser.parse_args()

    port = args.port if args.port else find_available_port()
    api_port = port + 1000
    node_id = f"node{port}"
    logger.info(f"Initializing blockchain on {port} and key rotation API on {api_port}")

    # Initialize key rotation (synchronous part)
    init_rotation_manager(node_id)
    from key_rotation.core import KeyRotationManager
    rotation_manager = KeyRotationManager(node_id=node_id, is_validator=args.validator)

    # Start key rotation API in a separate thread
    from key_rotation.main import main as rotation_main
    rotation_thread = threading.Thread(
        target=lambda: asyncio.run(rotation_main(node_id, args.validator, api_port, "127.0.0.1")),
        daemon=True
    )
    rotation_thread.start()

    # Initialize blockchain
    blockchain = Blockchain()
    
    # Parse bootstrap nodes
    bootstrap_nodes = []
    if args.bootstrap:
        bootstrap_nodes = [(node.split(":")[0], int(node.split(":")[1])) for node in args.bootstrap.split(",")]
    elif port != 5000:
        bootstrap_nodes = [("127.0.0.1", 5000)]

    # Set up network
    network = BlockchainNetwork(blockchain, node_id, "127.0.0.1", port, bootstrap_nodes)
    logger.info(f"Node {node_id} public key: {network.public_key}")

    network_thread = threading.Thread(target=network.run, daemon=True)
    network_thread.start()

    # Create GUI
    gui = BlockchainGUI(blockchain, network)
    
    # Set up asyncio loop to run alongside Tkinter
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    # Run initial async tasks
    loop.run_until_complete(blockchain.initialize())
    loop.run_until_complete(rotation_manager.start())
    loop.run_until_complete(run_async_tasks(gui, network, rotation_manager, loop))

    # Integrate asyncio with Tkinter
    def run_asyncio():
        try:
            loop.run_forever()
        except Exception as e:
            logger.error(f"Asyncio loop error: {e}")
        gui.root.after(100, run_asyncio)  # Reschedule if stopped

    gui.root.after(100, run_asyncio)

    # Signal handlers
    signal.signal(signal.SIGINT, lambda s, f: shutdown(gui, network, rotation_manager, loop))
    signal.signal(signal.SIGTERM, lambda s, f: shutdown(gui, network, rotation_manager, loop))

    # Run Tkinter main loop in the main thread
    gui.run()

if __name__ == "__main__":
    main()