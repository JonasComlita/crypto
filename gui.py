import tkinter as tk
from tkinter import ttk, messagebox, filedialog, simpledialog
import time
from queue import Queue
from typing import Dict, List, Optional
from blockchain import Blockchain, Block, Transaction, TransactionType, NetworkService
from network import BlockchainNetwork
from utils import SecurityUtils, generate_wallet, derive_key, Fernet
import logging
import asyncio
from security import SecurityMonitor, MFAManager, KeyBackupManager
import os
from PIL import ImageTk
import queue
import threading
import gc
import concurrent.futures
import sys
import random
# Configure logging
logger = logging.getLogger(__name__)

class BlockchainGUI:
    from blockchain import Blockchain
    def __init__(self, blockchain, network, mfa_manager=None, backup_manager=None):
        self.blockchain = blockchain
        self.blockchain.subscribe("new_block", self.on_new_block)
        self.blockchain.subscribe("new_transaction", self.on_new_transaction)
        self.network = network
        self.mfa_manager = mfa_manager
        self.backup_manager = backup_manager
        self.mining = False
        self._shutdown_in_progress = False  # Add this if not already present
        self.loop = None  # Will be set by main.py
        
        self.root = tk.Tk()
        self.root.title("OriginalCoin Blockchain")
        self.root.geometry("600x600")  # Set initial window size

        # Queue for thread-safe updates
        self.update_queue = queue.Queue()
        self.root.after(100, self.process_queue)
        
        # Style configuration
        self.style = ttk.Style()
        self.style.configure('Mining.TButton', background='green')
        self.style.configure('Header.TLabel', font=('Helvetica', 12, 'bold'))
        self.style.configure('Info.TLabel', font=('Helvetica', 10))
        
        # Initialize wallet variables
        self.selected_wallet = tk.StringVar()
        self.amount_var = tk.StringVar()
        self.recipient_var = tk.StringVar()
        
        # Create main container with padding
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.main_frame)
        self.notebook.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Initialize tabs
        self.init_wallet_tab()
        self.init_mining_tab()  # This creates self.mining_frame
        self.init_network_tab()

        self.schedule_balance_updates()
        self.update_network_tab()
        
        # Create mining button in the controls frame
        self.mining_btn = ttk.Button(
            # Use the controls_frame inside the mining_frame
            self.mining_frame.winfo_children()[0],  # First child is the controls_frame
            text="Start Mining", 
            command=self.toggle_mining,
            style='Mining.TButton'
        )
        self.mining_btn.grid(row=0, column=0, padx=5, pady=5)
        
        # Status bar at bottom
        self.status_var = tk.StringVar(value="Ready")
        self.status_bar = ttk.Label(
            self.root,
            textvariable=self.status_var,
            relief=tk.SUNKEN,
            padding=(5, 2)
        )
        self.status_bar.grid(row=1, column=0, sticky=(tk.W, tk.E))
        
        # Rest of the initialization remains the same
        self.update_queue = queue.Queue()
        
        # Add security status indicators
        self.mfa_status = tk.StringVar(value="MFA: Not Configured")
        self.backup_status = tk.StringVar(value="Backup: Not Configured")
        
        # Add security controls to the interface
        self.add_security_controls()

        # Bind window close event
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Add session management
        self.session_timeout = 30 * 60  # 30 minutes
        self.last_activity = time.time()
        
        # Add rate limiting
        self.mfa_attempts = {}
        self.max_mfa_attempts = 3
        self.mfa_lockout_time = 300  # 5 minutes

        # Add caching
        self.cache = {}
        self.cache_timeout = 60  # 1 minute
        
        # Batch processing
        self.update_batch_size = 100
        self.update_interval = 2000  # 2 seconds

    def init_wallet_tab(self):
        """Initialize wallet management tab"""
        wallet_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(wallet_frame, text="Wallet")
        
        # Wallet selection frame
        select_frame = ttk.LabelFrame(wallet_frame, text="Wallet Selection", padding="5")
        select_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        # Wallet dropdown and refresh button
        self.wallet_dropdown = ttk.OptionMenu(
            select_frame,
            self.selected_wallet,
            "Select Wallet",
            *[]
        )
        self.wallet_dropdown.grid(row=0, column=0, padx=5, pady=5)
        
        refresh_btn = ttk.Button(
            select_frame,
            text="↻ Refresh",
            command=self.update_wallet_dropdown
        )
        refresh_btn.grid(row=0, column=1, padx=5, pady=5)
        
        new_wallet_btn = ttk.Button(
            select_frame,
            text="+ New Wallet",
            command=self.create_new_wallet
        )
        new_wallet_btn.grid(row=0, column=2, padx=5, pady=5)

        # Change password button
        change_pass_btn = ttk.Button(
            select_frame,
            text="Change Wallet Password",
            command=self.handle_change_password
        )
        change_pass_btn.grid(row=0, column=3, padx=5, pady=5)
        
        # Balance display
        self.balance_label = ttk.Label(
            wallet_frame,
            text="Balance: 0 ORIG",
            style='Header.TLabel'
        )
        self.balance_label.grid(row=1, column=0, pady=10)
        
        # Transaction frame
        tx_frame = ttk.LabelFrame(wallet_frame, text="Send Transaction", padding="5")
        tx_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        # Amount input
        ttk.Label(tx_frame, text="Amount:").grid(row=0, column=0, padx=5, pady=5)
        self.amount_var = tk.StringVar()
        amount_entry = ttk.Entry(
            tx_frame,
            textvariable=self.amount_var,
            width=40
        )
        amount_entry.grid(row=0, column=1, padx=5, pady=5)
        
        # Recipient input
        ttk.Label(tx_frame, text="Recipient:").grid(row=1, column=0, padx=5, pady=5)
        self.recipient_var = tk.StringVar()
        recipient_entry = ttk.Entry(
            tx_frame,
            textvariable=self.recipient_var,
            width=40
        )
        recipient_entry.grid(row=1, column=1, padx=5, pady=5)
        
        # Send button
        send_btn = ttk.Button(
            tx_frame,
            text="Send",
            command=self.send_transaction
        )
        send_btn.grid(row=2, column=0, columnspan=2, pady=10)
        
        # Transaction history
        history_frame = ttk.LabelFrame(wallet_frame, text="Transaction History", padding="5")
        history_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        
        # Treeview for transaction history
        self.history_tree = ttk.Treeview(
            history_frame,
            columns=("timestamp", "from", "to", "amount"),
            show="headings",
            height=6
        )
        
        # Configure columns
        self.history_tree.heading("timestamp", text="Time")
        self.history_tree.heading("from", text="From")
        self.history_tree.heading("to", text="To")
        self.history_tree.heading("amount", text="Amount")
        
        self.history_tree.column("timestamp", width=150)
        self.history_tree.column("from", width=150)
        self.history_tree.column("to", width=150)
        self.history_tree.column("amount", width=100)
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(
            history_frame,
            orient=tk.VERTICAL,
            command=self.history_tree.yview
        )
        self.history_tree.configure(yscrollcommand=scrollbar.set)
        
        # Grid history components
        self.history_tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))

    def init_mining_tab(self):
        """Initialize mining control tab"""
        # Create the mining frame and set it as an instance attribute
        self.mining_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(self.mining_frame, text="Mining")
        
        # Mining controls
        controls_frame = ttk.LabelFrame(self.mining_frame, text="Mining Controls", padding="10")
        controls_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=5)
        
        # The mining button will be created in the __init__ method after this method is called
        
        # Mining stats
        stats_frame = ttk.LabelFrame(self.mining_frame, text="Mining Statistics", padding="10")
        stats_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=5)
        
        self.hashrate_var = tk.StringVar(value="Hashrate: 0 H/s")
        self.blocks_mined_var = tk.StringVar(value="Blocks Mined: 0")
        
        ttk.Label(
            stats_frame,
            textvariable=self.hashrate_var,
            style='Info.TLabel'
        ).grid(row=0, column=0, pady=2)
        
        ttk.Label(
            stats_frame,
            textvariable=self.blocks_mined_var,
            style='Info.TLabel'
        ).grid(row=1, column=0, pady=2)
        
        # Recent blocks
        blocks_frame = ttk.LabelFrame(self.mining_frame, text="Recent Blocks", padding="10")
        blocks_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        
        self.blocks_tree = ttk.Treeview(
            blocks_frame,
            columns=("index", "timestamp", "transactions", "hash"),
            show="headings",
            height=6
        )
        
        # Configure columns
        self.blocks_tree.heading("index", text="#")
        self.blocks_tree.heading("timestamp", text="Time")
        self.blocks_tree.heading("transactions", text="Transactions")
        self.blocks_tree.heading("hash", text="Hash")
        
        self.blocks_tree.column("index", width=50)
        self.blocks_tree.column("timestamp", width=150)
        self.blocks_tree.column("transactions", width=100)
        self.blocks_tree.column("hash", width=200)
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(
            blocks_frame,
            orient=tk.VERTICAL,
            command=self.blocks_tree.yview
        )
        self.blocks_tree.configure(yscrollcommand=scrollbar.set)
        
        # Grid blocks components
        self.blocks_tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))

    def init_network_tab(self):
        """Initialize network information tab"""
        network_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(network_frame, text="Network")
        
        # Node information
        info_frame = ttk.LabelFrame(network_frame, text="Node Information", padding="10")
        info_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=5)

        # Add sync button
        sync_btn = ttk.Button(
            info_frame,
            text="Sync Now",
            command=self.sync_chain
        )
        sync_btn.grid(row=0, column=0, padx=5, pady=5)
        
        self.node_id_var = tk.StringVar(value=f"Node ID: {self.network.node_id}")
        self.peers_var = tk.StringVar(value="Connected Peers: 0")
        
        ttk.Label(
            info_frame,
            textvariable=self.node_id_var,
            style='Info.TLabel'
        ).grid(row=0, column=0, pady=2)
        
        ttk.Label(
            info_frame,
            textvariable=self.peers_var,
            style='Info.TLabel'
        ).grid(row=1, column=0, pady=2)
        
        # Connected peers
        peers_frame = ttk.LabelFrame(network_frame, text="Connected Peers", padding="10")
        peers_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        
        self.peers_tree = ttk.Treeview(
            peers_frame,
            columns=("node_id", "address", "connected_since"),
            show="headings",
            height=6
        )
        
        # Configure columns
        self.peers_tree.heading("node_id", text="Node ID")
        self.peers_tree.heading("address", text="Address")
        self.peers_tree.heading("connected_since", text="Connected Since")
        
        self.peers_tree.column("node_id", width=150)
        self.peers_tree.column("address", width=150)
        self.peers_tree.column("connected_since", width=150)
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(
            peers_frame,
            orient=tk.VERTICAL,
            command=self.peers_tree.yview
        )
        self.peers_tree.configure(yscrollcommand=scrollbar.set)
        
        # Grid peers components
        self.peers_tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))

    def update_network_tab(self):
        """Update the Network tab with current network information."""
        try:
            # Update node ID
            self.node_id_var.set(f"Node ID: {self.network.node_id}")

            # Check if loop is initialized
            if not self.loop or not self.loop.is_running():
                logger.debug("Event loop not ready, scheduling network tab update later")
                self.root.after(1000, self.update_network_tab)
                return

            # Update peer count and list
            async def fetch_network_data():
                try:
                    # Get peer data from the network
                    peers = self.network.peers  # Assuming self.network.peers is a dict
                    peer_count = len(peers)
                    self.update_queue.put(
                        lambda: self.peers_var.set(f"Connected Peers: {peer_count}")
                    )

                    # Clear and update the peers tree
                    def update_peers_tree():
                        self.peers_tree.delete(*self.peers_tree.get_children())
                        current_time = time.time()
                        for peer_id, peer_data in peers.items():
                            # Assuming peer_data has 'host', 'port', and 'last_seen'
                            host = peer_data.get("host", "Unknown")
                            port = peer_data.get("port", "N/A")
                            connected_since = peer_data.get("last_seen", current_time)
                            if isinstance(connected_since, (int, float)):
                                connected_since_str = time.strftime(
                                    "%Y-%m-%d %H:%M:%S", time.localtime(connected_since)
                                )
                            else:
                                connected_since_str = str(connected_since)
                            self.peers_tree.insert(
                                "",
                                "end",
                                values=(peer_id, f"{host}:{port}", connected_since_str),
                            )
                    
                    self.update_queue.put(update_peers_tree)
                except Exception as e:
                    logger.error(f"Error fetching network data: {str(e)}")
                    self.update_queue.put(
                        lambda: self.show_error(f"Failed to update network tab: {str(e)}")
                    )

            # Run the async fetch in the event loop
            asyncio.run_coroutine_threadsafe(fetch_network_data(), self.loop)
            logger.debug("Network tab update scheduled")
        except Exception as e:
            logger.error(f"Error in update_network_tab: {str(e)}")
            self.show_error(f"Network tab update failed: {str(e)}")

    async def start_network_sync(self):
        """Start network synchronization"""
        try:
            # Check if the loop is set
            if not self.loop:
                logger.error("Cannot start network sync: loop not initialized")
                return
            
            await self.blockchain.sync_with_network()
            if hasattr(self.network, 'start_periodic_sync'):
                # Track the periodic sync task
                self.loop.create_task(self.network.start_periodic_sync(interval=10))
                logger.info("Started periodic network synchronization")
        except Exception as e:
            logger.error(f"Failed to start network sync: {e}", exc_info=True)

    def sync_chain(self):
        """Synchronize the blockchain with the network"""
        try:
            self.status_var.set("Syncing with network...")
            async def do_sync():
                try:
                    result = await self.blockchain.sync_with_network()
                    wallet = self.selected_wallet.get()
                    if wallet != "Select Wallet" and wallet:
                        await self.update_async_balance(wallet)
                        await self.update_async_transaction_history(wallet)
                    self.update_queue.put(lambda: self.status_var.set(
                        "Sync completed" if result else "Sync failed or no updates found"))
                except Exception as e:
                    logger.error(f"Chain sync error: {e}", exc_info=True)
                    self.update_queue.put(lambda: self.status_var.set(f"Sync error: {str(e)}"))
            
            asyncio.run_coroutine_threadsafe(do_sync(), self.loop)
        except Exception as e:
            self.show_error(f"Failed to sync chain: {str(e)}")

    def process_queue(self):
        """Process the update queue for thread-safe UI updates"""
        try:
            while not self.update_queue.empty():
                try:
                    func = self.update_queue.get_nowait()
                    logger.debug(f"Processing queue item: {func}")
                    func()
                except Exception as queue_error:
                    logger.error(f"Error processing queue item: {queue_error}", exc_info=True)
        except Exception as e:
            logger.error(f"Queue processing error: {e}")
        self.root.after(100, self.process_queue)

    async def on_new_block(self, block):
        if self.selected_wallet.get() != "Select Wallet":
            await self.update_async_balance(self.selected_wallet.get())
            await self.update_async_transaction_history(self.selected_wallet.get())

    def start_mining_callback(self):
        """Wrapper for starting mining with error handling"""
        try:
            logger.info("Starting mining callback triggered")
            wallet_address = self.selected_wallet.get()

            async def get_or_create_address():
                try:
                    addresses = await self.blockchain.get_all_addresses()
                    if addresses:
                        return addresses[0]
                    else:
                        return await self.blockchain.create_wallet()
                except Exception as addr_error:
                    logger.error(f"Error getting or creating wallet address: {addr_error}")
                    raise

            async def start_mining_coroutine():
                try:
                    # Get or create wallet address
                    wallet_address = await get_or_create_address()
                    
                    # Sanity check
                    if not wallet_address:
                        error_msg = "No wallet available for mining"
                        logger.error(error_msg)
                        self.update_queue.put(lambda: self.show_error(error_msg))
                        return

                    # Set the selected wallet if not already set
                    if self.selected_wallet.get() == "Select Wallet":
                        self.selected_wallet.set(wallet_address)

                    # Sync the network
                    await self.blockchain.sync_with_network()
                    
                    # Start mining
                    result = await self.blockchain.start_mining(wallet_address)
                    
                    if result:
                        self.update_queue.put(self.mining_started)
                        await self.update_async_balance(wallet_address)
                    else:
                        error_msg = "Failed to start mining"
                        logger.error(error_msg)
                        self.update_queue.put(lambda: self.show_error(error_msg))
                    
                except Exception as mining_error:
                    error_msg = f"Mining start failed: {str(mining_error)}"
                    logger.error(error_msg, exc_info=True)
                    self.update_queue.put(lambda: self.show_error(error_msg))

            # Run the coroutine in the event loop
            asyncio.run_coroutine_threadsafe(start_mining_coroutine(), self.loop)

        except Exception as outer_error:
            error_msg = f"Mining initialization error: {str(outer_error)}"
            logger.error(error_msg, exc_info=True)
            self.update_queue.put(lambda: self.show_error(error_msg))
            
    def show_error(self, message):
        """Thread-safe method to show error message"""
        logger.error(message)
        if hasattr(self, 'status_var'):
            self.status_var.set(message)
        if threading.current_thread() is threading.main_thread():
            messagebox.showerror("Error", message)
        else:
            self.root.after(0, lambda: messagebox.showerror("Error", message))

    def toggle_mining(self):
        """Toggle mining on and off"""
        try:
            if not self.mining:
                # Start mining
                self.start_mining_callback()
            else:
                # Stop mining
                def stop_mining_thread():
                    try:
                        logger.info("Stopping mining...")
                        result = asyncio.run_coroutine_threadsafe(
                            self.blockchain.stop_mining(), self.loop
                        ).result(timeout=5)
                        self.update_queue.put(
                            lambda: self.mining_stopped() if result else self.show_error("Failed to stop mining")
                        )
                    except Exception as e:
                        logger.error(f"Stop mining thread exception: {str(e)}")
                        self.update_queue.put(lambda: self.show_error(f"Mining stop failed: {str(e)}"))

                threading.Thread(target=stop_mining_thread, daemon=True).start()
        except Exception as e:
            self.show_error(f"Mining toggle failed: {str(e)}")

    def mining_started(self):
        """Update UI when mining starts successfully"""
        self.mining = True
        self.mining_btn.configure(text="Stop Mining")
        self.status_var.set("Mining started")
        logger.info("Mining UI updated to started state")
        self.start_mining_stats_update()
        wallet = self.selected_wallet.get()
        if wallet != "Select Wallet":
            asyncio.run_coroutine_threadsafe(self.update_async_balance(wallet), self.loop)

    def mining_stopped(self):
        """Update UI when mining stops successfully"""
        self.mining = False
        self.mining_btn.configure(text="Start Mining")
        self.status_var.set("Mining stopped")
        logger.info("Mining UI updated to stopped state")

    def start_mining_stats_update(self):
        """Periodically update mining stats while mining is active"""
        async def update_stats():
            if not self.mining:
                return
            try:
                hashrate = await self.blockchain.get_hashrate()
                # Get blocks mined directly from the blockchain's miner
                blocks_mined = self.blockchain.miner.blocks_mined if hasattr(self.blockchain.miner, 'blocks_mined') else 0
                
                logger.info(f"Updating mining stats - Hashrate: {hashrate:.2f} H/s, Blocks: {blocks_mined}")
                self.hashrate_var.set(f"Hashrate: {hashrate:.2f} H/s")
                self.blocks_mined_var.set(f"Blocks Mined: {blocks_mined}")
                
                # Update recent blocks
                self.update_queue.put(lambda: self.update_blocks_tree(self.blockchain.chain[-6:]))
                logger.info(f"Updated stats - Hashrate: {hashrate:.2f} H/s, Blocks: {blocks_mined}")
            except Exception as e:
                error_msg = f"Stats update error: {str(e)}"
                logger.error(f"Stats update failed: {e}")
                self.update_queue.put(lambda: self.status_var.set(error_msg))
        
        def schedule_next_update():
            if self.mining:
                # Schedule the next update
                logger.info("Scheduling next mining stats update")
                asyncio.run_coroutine_threadsafe(update_stats(), self.loop)
                self.root.after(1000, schedule_next_update)
            else:
                logger.info("Mining stopped, stopping stats update")

        # Start the update cycle
        logger.info("Starting mining stats update cycle")
        asyncio.run_coroutine_threadsafe(update_stats(), self.loop)
        self.root.after(1000, schedule_next_update)

    def update_blocks_tree(self, blocks):
        """Update the blocks tree with the given blocks"""
        self.blocks_tree.delete(*self.blocks_tree.get_children())
        for block in reversed(blocks):
            self.blocks_tree.insert(
                "", "end",
                values=(block.index, block.timestamp, len(block.transactions), block.hash[:20] + "...")
            )

    def _handle_new_wallet(self):
        """Handle creating a new wallet"""
        self.loop.create_task(self._create_new_wallet())

    def create_new_wallet(self):
        """Create a new wallet"""
        try:
            # Define the async wallet creation coroutine
            async def create_wallet_coroutine():
                try:
                    address = await self.blockchain.create_wallet()
                    self.update_queue.put(lambda: self.wallet_created(address))
                except Exception as e:
                    logger.error(f"Wallet creation exception: {str(e)}")
                    self.update_queue.put(lambda: self.show_error(f"Wallet creation failed: {str(e)}"))

            # Run the coroutine in the event loop thread-safely
            asyncio.run_coroutine_threadsafe(create_wallet_coroutine(), self.loop)
        except Exception as e:
            self.show_error(f"Wallet creation error: {str(e)}")

    def wallet_created(self, address):
        """Handle successful wallet creation"""
        self.update_wallet_dropdown()
        self.selected_wallet.set(address)
        self.status_var.set(f"Created new wallet: {address[:10]}...")
        asyncio.run_coroutine_threadsafe(self.update_async_balance(address), self.loop)

    def update_wallet_dropdown(self):
        """Update the wallet dropdown with available addresses"""
        try:
            # Use a future to manage the async operation
            async def fetch_addresses_coroutine():
                try:
                    # Retry up to 3 times with a delay if no addresses are found
                    for attempt in range(3):
                        addresses = await self.blockchain.get_all_addresses()
                        if addresses or attempt == 2:  # Proceed if addresses found or last attempt
                            self.update_queue.put(lambda: self.populate_wallet_dropdown(addresses))
                            break
                        logger.debug(f"No addresses found, retrying... (attempt {attempt + 1}/3)")
                        await asyncio.sleep(1)  # Wait 1 second before retrying
                except Exception as e:
                    logger.error(f"Address fetch exception: {e}")
                    self.update_queue.put(
                        lambda: self.show_error(f"Failed to get wallet addresses: {e}")
                    )

            # Run the coroutine in the event loop thread-safely
            logger.info("Updating wallet dropdown")
            future = asyncio.run_coroutine_threadsafe(fetch_addresses_coroutine(), self.loop)

            # Add a timeout handler to prevent blocking
            def check_future():
                try:
                    # Check if the future is done
                    if future.done():
                        # If there's an exception, it will be raised here
                        future.result(timeout=0)
                    else:
                        # If not done, schedule another check
                        self.root.after(100, check_future)
                except concurrent.futures.TimeoutError:
                    logger.warning("Wallet address fetch timed out")
                    self.update_queue.put(
                        lambda: self.show_error("Wallet address fetch timed out")
                    )
                except Exception as e:
                    logger.error(f"Wallet dropdown update error: {str(e)}")
                    self.update_queue.put(
                        lambda: self.show_error(f"Failed to update wallet dropdown: {str(e)}")
                    )

            # Start checking the future
            self.root.after(100, check_future)

        except Exception as e:
            logger.error(f"Wallet dropdown update initialization error: {str(e)}")
            self.show_error(f"Failed to initiate wallet dropdown update: {str(e)}")

    
    def populate_wallet_dropdown(self, addresses):
        """Populate the wallet dropdown with the given addresses"""
        try:
            # Clear existing menu items
            menu = self.wallet_dropdown["menu"]
            menu.delete(0, "end")
            
            if not addresses:
                # If no addresses, set to "Select Wallet"
                self.selected_wallet.set("Select Wallet")
                menu.add_command(
                    label="Select Wallet", 
                    command=lambda: self.selected_wallet.set("Select Wallet")
                )
                logger.info("No wallet addresses found")
                return
                    
            # Add wallet addresses
            for address in addresses:
                menu.add_command(
                    label=address[:10] + "...",  # Show shortened address
                    command=lambda a=address: self.on_wallet_selected(a)
                )
            
            # Log the number of wallets found
            logger.info(f"Populated wallet dropdown with {len(addresses)} addresses")
            
            # Select the first wallet by default if none is selected
            current = self.selected_wallet.get()
            if current == "Select Wallet" or current not in addresses:
                logger.info(f"Automatically selecting first wallet: {addresses[0]}")
                # Directly set the wallet and update balance
                self.selected_wallet.set(addresses[0])
                asyncio.run_coroutine_threadsafe(self.update_async_balance(addresses[0]), self.loop)
                asyncio.run_coroutine_threadsafe(self.update_async_transaction_history(addresses[0]), self.loop)
            
        except Exception as e:
            logger.error(f"Error populating wallet dropdown: {str(e)}", exc_info=True)
            self.show_error(f"Failed to populate wallet dropdown: {str(e)}")

    def on_wallet_selected(self, address):
        """Handle wallet selection with MFA verification"""
        try:
            logger.debug(f"Starting wallet selection for address: {address}")
            
            async def verify_and_select():
                if self.mfa_manager:
                    verified = await self.verify_mfa()
                    if not verified:
                        self.update_queue.put(
                            lambda: messagebox.showerror("Error", "MFA verification failed")
                        )
                        return
                
                logger.debug(f"Setting selected wallet to: {address}")
                self.selected_wallet.set(address)
                
                # Update the balance immediately
                await self.update_async_balance(address)
                await self.update_async_transaction_history(address)
                logger.debug(f"Balance updated for wallet: {address}")
            
            # Create and run the coroutine to verify and select the wallet
            asyncio.run_coroutine_threadsafe(verify_and_select(), self.loop)
                
        except Exception as e:
            logger.error(f"Error in wallet selection: {str(e)}", exc_info=True)
            self.update_queue.put(
                lambda: messagebox.showerror("Error", f"Error selecting wallet: {str(e)}")
            )

    def prompt_password(self, title: str, message: str) -> str:
        """Prompt user for a password"""
        password = simpledialog.askstring(
            title,
            message,
            show='*',
            parent=self.root
        )
        return password

    def handle_change_password(self):
        """Handle password change button click"""
        try:
            async def verify_and_change():
                # Verify MFA if enabled
                if self.mfa_manager:
                    if not await self.verify_mfa():
                        self.update_queue.put(
                            lambda: self.show_error("MFA verification required to change password")
                        )
                        return

                # Prompt and verify old password first
                old_password = self.prompt_password("Current Password", "Enter current wallet password:")
                if not old_password:
                    return

                # Explicitly test the old password
                try:
                    # Temporarily set KeyManager password to test decryption
                    original_password = self.blockchain.key_manager.password
                    self.blockchain.key_manager.password = old_password
                    await self.blockchain.key_manager.load_keys()  # Will raise if incorrect
                    self.blockchain.key_manager.password = original_password  # Restore on success
                except ValueError as e:
                    if "Incorrect password" in str(e):
                        self.update_queue.put(
                            lambda: self.show_error("Incorrect current password")
                        )
                    else:
                        self.update_queue.put(
                            lambda: self.show_error(f"Password verification failed: {str(e)}")
                        )
                    return

                # Prompt for new password
                new_password = self.prompt_password("New Password", "Enter new wallet password:")
                if not new_password:
                    return

                # Confirm new password
                confirm_password = self.prompt_password("Confirm Password", "Confirm new wallet password:")
                if new_password != confirm_password:
                    self.update_queue.put(
                        lambda: self.show_error("Confirmed password does not match new password")
                    )
                    return

                # Change the password
                await self.blockchain.change_wallet_password(new_password)

                self.update_queue.put(
                    lambda: messagebox.showinfo("Success", "Password changed successfully")
                )

            asyncio.run_coroutine_threadsafe(verify_and_change(), self.loop)
        except Exception as e:
            self.show_error(f"Error initiating password change: {str(e)}")

    async def verify_mfa(self):
        """Verify MFA code for the selected wallet"""
        if not self.mfa_manager:
            return True
        
        try:
            wallet_address = self.selected_wallet.get()
            if wallet_address == "Select Wallet" or not wallet_address:
                logger.info("No wallet selected, skipping MFA verification")
                return True
            
            is_configured = await self.mfa_manager.is_mfa_configured(wallet_address)
            if not is_configured:
                logger.info(f"MFA not configured for wallet {wallet_address}, skipping verification")
                return True
            
            code_future = asyncio.Future()
            
            def get_code():
                try:
                    code = simpledialog.askstring("MFA Required", "Enter MFA code:", show='*', parent=self.root)
                    self.loop.call_soon_threadsafe(lambda: code_future.set_result(code if code else None))
                except Exception as e:
                    logger.error(f"Error getting MFA code: {e}")
                    self.loop.call_soon_threadsafe(lambda: code_future.set_result(None))
            
            self.update_queue.put(get_code)
            code = await code_future
            if not code:
                return False
            
            result = self.mfa_manager.verify_mfa(wallet_address, code)
            logger.info(f"MFA verification {'successful' if result else 'failed'} for wallet {wallet_address}")
            return result
        except Exception as e:
            logger.error(f"MFA verification error: {e}", exc_info=True)
            return False

    async def update_async_balance(self, address):
        """Update the balance display asynchronously"""
        try:
            logger.debug(f"Fetching balance for address: {address}")
            balance = await self.blockchain.get_balance(address)
            logger.info(f"Balance for {address}: {balance:.2f} ORIG")
            
            # Add security indicator if wallet is backed up
            backup_status = ""
            if self.backup_manager:
                is_backed_up = await self.backup_manager.is_wallet_backed_up(address)
                if is_backed_up:
                    backup_status = "✓"
                    
            balance_text = f"Balance: {balance:.2f} ORIG {backup_status}"
            self.update_queue.put(lambda: self.balance_label.config(text=balance_text))
            
        except Exception as e:
            logger.error(f"Balance update error for {address}: {str(e)}", exc_info=True)
            self.update_queue.put(lambda: self.show_error(f"Failed to update balance: {str(e)}"))


    def schedule_balance_updates(self):
        """Periodically update the balance and check blockchain status"""
        async def update_chain_info():
            try:
                latest_block = self.blockchain.get_latest_block()
                if latest_block:
                    self.update_queue.put(lambda: self.status_var.set(f"Block: {latest_block.index}"))
                wallet = self.selected_wallet.get()
                if wallet and wallet != "Select Wallet":
                    balance = await self.blockchain.get_balance(wallet)
                    self.update_queue.put(lambda: self.balance_label.config(text=f"Balance: {balance:.2f} ORIG"))
                    logger.debug(f"Updated balance for {wallet}: {balance:.2f} ORIG")
                    if random.random() < 0.1:
                        logger.debug("Triggering chain synchronization")
                        await self.blockchain.sync_with_network()
            except Exception as e:
                logger.error(f"Chain update error: {str(e)}")
                self.update_queue.put(lambda: self.show_error(f"Chain update error: {str(e)}"))

        def schedule_next():
            if not self._shutdown_in_progress:
                # Check if loop is initialized before scheduling async task
                if hasattr(self, 'loop') and self.loop is not None:
                    try:
                        # Schedule the update
                        future = asyncio.run_coroutine_threadsafe(update_chain_info(), self.loop)
                        
                        # Add done callback to handle potential exceptions
                        def handle_future_result(fut):
                            try:
                                # Retrieve the result to ensure exceptions are raised
                                fut.result()
                            except Exception as e:
                                logger.error(f"Async balance update error: {str(e)}")
                        
                        future.add_done_callback(handle_future_result)
                    except Exception as e:
                        logger.error(f"Error scheduling async task: {str(e)}")
                else:
                    logger.debug("Event loop not ready yet, will retry on next schedule")
                
                # Schedule the next call regardless
                self.root.after(10000, schedule_next)  # 10000 ms = 10 seconds
            else:
                logger.info("Shutdown in progress, stopping balance updates")

        try:
            logger.info("Starting balance update scheduler")
            # Defer first update slightly to allow loop initialization
            self.root.after(1000, schedule_next)
        except Exception as e:
            logger.error(f"Error starting balance updates: {e}")
            self.show_error(f"Failed to start balance updates: {e}")
                

    async def update_async_transaction_history(self, address):
        """Update the transaction history asynchronously"""
        try:
            transactions = await self.blockchain.get_transactions_for_address(address)
            
            # Update the GUI from the main thread
            def update_history_tree():
                # Clear existing items
                self.history_tree.delete(*self.history_tree.get_children())
                
                # Add transactions to tree
                for tx in transactions:
                    self.history_tree.insert(
                        "",
                        "end",
                        values=(
                            tx.timestamp,
                            tx.sender[:10] + "..." if tx.sender else "Coinbase",
                            tx.recipient[:10] + "..." if tx.recipient else "N/A",
                            tx.amount
                        )
                    )
                    
            self.update_queue.put(update_history_tree)
            
        except Exception as e:
            logger.error(f"Transaction history update error: {str(e)}")
            self.update_queue.put(lambda: self.show_error(f"Failed to update transaction history: {str(e)}"))

    async def on_new_transaction(self, tx):
        if self.selected_wallet.get() != "Select Wallet":
            wallet = self.selected_wallet.get()
            if tx.sender == wallet or tx.recipient == wallet:
                await self.update_async_balance(wallet)
                await self.update_async_transaction_history(wallet)

    def send_transaction(self):
        """Send a transaction with security checks"""
        try:
            async def verify_and_send():
                if self.mfa_manager:
                    if not await self.verify_mfa():
                        self.update_queue.put(
                            lambda: messagebox.showerror("Error", "MFA verification failed")
                        )
                        return

                sender = self.selected_wallet.get()
                recipient = self.recipient_var.get()
                
                try:
                    amount = float(self.amount_var.get())
                except ValueError:
                    messagebox.showerror("Error", "Invalid amount")
                    return
        
                    
                if sender and recipient and amount > 0:
                    # Get sender's wallet info
                    async def send_transaction_async():
                        try:
                            wallet = await self.blockchain.get_wallet(sender)
                            if not wallet:
                                self.update_queue.put(lambda: messagebox.showerror("Error", "Wallet not found"))
                                return
                            
                            # Create and send transaction
                            tx = await self.blockchain.create_transaction(
                                wallet['private_key'],
                                sender,
                                recipient,
                                amount,
                                fee=0.001
                            )
                            
                            if not tx:
                                self.update_queue.put(lambda: messagebox.showerror("Error", "Failed to create transaction"))
                                return
                            
                            # Add to mempool
                            success = await self.blockchain.add_transaction_to_mempool(tx)
                            
                            if success:
                                # Backup keys after transaction if enabled
                                if self.backup_manager:
                                    await self.backup_manager.backup_transaction(tx)
                                
                                self.update_queue.put(lambda: messagebox.showinfo("Success", "Transaction sent successfully"))
                                
                                # Clear inputs and update display
                                self.update_queue.put(lambda: self.amount_var.set(""))
                                self.update_queue.put(lambda: self.recipient_var.set(""))
                                
                                # Update balance and transaction history
                                await self.update_async_balance(sender)
                                await self.update_async_transaction_history(sender)
                            else:
                                self.update_queue.put(lambda: messagebox.showerror("Error", "Transaction rejected by mempool"))
                        except Exception as e:
                            logger.error(f"Transaction send error: {str(e)}")
                            self.update_queue.put(lambda: messagebox.showerror("Error", f"Transaction failed: {str(e)}"))
                    
                    # Run the transaction in the event loop
                    asyncio.run_coroutine_threadsafe(send_transaction_async(), self.loop)
                else:
                    messagebox.showerror("Error", "Please fill all fields correctly")
                
            # Run the verification and transaction coroutine
            asyncio.run_coroutine_threadsafe(verify_and_send(), self.loop)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to send transaction: {str(e)}")

    def main_loop(self):
        """Run the asyncio event loop in a separate thread"""
        asyncio.set_event_loop(self.loop)
        try:
            self.loop.run_forever()
        except Exception as e:
            logger.error(f"Event loop error: {e}")
        finally:
            # Cleanup on loop stop
            if not self.loop.is_closed():
                pending = asyncio.all_tasks(self.loop)
                for task in pending:
                    task.cancel()
                try:
                    self.loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                except Exception as e:
                    logger.error(f"Error canceling tasks: {e}")
                self.loop.close()
                logger.info("Asyncio event loop closed")

    async def setup_mfa(self):
        """Handle MFA setup"""
        if not self.mfa_manager:
            return
            
        try:
            secret = self.mfa_manager.generate_mfa_secret(self.network.node_id)
            qr_code = self.mfa_manager.get_mfa_qr(
                self.network.node_id,
                f"OriginalCoin-{self.network.node_id}"
            )
            
            # Show QR code in a new window
            self.update_queue.put(lambda: self.show_qr_code(qr_code))
            
        except Exception as e:
            logger.error(f"MFA setup error: {e}")
            self.update_queue.put(lambda: messagebox.showerror("MFA Setup Error", str(e)))

    async def backup_keys(self):
        """Handle key backup"""
        if not self.backup_manager:
            return
            
        try:
            # Get password from UI thread
            password_future = asyncio.Future()
            self.update_queue.put(lambda: self.get_backup_password_wrapper(password_future))
            
            # Wait for password input
            password = await password_future
            if not password:
                return
                
            # Execute backup
            await self.backup_manager.create_backup(
                self.network.get_keys(),
                password
            )
            
            # Update UI
            self.update_queue.put(lambda: messagebox.showinfo("Success", "Keys backed up successfully"))
            
        except Exception as e:
            logger.error(f"Backup error: {e}")
            self.update_queue.put(lambda: messagebox.showerror("Backup Error", str(e)))

    def get_backup_password_wrapper(self, future):
        """UI thread wrapper for getting backup password"""
        password = self.get_backup_password()
        if self.loop.is_running():
            self.loop.call_soon_threadsafe(lambda: future.set_result(password))

    async def restore_keys(self):
        """Handle key restoration"""
        if not self.backup_manager:
            return
            
        try:
            # Get password from UI thread
            password_future = asyncio.Future()
            self.update_queue.put(lambda: self.get_backup_password_wrapper(password_future))
            
            # Wait for password input
            password = await password_future
            if not password:
                return
                
            # Get file path from UI thread
            file_path_future = asyncio.Future()
            self.update_queue.put(lambda: self.get_file_path_wrapper(file_path_future))
            
            # Wait for file path selection
            file_path = await file_path_future
            if not file_path:
                return
                
            # Execute restore
            keys = await self.backup_manager.restore_backup(file_path, password)
            await self.network.restore_keys(keys)
            
            # Update UI
            self.update_queue.put(lambda: messagebox.showinfo("Success", "Keys restored successfully"))
                
        except Exception as e:
            logger.error(f"Restore error: {e}")
            self.update_queue.put(lambda: messagebox.showerror("Restore Error", str(e)))

    def get_file_path_wrapper(self, future):
        """UI thread wrapper for getting file path"""
        file_path = filedialog.askopenfilename(
            title="Select Backup File",
            filetypes=[("Encrypted Backup", "*.enc")]
        )
        if self.loop.is_running():
            self.loop.call_soon_threadsafe(lambda: future.set_result(file_path))

    def get_backup_password(self) -> str:
        """Get password from user for backup/restore"""
        password = simpledialog.askstring(
            "Password Required",
            "Enter backup password:",
            show='*'
        )
        return password

    def show_qr_code(self, qr_code):
        """Display QR code in a new window"""
        qr_window = tk.Toplevel(self.root)
        qr_window.title("MFA Setup")
        
        # Convert QR code to PhotoImage
        qr_image = ImageTk.PhotoImage(qr_code)
        
        # Display QR code
        ttk.Label(qr_window, image=qr_image).pack(padx=10, pady=10)
        qr_window.qr_image = qr_image  # Keep reference to prevent garbage collection
        
        # Add instructions
        ttk.Label(
            qr_window,
            text="Scan this QR code with your authenticator app"
        ).pack(padx=10, pady=5)

    def add_security_controls(self):
        """Add security controls to the interface"""
        security_frame = ttk.LabelFrame(self.main_frame, text="Security Controls", padding="5")
        security_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        # MFA Setup button
        mfa_btn = ttk.Button(
            security_frame,
            text="Setup MFA",
            command=self.handle_mfa_setup
        )
        mfa_btn.grid(row=0, column=0, padx=5, pady=5)
        
        # Backup Controls
        backup_btn = ttk.Button(
            security_frame,
            text="Backup Keys",
            command=self.handle_backup_keys
        )
        backup_btn.grid(row=0, column=1, padx=5, pady=5)
        
        restore_btn = ttk.Button(
            security_frame,
            text="Restore Keys",
            command=self.handle_restore_keys
        )
        restore_btn.grid(row=0, column=2, padx=5, pady=5)

    # Updated handle_mfa_setup method for BlockchainGUI class
    def handle_mfa_setup(self):
        """Handle MFA setup button click for the selected wallet"""
        try:
            async def do_mfa_setup():
                try:
                    if not self.mfa_manager:
                        return
                    
                    wallet_address = self.selected_wallet.get()
                    if wallet_address == "Select Wallet" or not wallet_address:
                        self.update_queue.put(lambda: self.show_error("Please select a wallet first"))
                        return
                    
                    # Generate MFA secret for the wallet
                    secret = await self.mfa_manager.generate_mfa_secret(wallet_address)
                    logger.info(f"Generated MFA secret for wallet {wallet_address}")
                    
                    # Generate QR code for the wallet
                    qr_code = await self.mfa_manager.get_mfa_qr(
                        wallet_address,
                        f"OriginalCoin-Wallet-{wallet_address[:10]}"
                    )
                    logger.info(f"Generated QR code for wallet {wallet_address}")
                    
                    self.update_queue.put(lambda: self.show_qr_code(qr_code))
                except Exception as e:
                    logger.error(f"MFA setup error: {e}")
                    self.update_queue.put(lambda: messagebox.showerror("MFA Setup Error", str(e)))
            
            asyncio.run_coroutine_threadsafe(do_mfa_setup(), self.loop)
        except Exception as e:
            logger.error(f"Error starting MFA setup: {e}")
            self.show_error(f"Failed to start MFA setup: {e}")

    def handle_backup_keys(self):
        """Handle backup keys button click"""
        asyncio.run_coroutine_threadsafe(self.backup_keys(), self.loop)

    def handle_restore_keys(self):
        """Handle restore keys button click"""
        asyncio.run_coroutine_threadsafe(self.restore_keys(), self.loop)

    def on_closing(self):
        """Handle window closing event with robust cleanup"""
        if self._shutdown_in_progress:
            logger.info("Shutdown already in progress, skipping duplicate call")
            return
        self._shutdown_in_progress = True
        self.status_var.set("Shutting down...")
        logger.info("Initiating GUI shutdown")

        if self.mining:
            future = asyncio.run_coroutine_threadsafe(self.blockchain.stop_mining(), self.loop)
            future.result(timeout=10)

        if self.network:
            future = asyncio.run_coroutine_threadsafe(self.network.stop(), self.loop)
            future.result(timeout=15)
            logger.info("Network stopped")

        if hasattr(self.blockchain, 'shutdown'):
            future = asyncio.run_coroutine_threadsafe(self.blockchain.shutdown(), self.loop)
            future.result(timeout=10)

        async def shutdown_loop():
            pending = asyncio.all_tasks(self.loop)
            logger.info(f"Found {len(pending)} pending tasks")
            for task in pending:
                coro_name = task.get_coro().__qualname__
                logger.debug(f"Pending task: {coro_name}, done: {task.done()}")

            pending = asyncio.all_tasks(self.loop)
            for task in pending:
                if not task.done():
                    coro_name = task.get_coro().__qualname__
                    logger.debug(f"Cancelling task: {coro_name}")
                    task.cancel()

            if pending:
                done, still_pending = await asyncio.wait(pending, timeout=15)
                if still_pending:
                    logger.warning(f"{len(still_pending)} tasks still pending: {[t.get_coro().__qualname__ for t in still_pending]}")

        future = asyncio.run_coroutine_threadsafe(shutdown_loop(), self.loop)
        try:
            future.result(timeout=25)
        except (concurrent.futures.TimeoutError, concurrent.futures.CancelledError):
            logger.warning("Asyncio shutdown timed out or cancelled")

        self.root.quit()
        self.root.update()
        self.root.destroy()
        logger.info("GUI shutdown completed successfully")

    def run(self):
        """Run the GUI with proper initialization"""
        async def initialize_and_run():
            try:
                # Ensure blockchain is fully initialized
                await self.blockchain.initialize()
                logger.info("Blockchain fully initialized before GUI start")
                
                # Update wallet dropdown
                self.update_queue.put(lambda: self.update_wallet_dropdown())
                
                # Give time for wallet dropdown to populate
                await asyncio.sleep(0.5)
                
                # Get the first wallet address and update its balance
                addresses = await self.blockchain.get_all_addresses()
                if addresses:
                    first_address = addresses[0]
                    logger.info(f"Setting initial wallet to: {first_address}")
                    self.update_queue.put(lambda: self.selected_wallet.set(first_address))
                    await self.update_async_balance(first_address)
                    await self.update_async_transaction_history(first_address)
                    logger.info(f"Updated balance for wallet: {first_address}")
            except Exception as e:
                logger.error(f"Error during initialization: {e}", exc_info=True)
                self.update_queue.put(lambda: self.show_error(f"Initialization failed: {e}"))

        # Run initialization in the event loop
        asyncio.run_coroutine_threadsafe(initialize_and_run(), self.loop)
        
        # Wait a moment to let initialization tasks start
        time.sleep(1)
        
        # Start the mainloop
        self.root.mainloop()


async def initialize_security(node_id: str) -> tuple:
    """Initialize security components"""
    security_monitor = SecurityMonitor()
    mfa_manager = MFAManager()
    backup_manager = KeyBackupManager(
        backup_dir=os.path.join('data', 'key_backups')
    )
    
    # Start security monitoring
    asyncio.create_task(security_monitor.analyze_patterns())
    
    return security_monitor, mfa_manager, backup_manager