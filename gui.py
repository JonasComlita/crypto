import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog, simpledialog
import tkinter.simpledialog
import json
import re
import time
from queue import Queue
from typing import Dict, List, Optional
from blockchain import Blockchain, Block, Transaction, TransactionType
from network import BlockchainNetwork
from utils import SecurityUtils, generate_wallet, derive_key, Fernet
import logging
import asyncio
from security import SecurityMonitor, MFAManager, KeyBackupManager
import os
from PIL import Image, ImageTk
import queue
import threading
import datetime
import gc
import concurrent.futures

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BlockchainGUI:
    from blockchain import Blockchain
    def __init__(self, blockchain, network, mfa_manager=None, backup_manager=None):
        self.blockchain = blockchain
        self.network = network
        self.mfa_manager = mfa_manager
        self.backup_manager = backup_manager
        self.mining = False
        self.loop = asyncio.new_event_loop()
        self.loop_thread = threading.Thread(target=self.main_loop, daemon=True)
        self.shutdown_event = threading.Event()
        
        self.root = tk.Tk()
        self.root.title("OriginalCoin Blockchain")
        self.root.geometry("800x600")  # Set initial window size

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
        
        # Start the asyncio loop in a separate thread
        self.loop_thread.start()
        logger.info("Event loop thread started")

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
            width=20
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

    def process_queue(self):
        """Process the update queue for thread-safe UI updates"""
        try:
            while not self.update_queue.empty():
                func = self.update_queue.get_nowait()
                func()
        except Exception as e:
            logger.error(f"Queue processing error: {e}")
        self.root.after(100, self.process_queue)

    def start_mining_callback(self):
        """Wrapper for starting mining with error handling"""
        try:
            logger.info("Starting mining callback triggered")
            wallet_address = self.selected_wallet.get()
            if not wallet_address or wallet_address == "Select Wallet":
                # Run get_all_addresses async through the loop
                async def get_addresses():
                    addresses = await self.blockchain.get_all_addresses()
                    if not addresses:
                        self.update_queue.put(lambda: self.show_error("No wallet available for mining"))
                        return None
                    return addresses[0]
                
                # Get the result and continue with mining
                future = asyncio.run_coroutine_threadsafe(get_addresses(), self.loop)
                wallet_address = future.result(timeout=5)
                if not wallet_address:
                    return
                logger.info(f"Selected wallet address: {wallet_address}")

            # Define the async mining start coroutine
            async def start_mining_coroutine():
                try:
                    result = await self.blockchain.start_mining(wallet_address)
                    if result:
                        self.update_queue.put(self.mining_started)
                    else:
                        self.update_queue.put(lambda: self.show_error("Failed to start mining"))
                except Exception as e:
                    logger.error(f"Mining coroutine exception: {str(e)}")
                    self.update_queue.put(lambda: self.show_error(f"Mining start failed: {str(e)}"))

            # Run the coroutine in the event loop thread-safely
            asyncio.run_coroutine_threadsafe(start_mining_coroutine(), self.loop)

        except Exception as e:
            logger.error(f"Mining callback exception: {str(e)}")
            self.show_error(f"Mining initialization error: {str(e)}")
            
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
                blocks_mined = len(self.blockchain.chain)

                self.update_queue.put(lambda: self.hashrate_var.set(f"Hashrate: {hashrate:.2f} H/s"))
                self.update_queue.put(lambda: self.blocks_mined_var.set(f"Blocks Mined: {blocks_mined}"))
                
                # Update recent blocks
                self.update_queue.put(lambda: self.update_blocks_tree(self.blockchain.chain[-6:]))
                logger.info(f"Updated stats - Hashrate: {hashrate:.2f} H/s, Blocks: {blocks_mined}")
            except Exception as e:
                logger.error(f"Stats update failed: {e}")
                self.update_queue.put(lambda: self.status_var.set(f"Stats update error: {str(e)}"))
        
        def schedule_next_update():
            if self.mining:
                # Schedule the next update
                asyncio.run_coroutine_threadsafe(update_stats(), self.loop)
                self.root.after(5000, schedule_next_update)

        # Start the update cycle
        asyncio.run_coroutine_threadsafe(update_stats(), self.loop)
        self.root.after(5000, schedule_next_update)

    def update_blocks_tree(self, blocks):
        """Update the blocks tree with the given blocks"""
        self.blocks_tree.delete(*self.blocks_tree.get_children())
        for block in reversed(blocks):
            self.blocks_tree.insert(
                "", "end",
                values=(block.index, block.timestamp, len(block.transactions), block.hash[:20] + "...")
            )

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

    def update_wallet_dropdown(self):
        """Update the wallet dropdown with available addresses"""
        try:
            # Define the async address fetch coroutine
            async def fetch_addresses_coroutine():
                try:
                    addresses = await self.blockchain.get_all_addresses()
                    self.update_queue.put(lambda: self.populate_wallet_dropdown(addresses))
                except Exception as e:
                    logger.error(f"Address fetch exception: {str(e)}")
                    self.update_queue.put(lambda: self.show_error(f"Failed to get wallet addresses: {str(e)}"))

            # Run the coroutine in the event loop thread-safely
            asyncio.run_coroutine_threadsafe(fetch_addresses_coroutine(), self.loop)
        except Exception as e:
            self.show_error(f"Wallet dropdown update error: {str(e)}")

    def populate_wallet_dropdown(self, addresses):
        """Populate the wallet dropdown with the given addresses"""
        menu = self.wallet_dropdown["menu"]
        menu.delete(0, "end")
        
        for address in addresses:
            menu.add_command(
                label=address[:10] + "...",  # Show shortened address
                command=lambda a=address: self.on_wallet_selected(a)
            )

    def on_wallet_selected(self, address):
        """Handle wallet selection with MFA verification"""
        try:
            if self.mfa_manager and not self.verify_mfa():
                messagebox.showerror("Error", "MFA verification failed")
                return
                
            self.selected_wallet.set(address)
            
            # Update balance and transaction history asynchronously
            async def update_wallet_info():
                await self.update_async_balance(address)
                await self.update_async_transaction_history(address)
            
            asyncio.run_coroutine_threadsafe(update_wallet_info(), self.loop)
            
        except Exception as e:
            messagebox.showerror("Error", f"Error selecting wallet: {str(e)}")

    def verify_mfa(self):
        """Verify MFA code if enabled"""
        if not self.mfa_manager:
            return True
            
        code = simpledialog.askstring("MFA Required", "Enter MFA code:", show='*')
        if not code:
            return False
            
        return self.mfa_manager.verify_code(self.network.node_id, code)

    async def update_async_balance(self, address):
        """Update the balance display asynchronously"""
        try:
            balance = await self.blockchain.get_balance(address)
            
            # Add security indicator if wallet is backed up
            backup_status = ""
            if self.backup_manager:
                is_backed_up = await self.backup_manager.is_wallet_backed_up(address)
                if is_backed_up:
                    backup_status = "✓"
                    
            balance_text = f"Balance: {balance:.2f} ORIG {backup_status}"
            self.update_queue.put(lambda: self.balance_label.config(text=balance_text))
            
        except Exception as e:
            logger.error(f"Balance update error: {str(e)}")
            self.update_queue.put(lambda: self.show_error(f"Failed to update balance: {str(e)}"))

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

    def send_transaction(self):
        """Send a transaction with security checks"""
        try:
            # Verify MFA before transaction
            if self.mfa_manager and not self.verify_mfa():
                messagebox.showerror("Error", "MFA verification required")
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

    def handle_mfa_setup(self):
        """Handle MFA setup button click"""
        asyncio.run_coroutine_threadsafe(self.setup_mfa(), self.loop)

    def handle_backup_keys(self):
        """Handle backup keys button click"""
        asyncio.run_coroutine_threadsafe(self.backup_keys(), self.loop)

    def handle_restore_keys(self):
        """Handle restore keys button click"""
        asyncio.run_coroutine_threadsafe(self.restore_keys(), self.loop)

    def on_closing(self):
        """Improved application shutdown handler for GUI window close events"""
        logger.info("Starting application shutdown...")
        try:
            # Display shutdown message
            self.status_var.set("Shutting down...")
            
            # Flag to prevent multiple shutdown attempts
            if hasattr(self, '_shutdown_in_progress') and self._shutdown_in_progress:
                logger.info("Shutdown already in progress")
                return
            self._shutdown_in_progress = True
            
            # Define cleanup coroutine
            async def cleanup():
                try:
                    # Stop mining if active
                    if self.mining:
                        await self.blockchain.stop_mining()
                        logger.info("Mining stopped during shutdown")
                        
                    # Network cleanup
                    if self.network:
                        await self.network.cleanup()
                        logger.info("Network cleanup completed")
                        
                    # Blockchain shutdown
                    if hasattr(self.blockchain, 'shutdown'):
                        await self.blockchain.shutdown()
                        logger.info("Blockchain shutdown completed")
                except Exception as e:
                    logger.error(f"Cleanup error: {e}")
                finally:
                    # Signal event loop to stop
                    self.shutdown_event.set()
                    
                    # Only stop the loop if it's still running
                    try:
                        if not self.loop.is_closed():
                            self.loop.stop()
                    except Exception as e:
                        logger.error(f"Error stopping loop: {e}")
            
            # Create a future to track cleanup progress
            cleanup_future = asyncio.run_coroutine_threadsafe(cleanup(), self.loop)
            
            # Schedule GUI exit after a short delay
            self.root.after(500, self._delayed_exit, cleanup_future)
            
        except Exception as e:
            logger.error(f"Error starting shutdown: {e}")
            # Force destroy as a last resort
            try:
                self.root.destroy()
            except:
                pass

    def _delayed_exit(self, cleanup_future):
        """Handle delayed GUI exit after cleanup starts"""
        try:
            # Check if cleanup is done or wait a short time
            try:
                cleanup_future.result(timeout=3)
                logger.info("Async cleanup completed successfully")
            except concurrent.futures.TimeoutError:
                logger.warning("Async cleanup timed out, proceeding with GUI exit")
            except Exception as e:
                logger.error(f"Error in async cleanup: {e}")
            
            # Destroy the GUI
            self.root.quit()
            self.root.destroy()
            logger.info("GUI destroyed")
            
        except Exception as e:
            logger.error(f"Error during GUI exit: {e}")
            # Force exit as a last resort
            try:
                import os
                os._exit(1)
            except:
                pass

    def exit(self):
        """Add an explicit exit method for external shutdown calls"""
        if not hasattr(self, '_shutdown_in_progress') or not self._shutdown_in_progress:
            self.on_closing()
        else:
            logger.info("Exit called while shutdown in progress")

    def cleanup_resources(self):
        """Periodic cleanup of resources"""
        # Clear old cache entries
        current_time = time.time()
        self.cache = {k: v for k, v in self.cache.items() 
                      if current_time - v['timestamp'] < self.cache_timeout}
        
        # Clear sensitive data from memory
        if hasattr(self, 'clear_clipboard'):
            self.clear_clipboard()
            
        # Force garbage collection
        gc.collect()

    def clear_clipboard(self):
        """Clear sensitive data from clipboard"""
        try:
            self.root.clipboard_clear()
            self.root.clipboard_append("")
        except Exception as e:
            logger.error(f"Failed to clear clipboard: {e}")

    def run(self):
        """Run the blockchain GUI"""
        # Initial data loading
        self.update_wallet_dropdown()
        
        # Start UI main loop
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