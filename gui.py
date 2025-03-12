import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog, simpledialog
import tkinter.simpledialog
import json
import re
import time
from queue import Queue
from typing import Dict, List
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

        self.loop_thread.start()
        logger.info("Event loop thread started")

    def process_queue(self):
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
                addresses = self.blockchain.get_all_addresses()
                if not addresses:
                    logger.error("No wallet available for mining")
                    self.show_error("No wallet available for mining")
                    return
                wallet_address = addresses[0]
                logger.info(f"Selected wallet address: {wallet_address}")

            # Define the async mining start coroutine
            async def start_mining_coroutine():
                try:
                    # Pass the event loop to the miner
                    self.blockchain.miner.loop = self.loop
                    # Start mining asynchronously
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
            messagebox.showerror("Mining Error", message)
        else:
            self.root.after(0, lambda: messagebox.showerror("Mining Error", message))

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

    def handle_mining_result(self, future):
        """Handle the result of the mining task"""
        try:
            # This will either raise an exception or return a boolean
            result = future.result()
            
            # Use root.after to update GUI thread safely
            if result:
                self.root.after(0, self.mining_started)
            else:
                self.root.after(0, lambda: self.mining_stopped("Unknown error"))
        except Exception as e:
            self.root.after(0, lambda: self.mining_stopped(str(e)))

    def start_mining_stats_update(self):
        """Periodically update mining stats while mining is active"""
        async def update_stats():
            if not self.mining:
                return
            try:
                hashrate = await self.blockchain.get_hashrate()  # Ensure this method exists
                logger.info(f"Updating stats - Hashrate: {hashrate}")
                blocks_mined = len(self.blockchain.chain)

                self.root.after(0, lambda: self.hashrate_var.set(f"Hashrate: {hashrate:.2f} H/s"))
                self.root.after(0, lambda: self.blocks_mined_var.set(f"Blocks Mined: {blocks_mined}"))
                
                self.root.after(0, lambda: self.blocks_tree.delete(*self.blocks_tree.get_children()))

                # Update recent blocks
                for block in reversed(self.blockchain.chain[-6:]):
                    self.root.after(0, lambda b=block: self.blocks_tree.insert(
                        "", "end",
                        values=(b.index, b.timestamp, len(b.transactions), b.hash[:20] + "...")
                    ))
                logger.info(f"Updated stats - Hashrate: {hashrate:.2f} H/s, Blocks: {blocks_mined}")
            except Exception as e:
                logger.error(f"Stats update failed: {e}")
                self.root.after(0, lambda: self.status_var.set(f"Stats update error: {str(e)}"))
        
        def schedule_next_update():
            if self.mining:
                # Schedule the next update
                asyncio.run_coroutine_threadsafe(update_stats(), self.loop)
                self.root.after(5000, schedule_next_update)

        # Start the update cycle
        asyncio.run_coroutine_threadsafe(update_stats(), self.loop)
        self.root.after(5000, schedule_next_update)

    def handle_mining_stats_update(self, future):
        """Handle mining stats update results"""
        try:
            future.result()  # This will raise any exceptions
        except Exception as e:
            logger.error(f"Mining stats update error: {e}")
            # Optionally stop mining or reset UI

    async def update_mining_stats(self):
        hashrate = await self.blockchain.get_hashrate()
        self.hashrate_var.set(f"Hashrate: {hashrate:.2f} H/s")
        self.blocks_mined_var.set(f"Blocks Mined: {len(self.blockchain.chain)}")
        """Update mining statistics display"""
        if not hasattr(self, 'hashrate_var'):
            return
            
        hashrate = self.blockchain.get_hashrate()
        self.hashrate_var.set(f"Hashrate: {hashrate:.2f} H/s")
        self.blocks_mined_var.set(f"Blocks Mined: {len(self.blockchain.chain)}")
        
        # Update recent blocks display
        self.blocks_tree.delete(*self.blocks_tree.get_children())
        for block in reversed(self.blockchain.chain[-6:]):
            self.blocks_tree.insert(
                "",
                0,
                values=(
                    block.index,
                    block.timestamp,
                    len(block.transactions),
                    block.hash[:20] + "..."
                )
            )

    def create_new_wallet(self):
        """Create a new wallet"""
        try:
            address = self.blockchain.create_wallet()
            self.update_wallet_dropdown()
            self.selected_wallet.set(address)
            self.status_var.set(f"Created new wallet: {address[:10]}...")
        except Exception as e:
            self.status_var.set(f"Error creating wallet: {str(e)}")

    def update_wallet_dropdown(self):
        """Update the wallet dropdown with available addresses"""
        try:
            addresses = self.blockchain.get_all_addresses()
            
            menu = self.wallet_dropdown["menu"]
            menu.delete(0, "end")
            
            for address in addresses:
                menu.add_command(
                    label=address[:10] + "...",  # Show shortened address
                    command=lambda a=address: self.on_wallet_selected(a)
                )
        except Exception as e:
            messagebox.showerror("Error", f"Failed to update wallet list: {str(e)}")

    def on_wallet_selected(self, address):
        """Handle wallet selection with MFA verification"""
        try:
            if self.mfa_manager and not self.verify_mfa():
                messagebox.showerror("Error", "MFA verification failed")
                return
                
            self.selected_wallet.set(address)
            self.update_balance()
            self.update_transaction_history()
            
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

    def update_balance(self):
        """Update the balance display with security checks"""
        try:
            address = self.selected_wallet.get()
            balance = 0
            if address and address != "Select Wallet":
                balance = self.blockchain.get_balance(address)
                # Add security indicator if wallet is backed up
                backup_status = "✓" if self.backup_manager and \
                    self.backup_manager.is_wallet_backed_up(address) else ""
                self.balance_label.config(
                    text=f"Balance: {balance:.2f} ORIG {backup_status}"
                )
        except Exception as e:
            messagebox.showerror("Error", f"Failed to update balance: {str(e)}")

    def send_transaction(self):
        """Send a transaction with security checks"""
        try:
            # Verify MFA before transaction
            if self.mfa_manager and not self.verify_mfa():
                messagebox.showerror("Error", "MFA verification required")
                return
                
            sender = self.selected_wallet.get()
            recipient = self.recipient_var.get()
            amount = float(self.amount_var.get())
            
            if sender and recipient and amount > 0:
                # Create and send transaction
                tx = self.blockchain.create_transaction(sender, recipient, amount)
                
                # Backup keys after transaction if enabled
                if self.backup_manager:
                    asyncio.run(self.backup_manager.backup_transaction(tx))
                
                messagebox.showinfo("Success", "Transaction sent successfully")
                
                # Clear inputs and update display
                self.amount_var.set("")
                self.recipient_var.set("")
                self.update_balance()
                self.update_transaction_history()
            else:
                messagebox.showerror("Error", "Please fill all fields correctly")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to send transaction: {str(e)}")

    def update_transaction_history(self):
        """Update the transaction history display"""
        try:
            # Add pagination
            # Implement memory-efficient data loading
            # Add cleanup for old data
            
            # Clear existing items
            for item in self.history_tree.get_children():
                self.history_tree.delete(item)
            
            # Get transactions for selected wallet
            address = self.selected_wallet.get()
            if address and address != "Select Wallet":
                transactions = self.blockchain.get_transactions_for_address(address)
                
                # Add transactions to tree
                for tx in transactions:
                    self.history_tree.insert(
                        "",
                        "end",
                        values=(
                            tx.timestamp,
                            tx.sender[:10] + "...",
                            tx.recipient[:10] + "...",
                            tx.amount
                        )
                    )
        except Exception as e:
            messagebox.showerror("Error", f"Failed to update transaction history: {str(e)}")

    async def update_gui(self):
        """Process any pending GUI updates and schedule next update"""
        try:
            # Handle all pending updates
            while True:
                try:
                    # Non-blocking queue check
                    update_func = self.update_queue.get_nowait()
                    update_func()
                except queue.Empty:
                    break
                    
            # Update GUI elements
            await self.update_balance()
            await self.update_transaction_history()
            await self.update_mining_stats()
            await self.update_chain_info()
            await self.update_mempool_info()
            
        except Exception as e:
            logger.error(f"Error in GUI update: {e}")
            
        finally:
            # Schedule next update
            self.root.after(1000, lambda: asyncio.run_coroutine_threadsafe(self.update_gui(), asyncio.get_event_loop()))

    def safe_update(self, func):
        """Thread-safe way to schedule GUI updates"""
        self.update_queue.put(func)

    def exit(self):
        try:
            asyncio.run_coroutine_threadsafe(self.miner.stop_mining(), self.network.loop)
        except Exception as e:
            logger.error(f"Error stopping miner during exit: {e}")
        self.root.quit()

    def run(self):
        self.update_wallet_dropdown()
        self.root.mainloop()
    
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
            self.show_qr_code(qr_code)
            
        except Exception as e:
            messagebox.showerror("MFA Setup Error", str(e))

    async def backup_keys(self):
        """Handle key backup"""
        if not self.backup_manager:
            return
            
        try:
            password = self.get_backup_password()
            if not password:
                return
                
            await self.backup_manager.create_backup(
                self.network.get_keys(),
                password
            )
            messagebox.showinfo("Success", "Keys backed up successfully")
            
        except Exception as e:
            messagebox.showerror("Backup Error", str(e))

    async def restore_keys(self):
        """Handle key restoration"""
        if not self.backup_manager:
            return
            
        try:
            password = self.get_backup_password()
            if not password:
                return
                
            file_path = filedialog.askopenfilename(
                title="Select Backup File",
                filetypes=[("Encrypted Backup", "*.enc")]
            )
            
            if file_path:
                keys = await self.backup_manager.restore_backup(file_path, password)
                await self.network.restore_keys(keys)
                messagebox.showinfo("Success", "Keys restored successfully")
                
        except Exception as e:
            messagebox.showerror("Restore Error", str(e))

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

    def create_blockchain_tab(self):
        """Create the blockchain information tab"""
        blockchain_tab = ttk.Frame(self.notebook)
        self.notebook.add(blockchain_tab, text="Blockchain")
        
        # Blockchain Info Frame
        info_frame = ttk.LabelFrame(blockchain_tab, text="Blockchain Info")
        info_frame.pack(fill="x", padx=5, pady=5)
        
        # Add blockchain information
        self.chain_length = tk.StringVar(value="Chain Length: 0")
        ttk.Label(info_frame, textvariable=self.chain_length).pack(pady=5)
        
        self.last_block = tk.StringVar(value="Last Block: None")
        ttk.Label(info_frame, textvariable=self.last_block).pack(pady=5)
        
        # Add refresh button
        ttk.Button(
            info_frame,
            text="Refresh",
            command=self.refresh_blockchain_info
        ).pack(pady=5)

    def create_network_tab(self):
        """Create the network information tab"""
        network_tab = ttk.Frame(self.notebook)
        self.notebook.add(network_tab, text="Network")
        
        # Network Info Frame
        info_frame = ttk.LabelFrame(network_tab, text="Network Info")
        info_frame.pack(fill="x", padx=5, pady=5)
        
        # Add network information
        self.node_id = tk.StringVar(value=f"Node ID: {self.network.node_id}")
        ttk.Label(info_frame, textvariable=self.node_id).pack(pady=5)
        
        self.peer_count = tk.StringVar(value="Connected Peers: 0")
        ttk.Label(info_frame, textvariable=self.peer_count).pack(pady=5)
        
        # Add refresh button
        ttk.Button(
            info_frame,
            text="Refresh",
            command=self.refresh_network_info
        ).pack(pady=5)

    def create_transaction_tab(self):
        """Create the transaction management tab"""
        transaction_tab = ttk.Frame(self.notebook)
        self.notebook.add(transaction_tab, text="Transactions")
        
        # Transaction Creation Frame
        create_frame = ttk.LabelFrame(transaction_tab, text="Create Transaction")
        create_frame.pack(fill="x", padx=5, pady=5)
        
        # Add transaction form
        ttk.Label(create_frame, text="Recipient:").pack(pady=2)
        self.recipient_entry = ttk.Entry(create_frame)
        self.recipient_entry.pack(fill="x", padx=5, pady=2)
        
        ttk.Label(create_frame, text="Amount:").pack(pady=2)
        self.amount_entry = ttk.Entry(create_frame)
        self.amount_entry.pack(fill="x", padx=5, pady=2)
        
        ttk.Button(
            create_frame,
            text="Send Transaction",
            command=self.send_transaction
        ).pack(pady=5)
        
        # Transaction History Frame
        history_frame = ttk.LabelFrame(transaction_tab, text="Transaction History")
        history_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Add transaction history list
        self.transaction_list = tk.Listbox(history_frame)
        self.transaction_list.pack(fill="both", expand=True, padx=5, pady=5)

    def refresh_blockchain_info(self):
        """Update blockchain information display"""
        self.chain_length.set(f"Chain Length: {len(self.blockchain.chain)}")
        if self.blockchain.chain:
            last_block = self.blockchain.chain[-1]
            self.last_block.set(f"Last Block: {last_block.hash[:10]}...")

    def refresh_network_info(self):
        """Update network information display"""
        self.peer_count.set(f"Connected Peers: {len(self.network.peers)}")

    def add_security_controls(self):
        """Add security controls to the interface"""
        security_frame = ttk.LabelFrame(self.main_frame, text="Security Controls", padding="5")
        security_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        # MFA Setup button
        ttk.Button(
            security_frame,
            text="Setup MFA",
            command=lambda: asyncio.run(self.setup_mfa())
        ).grid(row=0, column=0, padx=5, pady=5)
        
        # Backup Controls
        ttk.Button(
            security_frame,
            text="Backup Keys",
            command=lambda: asyncio.run(self.backup_keys())
        ).grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Button(
            security_frame,
            text="Restore Keys",
            command=lambda: asyncio.run(self.restore_keys())
        ).grid(row=0, column=2, padx=5, pady=5)

    def on_closing(self):
        logger.info("Starting application shutdown...")
        try:
            if self.mining:
                asyncio.run_coroutine_threadsafe(self.blockchain.stop_mining(), self.loop).result(timeout=10)
                logger.info("Mining stopped during shutdown")
            if self.network:
                asyncio.run_coroutine_threadsafe(self.network.cleanup(), self.loop).result(timeout=5)
                logger.info("Network cleanup completed")
            self.shutdown_event.set()
            self.loop.call_soon_threadsafe(self.loop.stop)
            self.root.quit()
            self.root.destroy()
            logger.info("GUI destroyed")
            self.loop_thread.join(timeout=10)
            if self.loop_thread.is_alive():
                logger.warning("Event loop thread did not terminate cleanly")
                import psutil
                current_process = psutil.Process()
                children = current_process.children(recursive=True)
                for child in children:
                    logger.warning(f"Terminating child process: {child.pid}")
                    child.terminate()
            else:
                logger.info("Event loop thread terminated")
        except Exception as e:
            logger.error(f"Shutdown error: {e}")

    def cleanup_resources(self):
        """Periodic cleanup of resources"""
        # Clear old cache entries
        current_time = time.time()
        self.cache = {k: v for k, v in self.cache.items() 
                      if current_time - v['timestamp'] < self.cache_timeout}
        
        # Clear sensitive data
        self.clear_clipboard()
        gc.collect()

    def setup_monitoring(self):
        """Setup performance monitoring"""
        self.metrics = {
            'gui_updates': 0,
            'network_requests': 0,
            'transaction_count': 0,
            'response_times': []
        }
        
        # Add Prometheus metrics
        self.prometheus_client.start_http_server(8000)

async def initialize_security(node_id: str) -> tuple:
    security_monitor = SecurityMonitor()
    mfa_manager = MFAManager()
    backup_manager = KeyBackupManager(
        backup_dir=os.path.join('data', 'key_backups')
    )
    
    # Start security monitoring
    asyncio.create_task(security_monitor.analyze_patterns())
    
    return security_monitor, mfa_manager, backup_manager