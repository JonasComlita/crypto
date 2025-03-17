import tkinter as tk
from tkinter import ttk, messagebox, filedialog, simpledialog
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
from PIL import ImageTk
import queue
import threading
import concurrent.futures

logger = logging.getLogger(__name__)

class BlockchainGUI:
    def __init__(self, blockchain: Blockchain, network: BlockchainNetwork, mfa_manager=None, backup_manager=None):
        self.blockchain = blockchain
        self.network = network
        self.mfa_manager = mfa_manager
        self.backup_manager = backup_manager
        self.mining = False
        self._shutdown_in_progress = False
        self.loop = None  # Set by main.py
        self.loop_thread = None  # Set by main.py

        self.root = tk.Tk()
        self.root.title("OriginalCoin Blockchain")
        self.root.geometry("600x600")

        self.update_queue = queue.Queue()
        self.root.after(100, self.process_queue)

        # Style configuration
        self.style = ttk.Style()
        self.style.configure('Mining.TButton', background='green')
        self.style.configure('Header.TLabel', font=('Helvetica', 12, 'bold'))
        self.style.configure('Info.TLabel', font=('Helvetica', 10))

        self.selected_wallet = tk.StringVar(value="Select Wallet")
        self.amount_var = tk.StringVar()
        self.recipient_var = tk.StringVar()

        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        self.notebook = ttk.Notebook(self.main_frame)
        self.notebook.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        self.init_wallet_tab()
        self.init_mining_tab()
        self.init_network_tab()

        self.mining_btn = ttk.Button(
            self.mining_frame.winfo_children()[0],
            text="Start Mining",
            command=self.toggle_mining,
            style='Mining.TButton'
        )
        self.mining_btn.grid(row=0, column=0, padx=5, pady=5)

        self.status_var = tk.StringVar(value="Initializing...")
        self.status_bar = ttk.Label(
            self.root,
            textvariable=self.status_var,
            relief=tk.SUNKEN,
            padding=(5, 2)
        )
        self.status_bar.grid(row=1, column=0, sticky=(tk.W, tk.E))

        self.add_security_controls()

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Subscribe to blockchain events
        self.blockchain.subscribe("new_block", self.on_new_block)
        self.blockchain.subscribe("error", self.on_error)
        self.blockchain.subscribe("key_rotated", self.on_key_rotated)
        
    def init_wallet_tab(self):
        wallet_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(wallet_frame, text="Wallet")

        select_frame = ttk.LabelFrame(wallet_frame, text="Wallet Selection", padding="5")
        select_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)

        self.wallet_dropdown = ttk.OptionMenu(select_frame, self.selected_wallet, "Select Wallet", *[])
        self.wallet_dropdown.grid(row=0, column=0, padx=5, pady=5)

        refresh_btn = ttk.Button(select_frame, text="↻ Refresh", command=self.update_wallet_dropdown)
        refresh_btn.grid(row=0, column=1, padx=5, pady=5)

        new_wallet_btn = ttk.Button(select_frame, text="+ New Wallet", command=self.create_new_wallet)
        new_wallet_btn.grid(row=0, column=2, padx=5, pady=5)

        change_pass_btn = ttk.Button(select_frame, text="Change Password", command=self.handle_change_password)
        change_pass_btn.grid(row=0, column=3, padx=5, pady=5)

        self.balance_label = ttk.Label(wallet_frame, text="Balance: 0 ORIG", style='Header.TLabel')
        self.balance_label.grid(row=1, column=0, pady=10)

        tx_frame = ttk.LabelFrame(wallet_frame, text="Send Transaction", padding="5")
        tx_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)

        ttk.Label(tx_frame, text="Amount:").grid(row=0, column=0, padx=5, pady=5)
        amount_entry = ttk.Entry(tx_frame, textvariable=self.amount_var, width=40)
        amount_entry.grid(row=0, column=1, padx=5, pady=5)

        ttk.Label(tx_frame, text="Recipient:").grid(row=1, column=0, padx=5, pady=5)
        recipient_entry = ttk.Entry(tx_frame, textvariable=self.recipient_var, width=40)
        recipient_entry.grid(row=1, column=1, padx=5, pady=5)

        send_btn = ttk.Button(tx_frame, text="Send", command=self.send_transaction)
        send_btn.grid(row=2, column=0, columnspan=2, pady=10)

        history_frame = ttk.LabelFrame(wallet_frame, text="Transaction History", padding="5")
        history_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)

        self.history_tree = ttk.Treeview(history_frame, columns=("timestamp", "from", "to", "amount"), show="headings", height=6)
        self.history_tree.heading("timestamp", text="Time")
        self.history_tree.heading("from", text="From")
        self.history_tree.heading("to", text="To")
        self.history_tree.heading("amount", text="Amount")
        self.history_tree.column("timestamp", width=150)
        self.history_tree.column("from", width=150)
        self.history_tree.column("to", width=150)
        self.history_tree.column("amount", width=100)

        scrollbar = ttk.Scrollbar(history_frame, orient=tk.VERTICAL, command=self.history_tree.yview)
        self.history_tree.configure(yscrollcommand=scrollbar.set)
        self.history_tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))

    def init_mining_tab(self):
        self.mining_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(self.mining_frame, text="Mining")

        controls_frame = ttk.LabelFrame(self.mining_frame, text="Mining Controls", padding="10")
        controls_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=5)

        stats_frame = ttk.LabelFrame(self.mining_frame, text="Mining Statistics", padding="10")
        stats_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=5)

        self.hashrate_var = tk.StringVar(value="Hashrate: 0 H/s")
        self.blocks_mined_var = tk.StringVar(value="Blocks Mined: 0")
        ttk.Label(stats_frame, textvariable=self.hashrate_var, style='Info.TLabel').grid(row=0, column=0, pady=2)
        ttk.Label(stats_frame, textvariable=self.blocks_mined_var, style='Info.TLabel').grid(row=1, column=0, pady=2)

        blocks_frame = ttk.LabelFrame(self.mining_frame, text="Recent Blocks", padding="10")
        blocks_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)

        self.blocks_tree = ttk.Treeview(blocks_frame, columns=("index", "timestamp", "transactions", "hash"), show="headings", height=6)
        self.blocks_tree.heading("index", text="#")
        self.blocks_tree.heading("timestamp", text="Time")
        self.blocks_tree.heading("transactions", text="Transactions")
        self.blocks_tree.heading("hash", text="Hash")
        self.blocks_tree.column("index", width=50)
        self.blocks_tree.column("timestamp", width=150)
        self.blocks_tree.column("transactions", width=100)
        self.blocks_tree.column("hash", width=200)

        scrollbar = ttk.Scrollbar(blocks_frame, orient=tk.VERTICAL, command=self.blocks_tree.yview)
        self.blocks_tree.configure(yscrollcommand=scrollbar.set)
        self.blocks_tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))

    def init_network_tab(self):
        network_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(network_frame, text="Network")

        info_frame = ttk.LabelFrame(network_frame, text="Node Information", padding="10")
        info_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=5)

        sync_btn = ttk.Button(info_frame, text="Sync Now", command=self.sync_chain)
        sync_btn.grid(row=0, column=0, padx=5, pady=5)

        self.node_id_var = tk.StringVar(value=f"Node ID: {self.network.node_id}")
        self.peers_var = tk.StringVar(value="Connected Peers: 0")
        ttk.Label(info_frame, textvariable=self.node_id_var, style='Info.TLabel').grid(row=1, column=0, pady=2)
        ttk.Label(info_frame, textvariable=self.peers_var, style='Info.TLabel').grid(row=2, column=0, pady=2)

        peers_frame = ttk.LabelFrame(network_frame, text="Connected Peers", padding="10")
        peers_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)

        self.peers_tree = ttk.Treeview(peers_frame, columns=("node_id", "address", "connected_since"), show="headings", height=6)
        self.peers_tree.heading("node_id", text="Node ID")
        self.peers_tree.heading("address", text="Address")
        self.peers_tree.heading("connected_since", text="Connected Since")
        self.peers_tree.column("node_id", width=150)
        self.peers_tree.column("address", width=150)
        self.peers_tree.column("connected_since", width=150)

        scrollbar = ttk.Scrollbar(peers_frame, orient=tk.VERTICAL, command=self.peers_tree.yview)
        self.peers_tree.configure(yscrollcommand=scrollbar.set)
        self.peers_tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))

    def process_queue(self):
        """Process UI update queue"""
        try:
            while not self.update_queue.empty():
                func = self.update_queue.get_nowait()
                func()
        except Exception as e:
            logger.error(f"Queue processing error: {e}")
        if not self._shutdown_in_progress:
            self.root.after(100, self.process_queue)

    async def on_new_block(self, block: Block):
        """Handle new block event"""
        try:
            wallet = self.selected_wallet.get()
            if wallet != "Select Wallet":
                await self.update_async_balance(wallet)
                await self.update_async_transaction_history(wallet)
            self.update_queue.put(lambda: self.update_blocks_tree(self.blockchain.chain[-6:]))
            self.update_queue.put(lambda: self.status_var.set(f"New block: {block.index}"))
        except Exception as e:
            self.on_error(f"Error processing new block: {e}")

    async def on_error(self, error: str):
        """Handle error event"""
        self.update_queue.put(lambda: self.show_error(error))

    def show_error(self, message: str):
        """Display error message"""
        logger.error(message)
        self.status_var.set(message)
        messagebox.showerror("Error", message)

    # New handler for key_rotated event
    async def on_key_rotated(self, data: Dict[str, str]):
        """Handle key rotation event from key_rotation"""
        try:
            key_hash = data.get("new_key_hash", "unknown")
            logger.info(f"Key rotated with hash: {key_hash}")
            self.update_queue.put(lambda: self.status_var.set(f"Key rotated: {key_hash[:10]}..."))
        except Exception as e:
            self.on_error(f"Error handling key rotation: {e}")

    def update_wallet_dropdown(self):
        """Update wallet dropdown"""
        async def fetch_addresses():
            try:
                addresses = await self.blockchain.get_all_addresses()
                self.update_queue.put(lambda: self.populate_wallet_dropdown(addresses))
            except Exception as e:
                self.on_error(f"Failed to fetch wallets: {e}")
        asyncio.run_coroutine_threadsafe(fetch_addresses(), self.loop)

    def populate_wallet_dropdown(self, addresses: List[str]):
        menu = self.wallet_dropdown["menu"]
        menu.delete(0, "end")
        if not addresses:
            menu.add_command(label="Select Wallet", command=lambda: self.selected_wallet.set("Select Wallet"))
            self.selected_wallet.set("Select Wallet")
        else:
            for addr in addresses:
                menu.add_command(label=addr[:10] + "...", command=lambda a=addr: self.on_wallet_selected(a))
            if self.selected_wallet.get() == "Select Wallet" or self.selected_wallet.get() not in addresses:
                self.on_wallet_selected(addresses[0])

    def on_wallet_selected(self, address: str):
        async def update():
            try:
                if self.mfa_manager and not await self.verify_mfa():
                    self.on_error("MFA verification failed")
                    return
                self.selected_wallet.set(address)
                await self.update_async_balance(address)
                await self.update_async_transaction_history(address)
            except Exception as e:
                self.on_error(f"Error selecting wallet: {e}")
        asyncio.run_coroutine_threadsafe(update(), self.loop)

    async def update_async_balance(self, address: str):
        try:
            balance = await self.blockchain.get_balance(address)
            backup_status = "✓" if self.backup_manager and await self.backup_manager.is_wallet_backed_up(address) else ""
            self.update_queue.put(lambda: self.balance_label.config(text=f"Balance: {balance:.2f} ORIG {backup_status}"))
        except Exception as e:
            self.on_error(f"Failed to update balance: {e}")

    async def update_async_transaction_history(self, address: str):
        try:
            transactions = await self.blockchain.get_transactions_for_address(address)
            def update_tree():
                self.history_tree.delete(*self.history_tree.get_children())
                for tx in transactions:
                    self.history_tree.insert(
                        "", "end",
                        values=(tx.timestamp, tx.sender[:10] + "..." if tx.sender else "Coinbase",
                                tx.recipient[:10] + "..." if tx.recipient else "N/A", tx.amount)
                    )
            self.update_queue.put(update_tree)
        except Exception as e:
            self.on_error(f"Failed to update transaction history: {e}")

    def update_blocks_tree(self, blocks: List[Block]):
        self.blocks_tree.delete(*self.blocks_tree.get_children())  # Corrected from self.blocks_tree
        for block in reversed(blocks):
            self.blocks_tree.insert(
                "", "end",
                values=(block.index, block.timestamp, len(block.transactions), block.hash[:20] + "...")
            )

    def sync_chain(self):
        """Sync chain using network.get_chain()"""
        async def do_sync():
            try:
                self.status_var.set("Syncing with network...")
                chain_data = await self.network.request_chain()  # Assuming network.get_chain() is a typo for request_chain()
                if chain_data:
                    wallet = self.selected_wallet.get()
                    if wallet != "Select Wallet":
                        await self.update_async_balance(wallet)
                        await self.update_async_transaction_history(wallet)
                    self.update_queue.put(lambda: self.status_var.set("Sync completed"))
                else:
                    self.update_queue.put(lambda: self.status_var.set("No updates from sync"))
            except Exception as e:
                self.on_error(f"Sync failed: {e}")
        asyncio.run_coroutine_threadsafe(do_sync(), self.loop)

    def toggle_mining(self):
        if self.mining:
            async def stop():
                try:
                    await self.blockchain.stop_mining()
                    self.update_queue.put(self.mining_stopped)
                except Exception as e:
                    self.on_error(f"Failed to stop mining: {e}")
            asyncio.run_coroutine_threadsafe(stop(), self.loop)
        else:
            async def start():
                try:
                    wallet = self.selected_wallet.get()
                    if wallet == "Select Wallet":
                        addresses = await self.blockchain.get_all_addresses()
                        wallet = addresses[0] if addresses else await self.blockchain.create_wallet()
                        self.selected_wallet.set(wallet)
                    await self.blockchain.start_mining(wallet)
                    self.update_queue.put(self.mining_started)
                except Exception as e:
                    self.on_error(f"Failed to start mining: {e}")
            asyncio.run_coroutine_threadsafe(start(), self.loop)

    def mining_started(self):
        self.mining = True
        self.mining_btn.configure(text="Stop Mining")
        self.status_var.set("Mining started")
        self.start_mining_stats_update()

    def mining_stopped(self):
        self.mining = False
        self.mining_btn.configure(text="Start Mining")
        self.status_var.set("Mining stopped")

    def start_mining_stats_update(self):
        async def update_stats():
            if self.mining and not self._shutdown_in_progress:
                try:
                    hashrate = await self.blockchain.get_hashrate()
                    blocks_mined = self.blockchain.miner.blocks_mined if hasattr(self.blockchain.miner, 'blocks_mined') else 0
                    self.update_queue.put(lambda: self.hashrate_var.set(f"Hashrate: {hashrate:.2f} H/s"))
                    self.update_queue.put(lambda: self.blocks_mined_var.set(f"Blocks Mined: {blocks_mined}"))
                except Exception as e:
                    self.on_error(f"Stats update failed: {e}")
                await asyncio.sleep(1)
                asyncio.create_task(update_stats())
        if self.mining:
            asyncio.run_coroutine_threadsafe(update_stats(), self.loop)

    def create_new_wallet(self):
        async def create():
            try:
                address = await self.blockchain.create_wallet()
                self.update_queue.put(lambda: self.wallet_created(address))
            except Exception as e:
                self.on_error(f"Wallet creation failed: {e}")
        asyncio.run_coroutine_threadsafe(create(), self.loop)

    def wallet_created(self, address: str):
        self.update_wallet_dropdown()
        self.selected_wallet.set(address)
        self.status_var.set(f"Created wallet: {address[:10]}...")

    def handle_change_password(self):
        async def change():
            try:
                if self.mfa_manager and not await self.verify_mfa():
                    self.on_error("MFA verification failed")
                    return
                old_password = simpledialog.askstring("Current Password", "Enter current password:", show='*', parent=self.root)
                if not old_password or not await self.blockchain.key_manager.load_keys(password=old_password):
                    self.on_error("Incorrect current password")
                    return
                new_password = simpledialog.askstring("New Password", "Enter new password:", show='*', parent=self.root)
                confirm = simpledialog.askstring("Confirm Password", "Confirm new password:", show='*', parent=self.root)
                if new_password != confirm:
                    self.on_error("Passwords do not match")
                    return
                await self.blockchain.change_wallet_password(new_password)
                self.update_queue.put(lambda: messagebox.showinfo("Success", "Password changed"))
            except Exception as e:
                self.on_error(f"Password change failed: {e}")
        asyncio.run_coroutine_threadsafe(change(), self.loop)

    async def verify_mfa(self) -> bool:
        if not self.mfa_manager or not await self.mfa_manager.is_mfa_configured(self.selected_wallet.get()):
            return True
        code = await asyncio.get_event_loop().run_in_executor(None, lambda: simpledialog.askstring("MFA", "Enter MFA code:", show='*', parent=self.root))
        return code and self.mfa_manager.verify_mfa(self.selected_wallet.get(), code)

    def send_transaction(self):
        async def send():
            try:
                if self.mfa_manager and not await self.verify_mfa():
                    self.on_error("MFA verification failed")
                    return
                sender = self.selected_wallet.get()
                recipient = self.recipient_var.get()
                amount = float(self.amount_var.get())
                if sender == "Select Wallet" or not recipient or amount <= 0:
                    self.on_error("Invalid transaction details")
                    return
                wallet = await self.blockchain.get_wallet(sender)
                tx = await self.blockchain.create_transaction(wallet['private_key'], sender, recipient, amount, fee=0.001)
                if await self.blockchain.add_transaction_to_mempool(tx):
                    if self.backup_manager:
                        await self.backup_manager.backup_transaction(tx)
                    self.update_queue.put(lambda: [self.amount_var.set(""), self.recipient_var.set(""), messagebox.showinfo("Success", "Transaction sent")])
                    await self.update_async_balance(sender)
                    await self.update_async_transaction_history(sender)
                else:
                    self.on_error("Transaction rejected")
            except ValueError:
                self.on_error("Invalid amount")
            except Exception as e:
                self.on_error(f"Transaction failed: {e}")
        asyncio.run_coroutine_threadsafe(send(), self.loop)

    def update_network_tab(self):
        async def update():
            try:
                peers = self.network.peers
                self.update_queue.put(lambda: self.peers_var.set(f"Connected Peers: {len(peers)}"))
                def update_peers_tree():
                    self.peers_tree.delete(*self.peers_tree.get_children())
                    for peer_id, data in peers.items():
                        self.peers_tree.insert(
                            "", "end",
                            values=(peer_id, f"{data['host']}:{data['port']}",
                                    time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(data['last_seen'])))
                        )
                self.update_queue.put(update_peers_tree)
            except Exception as e:
                self.on_error(f"Network update failed: {e}")
        if not self._shutdown_in_progress:
            asyncio.run_coroutine_threadsafe(update(), self.loop)
            self.root.after(5000, self.update_network_tab)  # Periodic update every 5s

    def handle_mfa_setup(self):
        async def setup():
            try:
                wallet = self.selected_wallet.get()
                if wallet == "Select Wallet":
                    self.on_error("Select a wallet first")
                    return
                secret = await self.mfa_manager.generate_mfa_secret(wallet)
                qr_code = await self.mfa_manager.get_mfa_qr(wallet, f"OriginalCoin-Wallet-{wallet[:10]}")
                self.update_queue.put(lambda: self.show_qr_code(qr_code))
            except Exception as e:
                self.on_error(f"MFA setup failed: {e}")
        asyncio.run_coroutine_threadsafe(setup(), self.loop)

    def handle_backup_keys(self):
        async def backup():
            try:
                password = await asyncio.get_event_loop().run_in_executor(None, lambda: simpledialog.askstring("Password", "Enter backup password:", show='*', parent=self.root))
                if password:
                    await self.backup_manager.create_backup(self.network.private_key, password)
                    self.update_queue.put(lambda: messagebox.showinfo("Success", "Keys backed up"))
            except Exception as e:
                self.on_error(f"Backup failed: {e}")
        asyncio.run_coroutine_threadsafe(backup(), self.loop)

    def handle_restore_keys(self):
        async def restore():
            try:
                password = await asyncio.get_event_loop().run_in_executor(None, lambda: simpledialog.askstring("Password", "Enter backup password:", show='*', parent=self.root))
                if not password:
                    return
                file_path = await asyncio.get_event_loop().run_in_executor(None, lambda: filedialog.askopenfilename(title="Select Backup File", filetypes=[("Encrypted Backup", "*.enc")]))
                if file_path:
                    keys = await self.backup_manager.restore_backup(file_path, password)
                    # Assuming network has a method to update keys
                    self.network.private_key = keys['private_key']
                    self.network.public_key = keys['public_key']
                    self.update_queue.put(lambda: messagebox.showinfo("Success", "Keys restored"))
            except Exception as e:
                self.on_error(f"Restore failed: {e}")
        asyncio.run_coroutine_threadsafe(restore(), self.loop)

    def show_qr_code(self, qr_code):
        qr_window = tk.Toplevel(self.root)
        qr_window.title("MFA Setup")
        qr_image = ImageTk.PhotoImage(qr_code)
        ttk.Label(qr_window, image=qr_image).pack(padx=10, pady=10)
        qr_window.qr_image = qr_image
        ttk.Label(qr_window, text="Scan with authenticator app").pack(padx=10, pady=5)

    def add_security_controls(self):
        security_frame = ttk.LabelFrame(self.main_frame, text="Security Controls", padding="5")
        security_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        ttk.Button(security_frame, text="Setup MFA", command=self.handle_mfa_setup).grid(row=0, column=0, padx=5, pady=5)
        ttk.Button(security_frame, text="Backup Keys", command=self.handle_backup_keys).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(security_frame, text="Restore Keys", command=self.handle_restore_keys).grid(row=0, column=2, padx=5, pady=5)

    def on_closing(self):
        if self._shutdown_in_progress:
            return
        self._shutdown_in_progress = True
        self.status_var.set("Shutting down...")
        logger.info("GUI closing initiated")

        async def shutdown():
            try:
                if self.mining:
                    await self.blockchain.stop_mining()
                await self.network.stop()
                await self.blockchain.shutdown()
            except Exception as e:
                logger.error(f"Shutdown error: {e}")
        future = asyncio.run_coroutine_threadsafe(shutdown(), self.loop)
        try:
            future.result(timeout=15)
        except concurrent.futures.TimeoutError:
            logger.warning("Shutdown timed out")

        self.root.quit()
        self.root.destroy()
        logger.info("GUI shutdown completed")

    def run(self):
        """Run the GUI"""
        async def initialize():
            try:
                await self.blockchain.initialize()
                chain_data = await self.network.request_chain()  # Initial sync using network.get_chain-like functionality
                if chain_data:
                    await self.blockchain.replace_chain(chain_data)
                self.update_wallet_dropdown()
                self.update_network_tab()
                self.update_queue.put(lambda: self.status_var.set("Ready"))
            except Exception as e:
                self.on_error(f"Initialization failed: {e}")
        asyncio.run_coroutine_threadsafe(initialize(), self.loop)
        self.root.mainloop()