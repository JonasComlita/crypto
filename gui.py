import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import tkinter.simpledialog
import json
import re
import time
from queue import Queue
from typing import Dict, List
from blockchain import Blockchain, Block, Transaction, TransactionType, Miner
from network import BlockchainNetwork
from utils import SecurityUtils, generate_wallet, derive_key, Fernet
import logging
import asyncio

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BlockchainGUI:
    def __init__(self, blockchain: Blockchain, network: BlockchainNetwork, wallet_file: str = "wallets.json"):
        self.blockchain = blockchain
        self.network = network
        self.wallets = {}
        self.wallet_file = wallet_file
        self.miner = Miner(blockchain, blockchain.mempool, None)
        self.load_wallets()
        self.update_queue = Queue()
        self.root = tk.Tk()
        self.root.title("OriginalCoin GUI")
        self.root.geometry("900x1000")
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill='both', expand=True, padx=10, pady=10)
        self.create_widgets()
        self.blockchain.subscribe("new_block", self.on_new_block)
        self.blockchain.subscribe("new_transaction", self.on_new_transaction)
        self.root.after(1000, self.update_ui)
        self.root.protocol("WM_DELETE_WINDOW", self.exit)

    def create_widgets(self):
        self.create_header()
        self.create_wallet_section()
        self.create_mining_section()
        self.create_transaction_section()
        self.create_peer_section()
        self.create_status_panel()
        self.create_transaction_history()
        self.create_footer()

    def create_header(self):
        header_frame = ttk.Frame(self.main_frame)
        header_frame.grid(row=0, column=0, sticky='ew', pady=(0, 10))
        ttk.Label(header_frame, text="OriginalCoin", font=("Arial", 24, "bold")).pack(fill='x')

    def create_wallet_section(self):
        wallet_frame = ttk.LabelFrame(self.main_frame, text="Wallet", padding=10)
        wallet_frame.grid(row=1, column=0, sticky='ew', pady=5)
        ttk.Label(wallet_frame, text="Wallet:").grid(row=0, column=0, sticky='w', padx=5)
        self.wallet_entry = ttk.Entry(wallet_frame, width=20)
        self.wallet_entry.grid(row=0, column=1, sticky='ew', padx=5)
        self.wallet_entry.bind("<KeyRelease>", self.on_entry_change)
        self.wallet_var = tk.StringVar(value="")
        self.wallet_dropdown = ttk.OptionMenu(wallet_frame, self.wallet_var, "", *self.wallets.keys(), command=self.on_dropdown_select)
        self.wallet_dropdown.grid(row=0, column=2, sticky='ew', padx=5)
        ttk.Button(wallet_frame, text="Create Wallet", command=self.create_wallet).grid(row=0, column=3, sticky='e', padx=5)
        ttk.Button(wallet_frame, text="Save Wallets", command=self.save_wallets).grid(row=0, column=4, sticky='e', padx=5)
        wallet_frame.grid_columnconfigure(1, weight=1)

    def create_mining_section(self):
        mining_frame = ttk.LabelFrame(self.main_frame, text="Mining Controls", padding=10)
        mining_frame.grid(row=2, column=0, sticky='ew', pady=5)
        ttk.Button(mining_frame, text="Start Mining", command=self.start_mining).grid(row=0, column=0, sticky='w', padx=5)
        ttk.Button(mining_frame, text="Stop Mining", command=self.stop_mining).grid(row=0, column=1, sticky='e', padx=5)
        mining_frame.grid_columnconfigure(0, weight=1)
        mining_frame.grid_columnconfigure(1, weight=1)

    def create_transaction_section(self):
        send_frame = ttk.LabelFrame(self.main_frame, text="Send Transaction", padding=10)
        send_frame.grid(row=3, column=0, sticky='ew', pady=5)
        ttk.Label(send_frame, text="Recipient:").grid(row=0, column=0, sticky='w', padx=5)
        self.to_entry = ttk.Entry(send_frame, width=40)
        self.to_entry.grid(row=0, column=1, sticky='ew', padx=5)
        ttk.Label(send_frame, text="Amount:").grid(row=1, column=0, sticky='w', padx=5)
        self.amount_entry = ttk.Entry(send_frame)
        self.amount_entry.grid(row=1, column=1, sticky='ew', padx=5)
        ttk.Button(send_frame, text="Send Transaction", command=self.send_transaction).grid(row=1, column=2, sticky='e', padx=5)
        send_frame.grid_columnconfigure(1, weight=1)

    def create_peer_section(self):
        peer_frame = ttk.LabelFrame(self.main_frame, text="Peer Management", padding=10)
        peer_frame.grid(row=4, column=0, sticky='ew', pady=5)
        ttk.Label(peer_frame, text="Host:").grid(row=0, column=0, sticky='w', padx=5)
        self.peer_host_entry = ttk.Entry(peer_frame)
        self.peer_host_entry.grid(row=0, column=1, sticky='ew', padx=5)
        self.peer_host_entry.insert(0, "127.0.0.1")
        ttk.Label(peer_frame, text="Port:").grid(row=1, column=0, sticky='w', padx=5)
        self.peer_port_entry = ttk.Entry(peer_frame)
        self.peer_port_entry.grid(row=1, column=1, sticky='ew', padx=5)
        ttk.Button(peer_frame, text="Add Peer", command=self.add_peer).grid(row=1, column=2, sticky='e', padx=5)
        peer_frame.grid_columnconfigure(1, weight=1)

    def create_status_panel(self):
        status_frame = ttk.LabelFrame(self.main_frame, text="Status", padding=10)
        status_frame.grid(row=5, column=0, sticky='nsew', pady=5)
        ttk.Label(status_frame, text="Connected Peers:").grid(row=0, column=0, sticky='w', padx=5)
        peers_container = ttk.Frame(status_frame)
        peers_container.grid(row=1, column=0, columnspan=2, sticky='ew', pady=5)
        self.peer_listbox = tk.Listbox(peers_container, height=3)
        self.peer_listbox.pack(side='left', fill='x', expand=True)
        scrollbar = ttk.Scrollbar(peers_container, orient='vertical', command=self.peer_listbox.yview)
        scrollbar.pack(side='right', fill='y')
        self.peer_listbox.config(yscrollcommand=scrollbar.set)
        self.balance_label = ttk.Label(status_frame, text="Balance: 0.0")
        self.balance_label.grid(row=2, column=0, sticky='w', padx=5, pady=5)
        self.chain_height_label = ttk.Label(status_frame, text="Chain Height: 1")
        self.chain_height_label.grid(row=2, column=1, sticky='e', padx=5, pady=5)
        self.network_stats_label = ttk.Label(status_frame, text="Connected Peers: 0")
        self.network_stats_label.grid(row=3, column=0, sticky='w', padx=5, pady=5)
        self.update_peer_list()
        ttk.Label(status_frame, text="Logs:").grid(row=4, column=0, sticky='w', padx=5, pady=5)
        self.output = scrolledtext.ScrolledText(status_frame, width=70, height=10)
        self.output.grid(row=5, column=0, columnspan=2, sticky='nsew', padx=5, pady=5)
        ttk.Label(status_frame, text="Mempool:").grid(row=6, column=0, sticky='w', padx=5, pady=5)
        self.mempool_text = scrolledtext.ScrolledText(status_frame, width=70, height=3)
        self.mempool_text.grid(row=7, column=0, columnspan=2, sticky='nsew', padx=5, pady=5)
        status_frame.grid_columnconfigure(0, weight=1)
        status_frame.grid_columnconfigure(1, weight=1)
        self.main_frame.grid_rowconfigure(5, weight=1)

    def create_transaction_history(self):
        history_frame = ttk.LabelFrame(self.main_frame, text="Transaction History", padding=10)
        history_frame.grid(row=6, column=0, sticky='nsew', pady=5)
        self.history_text = scrolledtext.ScrolledText(history_frame, width=70, height=5)
        self.history_text.grid(row=0, column=0, sticky='nsew')
        history_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_rowconfigure(6, weight=1)

    def create_footer(self):
        bottom_frame = ttk.Frame(self.root)
        bottom_frame.pack(side='bottom', fill='x', pady=(0, 10))
        ttk.Button(bottom_frame, text="Exit", command=self.exit).pack(side='right', padx=10)

    def load_wallets(self):
        try:
            with open(self.wallet_file, 'r') as f:
                data = json.load(f)
                key, salt = data["key"], bytes.fromhex(data["salt"])
                cipher = Fernet(key)
                self.wallets = {name: {**w, "private_key": cipher.decrypt(w["private_key"].encode()).decode()}
                                for name, w in data["wallets"].items()}
        except Exception as e:
            logger.error(f"Failed to load wallets: {e}")
            self.wallets = {}

    def save_wallets(self):
        password = tkinter.simpledialog.askstring("Password", "Enter wallet password (min 8 chars, 1 upper, 1 lower, 1 digit):", show='*', parent=self.root)
        if not password or not re.match(r"^(?=.*[A-Z])(?=.*[a-z])(?=.*\d).{8,}$", password):
            messagebox.showerror("Error", "Password must be at least 8 characters with 1 uppercase, 1 lowercase, and 1 digit")
            return
        try:
            key, salt = derive_key(password)
            cipher = Fernet(key)
            encrypted_wallets = {name: {**w, "private_key": cipher.encrypt(w["private_key"].encode()).decode()}
                                for name, w in self.wallets.items()}
            with open(self.wallet_file, 'w') as f:
                json.dump({"key": key.decode(), "salt": salt.hex(), "wallets": encrypted_wallets}, f)
            self.update_wallet_dropdown()
        except Exception as e:
            logger.error(f"Failed to save wallets: {e}")
            messagebox.showerror("Error", f"Failed to save wallets: {e}")
            pass

    def update_wallet_dropdown(self):
        menu = self.wallet_dropdown["menu"]
        menu.delete(0, "end")
        options = list(self.wallets.keys())
        if not options:
            menu.add_command(label="No wallets", command=lambda: self.wallet_var.set(""))
        else:
            for name in options:
                menu.add_command(label=name, command=lambda n=name: self.wallet_var.set(n))
            if self.wallet_entry.get().strip() in options:
                self.wallet_var.set(self.wallet_entry.get().strip())
            elif options:
                self.wallet_var.set(options[0])
                pass

    def on_entry_change(self, event):
        name = self.wallet_entry.get().strip()
        if name in self.wallets:
            self.wallet_var.set(name)
        else:
            self.wallet_var.set("")
        self.update_balance()
        pass

    def on_dropdown_select(self, *args):
        name = self.wallet_var.get()
        if name and name != "No wallets":
            self.wallet_entry.delete(0, tk.END)
            self.wallet_entry.insert(0, name)
        self.update_balance()

    def update_balance(self):
        name = self.wallet_entry.get().strip()
        if name and name in self.wallets:
            # Schedule async call in network's loop
            future = asyncio.run_coroutine_threadsafe(
                self.blockchain.get_balance(self.wallets[name]["address"]), self.network.loop
            )
            balance = future.result()
            self.balance_label.config(text=f"Balance: {balance}")

    def create_wallet(self):
        name = tkinter.simpledialog.askstring("Input", "Enter wallet name (alphanumeric only):", parent=self.root)
        if not name or not name.isalnum() or any(n.lower() == name.lower() for n in self.wallets):
            messagebox.showerror("Error", "Wallet name must be unique and alphanumeric")
            return
        try:
            wallet = generate_wallet()
            self.wallets[name] = wallet
            self.save_wallets()
            self.output.insert(tk.END, f"Created wallet '{name}': Address: {wallet['address']}\n")
            self.wallet_entry.delete(0, tk.END)
            self.wallet_entry.insert(0, name)
            self.update_balance()
        except Exception as e:
            logger.error(f"Failed to create wallet: {e}")
            messagebox.showerror("Error", f"Failed to create wallet: {e}")
            pass

    def add_peer(self):
        host = self.peer_host_entry.get().strip()
        port_str = self.peer_port_entry.get().strip()
        if not host or not port_str or not port_str.isdigit():
            messagebox.showerror("Error", "Host and port must be valid")
            return
        port = int(port_str)
        if not (1 <= port <= 65535):
            messagebox.showerror("Error", "Port must be between 1 and 65535")
            return
        try:
            peer_id = f"node{port}"
            future = asyncio.run_coroutine_threadsafe(
                self.network.add_peer(peer_id, host, port, self.network.public_key), self.network.loop
            )
            success = future.result()
            if success:
                self.update_peer_list()
            else:
                messagebox.showerror("Error", f"Failed to add peer {peer_id}")
        except Exception as e:
            logger.error(f"Failed to add peer: {e}")
            messagebox.showerror("Error", f"Failed to add peer: {e}")

    def update_peer_list(self):
        self.peer_listbox.delete(0, tk.END)
        for peer_id, (host, port, _) in self.network.peers.items():
            self.peer_listbox.insert(tk.END, f"{peer_id}: {host}:{port}")
        self.network_stats_label.config(text=f"Connected Peers: {len(self.network.peers)}")
        pass

    def start_mining(self):
        name = self.wallet_entry.get().strip()
        if not name or name not in self.wallets:
            messagebox.showerror("Error", "Select a valid wallet")
            return
        try:
            self.miner.wallet_address = self.wallets[name]["address"]
            asyncio.run_coroutine_threadsafe(self.miner.start_mining(), self.network.loop)
            self.output.insert(tk.END, f"Mining started with wallet '{name}'\n")
        except Exception as e:
            logger.error(f"Failed to start mining: {e}")
            messagebox.showerror("Error", f"Mining failed: {e}")

    def stop_mining(self):
        try:
            asyncio.run_coroutine_threadsafe(self.miner.stop_mining(), self.network.loop)
            self.output.insert(tk.END, "Mining stopped\n")
        except Exception as e:
            logger.error(f"Failed to stop mining: {e}")
            messagebox.showerror("Error", f"Failed to stop mining: {e}")
            pass

    def send_transaction(self):
        from_name = self.wallet_entry.get().strip()
        to_address = self.to_entry.get().strip()
        amount = self.amount_entry.get().strip()
        if not from_name or not to_address or not amount:
            messagebox.showerror("Error", "Provide all fields")
            return
        if from_name not in self.wallets:
            messagebox.showerror("Error", f"Wallet '{from_name}' does not exist")
            return
        try:
            amount = float(amount)
            if amount <= 0:
                raise ValueError("Amount must be positive")
            wallet = self.wallets[from_name]
            future = asyncio.run_coroutine_threadsafe(self._send_transaction(wallet, to_address, amount), self.network.loop)
            tx = future.result()
            if tx:
                self.output.insert(tk.END, f"Transaction sent: {tx.tx_id[:8]}\n")
            else:
                balance = asyncio.run_coroutine_threadsafe(self.blockchain.get_balance(wallet["address"]), self.network.loop).result()
                messagebox.showerror("Error", f"Insufficient funds. Balance: {balance}, Required: {amount + 0.001}")
        except Exception as e:
            logger.error(f"Failed to send transaction: {e}")
            messagebox.showerror("Error", f"Invalid transaction: {e}")

    async def _send_transaction(self, wallet: Dict, to_address: str, amount: float) -> Transaction:
        tx = await self.blockchain.create_transaction(wallet["private_key"], wallet["address"], to_address, amount)
        if tx and await self.blockchain.add_transaction_to_mempool(tx):
            await self.network.broadcast_transaction(tx)
            return tx
        return None

    def on_new_block(self, block: Block):
        self.update_queue.put(("block", block))

    def on_new_transaction(self, tx: Transaction):
        self.update_queue.put(("transaction", tx))

    def update_ui(self):
        while not self.update_queue.empty():
            event_type, data = self.update_queue.get()
            if event_type == "block":
                self.output.insert(tk.END, f"New block mined: {data.header.index}\n")
                self.chain_height_label.config(text=f"Chain Height: {len(self.blockchain.chain)}")
                self.update_balance()
            elif event_type == "transaction":
                self.output.insert(tk.END, f"New transaction in mempool: {data.tx_id[:8]}\n")
        self.mempool_text.delete(1.0, tk.END)
        for tx in self.blockchain.mempool.transactions.values():
            status = "Pending"
            for block in self.blockchain.chain[-6:]:
                if any(t.tx_id == tx.tx_id for t in block.transactions):
                    status = "Confirmed"
                    break
            self.mempool_text.insert(tk.END, f"{tx.tx_id[:8]}: {status} - {tx.outputs[0].amount} to {tx.outputs[0].recipient[:8]}\n")
        name = self.wallet_entry.get().strip()
        if name in self.wallets:
            address = self.wallets[name]["address"]
            self.history_text.delete(1.0, tk.END)
            for block in self.blockchain.chain:
                for tx in block.transactions:
                    if (tx.tx_type != TransactionType.COINBASE and any(i.public_key and SecurityUtils.public_key_to_address(i.public_key) == address for i in tx.inputs)) or any(o.recipient == address for o in tx.outputs):
                        direction = "Sent" if any(i.public_key and SecurityUtils.public_key_to_address(i.public_key) == address for i in tx.inputs) else "Received"
                        self.history_text.insert(tk.END, f"{tx.tx_id[:8]}: {direction} {tx.outputs[0].amount} at {time.ctime(block.header.timestamp)}\n")
        self.root.after(1000, self.update_ui)
        pass

    def exit(self):
        try:
            asyncio.run_coroutine_threadsafe(self.miner.stop_mining(), self.network.loop)
        except Exception as e:
            logger.error(f"Error stopping miner during exit: {e}")
        self.root.quit()

    def run(self):
        self.update_wallet_dropdown()
        self.root.mainloop()