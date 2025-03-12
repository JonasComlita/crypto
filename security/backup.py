import json
import base64
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import os
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class KeyBackupManager:
    def __init__(self, backup_dir: str):
        self.backup_dir = backup_dir
        os.makedirs(backup_dir, exist_ok=True)
        self.wallet_backups = {}  # Track backed up wallets
        
    def generate_backup_key(self, password: str) -> bytes:
        """Generate encryption key from password"""
        salt = os.urandom(16)
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        return key, salt
        
    async def create_backup(self, keys: dict, password: str) -> str:
        """Create encrypted backup of keys"""
        try:
            key, salt = self.generate_backup_key(password)
            f = Fernet(key)
            
            backup_data = {
                'timestamp': datetime.utcnow().isoformat(),
                'salt': base64.b64encode(salt).decode(),
                'keys': keys
            }
            
            encrypted_data = f.encrypt(json.dumps(backup_data).encode())
            
            backup_path = os.path.join(
                self.backup_dir,
                f"backup_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.enc"
            )
            
            with open(backup_path, 'wb') as f:
                f.write(encrypted_data)
                
            logger.info(f"Created encrypted backup: {backup_path}")
            return backup_path
            
        except Exception as e:
            logger.error(f"Backup creation failed: {e}")
            raise
            
    async def restore_backup(self, backup_path: str, password: str) -> dict:
        """Restore keys from encrypted backup"""
        try:
            with open(backup_path, 'rb') as f:
                encrypted_data = f.read()
                
            backup_data = json.loads(encrypted_data.decode())
            salt = base64.b64decode(backup_data['salt'])
            
            key = self.generate_backup_key(password, salt)
            f = Fernet(key)
            
            decrypted_data = json.loads(f.decrypt(encrypted_data).decode())
            logger.info(f"Successfully restored backup from {backup_path}")
            
            return decrypted_data['keys']
            
        except Exception as e:
            logger.error(f"Backup restoration failed: {e}")
            raise

    async def is_wallet_backed_up(self, wallet_address: str) -> bool:
        """Check if a wallet has been backed up"""
        return wallet_address in self.wallet_backups
        
    async def backup_transaction(self, transaction):
        """Track wallet backup status after transaction"""
        try:
            self.wallet_backups[transaction.sender] = True
            self.wallet_backups[transaction.recipient] = True
        except Exception as e:
            logger.error(f"Failed to track wallet backup: {e}") 