import pyotp
import qrcode
from PIL import Image
import base64
from typing import Optional, Dict, Union
import logging
from datetime import datetime, timedelta
import asyncio
import os
import json

logger = logging.getLogger(__name__)

class MFAManagerException(Exception):
    """Custom exception for MFA-related errors"""
    pass

class MFAManager:
    def __init__(self, config_dir: str = 'data/mfa'):
        """
        Initialize MFA Manager with configurable storage
        
        Args:
            config_dir (str): Directory to store MFA configuration
        """
        self.verified_sessions: Dict[str, datetime] = {}
        self.mfa_secrets: Dict[str, str] = {}
        self.config_dir = config_dir
        
        # Ensure config directory exists
        os.makedirs(config_dir, exist_ok=True)
        
        # Load existing MFA configurations
        self._load_mfa_configs()

    def _load_mfa_configs(self):
        """Load existing MFA configurations from disk"""
        try:
            config_file = os.path.join(self.config_dir, 'mfa_config.json')
            if os.path.exists(config_file):
                with open(config_file, 'r') as f:
                    saved_configs = json.load(f)
                    self.mfa_secrets = saved_configs.get('secrets', {})
        except Exception as e:
            logger.error(f"Error loading MFA configurations: {e}")

    def _save_mfa_configs(self):
        """Save MFA configurations to disk"""
        try:
            config_file = os.path.join(self.config_dir, 'mfa_config.json')
            config_data = {
                'secrets': self.mfa_secrets
            }
            with open(config_file, 'w') as f:
                json.dump(config_data, f)
        except Exception as e:
            logger.error(f"Error saving MFA configurations: {e}")

    async def generate_mfa_secret(self, user_id: str) -> str:
        """
        Generate new MFA secret for a user
        
        Args:
            user_id (str): Unique identifier for the user
        
        Returns:
            str: Generated MFA secret
        
        Raises:
            MFAManagerException: If secret generation fails
        """
        try:
            # Check if MFA is already configured
            if user_id in self.mfa_secrets:
                logger.warning(f"MFA already configured for {user_id}. Regenerating...")
            
            # Generate new secret
            secret = pyotp.random_base32()
            self.mfa_secrets[user_id] = secret
            
            # Persist configuration
            self._save_mfa_configs()
            
            logger.info(f"Generated MFA secret for {user_id}")
            return secret
        except Exception as e:
            logger.error(f"Error generating MFA secret: {e}")
            raise MFAManagerException(f"Failed to generate MFA secret: {e}")

    async def get_mfa_qr(self, user_id: str, username: str) -> Image.Image:
        """
        Generate QR code for MFA setup
        
        Args:
            user_id (str): Unique identifier for the user
            username (str): Username for the QR code
        
        Returns:
            PIL Image of the QR code
        
        Raises:
            MFAManagerException: If QR code generation fails
        """
        try:
            # Ensure MFA secret exists
            if user_id not in self.mfa_secrets:
                raise MFAManagerException(f"MFA not set up for user {user_id}")
            
            # Create TOTP object
            totp = pyotp.TOTP(self.mfa_secrets[user_id])
            
            # Generate provisioning URI
            provisioning_uri = totp.provisioning_uri(
                username,
                issuer_name="OriginalCoin"
            )
            
            # Create QR code
            qr = qrcode.QRCode(version=1, box_size=10, border=5)
            qr.add_data(provisioning_uri)
            qr.make(fit=True)
            
            return qr.make_image(fill_color="black", back_color="white")
        except Exception as e:
            logger.error(f"Error generating MFA QR code: {e}")
            raise MFAManagerException(f"Failed to generate MFA QR code: {e}")

    async def is_mfa_configured(self, user_id: str) -> bool:
        """
        Check if MFA is configured for a specific user
        
        Args:
            user_id (str): Unique identifier for the user
        
        Returns:
            bool: True if MFA is configured, False otherwise
        """
        return user_id in self.mfa_secrets

    def verify_mfa(self, user_id: str, code: str) -> bool:
        """
        Verify MFA code
        
        Args:
            user_id (str): Unique identifier for the user
            code (str): MFA code to verify
        
        Returns:
            bool: True if verification succeeds, False otherwise
        """
        try:
            # Check if MFA is configured
            if user_id not in self.mfa_secrets:
                logger.warning(f"No MFA secret found for user {user_id}")
                return False
            
            # Create TOTP object
            totp = pyotp.TOTP(self.mfa_secrets[user_id])
            
            # Verify code with additional window for clock skew
            if totp.verify(code, valid_window=1):
                # Record verified session
                self.verified_sessions[user_id] = datetime.now()
                logger.info(f"MFA verified for user {user_id}")
                return True
            
            logger.warning(f"Invalid MFA code for user {user_id}")
            return False
        except Exception as e:
            logger.error(f"MFA verification error for user {user_id}: {e}")
            return False

    def is_session_valid(self, user_id: str, max_age_minutes: int = 30) -> bool:
        """
        Check if user has a valid MFA session
        
        Args:
            user_id (str): Unique identifier for the user
            max_age_minutes (int): Maximum session duration in minutes
        
        Returns:
            bool: True if session is valid, False otherwise
        """
        if user_id not in self.verified_sessions:
            return False
        
        session_time = self.verified_sessions[user_id]
        is_valid = datetime.now() - session_time < timedelta(minutes=max_age_minutes)
        
        # Optional: Clean up expired sessions
        if not is_valid:
            del self.verified_sessions[user_id]
        
        return is_valid

    def reset_mfa(self, user_id: str) -> bool:
        """
        Reset MFA configuration for a user
        
        Args:
            user_id (str): Unique identifier for the user
        
        Returns:
            bool: True if reset successful, False otherwise
        """
        try:
            # Remove secrets and sessions
            if user_id in self.mfa_secrets:
                del self.mfa_secrets[user_id]
            
            if user_id in self.verified_sessions:
                del self.verified_sessions[user_id]
            
            # Update persistent storage
            self._save_mfa_configs()
            
            logger.info(f"MFA reset for user {user_id}")
            return True
        except Exception as e:
            logger.error(f"Error resetting MFA for user {user_id}: {e}")
            return False