import pyotp
import qrcode
import base64
from typing import Optional
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class MFAManager:
    def __init__(self):
        self.verified_sessions = {}
        self.mfa_secrets = {}
        
    async def generate_mfa_secret(self, user_id: str) -> str:
        """Generate new MFA secret for a user"""
        secret = pyotp.random_base32()
        self.mfa_secrets[user_id] = secret
        return secret
        
    async def get_mfa_qr(self, user_id: str, username: str) -> str:
        """Generate QR code for MFA setup"""
        if user_id not in self.mfa_secrets:
            raise ValueError("MFA not set up for this user")
            
        totp = pyotp.TOTP(self.mfa_secrets[user_id])
        provisioning_uri = totp.provisioning_uri(
            username,
            issuer_name="OriginalCoin"
        )
        
        qr = qrcode.QRCode(version=1, box_size=10, border=5)
        qr.add_data(provisioning_uri)
        qr.make(fit=True)
        
        return qr.make_image()
        
    async def verify_mfa(self, user_id: str, token: str) -> bool:
        """Verify MFA token"""
        if user_id not in self.mfa_secrets:
            logger.warning(f"MFA verification attempted for unregistered user {user_id}")
            return False
            
        totp = pyotp.TOTP(self.mfa_secrets[user_id])
        if totp.verify(token):
            self.verified_sessions[user_id] = datetime.now()
            return True
        
        logger.warning(f"Failed MFA verification for user {user_id}")
        return False
        
    def is_session_valid(self, user_id: str, max_age_minutes: int = 30) -> bool:
        """Check if user has a valid MFA session"""
        if user_id not in self.verified_sessions:
            return False
            
        session_time = self.verified_sessions[user_id]
        return datetime.now() - session_time < timedelta(minutes=max_age_minutes)
