from collections import defaultdict
import time
import logging
from typing import Dict, List, Set
import asyncio

logger = logging.getLogger(__name__)

class SecurityMonitor:
    def __init__(self):
        self.suspicious_ips: Set[str] = set()
        self.failed_attempts: Dict[str, List[float]] = defaultdict(list)
        self.blocked_ips: Set[str] = set()
        self.connection_history: Dict[str, List[float]] = defaultdict(list)
        
        # Configure thresholds
        self.FAILED_ATTEMPT_WINDOW = 300  # 5 minutes
        self.MAX_FAILED_ATTEMPTS = 5
        self.CONNECTION_RATE_WINDOW = 60  # 1 minute
        self.MAX_CONNECTIONS_PER_WINDOW = 30
        
    async def monitor_connection(self, ip: str) -> bool:
        """Returns True if connection should be allowed"""
        if ip in self.blocked_ips:
            logger.warning(f"Blocked connection attempt from {ip}")
            return False
            
        current_time = time.time()
        self.connection_history[ip].append(current_time)
        
        # Clean old connection history
        self.connection_history[ip] = [t for t in self.connection_history[ip] 
                                     if current_time - t <= self.CONNECTION_RATE_WINDOW]
        
        # Check connection rate
        if len(self.connection_history[ip]) > self.MAX_CONNECTIONS_PER_WINDOW:
            logger.warning(f"Rate limit exceeded for {ip}")
            self.suspicious_ips.add(ip)
            return False
            
        return True
        
    async def record_failed_attempt(self, ip: str, attempt_type: str):
        """Record failed authentication or validation attempts"""
        current_time = time.time()
        self.failed_attempts[ip].append(current_time)
        
        # Clean old attempts
        self.failed_attempts[ip] = [t for t in self.failed_attempts[ip] 
                                  if current_time - t <= self.FAILED_ATTEMPT_WINDOW]
        
        if len(self.failed_attempts[ip]) >= self.MAX_FAILED_ATTEMPTS:
            logger.error(f"Multiple failed attempts from {ip}, blocking")
            self.blocked_ips.add(ip)
            
    async def analyze_patterns(self):
        """Periodic analysis of security patterns"""
        while True:
            try:
                current_time = time.time()
                
                # Analyze connection patterns
                for ip, connections in self.connection_history.items():
                    recent_connections = len([t for t in connections 
                                           if current_time - t <= 60])
                    if recent_connections > self.MAX_CONNECTIONS_PER_WINDOW:
                        logger.warning(f"Suspicious connection pattern from {ip}")
                        self.suspicious_ips.add(ip)
                
                # Clean up old data
                await self.cleanup_old_data()
                
            except Exception as e:
                logger.error(f"Error in security pattern analysis: {e}")
            
            await asyncio.sleep(60)  # Run analysis every minute
            
    async def cleanup_old_data(self):
        """Clean up old security data"""
        current_time = time.time()
        
        # Clean up connection history older than 1 hour
        for ip in list(self.connection_history.keys()):
            self.connection_history[ip] = [t for t in self.connection_history[ip] 
                                         if current_time - t <= 3600]
            if not self.connection_history[ip]:
                del self.connection_history[ip]
