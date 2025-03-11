import asyncio
import logging
import os
import uuid
from flask import Flask
from dotenv import load_dotenv
from typing import Optional
from key_rotation.core import KeyRotationManager
from key_rotation.api import create_rotation_api

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

async def main(node_id: Optional[str] = None, is_validator: bool = False, port: int = 5000, host: str = "127.0.0.1") -> None:
    load_dotenv()
    node_id = node_id or os.getenv("NODE_ID") or str(uuid.uuid4())
    is_validator = is_validator or (os.getenv("IS_VALIDATOR", "false").lower() == "true")
    
    logger.info(f"Starting node {node_id}, validator: {is_validator}, host: {host}, port: {port}")
    
    rotation_manager = KeyRotationManager(node_id=node_id, is_validator=is_validator)
    await rotation_manager.start()
    
    app = Flask(__name__)
    create_rotation_api(app, rotation_manager)
    
    ssl_context = None
    cert_path = os.getenv("CERT_PATH")
    key_path = os.getenv("KEY_PATH")
    if cert_path and key_path and os.path.exists(cert_path) and os.path.exists(key_path):
        ssl_context = (cert_path, key_path)
        logger.info(f"Using HTTPS with cert: {cert_path}, key: {key_path}")
    else:
        logger.info("Running without HTTPS; cert or key not provided")

    try:
        from flask import cli
        cli.show_server_banner = lambda *x: None  # Suppress Flask banner
        await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: app.run(host=host, port=port, ssl_context=ssl_context, threaded=True)
        )
    except Exception as e:
        logger.error(f"Failed to start Flask app: {e}")
        raise
    finally:
        await rotation_manager.stop()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Shutting down gracefully")
    except Exception as e:
        logger.error(f"Startup failed: {e}", exc_info=True)