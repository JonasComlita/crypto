import asyncio
import logging
import os
import uuid
import ssl
from flask import Flask
from dotenv import load_dotenv
from typing import Optional
from key_rotation.core import KeyRotationManager
from key_rotation.api import create_rotation_api

logger = logging.getLogger(__name__)

async def main(node_id: Optional[str] = None, is_validator: bool = False, 
              port: int = 5000, host: str = "127.0.0.1", loop=None, shutdown_event: Optional[asyncio.Event] = None) -> None:
    load_dotenv()
    node_id = node_id or os.getenv("NODE_ID") or str(uuid.uuid4())
    is_validator = is_validator or (os.getenv("IS_VALIDATOR", "false").lower() == "true")
    
    logger.info(f"Starting node {node_id}, validator: {is_validator}, host: {host}, port: {port}")
    
    rotation_manager = KeyRotationManager(node_id=node_id, is_validator=is_validator)
    await rotation_manager.start()
    
    # Use the provided loop or the current one
    current_loop = loop or asyncio.get_event_loop()
    
    app = Flask(__name__)
    create_rotation_api(app, rotation_manager)
    
    # SSL setup with certificate generation
    cert_path = os.getenv("CERT_PATH", f"certs/{node_id}_{port}.crt")
    key_path = os.getenv("KEY_PATH", f"certs/{node_id}_{port}.key")
    os.makedirs("certs", exist_ok=True)
    
    if not (os.path.exists(cert_path) and os.path.exists(key_path)):
        logger.info(f"SSL certificates not found for {node_id} on port {port}. Generating self-signed certificates...")
        cmd = (
            f'openssl req -x509 -newkey rsa:2048 -keyout "{key_path}" '
            f'-out "{cert_path}" -days 365 -nodes -subj "/CN={node_id}"'
        )
        with open(os.devnull, 'w') as devnull:
            result = os.system(f"{cmd} > {os.devnull} 2>&1")
        if result != 0:
            raise RuntimeError(f"Failed to generate SSL certificates for {node_id} on port {port}")
        logger.info(f"Generated self-signed certificates: {cert_path}, {key_path}")
    else:
        logger.debug(f"Using existing SSL certificates: {cert_path}, {key_path}")
    
    # Configure SSL context
    ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    try:
        ssl_context.load_cert_chain(certfile=cert_path, keyfile=key_path)
        logger.info(f"HTTPS enabled with certificates for {node_id} on port {port}")
    except Exception as e:
        logger.error(f"Failed to load SSL certificates: {e}")
        raise

    # Run Flask in a background thread
    def run_flask():
        from flask import cli
        cli.show_server_banner = lambda *x: None  # Suppress Flask banner
        app.run(host=host, port=port, ssl_context=ssl_context, threaded=True)

    try:
        # Start Flask in an executor
        executor = current_loop.run_in_executor(None, run_flask)
        logger.info(f"Flask server started on https://{host}:{port}")
        
        # Use provided shutdown_event or create a local one
        shutdown_event = shutdown_event or asyncio.Event()
        await shutdown_event.wait()  # Wait until signalled to stop
    except Exception as e:
        logger.error(f"Failed to start Flask app: {e}", exc_info=True)
        raise
    finally:
        await rotation_manager.stop()
        if 'executor' in locals():
            current_loop.call_soon_threadsafe(lambda: executor.cancel() if not executor.done() else None)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Shutting down gracefully")
    except Exception as e:
        logger.error(f"Startup failed: {e}", exc_info=True)