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
import signal
from werkzeug.serving import make_server
from blockchain import Blockchain

logger = logging.getLogger(__name__)

class FlaskServer:
    def __init__(self, app: Flask, host: str, port: int, ssl_context):
        self.server = make_server(host, port, app, ssl_context=ssl_context)
        self.loop = None

    async def start(self):
        self.loop = asyncio.get_running_loop()
        await self.loop.run_in_executor(None, self.server.serve_forever)

    async def stop(self):
        self.server.shutdown()
        await asyncio.sleep(1)

async def main(node_id: Optional[str] = None, is_validator: bool = False,
              port: Optional[int] = None, host: str = "127.0.0.1", loop=None,
              shutdown_event: Optional[asyncio.Event] = None, blockchain: Optional[Blockchain] = None) -> None:
    load_dotenv()
    node_id = node_id or os.getenv("NODE_ID") or str(uuid.uuid4())
    is_validator = is_validator or os.getenv("IS_VALIDATOR", "false").lower() == "true"
    port = port or int(os.getenv("KEY_ROTATION_PORT", "5000"))

    logger.info(f"Starting node {node_id}, validator: {is_validator}, host: {host}, port: {port}")

    rotation_manager = KeyRotationManager(node_id=node_id, is_validator=is_validator, blockchain=blockchain)
    await rotation_manager.start()

    app = Flask(__name__)
    create_rotation_api(app, rotation_manager)

    # Dynamically construct cert_path and key_path based on node_id
    certs_dir = "data/certs"  # Adjust if your certs folder is elsewhere
    cert_path = os.path.join(certs_dir, f"node-{node_id}.crt")
    key_path = os.path.join(certs_dir, f"node-{node_id}.key")

    # Check if the dynamically constructed paths exist
    if not (os.path.exists(cert_path) and os.path.exists(key_path)):
        # Allow override with CERT_PATH and KEY_PATH if set in .env
        cert_path = os.getenv("CERT_PATH", cert_path)
        key_path = os.getenv("KEY_PATH", key_path)

        # If still not found, fall back to generating self-signed certificates
        if not (os.path.exists(cert_path) and os.path.exists(key_path)):
            logger.warning(f"Certificate files for node {node_id} not found at {cert_path} and {key_path}, generating self-signed certificates (not recommended for production)")
            cert_path = os.path.join(certs_dir, f"selfsigned-{node_id}_{port}.crt")
            key_path = os.path.join(certs_dir, f"selfsigned-{node_id}_{port}.key")
            os.makedirs(certs_dir, exist_ok=True)
            cmd = f'openssl req -x509 -newkey rsa:2048 -keyout "{key_path}" -out "{cert_path}" -days 365 -nodes -subj "/CN={node_id}"'
            if os.system(cmd) != 0:
                raise RuntimeError("Failed to generate SSL certificates")

    ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ssl_context.load_cert_chain(certfile=cert_path, keyfile=key_path)

    server = FlaskServer(app, host, port, ssl_context)
    server_task = asyncio.create_task(server.start())

    shutdown_event = shutdown_event or asyncio.Event()

    def signal_handler(sig, frame):
        logger.info(f"Received signal {sig}, shutting down...")
        shutdown_event.set()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        await shutdown_event.wait()
    finally:
        await server.stop()
        await rotation_manager.stop()
        server_task.cancel()
        try:
            await server_task
        except asyncio.CancelledError:
            pass
        logger.info("Key rotation service shut down")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        logger.error(f"Startup failed: {e}", exc_info=True)