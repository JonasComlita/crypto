import asyncio
from flask import Flask, request, jsonify
from functools import wraps
import logging
from typing import Callable, Dict, Any
from key_rotation.core import KeyRotationManager
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

logger = logging.getLogger(__name__)

def create_rotation_api(app: Flask, rotation_manager: 'KeyRotationManager') -> None:
    """Create API endpoints for the key rotation system."""
    limiter = Limiter(app=app, key_func=get_remote_address, default_limits=["100 per minute"])

    @app.errorhandler(Exception)
    async def handle_exception(e: Exception) -> tuple[Dict[str, str], int]:
        logger.error(f"Unhandled exception: {e}", exc_info=True)
        return jsonify({"error": "Internal server error", "message": str(e)}), 500

    def require_auth(f: Callable) -> Callable:
        @wraps(f)
        async def decorated(*args: Any, **kwargs: Any) -> tuple[Dict[str, str], int]:
            auth_header = request.headers.get("Authorization")
            if not auth_header or not auth_header.startswith("Bearer "):
                return jsonify({"error": "Invalid or missing Authorization header"}), 401
            token = auth_header.split(" ")[1]
            if not await rotation_manager.authenticate_peer(token):
                logger.warning(f"Authentication failed for token: {token[:10]}...")
                return jsonify({"error": "Authentication failed"}), 401
            return await f(*args, **kwargs)
        return decorated

    @app.route("/api/v1/node/info", methods=["GET"])
    @limiter.limit("10 per minute")
    async def get_node_info() -> tuple[Dict[str, Any], int]:
        return jsonify({
            "node_id": rotation_manager.node_id,
            "is_validator": rotation_manager.is_validator,
            "certificate": rotation_manager.pki.get_certificate_pem(),
            "public_key": rotation_manager.pki.get_public_key_pem()
        }), 200

    @app.route("/api/v1/nodes/register", methods=["POST"])
    @require_auth
    @limiter.limit("5 per minute")
    async def register_node() -> tuple[Dict[str, str], int]:
        data = request.get_json()
        required = ["node_id", "node_url", "public_key", "certificate"]
        if not data or not all(k in data for k in required):
            return jsonify({"error": f"Missing required fields: {', '.join(required)}"}), 400
        success = await rotation_manager.node_registry.register_node(
            data["node_id"], data["node_url"], data["public_key"], data["certificate"]
        )
        if success:
            logger.info(f"Node registered: {data['node_id']}")
        return jsonify({"status": "success" if success else "error"}), 200 if success else 500

    @app.route("/api/v1/nodes", methods=["GET"])
    @require_auth
    async def get_nodes() -> tuple[Dict[str, Any], int]:
        nodes = await rotation_manager.node_registry.get_all_nodes()
        return jsonify({"nodes": nodes}), 200

    @app.route("/api/v1/rotation/proposals", methods=["GET"])
    @require_auth
    async def get_proposals() -> tuple[Dict[str, Any], int]:
        proposals = await rotation_manager.consensus.get_active_proposals()
        return jsonify({"proposals": proposals}), 200

    @app.route("/api/v1/rotation/propose", methods=["POST"])
    @require_auth
    @limiter.limit("2 per hour")
    async def propose_rotation() -> tuple[Dict[str, Any], int]:
        if not rotation_manager.is_validator:
            return jsonify({"error": "Only validators can propose rotations"}), 403
        new_key = await rotation_manager.generate_secure_secret()
        key_hash = rotation_manager.hash_secret(new_key)
        proposal_id = await rotation_manager.propose_key_rotation(new_key, key_hash)
        if proposal_id:
            await rotation_manager.p2p.broadcast_proposal(proposal_id)
            logger.info(f"Proposed key rotation: {proposal_id}")
            return jsonify({"status": "success", "proposal_id": proposal_id}), 200
        return jsonify({"error": "Failed to create proposal"}), 500

    @app.route("/api/v1/rotation/vote", methods=["POST"])
    @require_auth
    async def vote_on_rotation() -> tuple[Dict[str, str], int]:
        if not rotation_manager.is_validator:
            return jsonify({"error": "Only validators can vote"}), 403
        data = request.get_json()
        if not data or "proposal_id" not in data or "approve" not in data:
            return jsonify({"error": "Missing proposal_id or approve"}), 400
        success = await rotation_manager.consensus.vote_on_proposal(data["proposal_id"], data["approve"])
        if success:
            await rotation_manager.p2p.broadcast_vote(data["proposal_id"], data["approve"])
            logger.info(f"Voted on proposal {data['proposal_id']}: {data['approve']}")
        return jsonify({"status": "success" if success else "error"}), 200 if success else 500

    @app.route("/api/v1/rotation/status/<proposal_id>", methods=["GET"])
    @require_auth
    async def get_proposal_status(proposal_id: str) -> tuple[Dict[str, Any], int]:
        status = await rotation_manager.consensus.check_proposal_status(proposal_id)
        return jsonify(status), 200 if "error" not in status else 404

    @app.route("/api/v1/rotation/finalize", methods=["POST"])
    @require_auth
    async def finalize_rotation() -> tuple[Dict[str, Any], int]:
        data = request.get_json()
        if not data or "proposal_id" not in data:
            return jsonify({"error": "Missing proposal_id"}), 400
        success, key_hash = await rotation_manager.consensus.finalize_proposal(data["proposal_id"])
        if success and data["proposal_id"] == rotation_manager.pending_proposal_id:
            await rotation_manager.apply_key_rotation()
            await rotation_manager.distribute_finalized_key()
            logger.info(f"Finalized rotation for proposal {data['proposal_id']}")
        return jsonify({"status": "success" if success else "error", "key_hash": key_hash or ""}), 200 if success else 500

    @app.route("/api/v1/p2p/message", methods=["POST"])
    async def receive_p2p_message() -> tuple[Dict[str, str], int]:
        data = request.get_json()
        if not data or "message" not in data or "signature" not in data or "sender" not in data.get("message", {}):
            return jsonify({"error": "Invalid message format"}), 400
        success = await rotation_manager.p2p.process_message(data["message"], data["signature"], data["message"]["sender"])
        return jsonify({"status": "success" if success else "error"}), 200 if success else 400

    @app.route("/api/v1/rotation/receive-key", methods=["POST"])
    async def receive_key() -> tuple[Dict[str, str], int]:
        data = request.get_json()
        if not data or "encrypted_key" not in data:
            return jsonify({"error": "Missing encrypted_key"}), 400
        success = await rotation_manager.receive_key(data["encrypted_key"])
        return jsonify({"status": "success" if success else "error"}), 200 if success else 500

    @app.route("/api/v1/auth/secret", methods=["GET"])
    @require_auth
    async def get_auth_secret() -> tuple[Dict[str, str], int]:
        return jsonify({"secret": await rotation_manager.get_current_auth_secret()}), 200

    @app.route("/api/v1/auth/validate", methods=["POST"])
    async def validate_auth() -> tuple[Dict[str, str], int]:
        data = request.get_json()
        if not data or "token" not in data:
            return jsonify({"error": "Missing token"}), 400
        return jsonify({"status": "valid" if await rotation_manager.authenticate_peer(data["token"]) else "invalid"}), 200

    @app.route("/api/v1/nodes/debug", methods=["GET"])
    @require_auth
    async def debug_nodes() -> tuple[Dict[str, Any], int]:
        nodes = await rotation_manager.node_registry.get_all_nodes()
        active_nodes = await rotation_manager.node_registry.get_active_nodes()
        return jsonify({"total_nodes": len(nodes), "active_nodes": len(active_nodes), "nodes": nodes, "active": active_nodes}), 200