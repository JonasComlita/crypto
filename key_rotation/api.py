import asyncio
from flask import Flask, request, jsonify
from functools import wraps
import logging
from typing import Callable, Dict, Any
from key_rotation.core import KeyRotationManager

logger = logging.getLogger(__name__)

def create_rotation_api(app: Flask, rotation_manager: 'KeyRotationManager') -> None:
    """Create API endpoints for the key rotation system."""
    
    @app.errorhandler(Exception)
    async def handle_exception(e: Exception) -> tuple[Dict[str, str], int]:
        logger.error(f"Unhandled exception: {e}", exc_info=True)
        return jsonify({"error": "Internal server error", "message": str(e)}), 500

    def require_auth(f: Callable) -> Callable:
        @wraps(f)
        async def decorated(*args: Any, **kwargs: Any) -> tuple[Dict[str, str], int]:
            auth_header = request.headers.get("Authorization")
            if not auth_header:
                return jsonify({"error": "Missing Authorization header"}), 401
            
            parts = auth_header.split()
            if len(parts) != 2 or parts[0].lower() != "bearer":
                return jsonify({"error": "Invalid Authorization header format"}), 401
            
            token = parts[1]
            if not await rotation_manager.authenticate_peer(token):
                return jsonify({"error": "Authentication failed"}), 401
            
            return await f(*args, **kwargs)
        return decorated

    @app.route("/api/v1/node/info", methods=["GET"])
    async def get_node_info() -> tuple[Dict[str, Any], int]:
        """Get this node's information."""
        try:
            return jsonify({
                "node_id": rotation_manager.node_id,
                "is_validator": rotation_manager.is_validator,
                "certificate": rotation_manager.pki.get_certificate_pem(),
                "public_key": rotation_manager.pki.get_public_key_pem()
            }), 200
        except Exception as e:
            logger.error(f"Failed to get node info: {e}")
            return jsonify({"error": "Failed to retrieve node info"}), 500

    @app.route("/api/v1/nodes/register", methods=["POST"])
    @require_auth
    async def register_node() -> tuple[Dict[str, str], int]:
        """Register a new node."""
        try:
            data = request.get_json()
            if not data or not all(k in data for k in ["node_id", "node_url", "public_key", "certificate"]):
                return jsonify({"error": "Missing required fields"}), 400
            
            success = await rotation_manager.node_registry.register_node(
                data["node_id"], data["node_url"], data["public_key"], data["certificate"]
            )
            return jsonify({"status": "success"}) if success else jsonify({"error": "Failed to register node"}), 200 if success else 500
        except Exception as e:
            logger.error(f"Node registration failed: {e}")
            return jsonify({"error": "Internal error during registration"}), 500

    @app.route("/api/v1/nodes", methods=["GET"])
    @require_auth
    async def get_nodes() -> tuple[Dict[str, Any], int]:
        """Get all registered nodes."""
        try:
            nodes = await rotation_manager.node_registry.get_all_nodes()
            return jsonify({"nodes": nodes}), 200
        except Exception as e:
            logger.error(f"Failed to get nodes: {e}")
            return jsonify({"error": "Failed to retrieve nodes"}), 500

    @app.route("/api/v1/rotation/proposals", methods=["GET"])
    @require_auth
    async def get_proposals() -> tuple[Dict[str, Any], int]:
        """Get active key rotation proposals."""
        try:
            proposals = await rotation_manager.consensus.get_active_proposals()
            return jsonify({"proposals": proposals}), 200
        except Exception as e:
            logger.error(f"Failed to get proposals: {e}")
            return jsonify({"error": "Failed to retrieve proposals"}), 500

    @app.route("/api/v1/rotation/propose", methods=["POST"])
    @require_auth
    async def propose_rotation() -> tuple[Dict[str, Any], int]:
        """Propose a new key rotation."""
        try:
            if not rotation_manager.is_validator:
                return jsonify({"error": "Only validators can propose rotations"}), 403
            
            new_key = await rotation_manager.generate_secure_secret()
            key_hash = rotation_manager.hash_secret(new_key)
            proposal_id = await rotation_manager.propose_key_rotation(new_key, key_hash)
            
            if proposal_id:
                await rotation_manager.p2p.broadcast_proposal(proposal_id)
                return jsonify({"status": "success", "proposal_id": proposal_id}), 200
            return jsonify({"error": "Failed to create proposal"}), 500
        except Exception as e:
            logger.error(f"Propose rotation failed: {e}", exc_info=True)
            return jsonify({"error": f"Server error: {str(e)}"}), 500

    @app.route("/api/v1/rotation/vote", methods=["POST"])
    @require_auth
    async def vote_on_rotation() -> tuple[Dict[str, str], int]:
        """Vote on a key rotation proposal."""
        try:
            data = request.get_json()
            if not data or "proposal_id" not in data or "approve" not in data:
                return jsonify({"error": "Missing required fields"}), 400
            
            if not rotation_manager.is_validator:
                return jsonify({"error": "Only validators can vote"}), 403
            
            proposal_id, approve = data["proposal_id"], data["approve"]
            success = await rotation_manager.consensus.vote_on_proposal(proposal_id, approve)
            if success:
                await rotation_manager.p2p.broadcast_vote(proposal_id, approve)
                return jsonify({"status": "success"}), 200
            return jsonify({"error": "Failed to record vote"}), 500
        except Exception as e:
            logger.error(f"Vote on rotation failed: {e}")
            return jsonify({"error": "Internal error during voting"}), 500

    @app.route("/api/v1/rotation/status/<proposal_id>", methods=["GET"])
    @require_auth
    async def get_proposal_status(proposal_id: str) -> tuple[Dict[str, Any], int]:
        """Get the status of a specific proposal."""
        try:
            status = await rotation_manager.consensus.check_proposal_status(proposal_id)
            if "error" in status:
                return jsonify({"error": status["error"]}), 404
            return jsonify(status), 200
        except Exception as e:
            logger.error(f"Failed to get proposal status: {e}")
            return jsonify({"error": "Failed to retrieve proposal status"}), 500

    @app.route("/api/v1/rotation/finalize", methods=["POST"])
    @require_auth
    async def finalize_rotation() -> tuple[Dict[str, Any], int]:
        """Finalize an approved key rotation."""
        try:
            data = request.get_json()
            if not data or "proposal_id" not in data:
                return jsonify({"error": "Missing proposal_id"}), 400
            
            proposal_id = data["proposal_id"]
            success, key_hash = await rotation_manager.consensus.finalize_proposal(proposal_id)
            if success:
                if proposal_id == rotation_manager.pending_proposal_id:
                    await rotation_manager.apply_key_rotation()
                    await rotation_manager.distribute_finalized_key()
                return jsonify({"status": "success", "key_hash": key_hash}), 200
            return jsonify({"error": "Failed to finalize proposal"}), 500
        except Exception as e:
            logger.error(f"Finalize rotation failed: {e}")
            return jsonify({"error": "Internal error during finalization"}), 500

    @app.route("/api/v1/p2p/message", methods=["POST"])
    async def receive_p2p_message() -> tuple[Dict[str, str], int]:
        """Receive P2P messages from other nodes."""
        try:
            data = request.get_json()
            if not data or "message" not in data or "signature" not in data:
                return jsonify({"error": "Invalid message format"}), 400
            
            message, signature = data["message"], data["signature"]
            if "sender" not in message:
                return jsonify({"error": "Invalid message: missing sender"}), 400
            
            success = await rotation_manager.p2p.process_message(message, signature, message["sender"])
            return jsonify({"status": "success"}) if success else jsonify({"error": "Failed to process message"}), 200 if success else 400
        except Exception as e:
            logger.error(f"Failed to process P2P message: {e}")
            return jsonify({"error": "Internal error processing message"}), 500

    @app.route("/api/v1/rotation/receive-key", methods=["POST"])
    async def receive_key() -> tuple[Dict[str, str], int]:
        """Receive a new encrypted key."""
        try:
            data = request.get_json()
            if not data or "encrypted_key" not in data:
                return jsonify({"error": "Missing encrypted_key"}), 400
            
            success = await rotation_manager.receive_key(data["encrypted_key"])
            return jsonify({"status": "success"}) if success else jsonify({"error": "Failed to process received key"}), 200 if success else 500
        except Exception as e:
            logger.error(f"Failed to receive key: {e}")
            return jsonify({"error": "Internal error receiving key"}), 500

    @app.route("/api/v1/auth/secret", methods=["GET"])
    @require_auth
    async def get_auth_secret() -> tuple[Dict[str, str], int]:
        """Get the current auth secret."""
        try:
            return jsonify({"secret": await rotation_manager.get_current_auth_secret()}), 200
        except Exception as e:
            logger.error(f"Failed to get auth secret: {e}")
            return jsonify({"error": "Failed to retrieve auth secret"}), 500

    @app.route("/api/v1/auth/validate", methods=["POST"])
    async def validate_auth() -> tuple[Dict[str, str], int]:
        """Validate an auth token."""
        try:
            data = request.get_json()
            if not data or "token" not in data:
                return jsonify({"error": "Missing token"}), 400
            
            if await rotation_manager.authenticate_peer(data["token"]):
                return jsonify({"status": "valid"}), 200
            return jsonify({"status": "invalid"}), 401
        except Exception as e:
            logger.error(f"Failed to validate auth: {e}")
            return jsonify({"error": "Internal error validating token"}), 500

    @app.route("/api/v1/nodes/debug", methods=["GET"])
    async def debug_nodes() -> tuple[Dict[str, Any], int]:
        """Get debugging info about registered nodes."""
        try:
            nodes = await rotation_manager.node_registry.get_all_nodes()
            active_nodes = await rotation_manager.node_registry.get_active_nodes()
            return jsonify({
                "total_nodes": len(nodes),
                "active_nodes": len(active_nodes),
                "nodes": nodes,
                "active": active_nodes
            }), 200
        except Exception as e:
            logger.error(f"Failed to debug nodes: {e}")
            return jsonify({"error": "Failed to retrieve node debug info"}), 500