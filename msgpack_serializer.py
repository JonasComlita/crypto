"""
MessagePack serialization for blockchain objects.
Provides efficient binary serialization with custom encoders/decoders.
"""

import msgpack
import base64
import logging
import datetime
from typing import Any, Dict, List, Optional, Tuple, Union, Callable

logger = logging.getLogger(__name__)

# Custom encoder for handling non-msgpack types
class BlockchainEncoder:
    """Custom encoder for blockchain objects to msgpack."""
    
    @staticmethod
    def encode(obj: Any) -> Dict[str, Any]:
        """
        Convert object to msgpack-compatible dictionary.
        
        Args:
            obj: Object to encode
            
        Returns:
            Dictionary representation
        """
        if hasattr(obj, 'to_dict'):
            # Use object's to_dict method if available
            return obj.to_dict()
        elif isinstance(obj, bytes):
            # Encode bytes as base64 string with marker
            return {'__bytes__': base64.b64encode(obj).decode('ascii')}
        elif isinstance(obj, datetime.datetime):
            # Encode datetime as ISO string with marker
            return {'__datetime__': obj.isoformat()}
        elif hasattr(obj, '__dict__'):
            # Generic object with attributes
            result = {'__class__': obj.__class__.__name__}
            result.update(obj.__dict__)
            return result
        else:
            # Try standard conversion
            return obj

# Custom decoder for handling encoded types
class BlockchainDecoder:
    """Custom decoder for blockchain objects from msgpack."""
    
    def __init__(self, blockchain_module=None):
        """
        Initialize decoder with optional blockchain module for reconstructing objects.
        
        Args:
            blockchain_module: Optional module containing blockchain classes
        """
        self.blockchain_module = blockchain_module
        
    def decode(self, obj: Dict[str, Any]) -> Any:
        """
        Convert msgpack dictionary back to original object.
        
        Args:
            obj: Dictionary to decode
            
        Returns:
            Decoded object
        """
        if not isinstance(obj, dict):
            return obj
            
        # Check for special encoded types
        if '__bytes__' in obj and len(obj) == 1:
            return base64.b64decode(obj['__bytes__'])
            
        if '__datetime__' in obj and len(obj) == 1:
            return datetime.datetime.fromisoformat(obj['__datetime__'])
            
        if '__class__' in obj and self.blockchain_module is not None:
            # Try to reconstruct class instance if we have the module
            class_name = obj.pop('__class__')
            if hasattr(self.blockchain_module, class_name):
                cls = getattr(self.blockchain_module, class_name)
                if hasattr(cls, 'from_dict'):
                    # Use from_dict class method if available
                    return cls.from_dict(obj)
                else:
                    # Try to create instance and set attributes
                    instance = cls.__new__(cls)
                    for key, value in obj.items():
                        setattr(instance, key, value)
                    return instance
            # Restore class name for fallback
            obj['__class__'] = class_name
            
        return obj

def serialize(obj: Any) -> bytes:
    """
    Serialize an object to MessagePack binary format.
    
    Args:
        obj: Python object to serialize
        
    Returns:
        MessagePack binary data
    """
    try:
        return msgpack.packb(obj, default=BlockchainEncoder.encode, use_bin_type=True)
    except Exception as e:
        logger.error(f"Serialization error: {e}")
        raise

def deserialize(data: bytes, blockchain_module=None) -> Any:
    """
    Deserialize MessagePack binary data to Python object.
    
    Args:
        data: MessagePack binary data
        blockchain_module: Optional module for class reconstruction
        
    Returns:
        Deserialized Python object
    """
    try:
        decoder = BlockchainDecoder(blockchain_module)
        return msgpack.unpackb(data, object_hook=decoder.decode, raw=False)
    except Exception as e:
        logger.error(f"Deserialization error: {e}")
        raise

def serialize_block(block) -> bytes:
    """
    Serialize a Block object to MessagePack.
    
    Args:
        block: Block object
        
    Returns:
        MessagePack binary data
    """
    if hasattr(block, 'to_dict'):
        return serialize(block.to_dict())
    else:
        return serialize(block)

def deserialize_block(data: bytes, blockchain_module=None):
    """
    Deserialize MessagePack data to a Block object.
    
    Args:
        data: MessagePack binary data
        blockchain_module: Optional module containing Block class
        
    Returns:
        Block object or dictionary
    """
    block_data = deserialize(data, blockchain_module)
    
    if blockchain_module and hasattr(blockchain_module, 'Block'):
        # Use the Block class from the provided module
        return blockchain_module.Block.from_dict(block_data)
    else:
        # Return as dictionary if no module provided
        return block_data

def serialize_transaction(transaction) -> bytes:
    """
    Serialize a Transaction object to MessagePack.
    
    Args:
        transaction: Transaction object
        
    Returns:
        MessagePack binary data
    """
    if hasattr(transaction, 'to_dict'):
        return serialize(transaction.to_dict())
    else:
        return serialize(transaction)

def deserialize_transaction(data: bytes, blockchain_module=None):
    """
    Deserialize MessagePack data to a Transaction object.
    
    Args:
        data: MessagePack binary data
        blockchain_module: Optional module containing Transaction class
        
    Returns:
        Transaction object or dictionary
    """
    tx_data = deserialize(data, blockchain_module)
    
    if blockchain_module and hasattr(blockchain_module, 'Transaction'):
        # Use the Transaction class from the provided module
        return blockchain_module.Transaction.from_dict(tx_data)
    else:
        # Return as dictionary if no module provided
        return tx_data

def deserialize_with_type(data: bytes, obj_type: str, blockchain_module=None):
    """
    Deserialize MessagePack data to a specific object type.
    
    Args:
        data: MessagePack binary data
        obj_type: String identifying object type ('block', 'transaction', etc.)
        blockchain_module: Optional module containing classes
        
    Returns:
        Deserialized object
    """
    if obj_type.lower() == 'block':
        return deserialize_block(data, blockchain_module)
    elif obj_type.lower() == 'transaction':
        return deserialize_transaction(data, blockchain_module)
    else:
        return deserialize(data, blockchain_module)
