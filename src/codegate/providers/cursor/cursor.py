import gzip
from dataclasses import dataclass
from enum import Enum
from io import BytesIO
from typing import Dict, Optional, Type, Union

import structlog
from google.protobuf.json_format import MessageToDict
from litellm import ChatCompletionRequest
from protobuf_inspector.types import StandardParser

import codegate.providers.cursor.cursor_pb2 as cursorpb
from codegate.providers.normalizer.base import ModelInputNormalizer

logger = structlog.get_logger("codegate").bind(origin="cursor_provider")

HandledCursorType = Union[
    cursorpb.GetChatRequest,
    cursorpb.StreamCppRequest,
]

# Type for all possible Cursor message types
CursorMessage = Union[cursorpb.GetChatRequest, cursorpb.StreamCppRequest]

class StreamChatNormalizer(ModelInputNormalizer):
    def normalize(self, data: cursorpb.GetChatRequest) -> ChatCompletionRequest:
        """
        Normalize Cursor's GetChatRequest messages into ChatCompletionRequest format.
        """
        return None

    def denormalize(self, data: ChatCompletionRequest) -> cursorpb.GetChatRequest:
        """
        Convert ChatCompletionRequest back to Cursor's GetChatRequest format.
        """
        return None


class StreamCppNormalizer(ModelInputNormalizer):
    def normalize(self, data: Dict) -> ChatCompletionRequest:
        """
        Normalize Cursor's StreamCpp messages into ChatCompletionRequest format.

        Args:
            data: Dictionary from StreamCppRequest containing a 'code' field
        """
        return None

    def denormalize(self, data: ChatCompletionRequest) -> Dict:
        """Convert ChatCompletionRequest back to Cursor's StreamCpp format."""
        return None

class FSUploadFileNormalizer(ModelInputNormalizer):
    def normalize(self, data: Dict) -> ChatCompletionRequest:
        """
        Normalize Cursor's StreamCpp messages into ChatCompletionRequest format.

        Args:
            data: Dictionary from StreamCppRequest containing a 'code' field
        """
        return None

    def denormalize(self, data: ChatCompletionRequest) -> Dict:
        """Convert ChatCompletionRequest back to Cursor's StreamCpp format."""
        return None

class ProtoMessageHandler:
    GRPC_PREFIX_LENGTH = 4
    CONNECT_PREFIX_LENGTH = 1
    TOTAL_PREFIX_LENGTH = GRPC_PREFIX_LENGTH + CONNECT_PREFIX_LENGTH

    @staticmethod
    def is_connect_frame(data: bytes) -> tuple[bool, dict]:
        """
        Check if a byte sequence starts with Connect protocol framing.
        Returns (is_connect, details) where details contains compression flag and length if found.

        Connect frame format:
        - 1 byte compression flag (0 or 1)
        - 4 bytes message length (big-endian uint32)
        - Payload with exactly the length specified
        """
        if len(data) < 5:  # Need at least 5 bytes for Connect framing
            return False, {"error": "Message too short"}

        compression_flag = data[0]
        if compression_flag not in (0, 1):
            return False, {"error": "Invalid compression flag"}

        # Extract 4-byte length (big-endian)
        length = int.from_bytes(data[1:5], byteorder='big')

        # Check if actual message length matches declared length
        expected_total = length + 5  # frame + payload
        if len(data) != expected_total:
            return False, {
                "error": "Message length mismatch",
                "expected": expected_total,
                "actual": len(data)
            }

        return True, {
            "compression_flag": compression_flag,
            "message_length": length,
            "total_length": expected_total
        }

    @staticmethod
    def decode_raw(headers: Dict[str, str], raw_message: bytes, message_class) -> bytes:
        logger.debug(f"Original length: {len(raw_message)}")
        logger.debug(f"Original message: {raw_message}")

        if headers.get("connect-protocol-version", None):
            logger.debug("Connect protocol detected")

            is_connect, details = ProtoMessageHandler.is_connect_frame(raw_message)
            if not is_connect:
                logger.error(f"Invalid Connect frame: {details}")
            else:
                logger.debug(f"Connect frame details: {details}")

            message_data = raw_message[ProtoMessageHandler.TOTAL_PREFIX_LENGTH:]
            logger.debug(f"message length: {len(message_data)}")
            logger.debug(f"message: {message_data}")
        else:
            message_data = raw_message

        if headers.get('connect-content-encoding', None) == 'gzip':
            logger.debug("Decompressing message")
            message_data = gzip.decompress(message_data)

        try:
            # Create parser and inspect message
            parser = StandardParser()
            parsed = parser.parse_message(BytesIO(message_data), "message")
            # Print the parsed structure
            print(parsed)
        except Exception as e:
            logger.error(f"Failed to parse message: {str(e)}")

        message = message_class()
        message.ParseFromString(message_data)
        return message

    @staticmethod
    def encode_with_prefixes(message) -> bytes:
        serialized = message.SerializeToString()
        length = len(serialized)

        grpc_prefix = length.to_bytes(4, 'big')
        connect_prefix = bytes([length & 0xFF])

        return grpc_prefix + connect_prefix + serialized

class CursorMethod(str, Enum):
    STREAM_CHAT = "/aiserver.v1.AiService/StreamChat"
    STREAM_CPP = "/aiserver.v1.AiService/StreamCpp"
    FS_UPLOAD_FILE = "/aiserver.v1.FileSyncService/FSUploadFile"


@dataclass
class CursorMessageConfig:
    """Configuration for a specific Cursor message type"""
    proto_class: Type[CursorMessage]
    normalizer_class: Type[ModelInputNormalizer]

    def create_normalizer(self) -> ModelInputNormalizer:
        """Create a new instance of the normalizer"""
        return self.normalizer_class()


class CursorProvider:
    message_config = {
        CursorMethod.STREAM_CHAT: CursorMessageConfig(
            proto_class=cursorpb.GetChatRequest,
            normalizer_class=StreamChatNormalizer,
        ),
        CursorMethod.STREAM_CPP: CursorMessageConfig(
            proto_class=cursorpb.StreamCppRequest,
            normalizer_class=StreamCppNormalizer,
        ),
        CursorMethod.FS_UPLOAD_FILE: CursorMessageConfig(
            proto_class=cursorpb.FSUploadFileRequest,
            normalizer_class=FSUploadFileNormalizer,
        ),
    }

    def __init__(self):
        # Initialize normalizers for each method
        self.normalizers = {
            method: config.create_normalizer()
            for method, config in self.message_config.items()
        }

    def decode_by_method(self, method: str, headers: Dict[str, str], raw_message: bytes) -> Optional[ChatCompletionRequest]:
        logger.debug(f"Decoding message for method: {method}")

        try:
            cursor_method = CursorMethod(method)
        except ValueError:
            logger.debug(f"Invalid method: {method}")
            return None

        config = self.message_config.get(cursor_method)
        if not config:
            logger.debug(f"Unhandled method: {method}")
            return None

        try:
            proto_message: Optional[CursorMessage] = ProtoMessageHandler.decode_raw(headers, raw_message, config.proto_class)

            if proto_message:
                # Convert proto message to dict for the normalizer
                dict_message = MessageToDict(
                    proto_message,
                    preserving_proto_field_name=True
                )
                logger.debug(f"Decoded message: {dict_message}")
                normalizer = self.normalizers[cursor_method]
                normalized =  normalizer.normalize(proto_message)
                return dict_message
            return None
        except Exception as e:
            logger.error(f"Failed to decode message: {str(e)}")
            return None
