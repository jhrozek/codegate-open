from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, Type, Union

import structlog
from google.protobuf.json_format import MessageToDict
from litellm import ChatCompletionRequest

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

class ProtoMessageHandler:
    GRPC_PREFIX_LENGTH = 4
    CONNECT_PREFIX_LENGTH = 1
    TOTAL_PREFIX_LENGTH = GRPC_PREFIX_LENGTH + CONNECT_PREFIX_LENGTH

    @staticmethod
    def decode_raw(raw_message: bytes, message_class) -> bytes:
        logger.debug(f"Original length: {len(raw_message)}")

        message_data = raw_message[ProtoMessageHandler.TOTAL_PREFIX_LENGTH:]
        logger.debug(f"message length: {len(message_data)}")
        logger.debug(f"message: {message_data}")

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
    }

    def __init__(self):
        # Initialize normalizers for each method
        self.normalizers = {
            method: config.create_normalizer()
            for method, config in self.message_config.items()
        }

    def decode_by_method(self, method: str, raw_message: bytes) -> Optional[ChatCompletionRequest]:
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
            proto_message: Optional[CursorMessage] = ProtoMessageHandler.decode_raw(raw_message, config.proto_class)

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
