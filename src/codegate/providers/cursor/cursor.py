import gzip
import struct
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, Flag
from io import BytesIO
from typing import Any, Dict, Optional, Type, Union

import structlog
from google.protobuf.json_format import MessageToDict
from litellm import ChatCompletionRequest
from protobuf_inspector.types import StandardParser

import codegate.providers.cursor.cursor_pb2 as cursorpb
from codegate.providers.normalizer.base import ModelInputNormalizer

logger = structlog.get_logger("codegate").bind(origin="cursor_provider")

# TODO: Expand? Or drop since we don't want to keep maintaining this?
HandledCursorType = Union[
    cursorpb.GetChatRequest,
    cursorpb.StreamCppRequest,
    cursorpb.FSUploadFileRequest,
    cursorpb.FSIsEnabledForUserRequest,
]

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

class EnvelopeFlags(Flag):
    """
    Connect protocol envelope flags
    """
    NONE = 0
    COMPRESSED = 0b00000001
    END_STREAM = 0b00000010


class ContentDecompressor(ABC):
    """
    Abstract base class for content decompression strategies
    """

    @abstractmethod
    def decompress(self, data: bytes) -> bytes:
        """Decompress the given bytes"""
        pass


class GzipDecompressor(ContentDecompressor):
    """Handles gzip compressed content"""

    def decompress(self, data: bytes) -> bytes:
        logger.debug("Decompressing gzip content")
        return gzip.decompress(data)


class PlainTextDecompressor(ContentDecompressor):
    """Passes through uncompressed content"""

    def decompress(self, data: bytes) -> bytes:
        logger.debug("Using plaintext pass-through")
        return data

@dataclass
class DecodedMessage:
    """Container for decoded message data"""
    payload: bytes
    details: Dict[str, Any]

class ProtoDecoder(ABC):
    """Abstract base class for protobuf message decoders"""

    def __init__(self):
        self.decompressors = {
            'gzip': GzipDecompressor(),
            'identity': PlainTextDecompressor(),
        }

    @abstractmethod
    def decode(self, headers: Dict[str, str], raw_message: bytes) -> DecodedMessage:
        """Decode raw message bytes into a protobuf message"""
        pass

    def _get_decompressor(self, headers: Dict[str, str]) -> ContentDecompressor:
        """Get appropriate decompressor based on content-encoding header"""
        encoding = headers.get('connect-content-encoding', 'identity')
        return self.decompressors.get(encoding, PlainTextDecompressor())

    def decompress_if_needed(self, headers: Dict[str, str], data: bytes) -> bytes:
        """Handle content decompression using appropriate strategy"""
        decompressor = self._get_decompressor(headers)
        return decompressor.decompress(data)


class ConnectDecoder(ProtoDecoder):
    """Decoder for Connect protocol messages with envelope framing"""

    ENVELOPE_HEADER_LENGTH = 5
    ENVELOPE_HEADER_PACK = ">BI"  # big-endian unsigned char + unsigned int

    def decode(self, headers: Dict[str, str], raw_message: bytes) -> DecodedMessage:
        """Decode a Connect protocol message with envelope framing"""
        if len(raw_message) < self.ENVELOPE_HEADER_LENGTH:
            raise ValueError(f"Message too short. Need at least {self.ENVELOPE_HEADER_LENGTH} bytes")

        # Decode envelope header
        flags_byte, payload_length = struct.unpack(
            self.ENVELOPE_HEADER_PACK,
            raw_message[:self.ENVELOPE_HEADER_LENGTH]
        )

        # Validate flags
        try:
            flags = EnvelopeFlags(flags_byte)
        except ValueError:
            raise ValueError(f"Invalid flags byte: {flags_byte}")

        # Check total message length
        expected_length = self.ENVELOPE_HEADER_LENGTH + payload_length
        if len(raw_message) != expected_length:
            raise ValueError(
                f"Message length mismatch. Expected {expected_length}, got {len(raw_message)}"
            )

        # Extract payload
        payload = raw_message[self.ENVELOPE_HEADER_LENGTH:]

        # Handle compression if needed
        if flags & EnvelopeFlags.COMPRESSED:
            payload = self.decompress_if_needed(headers, payload)

        return DecodedMessage(
            payload=payload,
            details={
                "flags": {
                    "raw": flags_byte,
                    "compressed": bool(flags & EnvelopeFlags.COMPRESSED),
                    "end_stream": bool(flags & EnvelopeFlags.END_STREAM)
                },
                "payload_length": payload_length,
                "total_length": expected_length
            }
        )

class RawProtoDecoder(ProtoDecoder):
    """Decoder for raw protobuf messages without envelope framing"""

    def decode(self, headers: Dict[str, str], raw_message: bytes) -> DecodedMessage:
        """Decode a raw protobuf message"""
        payload = self.decompress_if_needed(headers, raw_message)
        return DecodedMessage(
            payload=payload,
            details={
                "length": len(payload)
            }
        )

class ProtoMessageDecoder:
    """Factory class to get appropriate decoder based on message type"""

    def __init__(self):
        self.connect_decoder = ConnectDecoder()
        self.raw_decoder = RawProtoDecoder()

    def _get_decoder(self, headers: Dict[str, str]) -> ProtoDecoder:
        """Get the appropriate decoder based on headers"""
        if headers.get("content-type", None) == "application/connect+proto":
            logger.debug("Using Connect protocol decoder")
            return self.connect_decoder
        logger.debug("Using raw protobuf decoder")
        return self.raw_decoder

    def decode_message(self, headers: Dict[str, str], raw_message: bytes) -> Optional[DecodedMessage]:
        """Decode a message using the appropriate decoder"""
        try:
            decoder = self._get_decoder(headers)
            return decoder.decode(headers, raw_message)
        except Exception as e:
            logger.error(f"Failed to decode message: {str(e)}")
            return None


class ProtoMarshaler(ABC):
    """Abstract base class for protobuf message marshalers"""

    def __init__(self):
        self.compressors = {
            'gzip': lambda x: gzip.compress(x),
            'identity': lambda x: x,
        }

    @abstractmethod
    def marshal(self, message: HandledCursorType, headers: Dict[str, str]) -> bytes:
        """Marshal message to bytes format"""
        pass

    def _get_compressor(self, headers: Dict[str, str]):
        """Get appropriate compressor based on content-encoding header"""
        encoding = headers.get('connect-content-encoding', 'identity')
        return self.compressors.get(encoding, lambda x: x)


class ConnectMarshaler(ProtoMarshaler):
    """Marshaler for Connect protocol messages with envelope framing"""

    ENVELOPE_HEADER_PACK = ">BI"

    def marshal(self, message: HandledCursorType, headers: Dict[str, str]) -> bytes:
        serialized = message.SerializeToString()
        compressor = self._get_compressor(headers)
        payload = compressor(serialized)

        flags = EnvelopeFlags.NONE
        if headers.get('connect-content-encoding') == 'gzip':
            flags |= EnvelopeFlags.COMPRESSED
        flags |= EnvelopeFlags.END_STREAM  # Configurable if needed

        header = struct.pack(self.ENVELOPE_HEADER_PACK, flags.value, len(payload))
        return header + payload


class RawProtoMarshaler(ProtoMarshaler):
    """Marshaler for raw protobuf messages"""

    def marshal(self, message: HandledCursorType, headers: Dict[str, str]) -> bytes:
        serialized = message.SerializeToString()
        compressor = self._get_compressor(headers)
        return compressor(serialized)


class ProtoMessageMarshaler:
    """Factory class to get appropriate marshaler based on message type"""

    def __init__(self):
        self.connect_marshaler = ConnectMarshaler()
        self.raw_marshaler = RawProtoMarshaler()

    def _get_marshaler(self, headers: Dict[str, str]) -> ProtoMarshaler:
        if headers.get("content-type") == "application/connect+proto":
            return self.connect_marshaler
        return self.raw_marshaler

    def marshal_message(self, message: HandledCursorType, headers: Dict[str, str]) -> bytes:
        marshaler = self._get_marshaler(headers)
        return marshaler.marshal(message, headers)

class ProtoMessageHandler:
    def _dump_message(self, message):
        try:
            parser = StandardParser()
            parsed = parser.parse_message(BytesIO(message), "message")
            logger.debug(parsed)
        except Exception as e:
            logger.error(f"Failed to parse message: {str(e)}")

    @staticmethod
    def decode(
            headers: Dict[str, str],
            raw_message: bytes,
            message_class,
    ) -> bytes:
        logger.debug(f"Original length: {len(raw_message)}")
        logger.debug(f"Original message: {raw_message}")
        logger.debug(f"Headers: {headers}")

        proto_message = ProtoMessageDecoder().decode_message(headers, raw_message)
        if not proto_message:
            return None

        ProtoMessageHandler()._dump_message(proto_message.payload)

        message = message_class()
        message.ParseFromString(proto_message.payload)
        return message

    @staticmethod
    def encode(
            headers: Dict[str, str],
            message: HandledCursorType,
    ) -> bytes:
        return ProtoMessageMarshaler().marshal_message(message, headers)

class CursorMethod(str, Enum):
    STREAM_CHAT = "/aiserver.v1.AiService/StreamChat"
    STREAM_CPP = "/aiserver.v1.AiService/StreamCpp"
    FS_UPLOAD_FILE = "/aiserver.v1.FileSyncService/FSUploadFile"
    FSI_IS_ENABLED = "/aiserver.v1.FileSyncService/FSIsEnabledForUser"


@dataclass
class CursorMessageConfig:
    """Configuration for a specific Cursor message type"""
    proto_class: Type[HandledCursorType]
    normalizer_class: Type[ModelInputNormalizer]

    def create_normalizer(self) -> ModelInputNormalizer:
        """Create a new instance of the normalizer"""
        return self.normalizer_class()


class CursorProvider:
    """
    This is the intended flow:
    1. The raw message body is received along with the headers and the method
    2. If we don't handle the method, we just pass the data through.
    3. For any handled methods, we try to parse the message into a native Python type
        i) We first decode the message using the appropriate decoder. In order to decode it, we need to grab the raw protobuf message
           but that message might be compressed or wrapped in a connect protocol envelope.
            a) depending on the headers, we choose either a raw protobuf decoder or a connect protocol decoder.
               The connect protocol decoder would handle the envelope framing and potential decompression of the message.
               /inside the envelope/
            b) We then parse the raw protobuf message into a native Python type
        ii) We then convert the message into the native type
    4. (not implemented yet) We then normalize the message into a ChatCompletionRequest so it can be passed into a pipeline
    5. (not implemented yet) The pipeline will then return a ChatCompletionResponse which we will denormalize back into the original message format
       by packing into protobuf. We need to remember to update the headers in case we decompress a message that was originally compressed.
       or change the content length if we change the message size.
    """
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
        CursorMethod.FSI_IS_ENABLED: CursorMessageConfig(
            proto_class=cursorpb.FSIsEnabledForUserRequest,
            normalizer_class=FSUploadFileNormalizer,
        ),
    }

    def __init__(self):
        # Initialize normalizers for each method
        self.normalizers = {
            method: config.create_normalizer()
            for method, config in self.message_config.items()
        }

    def process_request(
            self,
            method: str,
            headers: Dict[str, str],
            raw_message: bytes,
    ) -> bytes:
        proto_native_message = self.decode(method, headers, raw_message)
        if not proto_native_message:
            # this means the method was not handled
            return raw_message

        if isinstance(proto_native_message, cursorpb.GetChatRequest):
            cursor_message = proto_native_message
            print("--------------------")
            print(cursor_message)
            print("--------------------")

        # TODO: Normalize the message and run the pipeline

        return self.encode(method, headers, proto_native_message)

    def encode(
            self,
            method: str,
            headers: Dict[str, str],
            raw_message: HandledCursorType,
    ) -> Optional[bytes]:
        logger.debug(f"Encoding message for method: {method}")
        return ProtoMessageHandler.encode(headers, raw_message)


    def decode(
            self,
            method: str,
            headers: Dict[str, str],
            raw_message: bytes,
    ) -> Optional[HandledCursorType]:
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
            proto_message: Optional[HandledCursorType] = ProtoMessageHandler.decode(headers, raw_message, config.proto_class)
            if not proto_message:
                return None

            # Convert proto message to dict for the normalizer
            dict_message = MessageToDict(
                proto_message,
                preserving_proto_field_name=True
            )
            logger.debug(f"Decoded message: {dict_message}")

            return proto_message
        except Exception as e:
            logger.error(f"Failed to decode message: {str(e)}")
            return None
