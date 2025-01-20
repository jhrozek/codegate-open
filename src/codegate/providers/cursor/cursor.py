import structlog
from google.protobuf.json_format import MessageToDict

import codegate.providers.cursor.cursor_pb2 as cursorpb

logger = structlog.get_logger("codegate").bind(origin="cursor_provider")


class ProtoMessageHandler:
    GRPC_PREFIX_LENGTH = 4
    CONNECT_PREFIX_LENGTH = 1
    TOTAL_PREFIX_LENGTH = GRPC_PREFIX_LENGTH + CONNECT_PREFIX_LENGTH

    @staticmethod
    def decode_raw(raw_message: bytes, message_class) -> Any:
        message = message_class()
        message.ParseFromString(raw_message[ProtoMessageHandler.TOTAL_PREFIX_LENGTH:])
        return message

    @staticmethod
    def encode_with_prefixes(message) -> bytes:
        serialized = message.SerializeToString()
        length = len(serialized)

        grpc_prefix = length.to_bytes(4, 'big')
        connect_prefix = bytes([length & 0xFF])

        return grpc_prefix + connect_prefix + serialized

class GrpcMessageDecoder:
    def __init__(self, proto_message_class):
        self.message_class = proto_message_class

    def decode_message(self, raw_message):
        try:
            # Skip initial length prefix (4 bytes)
            # TODO: only do this for proto+connect!
            message_data = raw_message[5:]

            logger.debug(f"Original length: {len(raw_message)}")
            logger.debug(f"message length: {len(message_data)}")
            logger.debug(f"message: {message_data}")

            # Parse the raw message
            message = self.message_class()
            message.ParseFromString(message_data)
            if isinstance(message,

            # Convert to dictionary
            decoded = MessageToDict(
                message,
                preserving_proto_field_name=True
            )
            return decoded
        except Exception as e:
            logger.error(f"Failed to decode message: {str(e)}")
            return None

class CursorProvider:
    message_type_map = {
        "/aiserver.v1.AiService/StreamChat": cursorpb.GetChatRequest,
        "/aiserver.v1.AiService/StreamCpp": cursorpb.StreamCppRequest,
    }

    def decode_by_method(self, method, raw_message):
        logger.debug(f"Decoding message for method: {method}")

        cls = self.message_type_map.get(method)
        if not cls:
            logger.debug(f"Unhandled method: {method}")
            return None

        decoder = GrpcMessageDecoder(cls)
        try:
            return decoder.decode_message(raw_message)
        except Exception as e:
            logger.error(f"Failed to decode message: {str(e)}")
            return None
