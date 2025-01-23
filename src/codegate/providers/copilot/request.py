import gzip
import uuid
from dataclasses import dataclass
from typing import Dict
from urllib.parse import urlparse

import httptools
import structlog

logger = structlog.get_logger("codegate").bind(origin="copilot_proxy")

MAX_BUFFER_SIZE = 10 * 1024 * 1024  # 10MB

class RequestState:
    def __init__(self):
        self.headers_complete = False
        self.message_complete = False
        self.parser = httptools.HttpRequestParser(self)
        self.headers: Dict[str, str] = {}
        self.body = bytearray()
        self.buffer = bytearray()
        self.method = None
        self.url = None
        self.version = None
        self.upgrade = False
        self.content_length = None
        self.chunked = False
        self.path = None
        self.query = None
        self.id = str(uuid.uuid4())
        # Add these fields to track chunked data
        self.chunks_received = []
        self.current_chunk_size = None
        self.current_chunk_data = bytearray()

    def on_url(self, url: bytes):
        self.url = url.decode('utf-8')
        # Parse URL to get path and query
        parsed = urlparse(self.url)
        self.path = parsed.path
        self.query = parsed.query

    def on_header(self, name: bytes, value: bytes):
        name = name.decode('utf-8').lower()  # Normalize header names
        value = value.decode('utf-8')
        self.headers[name] = value

        # Track important headers
        if name == 'content-length':
            self.content_length = int(value)
        elif name == 'transfer-encoding' and 'chunked' in value.lower():
            self.chunked = True

    def on_headers_complete(self):
        self.headers_complete = True
        self.method = self.parser.get_method().decode('utf-8')
        self.version = f"HTTP/{self.parser.get_http_version()}"
        self.upgrade = self.parser.should_upgrade()

    def on_chunk_complete(self):
        """Called when a chunk is complete"""
        logger.debug(f"Request ID {self.id} chunk complete")
        if len(self.current_chunk_data) > 0:  # Skip zero-length chunks
            self.chunks_received.append(bytes(self.current_chunk_data))
            logger.debug(f"Request ID {self.id} chunk received: {self.current_chunk_data}")
            logger.debug(f"Request ID {self.id} total chunks received: {len(self.chunks_received)}")
            logger.debug(f"Request ID {self.id} total chunked data received: {len(b''.join(self.chunks_received))}")
            self.current_chunk_data = bytearray() # reset current chunk data
            #self.body.extend(self.current_chunk_data)

    def on_body(self, body: bytes):
        """Called when body data is received"""
        if self.chunked:
            self.current_chunk_data.extend(body)
        else:
            self.body.extend(body)

    def on_message_complete(self):
        logger.debug(f"Request ID {self.id} Message complete")
        self.message_complete = True

    def feed_data(self, data: bytes) -> bool:
        if len(self.buffer) >= MAX_BUFFER_SIZE:
            logger.error(f"Request ID {self.id} body too large")
            raise ValueError("Request body too large")
        self.buffer.extend(data)

        logger.debug(f"Feeding data to request parser: {data}")
        try:
            self.parser.feed_data(data)
            return True
        except httptools.HttpParserUpgrade:
            self.upgrade = True
            return True
        except httptools.HttpParserError as e:
            logger.error(f"Parser error: {e}")
            return False

    def get_buffer(self) -> bytes:
        return bytes(self.buffer)

    def is_protobuf_request(self) -> bool:
        return self.headers.get('content-type', '').startswith('application/proto') or \
            self.headers.get('content-type', '').startswith('application/connect+proto')

    def get_body(self) -> bytes:
        raw_body = bytes(self.body) if not self.chunked else b''.join(self.chunks_received)

        content_encoding = self.headers.get('content-encoding', '').lower()
        if content_encoding == '':
            return raw_body

        if 'gzip' in content_encoding:
            try:
                decompressed_body = gzip.decompress(raw_body)
                logger.debug(f"Decompressed body {decompressed_body}")
                return decompressed_body
            except Exception as e:
                logger.error(f"Error decompressing gzipped body: {e}")
                return raw_body
        else:
            logger.warning(f"Unsupported content encoding: {content_encoding}")

        return raw_body

    def is_complete(self) -> bool:
        """Check if we have received the complete request"""
        if not self.headers_complete:
            return False

        # For requests without body
        if self.method in ('GET', 'HEAD', 'OPTIONS', 'CONNECT'):
            logger.debug("Request without body is complete")
            return True

        # For chunked requests, we need to wait for message_complete
        if self.chunked:
            logger.debug(f"Chunked request {self.id} complete status: {self.message_complete}")
            return self.message_complete

        # For content-length requests
        if self.content_length is not None:
            is_complete = len(self.body) >= self.content_length
            logger.debug(f"Content-Length request complete status: {is_complete} ({len(self.body)}/{self.content_length})")
            return is_complete

        return False

    def get_request_data(self) -> bytes:
        """Reconstruct the full HTTP request"""
        request_line = f"{self.method} {self.url} {self.version}\r\n"
        headers = "\r\n".join(f"{k}: {v}" for k, v in self.headers.items())
        return f"{request_line}{headers}\r\n\r\n".encode() + bytes(self.body)


@dataclass
class RequestOut:
    method: str
    url: str
    version: str
    headers: Dict[str, str]
    body: bytes

    def to_bytes(self) -> bytes:
        """Reconstruct HTTP request from modifications, optionally chunking the body"""
        request_line = f"{self.method} {self.url} {self.version}\r\n"

        chunked = self.headers.get('transfer-encoding', '').lower() == 'chunked'

        headers = self.headers.copy()
        if chunked:
            headers['transfer-encoding'] = 'chunked'
        elif self.body:
            headers['content-length'] = str(len(self.body))

        headers_str = "\r\n".join(f"{k}: {v}" for k, v in headers.items())
        request = f"{request_line}{headers_str}\r\n\r\n".encode()

        if chunked:
            chunk_size = hex(len(self.body))[2:] + '\r\n'
            return request + chunk_size.encode() + self.body + b'\r\n0\r\n\r\n'

        return request + self.body

def request_in_to_out(request_in: RequestState) -> RequestOut:
    """
    Reconstruction of the request from the RequestState object

    Will be useful for pipeline fall-throughs, e.g. requests that
    are not handled at all.
    """

    # this is temporary, the body would be returned by the pipeline
    # already compressed
    content_encoding = request_in.headers.get('content-encoding', '').lower()
    body = request_in.get_body()
    if 'gzip' in content_encoding and body:
        body = gzip.compress(body)

    return RequestOut(
        method=request_in.method,
        url=request_in.url,
        version=request_in.version,
        headers=request_in.headers,
        body=body,
    )

