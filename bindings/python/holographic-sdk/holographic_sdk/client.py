import hashlib
import json
import struct
from urllib.parse import quote
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import httpx

from ._transport import KeepAliveTransport, usable_for

try:
    import orjson
except ImportError:  # optional: `uv add holographic-memory-sdk[fast]`
    orjson = None

if TYPE_CHECKING:
    from .memory import AsyncMemoryClient, MemoryClient

_DEFAULT_MASK_SALT = b"holographic-sdk-mask-v1"
_MIN_SALT_LEN = 16


class HolographicError(httpx.HTTPStatusError):
    """The service rejected a request with its documented ``{"error": {"code", "message"}}`` body.

    ``status_code``, ``code`` and ``message`` come from that response. It is still an
    ``httpx.HTTPStatusError``, so code that catches that, such as the importer's retries, keeps working.
    """

    def __init__(self, message: str, code: str, response: httpx.Response) -> None:
        super().__init__(message, request=response.request, response=response)
        self.message = message
        self.code = code
        self.status_code = response.status_code


def _check(res: httpx.Response) -> None:
    """Raise `HolographicError` for an error in the service's shape, and `raise_for_status()` otherwise."""
    if res.is_success:
        return
    try:
        error = res.json()["error"]
        code, message = error["code"], error["message"]
    except (ValueError, KeyError, TypeError):
        res.raise_for_status()
        return
    raise HolographicError(message, code, res)


def _dumps(payload: Any) -> bytes:
    """Encode a JSON body; rejects NaN and infinity with ``ValueError`` like httpx's ``json=``."""
    if orjson is not None:
        body = orjson.dumps(payload)
        # orjson writes non-finite floats as null; re-encode those rare bodies with the stdlib
        # so they raise as before.
        if b"null" not in body:
            return body
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")


class _Routes:
    """Parsed endpoint URLs, so a call does not re-parse the service URL."""

    def __init__(self, base: str) -> None:
        self.batch = httpx.URL(f"{base}/api/v1/documents/batch")
        self.document = httpx.URL(f"{base}/api/v1/documents")
        self.query = httpx.URL(f"{base}/api/v1/query")
        self.maintain = httpx.URL(f"{base}/api/v1/index/maintain")


def _get_headers(api_key: Optional[str], tenant_id: Optional[str]) -> Dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    if tenant_id:
        headers["X-Tenant-ID"] = tenant_id
    return headers


class VectorMask:
    """Keyed signed permutation of embedding coordinates.

    Applied client-side so the server can rank vectors without the original
    coordinates. Inner products, norms and cosine similarity are preserved
    exactly, which also means the server learns every pairwise similarity.
    This is obfuscation, not encryption: known plain/masked pairs reveal the
    key material for the coordinates they cover.
    """

    def __init__(self, passphrase: str, salt: bytes = _DEFAULT_MASK_SALT):
        if not passphrase:
            raise ValueError("mask passphrase must not be empty")
        if len(salt) < _MIN_SALT_LEN:
            raise ValueError(f"mask salt must be at least {_MIN_SALT_LEN} bytes")
        self._key = hashlib.scrypt(
            passphrase.encode("utf-8"), salt=salt, n=2**15, r=8, p=1, maxmem=2**26, dklen=32
        )
        self._tables: Dict[int, Any] = {}

    def _table(self, dim: int):
        table = self._tables.get(dim)
        if table is None:
            prefix = hashlib.sha256(b"hms-dense-mask-v1" + self._key + struct.pack("<Q", dim))
            tags = []
            for idx in range(dim):
                h = prefix.copy()
                h.update(struct.pack("<I", idx))
                tags.append((h.digest(), idx))
            signs = [1.0 if tag[31] & 1 else -1.0 for tag, _ in tags]
            order = [idx for _, idx in sorted(tags)]
            table = self._tables[dim] = (order, signs)
        return table

    def apply(self, embedding: List[float]) -> List[float]:
        order, signs = self._table(len(embedding))
        return [embedding[idx] * signs[idx] for idx in order]


def _make_mask(zero_trust_key: str, mask_salt: Optional[bytes]) -> Optional[VectorMask]:
    if not zero_trust_key:
        return None
    return VectorMask(zero_trust_key, mask_salt or _DEFAULT_MASK_SALT)


def _mask_documents(mask: Optional[VectorMask], documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if mask is None:
        return documents
    return [
        {**doc, "vector": mask.apply(doc["vector"])} if doc.get("vector") else doc
        for doc in documents
    ]


class HolographicClient:
    """Core sync HTTP client for communicating with a Holographic Memory backend."""
    
    def __init__(
        self,
        url: str = "http://localhost:8080",
        zero_trust_key: str = "",
        api_key: Optional[str] = None,
        tenant_id: Optional[str] = None,
        timeout: int = 30,
        mask_salt: Optional[bytes] = None,
        transport: Optional[httpx.BaseTransport] = None,
    ):
        self.url = url.rstrip('/')
        self._mask = _make_mask(zero_trust_key, mask_salt)
        headers = _get_headers(api_key, tenant_id)
        self._routes = _Routes(self.url)
        if transport is None and usable_for(self.url):
            transport = KeepAliveTransport()
        self.client = httpx.Client(headers=headers, timeout=timeout, transport=transport)

    @property
    def memory(self) -> "MemoryClient":
        """Agent memory (remember, recall, forget, explain) over this client's connection."""
        from .memory import MemoryClient

        return MemoryClient(self)

    def close(self) -> None:
        self.client.close()

    def __enter__(self) -> "HolographicClient":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def encrypt_vector(self, embedding: List[float]) -> List[float]:
        """Mask an embedding when a `zero_trust_key` is configured. Not encryption; see `VectorMask`."""
        if self._mask is None or not embedding:
            return embedding
        return self._mask.apply(embedding)

    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """Batch upload documents to the holographic store."""
        if not documents:
            return
            
        docs_to_send = _mask_documents(self._mask, documents)
        res = self.client.post(self._routes.batch, content=_dumps(docs_to_send))
        if res.status_code == 404: # Fallback for old API
            for doc in docs_to_send:
                _check(self.client.post(self._routes.document, content=_dumps(doc)))
        else:
            _check(res)

    def delete_document(self, doc_id: str) -> None:
        res = self.client.delete(f"{self.url}/api/v1/documents/{quote(doc_id, safe='')}")
        _check(res)

    def maintain_document_index(self) -> Dict[str, Any]:
        res = self.client.post(self._routes.maintain, content=b"{}")
        _check(res)
        return res.json()

    def query(self, query_vector: List[float], top_k: int = 3, filter: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        encrypted_query = self.encrypt_vector(query_vector) if query_vector else []
        payload: Dict[str, Any] = {"query_vector": encrypted_query, "top_k": top_k}
        if filter:
            payload["filter"] = filter
            
        res = self.client.post(self._routes.query, content=_dumps(payload))
        _check(res)
        return res.json()


class AsyncHolographicClient:
    """Core async HTTP client for communicating with a Holographic Memory backend."""
    
    def __init__(
        self,
        url: str = "http://localhost:8080",
        zero_trust_key: str = "",
        api_key: Optional[str] = None,
        tenant_id: Optional[str] = None,
        timeout: int = 30,
        mask_salt: Optional[bytes] = None,
        transport: Optional[httpx.AsyncBaseTransport] = None,
    ):
        self.url = url.rstrip('/')
        self._mask = _make_mask(zero_trust_key, mask_salt)
        headers = _get_headers(api_key, tenant_id)
        self._routes = _Routes(self.url)
        self.client = httpx.AsyncClient(headers=headers, timeout=timeout, transport=transport)

    @property
    def memory(self) -> "AsyncMemoryClient":
        """Agent memory (remember, recall, forget, explain) over this client's connection."""
        from .memory import AsyncMemoryClient

        return AsyncMemoryClient(self)

    async def aclose(self) -> None:
        await self.client.aclose()

    async def __aenter__(self) -> "AsyncHolographicClient":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.aclose()

    def encrypt_vector(self, embedding: List[float]) -> List[float]:
        """Mask an embedding when a `zero_trust_key` is configured. Not encryption; see `VectorMask`."""
        if self._mask is None or not embedding:
            return embedding
        return self._mask.apply(embedding)

    async def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        if not documents:
            return
            
        docs_to_send = _mask_documents(self._mask, documents)
        res = await self.client.post(self._routes.batch, content=_dumps(docs_to_send))
        if res.status_code == 404: # Fallback for old API
            for doc in docs_to_send:
                res_single = await self.client.post(self._routes.document, content=_dumps(doc))
                _check(res_single)
        else:
            _check(res)

    async def delete_document(self, doc_id: str) -> None:
        res = await self.client.delete(f"{self.url}/api/v1/documents/{quote(doc_id, safe='')}")
        _check(res)

    async def maintain_document_index(self) -> Dict[str, Any]:
        res = await self.client.post(self._routes.maintain, content=b"{}")
        _check(res)
        return res.json()

    async def query(self, query_vector: List[float], top_k: int = 3, filter: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        encrypted_query = self.encrypt_vector(query_vector) if query_vector else []
        payload: Dict[str, Any] = {"query_vector": encrypted_query, "top_k": top_k}
        if filter:
            payload["filter"] = filter
            
        res = await self.client.post(self._routes.query, content=_dumps(payload))
        _check(res)
        return res.json()
