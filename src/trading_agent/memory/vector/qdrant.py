"""Optional Qdrant-backed :class:`~trading_agent.memory.vector.base.VectorStore`.

Selected when the user's ``vstore`` setting is ``"qdrant"`` (default stays
sqlite-vec). Same contract, same cosine-similarity scores, so swapping backends
is invisible to MemoryStore / reflection / hygiene.

Qdrant point ids must be ints or UUIDs, but callers use arbitrary string ids
(e.g. a lesson uuid hex). We map ``id -> uuid5(id)`` deterministically and keep
the caller's original id in the payload under :data:`_ORIG_ID` so reads return
exactly what was upserted. Cosine distance is configured at collection creation.

Tests run fully offline using Qdrant's embedded mode (``location=":memory:"`` or
an on-disk ``path=``); production points at a ``url``.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from typing import Any

from .base import Filter, Hit, StoredPoint

_ORIG_ID = "_orig_id"  # reserved payload key holding the caller's string id
_NAMESPACE = uuid.UUID("6f3c9a1e-0000-4000-8000-000000000d4d")  # stable, WS-D


def _point_uuid(id: str) -> str:
    return str(uuid.uuid5(_NAMESPACE, id))


class QdrantVectorStore:
    """VectorStore over qdrant-client. Collections are created lazily."""

    def __init__(
        self,
        *,
        url: str | None = None,
        path: str | None = None,
        location: str | None = None,
        api_key: str | None = None,
    ) -> None:
        try:
            from qdrant_client import QdrantClient
        except ImportError as exc:  # optional dependency
            raise RuntimeError(
                "qdrant vstore selected but qdrant-client is not installed "
                "(pip install 'trading-agent[memory]')"
            ) from exc
        if url:
            self._client = QdrantClient(url=url, api_key=api_key)
        elif path:
            self._client = QdrantClient(path=path)
        else:
            # Embedded, in-process — used by tests and small single-box installs.
            self._client = QdrantClient(location=location or ":memory:")

    # --- helpers -------------------------------------------------------------

    def _ensure_collection(self, collection: str, dim: int) -> None:
        from qdrant_client.models import Distance, VectorParams

        if not self._client.collection_exists(collection):
            self._client.create_collection(
                collection, vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
            )

    @staticmethod
    def _to_filter(flt: Filter | None) -> Any:
        if not flt:
            return None
        from qdrant_client.models import FieldCondition, MatchValue
        from qdrant_client.models import Filter as QFilter

        return QFilter(
            must=[FieldCondition(key=k, match=MatchValue(value=v)) for k, v in flt.items()]
        )

    @staticmethod
    def _strip(payload: dict[str, Any] | None) -> tuple[str, dict[str, Any]]:
        payload = dict(payload or {})
        orig = payload.pop(_ORIG_ID, "")
        return str(orig), payload

    @staticmethod
    def _as_floats(vector: Any) -> list[float]:
        """Flatten a retrieved vector (a plain list[float]) to floats; [] otherwise."""
        if not isinstance(vector, list):
            return []
        return [float(x) for x in vector if isinstance(x, (int, float))]

    # --- CONTRACTS triple ----------------------------------------------------

    def upsert(
        self, collection: str, id: str, vector: Sequence[float], payload: dict[str, Any]
    ) -> None:
        from qdrant_client.models import PointStruct

        self._ensure_collection(collection, len(vector))
        stored = dict(payload)
        stored[_ORIG_ID] = id
        self._client.upsert(
            collection,
            points=[PointStruct(id=_point_uuid(id), vector=list(vector), payload=stored)],
        )

    def search(
        self, collection: str, vector: Sequence[float], k: int, flt: Filter | None = None
    ) -> list[Hit]:
        if k <= 0 or not self._client.collection_exists(collection):
            return []
        result = self._client.query_points(
            collection,
            query=list(vector),
            limit=k,
            query_filter=self._to_filter(flt),
            with_payload=True,
        ).points
        hits: list[Hit] = []
        for p in result:
            orig, payload = self._strip(p.payload)
            hits.append(Hit(id=orig, score=float(p.score), payload=payload))
        return hits

    def delete(self, collection: str, id: str) -> None:
        if not self._client.collection_exists(collection):
            return
        self._client.delete(collection, points_selector=[_point_uuid(id)])

    # --- management helpers --------------------------------------------------

    def get(self, collection: str, id: str) -> StoredPoint | None:
        if not self._client.collection_exists(collection):
            return None
        recs = self._client.retrieve(
            collection, ids=[_point_uuid(id)], with_payload=True, with_vectors=True
        )
        if not recs:
            return None
        rec = recs[0]
        orig, payload = self._strip(rec.payload)
        return StoredPoint(id=orig or id, vector=self._as_floats(rec.vector), payload=payload)

    def iter_points(self, collection: str, flt: Filter | None = None) -> list[StoredPoint]:
        if not self._client.collection_exists(collection):
            return []
        points: list[StoredPoint] = []
        offset: Any = None
        while True:
            batch, offset = self._client.scroll(
                collection,
                scroll_filter=self._to_filter(flt),
                with_payload=True,
                with_vectors=True,
                limit=256,
                offset=offset,
            )
            for rec in batch:
                orig, payload = self._strip(rec.payload)
                points.append(
                    StoredPoint(id=orig, vector=self._as_floats(rec.vector), payload=payload)
                )
            if offset is None:
                break
        return points

    def set_payload(self, collection: str, id: str, payload: dict[str, Any]) -> None:
        if not self._client.collection_exists(collection):
            return
        stored = dict(payload)
        stored[_ORIG_ID] = id
        self._client.overwrite_payload(
            collection, payload=stored, points=[_point_uuid(id)]
        )

    def count(self, collection: str, flt: Filter | None = None) -> int:
        if not self._client.collection_exists(collection):
            return 0
        return int(
            self._client.count(collection, count_filter=self._to_filter(flt), exact=True).count
        )
