import json
from pathlib import Path

from pydantic_ai import FunctionToolset
from pydantic_ai.tools import AgentDepsT

from synagent.storage._models import StorageQuery, StorageRecord

_DEFAULT_PATH = Path.cwd() / ".synagent" / "storage.json"


class StorageToolset(FunctionToolset[AgentDepsT]):
    """Toolset for persisting and retrieving synthesis records."""

    include_return_schema = True

    def __init__(self, path: Path = _DEFAULT_PATH):
        super().__init__()
        self._path = path
        self.add_function(self.save_record, name="save_record")
        self.add_function(self.get_record, name="get_record")
        self.add_function(self.list_records, name="list_records")

    def _load(self) -> dict[str, dict]:
        if not self._path.exists():
            return {}
        return json.loads(self._path.read_text())

    def _flush(self, store: dict[str, dict]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(store, indent=2))

    async def save_record(self, record: StorageRecord) -> str:
        """Persists a record to storage and returns its ID.

        Args:
            record (StorageRecord): Record to save.

        Returns:
            str: The ID of the saved record.
        """
        store = self._load()
        store[record.id] = record.data
        self._flush(store)
        return record.id

    async def get_record(self, query: StorageQuery) -> StorageRecord | None:
        """Retrieves a record by ID or query filters.

        Args:
            query (StorageQuery): Lookup parameters.

        Returns:
            StorageRecord | None: The first matching record, or None if not found.
        """
        store = self._load()
        if query.id is not None:
            data = store.get(query.id)
            if data is None:
                return None
            if query.filters and not all(
                data.get(k) == v for k, v in query.filters.items()
            ):
                return None
            return StorageRecord(id=query.id, data=data)
        for rid, data in store.items():
            if all(data.get(k) == v for k, v in query.filters.items()):
                return StorageRecord(id=rid, data=data)
        return None

    async def list_records(self) -> list[StorageRecord]:
        """Lists all stored records.

        Returns:
            list[StorageRecord]: All records currently in storage.
        """
        store = self._load()
        return [StorageRecord(id=rid, data=data) for rid, data in store.items()]
