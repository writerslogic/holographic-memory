import inspect
import importlib.metadata
from semantic_kernel.memory.memory_store_base import MemoryStoreBase
from writerslogic_warm_sdk.semantic_kernel import WarmMemoryStore

print("semantic-kernel", importlib.metadata.version("semantic-kernel"))
print("abstract methods", sorted(WarmMemoryStore.__abstractmethods__))
for name in sorted(MemoryStoreBase.__abstractmethods__):
    print(name, inspect.signature(getattr(MemoryStoreBase, name)))
try:
    store = WarmMemoryStore()
except Exception as error:
    print(type(error).__name__, str(error))
else:
    print("instantiated")
    store._client.close()
