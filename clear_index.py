import sys
from pathlib import Path

# Force bot2 imports
sys.path.insert(0, str(Path(__file__).parent))

from src.vectorstore.chroma_store import VectorStore

def clear_all():
    # 1. Clear ChromaDB
    print("Connecting to ChromaDB Cloud...")
    store = VectorStore()
    try:
        store.delete_collection()
        print("Successfully deleted bot2 ChromaDB collection.")
    except Exception as e:
        print(f"Could not delete collection (maybe it doesn't exist?): {e}")

    # 2. Clear PDF Registry
    registry_path = Path(__file__).parent / "data" / "pdf_registry.json"
    if registry_path.exists():
        registry_path.unlink()
        print("Successfully wiped local PDF tracking registry.")
    else:
        print("No local PDF tracking registry found.")

if __name__ == "__main__":
    clear_all()
