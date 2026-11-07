import os
import json
from typing import Dict, Any, List

from evomm.retriever.base import AbsRetriever
from evomm.schemas import InMemoryPageStore, Hit, Page


class IndexRetriever(AbsRetriever):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.pages: List[Page] = []

    def load(self):
        index_dir = self.config.get("index_dir")
        try:
            # Construct InMemoryPageStore so pages load from disk
            self.page_store = InMemoryPageStore(dir_path=os.path.join(index_dir, "pages"))
        except Exception as e:
            print('cannot load index, error: ', e)

    def build(self, page_store: InMemoryPageStore):
        # New PageStore instance for persistence
        target_path = os.path.join(self.config.get("index_dir"), "pages")
        new_store = InMemoryPageStore(dir_path=target_path)
        pages = page_store._pages if hasattr(page_store, '_pages') else page_store.load()
        new_store.save(pages)
        self.page_store = new_store

    def update(self, page_store: InMemoryPageStore):
        self.build(page_store)

    def search(self, query_list: List[str], top_k: int = 10) -> List[List[Hit]]:
        hits: List[Hit] = []
        for query in query_list:
            # Parse query as comma-separated page indices
            try:
                page_index = [int(idx.strip()) for idx in query.split(',') if idx.strip().isdigit()]
            except ValueError:
                continue
                
            for pid in page_index:
                p = self.page_store.get(pid)
                if not p:
                    continue
                hits.append(Hit(
                    page_id=str(pid),
                    snippet=p.content,
                    source="page_index",
                    meta={}
                ))
        return [hits]