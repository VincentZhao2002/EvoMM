import os
import json
import numpy as np
import requests
from typing import Dict, Any, List, Optional
from FlagEmbedding import FlagAutoModel
import faiss

from evomm.retriever.base import AbsRetriever
from evomm.schemas import InMemoryPageStore, Hit, Page


def _build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """
    Build a FAISS index.
    embeddings: (n, dim) numpy array
    """
    dimension = embeddings.shape[1]
    # Inner-product index (cosine similarity after normalization)
    index = faiss.IndexFlatIP(dimension)
    # L2-normalize for cosine similarity (copy to avoid mutating inputs)
    embeddings_normalized = embeddings.copy()
    faiss.normalize_L2(embeddings_normalized)
    index.add(embeddings_normalized)
    return index


def _search_faiss_index(index: faiss.Index, query_embeddings: np.ndarray, top_k: int):
    """
    Search a FAISS index.
    index: FAISS index
    query_embeddings: (n_queries, dim) query vectors
    top_k: number of hits per query
    Returns: (scores_list, indices_list) with one (top_k,) array per query
    """
    query_embeddings_normalized = query_embeddings.copy()
    faiss.normalize_L2(query_embeddings_normalized)

    scores, indices = index.search(query_embeddings_normalized, top_k)

    scores_list = [scores[i] for i in range(len(query_embeddings))]
    indices_list = [indices[i] for i in range(len(query_embeddings))]

    return scores_list, indices_list


class DenseRetriever(AbsRetriever):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.pages = None
        self.index = None
        self.doc_emb = None

        self.api_url = config.get("api_url")  # e.g. "http://localhost:8001"
        self.use_api = self.api_url is not None

        if self.use_api:
            print(f"[DenseRetriever] API mode: {self.api_url}")
            self.model = None
            try:
                response = requests.get(f"{self.api_url}/health", timeout=5)
                if response.status_code == 200:
                    print(f"[DenseRetriever] API health OK: {response.json()}")
                else:
                    print(f"[DenseRetriever] warn: API returned status {response.status_code}")
            except Exception as e:
                print(f"[DenseRetriever] warn: could not reach API: {e}")
        else:
            model_name = config.get("model_name")
            print(f"[DenseRetriever] Local mode, loading model: {model_name}")
            try:
                import torch
                has_cuda = torch.cuda.is_available()
                default_device = "cuda:0" if has_cuda else "cpu"
                devices = config.get("devices", default_device)

                self.model = FlagAutoModel.from_finetuned(
                    model_name,
                    model_class=config.get("model_class", None),
                    normalize_embeddings=config.get("normalize_embeddings", True),
                    pooling_method=config.get("pooling_method", "cls"),
                    trust_remote_code=config.get("trust_remote_code", True),
                    query_instruction_for_retrieval=config.get("query_instruction_for_retrieval"),
                    use_fp16=config.get("use_fp16", False),
                    devices=devices
                )
                if self.model is None:
                    raise RuntimeError("Model load failed: FlagAutoModel.from_finetuned() returned None")
                print(f"[DenseRetriever] Model loaded on device(s): {devices}")
            except Exception as e:
                error_msg = (
                    f"[DenseRetriever] Model load failed: {e}\n"
                    f"  model_name: {model_name}\n"
                    f"  check: 1) name/path 2) network 3) disk space"
                )
                print(error_msg)
                raise RuntimeError(error_msg) from e

    def _index_dir(self) -> str:
        return self.config["index_dir"]

    def _pages_dir(self) -> str:
        return os.path.join(self._index_dir(), "pages")

    def _emb_path(self) -> str:
        return os.path.join(self._index_dir(), "doc_emb.npy")

    def _encode_via_api(self, texts: List[str], encode_type: str = "corpus") -> np.ndarray:
        """
        Encode texts through the remote API.

        Args:
            texts: list of strings
            encode_type: "corpus" or "query"

        Returns:
            embeddings as numpy array
        """
        if not texts:
            raise ValueError("[DenseRetriever] text list is empty; cannot encode")

        non_empty_texts = [t for t in texts if t and t.strip()]
        if not non_empty_texts:
            raise ValueError("[DenseRetriever] all texts are empty; cannot encode")

        if len(non_empty_texts) != len(texts):
            print(f"[DenseRetriever] warn: dropped {len(texts) - len(non_empty_texts)} empty strings")

        try:
            request_data = {
                "texts": non_empty_texts,
                "type": encode_type,
                "batch_size": self.config.get("batch_size", 32),
                "max_length": self.config.get("max_length", 512),
            }

            response = requests.post(
                f"{self.api_url}/encode",
                json=request_data,
                timeout=300
            )

            if response.status_code != 200:
                error_detail = ""
                try:
                    error_response = response.json()
                    error_detail = f" server: {error_response}"
                except Exception:
                    error_detail = f" body: {response.text[:500]}"

                error_msg = (
                    f"[DenseRetriever] API encode failed: {response.status_code} {response.reason}\n"
                    f"  URL: {self.api_url}/encode\n"
                    f"  args: texts={len(non_empty_texts)}, type={encode_type}, "
                    f"batch_size={request_data['batch_size']}, max_length={request_data['max_length']}\n"
                    f"{error_detail}"
                )
                print(error_msg)
                response.raise_for_status()

            result = response.json()
            embeddings = np.array(result["embeddings"], dtype=np.float32)

            if len(non_empty_texts) != len(texts):
                empty_indices = [i for i, t in enumerate(texts) if not t or not t.strip()]
                full_embeddings = np.zeros((len(texts), embeddings.shape[1]), dtype=np.float32)
                non_empty_idx = 0
                for i in range(len(texts)):
                    if i not in empty_indices:
                        full_embeddings[i] = embeddings[non_empty_idx]
                        non_empty_idx += 1
                embeddings = full_embeddings

            return embeddings
        except requests.exceptions.RequestException as e:
            error_msg = (
                f"[DenseRetriever] API encode failed (network): {e}\n"
                f"  URL: {self.api_url}/encode\n"
                f"  check: 1) service up 2) URL 3) connectivity"
            )
            print(error_msg)
            raise
        except Exception as e:
            print(f"[DenseRetriever] API encode failed: {e}")
            import traceback
            traceback.print_exc()
            raise

    def _encode_pages(self, pages: List[Page]) -> np.ndarray:
        texts = []
        for p in pages:
            header = p.header if p.header is not None else ""
            content = p.content if p.content is not None else ""
            text = (header + " " + content).strip()
            text = p.content
            texts.append(text)

        if self.use_api:
            return self._encode_via_api(texts, encode_type="corpus")
        else:
            if self.model is None:
                raise RuntimeError("DenseRetriever model not initialized; check load errors.")
            return self.model.encode_corpus(
                texts,
                batch_size=self.config.get("batch_size", 32),
                max_length=self.config.get("max_length", 512),
            )

    def load(self) -> None:
        """
        Restore from disk:
        - page snapshot
        - doc_emb.npy
        - faiss index
        """
        try:
            self.doc_emb = np.load(self._emb_path())
            self.index = _build_faiss_index(self.doc_emb)
            self.pages = InMemoryPageStore.load(self._pages_dir()).load()
        except Exception as e:
            print("DenseRetriever.load() failed, will need build():", e)

    def build(self, page_store: InMemoryPageStore) -> None:
        """Full rebuild of the vector index."""
        os.makedirs(self._pages_dir(), exist_ok=True)

        self.pages = page_store.load()

        self.doc_emb = self._encode_pages(self.pages)

        self.index = _build_faiss_index(self.doc_emb)

        temp_page_store = InMemoryPageStore(dir_path=self._pages_dir())
        temp_page_store.save(self.pages)
        np.save(self._emb_path(), self.doc_emb)

    def update(self, page_store: InMemoryPageStore) -> None:
        """
        Incremental update: if only tail pages changed, re-encode from the first
        differing index instead of the full corpus.
        """
        if not self.pages or self.doc_emb is None or self.index is None:
            self.build(page_store)
            return

        new_pages = page_store.load()
        old_pages = self.pages

        max_shared = min(len(new_pages), len(old_pages))
        diff_idx = max_shared
        for i in range(max_shared):
            if Page.equal(new_pages[i], old_pages[i]):
                continue
            diff_idx = i
            break

        changed = (diff_idx < max_shared) or (len(new_pages) != len(old_pages))
        if not changed:
            return

        keep_emb = self.doc_emb[:diff_idx]

        tail_pages = new_pages[diff_idx:]
        tail_emb = self._encode_pages(tail_pages)

        new_doc_emb = np.concatenate([keep_emb, tail_emb], axis=0)

        self.index = _build_faiss_index(new_doc_emb)

        self.pages = new_pages
        self.doc_emb = new_doc_emb

        temp_page_store = InMemoryPageStore(dir_path=self._pages_dir())
        temp_page_store.save(self.pages)
        np.save(self._emb_path(), self.doc_emb)

    def search(self, query_list: List[str], top_k: int = 10) -> List[List[Hit]]:
        """
        Input: multiple queries.
        Output: one aggregated Hit list: scores summed across queries per page_id, then top_k.
        """
        if self.index is None:
            self.load()
            if self.index is None:
                return [[] for _ in query_list]

        if self.use_api:
            queries_emb = self._encode_via_api(query_list, encode_type="query")
        else:
            queries_emb = self.model.encode_queries(
                query_list,
                batch_size=self.config.get("batch_size", 32),
                max_length=self.config.get("max_length", 512),
            )

        scores_list, indices_list = _search_faiss_index(self.index, queries_emb, top_k)

        page_scores: Dict[str, float] = {}
        page_hits: Dict[str, Hit] = {}

        for scores, indices in zip(scores_list, indices_list):
            for rank, (idx, sc) in enumerate(zip(indices, scores)):
                idx_int = int(idx)
                if idx_int < 0 or idx_int >= len(self.pages):
                    continue
                page = self.pages[idx_int]
                snippet = page.content
                page_id = str(idx_int)
                score = float(sc)

                if page_id in page_scores:
                    page_scores[page_id] += score
                else:
                    page_scores[page_id] = score
                    page_hits[page_id] = Hit(
                        page_id=page_id,
                        snippet=snippet,
                        source="vector",
                        meta={"rank": rank, "score": score},
                    )

        sorted_pages = sorted(page_scores.items(), key=lambda x: x[1], reverse=True)
        top_k_pages = sorted_pages[:top_k]

        final_hits: List[Hit] = []
        for rank, (page_id, total_score) in enumerate(top_k_pages):
            hit = page_hits[page_id]
            updated_meta = hit.meta.copy() if hit.meta else {}
            updated_meta["rank"] = rank
            updated_meta["score"] = total_score
            final_hits.append(
                Hit(
                    page_id=hit.page_id,
                    snippet=hit.snippet,
                    source=hit.source,
                    meta=updated_meta
                )
            )

        return [final_hits]
