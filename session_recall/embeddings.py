"""Embedding generation for Session Recall."""

from typing import List
import numpy as np


class EmbeddingModel:
    """Generate embeddings using sentence-transformers with nomic model."""

    def __init__(
        self,
        model_name: str = "nomic-ai/nomic-embed-text-v1.5",
        device: str | None = None,
        dimension: int = 768,
        doc_prefix: str = "search_document: ",
        query_prefix: str = "search_query: ",
    ):
        """Initialize the embedding model.

        Args:
            model_name: HuggingFace model name
            device: Device to use ('mps', 'cuda', 'cpu'). Auto-detects if None.
            dimension: Embedding dimensionality (must match the model).
            doc_prefix: Instruction prefix for documents. nomic needs
                "search_document: "; bge-m3 and many others are prefix-free ("").
            query_prefix: Instruction prefix for queries (nomic: "search_query: ").
        """
        self.model_name = model_name
        self._model = None
        self._device = device
        self.dimension = dimension
        self.doc_prefix = doc_prefix
        self.query_prefix = query_prefix

    @property
    def model(self):
        """Lazy load the model on first use."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            import torch

            # Auto-detect device
            if self._device is None:
                if torch.backends.mps.is_available():
                    self._device = "mps"
                elif torch.cuda.is_available():
                    self._device = "cuda"
                else:
                    self._device = "cpu"

            self._model = SentenceTransformer(
                self.model_name, trust_remote_code=True
            )
            self._model.to(self._device)

        return self._model

    @property
    def device(self) -> str:
        """Get the device being used."""
        # Trigger model load to determine device
        _ = self.model
        return self._device

    def embed(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Generate embeddings for a list of texts.

        Args:
            texts: List of texts to embed
            batch_size: Batch size for encoding

        Returns:
            NumPy array of embeddings, shape (len(texts), dimension)
        """
        if not texts:
            return np.array([])

        # Apply the model's document instruction prefix (empty for bge-m3 etc.)
        prefixed = [f"{self.doc_prefix}{t}" for t in texts]
        embeddings = self.model.encode(
            prefixed,
            batch_size=batch_size,
            show_progress_bar=len(texts) > 100,
            convert_to_numpy=True,
        )
        return embeddings

    def embed_query(self, query: str) -> np.ndarray:
        """Generate embedding for a search query.

        Args:
            query: Search query text

        Returns:
            NumPy array of shape (dimension,)
        """
        # Apply the model's query instruction prefix (empty for bge-m3 etc.)
        prefixed = f"{self.query_prefix}{query}"
        embedding = self.model.encode([prefixed], convert_to_numpy=True)
        return embedding[0]
