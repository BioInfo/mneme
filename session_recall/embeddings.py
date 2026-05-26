"""Embedding generation for Session Recall."""

from typing import List
import numpy as np


class EmbeddingModel:
    """Generate embeddings using sentence-transformers with nomic model."""

    def __init__(
        self,
        model_name: str = "nomic-ai/nomic-embed-text-v1.5",
        device: str | None = None,
    ):
        """Initialize the embedding model.

        Args:
            model_name: HuggingFace model name
            device: Device to use ('mps', 'cuda', 'cpu'). Auto-detects if None.
        """
        self.model_name = model_name
        self._model = None
        self._device = device
        self.dimension = 768

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

        # Nomic requires task prefix for better retrieval
        prefixed = [f"search_document: {t}" for t in texts]
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
        # Nomic uses different prefix for queries
        prefixed = f"search_query: {query}"
        embedding = self.model.encode([prefixed], convert_to_numpy=True)
        return embedding[0]
