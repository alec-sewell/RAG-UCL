import time
from typing import List, Callable, Any
from langchain.embeddings.base import Embeddings
from langchain_google_genai._common import GoogleGenerativeAIError


def _sleep_backoff(attempt: int) -> None:
    """Sleeps for a duration that increases exponentially with the attempt number."""
    time.sleep(2 ** attempt)


class RateLimitedEmbeddings(Embeddings):
    """
    A wrapper for an Embeddings object that adds rate limiting and retry logic.

    This class is designed to handle API rate limits, specifically the '429:
    Resource Exhausted' errors from services like Google Generative AI, by
    implementing an exponential backoff retry mechanism. It makes the error
    handling consistent across document and query embedding methods.
    """

    def __init__(self, inner: Embeddings, batch_size: int = 64, max_retries: int = 6):
        """
        Initializes the RateLimitedEmbeddings wrapper.
        """
        if not isinstance(max_retries, int) or max_retries <= 0:
            raise ValueError("max_retries must be a positive integer.")
        self.inner = inner
        self.batch_size = batch_size
        self.max_retries = max_retries

    def _embed_with_retry(self, embed_fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """
        Executes an embedding function with retry logic for rate limit errors.

        Returns:
            The result of the embedding function call.

        Raises:
            GoogleGenerativeAIError: If the request fails after all retries or for a
                                     non-rate-limit error.
        """
        for attempt in range(self.max_retries):
            try:
                return embed_fn(*args, **kwargs)
            except GoogleGenerativeAIError as e:
                # Retry on 429 (Resource Exhausted) errors.
                if "429" in str(e) or "exhausted" in str(e).lower():
                    if attempt + 1 < self.max_retries:
                        _sleep_backoff(attempt)
                        continue
                # Re-raise for non-429 errors or on the last attempt for a 429.
                raise e
        # This line should not be reachable but is failsafe.
        raise RuntimeError("Embedding failed after maximum retries.")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embeds a list of documents, processing them in batches with retry logic.

        Returns:
            A list of embeddings, where each embedding is a list of floats.
        """
        all_embeddings: List[List[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_embeddings = self._embed_with_retry(self.inner.embed_documents, batch)
            all_embeddings.extend(batch_embeddings)
        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        """
        Embeds a single query with retry logic.

        Returns:
            The embedding for the query as a list of floats.
        """
        return self._embed_with_retry(self.inner.embed_query, text)