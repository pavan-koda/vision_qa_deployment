"""
ColPali Vision Retriever
Specialized retriever for vision-based document understanding
Uses ColPali to index PDF pages as images and retrieve based on visual similarity
Perfect for 500+ page documents with diagrams, charts, and complex layouts
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import torch
from PIL import Image
import pickle

logger = logging.getLogger(__name__)


class ColPaliRetriever:
    """
    ColPali-based retriever that indexes PDF pages as images.
    Enables visual search across large documents.
    """

    def __init__(
        self,
        model_name: str = "vidore/colpali",
        device: str = None,
        use_half_precision: bool = True
    ):
        """
        Initialize ColPali retriever.

        Args:
            model_name: HuggingFace model name for ColPali
            device: Device to use (cuda/cpu), auto-detected if None
            use_half_precision: Use FP16 for faster inference
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.use_half_precision = use_half_precision and self.device == "cuda"

        logger.info(f"Initializing ColPali on device: {self.device}")

        try:
            # Import ColPali dependencies
            from colpali_engine.models import ColPali, ColPaliProcessor

            self.processor = ColPaliProcessor.from_pretrained(model_name)
            self.model = ColPali.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.use_half_precision else torch.float32
            ).to(self.device)

            self.model.eval()
            logger.info("ColPali model loaded successfully")

        except ImportError:
            logger.warning("ColPali not installed. Using fallback CLIP-based retriever")
            self._init_fallback_model()

    def _init_fallback_model(self):
        """Initialize CLIP as fallback if ColPali is not available."""
        try:
            from transformers import CLIPProcessor, CLIPModel

            model_name = "openai/clip-vit-large-patch14"
            logger.info(f"Loading fallback CLIP model: {model_name}")

            self.processor = CLIPProcessor.from_pretrained(model_name)
            self.model = CLIPModel.from_pretrained(model_name).to(self.device)
            self.model.eval()

            self.is_fallback = True
            logger.info("CLIP model loaded as fallback")

        except Exception as e:
            logger.error(f"Failed to load fallback model: {str(e)}")
            raise

    def encode_images(
        self,
        image_paths: List[str],
        batch_size: int = 8
    ) -> np.ndarray:
        """
        Encode images to embeddings.

        Args:
            image_paths: List of paths to images
            batch_size: Batch size for encoding

        Returns:
            NumPy array of embeddings
        """
        try:
            all_embeddings = []

            for i in range(0, len(image_paths), batch_size):
                batch_paths = image_paths[i:i + batch_size]

                # Load images
                images = []
                for img_path in batch_paths:
                    try:
                        img = Image.open(img_path).convert('RGB')
                        images.append(img)
                    except Exception as e:
                        logger.warning(f"Failed to load image {img_path}: {str(e)}")
                        continue

                if not images:
                    continue

                # Process images
                if hasattr(self, 'is_fallback') and self.is_fallback:
                    # CLIP encoding
                    inputs = self.processor(
                        images=images,
                        return_tensors="pt",
                        padding=True
                    ).to(self.device)

                    with torch.no_grad():
                        image_features = self.model.get_image_features(**inputs)
                        # Normalize embeddings
                        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                        embeddings = image_features.cpu().numpy()

                else:
                    # ColPali encoding
                    inputs = self.processor(images=images, return_tensors="pt").to(self.device)

                    with torch.no_grad():
                        embeddings = self.model(**inputs).last_hidden_state
                        # Mean pooling
                        embeddings = embeddings.mean(dim=1)
                        # Normalize
                        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
                        embeddings = embeddings.cpu().numpy()

                all_embeddings.append(embeddings)

            if not all_embeddings:
                return np.array([])

            return np.vstack(all_embeddings)

        except Exception as e:
            logger.error(f"Error encoding images: {str(e)}")
            return np.array([])

    def encode_query(self, query: str) -> np.ndarray:
        """
        Encode text query to embedding.

        Args:
            query: Text query

        Returns:
            Query embedding
        """
        try:
            if hasattr(self, 'is_fallback') and self.is_fallback:
                # CLIP text encoding
                inputs = self.processor(
                    text=[query],
                    return_tensors="pt",
                    padding=True
                ).to(self.device)

                with torch.no_grad():
                    text_features = self.model.get_text_features(**inputs)
                    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                    return text_features.cpu().numpy()[0]

            else:
                # ColPali query encoding
                inputs = self.processor(text=[query], return_tensors="pt").to(self.device)

                with torch.no_grad():
                    query_embedding = self.model(**inputs).last_hidden_state
                    query_embedding = query_embedding.mean(dim=1)
                    query_embedding = query_embedding / query_embedding.norm(dim=-1, keepdim=True)
                    return query_embedding.cpu().numpy()[0]

        except Exception as e:
            logger.error(f"Error encoding query: {str(e)}")
            return np.array([])

    def create_index(
        self,
        page_images: List[str],
        session_id: str,
        data_dir: str = "data",
        use_faiss: bool = True
    ) -> bool:
        """
        Create visual index from page images.

        Args:
            page_images: List of paths to page images
            session_id: Session identifier
            data_dir: Directory to save index
            use_faiss: Use FAISS for efficient search

        Returns:
            True if successful
        """
        try:
            logger.info(f"Creating visual index for {len(page_images)} pages...")

            # Encode all page images
            embeddings = self.encode_images(page_images)

            if embeddings.shape[0] == 0:
                logger.error("Failed to create embeddings")
                return False

            # Normalize embeddings
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / norms

            data_path = Path(data_dir)
            data_path.mkdir(exist_ok=True)

            if use_faiss:
                # Create FAISS index
                import faiss

                dimension = embeddings.shape[1]
                index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
                index.add(embeddings.astype('float32'))

                # Save index
                index_path = data_path / f"{session_id}_colpali.faiss"
                faiss.write_index(index, str(index_path))
                logger.info(f"Saved FAISS index to {index_path}")

            # Save page paths and embeddings
            metadata = {
                'page_images': page_images,
                'embeddings': embeddings,
                'num_pages': len(page_images)
            }

            metadata_path = data_path / f"{session_id}_colpali_meta.pkl"
            with open(metadata_path, 'wb') as f:
                pickle.dump(metadata, f)

            logger.info(f"Visual index created for session {session_id}")
            return True

        except Exception as e:
            logger.error(f"Error creating index: {str(e)}")
            return False

    def search(
        self,
        query: str,
        session_id: str,
        data_dir: str = "data",
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant pages using visual similarity.

        Args:
            query: Text query
            session_id: Session identifier
            data_dir: Directory with saved index
            top_k: Number of top results to return

        Returns:
            List of results with page info and scores
        """
        try:
            data_path = Path(data_dir)

            # Load metadata
            metadata_path = data_path / f"{session_id}_colpali_meta.pkl"
            if not metadata_path.exists():
                logger.error(f"Index not found for session {session_id}")
                return []

            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)

            # Encode query
            query_embedding = self.encode_query(query)

            if query_embedding.shape[0] == 0:
                logger.error("Failed to encode query")
                return []

            # Normalize query
            query_embedding = query_embedding / np.linalg.norm(query_embedding)

            # Search with FAISS if available
            index_path = data_path / f"{session_id}_colpali.faiss"
            if index_path.exists():
                import faiss

                index = faiss.read_index(str(index_path))
                scores, indices = index.search(
                    query_embedding.reshape(1, -1).astype('float32'),
                    top_k
                )

                results = []
                for idx, score in zip(indices[0], scores[0]):
                    if idx < len(metadata['page_images']):
                        results.append({
                            'page': idx + 1,
                            'image_path': metadata['page_images'][idx],
                            'score': float(score),
                            'rank': len(results) + 1
                        })

                return results

            else:
                # Fallback: compute similarities manually
                embeddings = metadata['embeddings']
                similarities = np.dot(embeddings, query_embedding)

                # Get top-k indices
                top_indices = np.argsort(similarities)[::-1][:top_k]

                results = []
                for rank, idx in enumerate(top_indices, 1):
                    results.append({
                        'page': idx + 1,
                        'image_path': metadata['page_images'][idx],
                        'score': float(similarities[idx]),
                        'rank': rank
                    })

                return results

        except Exception as e:
            logger.error(f"Error searching index: {str(e)}")
            return []


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    retriever = ColPaliRetriever()
    print("ColPali retriever initialized successfully")
