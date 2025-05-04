from PIL import Image
from fastapi import status, HTTPException
from db.vector_database import QdrantFlyerManager
from utils.embedding import Embedding
from db.models import Prompt

class RAG:
    def __init__(self):
        self.qdrant_manager = QdrantFlyerManager()
        self.embedding_models = Embedding()

    async def get_similar_flyers(self, query: Prompt, collection_name: str = "flyers"):
        """
        Retrieve similar flyers from Qdrant based on text, image, or both.
        - Text-only: Use text embedding.
        - Image-only: Use image embedding.
        - Text+image: Average text and image embeddings.
        """
        try:
            query_embedding = None
            if query.text and query.images:
                # Combine text and image embeddings by averaging
                text_embedding = self.embedding_models.input_text_embedding(query.text)
                image = Image.open(query.images[0])  # Process first image
                image_embedding = self.embedding_models.input_image_embedding(image)
                query_embedding = (text_embedding + image_embedding) / 2
            elif query.text:
                query_embedding = self.embedding_models.input_text_embedding(query.text)
            elif query.images:
                image = Image.open(query.images[0])  # Process first image
                query_embedding = self.embedding_models.input_image_embedding(image)
            else:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Prompt must contain text or at least one image."
                )

            # Perform similarity search
            search_results = await self.qdrant_manager.similarity_search(
                embedding=query_embedding.tolist(),
                search="text",
                collection_name=collection_name
            )

            if not search_results:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="No similar flyers found."
                )
            
            # Format results to be decoded
            formatted_results = [
                # return just the image vector 
                {
                    # Get payload associated values
                    "id": result.id,
                    "filename": result.payload.get("filename"),
                    "text": result.payload.get("text"),
                    "score": result.score,
                    # Get image vector only
                    "image_vector": result.vector.get("image"),
                    "text_vector": result.vector.get("text"),
                }
                for result in search_results
            ]

            return formatted_results

        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error in get_similar_flyers: {str(e)}"
            )