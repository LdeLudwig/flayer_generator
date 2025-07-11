from PIL import Image
from fastapi import status, HTTPException
from backend.db.vector_database import QdrantFlyerManager
from backend.services.embedding import Embedding
from backend.models.prompt import Prompt

class RAG:
    def __init__(self):
        self.qdrant_manager = QdrantFlyerManager()
        self.embedding_models = Embedding()

    async def get_similar_flyers(self, query: Prompt, collection_name: str = "flyers"):
        """
        Retrieve similar flyers from the vector database based on a text and/or image query.

        This method performs semantic similarity search by converting the input query into embeddings
        and finding the most similar flyers in the vector database. It supports multimodal queries
        (text + image), text-only queries, and image-only queries.

        Args:
            query (Prompt): A Prompt object containing the search query. Must have at least one of:
                - text (Optional[str]): Text description to search for
                - images (Optional[List[str]]): List of image file paths (only first image is used)
            collection_name (str, optional): Name of the Qdrant collection to search in.
                Defaults to "flyers".

        Returns:
            List[Dict]: A list of dictionaries containing similar flyer information. Each dictionary includes:
                - id: Unique identifier of the flyer
                - filename: Original filename of the flyer image
                - text: Text content associated with the flyer
                - score: Similarity score (higher = more similar)
                - image_vector: Image embedding vector
                - text_vector: Text embedding vector

        Raises:
            HTTPException:
                - 400 BAD REQUEST: If query contains neither text nor images
                - 404 NOT FOUND: If no similar flyers are found
                - 500 INTERNAL SERVER ERROR: If an error occurs during processing

        Note:
            - When both text and images are provided, embeddings are combined by averaging
            - Only the first image in the images list is processed
            - Returns up to 5 most similar flyers based on the similarity search configuration
        """
        try:
            query_embedding = None
            if query.text and len(query.images) > 0:
                # Combine text and image embeddings by averaging
                text_embedding = self.embedding_models.input_text_embedding(query.text)
                image = Image.open(query.images[0])  # Process first image
                image_embedding = self.embedding_models.input_image_embedding(image)
                query_embedding = (text_embedding + image_embedding) / 2
            elif query.text:
                query_embedding = self.embedding_models.input_text_embedding(query.text)
            elif len(query.images) > 0:
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
                search="image",
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