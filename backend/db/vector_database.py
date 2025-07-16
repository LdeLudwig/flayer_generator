import uuid
from typing import List, Tuple
from fastapi import status
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, SearchParams


class QdrantFlyerManager:
    IMAGE_VECTOR_SIZE = 512
    TEXT_VECTOR_SIZE = 512

    def __init__(self, qdrant_url: str = "http://localhost:6333"):
        self.qdrant_client = AsyncQdrantClient(qdrant_url)

    async def initialize_collection(self, collection: str) -> int:
        """
        Initialize a Qdrant collection for flyers.
        """
        try:
            # Create a collection if it doesn't exist
            collections = await self.qdrant_client.get_collections()
            if collection not in [c.name for c in collections.collections]:
                await self.qdrant_client.create_collection(
                    collection_name=collection,
                    vectors_config={
                        "image": VectorParams(
                            size=self.IMAGE_VECTOR_SIZE, distance=Distance.COSINE
                        ),
                        "text": VectorParams(
                            size=self.TEXT_VECTOR_SIZE, distance=Distance.COSINE
                        ),
                    },
                )
            return status.HTTP_200_OK

        except Exception as e:
            raise Exception("Error in initialize_collection: ", e)

    async def upsert_embeddings(
        self, embeddings: List[dict], pair_list: List[Tuple], collection_name: str
    ) -> int:
        """
        Upsert image and text embeddings to Qdrant.
        """
        points = []

        for idx, embedding in enumerate(embeddings):
            try:
                point = PointStruct(
                    id=str(uuid.uuid4()),
                    vector={
                        "image": embedding.get("image").tolist(),
                        "text": embedding.get("text").tolist(),
                    },
                    payload={
                        "filename": pair_list[idx][0].filename,
                        "text": pair_list[idx][1],
                        "max_length": 100,
                    },
                )
                points.append(point)

            except Exception as e:
                raise Exception(f"Error in upsert_embeddings: {e}")

        response = await self.qdrant_client.upsert(
            collection_name=collection_name, points=points
        )
        if response.status != "completed":
            return status.HTTP_500_INTERNAL_SERVER_ERROR
        return status.HTTP_200_OK

    async def similarity_search(
        self, embedding: List[float], search: str, collection_name: str
    ) -> list:
        """
        Perform similarity search in Qdrant using the prompt's embedding.
        Selects 'text' or 'image' vector based on prompt content.
        """
        try:
            response = await self.qdrant_client.query_points(
                collection_name=collection_name,
                query=embedding,
                with_vectors=True,
                with_payload=True,
                search_params=SearchParams(hnsw_ef=128, exact=False),
                using=search,
                limit=5,
            )

            return response.points
        except Exception as e:
            raise Exception(f"Error in similarity_search: {e}")
