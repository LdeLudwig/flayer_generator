import os
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

qdrant_client = AsyncQdrantClient("http://localhost:6333")
COLLECTION_NAME = "Flayers"
IMAGE_VECTOR_SIZE = 512
TEXT_VECTOR_SIZE = 512

async def session_db ():
    # Create a collection if it doesn't exist
    collections = await qdrant_client.get_collection()
    if not COLLECTION_NAME in [c.name for c in collections.collections]:
        await qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config={
                "image": VectorParams(size=IMAGE_VECTOR_SIZE, distance=Distance.COSINE),
                "text": VectorParams(size=TEXT_VECTOR_SIZE, distance=Distance.COSINE),
            },
        )
        print("Collection created")
    else:
        print("Collection already exists")

    return qdrant_client

async def upsert_flyer_embeddings(embedding, pair_list):
    points=[]
    for idx in enumerate(embedding):
        try:
            point_id = f"image_{idx}_{pair_list.get('image').file_name}"

            point = PointStruct(
                id = point_id,
                vector = {
                    "image": embedding.get('image').tolist(),
                    "camption": embedding.get('caption').tolist()
                },
                payload = {
                    "filename": pair_list[idx][0].file_name,
                    "caption": pair_list[idx][1],
                    "max_length": 100,
                    "similarity": embedding.get("similarity")  
                }
            )

            points.append(point)

        except Exception as e:
            raise Exception("Error in upsert_embeddings: ", e)
    
    await qdrant_client.upsert(
            collection_name=COLLECTION_NAME,
            points=points
        )
