from fastapi import FastAPI, status, HTTPException 
from api.dalle import router as dalle_router
from fastapi.middleware.cors import CORSMiddleware
from db.vector_database import QdrantFlyerManager
from utils.image_tratative import formatting_train_images_to_jpg
from utils.embedding import generate_caption, get_image_embeddings


app = FastAPI(title="Flayer Generator", description="Multimodal Flyer generator")

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the QdrantFlyerManager instance
qdrant_manager = QdrantFlyerManager()


@app.get("/")
def read_root():
    return {"Server is running!"}


@app.post("/train_embedding", tags=["train_embedding"])
async def train_embedding(collection_name: str):
    try:
        # Format images
        train_flyers = formatting_train_images_to_jpg()
        pair_list = generate_caption(train_flyers)
        image_embeddings = get_image_embeddings(pair_list)

        # Initialize collection (await async call)
        collection_response = await qdrant_manager.initialize_collection(collection_name)

        if collection_response == status.HTTP_200_OK:
            # Upsert embeddings (await async call)
            response = await qdrant_manager.upsert_embeddings(
                image_embeddings, pair_list, collection_name
            )
            if response == status.HTTP_200_OK:
                return {
                    "message": "Embedding trained successfully",
                    "status": response,
                }
            else:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Failed to upsert embeddings",
                )
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to initialize collection",
            )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error in train_embedding: {str(e)}"
        )

@app.get("/get_collection_info", tags=["collection"])
async def get_collection_info(collection_name: str):
    try:
        # Get collection info
        collection_info = await qdrant_manager.qdrant_client.get_collection(collection_name)
        return {
            print(collection_info)
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error in get_collection_info: {str(e)}"
        )

# include routers to use dalle llm
app.include_router(dalle_router)