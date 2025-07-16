import uvicorn
from fastapi.responses import RedirectResponse
from fastapi import FastAPI, status, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from db.vector_database import QdrantFlyerManager
from db.models import Prompt
from api.image_generator import router as api_router
from utils.image_tratative import formatting_train_images_to_jpg
from utils.embedding import Embedding
from utils.RAG import RAG


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
# Initialize Embedding instance
embedding = Embedding()
# Initialize RAG instance
rag = RAG()

# Define the upload directory
UPLOAD_DIR = "./uploads"


@app.get("/")
def read_root():
    return RedirectResponse(url="/docs")


@app.get("/get_collection_info", tags=["collection"])
async def get_collection_info(collection_name: str):
    try:
        # Get collection info
        collection_info = await qdrant_manager.qdrant_client.get_collection(
            collection_name
        )
        print(collection_info)  # Print for debugging
        return {"collection_info": collection_info}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error in get_collection_info: {str(e)}",
        )


@app.post("/train_embedding", tags=["train_embedding"])
async def train_embedding(collection_name: str):
    try:
        # Format images
        train_flyers = formatting_train_images_to_jpg()
        pair_list = embedding.generate_caption(train_flyers)
        image_embeddings = embedding.get_train_image_embeddings(pair_list)

        # Initialize collection (await async call)
        collection_response = await qdrant_manager.initialize_collection(
            collection_name
        )

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
            detail=f"Error in train_embedding: {str(e)}",
        )


@app.post("/get_rag_response", tags=["generate_flyer"])
async def get_rag_response(prompt: Prompt):
    try:
        # Perform similarity search
        response = await rag.get_similar_flyers(prompt)
        if response:
            return {"response": response}
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to retrieve response",
            )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error in get_rag_response: {str(e)}",
        )


# include routers to use dalle llm
app.include_router(api_router)


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
