from fastapi import FastAPI
from api.dalle import router as dalle_router
from fastapi.middleware.cors import CORSMiddleware
from db.database import session_db
from db.database import upsert_flyer_embeddings
from utils.image_tratative import formatting_train_images_to_jpg
from utils.embedding import generate_caption, get_image_embeddings


app = FastAPI(title="Flayer Generator", description="Multimodal Flyer generator")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],

)

# Dependency
db = session_db()


@app.get("/")
def read_root():
    return {"Server is running!"}


@app.post("/train_embedding/", tags=["train"])
async def train_embedding():
    try:
        train_flyers = formatting_train_images_to_jpg()
        pair_list = generate_caption(train_flyers)
        image_embeddings = get_image_embeddings(pair_list)
        await upsert_flyer_embeddings(image_embeddings, pair_list)
    
    except Exception as e:
        raise Exception("Error in train_embedding: ", e)
    
    return {"message": "Embedding trained successfully"}


# include routers to use dalle llm
app.include_router(dalle_router)