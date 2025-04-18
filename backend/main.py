from fastapi import FastAPI
from api.dalle import router as dalle_router
from fastapi.middleware.cors import CORSMiddleware
from db.database import session_db


app = FastAPI(title="Flayer Generator", description="Multimodal Flyer generator")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],

)

app.include_router(dalle_router)

# Dependency
db = session_db()


@app.get("/")
def read_root():
    return {"Server is running!"}
