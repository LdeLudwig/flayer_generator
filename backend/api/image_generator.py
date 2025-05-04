from PIL import Image
from fastapi import APIRouter, HTTPException, status
from langchain.prompts import PromptTemplate
from db.models import Prompt
from utils.RAG import RAG


router = APIRouter(
    prefix="/api",
)

# Initialize RAG instance
rag = RAG()


@router.post("/dalle")
def dalle():
    pass
    