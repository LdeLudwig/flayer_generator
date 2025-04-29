from fastapi.routing import APIRouter
from db.models import Prompt

router = APIRouter(
    prefix="/api",
)

@router.post("/dalle")
def dalle():

    """ TODO:
        - send prompt to dalle
        - get image from dalle
        - save image to db (maybe after results be more accurate)
        - return image
    """
    return {"message": "Hello World"}