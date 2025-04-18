from fastapi.routing import APIRouter
from ..db.models import Prompt

route = APIRouter(
    prefix="/api",
)

@route.get("/dalle")
def dalle():
    """ TODO:
        - send prompt to dalle
        - get image from dalle
        - save image to db
        - return image
    """
    return {"message": "Hello World"}

