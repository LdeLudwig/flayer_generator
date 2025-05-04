from pydantic import BaseModel
from typing import Optional, List

class Prompt(BaseModel):
    text: Optional[str] = None
    images: Optional[List[str]] = None