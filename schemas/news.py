from pydantic import BaseModel

class NewsInput(BaseModel):
    text: str
