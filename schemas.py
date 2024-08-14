from pydantic import BaseModel, validator
from utils import extract_year

class BookBase(BaseModel):
    book_id:int
    title: str
    author: str
    genre: str
    year_published: int
    summary: str

    # @validator('year_published', pre=True, always=True)
    # def preprocess_year(cls, v):
    #     print(type(v))
    #     # Call extract_year to handle different formats
    #     return extract_year(v) if isinstance(v, str) else v
    class Config:
        orm_mode = True


class ReviewBase(BaseModel):
    book_id: int
    user_id: str
    review_text: str
    rating: int

    class Config:
        orm_mode = True

class UserLogin(BaseModel):
    user_id: str
    password: str

class UserRegister(BaseModel):
    user_id: str
    username: str
    password: str