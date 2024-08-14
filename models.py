from sqlalchemy import Column, Integer, String, ForeignKey
from sqlalchemy.orm import relationship
from database import Base
import uuid

class Book(Base):
    __tablename__ = "books"

    book_id = Column(Integer, primary_key=True, index=True)
    title = Column(String, index=True)
    author = Column(String, index=True)
    genre = Column(String, index=True)
    year_published = Column(Integer)
    summary = Column(String)

    reviews = relationship("Review", back_populates="book")


class Review(Base):
    __tablename__ = "reviews"

    id = Column(Integer, primary_key=True, index=True)
    book_id = Column(Integer, ForeignKey("books.book_id"))
    user_id = Column(String)
    review_text = Column(String)
    rating = Column(Integer)

    book = relationship("Book", back_populates="reviews")

class User(Base):
    __tablename__ = "users"

    user_id = Column(String, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    password = Column(String)
    token = Column(String, default=str(uuid.uuid4()), unique=True)