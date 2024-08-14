from fastapi import FastAPI, Depends, HTTPException, status, Header
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session, selectinload
from sqlalchemy.ext.asyncio import AsyncSession
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix 
import pandas as pd
import numpy as np
import os, json, uuid, hashlib
from passlib.context import CryptContext
from sqlalchemy.future import select
from models import Book, Review, User
from schemas import BookBase, ReviewBase, UserLogin, UserRegister
from database import get_db, init_db, Base
from summarizer import TextSummarizer

from dotenv import load_dotenv
# Load environment variables from the .env file
load_dotenv()

# Database connection parameters from .env file
host = os.getenv('DB_HOST')
port = os.getenv('DB_PORT')
user = os.getenv('DB_USER')
password = os.getenv('DB_PASSWORD')
database_name = os.getenv('DB_NAME')

# Check if database exists, if not create it
def create_database_if_not_exists():
    engine = create_engine(f"postgresql://{user}:{password}@{host}:{port}/postgres")
    try:
        with engine.connect() as conn:
            conn.execute(text(f"CREATE DATABASE {database_name}"))
            print(f"Database {database_name} created successfully.")
    except OperationalError:
        print(f"Database {database_name} already exists.")
    finally:
        engine.dispose()

# Initialize database and tables
def initialize_db_and_tables():
    create_database_if_not_exists()
    
    # Now connect to the correct database
    engine = create_engine(f"postgresql://{user}:{password}@{host}:{port}/{database_name}")
    
    # Check and create tables if they don't exist
    Base.metadata.create_all(bind=engine)

    # Return the engine and session
    return engine

# FastAPI setup
app = FastAPI()
summarizer = TextSummarizer('config.json')

# Load configuration from config.json
with open("config.json") as config_file:
    config = json.load(config_file)

@app.on_event("startup")
async def on_startup():
    await init_db()

# DATABASE_URL = config["database_url"]
BOOK_SUMMARIZATION_TOKENS = config["book_summarization_tokens"]
REVIEW_SUMMARIZATION_TOKENS = config["review_summarization_tokens"]

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def generate_token():
    return str(uuid.uuid4())

# Hash a plain password
def hash_password(password: str):
    return pwd_context.hash(password)

def hash_password_sha256(password: str) -> str:
    """Hash a password using SHA-256."""
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password_sha256(plain_password: str, hashed_password: str) -> bool:
    """Verify the plain password against the hashed password."""
    return hash_password_sha256(plain_password) == hashed_password

@app.post("/register/")
async def register_user(user_reg: UserRegister, db: AsyncSession = Depends(get_db)):
    db_user = await db.execute(select(User).filter(User.user_id == user_reg.user_id))
    db_user = db_user.scalar_one_or_none()
    
    if db_user:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="User ID already registered")
    
    hashed_password = hash_password_sha256(user_reg.password)
    token = generate_token()
    new_user = User(user_id=user_reg.user_id, username=user_reg.username, password=hashed_password, token = token)

    db.add(new_user)
    await db.commit()
    await db.refresh(new_user)
    return {"message": "User registered successfully"}


# Function to authenticate user and generate UUID token
@app.post("/login/")
async def login(user_login: UserLogin, db: AsyncSession = Depends(get_db)):
    # Fetch the user by user_id
    result = await db.execute(select(User).filter(User.user_id == user_login.user_id))
    user = result.scalar_one_or_none()
    # Check if the user exists and the password matches
    if not user or not verify_password_sha256(user_login.password, user.password):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
    # Generate a new token for the user
    token = str(uuid.uuid4())
    user.token = token
    # Commit the new token to the database
    db.add(user)
    await db.commit()
    return {"token": token}


# Dependency to validate token
async def validate_token(authorization: str = Header(None), db: AsyncSession = Depends(get_db)):
    if not authorization:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authorization header missing")

    token = authorization.split(" ")[1]  # Assumes Bearer token
    user = await db.execute(select(User).filter(User.token == token))
    user = user.scalar_one_or_none()

    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
    return user

@app.post("/books_summary/")
async def create_book(book: BookBase, db: AsyncSession = Depends(get_db), token: str = Depends(validate_token)):
    # Summarize the book's summary
    book_summary = summarizer.process_text(book.summary, BOOK_SUMMARIZATION_TOKENS)

    # Insert the book into the database
    new_book = Book(
        book_id=book.book_id,
        title=book.title,
        author=book.author,
        genre=book.genre,
        year_published=book.year_published,
        summary=book_summary
    )
    db.add(new_book)
    db.commit()
    db.refresh(new_book)
    return new_book

@app.post("/reviews/")
async def create_review(review: ReviewBase, db: AsyncSession = Depends(get_db), token: str = Depends(validate_token)):
    # Summarize the review text
    review_summary = summarizer.process_text(review.review_text, REVIEW_SUMMARIZATION_TOKENS)


    # Insert the summarized review into the database
    new_review = Review(
        book_id=review.book_id,
        user_id=review.user_id,
        review_text=review_summary,
        rating=review.rating
    )
    db.add(new_review)
    await db.commit()
    await db.refresh(new_review)
#     return new_review


#     return recomm# Endpoint to retrieve all books
@app.get("/books/")
async def get_books(db: AsyncSession = Depends(get_db), token: str = Depends(validate_token)):
    books = await db.execute(select(Book))  # Query all books from the database
    return books.scalars().all()  # Return the list of all books

# Endpoint to retrieve a specific book by its ID
@app.get("/books/{id}")
async def get_book(id: int, db: AsyncSession = Depends(get_db), token: str = Depends(validate_token)):
    book = await db.get(Book, id)  # Fetch the book by ID
    if not book:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Book not found")  # Raise 404 if not found
    return book  # Return the retrieved book

# Endpoint to update a book's information by its ID
@app.put("/books/{id}")
async def update_book(id: int, book: BookBase, db: AsyncSession = Depends(get_db), token: str = Depends(validate_token)):
    db_book = await db.get(Book, id)  # Fetch the book by ID
    if not db_book:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Book not found")  # Raise 404 if not found
    
    # Update the book's fields
    db_book.title = book.title
    db_book.author = book.author
    db_book.genre = book.genre
    db_book.year_published = book.year_published
    db_book.summary = book.summary
    await db.commit()  # Commit the transaction
    await db.refresh(db_book)  # Refresh the instance
    return db_book  # Return the updated book

# Endpoint to delete a book by its ID
@app.delete("/books/{id}")
async def delete_book(id: int, db: AsyncSession = Depends(get_db), token: str = Depends(validate_token)):
    book = await db.get(Book, id)  # Fetch the book by ID
    if not book:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Book not found")  # Raise 404 if not found
    await db.delete(book)  # Delete the book from the database
    await db.commit()  # Commit the transaction
    return {"message": "Book deleted successfully"}  # Return a success message

# Endpoint to add a review for a book
@app.post("/books/{id}/reviews")
async def create_review(id: int, review: ReviewBase, db: AsyncSession = Depends(get_db), token: str = Depends(validate_token)):
    book = await db.get(Book, id)  # Fetch the book by ID
    if not book:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Book not found")  # Raise 404 if not found
    
    new_review = Review(
        book_id=id,
        user_id=review.user_id,
        review_text=review.review_text,
        rating=review.rating
    )
    db.add(new_review)  # Add the new review to the session
    await db.commit()  # Commit the transaction
    await db.refresh(new_review)  # Refresh the instance
    return new_review  # Return the created review

# Endpoint to retrieve all reviews for a book
@app.get("/books/{id}/reviews")
async def get_reviews(id: int, db: AsyncSession = Depends(get_db), token: str = Depends(validate_token)):
    reviews = await db.execute(select(Review).where(Review.book_id == id))  # Query all reviews for the book
    return reviews.scalars().all()  # Return the list of reviews

# Endpoint to get a summary and aggregated rating for a book
@app.get("/books/{id}/summary")
async def get_book_summary(id: int, db: AsyncSession = Depends(get_db), token: str = Depends(validate_token)):
    book = await db.get(Book, id)  # Fetch the book by ID
    if not book:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Book not found")  # Raise 404 if not found
    
    reviews = await db.execute(select(Review).where(Review.book_id == id))  # Query all reviews for the book
    reviews_df = pd.DataFrame([{'rating': r.rating} for r in reviews.scalars().all()])
    avg_rating = reviews_df['rating'].mean() if not reviews_df.empty else None  # Calculate the average rating
    return {"summary": book.summary, "average_rating": avg_rating}  # Return the summary and average rating

# Endpoint to get book recommendations based on collaborative filteringended_books.head(n).to_dict(orient='records')
# Endpoint to retrieve all books
# @app.get("/books/")
# async def get_books(db: AsyncSession = Depends(get_db), token: str = Depends(validate_token)):
#     books = await db.execute(select(Book))  # Query all books from the database
#     return books.scalars().all()  # Return the list of all books

# Endpoint to retrieve a specific book by its ID
@app.get("/books/{id}")
async def get_book(id: int, db: AsyncSession = Depends(get_db), token: str = Depends(validate_token)):
    book = await db.get(Book, id)  # Fetch the book by ID
    if not book:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Book not found")  # Raise 404 if not found
    return book  # Return the retrieved book

# Endpoint to update a book's information by its ID
@app.put("/books/{id}")
async def update_book(id: int, book: BookBase, db: AsyncSession = Depends(get_db), token: str = Depends(validate_token)):
    db_book = await db.get(Book, id)  # Fetch the book by ID
    if not db_book:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Book not found")  # Raise 404 if not found
    
    # Update the book's fields
    db_book.title = book.title
    db_book.author = book.author
    db_book.genre = book.genre
    db_book.year_published = book.year_published
    db_book.summary = book.summary
    await db.commit()  # Commit the transaction
    await db.refresh(db_book)  # Refresh the instance
    return db_book  # Return the updated book

# Endpoint to delete a book by its ID
@app.delete("/books/{id}")
async def delete_book(id: int, db: AsyncSession = Depends(get_db), token: str = Depends(validate_token)):
    book = await db.get(Book, id)  # Fetch the book by ID
    if not book:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Book not found")  # Raise 404 if not found
    await db.delete(book)  # Delete the book from the database
    await db.commit()  # Commit the transaction
    return {"message": "Book deleted successfully"}  # Return a success message

# Endpoint to add a review for a book
@app.post("/books/{id}/reviews")
async def create_review(id: int, review: ReviewBase, db: AsyncSession = Depends(get_db), token: str = Depends(validate_token)):
    book = await db.get(Book, id)  # Fetch the book by ID
    if not book:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Book not found")  # Raise 404 if not found
    
    new_review = Review(
        book_id=id,
        user_id=review.user_id,
        review_text=review.review_text,
        rating=review.rating
    )
    db.add(new_review)  # Add the new review to the session
    await db.commit()  # Commit the transaction
    await db.refresh(new_review)  # Refresh the instance
    return new_review  # Return the created review

# Endpoint to retrieve all reviews for a book
@app.get("/books/{id}/reviews")
async def get_reviews(id: int, db: AsyncSession = Depends(get_db), token: str = Depends(validate_token)):
    reviews = await db.execute(select(Review).where(Review.book_id == id))  # Query all reviews for the book
    return reviews.scalars().all()  # Return the list of reviews

# Endpoint to get a summary and aggregated rating for a book
@app.get("/books/{id}/summary")
async def get_book_summary(id: int, db: AsyncSession = Depends(get_db), token: str = Depends(validate_token)):
    book = await db.get(Book, id)  # Fetch the book by ID
    if not book:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Book not found")  # Raise 404 if not found
    
    reviews = await db.execute(select(Review).where(Review.book_id == id))  # Query all reviews for the book
    reviews_df = pd.DataFrame([{'rating': r.rating} for r in reviews.scalars().all()])
    avg_rating = reviews_df['rating'].mean() if not reviews_df.empty else None  # Calculate the average rating
    return {"summary": book.summary, "average_rating": avg_rating}  # Return the summary and average rating

# Endpoint to get book recommendations based on collaborative filtering
@app.get("/recommendations/")
async def recommend_books(token: str = Depends(validate_token), db: AsyncSession = Depends(get_db), n: int = 5):
    user = token.user_id  # The validated user

    # Fetch book and review data from the database
    books = await db.execute(select(Book))
    reviews = await db.execute(select(Review))
    books_df = pd.DataFrame([{'book_id': b.book_id, 'title': b.title, 'genre': b.genre} for b in books.scalars().all()])
    reviews_df = pd.DataFrame([{'user_id': r.user_id, 'book_id': r.book_id, 'rating': r.rating} for r in reviews.scalars().all()])

    # Calculate average rating for each book
    book_ratings = reviews_df.groupby('book_id')['rating'].mean().reset_index()
    book_ratings.columns = ['book_id', 'average_rating']

    # Merge the books data with their average ratings
    books_df = pd.merge(books_df, book_ratings, on='book_id')

    # Create a user-item interaction matrix (user_id x book_id)
    user_item_matrix = reviews_df.pivot_table(index='user_id', columns='book_id', values='rating')

    # Fill NaN values with zeros (no interaction)
    user_item_matrix.fillna(0, inplace=True)

    # Convert to sparse matrix format for efficiency
    user_item_sparse = csr_matrix(user_item_matrix.values)

    # Calculate cosine similarity between users
    user_similarity = cosine_similarity(user_item_sparse)

    # Calculate cosine similarity between books based on genre
    genre_tfidf = TfidfVectorizer(stop_words='english')
    genre_matrix = genre_tfidf.fit_transform(books_df['genre'])
    genre_similarity = cosine_similarity(genre_matrix)

    # Find the user's similarity scores with all other users
    user_index = user_item_matrix.index.get_loc(user)
    sim_scores = user_similarity[user_index]

    # Calculate weighted ratings based on user similarity
    weighted_ratings = sim_scores.dot(user_item_matrix.values) / np.array([np.abs(sim_scores).sum()])

    # Adjust by genre similarity
    genre_weighted_ratings = weighted_ratings.dot(genre_similarity) / np.array([np.abs(genre_similarity).sum()])

    # Get book indices sorted by the highest scores
    book_indices = np.argsort(genre_weighted_ratings)[::-1]

    # Filter out books that the user has already bought or reviewed
    user_reviewed_books = reviews_df[reviews_df['user_id'] == user]['book_id'].tolist()
    recommended_books = books_df.iloc[book_indices].loc[~books_df['book_id'].isin(user_reviewed_books)]

    # Sort the filtered books by their average rating in descending order
    recommended_books = recommended_books.sort_values(by='average_rating', ascending=False)

    # Return the top n recommended books
    return recommended_books.head(n).to_dict(orient='records')
