import pytest
import httpx
from httpx import AsyncClient

BASE_URL = "http://127.0.0.1:8000"


@pytest.fixture
async def client():
    """Fixture to create an AsyncClient for the tests."""
    async with AsyncClient(base_url=BASE_URL) as client:
        yield client


@pytest.mark.asyncio
async def test_login(client):
    """
    Test the login functionality.
    
    This test checks if a user can successfully login and obtain a token.
    """
    login_data = {"user_id": "1", "password": "pass123"}
    response = await client.post("/login/", json=login_data)
    
    assert response.status_code == 200
    assert "token" in response.json()


@pytest.mark.asyncio
async def test_add_book(client):
    """
    Test the add book functionality.
    
    This test checks if a user can add a new book to the database.
    """
    # First, login to get a token
    login_data = {"user_id": "1", "password": "pass123"}
    login_response = await client.post("/login/", json=login_data)
    token = login_response.json().get("token")
    headers = {"Authorization": f"Bearer {token}"}
    
    # Add a new book
    new_book = {
        "title": "New Book",
        "author": "Author Name",
        "genre": "Fiction",
        "year_published": 2023,
        "summary": "This is a new book."
    }
    response = await client.post("/books/", json=new_book, headers=headers)
    
    assert response.status_code == 200
    assert response.json()["title"] == "New Book"


@pytest.mark.asyncio
async def test_get_all_books(client):
    """
    Test the get all books functionality.
    
    This test checks if a user can retrieve a list of all books from the database.
    """
    # First, login to get a token
    login_data = {"user_id": "1", "password": "pass123"}
    login_response = await client.post("/login/", json=login_data)
    token = login_response.json().get("token")
    headers = {"Authorization": f"Bearer {token}"}
    
    # Get all books
    response = await client.get("/books/", headers=headers)
    
    assert response.status_code == 200
    assert isinstance(response.json(), list)


@pytest.mark.asyncio
async def test_get_book_by_id(client):
    """
    Test the get book by ID functionality.
    
    This test checks if a user can retrieve a specific book by its ID.
    """
    # First, login to get a token
    login_data = {"user_id": "1", "password": "pass123"}
    login_response = await client.post("/login/", json=login_data)
    token = login_response.json().get("token")
    headers = {"Authorization": f"Bearer {token}"}
    
    # Assuming there's at least one book with ID 1
    response = await client.get("/books/1", headers=headers)
    
    assert response.status_code == 200
    assert response.json()["book_id"] == 1


@pytest.mark.asyncio
async def test_update_book(client):
    """
    Test the update book functionality.
    
    This test checks if a user can update the details of an existing book.
    """
    # First, login to get a token
    login_data = {"user_id": "1", "password": "pass123"}
    login_response = await client.post("/login/", json=login_data)
    token = login_response.json().get("token")
    headers = {"Authorization": f"Bearer {token}"}
    
    # Update an existing book
    update_book_data = {
        "title": "Updated Book",
        "author": "Updated Author",
        "genre": "Non-fiction",
        "year_published": 2024,
        "summary": "This is an updated book."
    }
    response = await client.put("/books/1", json=update_book_data, headers=headers)
    
    assert response.status_code == 200
    assert response.json()["title"] == "Updated Book"


@pytest.mark.asyncio
async def test_delete_book(client):
    """
    Test the delete book functionality.
    
    This test checks if a user can delete a book by its ID.
    """
    # First, login to get a token
    login_data = {"user_id": "1", "password": "pass123"}
    login_response = await client.post("/login/", json=login_data)
    token = login_response.json().get("token")
    headers = {"Authorization": f"Bearer {token}"}
    
    # Delete a book by ID
    response = await client.delete("/books/1", headers=headers)
    
    assert response.status_code == 200
    assert "message" in response.json()


@pytest.mark.asyncio
async def test_add_review(client):
    """
    Test the add review functionality.
    
    This test checks if a user can add a review for a specific book.
    """
    # First, login to get a token
    login_data = {"user_id": "1", "password": "pass123"}
    login_response = await client.post("/login/", json=login_data)
    token = login_response.json().get("token")
    headers = {"Authorization": f"Bearer {token}"}
    
    # Add a review for a book
    new_review = {
        "book_id": 1,
        "user_id": 1,
        "review_text": "This is a review.",
        "rating": 5
    }
    response = await client.post("/books/1/reviews", json=new_review, headers=headers)
    
    assert response.status_code == 200
    assert response.json()["review_text"] == "This is a review."


@pytest.mark.asyncio
async def test_get_reviews_for_book(client):
    """
    Test the get reviews for a book functionality.
    
    This test checks if a user can retrieve all reviews for a specific book.
    """
    # First, login to get a token
    login_data = {"user_id": "1", "password": "pass123"}
    login_response = await client.post("/login/", json=login_data)
    token = login_response.json().get("token")
    headers = {"Authorization": f"Bearer {token}"}
    
    # Get all reviews for a book
    response = await client.get("/books/1/reviews", headers=headers)
    
    assert response.status_code == 200
    assert isinstance(response.json(), list)


@pytest.mark.asyncio
async def test_get_book_summary_and_rating(client):
    """
    Test the get summary and aggregated rating for a book functionality.
    
    This test checks if a user can retrieve the summary and aggregated rating of a specific book.
    """
    # First, login to get a token
    login_data = {"user_id": "1", "password": "pass123"}
    login_response = await client.post("/login/", json=login_data)
    token = login_response.json().get("token")
    headers = {"Authorization": f"Bearer {token}"}
    
    # Get the summary and aggregated rating for a book
    response = await client.get("/books/1/summary", headers=headers)
    
    assert response.status_code == 200
    assert "summary" in response.json()
    assert "rating" in response.json()


@pytest.mark.asyncio
async def test_get_recommendations(client):
    """
    Test the get book recommendations functionality.
    
    This test checks if a user can retrieve a list of recommended books.
    """
    # First, login to get a token
    login_data = {"user_id": "1", "password": "pass123"}
    login_response = await client.post("/login/", json=login_data)
    token = login_response.json().get("token")
    headers = {"Authorization": f"Bearer {token}"}
    
    # Get book recommendations
    response = await client.get("/recommendations/", headers=headers)
    
    assert response.status_code == 200
    assert isinstance(response.json(), list)


@pytest.mark.asyncio
async def test_generate_summary(client):
    """
    Test the generate summary functionality.
    
    This test checks if a user can generate a summary for a given book content.
    """
    # First, login to get a token
    login_data = {"user_id": "1", "password": "pass123"}
    login_response = await client.post("/login/", json=login_data)
    token = login_response.json().get("token")
    headers = {"Authorization": f"Bearer {token}"}
    
    # Generate a summary for given book content
    book_content = {"summary": "This is the content of the book."}
    response = await client.post("/generate-summary/", json=book_content, headers=headers)
    
    assert response.status_code == 200
    assert "summary" in response.json()
