import requests

BASE_URL = "http://127.0.0.1:8000"

class APIClient:
    def __init__(self, base_url):
        self.base_url = base_url
        self.headers = {}

    def login(self, user_id: str, password: str) -> None:
        """Login and store the token for authentication."""
        login_data = {"user_id": user_id, "password": password}
        response = requests.post(f"{self.base_url}/login/", json=login_data)
        response.raise_for_status()
        token = response.json().get("token")
        self.headers = {"Authorization": f"Bearer {token}"}

    def get_all_books(self) -> list:
        """Get all books."""
        response = requests.get(f"{self.base_url}/books/", headers=self.headers)
        response.raise_for_status()
        return response.json()

    def get_book_by_id(self, book_id: int) -> dict:
        """Get a book by ID."""
        response = requests.get(f"{self.base_url}/books/{book_id}", headers=self.headers)
        response.raise_for_status()
        return response.json()

    def delete_book(self, book_id: int) -> dict:
        """Delete a book by ID."""
        response = requests.delete(f"{self.base_url}/books/{book_id}", headers=self.headers)
        response.raise_for_status()
        return response.json()

    def add_review(self, book_id: int, review_data: dict) -> dict:
        """Add a review for a book."""
        response = requests.post(f"{self.base_url}/books/{book_id}/reviews", json=review_data, headers=self.headers)
        response.raise_for_status()
        return response.json()

    def get_reviews_for_book(self, book_id: int) -> list:
        """Get all reviews for a book."""
        response = requests.get(f"{self.base_url}/books/{book_id}/reviews", headers=self.headers)
        response.raise_for_status()
        return response.json()

    def get_book_summary_and_rating(self, book_id: int) -> dict:
        """Get summary and aggregated rating for a book."""
        response = requests.get(f"{self.base_url}/books/{book_id}/summary", headers=self.headers)
        response.raise_for_status()
        return response.json()


# Usage Example:

client = APIClient(BASE_URL)

# 1. Login and get the token
user_id = 'user1'
password = 'pass123'
client.login(user_id=user_id, password=password)


# 2. Get all books
books = client.get_all_books()
print(books)

# 3. Get a book by ID
book = client.get_book_by_id(33790)
print(book)

# 4. Delete a book by ID
deleted_book_response = client.delete_book(67838)
print(deleted_book_response)

# 5. Add a review for a book
new_review = {
    "book_id": 1,
    "user_id": 'user3',
    "review_text": "This is a testing review.",
    "rating": 5
}
added_review = client.add_review(1, new_review)
print(added_review)

# 6. Get all reviews for a book
reviews = client.get_reviews_for_book(64010)
print(reviews)

# 7. Get summary and aggregated rating for a book
summary_and_rating = client.get_book_summary_and_rating(64010)
print(summary_and_rating)

