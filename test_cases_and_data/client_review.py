import requests
import csv

# Define the API endpoints
login_url = "http://localhost:8000/login/"
review_url = "http://localhost:8000/reviews/"

# Define login credentials
login_data = {
    "user_id": "user1",
    "password": "pass123"
}

# Login to get the token
response = requests.post(login_url, json=login_data)
response_data = response.json()

if response.status_code == 200:
    token = response_data.get("token")
    headers = {
        "Authorization": f"Bearer {token}"
    }
    print(token)
    # Load the data from the CSV file
    csv_file_path = "detailed_reviews_ratings.csv"

    with open(csv_file_path, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            review_data = {
                "book_id": int(row["book_id"]),
                "user_id": row["user_id"],
                "review_text": row["review_text"],
                "rating": int(row["rating"])
            }
            print(row)
            # Send the review data to the review API
            response = requests.post(review_url, json=review_data, headers=headers)

            if response.status_code == 200:
                print(f"Successfully inserted review for book_id: {row['book_id']}")
            else:
                print(f"Failed to insert review for book_id: {row['book_id']}. Error: {response.text}")
else:
    print(f"Login failed. Error: {response_data}")
