import requests
import csv
import json

csv_file = "books_data.csv"
import requests
import pandas as pd

# API URLs
LOGIN_URL = "http://localhost:8000/login/"
BOOKS_URL = "http://localhost:8000/books_summary/"

# User login credentials
user_login = {
    "user_id": "user2",
    "password": "pass234"
}

# Step 1: Login and get the token
response = requests.post(LOGIN_URL, json=user_login)
if response.status_code == 200:
    token = response.json().get("token")
else:
    print("Login failed:", response.json())
    exit()

# Step 2: post the new book and summary 
headers = {"Authorization": f"Bearer {token}"}
print(headers)
    
with open(csv_file, mode='r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        # Prepare the data payload
        data = {
            "book_id": int(row["book_id"]),
            "title": row["title"],
            "author": row["author"],
            "genre": row["genre"],
            "year_published": row["year_published"],
            "summary": row["summary"]
        }

        # Send POST request to the API
        response = requests.post(BOOKS_URL, json=data, headers=headers)
        if response.status_code == 200:
            print(f"Inserted book {row['title']} with summarized summary.")
        else:
            print(f"Failed to insert book {row['title']}: {response.status_code} - {response.text}")
