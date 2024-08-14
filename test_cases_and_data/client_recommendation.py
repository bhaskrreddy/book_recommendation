import requests
import pandas as pd

# API URLs
LOGIN_URL = "http://localhost:8000/login/"
RECOMMENDATIONS_URL = "http://localhost:8000/recommendations/"

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

# Step 2: Get book recommendations
headers = {"Authorization": f"Bearer {token}"}
response = requests.get(RECOMMENDATIONS_URL, headers=headers, params={"n": 5})

if response.status_code == 200:
    recommendations = response.json()
    recommended_books_df = pd.DataFrame(recommendations)
    print(recommended_books_df)
else:
    print("Failed to get recommendations:", response.json())
