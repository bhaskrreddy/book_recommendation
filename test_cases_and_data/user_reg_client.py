import requests

# Define the API endpoint for adding users
api_url = "http://localhost:8000/register/"

# Define the users to be added
users = [
    {"user_id": "user1", "username":"John", "password": "pass123"},
    {"user_id": "user2", "username":"Smith", "password": "pass234"},
    {"user_id": "user3", "username":"Linda", "password": "pass456"}
]

# Function to add users
def add_user(user_data):
    response = requests.post(api_url, json=user_data)
    if response.status_code == 200:
        print(f"User {user_data['user_id']} added successfully.")
    else:
        print(f"Failed to add user {user_data['user_id']}. Status code: {response.status_code}")

# Insert users into the database
for user in users:
    add_user(user)
