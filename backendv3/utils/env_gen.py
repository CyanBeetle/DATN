import secrets
import os

# Generate a secure SECRET_KEY
secret_key = secrets.token_hex(32)

# Get the directory of the script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Define the .env file path
env_path = os.path.join(script_dir, ".env")

# Define the .env content
env_content = f"""MONGO_CONNECTION_STRING=mongodb+srv://long:long@traffic.pwuvq.mongodb.net/Traffic?retryWrites=true&w=majority
JWT_ALGORITHM=HS256
DBNAME=Traffic
SECRET_KEY={secret_key}
"""

# Write to .env file in the same directory as the script
with open(env_path, "w") as env_file:
    env_file.write(env_content)

print(f".env file has been generated at {env_path} with a secure SECRET_KEY.")
