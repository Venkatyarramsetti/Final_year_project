from os import getenv
from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import MongoClient
from bson import ObjectId

# Load environment variables
load_dotenv()

MONGODB_URI = getenv("MONGODB_URI")

# MongoDB client
client = AsyncIOMotorClient(MONGODB_URI)
db = client.hazard_spotter

# Collections
users = db.users

class User:
    def __init__(self, email: str, name: str, hashed_password: str):
        self.email = email
        self.name = name
        self.hashed_password = hashed_password

    @classmethod
    async def get_by_email(cls, email: str):
        user_data = await users.find_one({"email": email})
        if user_data:
            return cls(**user_data)
        return None

    async def save(self):
        user_data = {
            "email": self.email,
            "name": self.name,
            "hashed_password": self.hashed_password
        }
        await users.insert_one(user_data)
        return self

    async def update(self):
        await users.update_one(
            {"email": self.email},
            {"$set": {
                "name": self.name,
                "hashed_password": self.hashed_password
            }}
        )
        return self

async def init_db():
    try:
        await client.server_info()
        # Create unique index on email
        await users.create_index("email", unique=True)
    except Exception as e:
        print(f"Error connecting to MongoDB: {e}")
        raise
