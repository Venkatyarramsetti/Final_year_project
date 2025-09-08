from os import getenv
import logging
from typing import Generator
from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import MongoClient
from bson import ObjectId

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

MONGODB_URI = getenv("MONGODB_URI")
if not MONGODB_URI:
    raise ValueError("MONGODB_URI environment variable is not set")

# MongoDB client
try:
    client = AsyncIOMotorClient(MONGODB_URI)
    db = client.hazard_spotter
    logger.info("Successfully connected to MongoDB")
except Exception as e:
    logger.error(f"Failed to connect to MongoDB: {e}")
    raise

# Collections
users = db.users

# For FastAPI dependency injection
async def get_db() -> Generator:
    try:
        yield db
    finally:
        # Clean up resources if needed
        pass

class User:
    def __init__(self, email: str, name: str, hashed_password: str, _id: ObjectId = None):
        self.id = _id
        self.email = email
        self.name = name
        self.hashed_password = hashed_password

    @classmethod
    async def get_by_email(cls, email: str):
        user_data = await users.find_one({"email": email})
        if user_data:
            return cls(
                _id=user_data.get("_id"),
                email=user_data.get("email"),
                name=user_data.get("name"),
                hashed_password=user_data.get("hashed_password")
            )
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
