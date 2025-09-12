from os import getenv
import logging
import time
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

# MongoDB connection with retry mechanism
def get_mongodb_client():
    MONGODB_URI = getenv("MONGODB_URI")
    if not MONGODB_URI:
        logger.warning("MONGODB_URI environment variable is not set. Using local fallback for development.")
        MONGODB_URI = "mongodb://localhost:27017/hazard_spotter"
    
    max_retries = 5
    retry_delay = 5  # seconds
    
    for attempt in range(max_retries):
        try:
            client = AsyncIOMotorClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
            # This will validate the connection
            client.admin.command('ping')
            logger.info("Successfully connected to MongoDB")
            return client
        except Exception as e:
            if attempt < max_retries - 1:
                logger.warning(f"Failed to connect to MongoDB (attempt {attempt+1}/{max_retries}): {e}")
                logger.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                logger.error(f"Failed to connect to MongoDB after {max_retries} attempts: {e}")
                # In production, we'll still return a client and let the application handle errors later
                # This allows the app to start even if MongoDB is temporarily unavailable
                return AsyncIOMotorClient(MONGODB_URI)

# Initialize client
client = get_mongodb_client()
db = client.hazard_spotter

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
        try:
            user_data = await users.find_one({"email": email})
            if user_data:
                return cls(
                    _id=user_data.get("_id"),
                    email=user_data.get("email"),
                    name=user_data.get("name"),
                    hashed_password=user_data.get("hashed_password")
                )
            return None
        except Exception as e:
            logger.error(f"Error retrieving user by email: {e}")
            return None

    async def save(self):
        try:
            user_data = {
                "email": self.email,
                "name": self.name,
                "hashed_password": self.hashed_password
            }
            await users.insert_one(user_data)
            return self
        except Exception as e:
            logger.error(f"Error saving user: {e}")
            raise

    async def update(self):
        try:
            await users.update_one(
                {"email": self.email},
                {"$set": {
                    "name": self.name,
                    "hashed_password": self.hashed_password
                }}
            )
            return self
        except Exception as e:
            logger.error(f"Error updating user: {e}")
            raise

async def init_db():
    try:
        if client:
            await client.admin.command('ping')
            # Create unique index on email
            await users.create_index("email", unique=True)
            logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        # Don't raise exception here to allow app to start even with DB issues
