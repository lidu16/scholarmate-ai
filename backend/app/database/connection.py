from motor.motor_asyncio import AsyncIOMotorClient
from app.config.settings import MONGODB_URI, DATABASE_NAME

client: AsyncIOMotorClient = None
db = None


async def connect_to_mongo():
    global client, db
    client = AsyncIOMotorClient(
        MONGODB_URI,
        serverSelectionTimeoutMS=5000,
        connectTimeoutMS=10000
    )
    db = client[DATABASE_NAME]
    await client.admin.command("ping")
    print(f"Connected to MongoDB Atlas -> database: {DATABASE_NAME}")


async def close_mongo_connection():
    global client
    if client:
        client.close()


def get_db():
    return db
    