import os
from dotenv import load_dotenv

load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI")
DATABASE_NAME = os.getenv("DATABASE_NAME", "scholarmate")

if not MONGODB_URI:
    raise RuntimeError("MONGODB_URI is not set in .env")