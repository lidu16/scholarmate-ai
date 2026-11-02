"""
Seed the `scholarships` collection in MongoDB Atlas.
Run once:  python -m app.database.seed_scholarships
Idempotent: re-running updates existing docs instead of duplicating.
"""

import asyncio
from motor.motor_asyncio import AsyncIOMotorClient
from app.config.settings import MONGODB_URI, DATABASE_NAME


SCHOLARSHIPS = [
    {
        "slug": "google-ai-scholarship",
        "name": "Google AI Scholarship",
        "provider": "Google",
        "country": "USA",
        "level": "Masters",
        "field": "Computer Science",
        "deadline": "2026-12-15",
        "amount": "$10,000",
        "description": "Supports graduate students pursuing AI-related degrees.",
    },
    {
        "slug": "mastercard-foundation-scholars",
        "name": "Mastercard Foundation Scholars",
        "provider": "Mastercard Foundation",
        "country": "Africa",
        "level": "Bachelors",
        "field": "Any",
        "deadline": "2026-11-30",
        "amount": "Full tuition",
        "description": "Full scholarship for academically talented yet economically disadvantaged African youth.",
    },
    {
        "slug": "daad-epos",
        "name": "DAAD EPOS",
        "provider": "DAAD",
        "country": "Germany",
        "level": "Masters",
        "field": "Development-related",
        "deadline": "2026-10-31",
        "amount": "€861/month + travel",
        "description": "Development-related postgraduate courses in Germany for students from developing countries.",
    },
    {
        "slug": "chevening",
        "name": "Chevening Scholarship",
        "provider": "UK Government",
        "country": "United Kingdom",
        "level": "Masters",
        "field": "Any",
        "deadline": "2026-11-05",
        "amount": "Full tuition + stipend",
        "description": "UK government's global scholarship programme for future leaders.",
    },
    {
        "slug": "erasmus-mundus",
        "name": "Erasmus Mundus Joint Masters",
        "provider": "European Commission",
        "country": "Europe",
        "level": "Masters",
        "field": "Any",
        "deadline": "2027-01-15",
        "amount": "€1,400/month + travel",
        "description": "Prestigious joint master's programmes across European universities.",
    },
    {
        "slug": "fulbright",
        "name": "Fulbright Foreign Student Program",
        "provider": "US Department of State",
        "country": "USA",
        "level": "Masters",
        "field": "Any",
        "deadline": "2026-10-15",
        "amount": "Full tuition + living",
        "description": "Graduate study in the US for international students.",
    },
    {
        "slug": "commonwealth-masters",
        "name": "Commonwealth Master's Scholarship",
        "provider": "Commonwealth Scholarship Commission",
        "country": "United Kingdom",
        "level": "Masters",
        "field": "Development-related",
        "deadline": "2026-12-01",
        "amount": "Full tuition + stipend",
        "description": "For talented students from low/middle-income Commonwealth countries.",
    },
    {
        "slug": "mandela-rhodes",
        "name": "Mandela Rhodes Scholarship",
        "provider": "Mandela Rhodes Foundation",
        "country": "South Africa",
        "level": "Masters",
        "field": "Any",
        "deadline": "2026-05-31",
        "amount": "Full tuition + stipend",
        "description": "Leadership-focused scholarship for African postgraduate students.",
    },
    {
        "slug": "gates-cambridge",
        "name": "Gates Cambridge Scholarship",
        "provider": "Gates Foundation",
        "country": "United Kingdom",
        "level": "PhD",
        "field": "Any",
        "deadline": "2026-12-03",
        "amount": "Full cost + stipend",
        "description": "Full-cost scholarship for outstanding international students at Cambridge.",
    },
    {
        "slug": "ethiopia-moe",
        "name": "Ethiopia Ministry of Education Scholarship",
        "provider": "Ethiopian MoE",
        "country": "Ethiopia",
        "level": "Masters",
        "field": "STEM",
        "deadline": "2026-09-30",
        "amount": "Partial funding",
        "description": "Domestic postgraduate scholarships for Ethiopian students in STEM fields.",
    },
]


async def seed():
    client = AsyncIOMotorClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
    db = client[DATABASE_NAME]
    coll = db.scholarships

    # Create useful indexes
    await coll.create_index("slug", unique=True)
    await coll.create_index([("country", 1), ("level", 1)])
    await coll.create_index([("name", "text"), ("description", "text"), ("field", "text")])

    for s in SCHOLARSHIPS:
        await coll.update_one(
            {"slug": s["slug"]},
            {"$set": s},
            upsert=True,
        )

    total = await coll.count_documents({})
    print(f"Seeded {len(SCHOLARSHIPS)} scholarships. Total in collection: {total}")

    client.close()


if __name__ == "__main__":
    asyncio.run(seed())