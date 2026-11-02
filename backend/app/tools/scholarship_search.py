"""
Scholarship search tool — queries MongoDB Atlas.
Async because Motor is async and the agent runs inside FastAPI's event loop.
"""

from app.database.connection import get_db


async def search_scholarships(query: str) -> str:
    """
    Search scholarships by free-text query.
    1) Try MongoDB full-text search (uses the text index on name/description/field).
    2) Fall back to regex match on name/country/field/level if no text hits.
    """
    db = get_db()
    if db is None:
        return "Database not connected."

    coll = db.scholarships

    # Pass 1: text index search
    try:
        cursor = coll.find(
            {"$text": {"$search": query}},
            {"score": {"$meta": "textScore"}},
        ).sort([("score", {"$meta": "textScore"})]).limit(5)
        results = await cursor.to_list(length=5)
    except Exception:
        results = []

    # Pass 2: regex fallback
    if not results:
        words = [w for w in query.split() if len(w) > 2]
        if words:
            pattern = "|".join(words)
            cursor = coll.find({
                "$or": [
                    {"name": {"$regex": pattern, "$options": "i"}},
                    {"country": {"$regex": pattern, "$options": "i"}},
                    {"field": {"$regex": pattern, "$options": "i"}},
                    {"level": {"$regex": pattern, "$options": "i"}},
                    {"description": {"$regex": pattern, "$options": "i"}},
                ]
            }).limit(5)
            results = await cursor.to_list(length=5)

    if not results:
        return "No scholarships matched your query."

    lines = [
        f"- {r['name']} | {r['country']} | {r['level']} | {r['field']} | "
        f"Deadline: {r['deadline']} | Amount: {r['amount']}"
        for r in results
    ]
    return "Found scholarships:\n" + "\n".join(lines)