"""
Mock scholarship search tool.
In Phase 6 this will hit a real database / RAG store.
For now it returns hard-coded sample data.
"""

SCHOLARSHIPS = [
    {
        "name": "Google AI Scholarship",
        "country": "USA",
        "level": "Masters",
        "field": "Computer Science",
        "deadline": "2026-12-15",
        "amount": "$10,000",
    },
    {
        "name": "Mastercard Foundation Scholars",
        "country": "Africa",
        "level": "Bachelors",
        "field": "Any",
        "deadline": "2026-11-30",
        "amount": "Full tuition",
    },
    {
        "name": "DAAD EPOS",
        "country": "Germany",
        "level": "Masters",
        "field": "Development-related",
        "deadline": "2026-10-31",
        "amount": "€861/month + travel",
    },
]


def search_scholarships(query: str) -> str:
    """
    Very naive search: return scholarships whose field/country
    loosely matches any word in the query.
    """
    query_lower = query.lower()
    matches = []
    for s in SCHOLARSHIPS:
        haystack = f"{s['name']} {s['country']} {s['field']} {s['level']}".lower()
        if any(word in haystack for word in query_lower.split() if len(word) > 2):
            matches.append(s)

    if not matches:
        return "No scholarships matched your query."

    lines = [f"- {m['name']} | {m['country']} | {m['level']} | {m['field']} | Deadline: {m['deadline']}" for m in matches]
    return "Found scholarships:\n" + "\n".join(lines)