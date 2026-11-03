import os
import unittest
from unittest.mock import patch

os.environ.setdefault("MONGODB_URI", "mongodb://localhost:27017")
os.environ.setdefault("GEMINI_API_KEY", "test-key-not-used")

from httpx import ASGITransport, AsyncClient

from app.agents.nodes import route_after_classify
from app.main import app
from app.services.applications import _clean


class BackendApiTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.client = AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        )

    async def asyncTearDown(self):
        await self.client.aclose()

    async def test_health_endpoint(self):
        response = await self.client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {"message": "ScholarMate AI Backend Running"},
        )

    async def test_student_email_is_validated(self):
        response = await self.client.post(
            "/students",
            json={"full_name": "Test Student", "email": "not-an-email"},
        )

        self.assertEqual(response.status_code, 422)

    async def test_student_creation_reports_unavailable_database(self):
        payload = {
            "full_name": "Test Student",
            "email": "student@example.com",
        }
        with patch("app.main.get_db", return_value=None):
            response = await self.client.post("/students", json=payload)

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["detail"], "Database not ready")

    async def test_application_status_is_validated(self):
        response = await self.client.patch(
            "/applications/invalid-id",
            json={"status": "not-a-status"},
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("Invalid status", response.json()["detail"])


class BackendLogicTests(unittest.TestCase):
    def test_application_document_is_cleaned_for_api_output(self):
        cleaned = _clean({"_id": "abc123", "status": "interested"})

        self.assertEqual(cleaned, {"status": "interested", "id": "abc123"})

    def test_empty_application_document_is_none(self):
        self.assertIsNone(_clean(None))

    def test_agent_routes_each_supported_tool_intent(self):
        routes = {
            "scholarship_search": "tool",
            "recommend_scholarships": "recommend",
            "match_score": "match_score",
            "eligibility_check": "eligibility",
            "application_plan": "application_plan",
            "track_application": "tracker",
            "prepare_documents": "checklist",
            "write_essay": "essay_draft",
            "review_essay": "essay_review",
        }
        for intent, expected_route in routes.items():
            with self.subTest(intent=intent):
                self.assertEqual(
                    route_after_classify({"tool_name": intent}),
                    expected_route,
                )

    def test_agent_uses_planner_for_general_intent(self):
        self.assertEqual(route_after_classify({"tool_name": None}), "planner")
if __name__ == "__main__":
    unittest.main()
