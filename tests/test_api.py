import io
import unittest
from PIL import Image
from fastapi.testclient import TestClient
from src.mroot.api import create_app


class FakePredictor:
    def predict(self, image):
        assert image.mode == "RGB"
        return {"rust": 0.5, "scab": 0.2, "leaf spot": 0.15, "black rot": 0.1, "mosaic": 0.05}


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.loads = 0
        def factory():
            self.loads += 1
            return FakePredictor()
        self.client = TestClient(create_app(factory))

    def test_health_does_not_load_models(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["classes"], 39)
        self.assertEqual(self.loads, 0)

    def test_rejects_invalid_and_missing_images(self):
        invalid = self.client.post("/predict", files={"file": ("bad.jpg", b"bad", "image/jpeg")})
        self.assertEqual(invalid.status_code, 400)
        self.assertEqual(self.client.post("/predict").status_code, 422)
        self.assertEqual(self.loads, 0)

    def test_top5_response_and_single_model_load(self):
        stream = io.BytesIO()
        Image.new("RGB", (12, 10), "green").save(stream, format="PNG")
        for _ in range(2):
            response = self.client.post("/predict", files={"file": ("leaf.png", stream.getvalue(), "image/png")})
            self.assertEqual(response.status_code, 200)
            body = response.json()
            self.assertEqual(body["prediction"], "rust")
            self.assertEqual(len(body["top5"]), 5)
            self.assertAlmostEqual(sum(row["probability"] for row in body["top5"]), 1.0)
        self.assertEqual(self.loads, 1)


if __name__ == "__main__":
    unittest.main()
