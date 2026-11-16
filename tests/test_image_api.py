"""
Test cases for image generation API endpoints
"""
import pytest
import json
import base64
import io
from PIL import Image
from fastapi.testclient import TestClient
from api.api import app

client = TestClient(app)


def create_test_image_base64(width: int = 512, height: int = 512) -> str:
    """Create a test image encoded as base64"""
    image = Image.new('RGB', (width, height), color='red')
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    image_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return f"data:image/png;base64,{image_data}"


class TestImageGenerationAPI:
    """Test class for image generation endpoints"""
    
    def test_generate_image_endpoint_exists(self):
        """Test that the image generation endpoint exists"""
        response = client.post("/api/generate-image", json={
            "prompt": "A beautiful landscape"
        })
        # Should not return 404 (endpoint exists)
        assert response.status_code != 404
    
    def test_generate_image_with_minimal_params(self):
        """Test image generation with minimal parameters"""
        request_data = {
            "prompt": "A cute cat sitting on a chair"
        }
        
        response = client.post("/api/generate-image", json=request_data)
        
        # Should accept the request (return job_id)
        if response.status_code == 200:
            data = response.json()
            assert "job_id" in data
            assert "message" in data
            assert len(data["job_id"]) > 0
        else:
            # If models aren't loaded, expect 503
            assert response.status_code == 503
    
    def test_generate_image_with_full_params(self):
        """Test image generation with all parameters"""
        request_data = {
            "prompt": "A detailed fantasy landscape with mountains and rivers",
            "negative_prompt": "blurry, low quality, distorted",
            "seed": 42,
            "steps": 25,
            "cfg": 7.5,
            "width": 768,
            "height": 512,
            "lora_paths": [],
            "lora_scales": []
        }
        
        response = client.post("/api/generate-image", json=request_data)
        
        if response.status_code == 200:
            data = response.json()
            assert "job_id" in data
            assert "message" in data
        else:
            assert response.status_code == 503  # Models not loaded
    
    def test_batch_images_endpoint(self):
        """Test batch image generation endpoint"""
        request_data = {
            "prompts": [
                "A red apple",
                "A blue car",
                "A green tree"
            ],
            "batch_size": 3,
            "steps": 20
        }
        
        response = client.post("/api/batch-images", json=request_data)
        
        if response.status_code == 200:
            data = response.json()
            assert "job_id" in data
            assert "message" in data
        else:
            assert response.status_code == 503
    
    def test_image_transfer_endpoint(self):
        """Test image transfer endpoint"""
        source_image = create_test_image_base64(256, 256)
        target_image = create_test_image_base64(256, 256)
        
        request_data = {
            "source_image": source_image,
            "target_image": target_image,
            "prompt": "Transfer the style from source to target",
            "transfer_strength": 0.8
        }
        
        response = client.post("/api/transfer-image", json=request_data)
        
        if response.status_code == 200:
            data = response.json()
            assert "job_id" in data
            assert "message" in data
        else:
            assert response.status_code == 503
    
    def test_image_download_endpoint(self):
        """Test image download endpoint"""
        # Test with a non-existent job ID
        response = client.get("/api/image/nonexistent123")
        assert response.status_code == 404
    
    def test_invalid_image_generation_params(self):
        """Test image generation with invalid parameters"""
        # Test with empty prompt
        response = client.post("/api/generate-image", json={
            "prompt": ""
        })
        # Should either accept empty prompt or return validation error
        assert response.status_code in [200, 422, 503]
        
        # Test with negative steps
        response = client.post("/api/generate-image", json={
            "prompt": "test",
            "steps": -5
        })
        # Should return validation error
        assert response.status_code == 422
        
        # Test with invalid CFG
        response = client.post("/api/generate-image", json={
            "prompt": "test",
            "cfg": -1.0
        })
        # Should return validation error
        assert response.status_code == 422
    
    def test_batch_images_validation(self):
        """Test batch image generation validation"""
        # Test with empty prompts list
        response = client.post("/api/batch-images", json={
            "prompts": []
        })
        # Should return validation error
        assert response.status_code == 422
        
        # Test with too large batch size
        response = client.post("/api/batch-images", json={
            "prompts": ["test1", "test2"],
            "batch_size": 20  # Exceeds max of 8
        })
        # Should return validation error
        assert response.status_code == 422
    
    def test_image_transfer_validation(self):
        """Test image transfer validation"""
        # Test with invalid base64 image
        response = client.post("/api/transfer-image", json={
            "source_image": "invalid_base64",
            "target_image": "invalid_base64",
            "prompt": "test"
        })
        # Should either return validation error or process with error
        assert response.status_code in [422, 503]
        
        # Test with invalid transfer strength
        source_image = create_test_image_base64()
        target_image = create_test_image_base64()
        
        response = client.post("/api/transfer-image", json={
            "source_image": source_image,
            "target_image": target_image,
            "prompt": "test",
            "transfer_strength": 2.0  # Should be between 0.0 and 1.0
        })
        # Should return validation error
        assert response.status_code == 422


if __name__ == "__main__":
    pytest.main([__file__, "-v"])