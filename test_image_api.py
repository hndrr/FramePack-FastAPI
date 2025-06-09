#!/usr/bin/env python3
"""Test script for image generation API"""

import requests
import json
import base64
import time


def test_image_generation():
    """Test single image generation endpoint"""
    print("Testing /api/generate-image endpoint...")

    url = "http://localhost:8080/api/generate-image"
    payload = {
        "prompt": "A beautiful sunset over mountains",
        "negative_prompt": "ugly, blurry",
        "width": 512,
        "height": 512,
        "steps": 20,
        "cfg": 7.5,
        "seed": 42,
    }

    try:
        response = requests.post(url, json=payload)
        print(f"Status Code: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            print(f"Success! Job ID: {result['job_id']}")
            print(f"Message: {result['message']}")
            return result["job_id"]
        else:
            print(f"Error: {response.text}")
            return None
    except Exception as e:
        print(f"Connection error: {e}")
        return None


def check_job_status(job_id):
    """Check job status"""
    print(f"\nChecking status for job {job_id}...")

    url = f"http://localhost:8080/status/{job_id}"

    try:
        response = requests.get(url)
        if response.status_code == 200:
            result = response.json()
            print(f"Status: {result['status']}")
            print(f"Progress: {result.get('progress', 0)}%")
            print(f"Info: {result.get('progress_info', '')}")
            return result
        else:
            print(f"Error: {response.text}")
            return None
    except Exception as e:
        print(f"Connection error: {e}")
        return None


def get_image_result(job_id):
    """Get the generated image"""
    print(f"\nGetting result for job {job_id}...")

    url = f"http://localhost:8080/api/image/{job_id}"

    try:
        response = requests.get(url)
        if response.status_code == 200:
            print("Success! Image downloaded.")
            with open(f"test_output_{job_id}.png", "wb") as f:
                f.write(response.content)
            print(f"Saved as test_output_{job_id}.png")
            return True
        else:
            print(f"Error: {response.text}")
            return False
    except Exception as e:
        print(f"Connection error: {e}")
        return False


def test_batch_generation():
    """Test batch image generation endpoint"""
    print("\nTesting /api/batch-images endpoint...")

    url = "http://localhost:8080/api/batch-images"
    payload = {
        "prompts": ["A red car", "A blue house", "A green tree"],
        "negative_prompt": "ugly, blurry",
        "batch_size": 3,
        "width": 512,
        "height": 512,
        "steps": 20,
    }

    try:
        response = requests.post(url, json=payload)
        print(f"Status Code: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            print(f"Success! Job ID: {result['job_id']}")
            return result["job_id"]
        else:
            print(f"Error: {response.text}")
            return None
    except Exception as e:
        print(f"Connection error: {e}")
        return None


def main():
    print("Image Generation API Test Suite")
    print("=" * 50)

    # Test single image generation
    job_id = test_image_generation()

    if job_id:
        # Wait for processing
        print("\nWaiting for job to complete...")
        max_attempts = 30
        for i in range(max_attempts):
            time.sleep(2)
            status = check_job_status(job_id)
            if status and status["status"] == "completed":
                get_image_result(job_id)
                break
            elif status and status["status"].startswith("failed"):
                print("Job failed!")
                break

    # Test batch generation
    batch_job_id = test_batch_generation()
    if batch_job_id:
        print(f"\nBatch job created: {batch_job_id}")
        # You can add similar status checking here

    print("\nTest complete!")


if __name__ == "__main__":
    main()
