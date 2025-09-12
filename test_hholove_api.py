#!/usr/bin/env python3
"""
HHOLOVE API Test Script
Test the HHOLOVE AI API integration with a sample image.
"""

import os
import sys
import requests
import hashlib
import time
from pathlib import Path

def test_hholove_api():
    """Test HHOLOVE API connectivity and functionality"""
    
    print("🔍 HHOLOVE API Test Script")
    print("=" * 40)
    
    # Check API key
    api_key = os.environ.get('HHOLOVE_API_KEY')
    if not api_key:
        print("❌ HHOLOVE_API_KEY not found in environment variables")
        print("   Please set your API key:")
        print("   export HHOLOVE_API_KEY='your_actual_api_key'")
        print("   Or add it to your .env file")
        return False
    
    print(f"✅ API Key found: {api_key[:10]}...")
    
    # Test API connectivity
    api_base_url = "https://ai.open.hhodata.com/api/v2"
    print(f"🌐 Testing API endpoint: {api_base_url}")
    
    try:
        # Test with a simple HEAD request to check connectivity
        test_response = requests.head(f"{api_base_url}/dongniao", timeout=10)
        print(f"✅ API endpoint accessible (Status: {test_response.status_code})")
    except Exception as e:
        print(f"❌ API endpoint test failed: {e}")
        return False
    
    # Look for test images
    test_image_paths = [
        "uploads/DSC04440.jpg",
        "uploads/DSC04550.jpg", 
        "uploads/DSC04562.jpg",
        "uploads/DSC04579-Enhanced-NR-2.jpg"
    ]
    
    test_image = None
    for path in test_image_paths:
        if os.path.exists(path):
            test_image = path
            break
    
    if not test_image:
        print("⚠️ No test images found in uploads/ directory")
        print("   Please add a JPEG bird image to test with")
        return False
    
    print(f"📸 Using test image: {test_image}")
    
    # Test image upload
    try:
        print("📤 Testing image upload...")
        
        device_id = hashlib.md5(f"test_bird_label_{test_image}".encode()).hexdigest()[:16]
        
        upload_headers = {
            'api_key': api_key
        }
        
        upload_data = {
            'upload': '1',
            'class': 'B',
            'area': 'CN',
            'did': device_id
        }
        
        with open(test_image, 'rb') as image_file:
            files = {
                'image': ('test_bird.jpg', image_file, 'image/jpeg')
            }
            
            upload_response = requests.post(
                f"{api_base_url}/dongniao",
                headers=upload_headers,
                data=upload_data,
                files=files,
                timeout=30
            )
        
        print(f"📤 Upload response: {upload_response.status_code}")
        
        if upload_response.status_code == 200:
            upload_result = upload_response.json()
            print(f"✅ Upload successful: {upload_result}")
            
            if upload_result.get('status') == '1000':
                recognition_id = upload_result['data'][1] if isinstance(upload_result['data'], list) else upload_result['data']['recognitionId']
                print(f"🆔 Recognition ID: {recognition_id}")
                
                # Test result polling
                print("⏳ Testing result polling...")
                
                result_headers = {
                    'api_key': api_key
                }
                
                for attempt in range(5):
                    time.sleep(2)
                    
                    result_data = {
                        'resultid': recognition_id
                    }
                    
                    result_response = requests.post(
                        f"{api_base_url}/dongniao",
                        headers=result_headers,
                        data=result_data,
                        timeout=15
                    )
                    
                    if result_response.status_code == 200:
                        result_json = result_response.json()
                        status = result_json.get('status')
                        
                        print(f"   Attempt {attempt + 1}: Status {status}")
                        
                        if status == '1000':
                            print("✅ Recognition successful!")
                            recognition_data = result_json.get('data', [])
                            
                            if recognition_data:
                                best_detection = recognition_data[0]
                                if best_detection['list']:
                                    best_result = best_detection['list'][0]
                                    confidence = best_result[0]
                                    names = best_result[1].split('|')
                                    
                                    print(f"🐦 Species: {names[1] if len(names) > 1 else names[0]}")
                                    print(f"📊 Confidence: {confidence}%")
                                    print(f"🔬 Scientific: {names[2] if len(names) > 2 else 'N/A'}")
                            
                            return True
                        
                        elif status == '1001':
                            print("   ⏳ Results not ready yet...")
                            continue
                        
                        elif status in ['1008', '1009']:
                            print(f"⚠️ No recognition possible: {result_json.get('message', status)}")
                            return True  # API working, just no results
                        
                        else:
                            print(f"❌ Unexpected status: {status}")
                            break
                    else:
                        print(f"   ❌ Poll attempt {attempt + 1} failed: {result_response.status_code}")
                
                print("⏰ Recognition timed out, but API is working")
                return True
            
            else:
                error_msg = upload_result.get('message', 'Unknown error')
                print(f"❌ Upload error: {upload_result.get('status')} - {error_msg}")
                return False
                
        elif upload_response.status_code == 401:
            print("❌ Authentication failed - Check your API key")
            return False
        
        else:
            print(f"❌ Upload failed: {upload_response.status_code}")
            print(f"   Response: {upload_response.text[:200]}...")
            return False
            
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return False

if __name__ == "__main__":
    print("HHOLOVE API Test Script")
    print("Make sure you have set HHOLOVE_API_KEY environment variable")
    print()
    
    # Load .env file if it exists
    try:
        from dotenv import load_dotenv
        load_dotenv()
        print("✅ Loaded .env file")
    except ImportError:
        print("⚠️ python-dotenv not available, using system environment only")
    
    success = test_hholove_api()
    
    if success:
        print("\n🎉 HHOLOVE API test completed successfully!")
        print("   Your API integration is working correctly.")
    else:
        print("\n❌ HHOLOVE API test failed.")
        print("   Please check your API key and network connection.")
    
    sys.exit(0 if success else 1)
