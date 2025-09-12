#!/usr/bin/env python3
"""
HHOLOVE AI API Reverse Engineering Script

This script helps analyze the HHOLOVE website to understand their real API format.
Run this while monitoring network traffic on their demo page.
"""

import requests
import json
import base64
from pathlib import Path

def analyze_hholove_demo():
    """
    Analyze the HHOLOVE demo page to understand API patterns
    """
    print("🔍 HHOLOVE AI API Reverse Engineering")
    print("=" * 50)
    
    # Step 1: Load their demo page and analyze
    demo_url = "https://ai.open.hhodata.com/#experience"
    
    print(f"📄 Loading demo page: {demo_url}")
    
    try:
        response = requests.get(demo_url, timeout=10)
        content = response.text
        
        print(f"✅ Page loaded, content length: {len(content)} chars")
        
        # Look for JavaScript code that might contain API calls
        print("\n🔍 Searching for API patterns in JavaScript...")
        
        # Common API patterns to search for
        api_patterns = [
            'fetch(',
            'XMLHttpRequest',
            'axios.',
            'api.',
            'upload',
            'recognize',
            'bird',
            'animal',
            'POST',
            'multipart',
            'base64',
            'FormData'
        ]
        
        found_patterns = []
        for pattern in api_patterns:
            if pattern in content:
                found_patterns.append(pattern)
        
        print(f"📋 Found API-related patterns: {found_patterns}")
        
        # Look for potential API endpoints
        print("\n🌐 Searching for potential API endpoints...")
        
        import re
        # Look for URLs that might be API endpoints
        url_patterns = re.findall(r'https?://[^\s\'"<>]+', content)
        api_urls = [url for url in url_patterns if any(keyword in url.lower() for keyword in ['api', 'service', 'recognize', 'upload'])]
        
        if api_urls:
            print("🎯 Potential API URLs found:")
            for url in set(api_urls):
                print(f"   - {url}")
        else:
            print("❌ No obvious API URLs found in page source")
            
        # Look for form elements
        print("\n📋 Searching for upload forms...")
        form_patterns = re.findall(r'<form[^>]*>.*?</form>', content, re.DOTALL | re.IGNORECASE)
        if form_patterns:
            print(f"📝 Found {len(form_patterns)} form(s)")
        
        # Look for JavaScript functions that might handle uploads
        print("\n🔧 Searching for upload-related JavaScript...")
        js_upload_patterns = [
            r'function\s+\w*upload\w*\s*\(',
            r'function\s+\w*recognize\w*\s*\(',
            r'\w*\.upload\s*=',
            r'\w*\.recognize\s*='
        ]
        
        for pattern in js_upload_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                print(f"🎯 Found JS pattern '{pattern}': {matches}")
        
    except Exception as e:
        print(f"❌ Error analyzing demo page: {e}")
    
    print("\n" + "=" * 50)
    print("🛠️  MANUAL INVESTIGATION STEPS:")
    print("=" * 50)
    print("1. Open https://ai.open.hhodata.com/#experience in Chrome/Firefox")
    print("2. Open Developer Tools (F12)")
    print("3. Go to Network tab")
    print("4. Upload an image using their demo")
    print("5. Look for XHR/Fetch requests in Network tab")
    print("6. Copy the request details:")
    print("   - URL")
    print("   - Method (POST/GET)")
    print("   - Headers")
    print("   - Request payload format")
    print("   - Response format")
    print("\n📋 EXPECTED API INFORMATION TO FIND:")
    print("- Real API endpoint URL")
    print("- Request Content-Type (multipart/form-data? application/json?)")
    print("- Image format (base64? binary? form field?)")
    print("- Authentication method")
    print("- Response JSON structure")
    
def save_test_image():
    """
    Save your test image for API testing
    """
    print("\n💾 Test Image Preparation")
    print("=" * 30)
    print("Save your bird image (the Brown-backed Three-toed Kingfisher) as:")
    print("   📁 test_bird.jpg")
    print("This will be used to test the real API once we discover the format.")

def create_api_template():
    """
    Create a template for the real API integration
    """
    template = '''
def recognize_bird_hholove_real(image_path, location=None):
    """
    Real HHOLOVE AI integration - update with discovered API format
    """
    try:
        # TODO: Replace these with REAL values from reverse engineering
        api_endpoint = "REPLACE_WITH_REAL_URL"  # From network analysis
        
        # TODO: Update authentication method
        headers = {
            "Authorization": "Bearer YOUR_API_KEY",  # Or other auth method
            "User-Agent": "BirdLabel/1.0"
        }
        
        # TODO: Update request format based on findings
        # Option 1: multipart/form-data
        with open(image_path, 'rb') as f:
            files = {'image': f}
            data = {'format': 'json'}  # Or other parameters
            response = requests.post(api_endpoint, files=files, data=data, headers=headers)
        
        # Option 2: JSON with base64
        # with open(image_path, 'rb') as f:
        #     image_data = base64.b64encode(f.read()).decode('utf-8')
        # payload = {
        #     'image': image_data,
        #     'format': 'base64'
        # }
        # response = requests.post(api_endpoint, json=payload, headers=headers)
        
        # TODO: Update response parsing
        if response.status_code == 200:
            data = response.json()
            # Parse response based on actual format
            return {
                'confidence': data.get('confidence', 0),
                'scientific_name': data.get('scientific_name', ''),
                'common_name': data.get('common_name', ''),
                'chinese_name': data.get('chinese_name', ''),
                'method': 'hholove_ai_real'
            }
        
    except Exception as e:
        print(f"❌ HHOLOVE API error: {e}")
        return None
'''
    
    print("\n📝 API Template Created")
    print("=" * 25)
    print("Once you discover the real API format, update the integration using this template.")
    
    # Save template to file
    Path("hholove_api_template.py").write_text(template)
    print("💾 Template saved as: hholove_api_template.py")

if __name__ == "__main__":
    analyze_hholove_demo()
    save_test_image()
    create_api_template()
    
    print("\n🎯 NEXT STEPS:")
    print("1. Run manual investigation on their website")
    print("2. Document the real API format")
    print("3. Update the integration code")
    print("4. Test with your bird image")
    print("\nExpected result: 棕背三趾翠鸟 (Brown-backed Three-toed Kingfisher)")
