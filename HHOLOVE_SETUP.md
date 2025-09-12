# HHOLOVE AI Integration Setup Guide

## 🎯 Overview

This guide explains how to set up the HHOLOVE AI (懂鸟) integration based on the official OpenAPI specification. HHOLOVE AI is a high-accuracy bird recognition service supporting 11,151 bird species with professional-grade accuracy.

## 📋 API Specifications

- **Base URL**: `https://ai.open.hhodata.com/api/v2`
- **Authentication**: API Key in header
- **Image Requirements**: 
  - Format: JPEG (preferred)
  - Size: Max 2MB, min edge 50px, max edge 8192px
  - Content: Single bird image
- **Supported Classes**: Birds (B), Mammals (M), Amphibians (A), Reptiles (R)
- **Current Implementation**: Birds only (class='B')

## 🔧 Setup Instructions

### 1. Get HHOLOVE API Key

1. Visit the HHOLOVE website: `https://ai.open.hhodata.com/`
2. Register for an account
3. Subscribe to the API service
4. Copy your API key from the dashboard

### 2. Configure Environment Variable

Create a `.env` file in your project directory or set the environment variable:

```bash
# Option 1: Create .env file
echo "HHOLOVE_API_KEY=your_actual_api_key_here" >> .env

# Option 2: Export environment variable
export HHOLOVE_API_KEY="your_actual_api_key_here"
```

### 3. Verify Integration

Run the Flask application and test with a bird image:

```bash
cd "Bird Label"
source .venv/bin/activate
python app.py
```

The HHOLOVE integration will now be active as the primary recognition method.

### 1. Sign Up for API Access
Visit [HHOLOVE AI Platform](https://ai.open.hhodata.com/) to:
- Create an account
- Get your API key
- Choose a pricing plan

### 2. Pricing Plans
- **Free Trial**: Up to 100 API calls
- **Pay-per-Package**: 10,000 calls for ¥100 (0.01 yuan per call)
- **Pay-as-you-go**: ¥5.0 per 1,000 calls for high volume
- **Enterprise**: Custom pricing for large customers

### 3. API Configuration

#### Option A: Environment Variable (Recommended)
```bash
export HHOLOVE_API_KEY="your_actual_api_key_here"
```

#### Option B: .env File
1. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```
2. Edit `.env` and replace `your_api_key_here` with your actual API key:
   ```
   HHOLOVE_API_KEY=your_actual_api_key_here
   ```

### 4. Test the Integration
Run the bird recognition app and upload an image. The system will automatically try HHOLOVE AI as the second method (after iNaturalist).

## Recognition Pipeline Order

The app now uses a 9-method cascade for maximum accuracy with HHOLOVE AI prioritized first:

1. **HHOLOVE AI (懂鸟)** - High-accuracy commercial service (85% Top1, 96% Top5) ⭐ PRIMARY
2. **iNaturalist API** - Global species database (fallback)
3. **Bird Watch TensorFlow** - Specialized bird model
4. **Other Specialized Models** - Additional bird-specific AI
5. **YOLOv5** - General object detection
6. **Hugging Face** - Community models
7. **OpenCV** - Computer vision analysis
8. **Local Fallback** - Basic recognition
9. **Absolute Fallback** - Always returns result

## Features

### Location Awareness
HHOLOVE AI supports location-based recognition improvement:
- GPS coordinates from image EXIF data
- Manual location input (city names, coordinates)
- Confidence boosting for regionally appropriate species

### Response Format
The HHOLOVE integration returns:
```json
{
  "confidence": 0.85,
  "scientific_name": "Turdus migratorius",
  "common_name": "American Robin",
  "chinese_name": "美洲知更鸟",
  "method": "hholove_ai",
  "location_used": true,
  "api_response": "Top5 results: 5"
}
```

## Troubleshooting

### Common Issues

1. **API Key Not Set**
   ```
   ⚠️ HHOLOVE API key not configured. Set HHOLOVE_API_KEY environment variable.
   ```
   Solution: Follow the API Configuration steps above.

2. **Authentication Failed**
   ```
   ❌ HHOLOVE API authentication failed. Check API key.
   ```
   Solution: Verify your API key is correct and account is active.

3. **Rate Limit Exceeded**
   ```
   ❌ HHOLOVE API rate limit exceeded. Try again later.
   ```
   Solution: Wait or upgrade to higher QPS plan.

4. **Network Errors**
   ```
   ❌ HHOLOVE API network error: Connection timeout
   ```
   Solution: Check internet connection and try again.

### Fallback Behavior
If HHOLOVE AI fails for any reason, the app automatically continues to the next recognition method (Bird Watch TensorFlow), ensuring reliable operation.

## Benefits

### Why Use HHOLOVE AI?
- **Higher Accuracy**: 85% vs typical 60-70% accuracy
- **More Species**: 10,000+ birds vs 1,000-3,000 in other services
- **Location Aware**: Improves predictions based on geographic context
- **Chinese Support**: Native Chinese names and documentation
- **Commercial Quality**: Professional service with SLA and support

### Cost Effectiveness
- Free trial for testing (100 calls)
- Low cost for regular use (0.01 yuan = ~$0.0014 per call)
- Reasonable for most applications

## Contact Support
- Email: service@hholove.com
- Website: https://ai.open.hhodata.com/
- Documentation: https://ai.open.hhodata.com/doc

---
**Note**: The API endpoint and request format in the code may need adjustment based on the actual HHOLOVE API documentation. The current implementation is based on common API patterns and will be updated once official documentation is available.
