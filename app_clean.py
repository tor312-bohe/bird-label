from flask import Flask, request, send_file, jsonify
import sqlite3
import os
from datetime import datetime
import json
from werkzeug.utils import secure_filename
from PIL import Image, ImageDraw, ImageFont
import tempfile
import logging
import traceback
import requests
from dotenv import load_dotenv
import hashlib

# Load environment variables from .env file
if os.path.exists('.env'):
    load_dotenv('.env')
    print("✅ Environment variables loaded from .env file")
else:
    print("⚠️ No .env file found")

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
RESOURCES_FOLDER = 'Resources'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(RESOURCES_FOLDER, exist_ok=True)

# API Configuration
HHOLOVE_API_KEY = os.getenv('HHOLOVE_API_KEY')
if not HHOLOVE_API_KEY:
    print("⚠️ Warning: HHOLOVE_API_KEY not found in environment variables")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def init_db():
    conn = sqlite3.connect('data/species.sqlite')
    cursor = conn.cursor()
    
    # Create tables if they don't exist
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS species (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        chinese_name TEXT NOT NULL,
        latin_name TEXT NOT NULL,
        common_name TEXT
    )
    ''')
    
    # Insert default species if table is empty
    cursor.execute('SELECT COUNT(*) FROM species')
    if cursor.fetchone()[0] == 0:
        default_species = [
            ("长耳鸮", "Asio otus", "Long-eared Owl"),
            ("红隼", "Falco tinnunculus", "Common Kestrel"),
            ("喜鹊", "Pica pica", "Eurasian Magpie"),
            ("麻雀", "Passer montanus", "Eurasian Tree Sparrow"),
            ("乌鸦", "Corvus corone", "Hooded Crow"),
            ("白头鹎", "Pycnonotus sinensis", "Light-vented Bulbul"),
            ("画眉", "Garrulax canorus", "Hwamei"),
            ("黄鹂", "Oriolus chinensis", "Black-naped Oriole"),
            ("翠鸟", "Alcedo atthis", "Common Kingfisher"),
            ("燕子", "Hirundo rustica", "Barn Swallow")
        ]
        
        cursor.executemany('INSERT INTO species (chinese_name, latin_name, common_name) VALUES (?, ?, ?)', default_species)
    
    conn.commit()
    conn.close()
    print("✅ Database initialized")

# Initialize database
init_db()

def get_location_from_ip():
    """Get user location from IP address"""
    try:
        # Use ipapi.co for IP geolocation (free and reliable)
        response = requests.get('http://ipapi.co/json/', timeout=5)
        if response.status_code == 200:
            data = response.json()
            city = data.get('city', '')
            region = data.get('region', '')
            country = data.get('country_name', '')
            
            # Format location string
            location_parts = [part for part in [city, region, country] if part]
            return ', '.join(location_parts) if location_parts else None
    except Exception as e:
        print(f"Location detection error: {e}")
        return None

def recognize_bird_hholove(image_path, location=None):
    """
    Use HHOLOVE AI API for bird recognition with encyclopedia data
    """
    if not HHOLOVE_API_KEY:
        return {"error": "HHOLOVE API key not configured", "success": False}
    
    try:
        url = "https://hholove-ai.p.rapidapi.com/image_identify_bird_v2"
        
        # Prepare the image file
        with open(image_path, 'rb') as image_file:
            files = {'image': image_file}
            
            headers = {
                'X-RapidAPI-Key': HHOLOVE_API_KEY,
                'X-RapidAPI-Host': 'hholove-ai.p.rapidapi.com'
            }
            
            # Prepare data with location if provided
            data = {}
            if location and location.strip():
                data['location'] = location.strip()
                print(f"🌍 Sending location to HHOLOVE: {location}")
            
            print("🔍 Calling HHOLOVE AI API for bird recognition...")
            response = requests.post(url, headers=headers, files=files, data=data, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                
                if result.get('success') and result.get('data'):
                    bird_data = result['data']
                    
                    # Comprehensive result with encyclopedia data
                    processed_result = {
                        "success": True,
                        "chinese_name": bird_data.get('chinese_name'),
                        "latin_name": bird_data.get('latin_name'),
                        "common_name": bird_data.get('common_name'),
                        "scientific_name": bird_data.get('latin_name'),  # alias for compatibility
                        "confidence": bird_data.get('accuracy', 0),
                        "location_used": bool(location and location.strip()),
                        "source": "HHOLOVE AI",
                        "encyclopedia_data": None
                    }
                    
                    # Extract encyclopedia data if available
                    introduce_data = bird_data.get('introduce', {})
                    if introduce_data and isinstance(introduce_data, dict):
                        processed_result["encyclopedia_data"] = {
                            "overview": introduce_data.get('综述', ''),
                            "physical_features": introduce_data.get('外形特征', ''),
                            "identification": introduce_data.get('区别辨识', ''),
                            "behavior": introduce_data.get('生活习性', ''),
                            "distribution": introduce_data.get('地理分布', ''),
                            "breeding": introduce_data.get('生长繁殖', ''),
                            "vocalizations": introduce_data.get('鸣叫特征', ''),
                            "conservation_status": introduce_data.get('保护现状', ''),
                            "classification": introduce_data.get('classification', {}),
                            "wikipedia_zh": introduce_data.get('wikipedia_zh'),
                            "wikipedia_en": introduce_data.get('wikipedia_en')
                        }
                        
                        print(f"📚 Encyclopedia data found for {bird_data.get('chinese_name', 'Unknown bird')}")
                        print(f"   Available sections: {[k for k, v in processed_result['encyclopedia_data'].items() if v and str(v).strip()]}")
                    
                    return processed_result
                else:
                    return {
                        "error": result.get('message', 'Bird not recognized'),
                        "success": False
                    }
            else:
                print(f"❌ HHOLOVE API error: {response.status_code} - {response.text}")
                return {
                    "error": f"API request failed (Status: {response.status_code})",
                    "success": False
                }
                
    except requests.exceptions.Timeout:
        return {"error": "Request timeout - please try again", "success": False}
    except requests.exceptions.RequestException as e:
        return {"error": f"Network error: {str(e)}", "success": False}
    except Exception as e:
        print(f"❌ HHOLOVE recognition error: {e}")
        traceback.print_exc()
        return {"error": "Recognition service temporarily unavailable", "success": False}

def search_species(query):
    """Search for species in the database"""
    conn = sqlite3.connect('data/species.sqlite')
    cursor = conn.cursor()
    
    # Search in Chinese name, pinyin, and Latin name
    cursor.execute('''
    SELECT chinese_name, latin_name, common_name 
    FROM species 
    WHERE chinese_name LIKE ? OR latin_name LIKE ? OR common_name LIKE ?
    ORDER BY 
        CASE 
            WHEN chinese_name LIKE ? THEN 1
            WHEN latin_name LIKE ? THEN 2
            ELSE 3
        END,
        chinese_name
    LIMIT 10
    ''', (f'%{query}%', f'%{query}%', f'%{query}%', f'{query}%', f'{query}%'))
    
    results = cursor.fetchall()
    conn.close()
    
    return [{"chinese_name": row[0], "latin_name": row[1], "common_name": row[2]} for row in results]

@app.route('/')
def index():
    return '''<!DOCTYPE html>
<html lang="zh">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Bird Labeler - 鸟类识别标注工具</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background-image: url('/static/background.jpg');
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            position: relative;
        }
        
        body::before {
            content: '';
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(0, 0, 0, 0.3);
            z-index: 1;
        }
        
        .container {
            position: relative;
            z-index: 2;
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            border-radius: 16px;
            padding: 2rem;
            width: 100%;
            max-width: 500px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
            border: 1px solid rgba(255, 255, 255, 0.2);
            color: white;
        }
        
        .header {
            text-align: center;
            margin-bottom: 2rem;
        }
        
        .logo {
            margin-bottom: 1rem;
        }
        
        .title {
            font-size: 2rem;
            margin-bottom: 0.5rem;
            font-weight: 300;
        }
        
        .subtitle {
            color: rgba(255, 255, 255, 0.8);
            font-size: 0.9rem;
        }
        
        .upload-area {
            border: 2px dashed rgba(255, 255, 255, 0.3);
            border-radius: 12px;
            padding: 3rem 2rem;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s ease;
            margin-bottom: 1.5rem;
            position: relative;
        }
        
        .upload-area:hover {
            border-color: rgba(255, 255, 255, 0.6);
            background: rgba(255, 255, 255, 0.05);
        }
        
        .upload-icon {
            margin-bottom: 1rem;
        }
        
        .upload-text {
            font-size: 1.1rem;
            margin-bottom: 0.5rem;
        }
        
        .upload-subtext {
            color: rgba(255, 255, 255, 0.7);
            font-size: 0.9rem;
        }
        
        .file-input {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            opacity: 0;
            cursor: pointer;
        }
        
        .button-group {
            display: flex;
            gap: 1rem;
            margin-bottom: 1.5rem;
        }
        
        .recognize-btn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border: none;
            color: white;
            padding: 0.75rem 1.5rem;
            border-radius: 8px;
            cursor: pointer;
            font-size: 1rem;
            font-weight: 500;
            transition: all 0.3s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            min-height: 48px;
        }
        
        .recognize-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
        }
        
        .recognize-btn.primary {
            background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
        }
        
        .recognize-btn.secondary {
            background: linear-gradient(135deg, #ff9a56 0%, #ff6b6b 100%);
        }
        
        .preview-image {
            width: 100%;
            max-height: 400px;
            object-fit: contain;
            border-radius: 8px;
            margin-bottom: 1rem;
        }
        
        .location-input {
            background: rgba(255, 255, 255, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.3);
            color: white;
            padding: 0.75rem;
            border-radius: 8px;
            width: 100%;
            font-size: 0.9rem;
        }
        
        .location-input::placeholder {
            color: rgba(255, 255, 255, 0.6);
        }
        
        .search-container {
            margin-bottom: 1.5rem;
        }
        
        .form-label {
            display: block;
            margin-bottom: 0.5rem;
            font-weight: 500;
        }
        
        .form-input {
            width: 100%;
            padding: 0.75rem;
            border: 1px solid rgba(255, 255, 255, 0.3);
            border-radius: 8px;
            background: rgba(255, 255, 255, 0.1);
            color: white;
            font-size: 1rem;
        }
        
        .form-input::placeholder {
            color: rgba(255, 255, 255, 0.6);
        }
        
        .search-results {
            margin-top: 0.5rem;
            max-height: 200px;
            overflow-y: auto;
        }
        
        .search-result {
            background: rgba(255, 255, 255, 0.1);
            padding: 0.75rem;
            margin-bottom: 0.5rem;
            border-radius: 6px;
            cursor: pointer;
            transition: background 0.2s ease;
        }
        
        .search-result:hover {
            background: rgba(255, 255, 255, 0.2);
        }
        
        .search-result-name {
            font-weight: 500;
            margin-bottom: 0.25rem;
        }
        
        .search-result-latin {
            font-style: italic;
            color: rgba(255, 255, 255, 0.8);
            font-size: 0.9rem;
        }
        
        .supported-formats {
            text-align: center;
            color: rgba(255, 255, 255, 0.6);
            font-size: 0.8rem;
            margin-top: 1rem;
        }
        
        @media (max-width: 600px) {
            .container {
                padding: 1.5rem;
                margin: 10px;
            }
            
            .title {
                font-size: 1.8rem;
            }
            
            .upload-area {
                padding: 2rem 1rem;
            }
            
            .button-group {
                flex-direction: column;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="logo"><img src="/resources/birdlogo.svg" alt="Bird Logo" style="width: 32px; height: 32px; filter: brightness(0) invert(1);"></div>
            <h1 class="title">Bird Labeler</h1>
            <p class="subtitle">Discover the beauty of nature by identifying bird species from your photos</p>
        </div>
        
        <!-- Upload Mode -->
        <div id="uploadMode">
            <form action="/upload" method="post" enctype="multipart/form-data" id="uploadForm">
                <div class="upload-area" id="uploadArea">
                    <div class="upload-icon"><img src="/resources/camera.svg" alt="Camera" style="width: 48px; height: 48px; filter: brightness(0) invert(1); opacity: 0.7;"></div>
                    <div class="upload-text">Click to upload an image</div>
                    <div class="upload-subtext">or drag and drop</div>
                    <input type="file" id="photo" name="photo" accept="image/*" required class="file-input">
                </div>
                
                <div class="button-group" id="mainButtons">
                    <button type="button" id="recognizeBirdBtn" class="recognize-btn primary" style="flex: 1;">
                        <img src="/resources/Scan.svg" alt="Scan" style="width: 20px; height: 20px; margin-right: 8px; vertical-align: middle; filter: brightness(0) invert(1);">
                        Recognize Bird
                    </button>
                    <button type="button" id="searchModeBtn" class="recognize-btn secondary" style="flex: 1;">
                        <img src="/resources/search.svg" alt="Search" style="width: 20px; height: 20px; margin-right: 8px; vertical-align: middle; filter: brightness(0) invert(1);">
                        Search by name
                    </button>
                </div>
                
                <div id="imagePreview" style="display: none;">
                    <img id="previewImg" class="preview-image" alt="Preview">
                    
                    <!-- Location Input -->
                    <div class="location-section" style="margin: 0.8rem 0;" id="locationSection">
                        <input type="text" id="location" name="location" placeholder="Detecting location..." readonly
                               style="width: 100%; padding: 0.75rem; border: 1px solid rgba(255, 255, 255, 0.3); border-radius: 8px; background: rgba(255, 255, 255, 0.1); color: white; font-size: 0.9rem;"
                               class="location-input">
                    </div>
                    
                    <!-- Recognize Button -->
                    <button type="button" id="recognizeBtn" class="recognize-btn primary" style="width: 100%; margin: 0.8rem 0;">
                        <img src="/resources/Scan.svg" alt="Scan" style="width: 20px; height: 20px; margin-right: 8px; vertical-align: middle; filter: brightness(0) invert(1);">
                        Recognize Bird
                    </button>
                    
                    <!-- Recognition Result -->
                    <div id="recognitionResult" style="display: none; background: rgba(255, 255, 255, 0.1); border-radius: 8px; padding: 1rem; margin: 1rem 0; max-height: 600px; overflow-y: auto;">
                        <div id="recognitionContent"></div>
                    </div>
                    
                    <button type="submit" id="submitBtn" class="recognize-btn" style="display: none;"><img src="/resources/tag.svg" alt="Tag" style="width: 20px; height: 20px; margin-right: 8px; vertical-align: middle; filter: brightness(0) invert(1);">Label Bird</button>
                    
                    <div style="text-align: center; margin-top: 1rem; display: none;" id="uploadAnotherContainer">
                        <button type="button" id="uploadAnotherBtn" style="background: none; border: none; color: rgba(255, 255, 255, 0.6); font-size: 0.9rem; cursor: pointer; text-decoration: underline;">
                            Upload another image
                        </button>
                    </div>
                    
                    <!-- Hidden inputs for form submission -->
                    <input type="hidden" id="cn" name="chinese_name">
                    <input type="hidden" id="la" name="latin_name">
                    <input type="hidden" name="position" value="bottom_right">
                </div>
            </form>
        </div>
        
        <!-- Search Mode -->
        <div id="searchMode" style="display: none;">
            <div class="search-container">
                <label class="form-label" for="search">Search Bird Species</label>
                <input type="text" id="search" class="form-input" placeholder="Enter Chinese name, pinyin, or Latin name..." autocomplete="off">
                <div class="search-results" id="searchResults"></div>
            </div>
            <button type="button" id="backToUploadBtn" class="recognize-btn secondary" style="margin-top: 2rem;">
                Back to Upload
            </button>
        </div>
        
        <div class="supported-formats" id="supportedFormats">
            Supported formats: JPG, PNG, WebP
        </div>
    </div>

    <script>
    let searchTimeout;
    const searchInput = document.getElementById('search');
    const searchResults = document.getElementById('searchResults');
    const cnInput = document.getElementById('cn');
    const laInput = document.getElementById('la');
    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('photo');
    const imagePreview = document.getElementById('imagePreview');
    const previewImg = document.getElementById('previewImg');
    const recognizeBtn = document.getElementById('recognizeBtn');
    const recognitionResult = document.getElementById('recognitionResult');
    const recognitionContent = document.getElementById('recognitionContent');

    // Auto-detect location
    function detectLocation() {
        const locationInput = document.getElementById('location');
        locationInput.placeholder = 'Detecting location...';
        
        fetch('/detect_location')
            .then(response => response.json())
            .then(data => {
                if (data.location) {
                    locationInput.value = data.location;
                    locationInput.placeholder = 'Current location';
                } else {
                    locationInput.value = '';
                    locationInput.placeholder = 'Location not detected (optional)';
                }
            })
            .catch(error => {
                console.log('Location detection failed:', error);
                locationInput.value = '';
                locationInput.placeholder = 'Enter location (optional)';
                // Make location input editable if detection fails
                locationInput.readonly = false;
            });
    }

    // File upload handling
    uploadArea.addEventListener('click', () => fileInput.click());
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.style.background = 'rgba(255, 255, 255, 0.1)';
    });
    uploadArea.addEventListener('dragleave', () => {
        uploadArea.style.background = '';
    });
    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.style.background = '';
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            fileInput.files = files;
            handleFileSelect();
        }
    });

    fileInput.addEventListener('change', handleFileSelect);

    function handleFileSelect() {
        const file = fileInput.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = (e) => {
                previewImg.src = e.target.result;
                imagePreview.style.display = 'block';
                document.getElementById('mainButtons').style.display = 'none';
                document.getElementById('supportedFormats').style.display = 'none';
                
                // Start location detection
                detectLocation();
            };
            reader.readAsDataURL(file);
        }
    }

    // Bird recognition
    recognizeBtn.addEventListener('click', function() {
        this.disabled = true;
        this.innerHTML = '<span style="display: inline-block; animation: spin 1s linear infinite;">🔍</span> Recognizing...';
        
        const formData = new FormData();
        formData.append('photo', fileInput.files[0]);
        
        const location = document.getElementById('location').value;
        if (location) {
            formData.append('location', location);
        }
        
        fetch('/recognize', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            console.log('Recognition result:', data);
            displayRecognitionResult(data);
            this.disabled = false;
            this.innerHTML = '<img src="/resources/Scan.svg" alt="Scan" style="width: 20px; height: 20px; margin-right: 8px; vertical-align: middle; filter: brightness(0) invert(1);">Recognize Bird';
        })
        .catch(error => {
            console.error('Recognition error:', error);
            recognitionContent.innerHTML = `
                <div style="color: #FCA5A5; text-align: center;">
                    ❌ Recognition failed<br>
                    <small style="color: rgba(255, 255, 255, 0.7);">Please try again or check your connection</small>
                </div>
            `;
            recognitionResult.style.display = 'block';
            this.disabled = false;
            this.innerHTML = '<img src="/resources/Scan.svg" alt="Scan" style="width: 20px; height: 20px; margin-right: 8px; vertical-align: middle; filter: brightness(0) invert(1);">Recognize Bird';
        });
    });

    function displayRecognitionResult(data) {
        if (data.success) {
            // Populate hidden fields for form submission
            cnInput.value = data.chinese_name || '';
            laInput.value = data.scientific_name || data.latin_name || '';
            
            // Create name display
            let nameDisplay = '';
            if (data.chinese_name) {
                nameDisplay += `<div style="margin-bottom: 0.5rem;"><strong>${data.chinese_name}</strong></div>`;
            }
            if (data.common_name) {
                nameDisplay += `<div style="color: rgba(255, 255, 255, 0.9); margin-bottom: 0.5rem;">${data.common_name}</div>`;
            }
            if (data.scientific_name) {
                nameDisplay += `<div style="font-style: italic; color: rgba(255, 255, 255, 0.8); margin-bottom: 0.5rem;">${data.scientific_name}</div>`;
            }
            
            // Show recognition result with encyclopedia data
            const locationUsed = document.getElementById('location').value;
            let resultHTML = `
                <div style="color: white; text-align: center;">
                    <div style="font-size: 1.1rem; margin-bottom: 0.5rem;">
                        ✓ <strong>Bird Identified!</strong>
                    </div>
                    ${nameDisplay}
                    <div style="font-size: 0.9rem; color: rgba(255, 255, 255, 0.7); margin-bottom: 0.5rem;">
                        Confidence: ${Math.round(data.confidence * 100)}%
                    </div>
                    ${locationUsed ? `
                    <div style="font-size: 0.8rem; color: rgba(255, 255, 255, 0.6);">
                        📍 Location: ${locationUsed}
                        ${data.location_used ? ' ✨ (Enhanced with location data)' : ''}
                    </div>
                    ` : ''}
                </div>
            `;
            
            // Add comprehensive encyclopedia information if available
            if (data.encyclopedia_data) {
                resultHTML += `
                    <div style="margin-top: 1.5rem; padding-top: 1rem; border-top: 1px solid rgba(255,255,255,0.3); text-align: left;">
                        <h4 style="color: white; text-align: center; margin-bottom: 1rem; font-size: 1rem;">🐦 ${data.chinese_name} 详细介绍</h4>
                `;
                
                // Add all encyclopedia sections
                const sections = [
                    { key: 'overview', title: '📋 综述' },
                    { key: 'physical_features', title: '🔍 外形特征' },
                    { key: 'identification', title: '🔎 区别辨识' },
                    { key: 'behavior', title: '🦅 生活习性' },
                    { key: 'distribution', title: '🌍 地理分布' },
                    { key: 'breeding', title: '🥚 生长繁殖' },
                    { key: 'vocalizations', title: '🎵 鸣叫特征' },
                    { key: 'conservation_status', title: '🛡️ 保护现状' }
                ];
                
                sections.forEach(section => {
                    const sectionData = data.encyclopedia_data[section.key];
                    if (sectionData && sectionData.trim()) {
                        resultHTML += `
                            <div style="margin-bottom: 1rem; background: rgba(255,255,255,0.05); padding: 0.8rem; border-radius: 6px;">
                                <div style="color: rgba(255,255,255,0.95); font-weight: bold; margin-bottom: 0.5rem; font-size: 0.9rem;">${section.title}</div>
                                <div style="color: rgba(255,255,255,0.85); font-size: 0.85rem; line-height: 1.4;">${sectionData}</div>
                            </div>
                        `;
                    }
                });
                
                // Add links if available
                if (data.encyclopedia_data.wikipedia_zh || data.encyclopedia_data.wikipedia_en) {
                    resultHTML += `
                        <div style="text-align: center; margin-top: 1rem; padding-top: 0.8rem; border-top: 1px solid rgba(255,255,255,0.2);">
                            <div style="color: rgba(255,255,255,0.95); font-weight: bold; margin-bottom: 0.5rem; font-size: 0.9rem;">🔗 更多信息</div>
                            ${data.encyclopedia_data.wikipedia_zh ? `
                                <a href="${data.encyclopedia_data.wikipedia_zh}" target="_blank" 
                                   style="color: rgba(135,206,235,0.9); text-decoration: none; font-size: 0.8rem; margin-right: 1rem;">
                                    📚 中文维基百科
                                </a>
                            ` : ''}
                            ${data.encyclopedia_data.wikipedia_en ? `
                                <a href="${data.encyclopedia_data.wikipedia_en}" target="_blank" 
                                   style="color: rgba(135,206,235,0.9); text-decoration: none; font-size: 0.8rem;">
                                    📖 English Wikipedia
                                </a>
                            ` : ''}
                        </div>
                    `;
                }
                
                resultHTML += `</div>`;
            }
            
            recognitionContent.innerHTML = resultHTML;
            recognitionResult.style.display = 'block';
            
            // Hide the location section after successful recognition
            const locationSection = document.getElementById('locationSection');
            if (locationSection) {
                locationSection.style.display = 'none';
            }
            
            // Hide the recognize button and show submit button
            recognizeBtn.style.display = 'none';
            document.getElementById('submitBtn').style.display = 'block';
            document.getElementById('uploadAnotherContainer').style.display = 'block';
        } else {
            recognitionContent.innerHTML = `
                <div style="color: white; text-align: center;">
                    <div style="color: #FCA5A5; margin-bottom: 0.5rem;">
                        ! Could not identify bird
                    </div>
                    <div style="font-size: 0.9rem; color: rgba(255, 255, 255, 0.7);">
                        ${data.error || 'Please try a clearer image or enter the information manually.'}
                    </div>
                </div>
            `;
            recognitionResult.style.display = 'block';
        }
    }

    // Search functionality
    searchInput.addEventListener('input', function() {
        const query = this.value.trim();
        
        clearTimeout(searchTimeout);
        searchTimeout = setTimeout(() => {
            if (query.length >= 1) {
                fetch(`/search?q=${encodeURIComponent(query)}`)
                    .then(response => response.json())
                    .then(data => {
                        displaySearchResults(data);
                    });
            } else {
                searchResults.innerHTML = '';
            }
        }, 300);
    });

    function displaySearchResults(results) {
        if (results.length === 0) {
            searchResults.innerHTML = '<div style="color: rgba(255, 255, 255, 0.6); padding: 1rem; text-align: center;">No results found</div>';
            return;
        }
        
        searchResults.innerHTML = results.map(result => `
            <div class="search-result" onclick="selectBird('${result.chinese_name}', '${result.latin_name}')">
                <div class="search-result-name">${result.chinese_name}</div>
                <div class="search-result-latin">${result.latin_name}</div>
                ${result.common_name ? `<div style="color: rgba(255, 255, 255, 0.7); font-size: 0.8rem;">${result.common_name}</div>` : ''}
            </div>
        `).join('');
    }

    function selectBird(chineseName, latinName) {
        cnInput.value = chineseName;
        laInput.value = latinName;
        
        // Show selected bird
        recognitionContent.innerHTML = `
            <div style="color: white; text-align: center;">
                <div style="font-size: 1.1rem; margin-bottom: 0.5rem;">
                    ✓ <strong>Bird Selected</strong>
                </div>
                <div style="margin-bottom: 0.5rem;"><strong>${chineseName}</strong></div>
                <div style="font-style: italic; color: rgba(255, 255, 255, 0.8);">${latinName}</div>
            </div>
        `;
        recognitionResult.style.display = 'block';
        document.getElementById('submitBtn').style.display = 'block';
        
        // Clear search
        searchInput.value = '';
        searchResults.innerHTML = '';
    }

    // Mode switching
    document.getElementById('searchModeBtn').addEventListener('click', function() {
        document.getElementById('uploadMode').style.display = 'none';
        document.getElementById('searchMode').style.display = 'block';
        searchInput.focus();
    });

    document.getElementById('backToUploadBtn').addEventListener('click', function() {
        document.getElementById('searchMode').style.display = 'none';
        document.getElementById('uploadMode').style.display = 'block';
        searchResults.innerHTML = '';
        searchInput.value = '';
    });

    document.getElementById('uploadAnotherBtn').addEventListener('click', function() {
        // Reset form
        fileInput.value = '';
        imagePreview.style.display = 'none';
        recognitionResult.style.display = 'none';
        document.getElementById('mainButtons').style.display = 'flex';
        document.getElementById('supportedFormats').style.display = 'block';
        document.getElementById('submitBtn').style.display = 'none';
        document.getElementById('uploadAnotherContainer').style.display = 'none';
        recognizeBtn.style.display = 'block';
        document.getElementById('locationSection').style.display = 'block';
        
        // Clear hidden inputs
        cnInput.value = '';
        laInput.value = '';
    });

    // Add CSS animation for spinner
    const style = document.createElement('style');
    style.textContent = `
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    `;
    document.head.appendChild(style);
    </script>
</body>
</html>'''

@app.route('/detect_location')
def detect_location():
    """Endpoint to detect user location"""
    location = get_location_from_ip()
    return jsonify({"location": location})

@app.route('/recognize', methods=['POST'])
def recognize():
    """Recognize bird from uploaded image"""
    if 'photo' not in request.files:
        return jsonify({'error': 'No image provided', 'success': False})
    
    file = request.files['photo']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file format', 'success': False})
    
    # Get location if provided
    location = request.form.get('location', '').strip()
    
    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        # Use HHOLOVE AI for recognition
        result = recognize_bird_hholove(filepath, location)
        
        return jsonify(result)
        
    except Exception as e:
        print(f"Recognition error: {e}")
        traceback.print_exc()
        return jsonify({'error': 'Recognition failed', 'success': False})

@app.route('/search')
def search():
    """Search for bird species"""
    query = request.args.get('q', '').strip()
    if not query:
        return jsonify([])
    
    results = search_species(query)
    return jsonify(results)

@app.route('/resources/<path:filename>')
def resources(filename):
    """Serve static resources"""
    return send_file(os.path.join(RESOURCES_FOLDER, filename))

@app.route('/static/<path:filename>')
def static_files(filename):
    """Serve static files like background image"""
    return send_file(filename)

@app.route('/upload', methods=['POST'])
def upload():
    """Handle image upload and create labeled version"""
    if 'photo' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['photo']
    chinese_name = request.form.get('chinese_name', '')
    latin_name = request.form.get('latin_name', '')
    position = request.form.get('position', 'bottom_right')
    
    if not chinese_name or not latin_name:
        return jsonify({'error': 'Bird information missing'})
    
    try:
        # Save original file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        # Create labeled image
        output_filename = f"captioned_{timestamp}.jpg"
        output_path = os.path.join(OUTPUT_FOLDER, output_filename)
        
        # Add text to image
        add_text_to_image(filepath, output_path, chinese_name, latin_name, position)
        
        return send_file(output_path, as_attachment=True, download_name=output_filename)
        
    except Exception as e:
        print(f"Upload error: {e}")
        traceback.print_exc()
        return jsonify({'error': f'Processing failed: {str(e)}'})

def add_text_to_image(input_path, output_path, chinese_name, latin_name, position='bottom_right'):
    """Add bird species text labels to image"""
    with Image.open(input_path) as img:
        # Convert to RGB if necessary
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        draw = ImageDraw.Draw(img)
        
        # Try to use a better font
        font_size = max(20, img.width // 40)  # Dynamic font size
        
        try:
            # Try to load a system font (works on most systems)
            if os.name == 'nt':  # Windows
                font_path = "C:/Windows/Fonts/msyh.ttc"  # Microsoft YaHei for Chinese
            elif os.path.exists("/System/Library/Fonts/PingFang.ttc"):  # macOS
                font_path = "/System/Library/Fonts/PingFang.ttc"
            elif os.path.exists("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):  # Linux
                font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
            else:
                raise OSError("No suitable font found")
                
            font = ImageFont.truetype(font_path, font_size)
        except (OSError, IOError):
            # Fallback to default font
            font = ImageFont.load_default()
        
        # Prepare text
        text_chinese = chinese_name
        text_latin = latin_name
        
        # Calculate text dimensions
        chinese_bbox = draw.textbbox((0, 0), text_chinese, font=font)
        latin_bbox = draw.textbbox((0, 0), text_latin, font=font)
        
        chinese_width = chinese_bbox[2] - chinese_bbox[0]
        chinese_height = chinese_bbox[3] - chinese_bbox[1]
        latin_width = latin_bbox[2] - latin_bbox[0]
        latin_height = latin_bbox[3] - latin_bbox[1]
        
        max_width = max(chinese_width, latin_width)
        total_height = chinese_height + latin_height + 10  # 10px spacing
        
        # Calculate position
        margin = 20
        if position == 'bottom_right':
            x = img.width - max_width - margin
            y = img.height - total_height - margin
        elif position == 'bottom_left':
            x = margin
            y = img.height - total_height - margin
        elif position == 'top_right':
            x = img.width - max_width - margin
            y = margin
        else:  # top_left
            x = margin
            y = margin
        
        # Draw background rectangle for better readability
        bg_padding = 15
        bg_x1 = x - bg_padding
        bg_y1 = y - bg_padding
        bg_x2 = x + max_width + bg_padding
        bg_y2 = y + total_height + bg_padding
        
        draw.rectangle([bg_x1, bg_y1, bg_x2, bg_y2], fill=(0, 0, 0, 128))
        
        # Draw text
        draw.text((x, y), text_chinese, font=font, fill='white')
        draw.text((x, y + chinese_height + 10), text_latin, font=font, fill='white')
        
        # Save the image
        img.save(output_path, 'JPEG', quality=95)

if __name__ == '__main__':
    print("✅ Environment variables loaded from .env file")
    print("✅ Database initialized")
    print("🐦 启动鸟类标注分享工具...")
    print("📱 访问 http://127.0.0.1:8001 开始使用")
    print("🔍 支持搜索功能，数据库已包含常见鸟类")
    print()
    print("按 Ctrl+C 停止服务器")
    app.run(host='127.0.0.1', port=8001, debug=True)
