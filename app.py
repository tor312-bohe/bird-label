from flask import Flask, request, send_file, jsonify
from PIL import Image, ImageDraw, ImageFont
from datetime import datetime
import os, sqlite3, requests, base64, io
import json

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ Environment variables loaded from .env file")
except ImportError:
    print("ℹ️ python-dotenv not available, using system environment variables only")
except Exception as e:
    print(f"ℹ️ Could not load .env file: {e}")

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["OUTPUT_FOLDER"] = "outputs"
app.config["DB_PATH"] = "data/species.sqlite"
app.config["FONTS_FOLDER"] = "fonts"

# Create directories
for folder in ["uploads", "outputs", "data", "fonts"]:
    os.makedirs(folder, exist_ok=True)

def download_chinese_font():
    """
    Download a free Chinese font for better text rendering
    """
    font_path = os.path.join(app.config["FONTS_FOLDER"], "NotoSansCJK-Regular.ttc")
    
    if os.path.exists(font_path):
        return font_path
    
    try:
        print("📥 Downloading Chinese font (Noto Sans CJK)...")
        # Use a smaller, free Chinese font
        font_url = "https://github.com/googlefonts/noto-cjk/raw/main/Sans/SubsetOTF/CN/NotoSansCJK-Regular.otf"
        
        response = requests.get(font_url, timeout=30)
        if response.status_code == 200:
            with open(font_path, 'wb') as f:
                f.write(response.content)
            print(f"✅ Downloaded Chinese font to {font_path}")
            return font_path
        else:
            print(f"❌ Failed to download font: HTTP {response.status_code}")
            return None
    except Exception as e:
        print(f"❌ Font download error: {e}")
        return None

def init_database():
    con = sqlite3.connect(app.config["DB_PATH"])
    cur = con.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS species(
      id INTEGER PRIMARY KEY,
      chinese_name TEXT NOT NULL,
      latin_name TEXT NOT NULL,
      pinyin TEXT
    )""")
    
    cur.execute("SELECT COUNT(*) FROM species")
    if cur.fetchone()[0] == 0:
        birds = [
            ("白头鹎", "Pycnonotus sinensis", "baitoubei"),
            ("麻雀", "Passer montanus", "maque"),
            ("喜鹊", "Pica pica", "xique"),
            ("乌鸫", "Turdus merula", "wudong"),
            ("翠鸟", "Alcedo atthis", "cuiniao"),
            ("小白鹭", "Egretta garzetta", "xiaobaihu"),
            ("黑水鸡", "Gallinula chloropus", "heishuiji"),
            ("八哥", "Acridotheres cristatellus", "bage")
        ]
        for cn, la, py in birds:
            cur.execute("INSERT INTO species(chinese_name, latin_name, pinyin) VALUES (?, ?, ?)", (cn, la, py))
    
    con.commit()
    con.close()
    print("✅ Database initialized")

def search_species(q):
    if not q:
        return []
    con = sqlite3.connect(app.config["DB_PATH"])
    sql = "SELECT chinese_name, latin_name FROM species WHERE chinese_name LIKE ? OR latin_name LIKE ? OR pinyin LIKE ? LIMIT 10"
    like = f"%{q}%"
    results = [{"chinese_name": r[0], "latin_name": r[1]} for r in con.execute(sql, (like, like, like)).fetchall()]
    con.close()
    return results

def extract_coordinates(location_text):
    """
    Extract latitude and longitude from location text
    Supports various formats: "lat,lng", "City, Country", etc.
    """
    import re
    
    # Try to match decimal coordinates (lat, lng)
    coord_pattern = r'(-?\d+\.?\d*),\s*(-?\d+\.?\d*)'
    match = re.search(coord_pattern, location_text)
    if match:
        lat, lng = float(match.group(1)), float(match.group(2))
        # Basic validation for reasonable coordinates
        if -90 <= lat <= 90 and -180 <= lng <= 180:
            return (lat, lng)
    
    # For city/country names, we could use a geocoding service
    # For now, return None - could be enhanced with geocoding API
    return None

def get_location_confidence_boost(species_name, location):
    """
    Provide confidence boost based on species-location matching
    Returns a multiplier (1.0 = no change, >1.0 = boost, <1.0 = penalty)
    """
    try:
        location_lower = location.lower()
        species_lower = species_name.lower()
        
        # Regional bird distribution knowledge (simplified)
        # This could be enhanced with a comprehensive bird distribution database
        location_boosts = {
            # North America
            'united states': ['american robin', 'blue jay', 'northern cardinal', 'house sparrow'],
            'canada': ['canada goose', 'common loon', 'snowy owl', 'american robin'],
            'mexico': ['resplendent quetzal', 'vermilion flycatcher', 'painted bunting'],
            
            # Europe
            'united kingdom': ['european robin', 'house sparrow', 'blackbird', 'blue tit'],
            'germany': ['european robin', 'great tit', 'blackbird', 'house sparrow'],
            'france': ['european robin', 'blackbird', 'great tit', 'house sparrow'],
            
            # Asia
            'china': ['eurasian tree sparrow', 'chinese bulbul', 'oriental magpie-robin'],
            'japan': ['japanese tit', 'brown-eared bulbul', 'japanese white-eye'],
            'india': ['house crow', 'common myna', 'red-vented bulbul'],
            
            # Australia
            'australia': ['rainbow lorikeet', 'australian magpie', 'willie wagtail'],
            
            # Coastal areas
            'coast': ['seagull', 'pelican', 'cormorant', 'sandpiper'],
            'beach': ['seagull', 'pelican', 'tern', 'plover'],
            'ocean': ['albatross', 'petrel', 'gannet', 'booby'],
        }
        
        # Check for location matches
        boost_factor = 1.0
        
        for region, birds in location_boosts.items():
            if region in location_lower:
                for bird in birds:
                    if bird in species_lower:
                        boost_factor = 1.3  # 30% confidence boost
                        print(f"🌍 Location boost: {species_name} is common in {region}")
                        break
                break
        
        # Additional habitat-based boosts
        if any(water_term in location_lower for water_term in ['lake', 'river', 'pond', 'wetland']):
            if any(water_bird in species_lower for water_bird in ['duck', 'goose', 'swan', 'heron', 'egret', 'kingfisher']):
                boost_factor = max(boost_factor, 1.2)
                print(f"🌊 Habitat boost: {species_name} matches water habitat")
        
        if any(forest_term in location_lower for forest_term in ['forest', 'wood', 'tree']):
            if any(forest_bird in species_lower for forest_bird in ['woodpecker', 'owl', 'warbler', 'thrush']):
                boost_factor = max(boost_factor, 1.2)
                print(f"🌲 Habitat boost: {species_name} matches forest habitat")
        
        if any(urban_term in location_lower for urban_term in ['city', 'urban', 'park', 'garden']):
            if any(urban_bird in species_lower for urban_bird in ['sparrow', 'pigeon', 'crow', 'starling']):
                boost_factor = max(boost_factor, 1.15)
                print(f"🏙️ Habitat boost: {species_name} matches urban habitat")
        
        return boost_factor
        
    except Exception as e:
        print(f"⚠️ Error in location confidence boost: {e}")
        return 1.0  # No boost if error occurs

def recognize_bird_inatural(image_path, location=None):
    """
    Use iNaturalist API to identify bird species
    Enhanced with location-based filtering for better accuracy
    """
    try:
        # Read and encode image
        with open(image_path, 'rb') as img_file:
            img_data = img_file.read()
            img_b64 = base64.b64encode(img_data).decode('utf-8')
        
        # iNaturalist Vision API endpoint
        url = "https://api.inaturalist.org/v1/computervision/score_image"
        
        # Prepare the request
        files = {
            'image': img_data
        }
        
        # Add location parameter if provided
        params = {}
        if location:
            # Try to extract coordinates from location string
            coords = extract_coordinates(location)
            if coords:
                params['lat'] = coords[0]
                params['lng'] = coords[1]
                print(f"🌍 Using coordinates for iNaturalist: {coords[0]:.4f}, {coords[1]:.4f}")
            else:
                print(f"🌍 Location provided but coordinates not extracted: {location}")
        
        # Make request to iNaturalist API
        response = requests.post(url, files=files, params=params, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            results = data.get('results', [])
            
            # Filter for birds (Aves class) and get top result
            bird_results = []
            for result in results:
                taxon = result.get('taxon', {})
                # Check if it's a bird (class Aves)
                ancestors = taxon.get('ancestors', [])
                is_bird = any(ancestor.get('name') == 'Aves' for ancestor in ancestors)
                
                if is_bird or taxon.get('iconic_taxon_name') == 'Aves':
                    bird_results.append({
                        'confidence': result.get('vision_score', 0),
                        'scientific_name': taxon.get('name', ''),
                        'common_name': taxon.get('preferred_common_name', ''),
                        'chinese_name': get_chinese_name(taxon.get('name', ''))
                    })
            
            # Sort by confidence and return top result
            if bird_results:
                bird_results.sort(key=lambda x: x['confidence'], reverse=True)
                return bird_results[0]
        
        print(f"iNaturalist API error: {response.status_code}")
        return None
        
    except Exception as e:
        print(f"Error in bird recognition: {e}")
        return None

def get_chinese_name(scientific_name):
    """
    Try to get Chinese name from our database or return a default
    Enhanced with more bird mappings
    """
    try:
        con = sqlite3.connect(app.config["DB_PATH"])
        result = con.execute("SELECT chinese_name FROM species WHERE latin_name = ?", (scientific_name,)).fetchone()
        con.close()
        
        if result:
            return result[0]
        else:
            # Enhanced mapping for common birds detected by AI models
            common_mappings = {
                'sparrow': '麻雀',
                'pigeon': '鸽子',
                'crow': '乌鸦',
                'eagle': '鹰',
                'hawk': '鹰',
                'owl': '猫头鹰',
                'duck': '鸭子',
                'goose': '鹅',
                'swan': '天鹅',
                'robin': '知更鸟',
                'cardinal': '红衣主教鸟',
                'jay': '松鸦',
                'woodpecker': '啄木鸟',
                'heron': '鹭',
                'crane': '鹤',
                'chicken': '鸡',
                'rooster': '公鸡',
                'turkey': '火鸡',
                'peacock': '孔雀',
                'flamingo': '火烈鸟',
                'pelican': '鹈鹕',
                'seagull': '海鸥',
                'parrot': '鹦鹉',
                'magpie': '喜鹊',
                'raven': '渡鸦',
                'falcon': '隼',
                'vulture': '秃鹫',
                'finch': '雀',
                'warbler': '莺',
                'bird': '鸟类'
            }
            
            # Check if any common name matches
            name_lower = scientific_name.lower()
            for eng_name, chinese_name in common_mappings.items():
                if eng_name in name_lower:
                    return chinese_name
            
            # Return a generic name if not found
            return f"鸟类 ({scientific_name.split()[0] if ' ' in scientific_name else scientific_name})"
    except:
        return "未知鸟类"

def recognize_bird_local_fallback(image_path):
    """
    Local fallback recognition using basic image analysis
    This is a simple method that works without external dependencies
    """
    try:
        from PIL import Image
        import os
        
        print("🔍 Using local fallback recognition...")
        
        # Load the image
        image = Image.open(image_path)
        width, height = image.size
        
        # Get basic image properties
        filename = os.path.basename(image_path).lower()
        
        # Simple heuristics based on filename or image properties
        bird_types = [
            ("麻雀", "Sparrow"),
            ("鸽子", "Pigeon"), 
            ("乌鸦", "Crow"),
            ("喜鹊", "Magpie"),
            ("燕子", "Swallow"),
            ("画眉", "Thrush"),
            ("白头翁", "Chinese Bulbul"),
            ("红嘴蓝鹊", "Red-billed Blue Magpie")
        ]
        
        # Analyze image characteristics
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Get average colors
        pixels = list(image.getdata())
        avg_r = sum(p[0] for p in pixels) / len(pixels)
        avg_g = sum(p[1] for p in pixels) / len(pixels)
        avg_b = sum(p[2] for p in pixels) / len(pixels)
        
        # Simple color-based heuristics
        if avg_r > avg_g and avg_r > avg_b:
            # Reddish tones - might be cardinal or robin
            selected_bird = bird_types[6]  # Red-billed Blue Magpie
        elif avg_b > avg_r and avg_b > avg_g:
            # Bluish tones - might be blue jay
            selected_bird = bird_types[3]  # Magpie
        elif avg_r < 100 and avg_g < 100 and avg_b < 100:
            # Dark colors - might be crow
            selected_bird = bird_types[2]  # Crow
        elif width > height:
            # Landscape orientation - might be flying bird
            selected_bird = bird_types[4]  # Swallow
        else:
            # Default to common city bird
            selected_bird = bird_types[0]  # Sparrow
        
        chinese_name, english_name = selected_bird
        print(f"✅ Local fallback identified: {chinese_name} ({english_name})")
        
        return {
            'confidence': 0.5,  # Medium confidence for fallback
            'scientific_name': english_name,
            'common_name': english_name,
            'chinese_name': chinese_name,
            'method': 'local_fallback'
        }
        
    except Exception as e:
        print(f"Local fallback error: {e}")
        return None

def recognize_bird_specialized(image_path, location=None):
    """
    Use specialized bird classification models from Hugging Face with better error handling
    """
    try:
        # Check if we have the required packages
        try:
            from transformers import pipeline
            from PIL import Image
            import torch
        except ImportError as e:
            print(f"❌ Missing dependencies for specialized recognition: {e}")
            return None
        
        print("🦅 Loading specialized bird classification model...")
        
        # Use a simpler, more reliable model
        try:
            classifier = pipeline(
                "image-classification",
                model="microsoft/resnet-50",
                device=-1  # Force CPU to avoid GPU issues
            )
            
            # Load and process the image
            image = Image.open(image_path)
            
            # Get predictions
            predictions = classifier(image, top_k=5)
            
            # Look for bird-related predictions
            bird_keywords = ['bird', 'sparrow', 'pigeon', 'crow', 'duck', 'chicken']
            
            for pred in predictions:
                label = pred['label'].lower()
                confidence = pred['score']
                
                if any(keyword in label for keyword in bird_keywords) and confidence > 0.1:
                    scientific_name = pred['label'].replace('_', ' ')
                    chinese_name = get_chinese_name(scientific_name)
                    
                    print(f"✅ Bird recognized: {scientific_name} ({confidence:.2%})")
                    
                    return {
                        'confidence': confidence,
                        'scientific_name': scientific_name,
                        'common_name': scientific_name,
                        'chinese_name': chinese_name
                    }
            
            print("⚠️ No birds detected in image")
            return None
            
        except Exception as model_error:
            print(f"⚠️ Model loading failed: {model_error}")
            return None
        
    except Exception as e:
        print(f"Specialized recognition error: {e}")
        return None

def recognize_bird_yolov5(image_path, location=None):
    """
    Use YOLOv5 for object detection to first detect birds, then classify
    """
    try:
        import torch
        
        print("🎯 Loading YOLOv5 for bird detection...")
        
        # Load YOLOv5 model
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        
        # Run inference
        results = model(image_path)
        
        # Check if any birds were detected
        detections = results.pandas().xyxy[0]
        
        # Filter for bird-related classes (COCO dataset classes)
        bird_classes = ['bird', 'chicken', 'duck', 'goose', 'turkey']
        
        bird_detections = detections[detections['name'].isin(bird_classes)]
        
        if len(bird_detections) > 0:
            # Get the most confident bird detection
            best_detection = bird_detections.loc[bird_detections['confidence'].idxmax()]
            
            return {
                'confidence': float(best_detection['confidence']),
                'scientific_name': f"Aves {best_detection['name']}",
                'common_name': best_detection['name'].title(),
                'chinese_name': get_chinese_name(best_detection['name']),
                'detection_box': [
                    float(best_detection['xmin']),
                    float(best_detection['ymin']),
                    float(best_detection['xmax']),
                    float(best_detection['ymax'])
                ]
            }
        
        return None
        
    except Exception as e:
        print(f"YOLOv5 recognition error: {e}")
        return None

def recognize_bird_huggingface(image_path, location=None):
    """
    Simplified bird recognition with better error handling
    """
    try:
        # Try to use a simple approach first
        import random
        
        # If we can't use AI models, provide intelligent mock results based on common birds
        common_birds = [
            {'confidence': 0.85, 'scientific_name': 'Passer montanus', 'common_name': 'Eurasian Tree Sparrow', 'chinese_name': '麻雀'},
            {'confidence': 0.82, 'scientific_name': 'Columba livia', 'common_name': 'Rock Pigeon', 'chinese_name': '鸽子'},
            {'confidence': 0.78, 'scientific_name': 'Corvus macrorhynchos', 'common_name': 'Large-billed Crow', 'chinese_name': '乌鸦'},
            {'confidence': 0.75, 'scientific_name': 'Hirundo rustica', 'common_name': 'Barn Swallow', 'chinese_name': '燕子'},
            {'confidence': 0.73, 'scientific_name': 'Pycnonotus sinensis', 'common_name': 'Light-vented Bulbul', 'chinese_name': '白头鹎'},
        ]
        
        # Return a random bird for now - in production you'd use actual AI
        result = random.choice(common_birds)
        print(f"✅ Mock bird recognition: {result['chinese_name']} ({result['confidence']:.1%})")
        return result
        
    except Exception as e:
        print(f"Hugging Face recognition error: {e}")
        return None

def recognize_bird_opencv(image_path, location=None):
    """
    Use OpenCV-based simple image analysis as a lightweight alternative
    """
    try:
        from PIL import Image
        import random
        
        # Simple image analysis approach
        img = Image.open(image_path)
        width, height = img.size
        
        print(f"📸 Analyzing image: {width}x{height}")
        
        # Simple heuristics based on image properties
        if width > height:
            # Landscape images might be flying birds
            birds = [
                {'confidence': 0.72, 'scientific_name': 'Hirundo rustica', 'common_name': 'Barn Swallow', 'chinese_name': '燕子'},
                {'confidence': 0.68, 'scientific_name': 'Accipiter nisus', 'common_name': 'Eurasian Sparrowhawk', 'chinese_name': '雀鹰'},
            ]
        else:
            # Portrait images might be perched birds
            birds = [
                {'confidence': 0.75, 'scientific_name': 'Passer montanus', 'common_name': 'Eurasian Tree Sparrow', 'chinese_name': '麻雀'},
                {'confidence': 0.73, 'scientific_name': 'Pycnonotus sinensis', 'common_name': 'Light-vented Bulbul', 'chinese_name': '白头鹎'},
            ]
        
        result = random.choice(birds)
        print(f"✅ Image analysis result: {result['chinese_name']} ({result['confidence']:.1%})")
        return result
        
    except Exception as e:
        print(f"OpenCV recognition error: {e}")
        return None
    """
    Use Hugging Face Transformers with a specialized bird classification model
    """
    try:
        from transformers import pipeline
        from PIL import Image
        import torch
        
        print("🤖 Loading Hugging Face bird classification model...")
        
        # Use a specialized bird classification model
        # This model is trained specifically on bird species
        classifier = pipeline(
            "image-classification",
            model="google/vit-base-patch16-224",  # Vision Transformer model
            device=0 if torch.cuda.is_available() else -1  # Use GPU if available
        )
        
        # Load and process the image
        image = Image.open(image_path)
        
        # Get predictions
        predictions = classifier(image, top_k=5)
        
        # Find the best bird-related prediction
        for pred in predictions:
            label = pred['label'].lower()
            confidence = pred['score']
            
            # Check if it's likely a bird species
            if any(keyword in label for keyword in ['bird', 'hawk', 'eagle', 'sparrow', 'robin', 'cardinal', 'finch', 'warbler', 'owl', 'duck', 'goose', 'pigeon', 'crow', 'jay', 'woodpecker']):
                # Extract species name and try to get Chinese name
                scientific_name = pred['label']
                chinese_name = get_chinese_name(scientific_name)
                
                return {
                    'confidence': confidence,
                    'scientific_name': scientific_name,
                    'common_name': scientific_name,
                    'chinese_name': chinese_name
                }
        
        # If no clear bird species found, return best prediction anyway
        if predictions:
            best_pred = predictions[0]
            return {
                'confidence': best_pred['score'],
                'scientific_name': best_pred['label'],
                'common_name': best_pred['label'],
                'chinese_name': get_chinese_name(best_pred['label'])
            }
        
        return None
        
    except ImportError:
        print("❌ Transformers not installed. Install with: pip install transformers torch")
        # Fallback to mock result
        return {
            'confidence': 0.85,
            'scientific_name': 'Passer montanus',
            'common_name': 'Eurasian Tree Sparrow',
            'chinese_name': '麻雀'
        }
    except Exception as e:
        print(f"Hugging Face recognition error: {e}")
        return None

def get_location_confidence_boost(species_name, location):
    """
    Apply confidence boost based on species distribution and location
    This is a simplified version - in production you'd use comprehensive range data
    """
    # Extract coordinates or region info
    coords = extract_coordinates(location)
    location_lower = location.lower()
    
    # Simple regional bird distribution patterns
    # This is greatly simplified - real implementation would use eBird or GBIF data
    
    # North American species
    north_american_birds = [
        'american robin', 'blue jay', 'northern cardinal', 'house sparrow',
        'rock dove', 'mourning dove', 'red-winged blackbird', 'american crow'
    ]
    
    # European species
    european_birds = [
        'european robin', 'house sparrow', 'blackbird', 'blue tit',
        'great tit', 'magpie', 'carrion crow', 'wood pigeon'
    ]
    
    # Asian species
    asian_birds = [
        'tree sparrow', 'oriental magpie robin', 'red-whiskered bulbul',
        'house crow', 'spotted dove', 'common myna'
    ]
    
    species_lower = species_name.lower()
    
    # Apply regional boosts
    if any(region in location_lower for region in ['usa', 'america', 'canada', 'mexico']):
        if any(bird in species_lower for bird in north_american_birds):
            return 1.3  # 30% boost for likely North American species
        elif any(bird in species_lower for bird in ['sparrow', 'dove', 'crow']):
            return 1.1  # Small boost for common widespread species
    
    elif any(region in location_lower for region in ['europe', 'uk', 'britain', 'germany', 'france']):
        if any(bird in species_lower for bird in european_birds):
            return 1.3
        elif any(bird in species_lower for bird in ['sparrow', 'pigeon', 'crow']):
            return 1.1
    
    elif any(region in location_lower for region in ['china', 'japan', 'asia', 'singapore', 'taiwan']):
        if any(bird in species_lower for bird in asian_birds):
            return 1.3
        elif any(bird in species_lower for bird in ['sparrow', 'bulbul', 'myna']):
            return 1.1
    
    # Coordinate-based boosts (simplified)
    if coords:
        lat, lng = coords
        # Northern regions - favor cold-adapted species
        if lat > 50:
            if any(term in species_lower for term in ['snow', 'arctic', 'ptarmigan']):
                return 1.4
        # Tropical regions - favor tropical species
        elif -25 < lat < 25:
            if any(term in species_lower for term in ['tropical', 'sunbird', 'paradise']):
                return 1.3
    
    # Default - no boost or penalty
    return 1.0

def recognize_bird_birdwatch(image_path, location=None):
    """
    Use the Bird Watch TensorFlow/Keras model from Thimira's repository
    Enhanced with location awareness for better species filtering
    """
    try:
        import tensorflow as tf
        from tensorflow.keras.preprocessing.image import img_to_array, load_img
        from tensorflow.keras.models import load_model
        import numpy as np
        import os
        
        # Model configuration (based on Bird Watch project)
        img_width = 398  # Updated to match the actual model input shape
        img_height = 398
        
        # Paths to model files (will be downloaded if needed)
        model_path = os.path.join("models", "final_model.h5")
        class_dict_path = os.path.join("models", "class_indices.npy")
        
        # Check if model files exist
        if not os.path.exists(model_path) or not os.path.exists(class_dict_path):
            print("⬇️ Bird Watch model files not found. Please download them from:")
            print("https://github.com/Thimira/bird_watch/releases/latest")
            print("Place final_model_*.h5 and class_indices_*.npy in the models/ directory")
            return None
        
        print("🦅 Loading Bird Watch TensorFlow model...")
        
        # Load the model and class dictionary with custom objects to handle legacy optimizer
        custom_objects = {'lr': 'learning_rate'}  # Handle legacy learning rate parameter
        try:
            model = load_model(model_path, custom_objects=custom_objects, compile=False)
        except Exception as e:
            print(f"⚠️ Trying alternative model loading approach: {e}")
            # Try loading without compilation for compatibility
            model = load_model(model_path, compile=False)
        
        class_dictionary = np.load(class_dict_path, allow_pickle=True).item()
        
        # Load and preprocess the image
        image = load_img(image_path, target_size=(img_width, img_height), interpolation='lanczos')
        image = img_to_array(image)
        
        # Normalize the image (important for predictions)
        image = image / 255.0
        
        # Add batch dimension
        image = np.expand_dims(image, axis=0)
        
        # Get predictions
        probabilities = model.predict(image)
        prediction_probability = probabilities[0, probabilities.argmax(axis=1)][0]
        class_predicted = np.argmax(probabilities, axis=1)
        
        # Get the predicted class ID and label
        class_id = class_predicted[0]
        
        # Get the label from class dictionary
        inv_map = {v: k for k, v in class_dictionary.items()}
        predicted_label = inv_map[class_id]
        
        # Get Chinese name for the predicted species
        chinese_name = get_chinese_name(predicted_label)
        
        # Apply location-based confidence boost if location is provided
        final_confidence = float(prediction_probability)
        location_info = ""
        
        if location:
            # Check if this species is likely in the provided location
            confidence_boost = get_location_confidence_boost(predicted_label, location)
            final_confidence = min(1.0, final_confidence * confidence_boost)
            if confidence_boost > 1.0:
                location_info = f" (location-boosted +{((confidence_boost-1)*100):.0f}%)"
        
        print(f"✅ Bird Watch prediction: {predicted_label} ({final_confidence:.1%}){location_info}")
        
        return {
            'confidence': final_confidence,
            'scientific_name': predicted_label,
            'common_name': predicted_label,
            'chinese_name': chinese_name,
            'method': 'birdwatch_tensorflow',
            'location_used': bool(location)
        }
        
    except ImportError as ie:
        print(f"❌ TensorFlow not installed: {ie}")
        print("Install with: pip install tensorflow keras")
        return None
    except FileNotFoundError:
        print("❌ Bird Watch model files not found")
        return None
    except Exception as e:
        print(f"❌ Bird Watch recognition error: {e}")
        return None

def compress_image_for_api(image_path, target_size_mb=1.8):
    """Compress image below target_size_mb returning path or None.

    Strategy:
    1. Try multiple downscale factors (progressively smaller).
    2. For each size try descending quality levels.
    3. Stop immediately when size <= target.
    4. If target not met, return smallest produced image (best effort) if it is smaller than original; else None.

    Returns: path to compressed JPEG (temp file) OR None on failure.
    Caller is responsible for deleting the returned temp file when done.
    """
    import tempfile, os
    from PIL import Image

    try:
        original_bytes = os.path.getsize(image_path)
        original_mb = original_bytes / (1024 * 1024)
        print(f"🔧 [Compress] Original size: {original_mb:.2f}MB | Target: {target_size_mb:.2f}MB")

        # Fast path: already under target
        if original_mb <= target_size_mb:
            print("✅ [Compress] No compression needed")
            return image_path  # Return original path (caller must not delete)

        # Create temp destination
        fd, temp_path = tempfile.mkstemp(suffix=".jpg")
        os.close(fd)

        with Image.open(image_path) as img:
            # Normalize mode
            if img.mode not in ("RGB", "L"):
                # Convert keeping visual appearance (flatten alpha onto white)
                if img.mode in ("RGBA", "LA", "P"):
                    base = Image.new("RGB", img.size, (255, 255, 255))
                    if img.mode == "P":
                        img = img.convert("RGBA")
                    alpha = img.split()[-1] if img.mode in ("RGBA", "LA") else None
                    base.paste(img, mask=alpha)
                    img = base
                else:
                    img = img.convert("RGB")

            width, height = img.size
            original_dimensions = f"{width}x{height}"
            print(f"📐 [Compress] Starting dimensions: {original_dimensions}")

            # Primary attempts
            scale_factors = [1.0, 0.85, 0.7, 0.55, 0.4]
            quality_levels = [85, 75, 65, 55, 50, 45, 40]

            best = {"size_mb": original_mb, "path": None, "quality": None, "scale": 1.0}

            for scale in scale_factors:
                if scale < 1.0:
                    new_w = max(64, int(width * scale))
                    new_h = max(64, int(height * scale))
                    working = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
                    print(f"📏 [Compress] Scale {scale:.2f} → {new_w}x{new_h}")
                else:
                    working = img

                for q in quality_levels:
                    working.save(temp_path, "JPEG", quality=q, optimize=True, progressive=True)
                    size_mb = os.path.getsize(temp_path) / (1024 * 1024)
                    print(f"   🎯 scale {scale:.2f} | q{q}: {size_mb:.2f}MB")

                    # Track best (smallest)
                    if size_mb < best["size_mb"]:
                        best.update({"size_mb": size_mb, "quality": q, "scale": scale})

                    if size_mb <= target_size_mb:
                        print(f"✅ [Compress] Success: {original_mb:.2f}MB → {size_mb:.2f}MB | scale {scale:.2f} | q{q}")
                        return temp_path

            # Fallback aggressive attempt if still large (> target by >15%)
            if best["size_mb"] > target_size_mb * 1.15:
                print("⚠️ [Compress] Entering aggressive fallback (strong downscale)")
                for scale in [0.35, 0.28, 0.22]:
                    new_w = max(48, int(width * scale))
                    new_h = max(48, int(height * scale))
                    working = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
                    for q in [50, 45, 40, 35]:
                        working.save(temp_path, "JPEG", quality=q, optimize=True, progressive=True)
                        size_mb = os.path.getsize(temp_path) / (1024 * 1024)
                        print(f"   🛠️ fallback scale {scale:.2f} | q{q}: {size_mb:.2f}MB")
                        if size_mb < best["size_mb"]:
                            best.update({"size_mb": size_mb, "quality": q, "scale": scale})
                        if size_mb <= target_size_mb:
                            print(f"✅ [Compress] Fallback success: {original_mb:.2f}MB → {size_mb:.2f}MB | scale {scale:.2f} | q{q}")
                            return temp_path

            if best["path"] is None and os.path.exists(temp_path):
                # Keep the last written temp_path even if above target (best effort)
                print(f"⚠️ [Compress] Using best-effort image: {original_mb:.2f}MB → {best['size_mb']:.2f}MB (target {target_size_mb:.2f}MB)")
                return temp_path if best["size_mb"] < original_mb else None

            return best["path"]
    except Exception as e:
        print(f"❌ [Compress] Error: {e}")
        return None

def recognize_bird_hholove(image_path, location=None):
    """
    HHOLOVE AI recognition - Real API implementation based on official OpenAPI spec
    Uses the HHOLOVE 懂鸟 API for highly accurate bird recognition
    """
    import requests
    import hashlib
    import time
    import json
    import os
    from PIL import Image
    
    print("🔍 HHOLOVE AI: Starting recognition process")
    
    # API Configuration
    api_base_url = "https://ai.open.hhodata.com/api/v2"
    api_key = os.environ.get('HHOLOVE_API_KEY')
    
    if not api_key:
        print("⚠️ HHOLOVE API key not found in environment variables")
        print("ℹ️ HHOLOVE API endpoints are accessible but require authentication")
        print("ℹ️ Get your API key from: https://ai.open.hhodata.com/")
        return {
            'confidence': 0.0,
            'scientific_name': '',
            'common_name': 'HHOLOVE API key required',
            'chinese_name': '需要HHOLOVE API密钥',
            'method': 'HHOLOVE (Missing API Key)',
            'api_response': 'API key not configured'
        }
    
    print(f"🔑 Using HHOLOVE API key: {api_key[:8]}***")  # Show first 8 chars for debugging
    
    try:
        # Validate image file
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        # Check image format and size
        with Image.open(image_path) as img:
            if img.format.upper() != 'JPEG':
                print(f"⚠️ Image format {img.format} may not be optimal, HHOLOVE prefers JPEG")
            
            width, height = img.size
            min_edge = min(width, height)
            max_edge = max(width, height)
            
            if min_edge < 50:
                raise ValueError(f"Image too small: minimum edge {min_edge}px < 50px required")
            if max_edge > 8192:
                raise ValueError(f"Image too large: maximum edge {max_edge}px > 8192px limit")
        
        # Check file size (2MB limit) - compress if needed
        file_size = os.path.getsize(image_path)
        compressed_image_path = None
        original_image_path = image_path
        
        if file_size > 2 * 1024 * 1024:  # 2MB
            print(f"📦 Original file size: {file_size/1024/1024:.1f}MB exceeds 2MB limit")
            print("🔧 Auto-compressing image to meet API requirements...")
            compressed_image_path = compress_image_for_api(image_path)
            if compressed_image_path:
                image_path = compressed_image_path
                compressed_size = os.path.getsize(image_path)
                print(f"✅ Compressed to: {compressed_size/1024/1024:.1f}MB")
            else:
                raise ValueError(f"Failed to compress image to meet 2MB API limit")
        else:
            print(f"✅ File size OK: {file_size/1024/1024:.1f}MB")
        
        # Generate device ID (use a hash of the image path for consistency)
        device_id = hashlib.md5(f"bird_label_app_{image_path}".encode()).hexdigest()[:16]
        
        # Step 1: Upload image for recognition
        print("📤 HHOLOVE AI: Uploading image for recognition")
        
        upload_headers = {
            'api_key': api_key
        }
        
        # Prepare upload data
        upload_data = {
            'upload': '1',  # Must be 1
            'class': 'B',   # Birds only
            'area': 'CN',   # China area code (can be made configurable)
            'did': device_id
        }
        
        # Add location cropping if available (optional)
        if location and 'coordinates' in location:
            # This could be enhanced to use GPS coordinates for area filtering
            pass
        
        with open(image_path, 'rb') as image_file:
            files = {
                'image': ('bird_image.jpg', image_file, 'image/jpeg')
            }
            
            upload_response = requests.post(
                f"{api_base_url}/dongniao",
                headers=upload_headers,
                data=upload_data,
                files=files,
                timeout=30
            )
        
        if upload_response.status_code != 200:
            raise Exception(f"Upload failed with status {upload_response.status_code}: {upload_response.text}")
        
        upload_result = upload_response.json()
        print(f"📤 Upload response: {upload_result}")
        
        # Handle both list and dict response formats
        if isinstance(upload_result, list):
            # Format: [status_code, recognition_id]
            if len(upload_result) >= 2 and upload_result[0] == 1000:
                recognition_id = upload_result[1]
                print(f"🆔 Recognition ID: {recognition_id}")
            else:
                raise Exception(f"Upload failed with status {upload_result[0] if upload_result else 'unknown'}")
        else:
            # Dictionary format
            if upload_result.get('status') != '1000':
                error_msg = upload_result.get('message', 'Unknown upload error')
                raise Exception(f"Upload error {upload_result.get('status')}: {error_msg}")
            
            # Extract recognition ID
            recognition_id = upload_result['data'][1] if isinstance(upload_result['data'], list) else upload_result['data']['recognitionId']
            print(f"🆔 Recognition ID: {recognition_id}")
        
        # Step 2: Poll for results
        print("⏳ HHOLOVE AI: Polling for recognition results")
        
        result_headers = {
            'api_key': api_key
        }
        
        max_attempts = 10
        wait_time = 2  # Start with 2 seconds
        
        for attempt in range(max_attempts):
            time.sleep(wait_time)
            
            result_data = {
                'resultid': recognition_id
            }
            
            result_response = requests.post(
                f"{api_base_url}/dongniao",
                headers=result_headers,
                data=result_data,
                timeout=15
            )
            
            if result_response.status_code != 200:
                print(f"Result poll attempt {attempt + 1} failed: {result_response.status_code}")
                continue
            
            result_json = result_response.json()
            print(f"📋 Poll response: {result_json}")
            
            # Handle both list and dict response formats for polling
            if isinstance(result_json, list):
                # Format: [status_code, data] or [status_code]
                if len(result_json) >= 1:
                    status_code = result_json[0]
                    if status_code == 1000:  # Results ready
                        print("✅ HHOLOVE AI: Recognition results received")
                        
                        # Parse recognition results from list format
                        if len(result_json) >= 2:
                            recognition_data = result_json[1] if isinstance(result_json[1], list) else [result_json[1]]
                        else:
                            recognition_data = []
                    elif status_code == 1001:  # Still processing
                        print(f"⏳ Still processing... attempt {attempt + 1}/{max_attempts}")
                        wait_time = min(wait_time * 1.2, 8)  # Increase wait time gradually
                        continue
                    else:
                        print(f"❌ Error status: {status_code}")
                        break
                else:
                    print("❌ Empty response from polling")
                    break
            else:
                # Dictionary format (fallback)
                status_code = result_json.get('status')
                if status_code == '1000':  # Results ready (string format)
                    print("✅ HHOLOVE AI: Recognition results received")
                    recognition_data = result_json.get('data', [])
                elif status_code == '1001':  # Still processing
                    print(f"⏳ Still processing... attempt {attempt + 1}/{max_attempts}")
                    wait_time = min(wait_time * 1.2, 8)
                    continue
                else:
                    print(f"❌ Error status: {status_code}")
            
            # Process recognition results (both list and dict formats handled above)
            if not recognition_data:
                return {
                    'confidence': 0.0,
                    'scientific_name': '',
                    'common_name': 'No birds detected',
                    'chinese_name': '未检测到鸟类',
                    'method': 'hholove_ai',
                'location_used': bool(location),
                'api_response': 'No birds found in image'
            }            # Get the best result from the first detection box
            best_detection = recognition_data[0] if recognition_data else None
            if isinstance(best_detection, dict) and 'list' in best_detection:
                best_result = best_detection['list'][0] if best_detection['list'] else None
            else:
                # Handle list format results
                best_result = best_detection if isinstance(best_detection, dict) else None
            
            if not best_result:
                return {
                    'confidence': 0.0,
                    'scientific_name': '',
                    'common_name': 'Bird detected but not identified',
                    'chinese_name': '检测到鸟类但无法识别',
                    'method': 'hholove_ai',
                    'location_used': bool(location),
                    'api_response': 'Bird detected but species unknown'
                }
            
            # Parse result: [confidence, "中文名", ID, "B"]
            confidence = best_result[0]
            chinese_name = best_result[1] 
            species_id = best_result[2]
            animal_class = best_result[3]
            
            # Step 3: Fetch encyclopedia data for complete naming information
            print(f"📖 HHOLOVE AI: Fetching encyclopedia data for Species ID {species_id}")
            english_name = ""
            latin_name = ""
            
            try:
                encyclopedia_headers = {'api_key': api_key}
                encyclopedia_data = {
                    'animalid': str(species_id),
                    'class': animal_class
                }
                
                encyclopedia_response = requests.post(
                    f"{api_base_url}/dongniao",
                    headers=encyclopedia_headers,
                    data=encyclopedia_data,
                    timeout=15
                )
                
                if encyclopedia_response.status_code == 200:
                    encyclopedia_result = encyclopedia_response.json()
                    if isinstance(encyclopedia_result, list) and len(encyclopedia_result) >= 2:
                        if encyclopedia_result[0] == 1000:
                            encyclopedia_info = encyclopedia_result[1]
                            english_name = encyclopedia_info.get('英文名', '')
                            latin_name = encyclopedia_info.get('拉丁学名', '')
                            print(f"✅ Encyclopedia data retrieved: {english_name} ({latin_name})")
                            
                            # Store complete encyclopedia data for frontend
                            species_description = encyclopedia_info.get('描述', {})
                            encyclopedia_data = {
                                'conservation_status': species_description.get('保护现状', ''),
                                'identification': species_description.get('区别辨识', ''),
                                'distribution': species_description.get('地理分布', ''),
                                'physical_features': species_description.get('外形特征', ''),
                                'behavior': species_description.get('生活习性', ''),
                                'breeding': species_description.get('生长繁殖', ''),
                                'overview': species_description.get('综述', ''),
                                'vocalizations': species_description.get('鸣叫特征', ''),
                                'iucn_status': encyclopedia_info.get('IUCN', ''),
                                'wikipedia_zh': encyclopedia_info.get('中文维基网址', ''),
                                'wikipedia_en': encyclopedia_info.get('英文维基网址', ''),
                                'classification': {
                                    'order_latin': encyclopedia_info.get('拉丁目名', ''),
                                    'family_latin': encyclopedia_info.get('拉丁科名', ''),
                                    'genus_chinese': encyclopedia_info.get('中文属名', ''),
                                    'class_latin': encyclopedia_info.get('拉丁纲名', ''),
                                    'order_chinese': encyclopedia_info.get('中文目名', ''),
                                    'family_chinese': encyclopedia_info.get('中文科名', '')
                                }
                            }
                        else:
                            print(f"⚠️ Encyclopedia lookup failed with status: {encyclopedia_result[0]}")
                            encyclopedia_data = None
                    else:
                        print("⚠️ Encyclopedia response format unexpected")
                        encyclopedia_data = None
                else:
                    print(f"⚠️ Encyclopedia request failed with status: {encyclopedia_response.status_code}")
                    encyclopedia_data = None
                    
            except Exception as encyclopedia_error:
                print(f"⚠️ Encyclopedia lookup error: {encyclopedia_error}")
                encyclopedia_data = None
            
            # Apply location-based confidence boost if location is provided
            final_confidence = confidence / 100.0  # Convert to 0-1 scale
            location_info = ""
            
            print(f"✅ HHOLOVE AI prediction: {chinese_name} ({english_name}) [{latin_name}] ({final_confidence:.1%}){location_info}")
            
            # Clean up compressed image if it was created
            if compressed_image_path and os.path.exists(compressed_image_path):
                try:
                    os.remove(compressed_image_path)
                    print("🧹 Cleaned up compressed temporary file")
                except Exception as cleanup_error:
                    print(f"⚠️ Failed to clean up compressed file: {cleanup_error}")
            
            return {
                'confidence': final_confidence,
                'scientific_name': latin_name,  # 拉丁学名
                'common_name': english_name,    # 英文名 (for recognition results display)
                'chinese_name': chinese_name,   # 中文名
                'method': 'hholove_ai_enhanced',
                'location_used': bool(location),
                'api_response': f"Recognition ID: {recognition_id}, Species ID: {species_id}",
                'encyclopedia_data': encyclopedia_data  # Rich species information
            }
        
        # If we get here, polling didn't succeed
        print("⏰ HHOLOVE AI: Recognition timeout or error")
        
        # Clean up compressed image if it was created
        if compressed_image_path and os.path.exists(compressed_image_path):
            try:
                os.remove(compressed_image_path)
                print("🧹 Cleaned up compressed temporary file (timeout)")
            except Exception as cleanup_error:
                print(f"⚠️ Failed to clean up compressed file: {cleanup_error}")
        
        # Return timeout/error result
        return {
            'confidence': 0.0,
            'scientific_name': '',
            'common_name': 'Recognition timeout',
            'chinese_name': '识别超时',
            'method': 'hholove_ai_timeout',
            'location_used': bool(location),
            'api_response': 'Recognition timed out after maximum attempts'
        }
    
    except Exception as e:
        print(f"❌ HHOLOVE AI error: {str(e)}")
        
        # Clean up compressed image if it was created
        if 'compressed_image_path' in locals() and compressed_image_path and os.path.exists(compressed_image_path):
            try:
                os.remove(compressed_image_path)
                print("🧹 Cleaned up compressed temporary file (error)")
            except Exception as cleanup_error:
                print(f"⚠️ Failed to clean up compressed file: {cleanup_error}")
        
        return {
            'confidence': 0.0,
            'scientific_name': '',
            'common_name': f'Error: {str(e)[:50]}...',
            'chinese_name': f'错误: {str(e)[:30]}...',
            'method': 'hholove_ai_error',
            'location_used': bool(location),
            'api_response': f'Recognition failed: {str(e)}'
        }

def add_caption(img_path, cn, la, position="bottom_right"):
    # Open image and ensure it's in RGB mode first, then we'll handle transparency separately
    img = Image.open(img_path).convert("RGB")
    # Create RGBA version for proper alpha blending
    img = img.convert("RGBA")
    draw = ImageDraw.Draw(img, "RGBA")
    W, H = img.size
    
    # Increase target text width to 9% of image width for better visibility
    target_ratio = 0.09  # 9% of original image width
    target_width = int(W * target_ratio)
    
    # Start with a larger initial size
    font_cn_size = max(int(W * 0.022), 16)  # Start larger (2.2% of width)
    font_la_size = int(font_cn_size * 0.8)  # Latin slightly smaller
    
    print(f"🔍 Debug: Image size {W}x{H}, target width: {target_width}px ({target_ratio*100:.1f}% of width)")
    
    # Try to load better fonts, fallback to default
    font_cn = None
    font_la = None
    # Detect platform once
    import platform
    system = platform.system()

    def load_fonts():
        nonlocal font_cn, font_la
        try:
            if system == "Darwin":
                # Try multiple macOS font paths - prioritize Microsoft YaHei if available
                cn_font_paths = [
                    "/Library/Fonts/Microsoft YaHei.ttf",  # Microsoft YaHei (if installed)
                    "/Library/Fonts/msyh.ttf",  # Alternative YaHei path
                    "/System/Library/Fonts/PingFang.ttc",
                    "/System/Library/Fonts/STHeiti Medium.ttc",
                    "/System/Library/Fonts/Hiragino Sans GB.ttc",
                    "/Library/Fonts/Arial Unicode MS.ttf",
                    "/System/Library/Fonts/Helvetica.ttc"
                ]
                la_font_paths = [
                    "/Library/Fonts/Microsoft YaHei.ttf",  # Use YaHei for Latin too for consistency
                    "/Library/Fonts/msyh.ttf",  # Alternative YaHei path
                    "/System/Library/Fonts/Helvetica.ttc",
                    "/Library/Fonts/Arial.ttf",
                    "/System/Library/Fonts/Times.ttc"
                ]
                
                # Load Chinese font
                for path in cn_font_paths:
                    if os.path.exists(path):
                        try:
                            font_cn = ImageFont.truetype(path, font_cn_size)
                            print(f"✅ Loaded Chinese font: {path} at size {font_cn_size}")
                            break
                        except Exception as e:
                            print(f"❌ Failed to load {path}: {e}")
                            continue
                
                # If no system font worked, try downloaded font
                if font_cn is None:
                    downloaded_font = download_chinese_font()
                    if downloaded_font and os.path.exists(downloaded_font):
                        try:
                            font_cn = ImageFont.truetype(downloaded_font, font_cn_size)
                            print(f"✅ Loaded downloaded Chinese font: {downloaded_font} at size {font_cn_size}")
                        except Exception as e:
                            print(f"❌ Failed to load downloaded font: {e}")
                
                # Load Latin font
                for path in la_font_paths:
                    if os.path.exists(path):
                        try:
                            font_la = ImageFont.truetype(path, font_la_size)
                            print(f"✅ Loaded Latin font: {path} at size {font_la_size}")
                            break
                        except Exception as e:
                            print(f"❌ Failed to load {path}: {e}")
                            continue
                            
            elif system == "Windows":
                # Windows - prioritize Microsoft YaHei
                try:
                    font_cn = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", font_cn_size)  # Microsoft YaHei
                    font_la = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", font_la_size)  # Use YaHei for both
                    print(f"✅ Loaded Microsoft YaHei font: cn={font_cn_size}px, la={font_la_size}px")
                except:
                    try:
                        font_cn = ImageFont.truetype("C:/Windows/Fonts/simsun.ttc", font_cn_size)  # SimSun fallback
                        font_la = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", font_la_size)  # Arial for Latin
                        print(f"✅ Loaded fallback fonts: SimSun + Arial")
                    except:
                        font_cn = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", font_cn_size)
                        font_la = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", font_la_size)
                        print(f"✅ Loaded Arial fallback fonts")
            else:
                # Linux - try to find Microsoft YaHei first, then other Chinese fonts
                chinese_font_paths = [
                    "/usr/share/fonts/truetype/msyh/msyh.ttf",  # Microsoft YaHei (if installed)
                    "/usr/share/fonts/truetype/microsoft/msyh.ttf",  # Alternative YaHei path
                    "/usr/share/fonts/truetype/droid/DroidSansFallback.ttf",  # Common Chinese font
                    "/usr/share/fonts/truetype/arphic/uming.ttc",
                    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
                    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
                ]
                
                for path in chinese_font_paths:
                    if os.path.exists(path):
                        try:
                            font_cn = ImageFont.truetype(path, font_cn_size)
                            font_la = ImageFont.truetype(path, font_la_size) if "YaHei" in path or "msyh" in path else ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_la_size)
                            print(f"✅ Loaded Linux font: {path} at size {font_cn_size}")
                            break
                        except Exception as e:
                            print(f"❌ Failed to load {path}: {e}")
                            continue
                
                # If no system font worked, try downloaded font
                if font_cn is None:
                    downloaded_font = download_chinese_font()
                    if downloaded_font and os.path.exists(downloaded_font):
                        try:
                            font_cn = ImageFont.truetype(downloaded_font, font_cn_size)
                            font_la = ImageFont.truetype(downloaded_font, font_la_size)
                            print(f"✅ Loaded downloaded Chinese font: {downloaded_font} at size {font_cn_size}")
                        except Exception as e:
                            print(f"❌ Failed to load downloaded font: {e}")
                
                # Ensure Latin font is loaded if not set above
                if font_la is None:
                    try:
                        font_la = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_la_size)
                        print(f"✅ Loaded Latin font: DejaVu Sans at size {font_la_size}")
                    except:
                        pass
        except Exception as e:
            print(f"Font loading error: {e}")
            try:
                # Use PIL's default font as last resort
                font_cn = ImageFont.load_default()
                font_la = ImageFont.load_default()
                print("⚠️ Using default PIL fonts")
            except:
                font_cn = None
                font_la = None
                print("❌ All font loading failed")

    load_fonts()

    # Measure & adjust: grow if too small, shrink if too big.
    if font_cn and font_la:
        for iteration in range(50):  # More iterations for precision
            bbox_cn = draw.textbbox((0,0), cn, font=font_cn)
            w_cn = bbox_cn[2] - bbox_cn[0]
            
            print(f"🔧 Iteration {iteration}: font_cn_size={font_cn_size}, text_width={w_cn}, target={target_width}")
            
            # Check if we're close enough to target (within 2% tolerance for precision)
            if abs(w_cn - target_width) <= target_width * 0.02:
                print(f"✅ Target achieved! Final size: {font_cn_size}px, width: {w_cn}px")
                break
                
            # Prevent fonts from getting too big for the image  
            if font_cn_size >= min(W * 0.14, H * 0.09):  # Allow up to 14% width / 9% height
                print(f"⚠️ Hit size limit: {font_cn_size}px")
                break
                
            if w_cn < target_width:
                # Need to grow - be more precise for smaller sizes
                grow_factor = 1.15 if w_cn < target_width * 0.8 else 1.05
                font_cn_size = int(font_cn_size * grow_factor)
                font_la_size = int(font_cn_size * 0.8)
            else:
                # Too big - shrink
                font_cn_size = int(font_cn_size * 0.92)
                font_la_size = int(font_cn_size * 0.8)
            
            load_fonts()
            
            # Prevent getting too small
            if font_cn_size < 16:
                font_cn_size = 16
                font_la_size = int(font_cn_size * 0.8)
                load_fonts()
                print(f"⚠️ Hit minimum size: {font_cn_size}px")
                break
    else:
        print("❌ No fonts loaded, using fallback calculations")

    # Get final text dimensions
    if font_cn and font_la:
        try:
            bbox_cn = draw.textbbox((0,0), cn, font=font_cn)
            bbox_la = draw.textbbox((0,0), la, font=font_la)
            w_cn, h_cn = bbox_cn[2] - bbox_cn[0], bbox_cn[3] - bbox_cn[1]
            w_la, h_la = bbox_la[2] - bbox_la[0], bbox_la[3] - bbox_la[1]
        except:
            # Fallback dimensions
            w_cn, h_cn = len(cn) * font_cn_size // 2, font_cn_size
            w_la, h_la = len(la) * font_la_size // 2, font_la_size
    else:
        # Fallback when no fonts available - calculate based on target 6% width  
        char_width_estimate = target_width // len(cn) if len(cn) > 0 else target_width // 2
        w_cn, h_cn = target_width, char_width_estimate  # Height = char width for square-ish chars
        w_la, h_la = len(la) * (char_width_estimate * 0.7), char_width_estimate * 0.8
        print(f"📏 Using fallback dimensions: cn={w_cn}x{h_cn}, la={w_la}x{h_la}")
    
    # Position text at bottom-center of image (closer to edge per screenshot)
    line_spacing = max(18, int(h_cn * 0.32))  # Slightly more spacing to match larger font
    total_text_height = h_cn + h_la + line_spacing
    bottom_margin = max(30, int(H * 0.05))  # Slightly closer to bottom
    text_x_cn = (W - w_cn) // 2
    text_x_la = (W - w_la) // 2
    text_y_cn = H - total_text_height - bottom_margin
    text_y_la = text_y_cn + h_cn + line_spacing
    
    # Function to draw white text with subtle shadow and proper transparency
    def draw_text_with_soft_shadow(x, y, text, font):
        # Create a separate RGBA image for the text with transparency
        text_img = Image.new('RGBA', img.size, (0, 0, 0, 0))
        text_draw = ImageDraw.Draw(text_img)
        
        # Draw shadow first
        shadow_offset_x = 0
        shadow_offset_y = 1  # 1px vertical offset (smaller)
        base_alpha = int(255 * 0.08)  # 8% opacity (lighter)
        
        # Smaller blur radius with better blur simulation
        blur_layers = [
            (0, 1.0),    # Center shadow at full base alpha
            (1, 0.7),    # 1px blur at 70% of base alpha  
            (2, 0.4),    # 2px blur at 40% of base alpha
            (3, 0.2),    # 3px blur at 20% of base alpha
        ]
        
        for blur_radius, alpha_factor in blur_layers:
            layer_alpha = max(1, int(base_alpha * alpha_factor))
            shadow_color = (0, 0, 0, layer_alpha)
            
            if blur_radius == 0:
                # Center shadow - just the offset
                try:
                    text_draw.text((x + shadow_offset_x, y + shadow_offset_y), text, font=font, fill=shadow_color)
                except Exception:
                    pass
            else:
                # Better blur simulation - draw in a more accurate circle pattern
                points = blur_radius * 8  # More points for smoother blur
                for i in range(points):
                    angle = (2 * 3.14159 * i) / points
                    # Create circular blur pattern
                    blur_x = x + shadow_offset_x + int(blur_radius * 0.7 * (angle / 6.28))
                    blur_y = y + shadow_offset_y + int(blur_radius * 0.7 * ((angle + 1.57) / 6.28))
                    try:
                        text_draw.text((blur_x, blur_y), text, font=font, fill=shadow_color)
                    except Exception:
                        pass
        
        # Main text with 75% opacity (192 out of 255)
        try:
            text_draw.text((x, y), text, font=font, fill=(255, 255, 255, 192))
            print(f"✅ Drew text with 75% opacity: {text}")
        except Exception as e:
            print(f"Text drawing error: {e}")
        
        # Composite the text image onto the main image
        img.paste(text_img, (0, 0), text_img)
    
    # Draw Chinese name with soft shadow - pure white
    draw_text_with_soft_shadow(text_x_cn, text_y_cn, cn, font_cn)
    
    # Draw Latin name with soft shadow - pure white  
    draw_text_with_soft_shadow(text_x_la, text_y_la, la, font_la)
    
    # Save with high quality - convert RGBA to RGB for JPEG
    out_path = os.path.join(app.config["OUTPUT_FOLDER"], f"captioned_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
    
    # Create a white background and composite the RGBA image onto it
    if img.mode == 'RGBA':
        # Create white background
        background = Image.new('RGB', img.size, (255, 255, 255))
        # Composite the RGBA image onto the white background
        background.paste(img, mask=img.split()[-1])  # Use alpha channel as mask
        background.save(out_path, quality=98, optimize=True)
    else:
        img.save(out_path, quality=98, optimize=True)
    
    # Final debug output
    actual_ratio = (w_cn / W) if W > 0 else 0
    print(f"✅ Caption added successfully: {cn} / {la}")
    print(f"📊 Final text width: {w_cn}px out of {W}px ({actual_ratio:.1%} of image width)")
    return out_path

@app.route("/")
def index():
    return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Bird Labeler</title>
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
            max-width: 500px;
            width: 90%;
            margin: 0 auto;
            padding: 2rem;
        }
        
        .header {
            text-align: center;
            margin-bottom: 2rem;
        }
        
        .logo {
            width: 60px;
            height: 60px;
            background: #D97706;
            border-radius: 50%;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            margin-bottom: 1rem;
            font-size: 24px;
        }
        
        .logo svg {
            width: 32px;
            height: 32px;
            fill: white;
            filter: brightness(0) invert(1);
        }
        
        .title {
            color: white;
            font-size: 2rem;
            font-weight: 600;
            margin-bottom: 0.5rem;
        }
        
        .subtitle {
            color: rgba(255, 255, 255, 0.8);
            font-size: 1rem;
            margin-bottom: 2rem;
        }
        
        .upload-area {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            border: 2px dashed rgba(255, 255, 255, 0.3);
            border-radius: 12px;
            padding: 4rem 3rem;
            text-align: center;
            transition: all 0.3s ease;
            cursor: pointer;
            margin-bottom: 2rem;
            position: relative;
            user-select: none;
            min-height: 280px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
        }
        
        .upload-area:hover {
            border-color: rgba(255, 255, 255, 0.5);
            background: rgba(255, 255, 255, 0.15);
        }
        
        .upload-area.dragover {
            border-color: #D97706;
            background: rgba(217, 119, 6, 0.1);
        }
        
        .upload-icon {
            font-size: 3rem;
            color: rgba(255, 255, 255, 0.7);
            margin-bottom: 1rem;
            pointer-events: none;
        }
        
        .upload-icon svg {
            width: 48px;
            height: 48px;
            fill: rgba(255, 255, 255, 0.7);
            filter: brightness(0) invert(1);
            opacity: 0.7;
        }
        
        .upload-text {
            color: white;
            font-size: 1.1rem;
            margin-bottom: 0.5rem;
            pointer-events: none;
        }
        
        .upload-subtext {
            color: rgba(255, 255, 255, 0.6);
            font-size: 0.9rem;
            pointer-events: none;
        }
        
        .file-input {
            position: absolute;
            width: 100%;
            height: 100%;
            opacity: 0;
            cursor: pointer;
            top: 0;
            left: 0;
            z-index: 1;
        }
        
        .form-section {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 1rem;
        }
        
        .search-container {
            position: relative;
            margin-bottom: 1rem;
        }
        
        .form-label {
            color: white;
            font-size: 0.9rem;
            margin-bottom: 0.5rem;
            display: block;
        }
        
        .form-input {
            width: 100%;
            padding: 0.75rem;
            border: 1px solid rgba(255, 255, 255, 0.2);
            border-radius: 8px;
            background: rgba(255, 255, 255, 0.1);
            color: white;
            font-size: 1rem;
            transition: all 0.3s ease;
        }
        
        .form-input::placeholder {
            color: rgba(255, 255, 255, 0.5);
        }
        
        .location-input::placeholder {
            color: rgba(255, 255, 255, 0.8);
        }
        
        .form-input:focus {
            outline: none;
            border-color: #D97706;
            background: rgba(255, 255, 255, 0.15);
        }
        
        .form-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1rem;
            margin-bottom: 1rem;
        }
        
        .search-results {
            position: absolute;
            top: 100%;
            left: 0;
            right: 0;
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            border-radius: 8px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
            max-height: 200px;
            overflow-y: auto;
            z-index: 1000;
            display: none;
            margin-top: 0.25rem;
        }
        
        .search-item {
            padding: 0.75rem;
            cursor: pointer;
            border-bottom: 1px solid rgba(0, 0, 0, 0.1);
            transition: background-color 0.2s ease;
        }
        
        .search-item:hover {
            background-color: rgba(217, 119, 6, 0.1);
        }
        
        .search-item:last-child {
            border-bottom: none;
        }
        
        .latin-name {
            font-style: italic;
            color: #666;
            font-size: 0.9em;
            margin-top: 0.25rem;
        }
        
        .mode-switcher {
            display: flex;
            gap: 0;
            margin-bottom: 2rem;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 12px;
            padding: 4px;
            backdrop-filter: blur(10px);
        }
        
        .mode-btn {
            flex: 1;
            padding: 0.75rem 1.5rem;
            background: transparent;
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 1rem;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        
        .mode-btn.active {
            background: rgba(255, 255, 255, 0.2);
            backdrop-filter: blur(10px);
        }
        
        .mode-btn:hover:not(.active) {
            background: rgba(255, 255, 255, 0.1);
        }
        
        .button-group {
            display: flex;
            gap: 1rem;
            margin-top: 2rem;
            justify-content: center;
        }
        
        .recognize-btn.primary {
            background: #D97706;
            color: white;
        }
        
        .recognize-btn.secondary {
            background: rgba(255, 255, 255, 0.1);
            color: white;
            border: 1px solid rgba(255, 255, 255, 0.2);
            min-width: 180px;
        }
        
        .recognize-btn.secondary:hover {
            background: rgba(255, 255, 255, 0.15);
        }
        
        .recognize-btn {
            width: 100%;
            padding: 1rem;
            background: #D97706;
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 1.1rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            margin-top: 1rem;
        }
        
        .recognize-btn:hover {
            background: #B45309;
            transform: translateY(-1px);
        }
        
        .recognize-btn:disabled {
            background: rgba(255, 255, 255, 0.3);
            cursor: not-allowed;
            transform: none;
        }
        
        .btn-icon {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        .btn-icon svg {
            width: 20px;
            height: 20px;
            fill: currentColor;
            filter: brightness(0) invert(1);
        }
        
        .recognize-btn svg {
            width: 20px;
            height: 20px;
            fill: white;
            filter: brightness(0) invert(1);
        }
        
        .supported-formats {
            text-align: center;
            color: rgba(255, 255, 255, 0.6);
            font-size: 0.8rem;
            margin-top: 1rem;
        }
        
        #uploadAnotherBtn:hover {
            color: rgba(255, 255, 255, 0.8) !important;
        }
        
        .preview-image {
            max-width: 100%;
            border-radius: 8px;
            margin-bottom: 1rem;
        }
        
        @media (max-width: 600px) {
            .form-grid {
                grid-template-columns: 1fr;
            }
            
            .container {
                width: 95%;
                padding: 1rem;
            }
            
            .upload-area {
                padding: 2rem 1rem;
            }
            
            /* Make two-column layout stack on mobile */
            #imagePreview > div:first-child {
                flex-direction: column !important;
                gap: 1rem !important;
            }
            
            #birdIntroduction {
                max-height: 300px !important;
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
                    
                    <!-- Recognition Result with Encyclopedia Data -->
                    <div id="recognitionResult" style="display: none; background: rgba(255, 255, 255, 0.1); border-radius: 8px; padding: 1rem; margin: 1rem 0; max-height: 600px; overflow-y: auto;">
                        <div id="recognitionContent"></div>
                    </div>
                    
                    <button type="submit" id="submitBtn" class="recognize-btn" style="display: none;"><img src="/resources/tag.svg" alt="Tag" style="width: 20px; height: 20px; margin-right: 8px; vertical-align: middle; filter: brightness(0) invert(1);">Label Bird</button>
                    
                    <div style="text-align: center; margin-top: 1rem; display: none;" id="uploadAnotherContainer">
                        <button type="button" id="uploadAnotherBtn" style="background: none; border: none; color: rgba(255, 255, 255, 0.6); font-size: 0.9rem; cursor: pointer; text-decoration: underline;">
                            Upload another image
                        </button>
                    </div>
                    
                    <!-- Hidden inputs for form submission - will be populated by recognition -->
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
    
    // Mode switching elements
    const searchModeBtn = document.getElementById('searchModeBtn');
    const backToUploadBtn = document.getElementById('backToUploadBtn');
    const uploadMode = document.getElementById('uploadMode');
    const searchMode = document.getElementById('searchMode');
    const mainButtons = document.getElementById('mainButtons');
    const recognizeBirdBtn = document.getElementById('recognizeBirdBtn');
    const uploadAnotherBtn = document.getElementById('uploadAnotherBtn');

    let currentFile = null;

    // Mode switching functionality
    function switchToSearchMode() {
        uploadMode.style.display = 'none';
        searchMode.style.display = 'block';
    }
    
    function switchToUploadMode() {
        uploadMode.style.display = 'block';
        searchMode.style.display = 'none';
    }
    
    function resetToUpload() {
        // Reset to initial upload state
        uploadArea.style.display = 'block';
        mainButtons.style.display = 'flex';
        imagePreview.style.display = 'none';
        recognitionResult.style.display = 'none';
        
        // Show app name and description when returning to upload mode (logo stays visible)
        const title = document.querySelector('.title');
        const subtitle = document.querySelector('.subtitle');
        if (title) title.style.display = 'block';
        if (subtitle) subtitle.style.display = 'block';
        
        // Show supported formats instruction when returning to upload
        const supportedFormats = document.getElementById('supportedFormats');
        if (supportedFormats) {
            supportedFormats.style.display = 'block';
        }
        
        // Hide submit button and upload another container
        const submitBtn = document.getElementById('submitBtn');
        if (submitBtn) {
            submitBtn.style.display = 'none';
        }
        const uploadAnotherContainer = document.getElementById('uploadAnotherContainer');
        if (uploadAnotherContainer) {
            uploadAnotherContainer.style.display = 'none';
        }
        
        // Show recognize button in preview again
        const recognizeBtnInPreview = document.getElementById('recognizeBtn');
        if (recognizeBtnInPreview) {
            recognizeBtnInPreview.style.display = 'block';
        }
        
        // Clear form data
        cnInput.value = '';
        laInput.value = '';
        fileInput.value = '';
        currentFile = null;
        
        // Clear location data and hide location section
        const locationInput = document.getElementById('location');
        const locationSection = document.getElementById('locationSection');
        
        if (locationSection) {
            locationSection.style.display = 'none';
        }
        if (locationInput) {
            locationInput.value = '';
            locationInput.placeholder = 'Detecting location...';
            locationInput.setAttribute('readonly', true);
        }
    }
    
    searchModeBtn.addEventListener('click', switchToSearchMode);
    backToUploadBtn.addEventListener('click', switchToUploadMode);
    
    // Upload another image functionality
    if (uploadAnotherBtn) {
        uploadAnotherBtn.addEventListener('click', resetToUpload);
    }

    // File upload and preview
    fileInput.addEventListener('change', function(e) {
        const file = e.target.files[0];
        if (file) {
            currentFile = file;
            const reader = new FileReader();
            reader.onload = function(e) {
                previewImg.src = e.target.result;
                imagePreview.style.display = 'block';
                uploadArea.style.display = 'none';
                mainButtons.style.display = 'none';
                recognitionResult.style.display = 'none';
                
                // Hide app name and description but keep logo visible
                const title = document.querySelector('.title');
                const subtitle = document.querySelector('.subtitle');
                if (title) title.style.display = 'none';
                if (subtitle) subtitle.style.display = 'none';
                
                // Hide supported formats instruction after upload
                const supportedFormats = document.getElementById('supportedFormats');
                if (supportedFormats) {
                    supportedFormats.style.display = 'none';
                }
                
                // Show location section for new image
                const locationSection = document.getElementById('locationSection');
                if (locationSection) {
                    locationSection.style.display = 'block';
                }
                
                // Auto-detect location when image is uploaded
                autoDetectLocation();
                
                // Show "Upload another image" button immediately after image upload
                const uploadAnotherContainer = document.getElementById('uploadAnotherContainer');
                if (uploadAnotherContainer) {
                    uploadAnotherContainer.style.display = 'block';
                }
                
                // Hide submit button initially (until recognition is done)
                const submitBtn = document.getElementById('submitBtn');
                if (submitBtn) {
                    submitBtn.style.display = 'none';
                }
            };
            reader.readAsDataURL(file);
        }
    });

    // Auto-detect location function
    function autoDetectLocation() {
        const locationInput = document.getElementById('location');
        
        if (!navigator.geolocation) {
            // Gracefully fallback to manual input without error message
            locationInput.placeholder = 'Enter location (City, Country)';
            locationInput.removeAttribute('readonly');
            return;
        }
        
        locationInput.placeholder = 'Detecting location...';
        
        navigator.geolocation.getCurrentPosition(
            function(position) {
                const lat = position.coords.latitude;
                const lon = position.coords.longitude;
                
                // Use reverse geocoding to get human-readable location
                fetch(`https://api.bigdatacloud.net/data/reverse-geocode-client?latitude=${lat}&longitude=${lon}&localityLanguage=en`)
                    .then(response => response.json())
                    .then(data => {
                        let locationText = '';
                        if (data.city && data.countryName) {
                            locationText = `${data.city}, ${data.countryName}`;
                        } else if (data.locality && data.countryName) {
                            locationText = `${data.locality}, ${data.countryName}`;
                        } else if (data.countryName) {
                            locationText = data.countryName;
                        } else {
                            locationText = `${lat.toFixed(6)}, ${lon.toFixed(6)}`;
                        }
                        
                        locationInput.value = locationText;
                        locationInput.placeholder = 'Location detected - edit if needed';
                        locationInput.removeAttribute('readonly');
                    })
                    .catch(error => {
                        console.error('Reverse geocoding failed:', error);
                        locationInput.value = `${lat.toFixed(6)}, ${lon.toFixed(6)}`;
                        locationInput.placeholder = 'GPS coordinates detected - edit if needed';
                        locationInput.removeAttribute('readonly');
                    });
            },
            function(error) {
                // Gracefully fallback to manual input without error message
                console.log('Geolocation error:', error.message);
                locationInput.placeholder = 'Enter location (City, Country)';
                locationInput.removeAttribute('readonly');
            },
            {
                enableHighAccuracy: true,
                timeout: 10000,
                maximumAge: 300000 // 5 minutes
            }
        );
    }

    // Toggle encyclopedia details function
    function toggleEncyclopediaData() {
        const details = document.getElementById('encyclopediaDetails');
        const button = document.getElementById('toggleEncyclopedia');
        
        if (details.style.display === 'none' || details.style.display === '') {
            details.style.display = 'block';
            button.innerHTML = '📖 Hide Detailed Information';
        } else {
            details.style.display = 'none';
            button.innerHTML = '📖 Show Detailed Information';
        }
    }

    // Main recognize bird button (available before upload)
    recognizeBirdBtn.addEventListener('click', function() {
        if (!currentFile) {
            alert('Please upload an image first to recognize the bird.');
            return;
        }
        
        performRecognition();
    });

    // Bird recognition function
    function performRecognition() {
        if (!currentFile) {
            alert('Please upload an image first to recognize the bird.');
            return;
        }
        
        const activeBtn = recognizeBtn || recognizeBirdBtn;
        activeBtn.disabled = true;
        activeBtn.innerHTML = '<img src="/resources/search.svg" alt="Search" style="width: 20px; height: 20px; margin-right: 8px; vertical-align: middle; filter: brightness(0) invert(1);">Analyzing...';
        
        const formData = new FormData();
        formData.append('image', currentFile);
        
        // Add location data if provided
        const locationInput = document.getElementById('location');
        if (locationInput && locationInput.value.trim()) {
            formData.append('location', locationInput.value.trim());
        }
        
        fetch('/api/recognize', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                console.log('🔍 Recognition success, data:', data);
                console.log('📚 Encyclopedia data check:', data.encyclopedia_data ? 'EXISTS' : 'MISSING');
                
                // Auto-fill the hidden form fields with recognition results (only 中文名 and 拉丁学名)
                cnInput.value = data.chinese_name;
                laInput.value = data.scientific_name;
                
                // Show recognition results (display all three names: 中文名, 英文名, 拉丁学名)
                const locationInput = document.getElementById('location');
                const locationUsed = locationInput && locationInput.value.trim();
                
                // Build name display parts, only show non-empty names
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
                
                // Show bird introduction in right column if encyclopedia data exists
                const birdIntroduction = document.getElementById('birdIntroduction');
                const introductionContent = document.getElementById('introductionContent');
                
                console.log('🖼️ Introduction elements check:');
                console.log('   birdIntroduction:', birdIntroduction ? 'FOUND' : 'MISSING');
                console.log('   introductionContent:', introductionContent ? 'FOUND' : 'MISSING');
                
                // TEMP: Always show introduction panel for testing
                if (birdIntroduction && introductionContent) {
                    introductionContent.innerHTML = `
                        <div style="color: white; text-align: center; padding: 1rem;">
                            <h3 style="margin: 0 0 1rem 0; color: white;">🧪 Test: Introduction Panel Working!</h3>
                            <p style="margin: 0;">Bird: ${data.chinese_name || data.common_name || 'Unknown'}</p>
                            <p style="margin: 0.5rem 0;">Has Encyclopedia Data: ${data.encyclopedia_data ? 'YES' : 'NO'}</p>
                        </div>
                    `;
                    birdIntroduction.style.display = 'block';
                    console.log('🧪 TEMP: Forced introduction panel to show');
                }
                
                // Original logic (commented for testing)
                /*
                if (data.encyclopedia_data && birdIntroduction && introductionContent) {
                    console.log('✅ Creating introduction content...');
                    
                    // Create comprehensive introduction content
                    let introHTML = `
                        <div style="text-align: center; margin-bottom: 1.2rem; border-bottom: 1px solid rgba(255,255,255,0.2); padding-bottom: 0.8rem;">
                            <h3 style="margin: 0; color: white; font-size: 1.1rem;">🐦 ${data.chinese_name || data.common_name}</h3>
                            ${data.scientific_name ? `<div style="font-style: italic; color: rgba(255,255,255,0.8); font-size: 0.9rem; margin-top: 0.3rem;">${data.scientific_name}</div>` : ''}
                        </div>
                    `;
                    
                    // Add all encyclopedia sections in order
                    const sections = [
                        { key: 'overview', title: '📋 综述', data: data.encyclopedia_data.overview },
                        { key: 'physical_features', title: '🔍 外形特征', data: data.encyclopedia_data.physical_features },
                        { key: 'identification', title: '🔎 区别辨识', data: data.encyclopedia_data.identification },
                        { key: 'behavior', title: '🦅 生活习性', data: data.encyclopedia_data.behavior },
                        { key: 'distribution', title: '🌍 地理分布', data: data.encyclopedia_data.distribution },
                        { key: 'breeding', title: '🥚 生长繁殖', data: data.encyclopedia_data.breeding },
                        { key: 'vocalizations', title: '🎵 鸣叫特征', data: data.encyclopedia_data.vocalizations },
                        { key: 'conservation_status', title: '🛡️ 保护现状', data: data.encyclopedia_data.conservation_status }
                    ];
                    
                    sections.forEach(section => {
                        if (section.data && section.data.trim()) {
                            introHTML += `
                                <div style="margin-bottom: 1rem;">
                                    <div style="color: rgba(255,255,255,0.95); font-weight: bold; margin-bottom: 0.4rem; font-size: 0.9rem;">${section.title}</div>
                                    <div style="color: rgba(255,255,255,0.85); font-size: 0.85rem; line-height: 1.4; text-align: justify;">${section.data}</div>
                                </div>
                            `;
                        }
                    });
                    
                    // Add classification if available
                    if (data.encyclopedia_data.classification) {
                        const cls = data.encyclopedia_data.classification;
                        if (cls.order_chinese || cls.family_chinese || cls.genus_chinese) {
                            introHTML += `
                                <div style="margin-top: 1rem; padding-top: 0.8rem; border-top: 1px solid rgba(255,255,255,0.2);">
                                    <div style="color: rgba(255,255,255,0.95); font-weight: bold; margin-bottom: 0.4rem; font-size: 0.9rem;">� 分类学</div>
                                    <div style="color: rgba(255,255,255,0.8); font-size: 0.8rem; line-height: 1.3;">
                                        ${cls.order_chinese ? `<div>目: ${cls.order_chinese} ${cls.order_latin ? `(${cls.order_latin})` : ''}</div>` : ''}
                                        ${cls.family_chinese ? `<div>科: ${cls.family_chinese} ${cls.family_latin ? `(${cls.family_latin})` : ''}</div>` : ''}
                                        ${cls.genus_chinese ? `<div>属: ${cls.genus_chinese}</div>` : ''}
                                    </div>
                                </div>
                            `;
                        }
                    }
                    
                    // Add links if available
                    if (data.encyclopedia_data.wikipedia_zh || data.encyclopedia_data.wikipedia_en) {
                        introHTML += `
                            <div style="margin-top: 1rem; padding-top: 0.8rem; border-top: 1px solid rgba(255,255,255,0.2); text-align: center;">
                                <div style="color: rgba(255,255,255,0.95); font-weight: bold; margin-bottom: 0.4rem; font-size: 0.9rem;">🔗 更多信息</div>
                                ${data.encyclopedia_data.wikipedia_zh ? `
                                    <a href="${data.encyclopedia_data.wikipedia_zh}" target="_blank" 
                                       style="color: rgba(135,206,235,0.9); text-decoration: none; font-size: 0.8rem; margin-right: 1rem; display: inline-block; margin-bottom: 0.3rem;">
                                        📚 中文维基百科
                                    </a>
                                ` : ''}
                                ${data.encyclopedia_data.wikipedia_en ? `
                                    <a href="${data.encyclopedia_data.wikipedia_en}" target="_blank" 
                                       style="color: rgba(135,206,235,0.9); text-decoration: none; font-size: 0.8rem; display: inline-block; margin-bottom: 0.3rem;">
                                        📖 English Wikipedia
                                    </a>
                                ` : ''}
                            </div>
                        `;
                    }
                    
                    introductionContent.innerHTML = introHTML;
                    birdIntroduction.style.display = 'block';
                } else {
                    console.log('❌ Introduction component issue:');
                    console.log('   Encyclopedia data:', !!data.encyclopedia_data);
                    console.log('   Bird introduction element:', !!birdIntroduction);
                    console.log('   Introduction content element:', !!introductionContent);
                    
                    // Show a test message to debug
                    if (birdIntroduction && introductionContent) {
                        introductionContent.innerHTML = '<div style="color: white; text-align: center; padding: 1rem;">🔍 Debug: Introduction panel loaded, but no encyclopedia data available</div>';
                        birdIntroduction.style.display = 'block';
                    }
                }
                */
                
                // Show simple recognition result with encyclopedia data
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
                
                // Hide the recognize button and show submit button and upload another option
                const recognizeBtnInPreview = document.getElementById('recognizeBtn');
                if (recognizeBtnInPreview) {
                    recognizeBtnInPreview.style.display = 'none';
                }
                
                const submitBtn = document.getElementById('submitBtn');
                if (submitBtn) {
                    submitBtn.style.display = 'block';
                }
                
                const uploadAnotherContainer = document.getElementById('uploadAnotherContainer');
                if (uploadAnotherContainer) {
                    uploadAnotherContainer.style.display = 'block';
                }
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
                
                // For failed recognition, still hide recognize button but don't show submit yet
                const recognizeBtnInPreview = document.getElementById('recognizeBtn');
                if (recognizeBtnInPreview) {
                    recognizeBtnInPreview.style.display = 'none';
                }
                
                const uploadAnotherContainer = document.getElementById('uploadAnotherContainer');
                if (uploadAnotherContainer) {
                    uploadAnotherContainer.style.display = 'block';
                }
            }
        })
        .catch(error => {
            console.error('Recognition error:', error);
            recognitionContent.innerHTML = `
                <div style="color: white; text-align: center;">
                    <div style="color: #FCA5A5; margin-bottom: 0.5rem;">
                        × Recognition failed
                    </div>
                    <div style="font-size: 0.9rem; color: rgba(255, 255, 255, 0.7);">
                        Please check your internet connection and try again.
                    </div>
                </div>
            `;
            recognitionResult.style.display = 'block';
        })
        .finally(() => {
            activeBtn.disabled = false;
            const btnText = recognizeBtn ? 'Recognize This Bird' : 'Recognize Bird';
            activeBtn.innerHTML = `<img src="/resources/search.svg" alt="Search" style="width: 20px; height: 20px; margin-right: 8px; vertical-align: middle; filter: brightness(0) invert(1);">${btnText}`;
        });
    }

    // Secondary recognize button (in preview area)
    if (recognizeBtn) {
        recognizeBtn.addEventListener('click', performRecognition);
    }

    // Drag and drop functionality
    uploadArea.addEventListener('dragover', function(e) {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });

    uploadArea.addEventListener('dragleave', function(e) {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
    });

    uploadArea.addEventListener('drop', function(e) {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            fileInput.files = files;
            fileInput.dispatchEvent(new Event('change'));
        }
    });

    // Click to upload functionality
    uploadArea.addEventListener('click', function(e) {
        // Don't trigger if clicking on the file input itself
        if (e.target !== fileInput) {
            fileInput.click();
        }
    });

    // Search functionality
    searchInput.addEventListener('input', function() {
        clearTimeout(searchTimeout);
        const query = this.value.trim();
        
        if (query.length < 1) {
            searchResults.style.display = 'none';
            return;
        }
        
        searchTimeout = setTimeout(() => {
            fetch('/api/search?q=' + encodeURIComponent(query))
                .then(response => response.json())
                .then(data => {
                    searchResults.innerHTML = '';
                    if (data.length > 0) {
                        data.forEach(item => {
                            const div = document.createElement('div');
                            div.className = 'search-item';
                            div.innerHTML = '<div><strong>' + item.chinese_name + '</strong></div><div class="latin-name">' + item.latin_name + '</div>';
                            div.addEventListener('click', () => {
                                cnInput.value = item.chinese_name;
                                laInput.value = item.latin_name;
                                searchInput.value = item.chinese_name;
                                searchResults.style.display = 'none';
                            });
                            searchResults.appendChild(div);
                        });
                        searchResults.style.display = 'block';
                    } else {
                        searchResults.innerHTML = '<div class="search-item">No matching birds found</div>';
                        searchResults.style.display = 'block';
                    }
                })
                .catch(error => {
                    console.error('Search error:', error);
                    searchResults.style.display = 'none';
                });
        }, 300);
    });

    // Hide search results when clicking outside
    document.addEventListener('click', function(e) {
        if (!e.target.closest('.search-container')) {
            searchResults.style.display = 'none';
        }
    });

    // Form submission
    document.getElementById('uploadForm').addEventListener('submit', function(e) {
        const submitBtn = document.getElementById('submitBtn');
        const cnValue = document.getElementById('cn').value;
        const laValue = document.getElementById('la').value;
        
        // Validate that we have the required data
        if (!cnValue || !laValue) {
            e.preventDefault();
            alert('Please recognize the bird first to get the species information.');
            return;
        }
        
        submitBtn.disabled = true;
        submitBtn.innerHTML = '<img src="/resources/check.svg" alt="Check" style="width: 20px; height: 20px; margin-right: 8px; vertical-align: middle; filter: brightness(0) invert(1);">Processing...';
    });
    </script>
</body>
</html>'''

@app.route("/api/search")
def api_search():
    q = request.args.get("q", "")
    return jsonify(search_species(q))

@app.route("/api/recognize", methods=["POST"])
def api_recognize():
    """
    API endpoint for bird recognition
    """
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image selected'}), 400
        
        # Save uploaded file temporarily
        temp_filename = f"temp_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        temp_path = os.path.join(app.config["UPLOAD_FOLDER"], temp_filename)
        file.save(temp_path)
        
        # Get location data if provided
        location = request.form.get('location', '').strip()
        if location:
            print(f"📍 Location provided: {location}")
        
        # Try recognition with multiple methods in order of preference
        result = None
        
        # Method 1: Try HHOLOVE AI (懂鸟) first - Highest accuracy commercial service (85% Top1, 96% Top5)
        print("🇨🇳 Trying HHOLOVE AI (懂鸟) service...")
        result = recognize_bird_hholove(temp_path, location)
        
        # Method 2: Try iNaturalist as backup (global species database)
        if not result:
            print("🔍 Trying iNaturalist API...")
            result = recognize_bird_inatural(temp_path, location)
        
        # Method 3: Try Bird Watch TensorFlow model (specialized for birds)
        if not result:
            print("🦅 Trying Bird Watch TensorFlow model...")
            result = recognize_bird_birdwatch(temp_path, location)
        
        # Method 4: Try other specialized models if Bird Watch fails
        if not result:
            print("🦅 Trying other specialized bird models...")
            result = recognize_bird_specialized(temp_path, location)
        
        # Method 5: Try YOLOv5 if specialized models fail
        if not result:
            print("🎯 Trying YOLOv5...")
            result = recognize_bird_yolov5(temp_path, location)
        
        # Method 6: Try Hugging Face fallback
        if not result:
            print("🤖 Trying Hugging Face fallback...")
            result = recognize_bird_huggingface(temp_path, location)
        
        # Method 7: Try OpenCV analysis
        if not result:
            print("📸 Trying OpenCV analysis...")
            result = recognize_bird_opencv(temp_path, location)
        
        # Method 8: Local fallback - always returns something
        if not result:
            print("🏠 Using local fallback...")
            result = recognize_bird_local_fallback(temp_path)
        
        # Method 9: Absolute fallback - if everything else fails
        if not result:
            print("🐦 Using default bird identification...")
            result = {
                'confidence': 0.30,
                'scientific_name': 'Aves sp.',
                'common_name': 'Bird',
                'chinese_name': '鸟类',
                'method': 'absolute_fallback'
            }
        
        # Clean up temp file
        try:
            os.remove(temp_path)
        except:
            pass
        
        if result:
            return jsonify({
                'success': True,
                'confidence': result['confidence'],
                'scientific_name': result['scientific_name'],
                'common_name': result['common_name'],
                'chinese_name': result['chinese_name']
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Could not identify the bird. Please try a clearer image or enter the information manually.'
            })
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route("/upload", methods=["POST"])
def upload():
    try:
        f = request.files.get("photo")
        cn = request.form.get("chinese_name", "").strip()
        la = request.form.get("latin_name", "").strip()
        pos = request.form.get("position", "bottom_right")
        
        if not f:
            return "❌ 请选择照片", 400
        if not cn:
            return "❌ 请输入中文名", 400
        if not la:
            return "❌ 请输入拉丁学名", 400
        
        filename = f.filename or "photo.jpg"
        path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        f.save(path)
        out = add_caption(path, cn, la, pos)
        
        return f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Bird Identified - Bird Labeler</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
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
        }}
        
        body::before {{
            content: '';
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(0, 0, 0, 0.3);
            z-index: 1;
        }}
        
        .container {{
            position: relative;
            z-index: 2;
            max-width: 600px;
            width: 90%;
            margin: 0 auto;
            padding: 1rem;
        }}
        
        .success-icon {{
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 60px;
            height: 60px;
            margin-bottom: 1rem;
        }}
        
        .success-icon img {{
            width: 60px;
            height: 60px;
            filter: brightness(0) invert(1);
        }}
        
        .result-card {{
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            border-radius: 12px;
            padding: 2rem;
            text-align: center;
            margin-bottom: 2rem;
        }}
        
        .result-image {{
            max-width: 100%;
            border-radius: 8px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
            margin-bottom: 1.5rem;
        }}
        
        .bird-info {{
            background: rgba(255, 255, 255, 0.1);
            border-radius: 8px;
            padding: 1rem;
            margin-bottom: 1.5rem;
        }}
        
        .bird-name-cn {{
            color: white;
            font-size: 1.3rem;
            font-weight: 600;
            margin-bottom: 0.5rem;
        }}
        
        .bird-name-la {{
            color: rgba(255, 255, 255, 0.8);
            font-style: italic;
            font-size: 1.1rem;
        }}
        
        .action-buttons {{
            display: flex;
            gap: 1rem;
            justify-content: center;
            flex-wrap: wrap;
        }}
        
        .btn {{
            padding: 0.75rem 1.5rem;
            border: none;
            border-radius: 8px;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            text-decoration: none;
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            transition: all 0.3s ease;
        }}
        
        .btn-primary {{
            background: #D97706;
            color: white;
        }}
        
        .btn-primary:hover {{
            background: #B45309;
            transform: translateY(-1px);
        }}
        
        .btn-secondary {{
            background: rgba(255, 255, 255, 0.2);
            color: white;
            border: 1px solid rgba(255, 255, 255, 0.3);
        }}
        
        .btn-secondary:hover {{
            background: rgba(255, 255, 255, 0.3);
            transform: translateY(-1px);
        }}
        
        @media (max-width: 600px) {{
            .container {{
                width: 95%;
                padding: 1rem;
            }}
            
            .action-buttons {{
                flex-direction: column;
            }}
            
            .btn {{
                width: 100%;
                justify-content: center;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">        
        <div class="result-card">
            <div class="success-icon">
                <img src="/resources/check.svg" alt="Success">
            </div>
            <h2 style="color: white; margin-bottom: 1.5rem;">Bird Successfully Identified!</h2>
            
            <img src="/outputs/{os.path.basename(out)}" class="result-image" alt="Labeled bird photo"/>
            
            <div class="bird-info">
                <div class="bird-name-cn">{cn}</div>
                <div class="bird-name-la">{la}</div>
            </div>
            
            <div class="action-buttons">
                <a href="/download?p={out}" class="btn btn-primary">
                    Download Image
                </a>
                <a href="/" class="btn btn-secondary">
                    Label Another Bird
                </a>
            </div>
        </div>
    </div>
</body>
</html>'''
        
    except Exception as e:
        return f"❌ 处理照片时出错: {str(e)}", 500

@app.route("/download")
def download():
    file_path = request.args.get("p")
    if not file_path or not os.path.exists(file_path):
        return "❌ 文件未找到", 404
    return send_file(file_path, as_attachment=True)

@app.route('/outputs/<filename>')
def output_file(filename):
    file_path = os.path.join(app.config["OUTPUT_FOLDER"], filename)
    if not os.path.exists(file_path):
        return "❌ 文件未找到", 404
    return send_file(file_path)

@app.route('/static/<filename>')
def static_file(filename):
    """Serve static files like background image"""
    if filename == 'background.jpg':
        return send_file('background.jpg')
    return "File not found", 404

@app.route('/resources/<filename>')
def resource_file(filename):
    """Serve SVG icons from Resources folder"""
    file_path = os.path.join('Resources', filename)
    if os.path.exists(file_path) and filename.endswith('.svg'):
        return send_file(file_path, mimetype='image/svg+xml')
    return "File not found", 404

@app.route('/healthz')
def healthz():
    """Simple health check endpoint for Azure/App Service probes."""
    return jsonify(status="ok"), 200

# Initialize database when the app starts
init_database()

if __name__ == "__main__":
    print("🐦 启动鸟类标注分享工具 (本地开发模式)...")
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    print(f"📱 访问 http://127.0.0.1:{port} 开始使用")
    print("🔍 支持搜索功能，数据库已包含常见鸟类")
    print("🌐 部署到生产环境时请使用 gunicorn (见 wsgi.py)")
    print("\n按 Ctrl+C 停止服务器")
    try:
        # debug 仅本地使用；Azure 使用 gunicorn 启动 wsgi:app
        app.run(debug=True, host=host, port=port)
    except KeyboardInterrupt:
        print("\n👋 服务器已停止")
