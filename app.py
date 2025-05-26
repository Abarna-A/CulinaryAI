from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import google.generativeai as genai
from diffusers import StableDiffusionPipeline
import torch
import os
from dotenv import load_dotenv
from datetime import datetime
import uuid
import logging
import base64
from PIL import Image
import io
from concurrent.futures import ThreadPoolExecutor, TimeoutError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize Flask with CORS
app = Flask(__name__, static_url_path='/static')
CORS(app, resources={r"/*": {"origins": "*"}})

# Configure Gemini
try:
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    gemini_model = genai.GenerativeModel('gemini-1.5-flash')
    vision_model = genai.GenerativeModel('gemini-1.5-flash')
    logger.info("Gemini AI initialized successfully")
except Exception as e:
    logger.error(f"Gemini init failed: {str(e)}")
    gemini_model = None
    vision_model = None

# Initialize Stable Diffusion with caching
sd_pipe = None
try:
    sd_pipe = StableDiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-1-base",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        cache_dir="model_cache"
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sd_pipe = sd_pipe.to(device)
    logger.info(f"Stable Diffusion loaded on {device}")
except Exception as e:
    logger.error(f"Stable Diffusion init failed: {str(e)}")
    sd_pipe = None

# Create directories
os.makedirs('static/uploads', exist_ok=True)
os.makedirs('static/gallery', exist_ok=True)
os.makedirs('static/placeholder', exist_ok=True)
os.makedirs('model_cache', exist_ok=True)

import re

def clean_gemini_response(response_text):
    """
    Cleans and formats Gemini API response text for proper display in frontend.
    Handles **bold** formatting and other markdown-like elements.
    
    Args:
        response_text (str): Raw text response from Gemini API
        
    Returns:
        str: Cleaned text with proper HTML formatting
    """
    if not response_text:
        return ""
    
    # Replace **bold** with <strong> tags
    cleaned_text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', response_text)
    
    # Replace *italics* with <em> tags (if needed)
    cleaned_text = re.sub(r'\*(.*?)\*', r'<em>\1</em>', cleaned_text)
    
    # Replace numbered lists with proper HTML
    cleaned_text = re.sub(r'^\d+\.\s+(.*)$', r'<li>\1</li>', cleaned_text, flags=re.MULTILINE)
    
    # Replace bullet points with proper HTML
    cleaned_text = re.sub(r'^-\s+(.*)$', r'<li>\1</li>', cleaned_text, flags=re.MULTILINE)
    
    # Replace multiple newlines with single <br>
    cleaned_text = re.sub(r'\n{2,}', '<br><br>', cleaned_text)
    
    # Replace single newlines with <br> (except before/after list items)
    cleaned_text = re.sub(r'(?<!</li>)\n(?!<li>)', '<br>', cleaned_text)
    
    # Wrap lists in <ul> or <ol> tags
    if '<li>' in cleaned_text:
        if re.search(r'^\d+\.', response_text, flags=re.MULTILINE):
            cleaned_text = f'<ol>{cleaned_text}</ol>'
        else:
            cleaned_text = f'<ul>{cleaned_text}</ul>'
    
    return cleaned_text

def generate_recipe(ingredients):
    if not gemini_model:
        raise Exception("AI service unavailable")
    
    try:
        prompt = f"""Generate a detailed recipe using ONLY: {ingredients}
        Format as:
        1. Name: [Name]
        2. Ingredients: [List with quantities]
        3. Steps: [Numbered instructions]
        4. Time: [Total cooking time]
        5. Difficulty: [Easy/Medium/Hard]
        6. Chef's tips: [Optional tips]"""
        
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        logger.error(f"Recipe generation failed: {str(e)}")
        raise Exception("Failed to generate recipe")

def generate_image_with_timeout(recipe_name, timeout=3000):
    if not sd_pipe:
        return None
        
    def generate():
        try:
            filename = f"recipe_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}.png"
            image_path = f"static/gallery/{filename}"
            
            prompt = f"Professional food photography of {recipe_name}, 8K resolution, natural lighting, on marble countertop"
            image = sd_pipe(prompt, num_inference_steps=15).images[0]
            image.save(image_path)
            
            return f"/static/gallery/{filename}"
        except Exception as e:
            logger.error(f"Image generation failed: {str(e)}")
            return "/static/placeholder/recipe_placeholder.png"

    with ThreadPoolExecutor() as executor:
        future = executor.submit(generate)
        try:
            return future.result(timeout=timeout)
        except TimeoutError:
            logger.error("Image generation timed out")
            return "/static/placeholder/recipe_placeholder.png"

def analyze_food_image(image_data):
    if not vision_model:
        raise Exception("Vision AI service unavailable")
    
    try:
        if isinstance(image_data, str):
            image_data = base64.b64decode(image_data.split(',')[1])
        img = Image.open(io.BytesIO(image_data))
        
        prompt = """Analyze this food image and provide a detailed recipe:
        1. First identify the dish name
        2. List all visible ingredients with estimated quantities
        3. Provide step-by-step cooking instructions
        4. Estimate cooking time and difficulty level
        5. Offer chef's tips for best results
        
        Format your response exactly like this:
        - Dish Name: [name of the dish]
        - Ingredients: 
          - [ingredient 1]
          - [ingredient 2]
          - [etc...]
        - Instructions:
          1. [step 1]
          2. [step 2]
          3. [etc...]
        - Cooking Time: [time estimate]
        - Difficulty: [Easy/Medium/Hard]
        - Chef's Tips: [helpful tips]"""
        
        response = vision_model.generate_content([prompt, img])
        return response.text
    except Exception as e:
        logger.error(f"Image analysis failed: {str(e)}")
        raise Exception("Failed to analyze image")

@app.route('/upload', methods=['POST'])
def upload_image():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
            
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400
            
        # Read file directly without saving first
        image_data = file.read()
        
        analysis = analyze_food_image(image_data)
        
        # Parse the analysis into structured data
        recipe = {}
        sections = ['Dish Name', 'Ingredients', 'Instructions', 'Cooking Time', 'Difficulty', 'Chef\'s Tips']
        current_section = None
        
        for line in analysis.split('\n'):
            line = line.strip()
            if not line:
                continue
                
            for section in sections:
                if line.startswith(f"- {section}:"):
                    current_section = section
                    recipe[current_section] = line.replace(f"- {section}:", "").strip()
                    break
            else:
                if current_section and current_section in recipe:
                    recipe[current_section] += "\n" + line.replace("- ", "").strip()
        
        # Generate image of the dish
        recipe_name = recipe.get('Dish Name', 'Unknown Dish')
        image_url = generate_image_with_timeout(recipe_name) if sd_pipe else None
        
        return jsonify({
            "status": "success",
            "image_url": image_url,
            "recipe": recipe
        })
        
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        message = data.get("message", "").strip()
        ingredients = data.get("ingredients", "").strip()
        recipe = data.get("recipe", "").strip()
        
        if not message:
            return jsonify({"error": "Message is required"}), 400
        
        if not gemini_model:
            raise Exception("AI service unavailable")
        
        prompt = f"""
        You are a helpful cooking assistant. The user is working with these ingredients: {ingredients}
        
        Current recipe context: {recipe}
        
        User message: {message}
        
        Please respond helpfully, considering:
        1. If they ask for alternatives (like no oven), suggest practical substitutions
        2. If they ask for improvements, suggest additional ingredients that would enhance the dish
        3. Keep responses concise but informative (1-2 paragraphs max)
        4. For equipment alternatives, suggest common household solutions
        5. If they ask for future recommendations, suggest 1-2 key ingredients to add
        6. Format responses with clear headings and bullet points when appropriate
        """
        
        response = gemini_model.generate_content(prompt)
        cleaned_response = clean_gemini_response(response.text)
        
        return jsonify({
            "response": cleaned_response,
            "status": "success"
        })
        
    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        return jsonify({
            "error": str(e),
            "status": "error"
        }), 500

@app.route('/gallery')
def get_gallery():
    try:
        images = []
        for file in sorted(os.listdir('static/gallery'), reverse=True):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                images.append({
                    'url': f'/static/gallery/{file}',
                    'name': ' '.join(file.split('_')[1:-1])
                })
        return jsonify({'images': images[:10]})
    except Exception as e:
        logger.error(f"Gallery error: {str(e)}")
        return jsonify({'images': []})

@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)

@app.route("/ingredients", methods=["GET", "POST"])
def ingredients_to_recipe():
    if request.method == "POST":
        try:
            ingredients = request.form.get("ingredients", "").strip()
            if not ingredients:
                return jsonify({"error": "Please enter ingredients"}), 400
            
            recipe = generate_recipe(ingredients)
            recipe_name = recipe.split("\n")[0].replace("1. Name: ", "")
            image_url = generate_image_with_timeout(recipe_name)
            
            return jsonify({
                "recipe": recipe,
                "image_url": image_url,
                "status": "success"
            })
            
        except Exception as e:
            logger.error(f"Error: {str(e)}")
            return jsonify({
                "error": str(e),
                "status": "error"
            }), 500
    
    return render_template("ingredients.html")

@app.route("/image", methods=["GET"])
def image_to_recipe():
    return render_template("image.html")

@app.route("/")
def home():
    return render_template("home.html")

@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    return response

if __name__ == "__main__":
    if not os.path.exists("static/placeholder/recipe_placeholder.png"):
        from PIL import Image, ImageDraw
        img = Image.new('RGB', (512, 512), color=(220, 220, 220))
        d = ImageDraw.Draw(img)
        d.text((100, 256), "Recipe Image\n(Placeholder)", fill=(100, 100, 100))
        img.save("static/placeholder/recipe_placeholder.png")
    
    app.run(host="0.0.0.0", port=5000, debug=True, threaded=True)