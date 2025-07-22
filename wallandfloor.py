from flask import Flask, request, jsonify, send_file, send_from_directory
import cv2
import numpy as np
import os
from segment_anything import SamPredictor, sam_model_registry
from flask_cors import CORS
import base64

app = Flask(__name__, static_folder='static')
CORS(app)

# Initialize SAM model
model_type = "vit_h"
checkpoint_path = "C:\\Khizar\\Visual Room Replacement\\model\\sam_vit_h_4b8939.pth"
sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
predictor = SamPredictor(sam)

TEXTURE_FOLDER = 'static/textures'
app.config['TEXTURE_FOLDER'] = TEXTURE_FOLDER

@app.route('/')
def index():
    return send_file('index.html')

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)

def generate_wall_mask(image):
    h, w = image.shape[:2]
    input_points = np.array([[w//2, h//2], [0, 0], [w-1, 0], [0, h-1], [w-1, h-1]])
    input_labels = np.array([1, 0, 0, 0, 0])

    masks, scores, _ = predictor.predict(
        point_coords=input_points,
        point_labels=input_labels,
        multimask_output=True
    )

    best_mask_index = np.argmax(scores)
    return masks[best_mask_index]

def generate_floor_mask(image):
    h, w = image.shape[:2]
    # Use multiple points at the bottom of the image
    input_points = np.array([[w//4, h-10], [w//2, h-10], [3*w//4, h-10]])
    input_labels = np.array([1, 1, 1])

    masks, scores, _ = predictor.predict(
        point_coords=input_points,
        point_labels=input_labels,
        multimask_output=True
    )

    best_mask_index = np.argmax(scores)
    floor_mask = masks[best_mask_index]
    
    # Post-process the mask to improve floor coverage
    kernel = np.ones((5,5), np.uint8)
    floor_mask = cv2.dilate(floor_mask.astype(np.uint8), kernel, iterations=2)
    
    return floor_mask

@app.route('/customize_room', methods=['POST'])
def customize_room():
    # Get image, wall color, and floor texture from request
    image_file = request.files['image']
    wall_color = request.form['wallColor']
    floor_texture = request.form['floorTexture']

    # Read image
    image_data = image_file.read()
    nparr = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Set image for predictor
    predictor.set_image(image)

    # Step 1: Generate wall mask and apply wall color
    wall_mask = generate_wall_mask(image)
    h, w = image.shape[:2]
    wall_color_rgb = tuple(int(wall_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
    wall_color_image = np.full((h, w, 3), wall_color_rgb, dtype=np.uint8)
    
    result = image.copy()
    result[wall_mask] = wall_color_image[wall_mask]

    # Step 2: Generate floor mask and apply floor texture
    floor_mask = generate_floor_mask(image)
    floor_texture_path = os.path.join(app.config['TEXTURE_FOLDER'], floor_texture)
    floor_texture_image = cv2.imread(floor_texture_path)
    floor_texture_image = cv2.cvtColor(floor_texture_image, cv2.COLOR_BGR2RGB)
    floor_texture_image = cv2.resize(floor_texture_image, (w, h))
    
    # Ensure floor_mask is boolean
    floor_mask = floor_mask.astype(bool)
    
    # Apply floor texture
    result[floor_mask] = floor_texture_image[floor_mask]

    # Convert result to base64
    _, buffer = cv2.imencode('.png', cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
    result_base64 = base64.b64encode(buffer).decode('utf-8')

    return jsonify({'message': 'Room customized successfully', 'result': result_base64})

@app.route('/textures')
def get_textures():
    textures = os.listdir(app.config['TEXTURE_FOLDER'])
    return jsonify(textures)

if __name__ == '__main__':
    app.run(debug=True)