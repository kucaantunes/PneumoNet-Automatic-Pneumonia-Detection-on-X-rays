import os
import onnx
import onnxruntime as ort
import numpy as np
from flask import Flask, request, render_template, jsonify
from PIL import Image
from transformers import GPTNeoForCausalLM, AutoTokenizer

# Initialize Flask app
app = Flask(__name__)

# Path to save uploaded images
UPLOAD_FOLDER = 'static'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the ONNX model using ONNX Runtime
onnx_model_path = 'PneumoNet.onnx'  # Path to your exported ONNX model
session = ort.InferenceSession(onnx_model_path)

# Preprocess the image
def preprocess_image(image):
    # Convert image to RGB and resize
    image = image.convert('RGB')
    image = image.resize((227, 227))  # Resize to 227x227 for AlexNet
    image = np.array(image).astype(np.float32)
    
    # Normalize the image to [0, 1]
    image = image / 255.0
    
    # Convert to NCHW format (batch_size, channels, height, width)
    image = np.transpose(image, (2, 0, 1))  # Convert to CHW
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    
    return image

# Predict pneumonia from the image
def predict_pneumonia(image):
    # Preprocess the image
    processed_image = preprocess_image(image)
    
    # Get the input name of the model
    input_name = session.get_inputs()[0].name
    
    # Run the model and get the prediction
    outputs = session.run(None, {input_name: processed_image})
    
    # Assuming the model has two classes: 0 -> Normal, 1 -> Pneumonia
    prediction = np.argmax(outputs[0], axis=1)
    
    return 'Pneumonia' if prediction == 1 else 'Normal'

# Load GPT-Neo for generating responses
chatbot_model_name = "EleutherAI/gpt-neo-1.3B"  # Using GPT-Neo 1.3B
chatbot_tokenizer = AutoTokenizer.from_pretrained(chatbot_model_name)
chatbot_model = GPTNeoForCausalLM.from_pretrained(chatbot_model_name)

def generate_medical_report(prediction):
    input_text = f"Explain in detail the medical implications of detecting {prediction} from an X-ray image."
    inputs = chatbot_tokenizer(input_text, return_tensors="pt")
    outputs = chatbot_model.generate(
        inputs["input_ids"], 
        max_length=500,  # Reduce output length for clarity
        num_return_sequences=1, 
        pad_token_id=chatbot_tokenizer.eos_token_id,
        no_repeat_ngram_size=3,  # Reduce repetition in output
        top_p=0.9,  # Adjust sampling strategy for more coherent answers
        top_k=50  # Limit candidate words during generation
    )
    report = chatbot_tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Ensuring responses are precise, removing any redundant parts
    report = " ".join(report.split()[:200])  # Limit output to 100 words for clarity
    return report

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get the uploaded image file
        file = request.files['image']
        if file:
            # Save the image in the static folder
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'uploaded_image.jpg')
            file.save(file_path)
            
            # Read the image and make prediction
            image = Image.open(file_path)
            result = predict_pneumonia(image)
            report = generate_medical_report(result)
            return render_template('result.html', result=result, report=report, image_path=file_path)
    
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        file = request.files['image']
        if file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'uploaded_image.jpg')
            file.save(file_path)
            
            image = Image.open(file_path)
            result = predict_pneumonia(image)
            report = generate_medical_report(result)
            return render_template('result.html', result=result, report=report, image_path=file_path)
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.form['message']
    
    # Check if the user is asking about pneumonia in the X-ray
    if "pneumonia" in user_input.lower() and "x-ray" in user_input.lower():
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'uploaded_image.jpg')
        
        # Ensure an image was uploaded and processed
        if os.path.exists(file_path):
            image = Image.open(file_path)
            result = predict_pneumonia(image)
            
            # Specify focused areas if pneumonia is detected
            focus_areas = "lower lobes" if result == "Pneumonia" else "no abnormalities detected"
            response = f"The X-ray indicates {result}."
        else:
            response = "No X-ray image was uploaded or analyzed. Please upload an image first."
    else:
        # General chatbot response with GPT-Neo
        inputs = chatbot_tokenizer(user_input, return_tensors="pt")
        outputs = chatbot_model.generate(
            inputs["input_ids"], 
            max_length=100,  # Shorten response length
            num_return_sequences=1, 
            pad_token_id=chatbot_tokenizer.eos_token_id, 
            no_repeat_ngram_size=3, 
            top_p=0.9, 
            top_k=50
        )
        response = chatbot_tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = " ".join(response.split()[:50])  # Limit to 50 words for conciseness

    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.getenv('PORT', 5000)))
