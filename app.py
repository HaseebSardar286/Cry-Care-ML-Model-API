from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
from processing import process_and_predict
from pydub import AudioSegment

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Helper function to convert audio file to .wav format
def convert_to_wav(input_file_path, output_file_path):
    """Convert audio file to .wav format using pydub."""
    audio = AudioSegment.from_file(input_file_path)
    audio.export(output_file_path, format="wav")

# API Endpoint to handle predictions
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Check if the file is not already in .wav format, and convert it
        if not filename.endswith('.wav'):
            wav_file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{os.path.splitext(filename)[0]}.wav")
            convert_to_wav(file_path, wav_file_path)
            os.remove(file_path)  # Remove the original file if converted
            file_path = wav_file_path  # Update the path to the .wav file

        # Process and predict
        predictions = process_and_predict(file_path)
        os.remove(file_path)  # Clean up uploaded file after processing
        return jsonify(predictions)

# Run the server
if __name__ == '__main__':
    app.run(debug=True)
