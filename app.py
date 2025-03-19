from flask import Flask, request, jsonify, render_template
from sentence_transformers import SentenceTransformer, util

# Load the trained model
model_path = "sentence-transformers/all-MiniLM-L6-v2"
model = SentenceTransformer(model_path, device='cpu')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract data from form
    text1 = request.form.get("text1", "").strip()
    text2 = request.form.get("text2", "").strip()

    # Validate inputs
    if not text1 or not text2:
        return render_template('index.html', prediction_text='Error: Invalid input')

    # Compute embeddings and similarity
    embeddings = model.encode([text1, text2], convert_to_tensor=True, normalize_embeddings=True)
    similarity = (util.cos_sim(embeddings[0], embeddings[1]).item() + 1) / 2  # Convert [-1,1] → [0,1]

    return render_template('index.html', prediction_text='Similarity Score: {:.4f}'.format(similarity))

if __name__ == "__main__":
    app.run(debug=True)




# import os
# from flask import Flask, request, jsonify, render_template
# from sentence_transformers import SentenceTransformer, util

# app = Flask(__name__)

# MODEL_PATH = "model"

# if not os.path.exists(MODEL_PATH):
#     model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
#     model.save(MODEL_PATH)
# else:
#     model = SentenceTransformer(MODEL_PATH)

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route("/health", methods=["GET"])
# def health_check():
#     return jsonify({"status": "API is running"})

# @app.route("/compare", methods=["POST"])
# def compute_similarity():
#     data = request.form
#     if not data or "text1" not in data or "text2" not in data:
#         return render_template('index.html', prediction_text='Error: Invalid input')
    
#     embeddings = model.encode([data["text1"], data["text2"]], convert_to_tensor=True)
#     similarity = util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()
    
#     return render_template('index.html', prediction_text='Similarity Score: {:.4f}'.format(similarity))

# if __name__ == "__main__":
#     app.run(debug=True)