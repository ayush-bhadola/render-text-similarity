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
    try:
        data = request.get_json()
        text1 = data.get("text1", "").strip()
        text2 = data.get("text2", "").strip()

        if not text1 or not text2:
            return jsonify({"error": "Invalid input"}), 400

        embeddings = model.encode([text1, text2], convert_to_tensor=True, normalize_embeddings=True)
        similarity = (util.cos_sim(embeddings[0], embeddings[1]).item() + 1) / 2  # Convert [-1,1] → [0,1]

        return jsonify({"similarity score": round(similarity, 4)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
