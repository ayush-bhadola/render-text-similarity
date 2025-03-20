from flask import Flask, request, jsonify, render_template
from sentence_transformers import SentenceTransformer, util

# Load the trained model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")  # Returns the HTML form for users

@app.route("/predict", methods=["POST"])
def predict():
    # Check if request is JSON or form-data
    if request.is_json:  
        data = request.get_json()
        text1 = data.get("text1", "").strip()
        text2 = data.get("text2", "").strip()
        
        if not text1 or not text2:
            return jsonify({"error": "Invalid input, expected JSON with 'text1' and 'text2'"}), 400

        # Compute similarity
        embeddings = model.encode([text1, text2], convert_to_tensor=True, normalize_embeddings=True)
        similarity = (util.cos_sim(embeddings[0], embeddings[1]).item() + 1) / 2  # Normalize to [0,1]

        return jsonify({"similarity score": round(similarity, 4)})

    else:
        # Handle form submission (Web UI)
        text1 = request.form.get("text1", "").strip()
        text2 = request.form.get("text2", "").strip()

        if not text1 or not text2:
            return render_template("index.html", prediction_text="Error: Invalid input")

        embeddings = model.encode([text1, text2], convert_to_tensor=True, normalize_embeddings=True)
        similarity = (util.cos_sim(embeddings[0], embeddings[1]).item() + 1) / 2

        return render_template("index.html", prediction_text="Similarity Score: {:.4f}".format(similarity))

if __name__ == "__main__":
    app.run(debug=True)
