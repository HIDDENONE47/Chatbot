from flask import Flask, request, jsonify, render_template
from chatbot import ask

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask_question():
    data = request.get_json()
    question = data.get("question", "")

    if not question:
        return jsonify({"error": "No question provided"}), 400

    answer = ask(question)
    return jsonify({"question": question, "answer": answer})

if __name__ == "__main__":
    app.run(debug=True)