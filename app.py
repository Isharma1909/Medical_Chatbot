from flask import Flask, request, jsonify, render_template
from main import ask_question

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():

    user_message = request.json["message"]

    response = ask_question(user_message)

    return jsonify({
        "response": response
    })


if __name__ == "__main__":
    app.run(debug=True)