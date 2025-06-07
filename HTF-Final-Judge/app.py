from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import run  # Import the necessary functions from run.py

app = Flask(__name__)
# CORS(app, resources={r"/check_phishing": {"origins": "*"}})  # Enable CORS to allow requests from the Chrome extension
CORS(app)
@app.route("/")
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/dev')
def dev():
    return render_template('dev.html')

@app.route('/check_phishing', methods=['POST'])
def check_phishing():
    if not request.is_json:
        print("Request Content-Type is not application/json")
        print("Raw data received:", request.data)  # 👈 Add this
        return jsonify({"result": "Content-Type must be application/json"}), 415
    data = request.get_json()
    domain = data.get('url')  # Get the input URL from the JSON request body
    if not domain:
        return jsonify({"result": "invalid"}), 400

    result = run.process_url_input(domain)  # Call the function from run.py directly
    print(result)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, port=2002)
