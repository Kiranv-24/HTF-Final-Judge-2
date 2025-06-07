from flask_cors import CORS
import flask
app = Flask(__name__)
CORS(app, resources={r"/check_phishing": {"origins": "*"}})