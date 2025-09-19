import sys, os
from flask import Flask
from app.api import init_routes
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

load_dotenv()

app = Flask(__name__)
init_routes(app)

@app.route("/home")
def root():
    return {"message": "Backend server is running 🚀"}

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)