import datetime

from flask import Flask, jsonify, request, render_template
from flask_pymongo import PyMongo

app = Flask(__name__)
app.config["MONGO_URI"] = "mongodb://localhost:27017/myDatabase"
mongo = PyMongo(app)


@app.route('/')
def index():
    return render_template('index.html', utc_dt=datetime.datetime.utcnow())


@app.route("/")
def home_page():
    online_users = mongo.db.users.find({"online": True})
    return render_template("index.html",
                           online_users=online_users)


@app.route('/predict', methods=['POST'])
def predict():
    # 1 load image
    # 2 image -> tensor
    # 3 prediction
    # 4 return json
    if request.method == 'POST':
        return jsonify({'result': 1})


if __name__ == '__main__':
    app.run()

# https://www.mongodb.com/developer/languages/python/flask-python-mongodb/
