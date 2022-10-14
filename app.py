from flask import Flask, jsonify, request

app = Flask(__name__)


@app.route('/')
def hello_world():  # put application's code here
    return 'Hello World!'


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        return jsonify({'test': 'test_result'})


if __name__ == '__main__':
    app.run()
