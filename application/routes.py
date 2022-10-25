from application import app
from flask import render_template, request


@app.route('/')
@app.route('/index')
@app.route('/home')
def index():
    return render_template("index.html", login=False)


@app.route('/login')
def login():
    return render_template("login.html", login=True)


@app.route('/courses')
def courses():
    return render_template("courses.html", login=True)


@app.route('/register')
def register():
    return render_template("register.html", login=True)


@app.route('/infer', method=['POST'])
def success():
    if request.method == 'POST':
        print('dupa')
