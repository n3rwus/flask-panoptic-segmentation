from application import app
from flask import render_template, request


@app.route('/')
@app.route('/index')
@app.route('/home')
def index():
    return render_template("index.html", index=True)


@app.route('/login')
def login():
    return render_template("login.html", login=True)


@app.route('/courses')
def courses():
    courses_data = [
        {
            "courseID": "1111",
            "title": "PHP 102",
            "description": "Intro to PHP",
            "credits": 3,
            "term": "Fall, Spring"
        },
        {
            "courseID": "2222",
            "title": "Java 1",
            "description": "Intro to Java Programming",
            "credits": 4,
            "term": "Spring"
        },
        {
            "courseID": "3333",
            "title": "Adv PHP 201",
            "description": "Advanced PHP Programming",
            "credits": 3,
            "term": "Fall"
        },
        {
            "courseID": "4444",
            "title": "Angular 1",
            "description": "Intro to Angular",
            "credits": 3,
            "term": "Fall, Spring"
        },
        {
            "courseID": "5555",
            "title": "Java 2",
            "description": "Advanced Java Programming",
            "credits": 4,
            "term": "Fall"
        }
    ]
    return render_template("courses.html", courseData=courses_data, courses=True)


@app.route('/register')
def register():
    return render_template("register.html", register=True)


@app.route('/about')
def about():
    return render_template("about.html", about=True)

# @app.route('/infer', method=['POST'])
# def success():
#     if request.method == 'POST':
#         print('dupa')
