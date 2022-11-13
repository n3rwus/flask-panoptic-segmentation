from application import app
from flask import render_template, request, jsonify, redirect

from application.inference import get_prediction
from application.transform_pipeline import format_class_name


@app.route('/')
@app.route('/index')
@app.route('/home')
def index():
    return render_template("index.html", index=True)


@app.route('/login')
def login():
    return render_template("login.html", login=True)


@app.route('/courses/')
@app.route('/courses/<term>')
def courses(term='Spring 2019'):
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
    return render_template("courses.html", courseData=courses_data, courses=True, term=term)


@app.route('/register')
def register():
    return render_template("register.html", register=True)


@app.route('/about')
def about():
    return render_template("about.html", about=True)


@app.route('/enrollment')
def enrollment():
    course_id = request.args.get('courseID')
    title = request.args.get('title')
    term = request.args.get('term')
    return render_template("enrollment.html", enrollment=True, data={"id": course_id, "title": title, "term": term})


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        img_bytes = file.read()
        class_id, class_name = get_prediction(image_bytes=img_bytes)
        return jsonify({'class_id': class_id, 'class_name': class_name})


@app.route('/tmp', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files.get('file')
        if not file:
            return
        img_bytes = file.read()
        class_id, class_name = get_prediction(image_bytes=img_bytes)
        class_name = format_class_name(class_name)
        return render_template('result.html', class_id=class_id,
                               class_name=class_name)
    return render_template('tmp.html')