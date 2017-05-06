from flask import Flask, url_for, render_template, request, flash, redirect, jsonify #, session
from werkzeug.utils import secure_filename
import os

import threading
import subprocess
import uuid

app =Flask(__name__)

UPLOAD_FOLDER = os.path.join(app.static_folder, 'content')
# STYLE_FOLDER = os.path.join(app.static_folder, 'style')
CHECKPOINT_FOLDER = os.path.join(app.static_folder, 'models')
RESULT_FOLDER = os.path.join(app.static_folder, 'results')
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

SRC_FOLDER = 'src'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# app.config['STYLE_FOLDER'] = STYLE_FOLDER

background_scripts = {}

def run_evaluation(id, filename):
    print 'calling subprocess'
    subprocess.call(['python', SRC_FOLDER+'/test.flask.py',
        '--checkpoint', CHECKPOINT_FOLDER,
        '--in-path', os.path.join(UPLOAD_FOLDER, filename),
        '--out-path', RESULT_FOLDER])
    background_scripts[id] = True

# manager = Manager(app)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        _file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if _file.filename == '':
            flash('No file selected')
            return redirect(request.url)
        if _file and allowed_file(_file.filename):
            filename = secure_filename(_file.filename)
            _file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        # print 'filename:', filename
        return redirect(url_for('evaluate', filename=filename))
    else:
        return render_template('index.html')

@app.route('/evaluate/<filename>')
def evaluate(filename):
    id = str(uuid.uuid4())
    background_scripts[id] = False
    threading.Thread(target=lambda: run_evaluation(id, filename)).start()
    return render_template('processing.html', id=id, filename=filename)

@app.route('/get_result')
def get_result():
    id = request.args.get('id', None)
    if id not in background_scripts:
        abort(404)
    return jsonify(done=background_scripts[id])

@app.route('/show_result/<filename>')
def show_result(filename):
    # new_name = filename.split('.')[0]+'_output.'+filename.split('.')[1]
    # os.rename(os.path.join(RESULT_FOLDER, filename), os.path.join(RESULT_FOLDER, new_name))
    return '<img src="' + url_for('static', filename='results/'+filename) + '" />'

if __name__ == '__main__':
    app.run()
