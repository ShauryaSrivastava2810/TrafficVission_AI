from flask import Flask, request, jsonify
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"result": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"result": "No selected file"}), 400
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Call the Tkinter function
        MEDIA = request.form.get('media')
        threading.Thread(target=_body_, args=(MEDIA, file_path)).start()

        return jsonify({"result": "File uploaded successfully"}), 200


if __name__ == '__main__':
    app.run(debug=True)
