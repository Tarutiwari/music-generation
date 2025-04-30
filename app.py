from flask import Flask, render_template, request
from main import generate  # Import the generate function

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    user_input = ''
    generated_output = ''
    if request.method == 'POST':
        user_input = request.form['music_input']
        generated_output = generate(user_input)  # Call generate from main.py
    return render_template('index.html', user_input=user_input, generated_output=generated_output)

if __name__ == '__main__':
    app.run(debug=True)
