from flask import Flask, render_template, request
from chakma_model import text_to_text_nmt_output_chakma as chakma_translate
from chatgaiya_model import text_to_text_nmt_output_chatgaiya as chatgaiya_translate

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html', output_text="")

@app.route('/translate', methods=['POST'])
def translate():
    input_text = request.form.get('input_text','')
    translation_type = request.args.get('translation_type', '')
    output_text = translation_type

    if translation_type == 'chakma':
        output_text = chakma_translate.make_translation(input_text)
    elif translation_type == 'chatgaiya':
        output_text = chatgaiya_translate.make_translation(input_text)
    else:
        # Handle any other translation type or error scenario
        return "Invalid translation type"

    return render_template('test.html', input_text=input_text, translation_type=translation_type, output_text=output_text)
if __name__ == '__main__':
    app.run(debug=True)
