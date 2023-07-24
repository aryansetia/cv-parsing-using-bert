import io
import argparse
from flask_cors import CORS
import torch
from transformers import BertTokenizerFast, BertForTokenClassification
from flask import Flask, jsonify, render_template, request
from server.utils import preprocess_data, predict, idx2tag

app = Flask(__name__)
CORS(app)
app.config['JSON_SORT_KEYS'] = False

MAX_LEN = 500
NUM_LABELS = 12
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = 'bert-base-uncased'
STATE_DICT = torch.load("model-state.bin", map_location=DEVICE)
TOKENIZER = BertTokenizerFast("./vocab/vocab.txt", lowercase=True)

model = BertForTokenClassification.from_pretrained(
    'bert-base-uncased', state_dict=STATE_DICT['model_state_dict'], num_labels=NUM_LABELS)
model.to(DEVICE)


@app.route('/predict', methods=['GET','POST'])
def predict_api():

    if request.method == 'POST':
        # print(request.files.to_dict())    
        data = io.BytesIO(request.files.get('filename').read())
        resume_text = preprocess_data(data)
        entities = predict(model, TOKENIZER, idx2tag,
                           DEVICE, resume_text, MAX_LEN)
        return jsonify({'entities': entities})
        # return render_template('main.html', entities=entities)
    else:
        return render_template('main.html')

        


if __name__ == '__main__':
    app.run(host='0.0.0.0',port=5000)
