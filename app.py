from flask import Flask, request, jsonify
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from io import BytesIO
import base64
from torchvision import transforms
app = Flask(__name__)

class DreamFusion(nn.Module):
    def __init__(self):
        super(DreamFusion, self).__init__()
        self.text_encoder = TextEncoder()
        self.shape_decoder = ShapeDecoder()

    def forward(self, text_description):
        text_embedding = self.text_encoder(text_description)
        shape = self.shape_decoder(text_embedding)
        return shape

    def generate_3d_model(self, text_description):
        text_embedding = self.text_encoder(text_description)
        shape = self.shape_decoder(text_embedding)
        return shape.detach().numpy()

class TextEncoder(nn.Module):
    def __init__(self):
        super(TextEncoder, self).__init__()
        self.bert_model = BertModel.from_pretrained('bert-base-uncased')
        self.classification_module = nn.Linear(768, 128)

    def forward(self, text_description):
        input_ids = torch.tensor([self.bert_model.encode(text_description)])
        attention_mask = torch.tensor([[1]*len(input_ids)])
        outputs = self.bert_model(input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        text_embedding = self.classification_module(pooled_output)
        return text_embedding

class ShapeDecoder(nn.Module):
    def __init__(self):
        super(ShapeDecoder, self).__init__()
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 1024)

    def forward(self, text_embedding):
        x = torch.relu(self.fc1(text_embedding))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        shape = x.view(-1, 32, 32, 3)
        return shape

dreamfusion = DreamFusion()

@app.route('/generate', methods=['POST'])
def generate():
    data = request.get_json()
    text_description = data['textDescription']
    shape = dreamfusion.generate_3d_model(text_description)
    img = Image.fromarray(shape[0].astype('uint8'))
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return jsonify({'threeDModel': img_str})

if __name__ == '__main__':
    app.run(debug=True)