import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


class CrossEncoder(nn.Module):
    def __init__(self, model_name, dropout=0.2):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        hidden = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, 1)

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]
        cls = self.dropout(cls)
        return self.classifier(cls).squeeze(-1)

    def encode(self, input_ids, attention_mask):
        """Encode a single text and return its CLS representation (used for cosine similarity)."""
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return out.last_hidden_state[:, 0, :]


def load_model(assets_dir: str = 'assets'):

    with open(f'{assets_dir}/config.json') as f:
        config = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(f'{assets_dir}/tokenizer')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CrossEncoder(model_name=config['model_name'], dropout=0.2)
    checkpoint = torch.load(f'{assets_dir}/best_model.pt', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    return model, tokenizer, config, device


@torch.no_grad()
def predict(cv_text: str, jd_text: str,
            model, tokenizer, config, device) -> dict:

    max_len = config['max_len']
    thresh = config['threshold']

    # Cross-encoder: both texts are concatenated into a single forward pass
    enc = tokenizer(
        cv_text, jd_text,
        max_length=max_len,
        padding='max_length',
        truncation='longest_first',
        return_tensors='pt'
    )

    logit = model(
        input_ids=enc['input_ids'].to(device),
        attention_mask=enc['attention_mask'].to(device),
    )

    prob = torch.sigmoid(logit).item()
    label = int(prob >= thresh)
    confidence = abs(prob - 0.5) * 2

    # Compute cosine similarity by encoding each text independently through the backbone
    def encode_single(text):
        single_enc = tokenizer(
            text,
            max_length=max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return model.encode(
            single_enc['input_ids'].to(device),
            single_enc['attention_mask'].to(device)
        )

    u = F.normalize(encode_single(cv_text), p=2, dim=1)
    v = F.normalize(encode_single(jd_text), p=2, dim=1)
    cosine_sim = (u * v).sum().item()

    return {
        'label': label,
        'probability': round(prob, 4),
        'confidence': round(confidence, 4),
        'verdict': 'Accepted' if label == 1 else 'Rejected',
        'cosine_sim': round(cosine_sim, 4),
    }