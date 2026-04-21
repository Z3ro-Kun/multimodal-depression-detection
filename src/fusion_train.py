import os
import random
import torch
import torch.nn as nn
import torchaudio
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModel,
    Wav2Vec2Model,
    Wav2Vec2Processor
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======================
# PATHS (YOUR DIRECTORIES)
# ======================
BASE_AUDIO = "D:/daic_audio_subset"

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

CSV_PATH = os.path.join(BASE_DIR, "data", "processed", "daic_text_clean.csv")

TEXT_MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "hierarchical_text",
    "best_model.pt"
)

FUSION_SAVE_PATH = os.path.join(
    BASE_DIR,
    "models",
    "fusion_model2",
    "fusion_best_model.pt"
)

os.makedirs(os.path.dirname(FUSION_SAVE_PATH), exist_ok=True)

# ======================
# LOAD DATA
# ======================
df = pd.read_csv(CSV_PATH)
train_df = df[df["split"] == "train"].reset_index(drop=True)
dev_df = df[df["split"] == "dev"].reset_index(drop=True)

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.array([0, 1]),
    y=train_df["label"]
)

class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

# ======================
# TEXT MODEL (FROZEN)
# ======================
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

class HierarchicalTextModel(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.attention = nn.Linear(hidden_size, 1)
        self.classifier = nn.Linear(hidden_size, 2)

    def forward(self, input_ids, attention_mask):
        batch_size, num_chunks, seq_len = input_ids.shape

        input_ids = input_ids.view(-1, seq_len)
        attention_mask = attention_mask.view(-1, seq_len)

        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        cls_embeddings = outputs.last_hidden_state[:, 0, :]
        cls_embeddings = cls_embeddings.view(batch_size, num_chunks, -1)

        attn_scores = self.attention(cls_embeddings).squeeze(-1)
        attn_weights = torch.softmax(attn_scores, dim=1).unsqueeze(-1)

        weighted_sum = torch.sum(cls_embeddings * attn_weights, dim=1)
        logits = self.classifier(weighted_sum)

        return weighted_sum, logits  # important

text_model = HierarchicalTextModel(model_name).to(device)
text_model.load_state_dict(torch.load(TEXT_MODEL_PATH, map_location=device))
text_model.eval()

for param in text_model.parameters():
    param.requires_grad = False

# ======================
# AUDIO MODEL (FROZEN)
# ======================
audio_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
audio_encoder = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base").to(device)
audio_encoder.eval()

for param in audio_encoder.parameters():
    param.requires_grad = False

# ======================
# DATASET
# ======================
class FusionDataset(Dataset):
    def __init__(self, df, segment_seconds=20, max_length=512, stride=256):
        self.df = df
        self.segment_seconds = segment_seconds
        self.max_length = max_length
        self.stride = stride

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        pid = str(int(row["participant_id"]))
        label = int(row["label"])
        text = row["text"]

        # -------- TEXT --------
        encoding = tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            stride=self.stride,
            return_overflowing_tokens=True,
            return_attention_mask=True,
            padding="max_length"
        )

        chunks = []
        for i in range(len(encoding["input_ids"])):
            chunks.append((
                torch.tensor(encoding["input_ids"][i]),
                torch.tensor(encoding["attention_mask"][i])
            ))

        input_ids = torch.stack([c[0] for c in chunks])
        attention_mask = torch.stack([c[1] for c in chunks])

        # -------- AUDIO (3 x 20s) --------
        wav_path = os.path.join(
            BASE_AUDIO,
            f"{pid}_P",
            f"{pid}_AUDIO.wav"
        )

        waveform, sr = torchaudio.load(wav_path)
        waveform = waveform.mean(dim=0)

        segment_length = sr * self.segment_seconds
        segments = []

        for _ in range(3):
            if waveform.shape[0] <= segment_length:
                segment = waveform
            else:
                start = random.randint(0, waveform.shape[0] - segment_length)
                segment = waveform[start:start + segment_length]

            segments.append(segment)

        return input_ids, attention_mask, segments, torch.tensor(label)

train_loader = DataLoader(FusionDataset(train_df), batch_size=1, shuffle=True)
dev_loader = DataLoader(FusionDataset(dev_df), batch_size=1, shuffle=False)

# ======================
# FUSION MODEL
# ======================
class FusionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1536, 256)
        self.ln1 = nn.LayerNorm(256)   # ← changed (was BatchNorm)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, 2)

    def forward(self, text_emb, audio_emb):
        x = torch.cat([text_emb, audio_emb], dim=1)
        x = self.fc1(x)
        x = self.ln1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        return self.fc2(x)
fusion_model = FusionModel().to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.AdamW(fusion_model.parameters(), lr=1e-4)

# ======================
# EVALUATION
# ======================
def evaluate():
    fusion_model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for input_ids, attention_mask, segments, labels in dev_loader:

            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            text_emb, _ = text_model(input_ids, attention_mask)

            audio_embs = []
            for segment in segments:
                audio_inputs = audio_processor(
                    segment.squeeze(0),
                    sampling_rate=16000,
                    return_tensors="pt",
                    padding=True
                ).input_values.to(device)

                audio_outputs = audio_encoder(audio_inputs)
                emb = audio_outputs.last_hidden_state.mean(dim=1)
                audio_embs.append(emb)

            audio_emb = torch.mean(torch.stack(audio_embs), dim=0)

            logits = fusion_model(text_emb, audio_emb)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return accuracy_score(all_labels, all_preds), \
           f1_score(all_labels, all_preds, average="macro")

# ======================
# TRAIN
# ======================
best_f1 = 0

for epoch in range(6):
    fusion_model.train()
    total_loss = 0

    for input_ids, attention_mask, segments, labels in tqdm(train_loader):

        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            text_emb, _ = text_model(input_ids, attention_mask)

            audio_embs = []
            for segment in segments:
                audio_inputs = audio_processor(
                    segment.squeeze(0),
                    sampling_rate=16000,
                    return_tensors="pt",
                    padding=True
                ).input_values.to(device)

                audio_outputs = audio_encoder(audio_inputs)
                emb = audio_outputs.last_hidden_state.mean(dim=1)
                audio_embs.append(emb)

            audio_emb = torch.mean(torch.stack(audio_embs), dim=0)

        optimizer.zero_grad()
        logits = fusion_model(text_emb, audio_emb)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    acc, f1 = evaluate()

    print(f"\nEpoch {epoch+1}")
    print("Train Loss:", avg_loss)
    print("Val Accuracy:", acc)
    print("Val Macro F1:", f1)

    if f1 > best_f1:
        best_f1 = f1
        torch.save(fusion_model.state_dict(), FUSION_SAVE_PATH)
        print("Saved new best fusion model.")

print("Best Fusion Macro F1:", best_f1)
