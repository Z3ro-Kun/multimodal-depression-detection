import os
import torch
import torchaudio
import numpy as np
import pandas as pd
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModel,
    Wav2Vec2Model,
    Wav2Vec2Processor
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve
)
import matplotlib.pyplot as plt
import seaborn as sns

# ======================
# DEVICE
# ======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======================
# PATHS
# ======================
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
BASE_AUDIO = "D:/daic_audio_subset"
CSV_PATH = os.path.join(BASE_DIR, "data", "processed", "daic_text_clean.csv")
TEXT_MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "hierarchical_text",
    "best_model.pt"
)
FUSION_MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "fusion_model2",
    "fusion_best_model.pt"
)

# ======================
# LOAD DATA
# ======================
df = pd.read_csv(CSV_PATH)
dev_df = df[df["split"] == "dev"].reset_index(drop=True)

# ======================
# TEXT MODEL
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

        return weighted_sum, logits

text_model = HierarchicalTextModel(model_name).to(device)
text_model.load_state_dict(torch.load(TEXT_MODEL_PATH, map_location=device))
text_model.eval()

for p in text_model.parameters():
    p.requires_grad = False

# ======================
# AUDIO MODEL
# ======================
audio_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
audio_encoder = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base").to(device)
audio_encoder.eval()

for p in audio_encoder.parameters():
    p.requires_grad = False

# ======================
# DATASET
# ======================
class FusionDataset(Dataset):
    def __init__(self, df, segment_seconds=20):
        self.df = df
        self.segment_seconds = segment_seconds

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        pid = str(int(row["participant_id"]))
        label = int(row["label"])
        text = row["text"]

        encoding = tokenizer(
            text,
            truncation=True,
            max_length=512,
            stride=256,
            return_overflowing_tokens=True,
            return_attention_mask=True,
            padding="max_length"
        )

        input_ids = torch.stack([
            torch.tensor(ids) for ids in encoding["input_ids"]
        ])
        attention_mask = torch.stack([
            torch.tensor(mask) for mask in encoding["attention_mask"]
        ])

        wav_path = os.path.join(
            BASE_AUDIO,
            f"{pid}_P",
            f"{pid}_AUDIO.wav"
        )

        waveform, sr = torchaudio.load(wav_path)
        waveform = waveform.mean(dim=0)

        segment_length = sr * 20
        segments = []

        for _ in range(3):
            if waveform.shape[0] <= segment_length:
                segment = waveform
            else:
                start = np.random.randint(0, waveform.shape[0] - segment_length)
                segment = waveform[start:start + segment_length]
            segments.append(segment)

        return input_ids, attention_mask, segments, torch.tensor(label)

dev_loader = DataLoader(FusionDataset(dev_df), batch_size=1, shuffle=False)

# ======================
# FUSION MODEL
# ======================
class FusionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1536, 256)
        self.ln1 = nn.LayerNorm(256)
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
fusion_model.load_state_dict(torch.load(FUSION_MODEL_PATH, map_location=device))
fusion_model.eval()

# ======================
# EVALUATION
# ======================
all_preds = []
all_labels = []
all_probs = []

with torch.no_grad():
    for input_ids, attention_mask, segments, labels in dev_loader:

        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

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
        probs = torch.softmax(logits, dim=1)

        preds = torch.argmax(probs, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probs.extend(probs[:,1].cpu().numpy())

# ======================
# METRICS
# ======================
acc = accuracy_score(all_labels, all_preds)
macro_f1 = f1_score(all_labels, all_preds, average="macro")

print("\nAccuracy:", round(acc,4))
print("Macro F1:", round(macro_f1,4))
print("\nClassification Report:")
print(classification_report(all_labels, all_preds, digits=4))

# Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(5,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()

# ROC
fpr, tpr, _ = roc_curve(all_labels, all_probs)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
plt.plot([0,1], [0,1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

# PR Curve
precision, recall, _ = precision_recall_curve(all_labels, all_probs)
pr_auc = auc(recall, precision)

plt.figure()
plt.plot(recall, precision, label=f"AUC = {pr_auc:.3f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.show()

print("\nROC AUC:", round(roc_auc,4))
print("PR AUC:", round(pr_auc,4))


import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
import matplotlib.pyplot as plt

thresholds = np.arange(0.1, 0.91, 0.02)

macro_f1s = []
depressed_recalls = []
depressed_precisions = []

for t in thresholds:
    preds = (np.array(all_probs) >= t).astype(int)

    macro_f1 = f1_score(all_labels, preds, average="macro")
    recall_dep = recall_score(all_labels, preds, pos_label=1)
    precision_dep = precision_score(all_labels, preds, pos_label=1)

    macro_f1s.append(macro_f1)
    depressed_recalls.append(recall_dep)
    depressed_precisions.append(precision_dep)

# Find best macro F1 threshold
best_idx = np.argmax(macro_f1s)
best_threshold = thresholds[best_idx]

print("\nBest Threshold (Macro F1):", round(best_threshold, 3))
print("Best Macro F1:", round(macro_f1s[best_idx], 4))
print("Depressed Recall at best threshold:", round(depressed_recalls[best_idx],4))
print("Depressed Precision at best threshold:", round(depressed_precisions[best_idx],4))

# Plot F1 vs Threshold
plt.figure()
plt.plot(thresholds, macro_f1s)
plt.xlabel("Threshold")
plt.ylabel("Macro F1")
plt.title("Macro F1 vs Threshold")
plt.show()

# Plot Recall vs Threshold
plt.figure()
plt.plot(thresholds, depressed_recalls)
plt.xlabel("Threshold")
plt.ylabel("Depressed Recall")
plt.title("Depressed Recall vs Threshold")
plt.show()
