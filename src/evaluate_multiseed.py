import os
import random
import torch
import torch.nn as nn
import torchaudio
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, Wav2Vec2Model, Wav2Vec2Processor


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_AUDIO = "D:/daic_audio_subset"
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
CSV_PATH = os.path.join(BASE_DIR, "data", "processed", "daic_text_clean.csv")
TEXT_MODEL_PATH = os.path.join(BASE_DIR, "models", "hierarchical_text", "best_model.pt")
FUSION_MODEL_PATH = os.path.join(BASE_DIR, "models", "fusion_model2", "fusion_best_model.pt")

df = pd.read_csv(CSV_PATH)
test_df = df[df["split"] == "dev"].reset_index(drop=True)

print(f"Test samples: {len(test_df)}")

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
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
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

for param in text_model.parameters():
    param.requires_grad = False

audio_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
audio_encoder = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base").to(device)
audio_encoder.eval()

for param in audio_encoder.parameters():
    param.requires_grad = False


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

        input_ids = torch.stack([torch.tensor(ids) for ids in encoding["input_ids"]])
        attention_mask = torch.stack([torch.tensor(mask) for mask in encoding["attention_mask"]])

        wav_path = os.path.join(BASE_AUDIO, f"{pid}_P", f"{pid}_AUDIO.wav")
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

print("Models loaded successfully!")

# Evaluate across multiple random seeds for audio segments
seeds = [42, 123, 999, 555, 777, 888, 111, 222, 333, 444]

all_results = []

for seed in seeds:
    print(f"\n{'=' * 60}")
    print(f"Evaluating with seed {seed}")
    print(f"{'=' * 60}")

    set_seed(seed)

    test_loader = DataLoader(FusionDataset(test_df), batch_size=1, shuffle=False)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for input_ids, attention_mask, segments, labels in test_loader:
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

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")

    all_results.append({
        'seed': seed,
        'accuracy': acc,
        'f1': f1,
        'labels': all_labels,
        'preds': all_preds
    })

    print(f"Seed {seed}: Acc={acc:.4f}, F1={f1:.4f}")

# Calculate statistics
f1_scores = [r['f1'] for r in all_results]
accuracies = [r['accuracy'] for r in all_results]

print("\n" + "=" * 60)
print("FUSION MODEL2 - EVALUATION RESULTS (10 SEEDS)")
print("=" * 60)

print(f"\nF1 scores:")
for result in all_results:
    print(f"  Seed {result['seed']}: {result['f1']:.4f}")

print(f"\nMean Macro F1: {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")
print(f"Mean Accuracy: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
print(f"Min F1: {np.min(f1_scores):.4f}")
print(f"Max F1: {np.max(f1_scores):.4f}")
print(f"Variance: {np.var(f1_scores):.6f}")

# Best seed analysis
best_idx = np.argmax(f1_scores)
best_result = all_results[best_idx]

print("\n" + "=" * 60)
print(f"BEST SEED ({best_result['seed']}, F1={best_result['f1']:.4f})")
print("=" * 60)

labels = best_result['labels']
preds = best_result['preds']

print("\nClassification Report:")
print(classification_report(labels, preds, target_names=['Non-Depressed', 'Depressed'], digits=4))

print("\nConfusion Matrix:")
cm = confusion_matrix(labels, preds)
print(cm)
print(f"TN={cm[0, 0]}, FP={cm[0, 1]}, FN={cm[1, 0]}, TP={cm[1, 1]}")

from sklearn.metrics import precision_score, recall_score

print("\nPer-Class Metrics:")
print(
    f"Non-Depressed - Precision: {precision_score(labels, preds, pos_label=0):.4f}, Recall: {recall_score(labels, preds, pos_label=0):.4f}")
print(
    f"Depressed - Precision: {precision_score(labels, preds, pos_label=1):.4f}, Recall: {recall_score(labels, preds, pos_label=1):.4f}")