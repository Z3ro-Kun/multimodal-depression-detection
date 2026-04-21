import os
import random
import numpy as np
import torch
import torch.nn as nn
import torchaudio
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModel,
    Wav2Vec2Model,
    Wav2Vec2Processor
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_AUDIO = "D:/daic_audio_subset"
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
CSV_PATH = os.path.join(BASE_DIR, "data", "processed", "daic_text_clean.csv")
TEXT_MODEL_PATH = os.path.join(BASE_DIR, "models", "hierarchical_text", "best_model.pt")
FUSION_SAVE_DIR = os.path.join(BASE_DIR, "models", "fusion_multiseed")
os.makedirs(FUSION_SAVE_DIR, exist_ok=True)

df = pd.read_csv(CSV_PATH)
train_df = df[df["split"] == "train"].reset_index(drop=True)
dev_df = df[df["split"] == "dev"].reset_index(drop=True)

print(f"Training samples: {len(train_df)}")
print(f"Test samples: {len(dev_df)}")

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.array([0, 1]),
    y=train_df["label"]
)
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
print(f"Class weights: {class_weights}")

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


def evaluate_model(model, loader):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for input_ids, attention_mask, segments, labels in loader:
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
            logits = model(text_emb, audio_emb)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")

    return acc, f1, all_labels, all_preds


def train_and_evaluate(seed):
    print(f"\n{'=' * 60}")
    print(f"SEED {seed}")
    print(f"{'=' * 60}")

    set_seed(seed)

    train_loader = DataLoader(FusionDataset(train_df), batch_size=1, shuffle=True)
    dev_loader = DataLoader(FusionDataset(dev_df), batch_size=1, shuffle=False)

    model = FusionModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_f1 = 0
    best_model_state = None
    patience = 3
    patience_counter = 0

    for epoch in range(10):
        model.train()
        total_loss = 0

        for input_ids, attention_mask, segments, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
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
            logits = model(text_emb, audio_emb)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        torch.cuda.empty_cache()

        avg_loss = total_loss / len(train_loader)
        acc, f1, _, _ = evaluate_model(model, dev_loader)

        print(f"Epoch {epoch + 1}: Loss={avg_loss:.4f}, Acc={acc:.4f}, F1={f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            print(f"  → New best F1: {best_f1:.4f}")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    return best_f1, best_model_state


seeds = [42, 123, 999, 555, 777]
f1_scores = []
accuracies = []
all_results = []

best_overall_f1 = 0
best_overall_state = None
best_seed = None

for seed in seeds:
    best_f1, model_state = train_and_evaluate(seed)

    dev_loader = DataLoader(FusionDataset(dev_df), batch_size=1, shuffle=False)
    best_model = FusionModel().to(device)
    best_model.load_state_dict(model_state)

    acc, f1, labels, preds = evaluate_model(best_model, dev_loader)

    f1_scores.append(f1)
    accuracies.append(acc)
    all_results.append({
        'seed': seed,
        'f1': f1,
        'acc': acc,
        'labels': labels,
        'preds': preds
    })

    if f1 > best_overall_f1:
        best_overall_f1 = f1
        best_overall_state = model_state
        best_seed = seed

    print(f"\nSeed {seed} final: Acc={acc:.4f}, F1={f1:.4f}")

    del best_model
    torch.cuda.empty_cache()

print("\n" + "=" * 60)
print("FUSION MODEL RESULTS")
print("=" * 60)

print(f"\nF1 scores:")
for seed, f1 in zip(seeds, f1_scores):
    print(f"  Seed {seed}: {f1:.4f}")

print(f"\nMean Macro F1: {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")
print(f"Mean Accuracy: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
print(f"Min F1: {np.min(f1_scores):.4f}")
print(f"Max F1: {np.max(f1_scores):.4f}")
print(f"Variance: {np.var(f1_scores):.6f}")

print("\n" + "=" * 60)
print(f"BEST MODEL (Seed {best_seed}, F1={best_overall_f1:.4f})")
print("=" * 60)

best_result = [r for r in all_results if r['seed'] == best_seed][0]
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

torch.save(best_overall_state, os.path.join(FUSION_SAVE_DIR, f"best_fusion_seed{best_seed}.pt"))
print(f"\nBest model saved to {FUSION_SAVE_DIR}")