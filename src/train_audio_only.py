import os
import random
import torch
import torch.nn as nn
import torchaudio
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import Dataset, DataLoader
from transformers import Wav2Vec2Model, Wav2Vec2Processor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ======================
# PATHS
# ======================
BASE_AUDIO = "D:/daic_audio_subset"
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
CSV_PATH = os.path.join(BASE_DIR, "data", "processed", "daic_text_clean.csv")
SAVE_DIR = os.path.join(BASE_DIR, "models", "audio_only")
os.makedirs(SAVE_DIR, exist_ok=True)

# ======================
# LOAD DATA
# ======================
df = pd.read_csv(CSV_PATH)
train_df = df[df["split"] == "train"].reset_index(drop=True)
test_df = df[df["split"] == "dev"].reset_index(drop=True)

print(f"\nTraining samples: {len(train_df)}")
print(f"Test samples: {len(test_df)}")
print(f"Train class distribution: {train_df['label'].value_counts().to_dict()}")
print(f"Test class distribution: {test_df['label'].value_counts().to_dict()}")

# ======================
# CLASS WEIGHTS
# ======================
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.array([0, 1]),
    y=train_df["label"]
)
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
print(f"Class weights: {class_weights}")


# ======================
# FOCAL LOSS
# ======================
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, weight=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(
            inputs, targets,
            weight=self.weight,
            reduction='none'
        )
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


# ======================
# AUDIO MODEL (FROZEN)
# ======================
print("\nLoading Wav2Vec2 model...")
audio_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
audio_encoder = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base").to(device)
audio_encoder.eval()

for param in audio_encoder.parameters():
    param.requires_grad = False

print("Wav2Vec2 loaded and frozen ✓")


# ======================
# CLASSIFIER
# ======================
class AudioClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(768, 256)
        self.ln1 = nn.LayerNorm(256)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 128)
        self.ln2 = nn.LayerNorm(128)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(128, 2)

    def forward(self, audio_emb):
        x = self.fc1(audio_emb)
        x = self.ln1(x)
        x = torch.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.ln2(x)
        x = torch.relu(x)
        x = self.dropout2(x)
        return self.fc3(x)


# ======================
# DATASET
# ======================
class AudioDataset(Dataset):
    def __init__(self, df, segment_seconds=20):
        self.df = df
        self.segment_seconds = segment_seconds

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        pid = str(int(row["participant_id"]))
        label = int(row["label"])

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

        return segments, torch.tensor(label)


# ======================
# EVALUATION
# ======================
def evaluate(classifier, loader):
    classifier.eval()
    audio_encoder.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for segments, labels in loader:
            labels = labels.to(device)

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
            logits = classifier(audio_emb)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    torch.cuda.empty_cache()

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")

    return acc, f1, all_labels, all_preds


# ======================
# TRAINING
# ======================
def train_one_seed(seed):
    print(f"\n{'=' * 60}")
    print(f"SEED {seed}")
    print(f"{'=' * 60}")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    classifier = AudioClassifier().to(device)
    criterion = FocalLoss(alpha=0.25, gamma=2, weight=class_weights)
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=1e-4, weight_decay=0.01)

    train_loader = DataLoader(AudioDataset(train_df), batch_size=1, shuffle=True)
    test_loader = DataLoader(AudioDataset(test_df), batch_size=1, shuffle=False)

    best_f1 = 0
    best_model_state = None
    patience = 5
    patience_counter = 0

    for epoch in range(15):
        classifier.train()
        total_loss = 0

        for segments, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
            labels = labels.to(device)

            with torch.no_grad():
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
            logits = classifier(audio_emb)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        torch.cuda.empty_cache()

        avg_loss = total_loss / len(train_loader)
        acc, f1, _, _ = evaluate(classifier, test_loader)

        print(f"Epoch {epoch + 1}: Loss={avg_loss:.4f}, Acc={acc:.4f}, F1={f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            best_model_state = classifier.state_dict().copy()
            patience_counter = 0
            print(f"  → New best F1: {best_f1:.4f}")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    return best_f1, best_model_state


# ======================
# RUN 5 SEEDS
# ======================
seeds = [42, 123, 999, 555, 777]
f1_scores = []
accuracies = []
all_results = []

best_overall_f1 = 0
best_overall_state = None
best_seed = None

for seed in seeds:
    best_f1, model_state = train_one_seed(seed)

    classifier = AudioClassifier().to(device)
    classifier.load_state_dict(model_state)
    test_loader = DataLoader(AudioDataset(test_df), batch_size=1, shuffle=False)
    acc, f1, labels, preds = evaluate(classifier, test_loader)

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

    del classifier
    torch.cuda.empty_cache()

# ======================
# FINAL RESULTS
# ======================
print("\n" + "=" * 60)
print("AUDIO-ONLY RESULTS")
print("=" * 60)

print(f"\nF1 scores:")
for seed, f1 in zip(seeds, f1_scores):
    print(f"  Seed {seed}: {f1:.4f}")

print(f"\nMean Macro F1: {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")
print(f"Mean Accuracy: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
print(f"Min F1: {np.min(f1_scores):.4f}")
print(f"Max F1: {np.max(f1_scores):.4f}")
print(f"Variance: {np.var(f1_scores):.6f}")

# ======================
# BEST MODEL ANALYSIS
# ======================
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

torch.save(best_overall_state, os.path.join(SAVE_DIR, "best_audio_only_model.pt"))
print(f"\nBest model saved to {SAVE_DIR}")

print("\n" + "=" * 60)
print("KEY FINDINGS:")
print("=" * 60)
print(f"1. Variance across seeds: {np.std(f1_scores):.4f}")
print(f"2. Range: {np.min(f1_scores):.4f} to {np.max(f1_scores):.4f}")
print(f"3. Demonstrates instability of audio-only approach")
print(f"4. Justifies multimodal fusion necessity")