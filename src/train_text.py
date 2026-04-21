import os
os.environ["WANDB_DISABLED"] = "true"

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# -------------------------------
# PATHS
# -------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "daic_text_clean.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models", "hierarchical_text")
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pt")

os.makedirs(MODEL_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------
# LOAD DATA
# -------------------------------
df = pd.read_csv(DATA_PATH)

train_df = df[df["split"] == "train"].reset_index(drop=True)
dev_df = df[df["split"] == "dev"].reset_index(drop=True)

train_df["label"] = train_df["label"].astype(int)
dev_df["label"] = dev_df["label"].astype(int)

print("Train interviews:", len(train_df))
print("Dev interviews:", len(dev_df))

# -------------------------------
# CLASS WEIGHTS
# -------------------------------
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.array([0, 1]),
    y=train_df["label"]
)

class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
print("Class weights:", class_weights)

# -------------------------------
# TOKENIZER
# -------------------------------
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# -------------------------------
# DATASET
# -------------------------------
class HierarchicalDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=512, stride=256):
        self.samples = []

        for _, row in df.iterrows():
            text = row["text"]
            label = int(row["label"])

            encoding = tokenizer(
                text,
                truncation=True,
                max_length=max_length,
                stride=stride,
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

            self.samples.append((chunks, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def collate_fn(batch):
    max_chunks = max(len(sample[0]) for sample in batch)

    input_ids_batch = []
    attention_masks_batch = []
    labels = []

    for chunks, label in batch:
        padded_chunks = chunks + [
            (torch.zeros(512, dtype=torch.long),
             torch.zeros(512, dtype=torch.long))
        ] * (max_chunks - len(chunks))

        input_ids = torch.stack([c[0] for c in padded_chunks])
        attention_masks = torch.stack([c[1] for c in padded_chunks])

        input_ids_batch.append(input_ids)
        attention_masks_batch.append(attention_masks)
        labels.append(label)

    return (
        torch.stack(input_ids_batch).to(device),
        torch.stack(attention_masks_batch).to(device),
        torch.tensor(labels).to(device)
    )

train_dataset = HierarchicalDataset(train_df, tokenizer)
dev_dataset = HierarchicalDataset(dev_df, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
dev_loader = DataLoader(dev_dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)

# -------------------------------
# MODEL
# -------------------------------
class HierarchicalModel(nn.Module):
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

        outputs = self.encoder(input_ids=input_ids,
                               attention_mask=attention_mask)

        cls_embeddings = outputs.last_hidden_state[:, 0, :]
        cls_embeddings = cls_embeddings.view(batch_size, num_chunks, -1)

        attn_scores = self.attention(cls_embeddings).squeeze(-1)
        attn_weights = torch.softmax(attn_scores, dim=1).unsqueeze(-1)

        weighted_sum = torch.sum(cls_embeddings * attn_weights, dim=1)
        logits = self.classifier(weighted_sum)

        return logits

model = HierarchicalModel(model_name).to(device)

# -------------------------------
# LOSS & OPTIMIZER
# -------------------------------
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

# -------------------------------
# EVALUATION
# -------------------------------
def evaluate():
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for input_ids, attention_mask, labels in dev_loader:
            logits = model(input_ids, attention_mask)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")
    return acc, f1

# -------------------------------
# TRAINING WITH BEST SAVE
# -------------------------------
best_f1 = 0

num_epochs = 4

for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for input_ids, attention_mask, labels in tqdm(train_loader):
        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    val_acc, val_f1 = evaluate()

    print(f"\nEpoch {epoch+1}")
    print("Train Loss:", avg_loss)
    print("Val Accuracy:", val_acc)
    print("Val Macro F1:", val_f1)

    # Save best model
    if val_f1 > best_f1:
        best_f1 = val_f1
        torch.save(model.state_dict(), BEST_MODEL_PATH)
        print("Saved new best model.")

print("\nBest Macro F1 achieved:", best_f1)

# -------------------------------
# OPTIONAL: LOW LR REFINEMENT
# -------------------------------
print("\nLoading best model for refinement...")
model.load_state_dict(torch.load(BEST_MODEL_PATH))

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-6)

for epoch in range(2):  # 1–2 refinement epochs
    model.train()
    total_loss = 0

    for input_ids, attention_mask, labels in tqdm(train_loader):
        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    val_acc, val_f1 = evaluate()

    print(f"\nRefinement Epoch {epoch+1}")
    print("Train Loss:", avg_loss)
    print("Val Accuracy:", val_acc)
    print("Val Macro F1:", val_f1)

    if val_f1 > best_f1:
        best_f1 = val_f1
        torch.save(model.state_dict(), BEST_MODEL_PATH)
        print("Saved improved best model.")

print("\nFinal Best Macro F1:", best_f1)
