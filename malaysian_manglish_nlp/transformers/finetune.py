"""
Fine-tune distilbert-base-multilingual-cased for multi-task Manglish classification.

Tasks:
    1. Sentiment (3 classes: positive/negative/neutral)
    2. Emotion (8 classes: happy/sad/angry/fear/surprise/disgust/love/neutral)
    3. Intent (6 classes: question/statement/request/complaint/greeting/opinion)

Usage:
    python -m malaysian_manglish_nlp.transformers.finetune
    
    # Or programmatically:
    from malaysian_manglish_nlp.transformers.finetune import train
    train("datasets/manglish_labeled.jsonl", "output/manglish_multitask")
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import json
import os
from pathlib import Path

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader, random_split
    from transformers import (
        AutoModel,
        AutoTokenizer,
        get_linear_schedule_with_warmup,
    )
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# Label definitions
SENTIMENT_LABELS = ['positive', 'negative', 'neutral']
EMOTION_LABELS = ['happy', 'sad', 'angry', 'fear', 'surprise', 'disgust', 'love', 'neutral']
INTENT_LABELS = ['question', 'statement', 'request', 'complaint', 'greeting', 'opinion']

LABEL_MAPS = {
    'sentiment': {label: i for i, label in enumerate(SENTIMENT_LABELS)},
    'emotion': {label: i for i, label in enumerate(EMOTION_LABELS)},
    'intent': {label: i for i, label in enumerate(INTENT_LABELS)},
}

MODEL_NAME = 'distilbert-base-multilingual-cased'


class ManglishMultiTaskModel(nn.Module):
    """Multi-task classification model with shared encoder and 3 task heads.
    
    Architecture:
        - Shared encoder: distilbert-base-multilingual-cased
        - Head 1: Sentiment (3 classes)
        - Head 2: Emotion (8 classes)
        - Head 3: Intent (6 classes)
    """
    
    def __init__(self, encoder_name: str = MODEL_NAME, dropout: Any = 0.3) -> None:
        """Initialize the object.

        Args:
            encoder_name: Encoder name parameter.
            dropout: Dropout parameter.

        Returns:
            Result value.

        """
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name)
        hidden_size = self.encoder.config.hidden_size
        
        # Shared pooling + dropout
        self.dropout = nn.Dropout(dropout)
        
        # Task-specific classification heads
        self.sentiment_head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, len(SENTIMENT_LABELS)),
        )
        
        self.emotion_head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, len(EMOTION_LABELS)),
        )
        
        self.intent_head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, len(INTENT_LABELS)),
        )
    
    def forward(self, input_ids: Any, attention_mask: Any) -> Any:
        """Forward pass through shared encoder + all task heads.
        
        Args:
            input_ids: Token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
        
        Returns:
            dict: Logits for each task head.
        """
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        
        # Use [CLS] token representation
        cls_output = outputs.last_hidden_state[:, 0, :]
        cls_output = self.dropout(cls_output)
        
        return {
            'sentiment': self.sentiment_head(cls_output),
            'emotion': self.emotion_head(cls_output),
            'intent': self.intent_head(cls_output),
        }


class ManglishDataset(Dataset):
    """Dataset for multi-task Manglish classification.
    
    Expected JSONL format:
        {"text": "...", "sentiment": "positive", "emotion": "happy", "intent": "statement"}
    
    Args:
        data_path: Path to JSONL file.
        tokenizer: HuggingFace tokenizer.
        max_length: Max token length.
    """
    
    def __init__(self, data_path: str, tokenizer: Any, max_length: int = 128) -> None:
        """Initialize the object.

        Args:
            data_path: Data path parameter.
            tokenizer: Tokenizer parameter.
            max_length: Max length parameter.

        Returns:
            Result value.

        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    self.samples.append(json.loads(line))
    
    def __len__(self) -> int:
        """Return number of items in cache.

        Returns:
            Integer value.

        """
        return len(self.samples)
    
    def __getitem__(self, idx: Any) -> Any:
        """Internal helper for  getitem  .

        Args:
            idx: Idx parameter.

        Returns:
            Result value.

        """
        sample = self.samples[idx]
        text = sample['text']
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        
        item = {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
        }
        
        # Map labels to indices (handle missing labels gracefully)
        if 'sentiment' in sample:
            item['sentiment'] = torch.tensor(
                LABEL_MAPS['sentiment'].get(sample['sentiment'], 0), dtype=torch.long
            )
        if 'emotion' in sample:
            item['emotion'] = torch.tensor(
                LABEL_MAPS['emotion'].get(sample['emotion'], 7), dtype=torch.long
            )
        if 'intent' in sample:
            item['intent'] = torch.tensor(
                LABEL_MAPS['intent'].get(sample['intent'], 1), dtype=torch.long
            )
        
        return item


def evaluate(model: Any, dataloader: Any, device: Optional[Any] = None) -> Dict[str, Any]:
    """Evaluate model on a dataloader.
    
    Args:
        model: The model.
        dataloader: Validation dataloader.
        device: torch device.
    
    Returns:
        dict: Accuracy per task and average loss.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.eval()
    criterion = nn.CrossEntropyLoss()
    
    total_loss = 0.0
    correct = {'sentiment': 0, 'emotion': 0, 'intent': 0}
    total = 0
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            logits = model(input_ids, attention_mask)
            
            batch_loss = 0.0
            task_count = 0
            
            for task in ['sentiment', 'emotion', 'intent']:
                if task in batch:
                    labels = batch[task].to(device)
                    loss = criterion(logits[task], labels)
                    batch_loss += loss.item()
                    task_count += 1
                    
                    preds = logits[task].argmax(dim=1)
                    correct[task] += (preds == labels).sum().item()
            
            if task_count > 0:
                total_loss += batch_loss / task_count
            total += input_ids.size(0)
    
    num_batches = len(dataloader)
    results = {
        'avg_loss': total_loss / max(num_batches, 1),
        'accuracy': {},
    }
    
    for task in ['sentiment', 'emotion', 'intent']:
        results['accuracy'][task] = correct[task] / max(total, 1)
    
    return results


def train(data_path: str='datasets/manglish_labeled.jsonl', output_dir: str='resources/manglish_finetuned',
          epochs: Any = 5, batch_size: int=16, lr: float=2e-5, max_length: int=128) -> Dict[str, Any]:
    """Fine-tune the multi-task model.
    
    Args:
        data_path: Path to labeled JSONL data.
        output_dir: Directory to save model, tokenizer, and label mappings.
        epochs: Number of training epochs.
        batch_size: Batch size.
        lr: Learning rate.
        max_length: Max token sequence length.
    
    Returns:
        dict: Training results with final metrics.
    """
    if not HAS_TORCH:
        raise ImportError(
            "torch and transformers required. Install: pip install torch transformers"
        )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load tokenizer and create dataset
    print(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    print(f"Loading data from: {data_path}")
    dataset = ManglishDataset(data_path, tokenizer, max_length=max_length)
    print(f"Total samples: {len(dataset)}")
    
    # Train/val split (80/20) with fixed seed
    generator = torch.Generator().manual_seed(42)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)
    
    print(f"Train: {train_size}, Val: {val_size}")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    print(f"Initializing multi-task model...")
    model = ManglishMultiTaskModel(MODEL_NAME)
    model.to(device)
    
    # Optimizer with different LR for encoder vs heads
    encoder_params = list(model.encoder.parameters())
    head_params = (
        list(model.sentiment_head.parameters()) +
        list(model.emotion_head.parameters()) +
        list(model.intent_head.parameters())
    )
    
    optimizer = torch.optim.AdamW([
        {'params': encoder_params, 'lr': lr},
        {'params': head_params, 'lr': lr * 10},  # Higher LR for heads
    ], weight_decay=0.01)
    
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )
    
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    best_val_acc = 0.0
    history = []
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        
        for step, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            logits = model(input_ids, attention_mask)
            
            # Multi-task loss (average across available tasks)
            loss = torch.tensor(0.0, device=device)
            task_count = 0
            
            for task in ['sentiment', 'emotion', 'intent']:
                if task in batch:
                    labels = batch[task].to(device)
                    task_loss = criterion(logits[task], labels)
                    loss = loss + task_loss
                    task_count += 1
            
            if task_count > 0:
                loss = loss / task_count
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            epoch_loss += loss.item()
            
            if (step + 1) % 50 == 0:
                print(f"  Epoch {epoch+1}/{epochs}, Step {step+1}/{len(train_loader)}, "
                      f"Loss: {loss.item():.4f}")
        
        avg_train_loss = epoch_loss / max(len(train_loader), 1)
        
        # Validation
        val_results = evaluate(model, val_loader, device)
        avg_val_acc = sum(val_results['accuracy'].values()) / 3
        
        print(f"Epoch {epoch+1}/{epochs} - "
              f"Train Loss: {avg_train_loss:.4f}, "
              f"Val Loss: {val_results['avg_loss']:.4f}, "
              f"Val Acc: sentiment={val_results['accuracy']['sentiment']:.3f}, "
              f"emotion={val_results['accuracy']['emotion']:.3f}, "
              f"intent={val_results['accuracy']['intent']:.3f}")
        
        history.append({
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'val_loss': val_results['avg_loss'],
            'val_accuracy': val_results['accuracy'],
        })
        
        # Save best model
        if avg_val_acc > best_val_acc:
            best_val_acc = avg_val_acc
            _save_model(model, tokenizer, output_dir)
            print(f"  -> Saved best model (avg acc: {best_val_acc:.4f})")
    
    print(f"\nTraining complete. Best avg accuracy: {best_val_acc:.4f}")
    print(f"Model saved to: {output_dir}")
    
    return {
        'best_val_accuracy': best_val_acc,
        'history': history,
        'output_dir': output_dir,
    }


def _save_model(model: Any, tokenizer: Any, output_dir: str) -> None:
    """Save model weights, tokenizer, and label mappings.
    
    Args:
        model: Trained model.
        tokenizer: Tokenizer.
        output_dir: Output directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model state dict
    torch.save(model.state_dict(), os.path.join(output_dir, 'model.pt'))
    
    # Save tokenizer
    tokenizer.save_pretrained(output_dir)
    
    # Save label mappings and config
    config = {
        'model_name': MODEL_NAME,
        'label_maps': LABEL_MAPS,
        'labels': {
            'sentiment': SENTIMENT_LABELS,
            'emotion': EMOTION_LABELS,
            'intent': INTENT_LABELS,
        },
        'architecture': 'ManglishMultiTaskModel',
    }
    
    with open(os.path.join(output_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    train()
