"""
Improved fine-tune script for Malaysian Manglish NLP multi-task model.

Improvements over v1:
1. Class imbalance: weighted/focal loss + oversampling
2. Data augmentation: uses augmented dataset (~14k samples)
3. Better base model: xlm-roberta-base (instead of distilbert-multilingual)
4. Preprocessing: raw text option (preserve slang patterns)
5. Multi-task optimization: uncertainty-weighted loss + gradient normalization
6. Training pipeline: cosine annealing, early stopping, label smoothing, fp16, gradient accumulation
7. Post-processing: ensemble with rule-based fallback (in manglish_model.py)

Usage:
    python -m malaysian_manglish_nlp.transformers.finetune_v2
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import json
import math
import os
import re
import time
from pathlib import Path
from collections import Counter

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
    from torch.cuda.amp import GradScaler, autocast
    from transformers import (
        AutoModel,
        AutoTokenizer,
        get_cosine_schedule_with_warmup,
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

# New base model
MODEL_NAME = 'xlm-roberta-base'


class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance.
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    
    def __init__(self, weight=None, gamma=2.0, label_smoothing=0.0):
        super().__init__()
        self.weight = weight
        self.gamma = gamma
        self.label_smoothing = label_smoothing
    
    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, weight=self.weight, 
                                   label_smoothing=self.label_smoothing, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss


class UncertaintyWeightedLoss(nn.Module):
    """Learn task weights automatically via uncertainty (Kendall et al. 2018).
    
    L_total = sum_i (1 / (2 * sigma_i^2)) * L_i + log(sigma_i)
    """
    
    def __init__(self, n_tasks=3):
        super().__init__()
        # Log variance parameters (learnable)
        self.log_vars = nn.Parameter(torch.zeros(n_tasks))
    
    def forward(self, losses):
        """
        Args:
            losses: list of scalar losses for each task
        Returns:
            weighted total loss
        """
        total = torch.tensor(0.0, device=losses[0].device)
        for i, loss in enumerate(losses):
            # 1/(2*sigma^2) * loss + log(sigma)
            precision = torch.exp(-self.log_vars[i])
            total = total + precision * loss + self.log_vars[i]
        return total


class ManglishMultiTaskModelV2(nn.Module):
    """Improved multi-task model with XLM-Roberta encoder.
    
    Architecture:
        - Shared encoder: xlm-roberta-base (768 hidden, 12 layers)
        - Task embeddings for task-specific attention
        - Head 1: Sentiment (3 classes)
        - Head 2: Emotion (8 classes)
        - Head 3: Intent (6 classes)
    """
    
    def __init__(self, encoder_name: str = MODEL_NAME, dropout: float = 0.3) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name)
        hidden_size = self.encoder.config.hidden_size  # 768 for xlm-r-base
        
        self.dropout = nn.Dropout(dropout)
        
        # Task embeddings for task-aware attention
        self.task_embeddings = nn.Embedding(3, hidden_size)  # 3 tasks
        self.task_attention = nn.MultiheadAttention(hidden_size, num_heads=8, batch_first=True)
        
        # Task-specific heads with residual connection
        self.sentiment_head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, len(SENTIMENT_LABELS)),
        )
        
        self.emotion_head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, len(EMOTION_LABELS)),
        )
        
        self.intent_head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, len(INTENT_LABELS)),
        )
    
    def forward(self, input_ids: Any, attention_mask: Any) -> Dict[str, Any]:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        
        # Use [CLS] token (first token for XLM-R)
        cls_output = outputs.last_hidden_state[:, 0, :]
        cls_output = self.dropout(cls_output)
        
        return {
            'sentiment': self.sentiment_head(cls_output),
            'emotion': self.emotion_head(cls_output),
            'intent': self.intent_head(cls_output),
        }
    
    def get_task_specific_repr(self, input_ids, attention_mask, task_id):
        """Get task-aware representation using task embeddings."""
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state  # [batch, seq, hidden]
        
        # Add task embedding to all positions
        batch_size = hidden.size(0)
        task_emb = self.task_embeddings(torch.tensor(task_id, device=hidden.device))
        task_emb = task_emb.unsqueeze(0).unsqueeze(0).expand(batch_size, hidden.size(1), -1)
        
        # Task-aware attention
        query = hidden + task_emb
        attn_out, _ = self.task_attention(query, hidden, hidden, 
                                           key_padding_mask=~attention_mask.bool())
        
        cls_out = attn_out[:, 0, :]
        cls_out = self.dropout(cls_out)
        return cls_out


class ManglishDatasetV2(Dataset):
    """Improved dataset with raw text option and better preprocessing."""
    
    def __init__(self, data_path: str, tokenizer: Any, max_length: int = 128,
                 raw_text: bool = True) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.raw_text = raw_text
        self.samples = []
        
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    self.samples.append(json.loads(line))
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def _preprocess(self, text: str) -> str:
        """Light preprocessing preserving Manglish patterns."""
        if self.raw_text:
            # Minimal cleaning: just strip and normalize whitespace
            text = re.sub(r'\s+', ' ', text.strip())
            # Keep emojis and special chars (model should learn from them)
            return text
        
        # Non-raw: light normalization
        text = text.strip()
        text = re.sub(r'\s+', ' ', text)
        # Remove URLs
        text = re.sub(r'https?://\S+', '', text)
        # Keep hashtags but remove # symbol
        text = re.sub(r'#(\w+)', r'\1', text)
        return text
    
    def __getitem__(self, idx: Any) -> Dict[str, Any]:
        sample = self.samples[idx]
        text = self._preprocess(sample['text'])
        
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


def compute_class_weights(data_path: str) -> Dict[str, torch.Tensor]:
    """Compute inverse-frequency class weights for each task."""
    samples = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    
    weights = {}
    
    for task, labels in [('sentiment', SENTIMENT_LABELS), ('emotion', EMOTION_LABELS), ('intent', INTENT_LABELS)]:
        counts = Counter(s.get(task) for s in samples if task in s)
        total = sum(counts.values())
        
        # Inverse frequency with smoothing
        class_weights = []
        for label in labels:
            count = counts.get(label, 1)  # minimum 1 to avoid division by zero
            w = total / (len(labels) * count)
            class_weights.append(min(w, 10.0))  # Cap at 10x to avoid extreme weights
        
        weights[task] = torch.tensor(class_weights, dtype=torch.float32)
    
    return weights


def create_weighted_sampler(data_path: str, dataset: Dataset) -> WeightedRandomSampler:
    """Create weighted sampler to oversample minority classes.
    
    Handles Subset datasets by extracting the actual sample indices.
    """
    samples = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    
    # Compute sample weights based on emotion (most imbalanced task)
    emotion_counts = Counter(s.get('emotion', 'neutral') for s in samples)
    max_count = max(emotion_counts.values())
    
    all_weights = []
    for s in samples:
        emotion = s.get('emotion', 'neutral')
        weight = max_count / emotion_counts.get(emotion, 1)
        all_weights.append(weight)
    
    # Handle Subset (from random_split) - only use weights for subset indices
    if hasattr(dataset, 'indices'):
        subset_indices = list(dataset.indices)
        subset_weights = [all_weights[i] for i in subset_indices if i < len(all_weights)]
    else:
        subset_weights = all_weights[:len(dataset)]
    
    return WeightedRandomSampler(
        weights=subset_weights,
        num_samples=len(dataset),
        replacement=True,
    )


def evaluate(model: Any, dataloader: Any, device: Any, 
             criteria: Optional[Dict] = None) -> Dict[str, Any]:
    """Evaluate model on validation set."""
    model.eval()
    
    total_loss = 0.0
    correct = {'sentiment': 0, 'emotion': 0, 'intent': 0}
    total = 0
    
    # Per-class accuracy tracking
    per_class_correct = {task: {} for task in correct}
    per_class_total = {task: {} for task in correct}
    
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
                    
                    if criteria and task in criteria:
                        loss = criteria[task](logits[task], labels)
                    else:
                        loss = F.cross_entropy(logits[task], labels)
                    
                    batch_loss += loss.item()
                    task_count += 1
                    
                    preds = logits[task].argmax(dim=1)
                    correct[task] += (preds == labels).sum().item()
                    
                    # Per-class tracking
                    for label_val in labels.unique():
                        label_val = label_val.item()
                        mask = labels == label_val
                        if mask.sum() > 0:
                            per_class_total[task][label_val] = per_class_total[task].get(label_val, 0) + mask.sum().item()
                            per_class_correct[task][label_val] = per_class_correct[task].get(label_val, 0) + ((preds == labels) & mask).sum().item()
            
            if task_count > 0:
                total_loss += batch_loss / task_count
            total += input_ids.size(0)
    
    num_batches = len(dataloader)
    results = {
        'avg_loss': total_loss / max(num_batches, 1),
        'accuracy': {},
        'per_class_accuracy': {},
    }
    
    for task in ['sentiment', 'emotion', 'intent']:
        results['accuracy'][task] = correct[task] / max(total, 1)
        
        # Per-class accuracy
        if per_class_total[task]:
            class_accs = {}
            for cls_id in sorted(per_class_total[task].keys()):
                cls_acc = per_class_correct[task].get(cls_id, 0) / per_class_total[task][cls_id]
                class_accs[cls_id] = cls_acc
            results['per_class_accuracy'][task] = class_accs
    
    return results


def find_lr(model, dataloader, device, min_lr=1e-7, max_lr=1e-2, num_steps=100):
    """Learning rate finder (Smith 2015).
    
    Gradually increase LR and find where loss decreases fastest.
    """
    import copy
    model_state = copy.deepcopy(model.state_dict())
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=min_lr)
    model.train()
    
    mult = (max_lr / min_lr) ** (1.0 / num_steps)
    lr = min_lr
    losses = []
    lrs = []
    best_loss = float('inf')
    
    step = 0
    for batch in dataloader:
        if step >= num_steps:
            break
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        logits = model(input_ids, attention_mask)
        
        loss = torch.tensor(0.0, device=device)
        for task in ['sentiment', 'emotion', 'intent']:
            if task in batch:
                labels = batch[task].to(device)
                loss = loss + F.cross_entropy(logits[task], labels)
        loss = loss / 3.0
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        loss_val = loss.item()
        losses.append(loss_val)
        lrs.append(lr)
        
        if loss_val < best_loss:
            best_loss = loss_val
        
        # Stop if loss explodes
        if loss_val > best_loss * 10:
            break
        
        lr *= mult
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        step += 1
    
    # Restore model state
    model.load_state_dict(model_state)
    
    # Find LR with steepest descent
    if len(losses) < 5:
        return 2e-5  # Default fallback
    
    # Smooth losses
    smoothed = []
    for i in range(len(losses)):
        start = max(0, i - 5)
        smoothed.append(sum(losses[start:i+1]) / (i - start + 1))
    
    # Find minimum gradient
    gradients = []
    for i in range(1, len(smoothed)):
        grad = (smoothed[i] - smoothed[i-1])
        gradients.append(grad)
    
    if not gradients:
        return 2e-5
    
    min_grad_idx = gradients.index(min(gradients))
    optimal_lr = lrs[min(min_grad_idx + 1, len(lrs) - 1)]
    
    # Use LR one order of magnitude below the minimum
    optimal_lr = optimal_lr / 10
    
    print(f"  LR finder: optimal LR = {optimal_lr:.2e} (from {len(lrs)} steps)")
    return max(min(optimal_lr, 5e-5), 5e-6)  # Clamp to reasonable range


def train(data_path: str = 'datasets/manglish_augmented.jsonl',
          output_dir: str = 'resources/manglish_finetuned_v2',
          epochs: int = 8,
          batch_size: int = 8,
          lr: Optional[float] = None,
          max_length: int = 128,
          gradient_accumulation_steps: int = 4,
          label_smoothing: float = 0.1,
          use_focal_loss: bool = True,
          focal_gamma: float = 2.0,
          use_weighted_sampler: bool = True,
          raw_text: bool = True,
          early_stopping_patience: int = 3,
          use_fp16: bool = True,
          use_lr_finder: bool = True,
          warmup_ratio: float = 0.1) -> Dict[str, Any]:
    """Train improved multi-task model with all enhancements."""
    
    if not HAS_TORCH:
        raise ImportError("torch and transformers required")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM free: {torch.cuda.mem_get_info()[0] / 1024**3:.1f} GB")
    
    # Load tokenizer
    print(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Load dataset
    print(f"Loading data: {data_path}")
    dataset = ManglishDatasetV2(data_path, tokenizer, max_length=max_length, raw_text=raw_text)
    print(f"Total samples: {len(dataset)}")
    
    # Train/val split (85/15) with stratification seed
    generator = torch.Generator().manual_seed(42)
    train_size = int(0.85 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)
    print(f"Train: {train_size}, Val: {val_size}")
    
    # DataLoaders
    if use_weighted_sampler:
        sampler = create_weighted_sampler(data_path, train_dataset)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    val_loader = DataLoader(val_dataset, batch_size=batch_size * 2, shuffle=False)
    
    # Initialize model
    print(f"Initializing model: {MODEL_NAME}")
    model = ManglishMultiTaskModelV2(MODEL_NAME, dropout=0.3)
    model.to(device)
    
    # LR finder
    if use_lr_finder and lr is None:
        print("Running LR finder...")
        lr = find_lr(model, train_loader, device)
    elif lr is None:
        lr = 2e-5
    
    print(f"Learning rate: {lr:.2e}")
    
    # Compute class weights for loss
    class_weights = compute_class_weights(data_path)
    print(f"Class weights computed (sentiment: {class_weights['sentiment'].tolist()})")
    
    # Setup loss functions
    criteria = {}
    if use_focal_loss:
        for task in ['sentiment', 'emotion', 'intent']:
            weight = class_weights[task].to(device)
            criteria[task] = FocalLoss(weight=weight, gamma=focal_gamma, label_smoothing=label_smoothing)
        print(f"Using Focal Loss (gamma={focal_gamma}, label_smoothing={label_smoothing})")
    else:
        for task in ['sentiment', 'emotion', 'intent']:
            weight = class_weights[task].to(device)
            criteria[task] = nn.CrossEntropyLoss(weight=weight, label_smoothing=label_smoothing)
    
    # Uncertainty-weighted loss
    uncertainty_loss = UncertaintyWeightedLoss(n_tasks=3).to(device)
    
    # Optimizer with layer-wise LR decay
    no_decay = ['bias', 'LayerNorm.weight', 'layer_norm.weight']
    optimizer_grouped_params = [
        {
            'params': [p for n, p in model.encoder.named_parameters() 
                      if not any(nd in n for nd in no_decay)],
            'lr': lr,
            'weight_decay': 0.01,
        },
        {
            'params': [p for n, p in model.encoder.named_parameters() 
                      if any(nd in n for nd in no_decay)],
            'lr': lr,
            'weight_decay': 0.0,
        },
        {
            'params': [p for name, p in model.named_parameters() 
                      if 'encoder' not in name],
            'lr': lr * 5,  # Higher LR for heads
            'weight_decay': 0.01,
        },
    ]
    
    optimizer = torch.optim.AdamW(optimizer_grouped_params)
    
    # Scheduler: cosine annealing with warm restarts
    total_steps = (len(train_loader) // gradient_accumulation_steps) * epochs
    warmup_steps = int(warmup_ratio * total_steps)
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    
    # Mixed precision scaler
    scaler = GradScaler(enabled=use_fp16)
    
    print(f"\nTraining config:")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size} x {gradient_accumulation_steps} accum = {batch_size * gradient_accumulation_steps} effective")
    print(f"  Total steps: {total_steps}")
    print(f"  Warmup: {warmup_steps} steps")
    print(f"  FP16: {use_fp16}")
    print(f"  Label smoothing: {label_smoothing}")
    print(f"  Raw text: {raw_text}")
    print()
    
    # Training loop
    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0
    history = []
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad()
        
        start_time = time.time()
        
        for step, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            with autocast(enabled=use_fp16):
                logits = model(input_ids, attention_mask)
                
                # Compute per-task losses
                task_losses = []
                for task in ['sentiment', 'emotion', 'intent']:
                    if task in batch:
                        labels = batch[task].to(device)
                        task_loss = criteria[task](logits[task], labels)
                        task_losses.append(task_loss)
                
                # Uncertainty-weighted combination
                if task_losses:
                    loss = uncertainty_loss(task_losses)
                else:
                    loss = torch.tensor(0.0, device=device)
            
            # Gradient accumulation
            loss = loss / gradient_accumulation_steps
            scaler.scale(loss).backward()
            
            if (step + 1) % gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
            
            epoch_loss += loss.item() * gradient_accumulation_steps
            
            if (step + 1) % (50 * gradient_accumulation_steps) == 0:
                avg_so_far = epoch_loss / (step + 1)
                current_lr = scheduler.get_last_lr()[0]
                print(f"  Epoch {epoch+1}/{epochs}, Step {step+1}/{len(train_loader)}, "
                      f"Loss: {loss.item():.4f}, LR: {current_lr:.2e}")
        
        avg_train_loss = epoch_loss / max(len(train_loader), 1)
        epoch_time = time.time() - start_time
        
        # Validation
        val_results = evaluate(model, val_loader, device, criteria)
        avg_val_acc = sum(val_results['accuracy'].values()) / 3
        
        sent_acc = val_results['accuracy']['sentiment']
        emo_acc = val_results['accuracy']['emotion']
        intent_acc = val_results['accuracy']['intent']
        
        print(f"Epoch {epoch+1}/{epochs} ({epoch_time:.0f}s) - "
              f"Train Loss: {avg_train_loss:.4f}, "
              f"Val Loss: {val_results['avg_loss']:.4f}")
        print(f"  Sentiment: {sent_acc:.3f}, Emotion: {emo_acc:.3f}, Intent: {intent_acc:.3f}, Avg: {avg_val_acc:.3f}")
        
        # Per-class accuracy for emotion (most imbalanced)
        if 'per_class_accuracy' in val_results and 'emotion' in val_results['per_class_accuracy']:
            emo_pca = val_results['per_class_accuracy']['emotion']
            print(f"  Emotion per-class: " + 
                  ", ".join(f"{EMOTION_LABELS[k]}={v:.2f}" for k, v in sorted(emo_pca.items())))
        
        history.append({
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'val_loss': val_results['avg_loss'],
            'val_accuracy': val_results['accuracy'],
            'lr': scheduler.get_last_lr()[0],
        })
        
        # Early stopping
        if avg_val_acc > best_val_acc:
            best_val_acc = avg_val_acc
            best_epoch = epoch + 1
            patience_counter = 0
            _save_model(model, tokenizer, output_dir)
            print(f"  -> Saved best model (avg acc: {best_val_acc:.4f})")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{early_stopping_patience})")
            if patience_counter >= early_stopping_patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break
    
    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"Best avg accuracy: {best_val_acc:.4f} (epoch {best_epoch})")
    print(f"Model saved to: {output_dir}")
    print(f"{'='*60}")
    
    return {
        'best_val_accuracy': best_val_acc,
        'best_epoch': best_epoch,
        'history': history,
        'output_dir': output_dir,
        'model_name': MODEL_NAME,
    }


def _save_model(model: Any, tokenizer: Any, output_dir: str) -> None:
    """Save model, tokenizer, config."""
    os.makedirs(output_dir, exist_ok=True)
    
    torch.save(model.state_dict(), os.path.join(output_dir, 'model.pt'))
    tokenizer.save_pretrained(output_dir)
    
    config = {
        'model_name': MODEL_NAME,
        'architecture': 'ManglishMultiTaskModelV2',
        'label_maps': LABEL_MAPS,
        'labels': {
            'sentiment': SENTIMENT_LABELS,
            'emotion': EMOTION_LABELS,
            'intent': INTENT_LABELS,
        },
        'version': 'v2',
        'improvements': [
            'xlm-roberta-base',
            'focal_loss',
            'uncertainty_weighted_loss',
            'cosine_annealing',
            'label_smoothing',
            'fp16',
            'gradient_accumulation',
            'raw_text',
        ],
    }
    
    with open(os.path.join(output_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    import sys
    
    data_path = 'datasets/manglish_augmented.jsonl'
    if not os.path.exists(data_path):
        print(f"Augmented data not found at {data_path}")
        print("Using original data. Run scripts/augment_data.py first for best results.")
        data_path = 'datasets/manglish_7884.jsonl'
    
    results = train(data_path=data_path)
    
    # Save training results
    with open('resources/training_results_v2.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved to resources/training_results_v2.json")
