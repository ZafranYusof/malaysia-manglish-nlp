"""
Ensemble module combining fine-tuned model + rule-based fallback.

Uses confidence threshold: if model <60% confidence, uses rule-based fallback.
Integrates into manglish_model.py predict() pipeline.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import json
import os


CONFIDENCE_THRESHOLD = 0.60  # Below this, use rule-based fallback
PARTIAL_THRESHOLD = 0.70     # Below this, blend model + rule-based


def _rule_predict(text: str) -> Dict[str, Any]:
    """Rule-based prediction (from manglish_model.py demo_predict)."""
    from malaysian_manglish_nlp.transformers.manglish_model import demo_predict
    return demo_predict(text)


def _blend_predictions(model_pred: Dict[str, Any], rule_pred: Dict[str, Any],
                       model_confidence: float) -> Dict[str, Any]:
    """Blend model and rule-based predictions based on confidence."""
    blended = {}
    
    for task in ['sentiment', 'emotion', 'intent']:
        m_pred = model_pred.get(task, {})
        r_pred = rule_pred.get(task, {})
        
        m_conf = m_pred.get('confidence', 0.5)
        r_conf = r_pred.get('confidence', 0.5)
        
        # If model agrees with rule-based, boost confidence
        if m_pred.get('label') == r_pred.get('label'):
            blended[task] = {
                'label': m_pred.get('label'),
                'confidence': min(m_conf * 1.1, 0.99),
            }
        else:
            # Weight by confidence
            total_conf = m_conf + r_conf
            if total_conf > 0:
                m_weight = m_conf / total_conf
                r_weight = r_conf / total_conf
            else:
                m_weight = 0.5
                r_weight = 0.5
            
            if m_weight > r_weight:
                blended[task] = {
                    'label': m_pred.get('label'),
                    'confidence': round(m_conf * 0.85, 4),
                }
            else:
                blended[task] = {
                    'label': r_pred.get('label'),
                    'confidence': round(r_conf * 0.85, 4),
                }
    
    blended['_ensemble'] = True
    blended['_model_confidence'] = model_confidence
    return blended


def ensemble_predict(text: str, model_pred: Dict[str, Any],
                     threshold: float = CONFIDENCE_THRESHOLD) -> Dict[str, Any]:
    """Apply ensemble logic to model predictions.
    
    Args:
        text: Original input text.
        model_pred: Predictions from fine-tuned model.
        threshold: Confidence threshold for fallback.
    
    Returns:
        Final predictions with ensemble logic applied.
    """
    # Compute average model confidence across tasks
    confidences = []
    for task in ['sentiment', 'emotion', 'intent']:
        if task in model_pred:
            confidences.append(model_pred[task].get('confidence', 0.5))
    
    avg_confidence = sum(confidences) / max(len(confidences), 1)
    
    # High confidence: trust model entirely
    if avg_confidence >= PARTIAL_THRESHOLD:
        model_pred['_ensemble'] = False
        return model_pred
    
    # Low confidence: full rule-based fallback
    if avg_confidence < threshold:
        rule_pred = _rule_predict(text)
        result = {}
        for task in ['sentiment', 'emotion', 'intent']:
            r = rule_pred.get(task, {})
            result[task] = {
                'label': r.get('label', model_pred.get(task, {}).get('label')),
                'confidence': r.get('confidence', 0.5),
            }
        result['_ensemble'] = True
        result['_fallback'] = 'rule_based'
        result['_model_confidence'] = avg_confidence
        return result
    
    # Medium confidence: blend predictions
    rule_pred = _rule_predict(text)
    return _blend_predictions(model_pred, rule_pred, avg_confidence)


def ensemble_predict_batch(texts: list, model_preds: list,
                           threshold: float = CONFIDENCE_THRESHOLD) -> list:
    """Apply ensemble to batch predictions."""
    results = []
    for text, pred in zip(texts, model_preds):
        results.append(ensemble_predict(text, pred, threshold))
    return results
