#!/usr/bin/env python3
"""
Batch processor for malaysian-manglish-nlp.
Process multiple texts from file or stdin.
Usage:
  python batch.py --input texts.txt --output results.json --tasks normalize,sentiment,lang
  echo "nk pergi makan" | python batch.py --tasks normalize
"""

import sys
import json
import argparse
import os
import csv
from io import StringIO

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from normalize import normalize
from detect_lang import detect_language
from sentiment import analyze_sentiment
from clean import clean_text, clean_for_nlp
from formalize import formalize
from segment import segment_text


AVAILABLE_TASKS = ['normalize', 'lang', 'sentiment', 'clean', 'clean_nlp', 'formalize', 'segment']


def process_single(text, tasks):
    """Process a single text with specified tasks."""
    result = {'original': text}
    
    if 'normalize' in tasks:
        result['normalized'] = normalize(text)
    
    if 'lang' in tasks:
        result['language'] = detect_language(text)
    
    if 'sentiment' in tasks:
        result['sentiment'] = analyze_sentiment(text)
    
    if 'clean' in tasks:
        result['cleaned'] = clean_text(text)
    
    if 'clean_nlp' in tasks:
        result['cleaned_nlp'] = clean_for_nlp(text)
    
    if 'formalize' in tasks:
        result['formalized'] = formalize(text)
    
    if 'segment' in tasks:
        seg = segment_text(text)
        result['segments'] = seg['segments']
        result['switch_count'] = seg['switch_count']
        result['dominant_lang'] = seg['dominant_lang']
    
    return result


def process_batch(texts, tasks, progress=False):
    """Process multiple texts."""
    results = []
    total = len(texts)
    
    for i, text in enumerate(texts):
        text = text.strip()
        if not text:
            continue
        
        result = process_single(text, tasks)
        results.append(result)
        
        if progress and (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{total}...", file=sys.stderr)
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Batch process Manglish text')
    parser.add_argument('--input', '-i', help='Input file (one text per line). Omit for stdin.')
    parser.add_argument('--output', '-o', help='Output file (JSON). Omit for stdout.')
    parser.add_argument('--tasks', '-t', default='normalize,lang,sentiment',
                       help=f'Comma-separated tasks: {",".join(AVAILABLE_TASKS)}')
    parser.add_argument('--format', '-f', choices=['json', 'jsonl', 'csv'], default='json',
                       help='Output format (default: json)')
    parser.add_argument('--progress', '-p', action='store_true',
                       help='Show progress to stderr')
    
    args = parser.parse_args()
    
    # Parse tasks
    tasks = [t.strip() for t in args.tasks.split(',')]
    invalid = [t for t in tasks if t not in AVAILABLE_TASKS]
    if invalid:
        print(f"Invalid tasks: {invalid}. Available: {AVAILABLE_TASKS}", file=sys.stderr)
        sys.exit(1)
    
    # Read input
    if args.input:
        with open(args.input, 'r', encoding='utf-8') as f:
            texts = f.readlines()
    else:
        texts = sys.stdin.readlines()
    
    if not texts:
        print("No input texts.", file=sys.stderr)
        sys.exit(1)
    
    if args.progress:
        print(f"Processing {len(texts)} texts with tasks: {tasks}", file=sys.stderr)
    
    # Process
    results = process_batch(texts, tasks, progress=args.progress)
    
    # Output
    if args.format == 'json':
        output = json.dumps(results, indent=2, ensure_ascii=False)
    elif args.format == 'jsonl':
        output = '\n'.join(json.dumps(r, ensure_ascii=False) for r in results)
    elif args.format == 'csv':
        if not results:
            output = ''
        else:
            # Flatten for CSV
            buf = StringIO()
            fieldnames = ['original']
            if 'normalize' in tasks:
                fieldnames.append('normalized')
            if 'lang' in tasks:
                fieldnames.extend(['lang', 'bm_ratio', 'en_ratio'])
            if 'sentiment' in tasks:
                fieldnames.extend(['sentiment', 'sentiment_score'])
            if 'clean' in tasks:
                fieldnames.append('cleaned')
            if 'formalize' in tasks:
                fieldnames.append('formalized')
            
            writer = csv.DictWriter(buf, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            
            for r in results:
                row = {'original': r['original']}
                if 'normalized' in r:
                    row['normalized'] = r['normalized']
                if 'language' in r:
                    row['lang'] = r['language']['language']
                    row['bm_ratio'] = r['language']['bm_ratio']
                    row['en_ratio'] = r['language']['en_ratio']
                if 'sentiment' in r:
                    row['sentiment'] = r['sentiment']['sentiment']
                    row['sentiment_score'] = r['sentiment']['score']
                if 'cleaned' in r:
                    row['cleaned'] = r['cleaned']
                if 'formalized' in r:
                    row['formalized'] = r['formalized']
                writer.writerow(row)
            
            output = buf.getvalue()
    
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(output)
        if args.progress:
            print(f"Results written to {args.output}", file=sys.stderr)
    else:
        print(output)


if __name__ == '__main__':
    main()
