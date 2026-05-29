#!/usr/bin/env python3
"""
manglish-nlp CLI — Full NLP toolkit for Malaysian Manglish.

Usage:
    manglish analyze "text"
    manglish sentiment "text"
    manglish normalize "text"
    manglish translate "text" --to en
    manglish ner "text"
    manglish pos "text"
    manglish summarize --file input.txt
    manglish benchmark
    manglish profile "text"
    manglish --version
    manglish --help

Supports stdin pipe: echo "text" | manglish sentiment
"""

from __future__ import annotations

from typing import Any

import sys
import os
import json
import time
import argparse

# ANSI color codes (no external deps)
class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    BG_GREEN = "\033[42m"
    BG_RED = "\033[41m"
    BG_YELLOW = "\033[43m"

    @classmethod
    def disable(cls) -> None:
        """Disable colors (for piped output)."""
        for attr in dir(cls):
            if attr.isupper() and not attr.startswith("_"):
                setattr(cls, attr, "")


# Disable colors if not a TTY
if not sys.stdout.isatty():
    Colors.disable()


def c(color: Any, text: str) -> str:
    """Colorize text."""
    return f"{color}{text}{Colors.RESET}"


def header(title: str) -> str:
    """Print a section header."""
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'=' * 50}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}  {title}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'=' * 50}{Colors.RESET}\n")


def label(name: str, value: Any) -> str:
    """Print a labeled value."""
    print(f"  {Colors.DIM}{name}:{Colors.RESET} {value}")


def print_json(data: Any, colored: bool = True) -> None:
    """Pretty print JSON with optional coloring."""
    output = json.dumps(data, indent=2, ensure_ascii=False)
    if colored and sys.stdout.isatty():
        # Simple syntax highlighting
        output = output.replace('"', f'{Colors.GREEN}"{Colors.RESET}')
    print(output)


def timed(func: Any, *args: Any, **kwargs: Any) -> Any:
    """Run function and return (result, elapsed_ms)."""
    start = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed = (time.perf_counter() - start) * 1000
    return result, elapsed


def get_text_from_args_or_stdin(args: Any) -> str:
    """Get text from args or stdin pipe."""
    if hasattr(args, 'text') and args.text:
        return ' '.join(args.text) if isinstance(args.text, list) else args.text
    if not sys.stdin.isatty():
        return sys.stdin.read().strip()
    print(c(Colors.RED, "Error: No text provided. Pass text as argument or pipe via stdin."))
    sys.exit(1)


def cmd_analyze(args: Any) -> None:
    """Run all modules on text."""
    import malaysian_manglish_nlp

    text = get_text_from_args_or_stdin(args)
    header("manglish-nlp Full Analysis")
    label("Input", c(Colors.WHITE, text))
    print()

    # Normalize
    result, ms = timed(malaysian_manglish_nlp.normalize, text)
    print(f"  {c(Colors.BOLD, 'Normalized:')}  {result} {c(Colors.DIM, f'({ms:.1f}ms)')}")

    # Language detection
    result, ms = timed(malaysian_manglish_nlp.detect_language, text)
    lang = result if isinstance(result, str) else result.get('language', str(result))
    print(f"  {c(Colors.BOLD, 'Language:')}    {lang} {c(Colors.DIM, f'({ms:.1f}ms)')}")

    # Sentiment
    result, ms = timed(malaysian_manglish_nlp.sentiment, text)
    sent_label = result.get('sentiment', result.get('label', ''))
    score = result.get('score', result.get('confidence', 0))
    color = Colors.GREEN if sent_label == 'positive' else Colors.RED if sent_label == 'negative' else Colors.YELLOW
    print(f"  {c(Colors.BOLD, 'Sentiment:')}   {c(color, sent_label)} (score: {score:.2f}) {c(Colors.DIM, f'({ms:.1f}ms)')}")

    # POS tags
    result, ms = timed(malaysian_manglish_nlp.pos_tag, text)
    tags_str = ' '.join([f"{w}/{t}" for w, t in result])
    print(f"  {c(Colors.BOLD, 'POS Tags:')}    {tags_str} {c(Colors.DIM, f'({ms:.1f}ms)')}")

    # NER
    result, ms = timed(malaysian_manglish_nlp.ner_tag, text)
    if result:
        ents = ', '.join([f"{e['text']}[{e['type']}]" for e in result])
        print(f"  {c(Colors.BOLD, 'Entities:')}    {ents} {c(Colors.DIM, f'({ms:.1f}ms)')}")
    else:
        print(f"  {c(Colors.BOLD, 'Entities:')}    {c(Colors.DIM, 'None found')} {c(Colors.DIM, f'({ms:.1f}ms)')}")

    # Emotion
    result, ms = timed(malaysian_manglish_nlp.detect_emotion, text)
    emo = result.get('emotion', result.get('primary', ''))
    print(f"  {c(Colors.BOLD, 'Emotion:')}     {emo} {c(Colors.DIM, f'({ms:.1f}ms)')}")

    # Keywords
    result, ms = timed(malaysian_manglish_nlp.extract_keywords, text)
    if result:
        kws = ', '.join([kw['keyword'] for kw in result[:5]])
        print(f"  {c(Colors.BOLD, 'Keywords:')}    {kws} {c(Colors.DIM, f'({ms:.1f}ms)')}")

    print()


def cmd_sentiment(args: Any) -> None:
    """Sentiment analysis."""
    import malaysian_manglish_nlp

    text = get_text_from_args_or_stdin(args)
    result, ms = timed(malaysian_manglish_nlp.sentiment, text)

    if sys.stdout.isatty():
        sent = result.get('sentiment', result.get('label', ''))
        score = result.get('score', result.get('confidence', 0))
        color = Colors.GREEN if sent == 'positive' else Colors.RED if sent == 'negative' else Colors.YELLOW
        print(f"{c(Colors.BOLD, 'Sentiment:')} {c(color, sent)}")
        print(f"{c(Colors.BOLD, 'Score:')}     {score:.3f}")
        print(c(Colors.DIM, f"({ms:.1f}ms)"))
    else:
        print(json.dumps(result, ensure_ascii=False))


def cmd_normalize(args: Any) -> None:
    """Normalize shortforms."""
    import malaysian_manglish_nlp

    text = get_text_from_args_or_stdin(args)
    result, ms = timed(malaysian_manglish_nlp.normalize, text)

    if sys.stdout.isatty():
        print(f"{c(Colors.DIM, 'Input:')}  {text}")
        print(f"{c(Colors.BOLD, 'Output:')} {c(Colors.GREEN, result)}")
        print(c(Colors.DIM, f"({ms:.1f}ms)"))
    else:
        print(result)


def cmd_translate(args: Any) -> None:
    """Translate text."""
    import malaysian_manglish_nlp

    text = get_text_from_args_or_stdin(args)
    target = args.to if hasattr(args, 'to') and args.to else 'en'

    if target == 'en':
        result, ms = timed(malaysian_manglish_nlp.to_english, text)
    elif target in ('bm', 'ms', 'malay'):
        result, ms = timed(malaysian_manglish_nlp.to_malay, text)
    elif target == 'formal':
        result, ms = timed(malaysian_manglish_nlp.to_formal, text)
    else:
        result, ms = timed(malaysian_manglish_nlp.translate, text)

    if sys.stdout.isatty():
        print(f"{c(Colors.DIM, 'Input:')}  {text}")
        print(f"{c(Colors.BOLD, f'[{target}]:')}   {c(Colors.GREEN, result)}")
        print(c(Colors.DIM, f"({ms:.1f}ms)"))
    else:
        if isinstance(result, dict):
            print(json.dumps(result, ensure_ascii=False))
        else:
            print(result)


def cmd_ner(args: Any) -> None:
    """Named Entity Recognition."""
    import malaysian_manglish_nlp

    text = get_text_from_args_or_stdin(args)
    result, ms = timed(malaysian_manglish_nlp.ner_tag, text)

    if sys.stdout.isatty():
        if result:
            max_type_len = max(len(e['type']) for e in result)
            for e in result:
                type_color = {
                    'PERSON': Colors.CYAN,
                    'LOCATION': Colors.GREEN,
                    'ORGANIZATION': Colors.MAGENTA,
                    'MONEY': Colors.YELLOW,
                    'DATE': Colors.BLUE,
                }.get(e['type'], Colors.WHITE)
                print(f"  {c(type_color, e['type'].ljust(max_type_len))}  {e['text']}")
            print(c(Colors.DIM, f"\n({len(result)} entities, {ms:.1f}ms)"))
        else:
            print(c(Colors.DIM, f"No entities found. ({ms:.1f}ms)"))
    else:
        print(json.dumps(result, ensure_ascii=False))


def cmd_pos(args: Any) -> None:
    """Part-of-Speech tagging."""
    import malaysian_manglish_nlp

    text = get_text_from_args_or_stdin(args)
    result, ms = timed(malaysian_manglish_nlp.pos_tag, text)

    if sys.stdout.isatty():
        for word, tag in result:
            tag_color = {
                'NOUN': Colors.CYAN,
                'VERB': Colors.GREEN,
                'ADJ': Colors.YELLOW,
                'ADV': Colors.MAGENTA,
                'PRON': Colors.BLUE,
            }.get(tag, Colors.WHITE)
            print(f"  {word:15s} {c(tag_color, tag)}")
        print(c(Colors.DIM, f"\n({ms:.1f}ms)"))
    else:
        print(json.dumps(result, ensure_ascii=False))


def cmd_summarize(args: Any) -> None:
    """Summarize text or file."""
    import malaysian_manglish_nlp

    if hasattr(args, 'file') and args.file:
        try:
            with open(args.file, 'r', encoding='utf-8') as f:
                text = f.read()
        except FileNotFoundError:
            print(c(Colors.RED, f"Error: File not found: {args.file}"))
            sys.exit(1)
        except IOError as e:
            print(c(Colors.RED, f"Error reading file: {e}"))
            sys.exit(1)
    else:
        text = get_text_from_args_or_stdin(args)

    result, ms = timed(malaysian_manglish_nlp.summarize, text)

    if sys.stdout.isatty():
        print(f"{c(Colors.BOLD, 'Summary:')}")
        if isinstance(result, dict):
            print(f"  {result.get('summary', result)}")
        else:
            print(f"  {result}")
        print(c(Colors.DIM, f"\n({ms:.1f}ms)"))
    else:
        if isinstance(result, dict):
            print(json.dumps(result, ensure_ascii=False))
        else:
            print(result)


def cmd_benchmark(args: Any) -> None:
    """Run benchmark suite."""
    import malaysian_manglish_nlp

    header("manglish-nlp Benchmark")

    test_texts = [
        "aku nk pgi mkn kat kedai tu",
        "gila best makanan dia bro",
        "Jumpa kat KLCC esok bayar RM50",
        "sedih gila dengar berita tu tapi kena move on la kan",
        "weh kau dah siap assignment belum? deadline esok pagi",
    ]

    modules = [
        ("normalize", malaysian_manglish_nlp.normalize),
        ("sentiment", malaysian_manglish_nlp.sentiment),
        ("pos_tag", malaysian_manglish_nlp.pos_tag),
        ("ner_tag", malaysian_manglish_nlp.ner_tag),
        ("detect_language", malaysian_manglish_nlp.detect_language),
        ("detect_emotion", malaysian_manglish_nlp.detect_emotion),
        ("extract_keywords", malaysian_manglish_nlp.extract_keywords),
        ("tokenize", malaysian_manglish_nlp.tokenize),
        ("stem", malaysian_manglish_nlp.stem),
        ("clean", malaysian_manglish_nlp.clean),
    ]

    results = []
    for name, func in modules:
        times = []
        for text in test_texts:
            try:
                _, ms = timed(func, text)
                times.append(ms)
            except Exception:
                times.append(-1)

        avg = sum(t for t in times if t >= 0) / max(len([t for t in times if t >= 0]), 1)
        results.append((name, avg))

        speed_color = Colors.GREEN if avg < 5 else Colors.YELLOW if avg < 20 else Colors.RED
        bar_len = min(int(avg / 2), 40)
        bar = "█" * bar_len
        print(f"  {name:20s} {c(speed_color, f'{avg:6.2f}ms')} {c(Colors.DIM, bar)}")

    total_avg = sum(r[1] for r in results) / len(results)
    print(f"\n  {c(Colors.BOLD, 'Average:')}           {total_avg:.2f}ms per module")
    print(f"  {c(Colors.BOLD, 'Texts tested:')}      {len(test_texts)}")
    print(f"  {c(Colors.BOLD, 'Modules tested:')}    {len(modules)}")
    print()


def cmd_profile(args: Any) -> None:
    """Performance profile a text."""
    import malaysian_manglish_nlp

    text = get_text_from_args_or_stdin(args)
    header("Performance Profile")
    label("Input", text)
    print()

    modules = [
        ("normalize", malaysian_manglish_nlp.normalize),
        ("sentiment", malaysian_manglish_nlp.sentiment),
        ("pos_tag", malaysian_manglish_nlp.pos_tag),
        ("ner_tag", malaysian_manglish_nlp.ner_tag),
        ("detect_language", malaysian_manglish_nlp.detect_language),
        ("detect_emotion", malaysian_manglish_nlp.detect_emotion),
        ("extract_keywords", malaysian_manglish_nlp.extract_keywords),
        ("tokenize", malaysian_manglish_nlp.tokenize),
        ("stem", malaysian_manglish_nlp.stem),
        ("clean", malaysian_manglish_nlp.clean),
        ("formalize", malaysian_manglish_nlp.formalize),
        ("segment", malaysian_manglish_nlp.segment),
    ]

    total_ms = 0
    for name, func in modules:
        try:
            result, ms = timed(func, text)
            total_ms += ms
            speed_color = Colors.GREEN if ms < 5 else Colors.YELLOW if ms < 20 else Colors.RED
            status = c(Colors.GREEN, "OK")
            print(f"  {status} {name:20s} {c(speed_color, f'{ms:6.2f}ms')}")
        except Exception as e:
            print(f"  {c(Colors.RED, 'FAIL')} {name:20s} {c(Colors.RED, str(e)[:40])}")

    print(f"\n  {c(Colors.BOLD, 'Total:')} {total_ms:.2f}ms")
    print(f"  {c(Colors.BOLD, 'Throughput:')} ~{1000/max(total_ms, 0.01):.0f} texts/sec (full pipeline)")
    print()


def cmd_version(args: Any = None) -> None:
    """Show version."""
    import malaysian_manglish_nlp
    version = getattr(malaysian_manglish_nlp, '__version__', 'unknown')
    print(f"manglish-nlp v{version}")


def build_parser() -> Any:
    """Build the argument parser with subcommands."""
    parser = argparse.ArgumentParser(
        prog='manglish',
        description='manglish-nlp: Full NLP toolkit for Malaysian Manglish',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  manglish analyze "aku nk pgi mkn"
  manglish sentiment "gila best bro"
  manglish normalize "nk tnya brp"
  manglish translate "aku lapar" --to en
  echo "text here" | manglish sentiment
  manglish summarize --file article.txt
  manglish benchmark
        """,
    )
    parser.add_argument('--version', '-v', action='store_true', help='Show version')

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # analyze
    p = subparsers.add_parser('analyze', help='Run all modules, pretty output')
    p.add_argument('text', nargs='*', help='Text to analyze')

    # sentiment
    p = subparsers.add_parser('sentiment', help='Sentiment analysis')
    p.add_argument('text', nargs='*', help='Text to analyze')

    # normalize
    p = subparsers.add_parser('normalize', help='Normalize shortforms')
    p.add_argument('text', nargs='*', help='Text to normalize')

    # translate
    p = subparsers.add_parser('translate', help='Translate text')
    p.add_argument('text', nargs='*', help='Text to translate')
    p.add_argument('--to', default='en', choices=['en', 'bm', 'ms', 'malay', 'formal'],
                   help='Target language (default: en)')

    # ner
    p = subparsers.add_parser('ner', help='Named Entity Recognition')
    p.add_argument('text', nargs='*', help='Text to extract entities from')

    # pos
    p = subparsers.add_parser('pos', help='Part-of-Speech tagging')
    p.add_argument('text', nargs='*', help='Text to tag')

    # summarize
    p = subparsers.add_parser('summarize', help='Summarize text or file')
    p.add_argument('text', nargs='*', help='Text to summarize')
    p.add_argument('--file', '-f', help='File to summarize')

    # benchmark
    subparsers.add_parser('benchmark', help='Run benchmark suite')

    # profile
    p = subparsers.add_parser('profile', help='Performance profile')
    p.add_argument('text', nargs='*', help='Text to profile')

    return parser


def main() -> None:
    """Main entry point."""
    parser = build_parser()

    # Handle --version before subcommand parsing
    if '--version' in sys.argv or '-v' in sys.argv:
        cmd_version()
        sys.exit(0)

    args = parser.parse_args()

    if not args.command:
        # Check if stdin has data
        if not sys.stdin.isatty():
            # Default to analyze for piped input
            args.command = 'analyze'
            args.text = None
        else:
            parser.print_help()
            sys.exit(0)

    commands = {
        'analyze': cmd_analyze,
        'sentiment': cmd_sentiment,
        'normalize': cmd_normalize,
        'translate': cmd_translate,
        'ner': cmd_ner,
        'pos': cmd_pos,
        'summarize': cmd_summarize,
        'benchmark': cmd_benchmark,
        'profile': cmd_profile,
    }

    handler = commands.get(args.command)
    if handler:
        try:
            handler(args)
        except KeyboardInterrupt:
            print(c(Colors.DIM, "\nInterrupted."))
            sys.exit(130)
        except Exception as e:
            print(c(Colors.RED, f"Error: {e}"))
            sys.exit(1)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
