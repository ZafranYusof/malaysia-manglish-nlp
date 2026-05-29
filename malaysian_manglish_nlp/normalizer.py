"""Advanced normalizer for Malaysian text.

Handles elongated text, numbers, dates, money, phone numbers, URLs.
Inspired by malaya.normalizer.rules.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import re


def normalize_elongated(text: str) -> str:
    """Normalize elongated/repeated characters.
    
    Args:
        text: Input text with elongated words.
    
    Returns:
        str: Normalized text.
    
    Example:
        >>> normalize_elongated("bestttttt gilaaaa sedapppp")
        'best gila sedap'
        >>> normalize_elongated("hahahaha wkwkwk")
        'haha wkwk'
    """
    # Reduce 3+ repeated chars to 1
    result = re.sub(r'(.)\1{2,}', r'\1', text)
    
    # Normalize laugh patterns
    result = re.sub(r'\bhaha(ha)+\b', 'haha', result, flags=re.IGNORECASE)
    result = re.sub(r'\bwkwk(wk)+\b', 'wkwk', result, flags=re.IGNORECASE)
    result = re.sub(r'\blol(ol)+\b', 'lol', result, flags=re.IGNORECASE)
    result = re.sub(r'\bxixi(xi)+\b', 'xixi', result, flags=re.IGNORECASE)
    
    return result


def normalize_money(text: str) -> str:
    """Normalize money expressions to standard form.
    
    Args:
        text: Input text.
    
    Returns:
        str: Text with normalized money expressions.
    
    Example:
        >>> normalize_money("harga dia rm50 je")
        'harga dia RM50.00 je'
        >>> normalize_money("bayar 3.5k")
        'bayar RM3500.00'
    """
    # RM/rm + number
    def format_rm(match: Any) -> str:
        """Format ringgit amount to standard form.

        Args:
            match: Regex match object.

        Returns:
            Processed text string.

        """
        amount = match.group(1).replace(',', '')
        try:
            val = float(amount)
            return f'RM{val:.2f}'
        except ValueError:
            return match.group(0)
    
    result = re.sub(r'(?i)rm\s?([\d,]+\.?\d*)', format_rm, text)
    
    # Number + k/K (thousands)
    def format_k(match: Any) -> str:
        """Format k-suffix number to ringgit amount.

        Args:
            match: Regex match object.

        Returns:
            Processed text string.

        """
        num = float(match.group(1))
        return f'RM{num * 1000:.2f}'
    
    result = re.sub(r'(\d+\.?\d*)\s*[kK]\b', format_k, result)
    
    # Number + juta/million
    def format_juta(match: Any) -> str:
        """Format juta/million number to ringgit amount.

        Args:
            match: Regex match object.

        Returns:
            Processed text string.

        """
        num = float(match.group(1))
        return f'RM{num * 1000000:.2f}'
    
    result = re.sub(r'(\d+\.?\d*)\s*(?:juta|million)\b', format_juta, result, flags=re.IGNORECASE)
    
    return result


def normalize_phone(text: str) -> str:
    """Normalize Malaysian phone numbers to standard format.
    
    Args:
        text: Input text.
    
    Returns:
        str: Text with normalized phone numbers.
    
    Example:
        >>> normalize_phone("call 0123456789")
        'call 012-345 6789'
        >>> normalize_phone("whatsapp 60123456789")
        'whatsapp +60 12-345 6789'
    """
    def format_phone(match: Any) -> str:
        """Format phone number to standard form.

        Args:
            match: Regex match object.

        Returns:
            Processed text string.

        """
        digits = re.sub(r'[\s\-]', '', match.group(0))
        
        # Remove leading +
        if digits.startswith('+'):
            digits = digits[1:]
        
        # Handle 60 prefix
        if digits.startswith('60') and len(digits) >= 11:
            area = digits[2:4]
            rest = digits[4:]
            return f'+60 {area}-{rest[:3]} {rest[3:]}'
        elif digits.startswith('0') and len(digits) >= 10:
            area = digits[:3]
            rest = digits[3:]
            return f'{area}-{rest[:3]} {rest[3:]}'
        
        return match.group(0)
    
    result = re.sub(r'\+?6?0\d[\d\s\-]{8,12}', format_phone, text)
    return result


def normalize_date(text: str) -> str:
    """Normalize date expressions.
    
    Args:
        text: Input text.
    
    Returns:
        str: Text with normalized dates.
    
    Example:
        >>> normalize_date("jumpa 28/5/2026")
        'jumpa 28 Mei 2026'
    """
    months_bm = {
        '1': 'Januari', '01': 'Januari',
        '2': 'Februari', '02': 'Februari',
        '3': 'Mac', '03': 'Mac',
        '4': 'April', '04': 'April',
        '5': 'Mei', '05': 'Mei',
        '6': 'Jun', '06': 'Jun',
        '7': 'Julai', '07': 'Julai',
        '8': 'Ogos', '08': 'Ogos',
        '9': 'September', '09': 'September',
        '10': 'Oktober',
        '11': 'November',
        '12': 'Disember',
    }
    
    def format_date(match: Any) -> str:
        """Format date to standard form.

        Args:
            match: Regex match object.

        Returns:
            Processed text string.

        """
        day = match.group(1)
        month = match.group(2)
        year = match.group(3)
        
        month_name = months_bm.get(month, month)
        
        # Handle 2-digit year
        if len(year) == 2:
            year = '20' + year if int(year) < 50 else '19' + year
        
        return f'{int(day)} {month_name} {year}'
    
    # DD/MM/YYYY or DD-MM-YYYY
    result = re.sub(r'(\d{1,2})[/\-](\d{1,2})[/\-](\d{2,4})', format_date, text)
    
    return result


def normalize_time(text: str) -> str:
    """Normalize time expressions.
    
    Args:
        text: Input text.
    
    Returns:
        str: Text with normalized time.
    
    Example:
        >>> normalize_time("jumpa pukul 3pm")
        'jumpa pukul 3:00 PM'
        >>> normalize_time("meeting 1430")
        'meeting 2:30 PM'
    """
    # 3pm, 3.30pm, 3:30pm
    def format_time(match: Any) -> str:
        """Format time to 12-hour form.

        Args:
            match: Regex match object.

        Returns:
            Processed text string.

        """
        hour = int(match.group(1))
        minute = match.group(2) or '00'
        period = match.group(3).upper()
        return f'{hour}:{minute} {period}'
    
    result = re.sub(r'(\d{1,2})[:\.]?(\d{2})?\s*(am|pm|AM|PM)', format_time, text)
    
    # 24-hour format (1430 -> 2:30 PM)
    def format_24h(match: Any) -> str:
        """Convert 24-hour time to 12-hour form.

        Args:
            match: Regex match object.

        Returns:
            Processed text string.

        """
        hour = int(match.group(1))
        minute = match.group(2)
        period = 'AM' if hour < 12 else 'PM'
        display_hour = hour
        if hour > 12:
            display_hour = hour - 12
        elif hour == 0:
            display_hour = 12
        return f'{display_hour}:{minute} {period}'
    
    # Match 4-digit time patterns: 0000-2359
    result = re.sub(r'(?<![/\d])\b(0\d|1\d|2[0-3])([0-5]\d)\b(?!\s*(?:tahun|year|[/\-]))', format_24h, result)
    
    return result


def normalize_number(text: str) -> str:
    """Normalize number expressions to words (BM).
    
    Args:
        text: Input text.
    
    Returns:
        str: Text with numbers converted to BM words.
    
    Example:
        >>> normalize_number("ada 3 orang")
        'ada tiga orang'
    """
    num_words = {
        '0': 'sifar', '1': 'satu', '2': 'dua', '3': 'tiga',
        '4': 'empat', '5': 'lima', '6': 'enam', '7': 'tujuh',
        '8': 'lapan', '9': 'sembilan', '10': 'sepuluh',
        '11': 'sebelas', '12': 'dua belas', '20': 'dua puluh',
        '50': 'lima puluh', '100': 'seratus', '1000': 'seribu',
    }
    
    def replace_num(match: Any) -> str:
        """Replace number with BM word equivalent.

        Args:
            match: Regex match object.

        Returns:
            Processed text string.

        """
        num = match.group(0)
        if num in num_words:
            return num_words[num]
        return num  # Keep as-is if not in simple map
    
    # Only replace standalone numbers (not part of dates/phone/money)
    result = re.sub(r'\b(\d{1,4})\b', replace_num, text)
    return result


def normalize_url(text: str) -> str:
    """Normalize URLs to readable form.
    
    Args:
        text: Input text.
    
    Returns:
        str: Text with URLs simplified.
    
    Example:
        >>> normalize_url("check https://www.google.com/search?q=test")
        'check google.com'
    """
    def simplify_url(match: Any) -> str:
        """Simplify url.

        Args:
            match: Regex match object.

        Returns:
            Processed text string.

        """
        url = match.group(0)
        # Extract domain
        domain_match = re.search(r'(?:https?://)?(?:www\.)?([^/\s?]+)', url)
        if domain_match:
            return domain_match.group(1)
        return url
    
    result = re.sub(r'https?://\S+', simplify_url, text)
    return result


def normalize_all(text: str, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Apply all normalizations.
    
    Args:
        text: Input text.
        options: Which normalizations to apply. Default: all True.
            Keys: elongated, money, phone, date, time, number, url
    
    Returns:
        dict: Result with 'normalized' text and 'changes' list.
    
    Example:
        >>> normalize_all("besttt rm50 call 0123456789 jumpa 28/5/26 pukul 3pm")
        {'normalized': 'best RM50.00 call 012-345 6789 jumpa 28 Mei 2026 pukul 3:00 PM', ...}
    """
    if options is None:
        options = {
            'elongated': True, 'money': True, 'phone': True,
            'date': True, 'time': True, 'url': True,
            'number': False,  # Off by default (can be destructive)
        }
    
    result = text
    changes = []
    
    if options.get('elongated', True):
        new = normalize_elongated(result)
        if new != result:
            changes.append('elongated')
        result = new
    
    if options.get('url', True):
        new = normalize_url(result)
        if new != result:
            changes.append('url')
        result = new
    
    if options.get('money', True):
        new = normalize_money(result)
        if new != result:
            changes.append('money')
        result = new
    
    if options.get('phone', True):
        new = normalize_phone(result)
        if new != result:
            changes.append('phone')
        result = new
    
    if options.get('date', True):
        new = normalize_date(result)
        if new != result:
            changes.append('date')
        result = new
    
    if options.get('time', True):
        new = normalize_time(result)
        if new != result:
            changes.append('time')
        result = new
    
    if options.get('number', False):
        new = normalize_number(result)
        if new != result:
            changes.append('number')
        result = new
    
    return {
        'normalized': result,
        'original': text,
        'changes': changes,
    }
