import sys, re
sys.path.insert(0, '.')

text = "rindu gila kat family"
lower = text.lower()
pattern = r'rindu\s*(kat|dekat)?\s*(family|keluarga|mak|ayah|abang|kakak|adik|arwah|kampung)'
print(f"Text: {lower}")
print(f"Pattern match: {re.search(pattern, lower)}")

# Check what words module sees
words = set(re.findall(r'[a-zA-Z]+', lower))
print(f"Words: {words}")

# rindu is in love words, so love gets 1 * 1.2 weight * 1.5 intensifier = 1.8
# sad pattern should give 1.5
# love pattern 'rindu\s*(kau|ko|awak|dia|you|gila|sangat|sgt)' also matches!
love_pattern = r'rindu\s*(kau|ko|awak|dia|you|gila|sangat|sgt)'
print(f"Love pattern match: {re.search(love_pattern, lower)}")
