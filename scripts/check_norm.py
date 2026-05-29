import sys
sys.path.insert(0, '.')
from manglish_nlp import normalize

NORM_DATA = [
    ("nk pgi mkn", "nak pergi makan"),
    ("xde duit nk byr", "takde duit nak bayar"),
    ("aku tgh bz skrg", "aku tengah busy sekarang"),
    ("jgn lpe bwk bku", "jangan lupa bawa buku"),
    ("sy nk tnye skit", "saya nak tanya sikit"),
    ("dh smpi blm", "dah sampai belum"),
    ("xleh nk wt cmne", "tak boleh nak buat macam mana"),
    ("igt nk g jln2", "ingat nak pergi jalan-jalan"),
    ("tlg htr brg tu", "tolong hantar barang tu"),
    ("bpe hrge die", "berapa harga dia"),
    ("mmg x phm la", "memang tak faham la"),
    ("sbb tu la aku mls", "sebab tu la aku malas"),
]

for text, expected in NORM_DATA:
    result = normalize(text)
    status = "OK" if result == expected else "FAIL"
    if status == "FAIL":
        print(f'FAIL: "{text}"')
        print(f'  Expected: "{expected}"')
        print(f'  Got:      "{result}"')
        print()
