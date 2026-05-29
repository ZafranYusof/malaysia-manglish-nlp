import sys
sys.path.insert(0, '.')
from malaysian_manglish_nlp import sentiment

SENTIMENT_DATA = [
    ("best gila makanan dia", "positive"),
    ("sedap sangat nasi goreng tu", "positive"),
    ("power la presentation kau", "positive"),
    ("terbaik service kat sini", "positive"),
    ("syok gila concert semalam", "positive"),
    ("mantap betul design dia", "positive"),
    ("gempak la event tu", "positive"),
    ("padu gila skills dia", "positive"),
    ("solid la team ni", "positive"),
    ("legend betul mamat ni", "positive"),
    ("cantik gila view kat sini", "positive"),
    ("seronok sangat holiday kali ni", "positive"),
    ("enjoy betul aku hari ni", "positive"),
    ("grateful sangat dapat kawan macam ni", "positive"),
    ("tiptop la condition kereta tu", "positive"),
    ("lega sangat dah settle semua", "positive"),
    ("bangga gila tengok result dia", "positive"),
    ("puas hati la dengan outcome", "positive"),
    ("smooth je process dia", "positive"),
    ("on point la outfit dia hari ni", "positive"),
    ("mahal gila tapi tak sedap", "negative"),
    ("lambat gila delivery dia", "negative"),
    ("tak best langsung movie tu", "negative"),
    ("menyesal beli phone ni", "negative"),
    ("terrible la wifi kat sini", "negative"),
    ("frustrated gila assignment ni", "negative"),
    ("sampah la game ni", "negative"),
    ("rugi je bayar mahal", "negative"),
    ("hancur la plan aku", "negative"),
    ("stress gila kerja ni", "negative"),
    ("aku pergi kedai kejap", "neutral"),
    ("dia balik rumah dah", "neutral"),
    ("esok ada meeting pukul 3", "neutral"),
    ("aku nak makan nasi goreng", "neutral"),
    ("kelas start pukul 8 pagi", "neutral"),
    ("dia kerja kat KL sekarang", "neutral"),
    ("aku baru sampai rumah", "neutral"),
    ("hujan kat luar sekarang", "neutral"),
    ("aku tunggu kat depan", "neutral"),
    ("dia hantar mesej tadi", "neutral"),
]

for text, expected in SENTIMENT_DATA:
    result = sentiment(text)
    if result['sentiment'] != expected:
        print(f'FAIL: "{text}"')
        print(f'  Expected: {expected}, Got: {result["sentiment"]}')
        print(f'  Score: {result["score"]}, Pos: {result["positive_words"]}, Neg: {result["negative_words"]}')
        print()
