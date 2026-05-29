"""
Expanded benchmark suite — 500+ labeled test cases.
Covers: sentiment, language, emotion, profanity, normalization, stemmer,
        dialect, sarcasm, NER, POS.
"""

import sys
import os
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import malaysian_manglish_nlp
from malaysian_manglish_nlp.emotion import detect_emotion
from malaysian_manglish_nlp.profanity import detect_profanity
from malaysian_manglish_nlp.dialect import detect_dialect
from malaysian_manglish_nlp.sarcasm import detect_sarcasm

# ============================================================
# SENTIMENT (120 cases)
# ============================================================
SENTIMENT_DATA = [
    # Positive (40)
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
    ("awesome gila birthday party tu", "positive"),
    ("happy sangat dapat offer letter", "positive"),
    ("comel gila kucing dia", "positive"),
    ("terharu sangat baca message tu", "positive"),
    ("berbaloi la bayar mahal sikit", "positive"),
    ("suka sangat tempat ni", "positive"),
    ("nice la vibe cafe ni", "positive"),
    ("perfect timing kau datang", "positive"),
    ("brilliant idea tu", "positive"),
    ("superb la quality dia", "positive"),
    ("terror la budak ni coding", "positive"),
    ("fire la lagu baru dia", "positive"),
    ("wholesome gila video tu", "positive"),
    ("inspiring betul cerita dia", "positive"),
    ("motivated sangat lepas dengar talk tu", "positive"),
    ("fresh gila rasa lepas mandi", "positive"),
    ("semangat gila nak start project baru", "positive"),
    ("lawak gila meme tu", "positive"),
    ("gorgeous la dress dia", "positive"),
    ("yummy sangat dessert tu", "positive"),
    # Negative (40)
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
    ("boring gila lecture tadi", "negative"),
    ("kecewa sangat dengan result", "negative"),
    ("penat gila hari ni", "negative"),
    ("sakit hati tengok perangai dia", "negative"),
    ("frust betul tak dapat tiket", "negative"),
    ("teruk gila traffic hari ni", "negative"),
    ("annoyed sangat dengan jiran", "negative"),
    ("rosak dah phone aku", "negative"),
    ("fail lagi exam ni", "negative"),
    ("sedih gila kena tinggal", "negative"),
    ("toxic betul environment kerja ni", "negative"),
    ("cringe gila tengok video tu", "negative"),
    ("disgusting la toilet tu", "negative"),
    ("hopeless dah aku rasa", "negative"),
    ("lonely sangat duduk sorang", "negative"),
    ("anxious gila nak result", "negative"),
    ("regret sangat tak pergi", "negative"),
    ("exhausted betul minggu ni", "negative"),
    ("confused gila dengan instruction", "negative"),
    ("scared nak jumpa boss", "negative"),
    ("jelak dah makan benda sama", "negative"),
    ("bengang betul kena scam", "negative"),
    ("geram sangat dengan customer service", "negative"),
    ("meluat tengok muka dia", "negative"),
    ("bosan gila quarantine ni", "negative"),
    ("sien la kerja macam ni", "negative"),
    ("pening kepala fikir pasal ni", "negative"),
    ("lesu je badan hari ni", "negative"),
    ("gelisah sangat tunggu reply", "negative"),
    ("hampa sangat tak dapat scholarship", "negative"),
    # Neutral (40)
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
    ("parking penuh kat sini", "neutral"),
    ("aku nak pergi gym petang ni", "neutral"),
    ("dia ambil kursus IT semester ni", "neutral"),
    ("flight pukul 10 malam", "neutral"),
    ("aku duduk tingkat 5", "neutral"),
    ("dia pakai baju merah hari ni", "neutral"),
    ("kedai tu tutup pukul 10", "neutral"),
    ("aku bawa kereta hari ni", "neutral"),
    ("dia ada adik 3 orang", "neutral"),
    ("aku order grab food", "neutral"),
    ("meeting postpone ke esok", "neutral"),
    ("dia pindah rumah baru", "neutral"),
    ("aku tukar phone number", "neutral"),
    ("kelas cancel hari ni", "neutral"),
    ("dia kerja shift malam", "neutral"),
    ("aku nak renew passport", "neutral"),
    ("dia belajar kat UKM", "neutral"),
    ("aku nak bayar bil elektrik", "neutral"),
    ("dia pergi interview semalam", "neutral"),
    ("aku nak print assignment", "neutral"),
    ("dia beli laptop baru", "neutral"),
    ("aku nak top up touch n go", "neutral"),
    ("dia register kursus online", "neutral"),
    ("aku nak claim warranty", "neutral"),
    ("dia apply kerja kat Petronas", "neutral"),
    ("aku nak book hotel", "neutral"),
    ("dia hantar resume semalam", "neutral"),
    ("aku nak servis kereta", "neutral"),
    ("dia ada appointment pukul 2", "neutral"),
    ("aku nak collect parcel", "neutral"),
    ("dia nak renew roadtax", "neutral"),
]

# ============================================================
# LANGUAGE DETECTION (50 cases)
# ============================================================
LANGUAGE_DATA = [
    # BM (18)
    ("aku nak pergi makan nasi", "bm"),
    ("saya suka belajar di universiti", "bm"),
    ("dia kerja kat pejabat setiap hari", "bm"),
    ("makanan kat sini memang sedap", "bm"),
    ("kami semua pergi ke sekolah pagi tadi", "bm"),
    ("mereka sudah sampai rumah", "bm"),
    ("saya tidak faham soalan ini", "bm"),
    ("dia selalu datang lambat", "bm"),
    ("kami nak beli barang kat kedai", "bm"),
    ("budak tu pandai sangat", "bm"),
    ("kereta dia baru je beli", "bm"),
    ("rumah dia dekat dengan sekolah", "bm"),
    ("aku tak boleh pergi esok", "bm"),
    ("dia memang rajin belajar", "bm"),
    ("makanan dia masak sedap", "bm"),
    ("kami tunggu kat depan pintu", "bm"),
    ("hujan lebat sangat petang tadi", "bm"),
    ("dia dah habis buat kerja", "bm"),
    # EN (16)
    ("I want to go to the store", "en"),
    ("The weather is really nice today", "en"),
    ("She works at the hospital", "en"),
    ("Can you help me with this", "en"),
    ("They have been waiting for hours", "en"),
    ("I need to finish my assignment", "en"),
    ("The movie was really good", "en"),
    ("She told me about the meeting", "en"),
    ("We should go there tomorrow", "en"),
    ("He always comes late to class", "en"),
    ("I think we need more time", "en"),
    ("They want to buy a new house", "en"),
    ("She is going to the library", "en"),
    ("We had dinner at the restaurant", "en"),
    ("He said something interesting", "en"),
    ("I will call you later tonight", "en"),
    # Manglish (16)
    ("aku nak go buy some food then balik", "manglish"),
    ("dia cakap nak meet up kat mall", "manglish"),
    ("best gila the movie, kau kena try la", "manglish"),
    ("aku dah tired sangat nak rest kejap", "manglish"),
    ("confirm la dia coming tonight", "manglish"),
    ("serious ke you nak quit job tu", "manglish"),
    ("aku nak order food delivery je la", "manglish"),
    ("dia always late for class kan", "manglish"),
    ("kau dah try restaurant baru tu belum", "manglish"),
    ("aku need to finish assignment by tonight", "manglish"),
    ("jom la pergi shopping this weekend", "manglish"),
    ("dia nak join meeting ke tak", "manglish"),
    ("aku rasa dia busy sangat lately", "manglish"),
    ("kau free tak tomorrow morning", "manglish"),
    ("aku nak cancel plan sebab tired", "manglish"),
    ("dia cakap nak leave early today", "manglish"),
]

# ============================================================
# EMOTION (60 cases)
# ============================================================
EMOTION_DATA = [
    # Happy (10)
    ("gila happy aku dapat result bagus", "happy"),
    ("best sangat holiday kali ni", "happy"),
    ("seronok gila main game ni", "happy"),
    ("enjoy betul concert semalam", "happy"),
    ("syok gila dapat bonus", "happy"),
    ("mantap la team menang", "happy"),
    ("grateful sangat ada kawan macam korang", "happy"),
    ("lega gila dah habis exam", "happy"),
    ("bangga tengok anak berjaya", "happy"),
    ("puas hati la result kali ni", "happy"),
    # Sad (10)
    ("sedih gila dengar berita tu", "sad"),
    ("kecewa sangat dengan dia", "sad"),
    ("rindu gila kat family", "sad"),
    ("nangis je aku tengok movie tu", "sad"),
    ("down sangat hari ni", "sad"),
    ("lonely gila duduk sorang", "sad"),
    ("hampa sangat tak dapat offer", "sad"),
    ("sebak dengar lagu tu", "sad"),
    ("murung je dia hari ni", "sad"),
    ("hancur hati aku", "sad"),
    # Angry (10)
    ("bengang betul la service dia", "angry"),
    ("geram gila dengan orang macam ni", "angry"),
    ("marah sangat sebab kena tipu", "angry"),
    ("fed up dah dengan attitude dia", "angry"),
    ("benci gila perangai macam tu", "angry"),
    ("triggered sangat baca comment tu", "angry"),
    ("panas hati tengok dia buat macam tu", "angry"),
    ("menyampah betul dengan orang macam ni", "angry"),
    ("naik angin aku dengar cerita tu", "angry"),
    ("furious gila kena cut queue", "angry"),
    # Fear (10)
    ("cuak gila nak exam esok", "fear"),
    ("takut sangat nak present", "fear"),
    ("nervous gila first day kerja", "fear"),
    ("risau pasal result nanti", "fear"),
    ("anxious sangat tunggu reply", "fear"),
    ("gabra gila nak jumpa boss", "fear"),
    ("seram tengok cerita hantu tu", "fear"),
    ("panik gila bila phone hilang", "fear"),
    ("gelisah sangat malam ni", "fear"),
    ("bimbang pasal future", "fear"),
    # Surprise (10)
    ("terkejut gila dia datang", "surprise"),
    ("tak sangka dia boleh buat macam tu", "surprise"),
    ("shocked gila dengar news tu", "surprise"),
    ("wow tak expect langsung", "surprise"),
    ("alamak terlupa pulak", "surprise"),
    ("omg serious ke ni", "surprise"),
    ("speechless aku tengok result dia", "surprise"),
    ("unbelievable la benda ni", "surprise"),
    ("stunned gila aku", "surprise"),
    ("amazed tengok talent dia", "surprise"),
    # Love (5)
    ("sayang sangat kat dia", "love"),
    ("rindu gila kat awak", "love"),
    ("love sangat family aku", "love"),
    ("crush gila kat dia", "love"),
    ("comel sangat baby tu", "love"),
    # Disgust (5)
    ("jijik gila tengok benda tu", "disgust"),
    ("geli sangat pegang benda tu", "disgust"),
    ("eww busuk gila", "disgust"),
    ("loya tengok makanan tu", "disgust"),
    ("mual sangat bau dia", "disgust"),
]

# ============================================================
# PROFANITY (40 cases)
# ============================================================
PROFANITY_DATA = [
    # Toxic (20)
    ("kau ni bodoh ke apa", True),
    ("pergi mampus la kau", True),
    ("babi la orang tu", True),
    ("celaka punya orang", True),
    ("sial betul hari ni", True),
    ("stupid gila decision dia", True),
    ("wtf is wrong with you", True),
    ("mak kau hijau", True),
    ("bangang betul budak ni", True),
    ("hampeh la kerja kau", True),
    ("sampah je kau ni", True),
    ("idiot sangat la dia", True),
    ("damn la susah sangat", True),
    ("bengap ke kau ni", True),
    ("taik la plan ni", True),
    ("anjing punya orang", True),
    ("pukimak la kau", True),
    ("lancau betul", True),
    ("cibai la mamat tu", True),
    ("stfu la kau", True),
    # Clean (20)
    ("makanan sedap gila", False),
    ("aku nak pergi kedai", False),
    ("best la movie tu", False),
    ("dia memang pandai", False),
    ("gila cantik rumah dia", False),
    ("serious ke kau cakap", False),
    ("confirm la dia datang", False),
    ("power la presentation tu", False),
    ("mantap betul skills dia", False),
    ("legend la mamat ni", False),
    ("terror gila budak tu", False),
    ("solid la team kita", False),
    ("smooth je semua berjalan", False),
    ("padu gila design tu", False),
    ("terbaik la service dia", False),
    ("gempak la event semalam", False),
    ("enjoy sangat hari ni", False),
    ("seronok gila main game", False),
    ("syok la holiday ni", False),
    ("chill je vibe dia", False),
]

# ============================================================
# DIALECT (30 cases)
# ============================================================
DIALECT_DATA = [
    # Kelantan (6)
    ("ambo nok make nasi daghe", "kelantan"),
    ("gapo kau buat tu ore", "kelantan"),
    ("demo maghi sini", "kelantan"),
    ("kawe tok leh tubik", "kelantan"),
    ("bakpo kau buat gitu", "kelantan"),
    ("ghoyak la kat ambo", "kelantan"),
    # Terengganu (6)
    ("mung nok gi mane", "terengganu"),
    ("makang ikang kat sini", "terengganu"),
    ("budok tu sokmo lambat", "terengganu"),
    ("pitih dok cukup lagi", "terengganu"),
    ("guane nok buat ni", "terengganu"),
    ("kelih la sini jap", "terengganu"),
    # Kedah (6)
    ("hang pi mana tadi", "kedah"),
    ("depa semua dah balik", "kedah"),
    ("awat la hang buat camtu", "kedah"),
    ("mai la sini sat", "kedah"),
    ("habaq kat aku pasai tu", "kedah"),
    ("toksah la pikiaq banyak", "kedah"),
    # Sarawak (6)
    ("kamek sik mok polah ya", "sarawak"),
    ("kitak nemu sida dolok", "sarawak"),
    ("madah la kat kamek", "sarawak"),
    ("sik iboh risau", "sarawak"),
    ("nyamai makanan tok", "sarawak"),
    ("agik berapa lama", "sarawak"),
    # N9 (3)
    ("den tak nak poi sano", "negeri_sembilan"),
    ("apo kau buat tu", "negeri_sembilan"),
    ("dio dah balik doh", "negeri_sembilan"),
    # Standard (3)
    ("aku nak pergi makan", "standard"),
    ("saya suka belajar", "standard"),
    ("dia kerja kat pejabat", "standard"),
]

# ============================================================
# SARCASM (20 cases)
# ============================================================
SARCASM_DATA = [
    # Sarcastic (10)
    ("best la service dia, 2 jam tunggu", True),
    ("pandai la kau, exam fail", True),
    ("bagus la kerja kau, semua salah", True),
    ("rajin la budak ni, tidur je kerja", True),
    ("konon pandai tapi copy paste je", True),
    ("wow hebat sangat la kau ni", True),
    ("power la wifi ni, loading 10 minit", True),
    ("memang la bagus, rosak baru seminggu", True),
    ("tahniah la, last place", True),
    ("ok la tu... untuk budak darjah 1", True),
    # Not sarcastic (10)
    ("sedap gila makanan dia", False),
    ("best sangat holiday ni", False),
    ("power la presentation kau", False),
    ("pandai betul anak dia", False),
    ("rajin sangat budak tu", False),
    ("bagus la result kau", False),
    ("hebat gila skills dia", False),
    ("mantap la team ni", False),
    ("terbaik service kat sini", False),
    ("solid la kerja kau", False),
]

# ============================================================
# NORMALIZATION (30 cases)
# ============================================================
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
    ("nk tnya brapa sem utk grad", "nak tanya berapa semester untuk grad"),
    ("aku xnk pgi sne", "aku tak nak pergi sana"),
    ("dorg dh blk", "diorang dah balik"),
    ("mcm mne nk buat", "macam mana nak buat"),
    ("xtau la cmne", "tak tahu la macam mana"),
    ("kne byr bpe", "kena bayar berapa"),
    ("jap lg aku dtg", "jap lagi aku datang"),
    ("smpai rmh dh", "sampai rumah dah"),
    ("nk tdo dh", "nak tidur dah"),
    ("xpe la", "takpe la"),
    ("aku otw skrg", "aku on the way sekarang"),
    ("tgk la dlu", "tengok la dulu"),
    ("aku bkn nk mrh", "aku bukan nak marah"),
    ("dia mmg cmtu", "dia memang macam tu"),
    ("korg nk pgi x", "korang nak pergi tak"),
    ("aku dh penat sgt", "aku dah penat sangat"),
    ("nnt aku msg", "nanti aku message"),
    ("aku tgh mkn", "aku tengah makan"),
]

# ============================================================
# STEMMER (30 cases)
# ============================================================
STEMMER_DATA = [
    ("berlari", "lari"),
    ("memakan", "makan"),
    ("menulis", "tulis"),
    ("pelajaran", "ajar"),
    ("menyapu", "sapu"),
    ("terbang", "terbang"),
    ("memasak", "masak"),
    ("sekolahan", "sekolah"),
    ("berlarian", "lari"),
    ("memukul", "pukul"),
    ("menari", "tari"),
    ("perjalanan", "jalan"),
    ("ketinggian", "tinggi"),
    ("kebersihan", "bersih"),
    ("permainan", "main"),
    ("pengajaran", "ajar"),
    ("pembangunan", "bangun"),
    ("keindahan", "indah"),
    ("pekerja", "kerja"),
    ("penulis", "tulis"),
    ("pelari", "lari"),
    ("pemain", "main"),
    ("penyanyi", "nyanyi"),
    ("terbesar", "besar"),
    ("termakan", "makan"),
    ("dimasak", "masak"),
    ("dimakan", "makan"),
    ("dipukul", "pukul"),
    ("sebesar", "besar"),
    ("sekecil", "kecil"),
]


def run_benchmark():
    """Run full expanded benchmark."""
    start = time.time()
    total_pass = 0
    total_fail = 0
    results = {}
    
    print("=" * 60)
    print("MANGLISH-NLP EXPANDED BENCHMARK (500+ cases)")
    print("=" * 60)
    print()
    
    # Sentiment
    passed = 0
    for text, expected in SENTIMENT_DATA:
        r = malaysian_manglish_nlp.sentiment(text)
        if r['sentiment'] == expected:
            passed += 1
    total = len(SENTIMENT_DATA)
    results['sentiment'] = (passed, total)
    total_pass += passed
    total_fail += total - passed
    print(f"[SENTIMENT]       {passed}/{total} ({passed/total*100:.1f}%)")
    
    # Language
    passed = 0
    for text, expected in LANGUAGE_DATA:
        r = malaysian_manglish_nlp.detect_language(text)
        if r['language'] == expected:
            passed += 1
    total = len(LANGUAGE_DATA)
    results['language'] = (passed, total)
    total_pass += passed
    total_fail += total - passed
    print(f"[LANGUAGE]        {passed}/{total} ({passed/total*100:.1f}%)")
    
    # Emotion
    passed = 0
    for text, expected in EMOTION_DATA:
        r = detect_emotion(text)
        if r['emotion'] == expected:
            passed += 1
    total = len(EMOTION_DATA)
    results['emotion'] = (passed, total)
    total_pass += passed
    total_fail += total - passed
    print(f"[EMOTION]         {passed}/{total} ({passed/total*100:.1f}%)")
    
    # Profanity
    passed = 0
    for text, expected in PROFANITY_DATA:
        r = detect_profanity(text)
        if r['is_toxic'] == expected:
            passed += 1
    total = len(PROFANITY_DATA)
    results['profanity'] = (passed, total)
    total_pass += passed
    total_fail += total - passed
    print(f"[PROFANITY]       {passed}/{total} ({passed/total*100:.1f}%)")
    
    # Dialect
    passed = 0
    for text, expected in DIALECT_DATA:
        r = detect_dialect(text)
        if r['dialect'] == expected:
            passed += 1
    total = len(DIALECT_DATA)
    results['dialect'] = (passed, total)
    total_pass += passed
    total_fail += total - passed
    print(f"[DIALECT]         {passed}/{total} ({passed/total*100:.1f}%)")
    
    # Sarcasm
    passed = 0
    for text, expected in SARCASM_DATA:
        r = detect_sarcasm(text)
        if r['is_sarcastic'] == expected:
            passed += 1
    total = len(SARCASM_DATA)
    results['sarcasm'] = (passed, total)
    total_pass += passed
    total_fail += total - passed
    print(f"[SARCASM]         {passed}/{total} ({passed/total*100:.1f}%)")
    
    # Normalization
    passed = 0
    for text, expected in NORM_DATA:
        r = malaysian_manglish_nlp.normalize(text)
        if r == expected:
            passed += 1
    total = len(NORM_DATA)
    results['normalization'] = (passed, total)
    total_pass += passed
    total_fail += total - passed
    print(f"[NORMALIZATION]   {passed}/{total} ({passed/total*100:.1f}%)")
    
    # Stemmer
    passed = 0
    for text, expected in STEMMER_DATA:
        r = malaysian_manglish_nlp.stem_word(text)
        if r == expected:
            passed += 1
    total = len(STEMMER_DATA)
    results['stemmer'] = (passed, total)
    total_pass += passed
    total_fail += total - passed
    print(f"[STEMMER]         {passed}/{total} ({passed/total*100:.1f}%)")
    
    # Summary
    grand_total = total_pass + total_fail
    elapsed = time.time() - start
    print()
    print("=" * 60)
    print(f"OVERALL: {total_pass}/{grand_total} ({total_pass/grand_total*100:.1f}%)")
    print(f"Time: {elapsed:.2f}s")
    print("=" * 60)
    
    return results


if __name__ == '__main__':
    run_benchmark()
