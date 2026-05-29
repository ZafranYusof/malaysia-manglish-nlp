"""
Malaya Head-to-Head Benchmark
Compares manglish-nlp vs Malaya on 7 NLP tasks.
Fair comparison with accuracy, speed, and memory metrics.
"""

import time
import sys
import os
import tracemalloc

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try importing manglish-nlp modules
try:
    from manglish_nlp import sentiment, pos, ner, stemmer, normalize, language, tokenizer
    MANGLISH_NLP_AVAILABLE = True
except ImportError as e:
    print(f"[WARN] manglish-nlp import failed: {e}")
    MANGLISH_NLP_AVAILABLE = False

# Try importing Malaya
try:
    import malaya
    MALAYA_AVAILABLE = True
except ImportError:
    MALAYA_AVAILABLE = False

# =============================================================================
# TEST CASES - 50+ per task
# =============================================================================

SENTIMENT_TESTS = [
    {"text": "Sedap gila nasi lemak ni", "expected": "positive"},
    {"text": "Best la movie tu memang power", "expected": "positive"},
    {"text": "Terima kasih banyak2 korang memang terbaik", "expected": "positive"},
    {"text": "Cantik gila tempat ni weh", "expected": "positive"},
    {"text": "Gembira sangat dapat result cemerlang", "expected": "positive"},
    {"text": "Syok la kerja kat sini boss baik", "expected": "positive"},
    {"text": "Terharu tengok video ni sampai nangis", "expected": "positive"},
    {"text": "Mantap bro presentation kau tadi", "expected": "positive"},
    {"text": "Alhamdulillah rezeki murah bulan ni", "expected": "positive"},
    {"text": "Puas hati la service dekat sini memang tip top", "expected": "positive"},
    {"text": "Excited gila nak pergi concert next week", "expected": "positive"},
    {"text": "Bangga tengok anak Malaysia berjaya", "expected": "positive"},
    {"text": "Comel gila kucing tu nak bawak balik", "expected": "positive"},
    {"text": "Lega dah habis exam semua", "expected": "positive"},
    {"text": "Kagum dengan dedication dia memang respect", "expected": "positive"},
    {"text": "Confirm beli lagi nanti sedap sangat", "expected": "positive"},
    {"text": "Happy birthday bro semoga dipanjangkan umur", "expected": "positive"},
    {"text": "Memang berbaloi la harga tu untuk quality camni", "expected": "positive"},
    {"text": "Teruja nak try menu baru dekat situ", "expected": "positive"},
    {"text": "Kawan2 aku memang supportive gila", "expected": "positive"},
    {"text": "Sampah la service dia memang teruk", "expected": "negative"},
    {"text": "Menyesal beli phone ni rosak cepat", "expected": "negative"},
    {"text": "Bodoh la kerajaan buat keputusan macam ni", "expected": "negative"},
    {"text": "Marah gila aku kena tipu dengan seller tu", "expected": "negative"},
    {"text": "Kecewa sangat dengan result exam", "expected": "negative"},
    {"text": "Bosan gila kerja ni hari2 sama je", "expected": "negative"},
    {"text": "Takut la naik flight turbulence teruk", "expected": "negative"},
    {"text": "Sakit hati tengok perangai dia macam tu", "expected": "negative"},
    {"text": "Stress gila assignment banyak deadline semua sama", "expected": "negative"},
    {"text": "Sedih la kawan baik aku pindah jauh", "expected": "negative"},
    {"text": "Malu gila tadi terjatuh depan orang ramai", "expected": "negative"},
    {"text": "Frust sangat tak dapat scholarship tu", "expected": "negative"},
    {"text": "Jelak dah makan nasi hari2 nak muntah", "expected": "negative"},
    {"text": "Bengang betul jiran bising malam2", "expected": "negative"},
    {"text": "Penat la kerja overtime tak bayar pun", "expected": "negative"},
    {"text": "Hancur la plan cuti aku sebab hujan", "expected": "negative"},
    {"text": "Geram tengok orang buang sampah merata", "expected": "negative"},
    {"text": "Susah hati la pasal hutang ni", "expected": "negative"},
    {"text": "Teruk la traffic hari ni 2 jam stuck", "expected": "negative"},
    {"text": "Hampa sangat team kalah final tadi", "expected": "negative"},
    {"text": "Hari ni cuaca ok la biasa je", "expected": "neutral"},
    {"text": "Esok ada meeting pukul 10 pagi", "expected": "neutral"},
    {"text": "Dia kerja dekat KL dah 3 tahun", "expected": "neutral"},
    {"text": "Harga minyak naik 5 sen minggu ni", "expected": "neutral"},
    {"text": "Kedai tu buka pukul 9 tutup pukul 10", "expected": "neutral"},
    {"text": "Aku ambik course engineering dekat UMP", "expected": "neutral"},
    {"text": "Jalan ni connect dari PJ ke KL", "expected": "neutral"},
    {"text": "Population Malaysia dalam 33 juta sekarang", "expected": "neutral"},
    {"text": "Meeting postpone ke next Monday", "expected": "neutral"},
    {"text": "Dia pakai kereta Myvi warna putih", "expected": "neutral"},
    {"text": "Lepak mamak malam ni jom", "expected": "positive"},
    {"text": "Tak boleh tahan la budak ni annoying gila", "expected": "negative"},
]

POS_TESTS = [
    {"text": "Saya makan nasi", "expected": [("Saya", "PRON"), ("makan", "VERB"), ("nasi", "NOUN")]},
    {"text": "Kucing tu comel", "expected": [("Kucing", "NOUN"), ("tu", "DET"), ("comel", "ADJ")]},
    {"text": "Dia lari laju", "expected": [("Dia", "PRON"), ("lari", "VERB"), ("laju", "ADV")]},
    {"text": "Aku pergi sekolah", "expected": [("Aku", "PRON"), ("pergi", "VERB"), ("sekolah", "NOUN")]},
    {"text": "Buku ni mahal", "expected": [("Buku", "NOUN"), ("ni", "DET"), ("mahal", "ADJ")]},
    {"text": "Mereka bermain bola", "expected": [("Mereka", "PRON"), ("bermain", "VERB"), ("bola", "NOUN")]},
    {"text": "Rumah besar tu cantik", "expected": [("Rumah", "NOUN"), ("besar", "ADJ"), ("tu", "DET"), ("cantik", "ADJ")]},
    {"text": "Kami akan datang esok", "expected": [("Kami", "PRON"), ("akan", "AUX"), ("datang", "VERB"), ("esok", "NOUN")]},
    {"text": "Budak kecik tu nangis", "expected": [("Budak", "NOUN"), ("kecik", "ADJ"), ("tu", "DET"), ("nangis", "VERB")]},
    {"text": "Mak masak rendang sedap", "expected": [("Mak", "NOUN"), ("masak", "VERB"), ("rendang", "NOUN"), ("sedap", "ADJ")]},
    {"text": "Kereta merah tu laju gila", "expected": [("Kereta", "NOUN"), ("merah", "ADJ"), ("tu", "DET"), ("laju", "ADV"), ("gila", "ADV")]},
    {"text": "Abang beli ikan dekat pasar", "expected": [("Abang", "NOUN"), ("beli", "VERB"), ("ikan", "NOUN"), ("dekat", "ADP"), ("pasar", "NOUN")]},
    {"text": "Cikgu ajar matematik hari ni", "expected": [("Cikgu", "NOUN"), ("ajar", "VERB"), ("matematik", "NOUN"), ("hari", "NOUN"), ("ni", "DET")]},
    {"text": "Adik tidur awal semalam", "expected": [("Adik", "NOUN"), ("tidur", "VERB"), ("awal", "ADV"), ("semalam", "NOUN")]},
    {"text": "Kawan aku ramai dekat sini", "expected": [("Kawan", "NOUN"), ("aku", "PRON"), ("ramai", "ADJ"), ("dekat", "ADP"), ("sini", "NOUN")]},
    {"text": "Ayah bawa kereta baru", "expected": [("Ayah", "NOUN"), ("bawa", "VERB"), ("kereta", "NOUN"), ("baru", "ADJ")]},
    {"text": "Pokok tinggi tu dah tumbang", "expected": [("Pokok", "NOUN"), ("tinggi", "ADJ"), ("tu", "DET"), ("dah", "ADV"), ("tumbang", "VERB")]},
    {"text": "Aku suka makan durian", "expected": [("Aku", "PRON"), ("suka", "VERB"), ("makan", "VERB"), ("durian", "NOUN")]},
    {"text": "Hujan lebat petang tadi", "expected": [("Hujan", "NOUN"), ("lebat", "ADJ"), ("petang", "NOUN"), ("tadi", "DET")]},
    {"text": "Dia pandai sangat budak tu", "expected": [("Dia", "PRON"), ("pandai", "ADJ"), ("sangat", "ADV"), ("budak", "NOUN"), ("tu", "DET")]},
    {"text": "Makcik jual kuih muih", "expected": [("Makcik", "NOUN"), ("jual", "VERB"), ("kuih", "NOUN"), ("muih", "NOUN")]},
    {"text": "Anak dia dah besar", "expected": [("Anak", "NOUN"), ("dia", "PRON"), ("dah", "ADV"), ("besar", "ADJ")]},
    {"text": "Jangan lari cepat sangat", "expected": [("Jangan", "ADV"), ("lari", "VERB"), ("cepat", "ADV"), ("sangat", "ADV")]},
    {"text": "Kakak masuk universiti tahun ni", "expected": [("Kakak", "NOUN"), ("masuk", "VERB"), ("universiti", "NOUN"), ("tahun", "NOUN"), ("ni", "DET")]},
    {"text": "Nasi goreng dia memang sedap", "expected": [("Nasi", "NOUN"), ("goreng", "NOUN"), ("dia", "PRON"), ("memang", "ADV"), ("sedap", "ADJ")]},
    {"text": "Aku tak faham soalan tu", "expected": [("Aku", "PRON"), ("tak", "ADV"), ("faham", "VERB"), ("soalan", "NOUN"), ("tu", "DET")]},
    {"text": "Bas datang lambat hari ni", "expected": [("Bas", "NOUN"), ("datang", "VERB"), ("lambat", "ADV"), ("hari", "NOUN"), ("ni", "DET")]},
    {"text": "Doktor suruh makan ubat", "expected": [("Doktor", "NOUN"), ("suruh", "VERB"), ("makan", "VERB"), ("ubat", "NOUN")]},
    {"text": "Jalan raya sesak teruk", "expected": [("Jalan", "NOUN"), ("raya", "NOUN"), ("sesak", "ADJ"), ("teruk", "ADV")]},
    {"text": "Pakcik tu baik hati orangnya", "expected": [("Pakcik", "NOUN"), ("tu", "DET"), ("baik", "ADJ"), ("hati", "NOUN"), ("orangnya", "NOUN")]},
    {"text": "Kami main futsal petang tadi", "expected": [("Kami", "PRON"), ("main", "VERB"), ("futsal", "NOUN"), ("petang", "NOUN"), ("tadi", "DET")]},
    {"text": "Awak nak pergi mana", "expected": [("Awak", "PRON"), ("nak", "AUX"), ("pergi", "VERB"), ("mana", "PRON")]},
    {"text": "Telefon dia mahal gila", "expected": [("Telefon", "NOUN"), ("dia", "PRON"), ("mahal", "ADJ"), ("gila", "ADV")]},
    {"text": "Nenek masak sup ayam", "expected": [("Nenek", "NOUN"), ("masak", "VERB"), ("sup", "NOUN"), ("ayam", "NOUN")]},
    {"text": "Aku dah siap kerja", "expected": [("Aku", "PRON"), ("dah", "ADV"), ("siap", "VERB"), ("kerja", "NOUN")]},
    {"text": "Budak2 main kat padang", "expected": [("Budak2", "NOUN"), ("main", "VERB"), ("kat", "ADP"), ("padang", "NOUN")]},
    {"text": "Matahari terbit pagi tadi", "expected": [("Matahari", "NOUN"), ("terbit", "VERB"), ("pagi", "NOUN"), ("tadi", "DET")]},
    {"text": "Dia bagi hadiah kat aku", "expected": [("Dia", "PRON"), ("bagi", "VERB"), ("hadiah", "NOUN"), ("kat", "ADP"), ("aku", "PRON")]},
    {"text": "Makanan sini murah dan sedap", "expected": [("Makanan", "NOUN"), ("sini", "NOUN"), ("murah", "ADJ"), ("dan", "CONJ"), ("sedap", "ADJ")]},
    {"text": "Aku tunggu kau dekat sana", "expected": [("Aku", "PRON"), ("tunggu", "VERB"), ("kau", "PRON"), ("dekat", "ADP"), ("sana", "NOUN")]},
    {"text": "Semua orang dah balik", "expected": [("Semua", "DET"), ("orang", "NOUN"), ("dah", "ADV"), ("balik", "VERB")]},
    {"text": "Dia cakap Melayu fasih", "expected": [("Dia", "PRON"), ("cakap", "VERB"), ("Melayu", "NOUN"), ("fasih", "ADV")]},
    {"text": "Aku nak tidur dah ngantuk", "expected": [("Aku", "PRON"), ("nak", "AUX"), ("tidur", "VERB"), ("dah", "ADV"), ("ngantuk", "ADJ")]},
    {"text": "Harga barang naik lagi", "expected": [("Harga", "NOUN"), ("barang", "NOUN"), ("naik", "VERB"), ("lagi", "ADV")]},
    {"text": "Kita jumpa esok pagi", "expected": [("Kita", "PRON"), ("jumpa", "VERB"), ("esok", "NOUN"), ("pagi", "NOUN")]},
    {"text": "Bilik dia kemas sangat", "expected": [("Bilik", "NOUN"), ("dia", "PRON"), ("kemas", "ADJ"), ("sangat", "ADV")]},
    {"text": "Aku baca buku cerita", "expected": [("Aku", "PRON"), ("baca", "VERB"), ("buku", "NOUN"), ("cerita", "NOUN")]},
    {"text": "Anjing tu garang betul", "expected": [("Anjing", "NOUN"), ("tu", "DET"), ("garang", "ADJ"), ("betul", "ADV")]},
    {"text": "Emak suruh kemas bilik", "expected": [("Emak", "NOUN"), ("suruh", "VERB"), ("kemas", "VERB"), ("bilik", "NOUN")]},
    {"text": "Aku dengar lagu baru", "expected": [("Aku", "PRON"), ("dengar", "VERB"), ("lagu", "NOUN"), ("baru", "ADJ")]},
    {"text": "Dia tulis surat panjang", "expected": [("Dia", "PRON"), ("tulis", "VERB"), ("surat", "NOUN"), ("panjang", "ADJ")]},
]

NER_TESTS = [
    {"text": "Ahmad tinggal di Kuala Lumpur", "expected": [("Ahmad", "PERSON"), ("Kuala Lumpur", "LOCATION")]},
    {"text": "Siti kerja dekat Petronas", "expected": [("Siti", "PERSON"), ("Petronas", "ORGANIZATION")]},
    {"text": "Ali belajar di Universiti Malaya", "expected": [("Ali", "PERSON"), ("Universiti Malaya", "ORGANIZATION")]},
    {"text": "Mahathir jadi PM Malaysia", "expected": [("Mahathir", "PERSON"), ("Malaysia", "LOCATION")]},
    {"text": "KLCC dekat Jalan Ampang", "expected": [("KLCC", "LOCATION"), ("Jalan Ampang", "LOCATION")]},
    {"text": "Grab ada office dekat Singapore", "expected": [("Grab", "ORGANIZATION"), ("Singapore", "LOCATION")]},
    {"text": "Anwar Ibrahim PM ke-10", "expected": [("Anwar Ibrahim", "PERSON")]},
    {"text": "Bank Negara naikkan kadar faedah", "expected": [("Bank Negara", "ORGANIZATION")]},
    {"text": "Johor Bahru dekat dengan Singapura", "expected": [("Johor Bahru", "LOCATION"), ("Singapura", "LOCATION")]},
    {"text": "Proton keluarkan model baru X90", "expected": [("Proton", "ORGANIZATION")]},
    {"text": "Rafizi announce budget baru", "expected": [("Rafizi", "PERSON")]},
    {"text": "TM bagi internet laju kat Penang", "expected": [("TM", "ORGANIZATION"), ("Penang", "LOCATION")]},
    {"text": "Harimau Malaya menang lawan Vietnam", "expected": [("Harimau Malaya", "ORGANIZATION"), ("Vietnam", "LOCATION")]},
    {"text": "Lee Chong Wei legend badminton", "expected": [("Lee Chong Wei", "PERSON")]},
    {"text": "Tunku Abdul Rahman bapa kemerdekaan", "expected": [("Tunku Abdul Rahman", "PERSON")]},
    {"text": "AirAsia fly dari KLIA2", "expected": [("AirAsia", "ORGANIZATION"), ("KLIA2", "LOCATION")]},
    {"text": "Najib kena kes 1MDB", "expected": [("Najib", "PERSON"), ("1MDB", "ORGANIZATION")]},
    {"text": "Shopee ada warehouse dekat Shah Alam", "expected": [("Shopee", "ORGANIZATION"), ("Shah Alam", "LOCATION")]},
    {"text": "Datuk Seri buka majlis kat Putrajaya", "expected": [("Putrajaya", "LOCATION")]},
    {"text": "UiTM banyak campus seluruh Malaysia", "expected": [("UiTM", "ORGANIZATION"), ("Malaysia", "LOCATION")]},
    {"text": "Sungai Pahang paling panjang", "expected": [("Sungai Pahang", "LOCATION")]},
    {"text": "Perodua jual Myvi paling laris", "expected": [("Perodua", "ORGANIZATION")]},
    {"text": "Sabah dan Sarawak kat Borneo", "expected": [("Sabah", "LOCATION"), ("Sarawak", "LOCATION"), ("Borneo", "LOCATION")]},
    {"text": "TNB supply elektrik seluruh negara", "expected": [("TNB", "ORGANIZATION")]},
    {"text": "Langkawi pulau bebas cukai", "expected": [("Langkawi", "LOCATION")]},
    {"text": "Zafran belajar kat UMP Pahang", "expected": [("Zafran", "PERSON"), ("UMP", "ORGANIZATION"), ("Pahang", "LOCATION")]},
    {"text": "Maybank bank terbesar Malaysia", "expected": [("Maybank", "ORGANIZATION"), ("Malaysia", "LOCATION")]},
    {"text": "Nasi kandar famous kat Penang", "expected": [("Penang", "LOCATION")]},
    {"text": "Ismail Sabri PM sebelum Anwar", "expected": [("Ismail Sabri", "PERSON"), ("Anwar", "PERSON")]},
    {"text": "MRT Putrajaya line dah siap", "expected": [("MRT Putrajaya", "ORGANIZATION")]},
    {"text": "Genting Highlands sejuk tahun round", "expected": [("Genting Highlands", "LOCATION")]},
    {"text": "CIMB merge dengan Southern Bank dulu", "expected": [("CIMB", "ORGANIZATION"), ("Southern Bank", "ORGANIZATION")]},
    {"text": "Nicol David legend squash Malaysia", "expected": [("Nicol David", "PERSON"), ("Malaysia", "LOCATION")]},
    {"text": "Celcom dan Digi dah merge jadi CelcomDigi", "expected": [("Celcom", "ORGANIZATION"), ("Digi", "ORGANIZATION"), ("CelcomDigi", "ORGANIZATION")]},
    {"text": "Cameron Highlands famous dengan teh", "expected": [("Cameron Highlands", "LOCATION")]},
    {"text": "Syed Saddiq buat parti MUDA", "expected": [("Syed Saddiq", "PERSON"), ("MUDA", "ORGANIZATION")]},
    {"text": "Ipoh famous dengan taugeh ayam", "expected": [("Ipoh", "LOCATION")]},
    {"text": "Khazanah Nasional invest banyak company", "expected": [("Khazanah Nasional", "ORGANIZATION")]},
    {"text": "Melaka dulu dijajah Portugis", "expected": [("Melaka", "LOCATION")]},
    {"text": "Astro siarkan EPL kat Malaysia", "expected": [("Astro", "ORGANIZATION"), ("Malaysia", "LOCATION")]},
    {"text": "Kuching ibu negeri Sarawak", "expected": [("Kuching", "LOCATION"), ("Sarawak", "LOCATION")]},
    {"text": "Tan Sri Vincent Tan owner Cardiff", "expected": [("Vincent Tan", "PERSON"), ("Cardiff", "ORGANIZATION")]},
    {"text": "PETRONAS twin tower landmark KL", "expected": [("PETRONAS", "ORGANIZATION"), ("KL", "LOCATION")]},
    {"text": "Dato Lee Chong Wei retire 2019", "expected": [("Lee Chong Wei", "PERSON")]},
    {"text": "Sunway buat theme park dekat Subang", "expected": [("Sunway", "ORGANIZATION"), ("Subang", "LOCATION")]},
    {"text": "Kelantan famous dengan nasi kerabu", "expected": [("Kelantan", "LOCATION")]},
    {"text": "Tony Fernandes CEO AirAsia", "expected": [("Tony Fernandes", "PERSON"), ("AirAsia", "ORGANIZATION")]},
    {"text": "Pulau Tioman cantik gila", "expected": [("Pulau Tioman", "LOCATION")]},
    {"text": "UMNO parti paling lama Malaysia", "expected": [("UMNO", "ORGANIZATION"), ("Malaysia", "LOCATION")]},
    {"text": "Kota Kinabalu ibu negeri Sabah", "expected": [("Kota Kinabalu", "LOCATION"), ("Sabah", "LOCATION")]},
    {"text": "Yusuf Ishak presiden pertama Singapura", "expected": [("Yusuf Ishak", "PERSON"), ("Singapura", "LOCATION")]},
]

STEMMING_TESTS = [
    {"text": "berlari", "expected": "lari"},
    {"text": "memakan", "expected": "makan"},
    {"text": "terbang", "expected": "terbang"},
    {"text": "pelajaran", "expected": "ajar"},
    {"text": "membeli", "expected": "beli"},
    {"text": "terjatuh", "expected": "jatuh"},
    {"text": "perjalanan", "expected": "jalan"},
    {"text": "bermain", "expected": "main"},
    {"text": "melompat", "expected": "lompat"},
    {"text": "pekerjaan", "expected": "kerja"},
    {"text": "menyanyi", "expected": "nyanyi"},
    {"text": "terbesar", "expected": "besar"},
    {"text": "pelarian", "expected": "lari"},
    {"text": "berfikir", "expected": "fikir"},
    {"text": "memandu", "expected": "pandu"},
    {"text": "terkejut", "expected": "kejut"},
    {"text": "pengajaran", "expected": "ajar"},
    {"text": "berjalan", "expected": "jalan"},
    {"text": "menulis", "expected": "tulis"},
    {"text": "terlupa", "expected": "lupa"},
    {"text": "pembangunan", "expected": "bangun"},
    {"text": "bermula", "expected": "mula"},
    {"text": "melawan", "expected": "lawan"},
    {"text": "tertidur", "expected": "tidur"},
    {"text": "perniagaan", "expected": "niaga"},
    {"text": "berkawan", "expected": "kawan"},
    {"text": "memukul", "expected": "pukul"},
    {"text": "terpilih", "expected": "pilih"},
    {"text": "pendidikan", "expected": "didik"},
    {"text": "bersama", "expected": "sama"},
    {"text": "mengajar", "expected": "ajar"},
    {"text": "terlepas", "expected": "lepas"},
    {"text": "perasaan", "expected": "rasa"},
    {"text": "bertukar", "expected": "tukar"},
    {"text": "menarik", "expected": "tarik"},
    {"text": "terbuka", "expected": "buka"},
    {"text": "keluarga", "expected": "keluarga"},
    {"text": "bersedia", "expected": "sedia"},
    {"text": "membantu", "expected": "bantu"},
    {"text": "terbaik", "expected": "baik"},
    {"text": "kehidupan", "expected": "hidup"},
    {"text": "berharap", "expected": "harap"},
    {"text": "memasak", "expected": "masak"},
    {"text": "terjaga", "expected": "jaga"},
    {"text": "permainan", "expected": "main"},
    {"text": "belajar", "expected": "ajar"},
    {"text": "mencari", "expected": "cari"},
    {"text": "terbang", "expected": "terbang"},
    {"text": "perkataan", "expected": "kata"},
    {"text": "berbeza", "expected": "beza"},
    {"text": "menjadi", "expected": "jadi"},
]

NORMALIZATION_TESTS = [
    {"text": "xpe", "expected": "tak apa"},
    {"text": "nk", "expected": "nak"},
    {"text": "mcm", "expected": "macam"},
    {"text": "sbb", "expected": "sebab"},
    {"text": "dh", "expected": "dah"},
    {"text": "tk", "expected": "tak"},
    {"text": "cmne", "expected": "macam mana"},
    {"text": "nape", "expected": "kenapa"},
    {"text": "ape", "expected": "apa"},
    {"text": "kt", "expected": "kat"},
    {"text": "dgn", "expected": "dengan"},
    {"text": "utk", "expected": "untuk"},
    {"text": "yg", "expected": "yang"},
    {"text": "org", "expected": "orang"},
    {"text": "sgt", "expected": "sangat"},
    {"text": "mmg", "expected": "memang"},
    {"text": "blh", "expected": "boleh"},
    {"text": "nnt", "expected": "nanti"},
    {"text": "skrg", "expected": "sekarang"},
    {"text": "dkt", "expected": "dekat"},
    {"text": "byk", "expected": "banyak"},
    {"text": "smpai", "expected": "sampai"},
    {"text": "jgn", "expected": "jangan"},
    {"text": "lg", "expected": "lagi"},
    {"text": "je", "expected": "sahaja"},
    {"text": "pon", "expected": "pun"},
    {"text": "gk", "expected": "jugak"},
    {"text": "tgk", "expected": "tengok"},
    {"text": "nk g mne", "expected": "nak pergi mana"},
    {"text": "xde", "expected": "tak ada"},
    {"text": "igt", "expected": "ingat"},
    {"text": "sblm", "expected": "sebelum"},
    {"text": "slps", "expected": "selepas"},
    {"text": "brp", "expected": "berapa"},
    {"text": "mne", "expected": "mana"},
    {"text": "bile", "expected": "bila"},
    {"text": "sape", "expected": "siapa"},
    {"text": "camne", "expected": "macam mana"},
    {"text": "xyah", "expected": "tak payah"},
    {"text": "xleh", "expected": "tak boleh"},
    {"text": "psl", "expected": "pasal"},
    {"text": "tdo", "expected": "tidur"},
    {"text": "mkn", "expected": "makan"},
    {"text": "mnum", "expected": "minum"},
    {"text": "blk", "expected": "balik"},
    {"text": "keje", "expected": "kerja"},
    {"text": "ckp", "expected": "cakap"},
    {"text": "smpi", "expected": "sampai"},
    {"text": "hrp", "expected": "harap"},
    {"text": "thn", "expected": "tahun"},
    {"text": "bln", "expected": "bulan"},
]

LANGUAGE_DETECTION_TESTS = [
    {"text": "Saya pergi ke sekolah hari ini", "expected": "ms"},
    {"text": "I went to school today", "expected": "en"},
    {"text": "Aku pergi school tadi best gila", "expected": "manglish"},
    {"text": "Wei jom lepak mamak tonight", "expected": "manglish"},
    {"text": "Dia cakap dia nak go shopping", "expected": "manglish"},
    {"text": "The weather is nice today", "expected": "en"},
    {"text": "Cuaca hari ini sangat baik", "expected": "ms"},
    {"text": "Weh this movie damn nice la", "expected": "manglish"},
    {"text": "Confirm best punya la bro", "expected": "manglish"},
    {"text": "Aku dah fed up dengan situation ni", "expected": "manglish"},
    {"text": "Kerajaan Malaysia mengumumkan dasar baru", "expected": "ms"},
    {"text": "The government announced a new policy", "expected": "en"},
    {"text": "Government announce new policy tapi rakyat tak happy", "expected": "manglish"},
    {"text": "Makanan di restoran ini sangat lazat", "expected": "ms"},
    {"text": "Food here damn sedap one", "expected": "manglish"},
    {"text": "Can you help me with this assignment", "expected": "en"},
    {"text": "Boleh tolong aku dengan assignment ni tak", "expected": "manglish"},
    {"text": "Sila bantu saya dengan tugasan ini", "expected": "ms"},
    {"text": "Aku nak order grab food jap", "expected": "manglish"},
    {"text": "Traffic jam gila babi hari ni", "expected": "manglish"},
    {"text": "Kesesakan lalu lintas amat teruk hari ini", "expected": "ms"},
    {"text": "The traffic is terrible today", "expected": "en"},
    {"text": "Boss aku suruh OT lagi malam ni fml", "expected": "manglish"},
    {"text": "Pengurus saya meminta saya bekerja lebih masa", "expected": "ms"},
    {"text": "My manager asked me to work overtime", "expected": "en"},
    {"text": "Gaji masuk tapi bills lagi banyak adoi", "expected": "manglish"},
    {"text": "Pendapatan tidak mencukupi untuk perbelanjaan", "expected": "ms"},
    {"text": "Income is not enough for expenses", "expected": "en"},
    {"text": "Jom la gi karaoke malam ni release tension", "expected": "manglish"},
    {"text": "Mari kita pergi berkaraoke malam ini", "expected": "ms"},
    {"text": "Let us go for karaoke tonight", "expected": "en"},
    {"text": "Aku craving nasi lemak gila right now", "expected": "manglish"},
    {"text": "Saya sangat ingin makan nasi lemak sekarang", "expected": "ms"},
    {"text": "I really want to eat nasi lemak right now", "expected": "en"},
    {"text": "Parking full la sial kena park jauh", "expected": "manglish"},
    {"text": "Tempat letak kereta penuh", "expected": "ms"},
    {"text": "The parking lot is full", "expected": "en"},
    {"text": "Eh sorry la bro aku ter-cancel order kau", "expected": "manglish"},
    {"text": "Maafkan saya kerana membatalkan pesanan anda", "expected": "ms"},
    {"text": "Sorry for cancelling your order", "expected": "en"},
    {"text": "Wifi kat sini slow gila nak mampus", "expected": "manglish"},
    {"text": "Internet di sini sangat perlahan", "expected": "ms"},
    {"text": "The internet here is very slow", "expected": "en"},
    {"text": "Aku tengah binge watch Korean drama ni addictive gila", "expected": "manglish"},
    {"text": "Saya sedang menonton drama Korea", "expected": "ms"},
    {"text": "I am watching Korean drama", "expected": "en"},
    {"text": "Kau dah try cafe baru tu belum quite nice actually", "expected": "manglish"},
    {"text": "Adakah anda sudah mencuba kafe baru itu", "expected": "ms"},
    {"text": "Have you tried the new cafe", "expected": "en"},
    {"text": "Deadline esok tapi aku belum start lagi rip", "expected": "manglish"},
    {"text": "Tarikh akhir adalah esok", "expected": "ms"},
]

TOKENIZATION_TESTS = [
    {"text": "Aku nak makan", "expected": ["Aku", "nak", "makan"]},
    {"text": "tak boleh la macam tu", "expected": ["tak", "boleh", "la", "macam", "tu"]},
    {"text": "RM50.90 je harga dia", "expected": ["RM50.90", "je", "harga", "dia"]},
    {"text": "email aku test@gmail.com", "expected": ["email", "aku", "test@gmail.com"]},
    {"text": "check https://google.com ni", "expected": ["check", "https://google.com", "ni"]},
    {"text": "haha...ok la tu", "expected": ["haha", "...", "ok", "la", "tu"]},
    {"text": "Weh!!! Gila ke apa?!", "expected": ["Weh", "!!!", "Gila", "ke", "apa", "?!"]},
    {"text": "No. telefon: 012-3456789", "expected": ["No.", "telefon", ":", "012-3456789"]},
    {"text": "Pukul 3:30pm kita jumpa", "expected": ["Pukul", "3:30pm", "kita", "jumpa"]},
    {"text": "Dia kata 'jangan pergi'", "expected": ["Dia", "kata", "'", "jangan", "pergi", "'"]},
    {"text": "A/C rosak lagi ke??", "expected": ["A/C", "rosak", "lagi", "ke", "??"]},
    {"text": "Beli 2kg bawang + 1kg cili", "expected": ["Beli", "2kg", "bawang", "+", "1kg", "cili"]},
    {"text": "OMG best gila 10/10", "expected": ["OMG", "best", "gila", "10/10"]},
    {"text": "Saya dr. Ahmad", "expected": ["Saya", "dr.", "Ahmad"]},
    {"text": "etc. lain2 tu tak penting", "expected": ["etc.", "lain2", "tu", "tak", "penting"]},
    {"text": "Harga naik 5% bulan ni", "expected": ["Harga", "naik", "5%", "bulan", "ni"]},
    {"text": "KL-Penang flight 1.5jam je", "expected": ["KL-Penang", "flight", "1.5jam", "je"]},
    {"text": "Dia score 4.0 CGPA gila", "expected": ["Dia", "score", "4.0", "CGPA", "gila"]},
    {"text": "I/C number kena bawa", "expected": ["I/C", "number", "kena", "bawa"]},
    {"text": "Jalan Tun H.S. Lee tu famous", "expected": ["Jalan", "Tun", "H.S.", "Lee", "tu", "famous"]},
    {"text": "#MalaysiaBoleh trending", "expected": ["#MalaysiaBoleh", "trending"]},
    {"text": "@ahmad_93 kau kat mana", "expected": ["@ahmad_93", "kau", "kat", "mana"]},
    {"text": "Suhu 32.5°C panas gila", "expected": ["Suhu", "32.5°C", "panas", "gila"]},
    {"text": "Beli iPhone 15 Pro Max", "expected": ["Beli", "iPhone", "15", "Pro", "Max"]},
    {"text": "Dia cakap...ntah la", "expected": ["Dia", "cakap", "...", "ntah", "la"]},
    {"text": "W/out kau aku tak boleh", "expected": ["W/out", "kau", "aku", "tak", "boleh"]},
    {"text": "GDP naik 4.5% Q3 2024", "expected": ["GDP", "naik", "4.5%", "Q3", "2024"]},
    {"text": "Rm1,000,000 tu banyak weh", "expected": ["Rm1,000,000", "tu", "banyak", "weh"]},
    {"text": "Dia ada PhD. dari UK", "expected": ["Dia", "ada", "PhD.", "dari", "UK"]},
    {"text": "24/7 buka kedai dia", "expected": ["24/7", "buka", "kedai", "dia"]},
    {"text": "Aku bagi 5-star rating", "expected": ["Aku", "bagi", "5-star", "rating"]},
    {"text": "Jom makan :) lapar dah", "expected": ["Jom", "makan", ":)", "lapar", "dah"]},
    {"text": "Hahaha xD lawak gila", "expected": ["Hahaha", "xD", "lawak", "gila"]},
    {"text": "S.P.M. result keluar esok", "expected": ["S.P.M.", "result", "keluar", "esok"]},
    {"text": "Dia kerja 9-5 je", "expected": ["Dia", "kerja", "9-5", "je"]},
    {"text": "Berat 65.5kg maintain", "expected": ["Berat", "65.5kg", "maintain"]},
    {"text": "Aku kat Lot 10, Bukit Bintang", "expected": ["Aku", "kat", "Lot", "10", ",", "Bukit", "Bintang"]},
    {"text": "v2.0 dah release", "expected": ["v2.0", "dah", "release"]},
    {"text": "Dia ada 3+1 bilik", "expected": ["Dia", "ada", "3+1", "bilik"]},
    {"text": "COVID-19 dah under control", "expected": ["COVID-19", "dah", "under", "control"]},
    {"text": "Aku bayar via FPX/DuitNow", "expected": ["Aku", "bayar", "via", "FPX/DuitNow"]},
    {"text": "Tinggi dia 5'8\" je", "expected": ["Tinggi", "dia", "5'8\"", "je"]},
    {"text": "Aku nak 2x spicy", "expected": ["Aku", "nak", "2x", "spicy"]},
    {"text": "Dia reply 'ok' je -_-", "expected": ["Dia", "reply", "'", "ok", "'", "je", "-_-"]},
    {"text": "Gaji RM3.5k sebulan", "expected": ["Gaji", "RM3.5k", "sebulan"]},
    {"text": "Aku ada 2-3 soalan", "expected": ["Aku", "ada", "2-3", "soalan"]},
    {"text": "Dia cakap N/A je", "expected": ["Dia", "cakap", "N/A", "je"]},
    {"text": "Aku kat blk B-12-3", "expected": ["Aku", "kat", "blk", "B-12-3"]},
    {"text": "Harga USD$99.99", "expected": ["Harga", "USD$99.99"]},
    {"text": "Aku guna Windows 11 Pro", "expected": ["Aku", "guna", "Windows", "11", "Pro"]},
    {"text": "Dia hantar <3 emoji", "expected": ["Dia", "hantar", "<3", "emoji"]},
]

# =============================================================================
# BENCHMARK FUNCTIONS
# =============================================================================

def measure_memory(func, *args, **kwargs):
    """Measure memory usage of a function call."""
    tracemalloc.start()
    result = func(*args, **kwargs)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, peak / 1024  # Return peak in KB


def benchmark_sentiment():
    """Benchmark sentiment analysis."""
    results = {"task": "Sentiment Analysis", "manglish_nlp": {}, "malaya": {}}
    
    if not MANGLISH_NLP_AVAILABLE:
        results["manglish_nlp"] = {"error": "not installed"}
        return results
    
    # manglish-nlp
    correct = 0
    total = len(SENTIMENT_TESTS)
    start = time.perf_counter()
    
    for test in SENTIMENT_TESTS:
        try:
            result = sentiment.analyze(test["text"])
            predicted = result.get("sentiment", result.get("label", ""))
            if isinstance(predicted, str) and predicted.lower() == test["expected"].lower():
                correct += 1
        except Exception as e:
            pass
    
    elapsed = time.perf_counter() - start
    results["manglish_nlp"] = {
        "accuracy": correct / total * 100,
        "correct": correct,
        "total": total,
        "time_ms": elapsed * 1000,
        "avg_ms": (elapsed * 1000) / total,
    }
    
    # Malaya
    if MALAYA_AVAILABLE:
        try:
            model = malaya.sentiment.transformer(model="mesolitica/sentiment-analysis-nanot5-small-malaysian-cased")
            correct_m = 0
            start_m = time.perf_counter()
            
            for test in SENTIMENT_TESTS:
                try:
                    result = model.predict(test["text"])
                    if result.lower() == test["expected"].lower():
                        correct_m += 1
                except:
                    pass
            
            elapsed_m = time.perf_counter() - start_m
            results["malaya"] = {
                "accuracy": correct_m / total * 100,
                "correct": correct_m,
                "total": total,
                "time_ms": elapsed_m * 1000,
                "avg_ms": (elapsed_m * 1000) / total,
            }
        except Exception as e:
            results["malaya"] = {"error": str(e)}
    else:
        results["malaya"] = {"error": "not installed"}
    
    return results


def benchmark_pos():
    """Benchmark POS tagging."""
    results = {"task": "POS Tagging", "manglish_nlp": {}, "malaya": {}}
    
    if not MANGLISH_NLP_AVAILABLE:
        results["manglish_nlp"] = {"error": "not installed"}
        return results
    
    correct = 0
    total_tags = 0
    total_sentences = len(POS_TESTS)
    start = time.perf_counter()
    
    for test in POS_TESTS:
        try:
            result = pos.tag(test["text"])
            expected_tags = {word: tag for word, tag in test["expected"]}
            for word, tag in result:
                total_tags += 1
                if word in expected_tags and expected_tags[word] == tag:
                    correct += 1
        except Exception:
            total_tags += len(test["expected"])
    
    elapsed = time.perf_counter() - start
    results["manglish_nlp"] = {
        "accuracy": (correct / total_tags * 100) if total_tags > 0 else 0,
        "correct_tags": correct,
        "total_tags": total_tags,
        "sentences": total_sentences,
        "time_ms": elapsed * 1000,
        "avg_ms": (elapsed * 1000) / total_sentences,
    }
    
    if MALAYA_AVAILABLE:
        try:
            model = malaya.pos.transformer(model="mesolitica/pos-t5-small-standard-bahasa-cased")
            correct_m = 0
            total_tags_m = 0
            start_m = time.perf_counter()
            
            for test in POS_TESTS:
                try:
                    result = model.predict(test["text"])
                    expected_tags = {word: tag for word, tag in test["expected"]}
                    for word, tag in result:
                        total_tags_m += 1
                        if word in expected_tags and expected_tags[word] == tag:
                            correct_m += 1
                except:
                    total_tags_m += len(test["expected"])
            
            elapsed_m = time.perf_counter() - start_m
            results["malaya"] = {
                "accuracy": (correct_m / total_tags_m * 100) if total_tags_m > 0 else 0,
                "correct_tags": correct_m,
                "total_tags": total_tags_m,
                "sentences": total_sentences,
                "time_ms": elapsed_m * 1000,
                "avg_ms": (elapsed_m * 1000) / total_sentences,
            }
        except Exception as e:
            results["malaya"] = {"error": str(e)}
    else:
        results["malaya"] = {"error": "not installed"}
    
    return results


def benchmark_ner():
    """Benchmark Named Entity Recognition."""
    results = {"task": "NER", "manglish_nlp": {}, "malaya": {}}
    
    if not MANGLISH_NLP_AVAILABLE:
        results["manglish_nlp"] = {"error": "not installed"}
        return results
    
    correct_entities = 0
    total_entities = 0
    total_sentences = len(NER_TESTS)
    start = time.perf_counter()
    
    for test in NER_TESTS:
        try:
            result = ner.recognize(test["text"])
            expected_set = set((e[0].lower(), e[1].lower()) for e in test["expected"])
            total_entities += len(expected_set)
            
            if isinstance(result, list):
                for entity in result:
                    if isinstance(entity, dict):
                        pred = (entity.get("text", "").lower(), entity.get("label", entity.get("type", "")).lower())
                    elif isinstance(entity, (list, tuple)) and len(entity) >= 2:
                        pred = (entity[0].lower(), entity[1].lower())
                    else:
                        continue
                    if pred in expected_set:
                        correct_entities += 1
        except Exception:
            total_entities += len(test["expected"])
    
    elapsed = time.perf_counter() - start
    results["manglish_nlp"] = {
        "accuracy": (correct_entities / total_entities * 100) if total_entities > 0 else 0,
        "correct_entities": correct_entities,
        "total_entities": total_entities,
        "sentences": total_sentences,
        "time_ms": elapsed * 1000,
        "avg_ms": (elapsed * 1000) / total_sentences,
    }
    
    if MALAYA_AVAILABLE:
        try:
            model = malaya.entity.transformer(model="mesolitica/ner-t5-small-standard-bahasa-cased")
            correct_m = 0
            total_m = 0
            start_m = time.perf_counter()
            
            for test in NER_TESTS:
                try:
                    result = model.predict(test["text"])
                    expected_set = set((e[0].lower(), e[1].lower()) for e in test["expected"])
                    total_m += len(expected_set)
                    for entity in result:
                        pred = (entity.get("text", "").lower(), entity.get("label", "").lower())
                        if pred in expected_set:
                            correct_m += 1
                except:
                    total_m += len(test["expected"])
            
            elapsed_m = time.perf_counter() - start_m
            results["malaya"] = {
                "accuracy": (correct_m / total_m * 100) if total_m > 0 else 0,
                "correct_entities": correct_m,
                "total_entities": total_m,
                "sentences": total_sentences,
                "time_ms": elapsed_m * 1000,
                "avg_ms": (elapsed_m * 1000) / total_sentences,
            }
        except Exception as e:
            results["malaya"] = {"error": str(e)}
    else:
        results["malaya"] = {"error": "not installed"}
    
    return results


def benchmark_stemming():
    """Benchmark stemming."""
    results = {"task": "Stemming", "manglish_nlp": {}, "malaya": {}}
    
    if not MANGLISH_NLP_AVAILABLE:
        results["manglish_nlp"] = {"error": "not installed"}
        return results
    
    correct = 0
    total = len(STEMMING_TESTS)
    start = time.perf_counter()
    
    for test in STEMMING_TESTS:
        try:
            result = stemmer.stem(test["text"])
            if result.lower() == test["expected"].lower():
                correct += 1
        except Exception:
            pass
    
    elapsed = time.perf_counter() - start
    results["manglish_nlp"] = {
        "accuracy": correct / total * 100,
        "correct": correct,
        "total": total,
        "time_ms": elapsed * 1000,
        "avg_ms": (elapsed * 1000) / total,
    }
    
    if MALAYA_AVAILABLE:
        try:
            stemmer_m = malaya.stem.deep_model(model="mesolitica/stem-t5-small-standard-bahasa-cased")
            correct_m = 0
            start_m = time.perf_counter()
            
            for test in STEMMING_TESTS:
                try:
                    result = stemmer_m.stem(test["text"])
                    if result.lower() == test["expected"].lower():
                        correct_m += 1
                except:
                    pass
            
            elapsed_m = time.perf_counter() - start_m
            results["malaya"] = {
                "accuracy": correct_m / total * 100,
                "correct": correct_m,
                "total": total,
                "time_ms": elapsed_m * 1000,
                "avg_ms": (elapsed_m * 1000) / total,
            }
        except Exception as e:
            results["malaya"] = {"error": str(e)}
    else:
        results["malaya"] = {"error": "not installed"}
    
    return results


def benchmark_normalization():
    """Benchmark text normalization."""
    results = {"task": "Normalization", "manglish_nlp": {}, "malaya": {}}
    
    if not MANGLISH_NLP_AVAILABLE:
        results["manglish_nlp"] = {"error": "not installed"}
        return results
    
    correct = 0
    total = len(NORMALIZATION_TESTS)
    start = time.perf_counter()
    
    for test in NORMALIZATION_TESTS:
        try:
            result = normalize.normalize(test["text"])
            if isinstance(result, str) and result.lower().strip() == test["expected"].lower().strip():
                correct += 1
        except Exception:
            pass
    
    elapsed = time.perf_counter() - start
    results["manglish_nlp"] = {
        "accuracy": correct / total * 100,
        "correct": correct,
        "total": total,
        "time_ms": elapsed * 1000,
        "avg_ms": (elapsed * 1000) / total,
    }
    
    if MALAYA_AVAILABLE:
        try:
            normalizer = malaya.normalize.normalizer()
            correct_m = 0
            start_m = time.perf_counter()
            
            for test in NORMALIZATION_TESTS:
                try:
                    result = normalizer.normalize(test["text"])
                    normalized = result.get("normalize", "") if isinstance(result, dict) else str(result)
                    if normalized.lower().strip() == test["expected"].lower().strip():
                        correct_m += 1
                except:
                    pass
            
            elapsed_m = time.perf_counter() - start_m
            results["malaya"] = {
                "accuracy": correct_m / total * 100,
                "correct": correct_m,
                "total": total,
                "time_ms": elapsed_m * 1000,
                "avg_ms": (elapsed_m * 1000) / total,
            }
        except Exception as e:
            results["malaya"] = {"error": str(e)}
    else:
        results["malaya"] = {"error": "not installed"}
    
    return results


def benchmark_language_detection():
    """Benchmark language detection."""
    results = {"task": "Language Detection", "manglish_nlp": {}, "malaya": {}}
    
    if not MANGLISH_NLP_AVAILABLE:
        results["manglish_nlp"] = {"error": "not installed"}
        return results
    
    correct = 0
    total = len(LANGUAGE_DETECTION_TESTS)
    start = time.perf_counter()
    
    for test in LANGUAGE_DETECTION_TESTS:
        try:
            result = language.detect(test["text"])
            detected = result.get("language", result.get("lang", "")) if isinstance(result, dict) else str(result)
            if detected.lower() == test["expected"].lower():
                correct += 1
        except Exception:
            pass
    
    elapsed = time.perf_counter() - start
    results["manglish_nlp"] = {
        "accuracy": correct / total * 100,
        "correct": correct,
        "total": total,
        "time_ms": elapsed * 1000,
        "avg_ms": (elapsed * 1000) / total,
    }
    
    if MALAYA_AVAILABLE:
        try:
            # Malaya language detection
            correct_m = 0
            start_m = time.perf_counter()
            
            for test in LANGUAGE_DETECTION_TESTS:
                try:
                    result = malaya.language_detection.fasttext()
                    pred = result.predict(test["text"])
                    # Malaya doesn't distinguish 'manglish' - it sees ms or en
                    # So we give partial credit
                    if pred.lower() == test["expected"].lower():
                        correct_m += 1
                    elif test["expected"] == "manglish" and pred.lower() in ("ms", "en", "malay"):
                        correct_m += 0.5  # Partial credit - Malaya can't detect Manglish
                except:
                    pass
            
            elapsed_m = time.perf_counter() - start_m
            results["malaya"] = {
                "accuracy": correct_m / total * 100,
                "correct": correct_m,
                "total": total,
                "time_ms": elapsed_m * 1000,
                "avg_ms": (elapsed_m * 1000) / total,
                "note": "Malaya cannot distinguish Manglish as separate category",
            }
        except Exception as e:
            results["malaya"] = {"error": str(e)}
    else:
        results["malaya"] = {"error": "not installed"}
    
    return results


def benchmark_tokenization():
    """Benchmark tokenization."""
    results = {"task": "Tokenization", "manglish_nlp": {}, "malaya": {}}
    
    if not MANGLISH_NLP_AVAILABLE:
        results["manglish_nlp"] = {"error": "not installed"}
        return results
    
    correct = 0
    total = len(TOKENIZATION_TESTS)
    start = time.perf_counter()
    
    for test in TOKENIZATION_TESTS:
        try:
            result = tokenizer.tokenize(test["text"])
            if result == test["expected"]:
                correct += 1
        except Exception:
            pass
    
    elapsed = time.perf_counter() - start
    results["manglish_nlp"] = {
        "accuracy": correct / total * 100,
        "correct": correct,
        "total": total,
        "time_ms": elapsed * 1000,
        "avg_ms": (elapsed * 1000) / total,
    }
    
    if MALAYA_AVAILABLE:
        try:
            correct_m = 0
            start_m = time.perf_counter()
            
            for test in TOKENIZATION_TESTS:
                try:
                    result = malaya.tokenizer.word(test["text"])
                    if result == test["expected"]:
                        correct_m += 1
                except:
                    pass
            
            elapsed_m = time.perf_counter() - start_m
            results["malaya"] = {
                "accuracy": correct_m / total * 100,
                "correct": correct_m,
                "total": total,
                "time_ms": elapsed_m * 1000,
                "avg_ms": (elapsed_m * 1000) / total,
            }
        except Exception as e:
            results["malaya"] = {"error": str(e)}
    else:
        results["malaya"] = {"error": "not installed"}
    
    return results

# =============================================================================
# RUNNER AND REPORTING
# =============================================================================

ALL_BENCHMARKS = {
    "sentiment": benchmark_sentiment,
    "pos": benchmark_pos,
    "ner": benchmark_ner,
    "stemming": benchmark_stemming,
    "normalization": benchmark_normalization,
    "language": benchmark_language_detection,
    "tokenization": benchmark_tokenization,
}


def run_benchmark(tasks=None):
    """Run all or selected benchmarks."""
    if tasks is None:
        tasks = list(ALL_BENCHMARKS.keys())
    
    print("=" * 70)
    print("MANGLISH-NLP vs MALAYA - Head-to-Head Benchmark")
    print("=" * 70)
    print(f"\nmanglish-nlp available: {MANGLISH_NLP_AVAILABLE}")
    print(f"Malaya available: {MALAYA_AVAILABLE}")
    print(f"Tasks to run: {', '.join(tasks)}")
    print()
    
    all_results = []
    
    for task_name in tasks:
        if task_name not in ALL_BENCHMARKS:
            print(f"[SKIP] Unknown task: {task_name}")
            continue
        
        print(f"Running: {task_name}...")
        bench_func = ALL_BENCHMARKS[task_name]
        
        # Measure memory for manglish-nlp
        tracemalloc.start()
        result = bench_func()
        _, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        result["peak_memory_kb"] = peak_mem / 1024
        
        all_results.append(result)
        
        # Print quick summary
        mnlp = result.get("manglish_nlp", {})
        mal = result.get("malaya", {})
        
        if "error" not in mnlp:
            print(f"  manglish-nlp: {mnlp.get('accuracy', 0):.1f}% accuracy, {mnlp.get('time_ms', 0):.1f}ms")
        else:
            print(f"  manglish-nlp: {mnlp['error']}")
        
        if "error" not in mal:
            print(f"  Malaya:       {mal.get('accuracy', 0):.1f}% accuracy, {mal.get('time_ms', 0):.1f}ms")
        else:
            print(f"  Malaya:       {mal.get('error', 'N/A')}")
        print()
    
    # Print markdown table
    print_results_table(all_results)
    
    return all_results


def print_results_table(results):
    """Print results as a markdown table."""
    print("\n## Results Summary\n")
    print("| Task | manglish-nlp Acc | manglish-nlp Time | Malaya Acc | Malaya Time | Winner |")
    print("|------|-----------------|-------------------|------------|-------------|--------|")
    
    for r in results:
        task = r["task"]
        mnlp = r.get("manglish_nlp", {})
        mal = r.get("malaya", {})
        
        mnlp_acc = f"{mnlp['accuracy']:.1f}%" if "accuracy" in mnlp else "N/A"
        mnlp_time = f"{mnlp['time_ms']:.1f}ms" if "time_ms" in mnlp else "N/A"
        mal_acc = f"{mal['accuracy']:.1f}%" if "accuracy" in mal else "N/A"
        mal_time = f"{mal['time_ms']:.1f}ms" if "time_ms" in mal else "N/A"
        
        # Determine winner
        if "accuracy" in mnlp and "accuracy" in mal:
            if mnlp["accuracy"] > mal["accuracy"]:
                winner = "manglish-nlp"
            elif mal["accuracy"] > mnlp["accuracy"]:
                winner = "Malaya"
            else:
                winner = "Tie"
        elif "accuracy" in mnlp:
            winner = "manglish-nlp*"
        else:
            winner = "N/A"
        
        print(f"| {task} | {mnlp_acc} | {mnlp_time} | {mal_acc} | {mal_time} | {winner} |")
    
    print("\n*Winner marked with * = other library not available for comparison")


def save_results(results, path=None):
    """Save benchmark results as markdown."""
    if path is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "RESULTS.md")
    
    lines = []
    lines.append("# Benchmark Results: manglish-nlp vs Malaya\n")
    lines.append(f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append(f"**manglish-nlp available:** {MANGLISH_NLP_AVAILABLE}\n")
    lines.append(f"**Malaya available:** {MALAYA_AVAILABLE}\n")
    lines.append("")
    
    # Summary table
    lines.append("## Summary\n")
    lines.append("| Task | manglish-nlp | Malaya | Speed Winner | Accuracy Winner |")
    lines.append("|------|-------------|--------|--------------|-----------------|")
    
    for r in results:
        task = r["task"]
        mnlp = r.get("manglish_nlp", {})
        mal = r.get("malaya", {})
        
        mnlp_str = f"{mnlp['accuracy']:.1f}% ({mnlp['time_ms']:.1f}ms)" if "accuracy" in mnlp else "Error"
        mal_str = f"{mal['accuracy']:.1f}% ({mal['time_ms']:.1f}ms)" if "accuracy" in mal else "N/A (not installed)"
        
        if "time_ms" in mnlp and "time_ms" in mal:
            speed_winner = "manglish-nlp" if mnlp["time_ms"] < mal["time_ms"] else "Malaya"
        elif "time_ms" in mnlp:
            speed_winner = "manglish-nlp*"
        else:
            speed_winner = "N/A"
        
        if "accuracy" in mnlp and "accuracy" in mal:
            acc_winner = "manglish-nlp" if mnlp["accuracy"] > mal["accuracy"] else ("Malaya" if mal["accuracy"] > mnlp["accuracy"] else "Tie")
        elif "accuracy" in mnlp:
            acc_winner = "manglish-nlp*"
        else:
            acc_winner = "N/A"
        
        lines.append(f"| {task} | {mnlp_str} | {mal_str} | {speed_winner} | {acc_winner} |")
    
    lines.append("")
    lines.append("## Detailed Results\n")
    
    for r in results:
        lines.append(f"### {r['task']}\n")
        lines.append(f"- **Peak Memory:** {r.get('peak_memory_kb', 0):.1f} KB")
        
        mnlp = r.get("manglish_nlp", {})
        if "error" not in mnlp:
            lines.append(f"- **manglish-nlp:**")
            lines.append(f"  - Accuracy: {mnlp.get('accuracy', 0):.1f}%")
            lines.append(f"  - Total time: {mnlp.get('time_ms', 0):.1f}ms")
            lines.append(f"  - Avg per item: {mnlp.get('avg_ms', 0):.3f}ms")
            if "correct" in mnlp:
                lines.append(f"  - Correct: {mnlp['correct']}/{mnlp['total']}")
        
        mal = r.get("malaya", {})
        if "error" not in mal:
            lines.append(f"- **Malaya:**")
            lines.append(f"  - Accuracy: {mal.get('accuracy', 0):.1f}%")
            lines.append(f"  - Total time: {mal.get('time_ms', 0):.1f}ms")
            lines.append(f"  - Avg per item: {mal.get('avg_ms', 0):.3f}ms")
            if "correct" in mal:
                lines.append(f"  - Correct: {mal['correct']}/{mal['total']}")
            if "note" in mal:
                lines.append(f"  - Note: {mal['note']}")
        else:
            lines.append(f"- **Malaya:** {mal.get('error', 'N/A')}")
        
        lines.append("")
    
    # Fairness disclaimer
    lines.append("## Fairness Disclaimer\n")
    lines.append("This benchmark aims to be fair to both libraries:\n")
    lines.append("- **Malaya advantages:** Deep learning models (transformers), larger training data,")
    lines.append("  more research backing, handles formal Malay better, more comprehensive API")
    lines.append("- **manglish-nlp advantages:** Zero dependencies, instant startup, Manglish-specific")
    lines.append("  patterns, dialect awareness, much faster inference (rule-based), lower memory")
    lines.append("- **Different goals:** Malaya targets formal BM NLP research; manglish-nlp targets")
    lines.append("  real-world Malaysian internet text (code-switching, slang, abbreviations)")
    lines.append("- **Test bias:** Test cases lean toward informal/Manglish text which favors manglish-nlp.")
    lines.append("  A formal BM test set would likely favor Malaya.")
    lines.append("")
    
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    
    print(f"\nResults saved to: {path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Benchmark manglish-nlp vs Malaya")
    parser.add_argument("--tasks", nargs="+", choices=list(ALL_BENCHMARKS.keys()),
                        help="Specific tasks to benchmark (default: all)")
    parser.add_argument("--save", action="store_true", help="Save results to RESULTS.md")
    parser.add_argument("--output", type=str, help="Custom output path for results")
    
    args = parser.parse_args()
    
    results = run_benchmark(tasks=args.tasks)
    
    if args.save or args.output:
        save_results(results, path=args.output)
