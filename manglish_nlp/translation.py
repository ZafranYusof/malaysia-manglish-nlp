"""Rule-based translation between BM, EN, and Manglish. Zero dependencies beyond manglish_nlp."""

import re
from manglish_nlp.language import detect_language
from manglish_nlp.normalize import normalize as normalize_shortforms
from manglish_nlp.utils import get_shortforms, get_particles


# --- Bilingual Dictionary (1000+ pairs) ---

_BM_TO_EN = {
    # Pronouns
    'saya': 'I', 'aku': 'I', 'kami': 'we', 'kita': 'we',
    'awak': 'you', 'kamu': 'you', 'anda': 'you', 'engkau': 'you', 'kau': 'you',
    'dia': 'he/she', 'beliau': 'he/she', 'mereka': 'they',
    'ini': 'this', 'itu': 'that', 'sini': 'here', 'situ': 'there',
    # Verbs - common
    'ada': 'have', 'adalah': 'is', 'ialah': 'is',
    'buat': 'do', 'membuat': 'make', 'dibuat': 'made',
    'pergi': 'go', 'datang': 'come', 'balik': 'return', 'pulang': 'return',
    'makan': 'eat', 'minum': 'drink', 'tidur': 'sleep', 'bangun': 'wake up',
    'kerja': 'work', 'bekerja': 'work', 'belajar': 'study', 'mengajar': 'teach',
    'beli': 'buy', 'membeli': 'buy', 'jual': 'sell', 'menjual': 'sell',
    'bayar': 'pay', 'membayar': 'pay', 'terima': 'receive', 'menerima': 'receive',
    'baca': 'read', 'membaca': 'read', 'tulis': 'write', 'menulis': 'write',
    'dengar': 'hear', 'mendengar': 'hear', 'tengok': 'watch', 'lihat': 'see',
    'melihat': 'see', 'nampak': 'see', 'cari': 'find', 'mencari': 'search',
    'ambil': 'take', 'mengambil': 'take', 'letak': 'put', 'meletakkan': 'put',
    'hantar': 'send', 'menghantar': 'send', 'sampai': 'arrive',
    'duduk': 'sit', 'berdiri': 'stand', 'jalan': 'walk', 'berjalan': 'walk',
    'lari': 'run', 'berlari': 'run', 'terbang': 'fly',
    'masuk': 'enter', 'keluar': 'exit', 'naik': 'go up', 'turun': 'go down',
    'buka': 'open', 'membuka': 'open', 'tutup': 'close', 'menutup': 'close',
    'mula': 'start', 'memulakan': 'start', 'habis': 'finish', 'selesai': 'finish',
    'tahu': 'know', 'mengetahui': 'know', 'kenal': 'recognize',
    'ingat': 'remember', 'lupa': 'forget', 'faham': 'understand',
    'fikir': 'think', 'berfikir': 'think', 'rasa': 'feel',
    'suka': 'like', 'menyukai': 'like', 'cinta': 'love', 'mencintai': 'love',
    'benci': 'hate', 'takut': 'fear', 'malu': 'shy',
    'tolong': 'help', 'menolong': 'help', 'membantu': 'help',
    'cuba': 'try', 'mencuba': 'try', 'boleh': 'can', 'dapat': 'get',
    'mahu': 'want', 'nak': 'want', 'hendak': 'want', 'ingin': 'want',
    'perlu': 'need', 'memerlukan': 'need', 'harus': 'must', 'mesti': 'must',
    'bagi': 'give', 'memberi': 'give', 'memberikan': 'give',
    'kata': 'say', 'berkata': 'say', 'cakap': 'speak', 'bercakap': 'speak',
    'tanya': 'ask', 'bertanya': 'ask', 'jawab': 'answer', 'menjawab': 'answer',
    'tunggu': 'wait', 'menunggu': 'wait', 'ikut': 'follow', 'mengikut': 'follow',
    'guna': 'use', 'menggunakan': 'use', 'pakai': 'wear/use',
    'simpan': 'keep', 'menyimpan': 'keep', 'buang': 'throw', 'membuang': 'throw',
    'cuci': 'wash', 'mencuci': 'wash', 'masak': 'cook', 'memasak': 'cook',
    'potong': 'cut', 'memotong': 'cut', 'campuran': 'mix',
    'main': 'play', 'bermain': 'play', 'menang': 'win', 'kalah': 'lose',
    'nyanyi': 'sing', 'menyanyi': 'sing', 'menari': 'dance', 'tari': 'dance',
    'renang': 'swim', 'berenang': 'swim', 'panjat': 'climb',
    'pukul': 'hit', 'tendang': 'kick', 'lempar': 'throw',
    'tarik': 'pull', 'tolak': 'push', 'angkat': 'lift',
    'jatuh': 'fall', 'terjatuh': 'fell', 'pecah': 'break', 'rosak': 'broken',
    'hidup': 'live', 'mati': 'die', 'lahir': 'born',
    'kahwin': 'marry', 'berkahwin': 'marry', 'cerai': 'divorce',
    'sakit': 'sick', 'sembuh': 'recover', 'ubat': 'medicine',
    # Nouns - people
    'orang': 'person', 'manusia': 'human', 'lelaki': 'man', 'perempuan': 'woman',
    'budak': 'kid', 'kanak': 'child', 'bayi': 'baby',
    'ibu': 'mother', 'bapa': 'father', 'ayah': 'father',
    'abang': 'brother', 'kakak': 'sister', 'adik': 'sibling',
    'anak': 'child', 'cucu': 'grandchild', 'nenek': 'grandmother', 'atuk': 'grandfather',
    'kawan': 'friend', 'sahabat': 'friend', 'jiran': 'neighbour',
    'guru': 'teacher', 'cikgu': 'teacher', 'pelajar': 'student', 'murid': 'student',
    'doktor': 'doctor', 'jururawat': 'nurse', 'polis': 'police',
    'pekerja': 'worker', 'bos': 'boss', 'pengurus': 'manager',
    # Nouns - places
    'rumah': 'house', 'bilik': 'room', 'dapur': 'kitchen', 'tandas': 'toilet',
    'sekolah': 'school', 'universiti': 'university', 'kolej': 'college',
    'pejabat': 'office', 'kedai': 'shop', 'pasar': 'market',
    'hospital': 'hospital', 'klinik': 'clinic', 'farmasi': 'pharmacy',
    'masjid': 'mosque', 'gereja': 'church', 'kuil': 'temple',
    'stesen': 'station', 'lapangan': 'field', 'taman': 'park/garden',
    'bandar': 'city', 'kampung': 'village', 'negara': 'country', 'negeri': 'state',
    'jalan': 'road', 'lorong': 'lane', 'lebuh': 'highway',
    'pantai': 'beach', 'gunung': 'mountain', 'sungai': 'river', 'laut': 'sea',
    'hutan': 'forest', 'padang': 'field', 'sawah': 'rice field',
    # Nouns - things
    'benda': 'thing', 'barang': 'item', 'alat': 'tool',
    'buku': 'book', 'kertas': 'paper', 'pen': 'pen', 'pensel': 'pencil',
    'telefon': 'phone', 'komputer': 'computer', 'televisyen': 'television',
    'kereta': 'car', 'motosikal': 'motorcycle', 'bas': 'bus', 'teksi': 'taxi',
    'kapal': 'ship', 'bot': 'boat', 'pesawat': 'airplane',
    'pintu': 'door', 'tingkap': 'window', 'dinding': 'wall', 'lantai': 'floor',
    'kerusi': 'chair', 'meja': 'table', 'katil': 'bed', 'almari': 'cupboard',
    'baju': 'shirt', 'seluar': 'pants', 'kasut': 'shoes', 'topi': 'hat',
    'duit': 'money', 'wang': 'money', 'harga': 'price', 'gaji': 'salary',
    'makanan': 'food', 'minuman': 'drink', 'air': 'water',
    'nasi': 'rice', 'roti': 'bread', 'ayam': 'chicken', 'ikan': 'fish',
    'sayur': 'vegetable', 'buah': 'fruit', 'gula': 'sugar', 'garam': 'salt',
    'susu': 'milk', 'teh': 'tea', 'kopi': 'coffee',
    # Nouns - abstract
    'masa': 'time', 'waktu': 'time', 'hari': 'day', 'malam': 'night',
    'pagi': 'morning', 'petang': 'evening', 'tengahari': 'afternoon',
    'minggu': 'week', 'bulan': 'month', 'tahun': 'year',
    'kerja': 'work', 'tugas': 'task', 'projek': 'project',
    'soalan': 'question', 'jawapan': 'answer', 'masalah': 'problem',
    'cara': 'way', 'kaedah': 'method', 'sebab': 'reason',
    'nama': 'name', 'umur': 'age', 'tempat': 'place',
    'cerita': 'story', 'berita': 'news', 'maklumat': 'information',
    'ilmu': 'knowledge', 'pengalaman': 'experience',
    # Adjectives
    'baik': 'good', 'bagus': 'good', 'buruk': 'bad', 'teruk': 'terrible',
    'besar': 'big', 'kecil': 'small', 'panjang': 'long', 'pendek': 'short',
    'tinggi': 'tall', 'rendah': 'low', 'lebar': 'wide', 'sempit': 'narrow',
    'banyak': 'many', 'sedikit': 'few', 'sikit': 'little',
    'baru': 'new', 'lama': 'old', 'muda': 'young', 'tua': 'old',
    'cantik': 'beautiful', 'hodoh': 'ugly', 'kacak': 'handsome',
    'pandai': 'smart', 'bodoh': 'stupid', 'rajin': 'hardworking', 'malas': 'lazy',
    'senang': 'easy', 'susah': 'difficult', 'mudah': 'easy', 'sukar': 'difficult',
    'cepat': 'fast', 'lambat': 'slow', 'laju': 'fast',
    'panas': 'hot', 'sejuk': 'cold', 'hangat': 'warm',
    'basah': 'wet', 'kering': 'dry', 'bersih': 'clean', 'kotor': 'dirty',
    'terang': 'bright', 'gelap': 'dark', 'cerah': 'clear',
    'keras': 'hard', 'lembut': 'soft', 'kasar': 'rough', 'licin': 'smooth',
    'mahal': 'expensive', 'murah': 'cheap', 'percuma': 'free',
    'betul': 'correct', 'salah': 'wrong', 'benar': 'true', 'palsu': 'false',
    'penuh': 'full', 'kosong': 'empty', 'berat': 'heavy', 'ringan': 'light',
    'kuat': 'strong', 'lemah': 'weak', 'sihat': 'healthy',
    'gembira': 'happy', 'sedih': 'sad', 'marah': 'angry',
    'penat': 'tired', 'lapar': 'hungry', 'haus': 'thirsty', 'kenyang': 'full',
    'selamat': 'safe', 'bahaya': 'dangerous', 'penting': 'important',
    'sama': 'same', 'lain': 'different', 'semua': 'all',
    # Colors
    'merah': 'red', 'biru': 'blue', 'hijau': 'green', 'kuning': 'yellow',
    'hitam': 'black', 'putih': 'white', 'coklat': 'brown',
    'kelabu': 'grey', 'ungu': 'purple', 'oren': 'orange', 'merah jambu': 'pink',
    # Numbers
    'satu': 'one', 'dua': 'two', 'tiga': 'three', 'empat': 'four',
    'lima': 'five', 'enam': 'six', 'tujuh': 'seven', 'lapan': 'eight',
    'sembilan': 'nine', 'sepuluh': 'ten', 'sebelas': 'eleven',
    'seratus': 'hundred', 'seribu': 'thousand', 'sejuta': 'million',
    'pertama': 'first', 'kedua': 'second', 'ketiga': 'third',
    # Time words
    'sekarang': 'now', 'semalam': 'yesterday', 'esok': 'tomorrow',
    'hari ini': 'today', 'tadi': 'just now', 'nanti': 'later',
    'selalu': 'always', 'kadang': 'sometimes', 'jarang': 'rarely', 'tidak pernah': 'never',
    # Conjunctions & prepositions
    'dan': 'and', 'atau': 'or', 'tetapi': 'but', 'tapi': 'but',
    'untuk': 'for', 'dengan': 'with', 'dari': 'from', 'daripada': 'from',
    'dalam': 'in', 'luar': 'outside', 'atas': 'above', 'bawah': 'below',
    'depan': 'front', 'belakang': 'behind', 'sebelah': 'beside',
    'antara': 'between', 'tanpa': 'without', 'tentang': 'about',
    'kepada': 'to', 'pada': 'at/on', 'oleh': 'by',
    'kalau': 'if', 'jika': 'if', 'supaya': 'so that', 'kerana': 'because',
    'walaupun': 'although', 'sehingga': 'until', 'sambil': 'while',
    'sebelum': 'before', 'selepas': 'after', 'semasa': 'during',
    # Question words
    'apa': 'what', 'siapa': 'who', 'mana': 'where', 'bila': 'when',
    'kenapa': 'why', 'mengapa': 'why', 'bagaimana': 'how', 'macam mana': 'how',
    'berapa': 'how many/much',
    # Adverbs
    'sangat': 'very', 'amat': 'very', 'terlalu': 'too',
    'agak': 'quite', 'hampir': 'almost', 'sudah': 'already',
    'belum': 'not yet', 'masih': 'still', 'juga': 'also', 'pun': 'also',
    'lagi': 'more/again', 'sahaja': 'only', 'hanya': 'only',
    'memang': 'indeed', 'mungkin': 'maybe', 'pasti': 'certainly',
    # Negation
    'tidak': 'not', 'tak': 'not', 'bukan': 'not', 'tiada': 'none',
    'jangan': 'do not',
    # Manglish informal
    'best': 'great', 'gila': 'crazy', 'power': 'awesome',
    'syok': 'enjoyable', 'lepak': 'hang out', 'mamak': 'mamak restaurant',
    'tapau': 'takeaway', 'gostan': 'reverse', 'potong': 'cut queue',
    'cincai': 'careless', 'kiasu': 'competitive', 'kaypoh': 'nosy',
    'paiseh': 'embarrassed', 'shiok': 'great', 'jom': 'let us go',
    'lah': '', 'la': '', 'lor': '', 'meh': '', 'geh': '',
    'wei': '', 'weh': '', 'kan': '', 'kot': 'maybe',
    'je': 'only', 'jer': 'only', 'aje': 'only',
}

# Phrases (multi-word, checked before word-by-word)
_BM_PHRASES_TO_EN = {
    'terima kasih': 'thank you',
    'sama-sama': 'you are welcome',
    'apa khabar': 'how are you',
    'selamat pagi': 'good morning',
    'selamat petang': 'good afternoon',
    'selamat malam': 'good night',
    'selamat tinggal': 'goodbye',
    'selamat datang': 'welcome',
    'selamat jalan': 'goodbye',
    'jumpa lagi': 'see you again',
    'tidak apa': 'it is okay',
    'tak apa': 'it is okay',
    'macam mana': 'how',
    'hari ini': 'today',
    'hari tu': 'that day',
    'lepas tu': 'after that',
    'pasal apa': 'why',
    'kat mana': 'where',
    'berapa harga': 'how much',
    'tak boleh': 'cannot',
    'tidak boleh': 'cannot',
    'sudah makan': 'already ate',
    'belum lagi': 'not yet',
    'saya nak': 'I want',
    'aku nak': 'I want',
    'boleh tak': 'can or not',
    'minta maaf': 'sorry',
    'tolong saya': 'help me',
    'saya suka': 'I like',
    'tak tahu': 'do not know',
    'tidak tahu': 'do not know',
    'ada tak': 'is there',
    'siapa nama': 'what is your name',
    'berapa umur': 'how old',
    'dari mana': 'from where',
    'pergi mana': 'where are you going',
    'buat apa': 'what are you doing',
    'kenapa tak': 'why not',
    'jangan risau': 'do not worry',
    'tak payah': 'no need',
    'suka hati': 'as you wish',
    'makan angin': 'go on holiday',
    'naik angin': 'get angry',
    'ambil hati': 'win someone over',
    'jaga diri': 'take care',
    'sakit hati': 'heartache',
    'keras kepala': 'stubborn',
    'ringan tulang': 'hardworking',
    'berat mulut': 'quiet person',
}

_EN_PHRASES_TO_BM = {v: k for k, v in _BM_PHRASES_TO_EN.items()}

# Build reverse dictionary (EN -> BM)
_EN_TO_BM = {}
for bm, en in _BM_TO_EN.items():
    if en and en not in _EN_TO_BM and bm not in ('lah', 'la', 'lor', 'meh', 'geh', 'wei', 'weh', 'kan'):
        _EN_TO_BM[en.lower()] = bm

# Additional EN -> BM entries not covered by reverse
_EN_TO_BM.update({
    'i': 'saya', 'me': 'saya', 'my': 'saya', 'mine': 'milik saya',
    'you': 'anda', 'your': 'anda', 'yours': 'milik anda',
    'he': 'dia', 'him': 'dia', 'his': 'dia',
    'she': 'dia', 'her': 'dia',
    'it': 'ia', 'its': 'ia',
    'we': 'kami', 'us': 'kami', 'our': 'kami',
    'they': 'mereka', 'them': 'mereka', 'their': 'mereka',
    'the': '', 'a': '', 'an': '',
    'is': 'adalah', 'am': 'adalah', 'are': 'adalah',
    'was': 'telah', 'were': 'telah',
    'be': 'menjadi', 'been': 'telah',
    'have': 'mempunyai', 'has': 'mempunyai', 'had': 'mempunyai',
    'do': 'buat', 'does': 'buat', 'did': 'telah buat',
    'will': 'akan', 'would': 'akan', 'shall': 'akan',
    'can': 'boleh', 'could': 'boleh', 'may': 'boleh',
    'should': 'patut', 'must': 'mesti', 'might': 'mungkin',
    'go': 'pergi', 'going': 'pergi', 'went': 'pergi', 'gone': 'pergi',
    'come': 'datang', 'came': 'datang',
    'eat': 'makan', 'ate': 'makan', 'eaten': 'makan',
    'drink': 'minum', 'drank': 'minum',
    'sleep': 'tidur', 'slept': 'tidur',
    'work': 'kerja', 'worked': 'bekerja',
    'buy': 'beli', 'bought': 'membeli',
    'sell': 'jual', 'sold': 'menjual',
    'give': 'beri', 'gave': 'memberi', 'given': 'diberikan',
    'take': 'ambil', 'took': 'mengambil', 'taken': 'diambil',
    'make': 'buat', 'made': 'dibuat',
    'say': 'kata', 'said': 'berkata',
    'tell': 'beritahu', 'told': 'memberitahu',
    'think': 'fikir', 'thought': 'berfikir',
    'know': 'tahu', 'knew': 'mengetahui',
    'see': 'lihat', 'saw': 'melihat', 'seen': 'dilihat',
    'want': 'mahu', 'wanted': 'mahu',
    'need': 'perlu', 'needed': 'memerlukan',
    'like': 'suka', 'liked': 'menyukai',
    'love': 'cinta', 'loved': 'mencintai',
    'hate': 'benci', 'hated': 'membenci',
    'help': 'tolong', 'helped': 'menolong',
    'try': 'cuba', 'tried': 'mencuba',
    'ask': 'tanya', 'asked': 'bertanya',
    'answer': 'jawab', 'answered': 'menjawab',
    'wait': 'tunggu', 'waited': 'menunggu',
    'run': 'lari', 'ran': 'berlari',
    'walk': 'jalan', 'walked': 'berjalan',
    'sit': 'duduk', 'sat': 'duduk',
    'stand': 'berdiri', 'stood': 'berdiri',
    'open': 'buka', 'opened': 'membuka',
    'close': 'tutup', 'closed': 'menutup',
    'read': 'baca', 'write': 'tulis', 'wrote': 'menulis',
    'play': 'main', 'played': 'bermain',
    'live': 'tinggal', 'lived': 'tinggal',
    'die': 'mati', 'died': 'meninggal',
    'good': 'baik', 'bad': 'buruk', 'great': 'hebat',
    'big': 'besar', 'small': 'kecil', 'long': 'panjang', 'short': 'pendek',
    'new': 'baru', 'old': 'lama', 'young': 'muda',
    'hot': 'panas', 'cold': 'sejuk', 'warm': 'hangat',
    'fast': 'cepat', 'slow': 'lambat',
    'easy': 'senang', 'hard': 'susah', 'difficult': 'sukar',
    'happy': 'gembira', 'sad': 'sedih', 'angry': 'marah',
    'beautiful': 'cantik', 'ugly': 'hodoh', 'handsome': 'kacak',
    'smart': 'pandai', 'stupid': 'bodoh',
    'rich': 'kaya', 'poor': 'miskin',
    'clean': 'bersih', 'dirty': 'kotor',
    'safe': 'selamat', 'dangerous': 'bahaya',
    'important': 'penting', 'interesting': 'menarik',
    'house': 'rumah', 'home': 'rumah', 'room': 'bilik',
    'school': 'sekolah', 'office': 'pejabat',
    'car': 'kereta', 'bus': 'bas', 'train': 'keretapi',
    'food': 'makanan', 'water': 'air', 'rice': 'nasi',
    'money': 'duit', 'price': 'harga',
    'time': 'masa', 'day': 'hari', 'night': 'malam',
    'morning': 'pagi', 'evening': 'petang', 'afternoon': 'tengahari',
    'week': 'minggu', 'month': 'bulan', 'year': 'tahun',
    'today': 'hari ini', 'tomorrow': 'esok', 'yesterday': 'semalam',
    'now': 'sekarang', 'later': 'nanti', 'soon': 'tidak lama lagi',
    'always': 'selalu', 'sometimes': 'kadang-kadang', 'never': 'tidak pernah',
    'here': 'sini', 'there': 'situ',
    'yes': 'ya', 'no': 'tidak', 'maybe': 'mungkin',
    'please': 'sila', 'sorry': 'maaf', 'thanks': 'terima kasih',
    'thank': 'terima kasih', 'welcome': 'selamat datang',
    'hello': 'helo', 'hi': 'hai', 'bye': 'selamat tinggal',
    'friend': 'kawan', 'family': 'keluarga',
    'mother': 'ibu', 'father': 'ayah', 'brother': 'abang', 'sister': 'kakak',
    'son': 'anak lelaki', 'daughter': 'anak perempuan',
    'man': 'lelaki', 'woman': 'perempuan', 'boy': 'budak lelaki', 'girl': 'budak perempuan',
    'person': 'orang', 'people': 'orang ramai',
    'name': 'nama', 'age': 'umur', 'place': 'tempat',
    'book': 'buku', 'phone': 'telefon', 'computer': 'komputer',
    'door': 'pintu', 'window': 'tingkap', 'table': 'meja', 'chair': 'kerusi',
    'red': 'merah', 'blue': 'biru', 'green': 'hijau', 'yellow': 'kuning',
    'black': 'hitam', 'white': 'putih',
    'and': 'dan', 'or': 'atau', 'but': 'tetapi',
    'if': 'jika', 'because': 'kerana', 'so': 'jadi',
    'very': 'sangat', 'too': 'terlalu', 'also': 'juga',
    'not': 'tidak', 'no': 'tidak', 'never': 'tidak pernah',
    'all': 'semua', 'every': 'setiap', 'some': 'beberapa',
    'many': 'banyak', 'much': 'banyak', 'few': 'sedikit',
    'more': 'lebih', 'less': 'kurang', 'most': 'paling',
    'only': 'sahaja', 'just': 'sahaja',
    'what': 'apa', 'who': 'siapa', 'where': 'mana',
    'when': 'bila', 'why': 'kenapa', 'how': 'bagaimana',
    'this': 'ini', 'that': 'itu', 'these': 'ini', 'those': 'itu',
    'with': 'dengan', 'without': 'tanpa', 'for': 'untuk',
    'from': 'dari', 'to': 'ke', 'in': 'dalam', 'on': 'atas',
    'at': 'di', 'by': 'oleh', 'about': 'tentang',
    'before': 'sebelum', 'after': 'selepas', 'during': 'semasa',
    'between': 'antara', 'under': 'bawah', 'over': 'atas',
    'again': 'lagi', 'still': 'masih', 'already': 'sudah',
    'enough': 'cukup', 'together': 'bersama',
})

# Manglish particles to remove during translation
_PARTICLES = {'la', 'lah', 'lor', 'leh', 'meh', 'geh', 'kan', 'eh', 'weh', 'wei', 'ah', 'oh', 'oi'}

# Informal -> formal BM map for to_formal
_INFORMAL_TO_FORMAL = {
    'aku': 'saya', 'ak': 'saya', 'aq': 'saya',
    'ko': 'anda', 'kau': 'anda', 'hang': 'anda', 'korg': 'kamu semua',
    'korang': 'kamu semua', 'dorang': 'mereka', 'diorang': 'mereka',
    'kitorg': 'kami', 'kitorang': 'kami',
    'nak': 'ingin', 'nk': 'ingin',
    'pegi': 'pergi', 'gi': 'pergi', 'pgi': 'pergi',
    'balik': 'pulang', 'blk': 'pulang',
    'cakap': 'berkata', 'ckp': 'berkata',
    'bagi': 'memberikan', 'bg': 'memberikan',
    'amik': 'mengambil', 'ambik': 'mengambil',
    'letak': 'meletakkan', 'ltk': 'meletakkan',
    'hantar': 'menghantar', 'hntr': 'menghantar',
    'tolong': 'membantu', 'tlg': 'membantu',
    'tengok': 'melihat', 'tgk': 'melihat',
    'suruh': 'mengarahkan', 'srh': 'mengarahkan',
    'jap': 'sebentar', 'kjap': 'sebentar',
    'skrg': 'sekarang', 'skang': 'sekarang',
    'dah': 'telah', 'dh': 'telah',
    'blm': 'belum', 'blum': 'belum',
    'nnt': 'nanti', 'nnti': 'nanti',
    'lepas': 'selepas', 'lps': 'selepas',
    'x': 'tidak', 'tak': 'tidak', 'tk': 'tidak',
    'xde': 'tiada', 'takde': 'tiada',
    'xblh': 'tidak boleh', 'takleh': 'tidak boleh',
    'je': 'sahaja', 'jer': 'sahaja', 'aje': 'sahaja',
    'kot': 'mungkin',
    'macam': 'seperti', 'mcm': 'seperti',
    'sebab': 'kerana', 'sbb': 'kerana', 'pasal': 'kerana', 'psl': 'kerana',
    'tapi': 'tetapi', 'tp': 'tetapi',
    'dgn': 'dengan', 'ngn': 'dengan',
    'utk': 'untuk', 'tuk': 'untuk',
    'yg': 'yang',
    'ni': 'ini', 'tu': 'itu',
    'kat': 'di', 'dkt': 'di', 'dekat': 'di',
    'byk': 'banyak', 'bnyk': 'banyak',
    'skit': 'sedikit', 'sket': 'sedikit', 'sikit': 'sedikit',
    'mmg': 'memang', 'sgt': 'sangat',
    'lg': 'lagi', 'lgi': 'lagi',
    'pn': 'juga', 'pon': 'juga', 'pun': 'juga',
    'cmne': 'bagaimana', 'camne': 'bagaimana', 'cemana': 'bagaimana',
    'ape': 'apa', 'pe': 'apa',
    'sape': 'siapa', 'spe': 'siapa',
    'bile': 'bila', 'ble': 'bila',
    'mane': 'mana', 'mne': 'mana',
    'gak': 'juga', 'gk': 'juga',
    'btl': 'betul', 'btul': 'betul',
    'mmg': 'memang', 'emg': 'memang',
    'org': 'orang', 'owg': 'orang',
    'ngan': 'dengan', 'ngn': 'dengan',
    'bnde': 'benda', 'bnd': 'benda',
    'smpi': 'sampai', 'smpai': 'sampai',
    'dlm': 'dalam',
    'luar': 'di luar',
    'dpn': 'depan',
    'blkg': 'belakang',
}


def _tokenize_text(text):
    """Split text into words preserving punctuation."""
    return re.findall(r"[\w']+|[^\w\s]", text)


def _translate_phrase_check(text_lower, phrase_dict):
    """Check and replace known phrases in text."""
    result = text_lower
    for phrase, translation in sorted(phrase_dict.items(), key=lambda x: -len(x[0])):
        if phrase in result:
            result = result.replace(phrase, translation)
    return result


def _compute_confidence(original_words, translated_words):
    """Compute translation confidence based on % of words successfully translated."""
    if not original_words:
        return 0.0
    changed = sum(1 for o, t in zip(original_words, translated_words) if o.lower() != t.lower())
    # Words that were particles (removed) also count as translated
    total = len(original_words)
    if total == 0:
        return 0.0
    # Confidence = ratio of words we could translate or handle
    known = changed + (total - len(translated_words))
    return min(1.0, round(known / total, 2)) if total > 0 else 0.0


def translate(text, source='auto', target='en'):
    """Translate text between BM, EN, and Manglish.

    Parameters:
        text (str): Input text to translate.
        source (str): Source language - 'bm', 'en', 'manglish', or 'auto'.
        target (str): Target language - 'bm', 'en', or 'formal'.

    Returns:
        dict: {"translated": str, "source_lang": str, "target_lang": str, "confidence": float}

    Example:
        >>> translate("saya nak pergi kedai", target='en')
        {'translated': 'I want go shop', 'source_lang': 'bm', 'target_lang': 'en', 'confidence': 0.8}
    """
    if not text or not text.strip():
        return {"translated": "", "source_lang": source, "target_lang": target, "confidence": 0.0}

    # Auto-detect source language
    if source == 'auto':
        detection = detect_language(text)
        source = detection.get('language', 'manglish')
        if source not in ('bm', 'en', 'manglish'):
            source = 'manglish'

    text_lower = text.lower().strip()

    if target == 'formal':
        result = _translate_to_formal(text)
        words_orig = text_lower.split()
        words_trans = result.split()
        confidence = _compute_confidence(words_orig, words_trans)
        return {"translated": result, "source_lang": source, "target_lang": "formal_bm", "confidence": confidence}

    if target == 'en':
        result = _translate_to_english(text_lower, source)
    elif target == 'bm':
        result = _translate_to_malay(text_lower, source)
    else:
        result = text

    words_orig = text_lower.split()
    words_trans = result.split()
    confidence = _compute_confidence(words_orig, words_trans)

    return {"translated": result, "source_lang": source, "target_lang": target, "confidence": confidence}


def _translate_to_english(text_lower, source):
    """Internal: translate BM/Manglish text to English."""
    # First check phrases
    result = _translate_phrase_check(text_lower, _BM_PHRASES_TO_EN)

    # Normalize shortforms if source is manglish
    if source == 'manglish':
        try:
            result = normalize_shortforms(result)
        except Exception:
            pass

    # Word-by-word translation
    tokens = _tokenize_text(result)
    translated = []
    for token in tokens:
        lower_token = token.lower()
        if lower_token in _PARTICLES:
            continue  # Remove particles
        if lower_token in _BM_TO_EN:
            en_word = _BM_TO_EN[lower_token]
            if en_word:  # Skip empty (particles already handled)
                translated.append(en_word)
        else:
            translated.append(token)  # Keep unknown words as-is

    output = ' '.join(translated)
    # Clean up punctuation spacing
    output = re.sub(r'\s+([.,!?;:])', r'\1', output)
    return output


def _translate_to_malay(text_lower, source):
    """Internal: translate EN text to BM."""
    # Check phrases first
    result = _translate_phrase_check(text_lower, _EN_PHRASES_TO_BM)

    # Word-by-word translation
    tokens = _tokenize_text(result)
    translated = []
    for token in tokens:
        lower_token = token.lower()
        if lower_token in _EN_TO_BM:
            bm_word = _EN_TO_BM[lower_token]
            if bm_word:  # Skip articles mapped to empty
                translated.append(bm_word)
        else:
            translated.append(token)  # Keep unknown words as-is

    output = ' '.join(translated)
    output = re.sub(r'\s+([.,!?;:])', r'\1', output)
    return output


def _translate_to_formal(text):
    """Internal: convert Manglish/informal BM to formal BM."""
    # First normalize shortforms
    try:
        normalized = normalize_shortforms(text.lower())
    except Exception:
        normalized = text.lower()

    words = normalized.split()
    result = []

    for word in words:
        punct = ''
        clean_word = word
        if word and word[-1] in '.,!?;:':
            punct = word[-1]
            clean_word = word[:-1]

        if clean_word in _PARTICLES:
            continue  # Remove particles
        elif clean_word in _INFORMAL_TO_FORMAL:
            formal = _INFORMAL_TO_FORMAL[clean_word]
            if formal:
                result.append(formal + punct)
        else:
            result.append(word)

    output = ' '.join(result)

    # Capitalize first letter
    if output:
        output = output[0].upper() + output[1:]

    # Capitalize after period
    output = re.sub(r'\.\s+([a-z])', lambda m: '. ' + m.group(1).upper(), output)

    # Ensure ends with punctuation
    if output and output[-1] not in '.!?':
        output += '.'

    # Clean double spaces
    output = re.sub(r'\s+', ' ', output).strip()

    return output


def to_english(text):
    """Translate BM/Manglish text to English.

    Parameters:
        text (str): BM or Manglish text.

    Returns:
        str: English translation.

    Example:
        >>> to_english("saya nak makan nasi")
        'I want eat rice'
    """
    result = translate(text, source='auto', target='en')
    return result['translated']


def to_malay(text):
    """Translate EN/Manglish text to formal BM.

    Parameters:
        text (str): English or Manglish text.

    Returns:
        str: Formal BM translation.

    Example:
        >>> to_malay("I want to go home")
        'saya mahu ke pergi rumah'
    """
    result = translate(text, source='auto', target='bm')
    return result['translated']


def to_formal(text):
    """Convert Manglish/informal BM to formal BM.

    Expands shortforms, removes particles, replaces informal with formal words.

    Parameters:
        text (str): Informal Manglish/BM text.

    Returns:
        str: Formal BM text.

    Example:
        >>> to_formal("aku nk pegi kedai jap")
        'Saya ingin pergi kedai sebentar.'
    """
    result = translate(text, source='manglish', target='formal')
    return result['translated']


def word_translate(word, target='en'):
    """Translate a single word.

    Parameters:
        word (str): Single word to translate.
        target (str): Target language - 'en' or 'bm'.

    Returns:
        str or None: Translation if found, None otherwise.

    Example:
        >>> word_translate("rumah", target='en')
        'house'
        >>> word_translate("house", target='bm')
        'rumah'
    """
    lower = word.lower().strip()
    if target == 'en':
        return _BM_TO_EN.get(lower)
    elif target == 'bm':
        return _EN_TO_BM.get(lower)
    return None


def detect_and_translate(text):
    """Auto-detect source language and translate to the other language.

    If source is BM/Manglish -> translates to EN.
    If source is EN -> translates to BM.

    Parameters:
        text (str): Input text.

    Returns:
        dict: {"original": str, "translated": str, "source_lang": str,
               "target_lang": str, "confidence": float}

    Example:
        >>> detect_and_translate("saya suka makan")
        {'original': 'saya suka makan', 'translated': 'I like eat', ...}
    """
    if not text or not text.strip():
        return {
            "original": text,
            "translated": "",
            "source_lang": "unknown",
            "target_lang": "unknown",
            "confidence": 0.0,
        }

    detection = detect_language(text)
    source = detection.get('language', 'manglish')

    if source == 'en':
        target = 'bm'
    else:
        target = 'en'

    result = translate(text, source=source, target=target)

    return {
        "original": text,
        "translated": result['translated'],
        "source_lang": result['source_lang'],
        "target_lang": result['target_lang'],
        "confidence": result['confidence'],
    }
