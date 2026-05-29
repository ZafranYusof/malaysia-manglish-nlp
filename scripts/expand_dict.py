"""Expand shortforms dictionary to 500+ entries."""
import json
import os

DICT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'manglish_nlp', 'resources', 'dictionary.json')

with open(DICT_PATH, 'r', encoding='utf-8') as f:
    data = json.load(f)

# New shortforms to add (covering SMS, social media, gaming, daily conversation)
new_shortforms = {
    # Common daily
    'ape': 'apa', 'apo': 'apa', 'pe': 'apa', 'nape': 'kenapa', 'napew': 'kenapa',
    'sape': 'siapa', 'spe': 'siapa', 'cmne': 'macam mana', 'cne': 'macam mana',
    'mcmne': 'macam mana', 'gne': 'macam mana', 'mne': 'mana', 'mna': 'mana',
    'bile': 'bila', 'bla': 'bila', 'bpe': 'berapa', 'bpe2': 'berapa',
    
    # Pronouns extended
    'aq': 'aku', 'sy': 'saya', 'awk': 'awak', 'kowg': 'korang',
    'kwg': 'korang', 'ktorg': 'kitorang', 'ktowg': 'kitorang',
    'dorg': 'diorang', 'dowg': 'diorang', 'drg': 'diorang',
    
    # Verbs extended
    'pgi': 'pergi', 'dtg': 'datang', 'dtng': 'datang', 'blk': 'balik',
    'blik': 'balik', 'mkn': 'makan', 'mnm': 'minum', 'tdo': 'tidur',
    'tdr': 'tidur', 'bgun': 'bangun', 'bgn': 'bangun', 'kje': 'kerja',
    'krje': 'kerja', 'blaja': 'belajar', 'blajr': 'belajar', 'bjr': 'belajar',
    'hntr': 'hantar', 'htr': 'hantar', 'amek': 'ambil', 'amk': 'ambil',
    'tgk': 'tengok', 'tngk': 'tengok', 'dgr': 'dengar', 'dngar': 'dengar',
    'ckp': 'cakap', 'ckap': 'cakap', 'jwb': 'jawab', 'jwab': 'jawab',
    'tlg': 'tolong', 'tlng': 'tolong', 'mnt': 'minta', 'mnta': 'minta',
    'srh': 'suruh', 'sruh': 'suruh', 'bwt': 'buat', 'bt': 'buat',
    'gne': 'guna', 'pkai': 'pakai', 'pki': 'pakai', 'bke': 'buka',
    'ttup': 'tutup', 'ttp': 'tutup', 'msk': 'masuk', 'kluar': 'keluar',
    'klr': 'keluar', 'nk': 'naik', 'trn': 'turun', 'trun': 'turun',
    'jln': 'jalan', 'lri': 'lari', 'ddk': 'duduk', 'duk': 'duduk',
    'brdri': 'berdiri', 'smpai': 'sampai', 'smpi': 'sampai',
    'igt': 'ingat', 'ingt': 'ingat', 'lpe': 'lupa', 'lpa': 'lupa',
    'fhm': 'faham', 'phm': 'faham', 'thu': 'tahu', 'tw': 'tahu',
    'knal': 'kenal', 'knl': 'kenal', 'cbe': 'cuba', 'cba': 'cuba',
    'trime': 'terima', 'trma': 'terima', 'byr': 'bayar', 'bya': 'bayar',
    'bli': 'beli', 'jl': 'jual', 'pnjm': 'pinjam', 'pjm': 'pinjam',
    'smpn': 'simpan', 'smpen': 'simpan', 'bng': 'buang',
    
    # Adjectives extended
    'cntik': 'cantik', 'cntk': 'cantik', 'hdp': 'hodoh', 'pdai': 'pandai',
    'pndai': 'pandai', 'bdh': 'bodoh', 'bdoh': 'bodoh', 'rjin': 'rajin',
    'rjn': 'rajin', 'mls': 'malas', 'pnt': 'penat', 'pnat': 'penat',
    'sdp': 'sedap', 'sdap': 'sedap', 'mhl': 'mahal', 'mrh': 'murah',
    'ssh': 'susah', 'ssah': 'susah', 'sng': 'senang', 'snang': 'senang',
    'cpt': 'cepat', 'cpat': 'cepat', 'lmbt': 'lambat', 'lbt': 'lambat',
    'dkt': 'dekat', 'juh': 'jauh', 'bsr': 'besar', 'bsar': 'besar',
    'kcik': 'kecil', 'kcl': 'kecil', 'tggi': 'tinggi', 'tgi': 'tinggi',
    'rndh': 'rendah', 'pjg': 'panjang', 'pnjg': 'panjang',
    'pndk': 'pendek', 'pdk': 'pendek', 'bru': 'baru', 'lme': 'lama',
    'lma': 'lama', 'mde': 'muda', 'tue': 'tua',
    'pns': 'panas', 'sjuk': 'sejuk', 'sjk': 'sejuk',
    'bsh': 'basah', 'krg': 'kering', 'brsh': 'bersih', 'ktr': 'kotor',
    'grm': 'geram', 'bngang': 'bengang', 'bngng': 'bengang',
    
    # Nouns extended
    'org': 'orang', 'owg': 'orang', 'bdk': 'budak', 'bdak': 'budak',
    'kwn': 'kawan', 'kwan': 'kawan', 'rmh': 'rumah', 'umh': 'rumah',
    'sklh': 'sekolah', 'skola': 'sekolah', 'kdi': 'kedai', 'kdai': 'kedai',
    'keta': 'kereta', 'krete': 'kereta', 'fon': 'telefon', 'hp': 'telefon',
    'duit': 'duit', 'dwit': 'duit', 'hrge': 'harga', 'hrg': 'harga',
    'gji': 'gaji', 'mse': 'masa', 'wktu': 'waktu', 'wkt': 'waktu',
    'hri': 'hari', 'mggu': 'minggu', 'mgu': 'minggu', 'bln': 'bulan',
    'thn': 'tahun', 'thun': 'tahun', 'tmpt': 'tempat', 'tpt': 'tempat',
    'blik': 'bilik', 'blk2': 'bilik', 'pjbt': 'pejabat', 'opis': 'pejabat',
    'mknn': 'makanan', 'mnmn': 'minuman', 'bju': 'baju', 'sluar': 'seluar',
    'kski': 'kasut', 'ksut': 'kasut', 'beg': 'beg', 'tket': 'tiket',
    
    # Time extended
    'pg': 'pagi', 'pgi2': 'pagi', 'ptg': 'petang', 'ptng': 'petang',
    'mlm': 'malam', 'mlam': 'malam', 'tgh': 'tengah', 'tghari': 'tengahari',
    'smlm': 'semalam', 'smalam': 'semalam', 'esk': 'esok', 'bsk': 'esok',
    'lps': 'lepas', 'lpas': 'lepas', 'sblm': 'sebelum', 'sblum': 'sebelum',
    'slps': 'selepas', 'slpas': 'selepas', 'skrg': 'sekarang', 'skang': 'sekarang',
    'kjap': 'kejap', 'jap': 'kejap', 'jjap': 'kejap',
    'lme2': 'lama-lama', 'slalu': 'selalu', 'sllu': 'selalu',
    'kdg2': 'kadang-kadang', 'kdng': 'kadang-kadang',
    
    # Negation/modals extended
    'xde': 'takde', 'xda': 'takda', 'xblh': 'tak boleh', 'xleh': 'tak boleh',
    'xnk': 'tak nak', 'xmo': 'tak mahu', 'xpe': 'takpe', 'xpew': 'takpe',
    'xksh': 'tak kisah', 'xksah': 'tak kisah', 'xthu': 'tak tahu',
    'xtw': 'tak tahu', 'xphm': 'tak faham', 'xfhm': 'tak faham',
    'xsmpat': 'tak sempat', 'xsmpt': 'tak sempat',
    'blh': 'boleh', 'bole': 'boleh', 'dpt': 'dapat', 'dpat': 'dapat',
    'kne': 'kena', 'kna': 'kena', 'wjb': 'wajib', 'ptut': 'patut',
    'ptot': 'patut', 'mgkn': 'mungkin', 'mgkin': 'mungkin',
    
    # Connectors extended
    'sbb': 'sebab', 'sbab': 'sebab', 'psl': 'pasal', 'psal': 'pasal',
    'tp': 'tapi', 'tpi': 'tapi', 'dgn': 'dengan', 'ngn': 'dengan',
    'utk': 'untuk', 'tuk': 'untuk', 'ntuk': 'untuk',
    'dlm': 'dalam', 'dlam': 'dalam', 'lgi': 'lagi', 'lg': 'lagi',
    'pn': 'pun', 'pon': 'pun', 'jgk': 'jugak', 'jgak': 'jugak',
    'mmg': 'memang', 'mng': 'memang', 'sgt': 'sangat', 'sngt': 'sangat',
    'agk': 'agak', 'krg': 'kurang', 'kurg': 'kurang', 'lbh': 'lebih',
    'lbih': 'lebih', 'plg': 'paling', 'pling': 'paling',
    
    # Particles/fillers
    'jela': 'je la', 'jgn': 'jangan', 'jngn': 'jangan',
    'cmtu': 'macam tu', 'cmni': 'macam ni', 'gtu': 'gitu',
    'gni': 'gini', 'gler': 'giler', 'glr': 'giler',
    
    # Social media / internet
    'dm': 'direct message', 'pm': 'private message', 'ig': 'instagram',
    'fb': 'facebook', 'tw': 'twitter', 'yt': 'youtube', 'tt': 'tiktok',
    'wp': 'whatsapp', 'tele': 'telegram', 'gc': 'group chat',
    'pfp': 'profile picture', 'bio': 'biography', 'acc': 'account',
    'notif': 'notification', 'noti': 'notification',
    'ss': 'screenshot', 'vid': 'video', 'pic': 'picture',
    'repost': 'repost', 'rt': 'retweet', 'fav': 'favourite',
    
    # Gaming/internet slang
    'gg': 'good game', 'wp': 'well played', 'ez': 'easy',
    'noob': 'noob', 'pro': 'professional', 'afk': 'away from keyboard',
    'brb': 'be right back', 'gtg': 'got to go', 'g2g': 'got to go',
    'idk': 'i dont know', 'idc': 'i dont care', 'nvm': 'nevermind',
    'tbh': 'to be honest', 'imo': 'in my opinion', 'fyi': 'for your information',
    'btw': 'by the way', 'omg': 'oh my god', 'lmao': 'laughing',
    'rofl': 'laughing', 'smh': 'shaking my head', 'fml': 'frustrated',
    'tfw': 'that feeling when', 'mfw': 'my face when',
    'asap': 'as soon as possible', 'eta': 'estimated time arrival',
    'otw': 'on the way', 'omw': 'on my way',
    
    # Emotions/reactions
    'hpy': 'happy', 'sdih': 'sedih', 'sdh': 'sedih',
    'mrah': 'marah', 'mrh2': 'marah', 'tkt': 'takut', 'tkut': 'takut',
    'mlu': 'malu', 'malu2': 'malu', 'rndu': 'rindu', 'rndo': 'rindu',
    'syg': 'sayang', 'syng': 'sayang', 'bnci': 'benci', 'bci': 'benci',
    'jles': 'jeles', 'jls': 'jeles', 'cmbru': 'cemburu', 'cmburu': 'cemburu',
    
    # Food/daily life
    'nsik': 'nasi', 'nsk': 'nasi', 'lauk': 'lauk', 'lk': 'lauk',
    'kpi': 'kopi', 'teh': 'teh', 'aym': 'ayam', 'ikn': 'ikan',
    'syur': 'sayur', 'bh': 'buah', 'tlur': 'telur', 'tlr': 'telur',
    'grg': 'goreng', 'greng': 'goreng', 'rbs': 'rebus', 'bkr': 'bakar',
    
    # Places
    'msjd': 'masjid', 'msjid': 'masjid', 'hsptal': 'hospital',
    'hsptl': 'hospital', 'stesyen': 'stesen', 'stsn': 'stesen',
    'lpgn': 'lapangan', 'psr': 'pasar', 'psar': 'pasar',
    'bndr': 'bandar', 'bdr': 'bandar', 'kmpg': 'kampung', 'kpg': 'kampung',
    
    # Education
    'uni': 'universiti', 'kolej': 'kolej', 'klj': 'kolej',
    'kls': 'kelas', 'klas': 'kelas', 'subjek': 'subjek', 'sbjk': 'subjek',
    'asgmnt': 'assignment', 'asgn': 'assignment', 'hmwk': 'homework',
    'lect': 'lecture', 'lectr': 'lecturer', 'prof': 'professor',
    'sem': 'semester', 'smstr': 'semester', 'xm': 'exam', 'exm': 'exam',
    'rslt': 'result', 'grd': 'grade', 'cgpa': 'cgpa', 'ptr': 'pointer',
    
    # Work
    'bos': 'boss', 'mgr': 'manager', 'clg': 'colleague',
    'mtg': 'meeting', 'mting': 'meeting', 'ddline': 'deadline',
    'prjct': 'project', 'prjk': 'projek', 'rprt': 'report',
    'slry': 'salary', 'ot': 'overtime', 'mc': 'medical certificate',
    'lve': 'leave', 'cti': 'cuti', 'rsign': 'resign',
    
    # Transport
    'tren': 'train', 'trn2': 'train', 'bss': 'bas', 'teksi': 'teksi',
    'grb': 'grab', 'mrt': 'mrt', 'lrt': 'lrt', 'ktm': 'ktm',
    'jm': 'jam', 'trfik': 'trafik', 'trfk': 'trafik',
    'prkng': 'parking', 'prkg': 'parking', 'tol': 'tol',
    
    # Shopping
    'brg': 'barang', 'brng': 'barang', 'dskaun': 'diskaun', 'dskn': 'diskaun',
    'prmo': 'promo', 'sle': 'sale', 'byr': 'bayar', 'csh': 'cash',
    'trf': 'transfer', 'bnk': 'bank', 'atm': 'atm',
    
    # Health
    'skt': 'sakit', 'skit': 'sakit', 'dmm': 'demam', 'dmam': 'demam',
    'btk': 'batuk', 'btuk': 'batuk', 'slsma': 'selesema', 'slsm': 'selesema',
    'pning': 'pening', 'png': 'pening', 'mntah': 'muntah', 'mntk': 'muntah',
    'ubt': 'ubat', 'ubat': 'ubat', 'klnik': 'klinik', 'klnk': 'klinik',
    'dktr': 'doktor', 'dktor': 'doktor',
    
    # Weather
    'pns': 'panas', 'sjuk': 'sejuk', 'hjn': 'hujan', 'hujn': 'hujan',
    'mndg': 'mendung', 'mdung': 'mendung', 'rbut': 'ribut', 'rbt': 'ribut',
    'crah': 'cerah', 'crh': 'cerah',
    
    # Misc common
    'sbnrnya': 'sebenarnya', 'sbnarnya': 'sebenarnya', 'sbnr': 'sebenar',
    'btl': 'betul', 'btul': 'betul', 'slh': 'salah', 'slah': 'salah',
    'msalah': 'masalah', 'mslh': 'masalah', 'hal': 'hal',
    'cte': 'cerita', 'crita': 'cerita', 'crt': 'cerita',
    'jwpn': 'jawapan', 'jwpan': 'jawapan', 'slsai': 'selesai', 'slsai': 'selesai',
    'hbis': 'habis', 'hbs': 'habis', 'mle': 'mula', 'mla': 'mula',
    'smpai': 'sampai', 'smpi': 'sampai', 'dri': 'dari', 'dr': 'dari',
    'sni': 'sini', 'stu': 'situ', 'sne': 'sana',
    'cmpur': 'campur', 'cmpr': 'campur', 'pcah': 'pecah', 'pch': 'pecah',
    'rsmi': 'rasmi', 'prsn': 'perasaan', 'prsan': 'perasaan',
    'pglmn': 'pengalaman', 'pglman': 'pengalaman',
    'pndpt': 'pendapat', 'pndpat': 'pendapat',
    'kputusn': 'keputusan', 'kptsn': 'keputusan',
    'msyrt': 'masyarakat', 'msyrkt': 'masyarakat',
    'kluarge': 'keluarga', 'klrga': 'keluarga', 'klg': 'keluarga',
}

# Merge (don't overwrite existing)
added = 0
for k, v in new_shortforms.items():
    if k not in data['shortforms']:
        data['shortforms'][k] = v
        added += 1

data['version'] = '3.0.0'

with open(DICT_PATH, 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"Added {added} new shortforms. Total: {len(data['shortforms'])}")
