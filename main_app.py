import streamlit as st
import pandas as pd
import re
import json
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq

# ==========================================
# 1. KONFIGURASI & CSS
# ==========================================
st.set_page_config(page_title="AI Course Advisor UBM", page_icon="🎓", layout="wide")

def local_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Segoe+UI&display=swap');
    html, body, [class*="css"] { font-family: 'Segoe UI', sans-serif; }
    
    .stApp { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); background-attachment: fixed; }
    h1, h2, h3, h4, p, label, .stMarkdown, .stChatInput { color: white !important; }
    .stChatInput textarea { background-color: #2D3748 !important; color: white !important; }
    
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] div,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] .stRadio,
    section[data-testid="stSidebar"] .stSelectbox,
    section[data-testid="stSidebar"] .stSlider {
        color: white !important;
    }
    
    .result-card { background: #f0f2f6; padding: 20px; border-radius: 15px; margin-bottom: 15px; border-left: 5px solid #667eea; }
    .result-card h3 { color: #31333F !important; margin: 0; }
    .result-card p, .result-card li { color: #31333F !important; }
    
    div[data-testid="stChatMessage"] { background-color: rgba(255, 255, 255, 0.1); border-radius: 10px; padding: 10px; margin-bottom: 10px; border: 1px solid rgba(255,255,255,0.2); }
    </style>
    """, unsafe_allow_html=True)

local_css()

# ==========================================
# 2. LOGIKA DATA & AI (BACKEND)
# ==========================================

if "messages" not in st.session_state:
    st.session_state.messages = []
if "bookmarks" not in st.session_state:
    st.session_state.bookmarks = []
if 'menu' not in st.session_state:
    st.session_state['menu'] = "🔍 Cari Jurusan (Database)"

# Files to persist bookmarks
BOOKMARK_JSON = 'bookmarks.json'
BOOKMARK_CSV = 'bookmarks.csv'

def load_bookmarks_from_file():
    # Prefer JSON, fallback to CSV
    try:
        if os.path.exists(BOOKMARK_JSON):
            with open(BOOKMARK_JSON, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data
    except Exception:
        pass
    try:
        if os.path.exists(BOOKMARK_CSV):
            df = pd.read_csv(BOOKMARK_CSV)
            return df.to_dict('records')
    except Exception:
        pass
    return []

def save_bookmarks_to_file(bookmarks):
    try:
        # JSON
        with open(BOOKMARK_JSON, 'w', encoding='utf-8') as f:
            json.dump(bookmarks, f, ensure_ascii=False, indent=2)
        # CSV
        if bookmarks:
            pd.DataFrame(bookmarks).to_csv(BOOKMARK_CSV, index=False)
        else:
            # write an empty csv with headers
            pd.DataFrame(columns=['Course', 'Program', 'Similarity Score', 'Difficulty', 'Advice']).to_csv(BOOKMARK_CSV, index=False)
    except Exception as e:
        # we don't want to crash the app for persistence errors
        st.warning(f"Gagal menyimpan bookmark: {e}")

# Load persisted bookmarks into session state on startup
if not st.session_state.bookmarks:
    st.session_state.bookmarks = load_bookmarks_from_file()

@st.cache_data
def load_data():
    try:
        df = pd.read_csv('List Mata Kuliah UBM.xlsx - Sheet1.csv')
        df = df.dropna(subset=['Course']) 
        df['combined_features'] = df['Course'].astype(str) + ' ' + df['Program'].astype(str)
        return df
    except FileNotFoundError:
        return pd.DataFrame()

# --- FUNGSI BARU: TRANSLATE NIAT USER PAKE AI ---
def get_keywords_via_ai(user_query):
    """
    Jika database tidak menemukan 'makan', fungsi ini meminta AI
    menerjemahkannya menjadi 'kuliner', 'food', 'tata boga', dll.
    """
    try:
        if "GROQ_API_KEY" in st.secrets:
            client = Groq(api_key=st.secrets["GROQ_API_KEY"])
            
            # Prompt cerdas untuk ekstraksi keyword
            prompt = f"""
            Tugas: Ubah input user yang santai menjadi kata kunci akademis/jurusan kuliah.
            Input User: "{user_query}"
            
            Contoh:
            - Input: "Suka makan" -> Output: kuliner tata boga food beverage hospitality
            - Input: "Suka debat" -> Output: hukum komunikasi hubungan internasional
            - Input: "Suka main game" -> Output: informatika desain game multimedia
            
            Output HANYA kata kuncinya saja (dipisah spasi). Jangan ada kata pengantar.
            """
            
            completion = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3, # Rendah agar fokus dan tidak ngarang
                max_tokens=50,
            )
            return completion.choices[0].message.content
    except:
        return user_query # Jika error, kembalikan query asli
    return user_query

# --- Helpers Lama ---
KEYWORD_MAPPING = {
    "menggambar": "desain visual art seni fotografi kreatif sketsa ilustrasi grafis",
    "jualan": "marketing bisnis manajemen pemasaran retail sales perdagangan kewirausahaan entrepreneur",
    "ngoding": "teknologi informasi sistem komputer data algoritma programming python web software aplikasi digital",
    "hitung": "akuntansi statistika matematika ekonomi keuangan pajak finance analisis",
    "jalan-jalan": "pariwisata hospitality hotel tour travel guide tourism wisata perhotelan",
    "masak": "food beverage tata boga kitchen pastry kuliner makanan minuman chef", # Keyword manual tetap ada sebagai backup cepat
}

def get_course_advice(course_name):
    course_lower = course_name.lower()
    if any(x in course_lower for x in ['matematika', 'statistik', 'akuntansi']):
        return "💡 Tips: Pahami konsep dasar, jangan cuma hafal rumus. Latihan soal kuncinya!"
    elif any(x in course_lower for x in ['coding', 'algoritma', 'data']):
        return "💻 Tips: Praktek (ngoding) lebih efektif daripada baca teori. Jangan takut error!"
    elif any(x in course_lower for x in ['desain', 'gambar', 'art']):
        return "🎨 Tips: Perbanyak lihat referensi (Pinterest) dan bangun portofolio."
    elif any(x in course_lower for x in ['bisnis', 'manajemen']):
        return "📊 Tips: Pelajari studi kasus nyata perusahaan dan latih skill presentasi."
    else:
        return "📝 Tips: Catat poin penting dosen dan aktif bertanya di kelas."

def get_course_difficulty(course_name):
    name = (course_name or "").lower()
    if any(k in name for k in ['matematika', 'kalkulus', 'statistika', 'fisika']): return 5
    elif any(k in name for k in ['algoritma', 'program', 'akuntansi']): return 4
    elif any(k in name for k in ['desain', 'bahasa', 'komunikasi']): return 2
    return 3


def bookmark_course(course, program, similarity, difficulty, advice):
    """Callback to add a course to bookmarks (safe for Streamlit on_click)."""
    c = str(course).strip()
    existing = [b for b in st.session_state.bookmarks if b.get('Course') == c]
    if not existing:
        st.session_state.bookmarks.append({
            'Course': c,
            'Program': program,
            'Similarity Score': similarity,
            'Difficulty': int(difficulty) if difficulty is not None else get_course_difficulty(c),
            'Advice': advice
        })
        save_bookmarks_to_file(st.session_state.bookmarks)
        try:
            # Tetap pertahankan ini jika Anda ingin otomatis pindah ke halaman bookmark setelah simpan
            st.query_params = {'menu': ['bookmarks']} 
        except Exception:
            pass
        # BLOK st.rerun() SUDAH DIHAPUS DI SINI. Streamlit akan otomatis rerun.
        
    else:
        # If already exists, keep UX consistent
        try:
            st.info("Sudah ada di Bookmark")
        except Exception:
            pass


def remove_bookmark(course):
    """Remove a bookmark by course name. Safe to call from `on_click`."""
    c = str(course).strip()
    before = len(st.session_state.bookmarks)
    st.session_state.bookmarks = [b for b in st.session_state.bookmarks if b.get('Course') != c]
    if len(st.session_state.bookmarks) < before:
        save_bookmarks_to_file(st.session_state.bookmarks)
        try:
            st.success(f"Dihapus: {c}")
        except Exception:
            pass
    else:
        try:
            st.info("Bookmark tidak ditemukan")
        except Exception:
            pass


def remove_bookmark_by_index(idx):
    """Remove a bookmark by index (safer for URL-triggered actions)."""
    try:
        idx = int(idx)
        if 0 <= idx < len(st.session_state.bookmarks):
            removed = st.session_state.bookmarks.pop(idx)
            save_bookmarks_to_file(st.session_state.bookmarks)
            try:
                st.success(f"Dihapus: {removed.get('Course')}")
            except Exception:
                pass
            return True
    except Exception:
        pass
    try:
        st.info("Bookmark tidak ditemukan")
    except Exception:
        pass
    return False


def confirm_delete_yes(course):
    """Handler for confirming deletion from modal/buttons."""
    remove_bookmark(course)
    st.session_state['confirm_delete'] = None
    try:
        st.rerun()
    except Exception:
        pass


def confirm_delete_no():
    """Handler for cancelling deletion."""
    st.session_state['confirm_delete'] = None
    try:
        st.rerun()
    except Exception:
        pass


def request_remove_bookmark(course):
    """Set a session flag to confirm bookmark deletion."""
    st.session_state['confirm_delete'] = str(course).strip()

def recommend_career_paths(courses_list):
    if not courses_list: return []
    careers = set()
    programs = [c['Program'] for c in courses_list]
    mapping = {
        'Informatika': ['Software Engineer', 'App Developer', 'Tech Lead'],
        'Data Science': ['Data Scientist', 'AI Engineer', 'Data Analyst'],
        'Bisnis': ['Entrepreneur', 'Business Manager', 'Consultant'],
        'Desain': ['Art Director', 'UI/UX Designer', 'Creative Lead'],
        'Akuntansi': ['Auditor', 'Financial Analyst', 'Tax Consultant'],
        'Hospitality': ['Hotel Manager', 'Executive Chef', 'Travel Consultant']
    }
    for prog in programs:
        for key, vals in mapping.items():
            if key in prog: careers.update(vals)
    return list(careers)[:5]

def process_negation(user_input):
    negation_patterns = [r'\b(tidak\s+suka|gak\s+suka|benci|anti)\s+(\w+)']
    cleaned_text = user_input.lower()
    words_to_remove = []
    for pattern in negation_patterns:
        matches = re.finditer(pattern, cleaned_text)
        for match in matches:
            if len(match.groups()) >= 2:
                words_to_remove.append(match.group(2))
                cleaned_text = cleaned_text.replace(match.group(0), '')
    return cleaned_text, words_to_remove

def expand_query(user_query):
    expanded = user_query.lower()
    for key, val in KEYWORD_MAPPING.items():
        if key in expanded: expanded += ' ' + val
    return expanded

def get_recommendations(user_query, df, words_to_remove=None):
    if df.empty or not user_query.strip(): return pd.DataFrame()
    df_filtered = df.copy()
    if words_to_remove:
        for word in words_to_remove:
            df_filtered = df_filtered[~df_filtered['combined_features'].str.lower().str.contains(word, na=False)]
    if df_filtered.empty: return pd.DataFrame()
    
    expanded_query = expand_query(user_query)
    vectorizer = TfidfVectorizer()
    try:
        tfidf_matrix = vectorizer.fit_transform(df_filtered['combined_features'])
        query_vec = vectorizer.transform([expanded_query])
        scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
        df_filtered['Similarity Score'] = (scores * 100).round(1)
        return df_filtered[df_filtered['Similarity Score'] > 10.0].sort_values('Similarity Score', ascending=False).head(5)
    except:
        return pd.DataFrame()

# ==========================================
# 3. HALAMAN 1: CARI JURUSAN (LENGKAP)
# ==========================================
def page_recommendation():
    st.title("🔍 Cari Jurusan & Matkul")
    st.markdown("Analisis minatmu secara mendalam berdasarkan database kampus.")
    
    df = load_data()
    
    with st.sidebar:
        st.header("Filter Pencarian")
        prog_list = ["Semua Jurusan"] + sorted(df['Program'].unique().tolist()) if not df.empty else []
        sel_prog = st.selectbox("Jurusan Spesifik:", prog_list)
        diff_range = st.slider("Filter Kesulitan (Bintang):", 1, 5, (1, 5))
    
    user_input = st.text_area("Ceritakan minatmu:", height=100, placeholder="Contoh: Saya suka banget makan...")
    
    if st.button("Analisis Minat 🚀"):
        if not user_input:
            st.warning("Isi dulu minat kamu ya!")
        else:
            st.markdown("---")
            clean_text, ignored = process_negation(user_input)
            
            # Filter Data Awal
            df_filter = df.copy()
            if sel_prog != "Semua Jurusan": 
                df_filter = df_filter[df_filter['Program'] == sel_prog]
            
            # 1. PERCOBAAN PERTAMA: Cari Langsung
            recs = get_recommendations(clean_text, df_filter, ignored)
            
            # 2. PERCOBAAN KEDUA: Jika Kosong, Panggil Bantuan AI (SMART FALLBACK)
            if recs.empty:
                with st.spinner("Hmm, mencari hubungan minatmu dengan jurusan yang ada..."):
                    # Minta AI menerjemahkan "Suka makan" -> "Kuliner Tata Boga"
                    ai_keywords = get_keywords_via_ai(clean_text)
                    st.caption(f"🤖 AI mendeteksi minat terkait: *{ai_keywords}*")
                    
                    # Cari ulang pakai kata kunci dari AI
                    recs = get_recommendations(ai_keywords, df_filter, ignored)
            
            # Tampilkan Hasil
            if not recs.empty:
                recs['Difficulty'] = recs['Course'].apply(get_course_difficulty)
                recs = recs[(recs['Difficulty'] >= diff_range[0]) & (recs['Difficulty'] <= diff_range[1])]
                
                if not recs.empty:
                    st.success(f"✅ Ditemukan {len(recs)} Mata Kuliah yang pas!")

                    # reset index to have stable small integer indices for button keys
                    recs = recs.reset_index(drop=True)
                    for i, row in recs.iterrows():
                        stars = '★' * int(row['Difficulty']) + '☆' * (5 - int(row['Difficulty']))
                        advice = get_course_advice(row['Course'])

                        st.markdown(f"""
                        <div class="result-card">
                            <h3>{row['Course']}</h3>
                            <p>🎓 {row['Program']} | ⭐ Kecocokan: {row['Similarity Score']}%</p>
                            <p>Tingkat Kesulitan: <span style="color:#f1c40f; font-size:18px;">{stars}</span></p>
                        </div>
                        """, unsafe_allow_html=True)
                        col_a, col_b = st.columns([4,1])
                        with col_a:
                            with st.expander(f"💡 Tips Sukses Mata Kuliah Ini"):
                                st.info(advice)
                        with col_b:
                            # Bookmark button uses on_click callback for correctness
                            btn_key = f"bookmark_{i}"
                            st.button("📌 Bookmark", key=btn_key, on_click=bookmark_course, args=(row['Course'], row['Program'], row.get('Similarity Score', None), int(row['Difficulty']) if 'Difficulty' in row else get_course_difficulty(row['Course']), advice))
            else:
                st.error("Waduh, database kami belum punya matkul yang cocok, meskipun sudah dibantu AI.")
                st.info("Cobalah ngobrol langsung di menu '🤖 Chat Bebas (AI)' untuk saran lebih lanjut.")

# ==========================================
# 4. HALAMAN 2: CHAT AI (FACE-TO-FACE)
# ==========================================
def page_chat_ai():
    st.title("🤖 Ngobrol Bareng AI")
    st.caption("Tanya apa saja seputar kuliah, curhat, atau tips belajar. AI akan menjawab secara real-time!")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Tanya sesuatu..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            try:
                if "GROQ_API_KEY" in st.secrets:
                    client = Groq(api_key=st.secrets["GROQ_API_KEY"])
                    
                    messages_payload = [
                        {"role": "system", "content": "Kamu adalah Advisor Kampus UBM yang gaul, seru, dan suportif. Gunakan bahasa Indonesia santai dan emoji."}
                    ] + [
                        {"role": m["role"], "content": m["content"]} for m in st.session_state.messages
                    ]
                    
                    completion = client.chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        messages=messages_payload,
                        temperature=0.7,
                        max_tokens=1024,
                        stream=True, 
                    )
                    
                    for chunk in completion:
                        if chunk.choices[0].delta.content:
                            full_response += chunk.choices[0].delta.content
                            message_placeholder.markdown(full_response + "▌")
                    message_placeholder.markdown(full_response)
                else:
                    full_response = "⚠️ API Key Groq belum dipasang."
                    message_placeholder.error(full_response)
            except Exception as e:
                full_response = f"Error: {str(e)}"
                message_placeholder.error(full_response)
            
            st.session_state.messages.append({"role": "assistant", "content": full_response})

# --- LOGIKA BACKEND YANG PERLU DIUBAH ---

def confirm_delete_yes(course):
    """Handler for confirming deletion from modal/buttons."""
    remove_bookmark(course)
    st.session_state['confirm_delete'] = None

def confirm_delete_no():
    """Handler for cancelling deletion."""
    st.session_state['confirm_delete'] = None

# ==========================================
# 4.5 HALAMAN: BOOKMARKS (MATA KULIAH TERSIMPAN)
# ==========================================
def page_bookmarks():
    st.title("📜 Bookmark (Mata Kuliah Tersimpan)")
    st.markdown("Daftar mata kuliah yang kamu simpan. Hapus jika sudah tidak perlu.")

    if not st.session_state.bookmarks:
        st.info("Belum ada bookmark. Simpan mata kuliah dari hasil pencarian menggunakan tombol 📌.")
        st.session_state['confirm_delete'] = None
        return

    # Ambil nama mata kuliah yang sedang menunggu konfirmasi hapus
    to_delete = st.session_state.get('confirm_delete')

    # --- MEMULAI PERULANGAN UNTUK SETIAP BOOKMARK ---
    for i, b in enumerate(st.session_state.bookmarks):
        
        # 1. Tampilkan Kartu Mata Kuliah
        stars = '★' * int(b.get('Difficulty', 3)) + '☆' * (5 - int(b.get('Difficulty', 3)))
        st.markdown(f"""
        <div class="result-card">
            <h3>{b['Course']}</h3>
            <p>🎓 {b.get('Program', '-')} | ⭐ Kecocokan: {b.get('Similarity Score', '-')}%</p>
            <p>Tingkat Kesulitan: <span style="color:#f1c40f; font-size:18px;">{stars}</span></p>
        </div>
        """, unsafe_allow_html=True)

        with st.expander("💡 Tips Sukses Mata Kuliah Ini"):
            st.info(b.get('Advice', get_course_advice(b['Course'])))

        btn_key = f"request_remove_{i}_{re.sub(r'\\W+', '_', b['Course'])}"
        
        # 2. Tampilkan Tombol Hapus Bookmark
        st.button("🗑️ Hapus Bookmark", key=btn_key, on_click=request_remove_bookmark, args=(b['Course'],))

        
        # --- 3. LOGIKA KONFIRMASI (BERADA DI DALAM LOOP) ---
        # Konfirmasi hanya ditampilkan jika:
        # a. Ada item yang akan dihapus (to_delete is not None)
        # b. Item yang akan dihapus SAMA dengan mata kuliah saat ini (b['Course'])
        if to_delete and to_delete == b['Course']:
            safe_key_yes = f"inline_yes_{re.sub(r'\\W+', '_', to_delete)}"
            safe_key_no = f"inline_no_{re.sub(r'\\W+', '_', to_delete)}"
            
            # Catatan: Kita menggunakan fallback inline karena st.modal (jika tersedia)
            # adalah pop-up yang mengambang, bukan elemen yang bisa diletakkan di bawah course.

            st.warning(f"⚠️ **KONFIRMASI PENGHAPUSAN:** Yakin ingin menghapus '{to_delete}'?")
            
            # Tombol "Ya, Hapus"
            col_y_inline, col_n_inline = st.columns([1,1])
            with col_y_inline:
                st.button("✅ Ya, Hapus", key=safe_key_yes, on_click=confirm_delete_yes, args=(to_delete,), type="primary")
            with col_n_inline:
                st.button("❌ Batal", key=safe_key_no, on_click=confirm_delete_no)
        
        st.markdown("<hr style='border: 1px solid #333333;'>", unsafe_allow_html=True)


    # --- PENTING: HAPUS SEMUA LOGIKA KONFIRMASI YANG SEBELUMNYA ADA DI BAWAH LOOP ---
    # Jika sebelumnya ada blok konfirmasi di luar perulangan, pastikan sudah dihapus.

# ==========================================
# 5. NAVIGASI UTAMA
# ==========================================
def main():
    if 'app_started' not in st.session_state:
        st.session_state['app_started'] = False

    if not st.session_state['app_started']:
        st.markdown("""
            <div style="text-align: center; padding: 50px; color: white;">
                <div style="font-size: 80px; margin-bottom:20px;">🎓</div>
                <h1 style="color:white !important; font-size: 3em;">AI Course Advisor</h1>
                <p style="color:white !important; font-size: 1.2em;">Konsultan Akademik Pribadimu, 24/7.</p>
            </div>
        """, unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1,1,1])
        with col2:
            if st.button("Mulai Konsultasi 🚀", use_container_width=True):
                st.session_state['app_started'] = True
                st.rerun()
    else:
        # Check query params for navigation fallback BEFORE widgets are created
        try:
            qp = st.query_params
            # handle confirm yes/no via query params (used by HTML fallback modal)
            if 'confirm_index' in qp:
                try:
                    raw = qp.get('confirm_index', [None])[0]
                    if raw is not None:
                        from urllib.parse import unquote_plus
                        idx_raw = unquote_plus(raw)
                        remove_bookmark_by_index(idx_raw)
                        st.session_state['confirm_delete'] = None
                except Exception:
                    pass
                finally:
                    try:
                        st.query_params = {}
                    except Exception:
                        pass
                try:
                    st.rerun()
                except Exception:
                    pass
            elif 'confirm_yes' in qp:
                # backward-compatible: try deletion by name if index not provided
                try:
                    raw = qp.get('confirm_yes', [None])[0]
                    if raw:
                        from urllib.parse import unquote_plus
                        course_to_delete = unquote_plus(raw)
                        remove_bookmark(course_to_delete)
                        st.session_state['confirm_delete'] = None
                except Exception:
                    pass
                finally:
                    try:
                        st.query_params = {}
                    except Exception:
                        pass
                try:
                    st.rerun()
                except Exception:
                    pass
            if 'confirm_no' in qp:
                # user cancelled via HTML fallback; clear state
                try:
                    st.session_state['confirm_delete'] = None
                except Exception:
                    pass
                finally:
                    try:
                        st.query_params = {}
                    except Exception:
                        pass
                try:
                    st.rerun()
                except Exception:
                    pass

            if qp.get('menu') == ['bookmarks']:
                st.session_state['menu'] = "📜 Bookmark (Mata Kuliah Tersimpan)"
                # clear query params
                try:
                    st.query_params = {}
                except Exception:
                    pass
        except Exception:
            # In some runtimes, query_params may not be available; ignore
            pass

        with st.sidebar:
            st.title("Menu Aplikasi")
            # bind radio directly to session_state key 'menu' so we can programmatically change it
            st.radio("Pilih Mode:", ["🔍 Cari Jurusan (Database)", "🤖 Chat Bebas (AI)", "📜 Bookmark (Mata Kuliah Tersimpan)"], key='menu')
            st.markdown("---")
            if st.button("🏠 Kembali ke Depan"):
                st.session_state['app_started'] = False
                st.rerun()

        menu = st.session_state.get('menu', "🔍 Cari Jurusan (Database)")
        if menu == "🔍 Cari Jurusan (Database)":
            page_recommendation()
        elif menu == "🤖 Chat Bebas (AI)":
            page_chat_ai()
        elif menu == "📜 Bookmark (Mata Kuliah Tersimpan)":
            page_bookmarks()

if __name__ == "__main__":
    main()
