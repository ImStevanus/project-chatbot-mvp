import streamlit as st
import pandas as pd
import re
import json
import os
import plotly.express as px
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
    
    /* Ganti warna header tabel di Streamlit */
    .stDataFrame the-ad-hoc-table-header, 
    .stDataFrame thead th {
        background-color: #764ba2 !important; 
        color: white !important;
    }
    
    div[data-testid="stChatMessage"] { background-color: rgba(255, 255, 255, 0.1); border-radius: 10px; padding: 10px; margin-bottom: 10px; border: 1px solid rgba(255,255,255,0.2); }
    </style>
    """, unsafe_allow_html=True)

local_css()

# ==========================================
# 2. LOGIKA DATA & AI (BACKEND)
# ==========================================

# --- FUNGSI UTAMA VISUALISASI ---
def create_interest_map(results):
    """
    Membuat peta minat interaktif (Bubble Chart) berdasarkan hasil analisis.
    """
    if not results:
        st.info("Tidak ada hasil yang tersedia untuk visualisasi.")
        return

    df = pd.DataFrame(results)
    df['Similarity Score'] = pd.to_numeric(df['Similarity Score'], errors='coerce')
    df['Difficulty'] = pd.to_numeric(df['Difficulty'], errors='coerce')
    df.dropna(subset=['Similarity Score', 'Difficulty'], inplace=True)
    df['Cluster'] = df['Program'].apply(lambda x: str(x).split()[0] if isinstance(x, str) else 'Lain-lain')
    
    fig = px.scatter(
        df, 
        x='Similarity Score', 
        y='Difficulty', 
        size='Similarity Score', 
        color='Cluster', 
        hover_name='Course', 
        size_max=60, 
        title='Peta Kecocokan Mata Kuliah Berdasarkan Minat'
    )

    fig.update_layout(
        xaxis_title="Kecocokan Minat (Skor Lebih Tinggi = Lebih Baik)",
        yaxis_title="Tingkat Kesulitan (5 = Sangat Sulit)",
        showlegend=True,
        height=500
    )
    
    fig.update_xaxes(range=[0, 100])
    fig.update_yaxes(range=[0, 5])

    st.plotly_chart(fig, use_container_width=True)


# --- INISIALISASI SESSION STATE ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "bookmarks" not in st.session_state:
    st.session_state.bookmarks = []
if 'menu' not in st.session_state:
    st.session_state['menu'] = "🔍 Cari Jurusan (Database)"
# --- INISIALISASI STATE FITUR #2 (PERBANDINGAN) ---
if "compare_list" not in st.session_state:
    st.session_state.compare_list = []
if "ai_compare_request" not in st.session_state:
    st.session_state.ai_compare_request = False
# --- INISIALISASI STATE FITUR #3 (SIMULASI DAMPAK) ---
if "impact_course" not in st.session_state:
    st.session_state.impact_course = None
if "impact_result" not in st.session_state:
    st.session_state.impact_result = None
# --- INISIALISASI STATE FITUR #4 (ANALISIS JALUR) ---
if "path_query" not in st.session_state:
    st.session_state.path_query = None
if "path_analysis" not in st.session_state:
    st.session_state.path_analysis = None
# ----------------------------------------------------


# Files to persist bookmarks
BOOKMARK_JSON = 'bookmarks.json'
BOOKMARK_CSV = 'bookmarks.csv'

def load_bookmarks_from_file():
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
        with open(BOOKMARK_JSON, 'w', encoding='utf-8') as f:
            json.dump(bookmarks, f, ensure_ascii=False, indent=2)
        if bookmarks:
            pd.DataFrame(bookmarks).to_csv(BOOKMARK_CSV, index=False)
        else:
            pd.DataFrame(columns=['Course', 'Program', 'Similarity Score', 'Difficulty', 'Advice']).to_csv(BOOKMARK_CSV, index=False)
    except Exception as e:
        st.warning(f"Gagal menyimpan bookmark: {e}")

# Load persisted bookmarks into session state on startup
if not st.session_state.bookmarks:
    st.session_state.bookmarks = load_bookmarks_from_file()

@st.cache_data
def load_data():
    try:
        # PENTING: Pastikan nama file CSV ini benar
        df = pd.read_csv('List Mata Kuliah UBM.xlsx - Sheet1.csv')
        df = df.dropna(subset=['Course']) 
        df['combined_features'] = df['Course'].astype(str) + ' ' + df['Program'].astype(str)
        return df
    except FileNotFoundError:
        st.error("File data 'List Mata Kuliah UBM.xlsx - Sheet1.csv' tidak ditemukan. Pastikan sudah ada.")
        return pd.DataFrame()

# --- FUNGSI AI UNTUK TRANSLASI MINAT ---
def get_keywords_via_ai(user_query):
    try:
        if "GROQ_API_KEY" in st.secrets:
            client = Groq(api_key=st.secrets["GROQ_API_KEY"])
            
            prompt = f"""
            Tugas: Ubah input user yang santai menjadi kata kunci akademis/jurusan kuliah.
            Input User: "{user_query}"
            
            Contoh:
            - Input: "Suka makan" -> Output: kuliner tata boga food beverage hospitality
            
            Output HANYA kata kuncinya saja (dipisah spasi). Jangan ada kata pengantar.
            """
            
            completion = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=50,
            )
            return completion.choices[0].message.content
    except:
        return user_query
    return user_query

# --- Helpers Lain ---
KEYWORD_MAPPING = {
    "menggambar": "desain visual art seni fotografi kreatif sketsa ilustrasi grafis",
    "jualan": "marketing bisnis manajemen pemasaran retail sales perdagangan kewirausahaan entrepreneur",
    "ngoding": "teknologi informasi sistem komputer data algoritma programming python web software aplikasi digital",
    "hitung": "akuntansi statistika matematika ekonomi keuangan pajak finance analisis",
    "jalan-jalan": "pariwisata hospitality hotel tour travel guide tourism wisata perhotelan",
    "masak": "food beverage tata boga kitchen pastry kuliner makanan minuman chef", 
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

def expand_query(user_query):
    expanded = user_query.lower()
    for key, val in KEYWORD_MAPPING.items():
        if key in expanded: expanded += ' ' + val
    return expanded

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

# --- FUNGSI CALLBACK & LOGIKA FITUR #1, #2, #3, dan #4 ---

def bookmark_course(course, program, similarity, difficulty, advice):
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
            st.query_params = {'menu': ['bookmarks']} 
        except Exception:
            pass
    else:
        try:
            st.info("Sudah ada di Bookmark")
        except Exception:
            pass

def remove_bookmark(course):
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

def request_remove_bookmark(course):
    st.session_state['confirm_delete'] = str(course).strip()
    
def confirm_delete_yes(course):
    remove_bookmark(course)
    st.session_state['confirm_delete'] = None

def confirm_delete_no():
    st.session_state['confirm_delete'] = None

# --- FUNGSI FITUR #2 (PERBANDINGAN) ---
def toggle_compare(course):
    """Menambah atau menghapus mata kuliah dari daftar perbandingan."""
    c = str(course).strip()
    if c in st.session_state.compare_list:
        st.session_state.compare_list.remove(c)
        try:
            st.success(f"Dihapus dari Perbandingan: {c}")
        except:
            pass
    else:
        if len(st.session_state.compare_list) < 3: # Batasi maksimum 3
            st.session_state.compare_list.append(c)
            try:
                st.info(f"Ditambahkan ke Perbandingan: {c}")
            except:
                pass
        else:
            try:
                st.warning("Maksimal 3 mata kuliah untuk dibandingkan.")
            except:
                pass

def analyze_comparison_with_ai(data):
    """Meminta Groq menganalisis perbandingan data."""
    try:
        if "GROQ_API_KEY" in st.secrets:
            client = Groq(api_key=st.secrets["GROQ_API_KEY"])
            
            summary = "\n".join([f"- {d['Course']} ({d['Program']}, Kecocokan {d['Similarity Score']}%, Kesulitan {d['Difficulty']}/5). Tips: {d['Advice']}" for d in data])
            
            prompt = f"""
            Tugas: Analisis secara singkat (maksimal 3 paragraf) data mata kuliah berikut:
            {summary}

            Berikan saran final yang gaul dan dukung pengguna untuk memilih berdasarkan data di atas. Gunakan bahasa Indonesia santai dan emoji.
            """
            
            completion = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,
                max_tokens=512,
            )
            return completion.choices[0].message.content
    except Exception as e:
        return f"Gagal mendapatkan insight AI. Error: {str(e)}"
    return "Tidak ada Insight AI."

def display_comparison_table():
    if not st.session_state.compare_list:
        return

    data = [b for b in st.session_state.bookmarks if b['Course'] in st.session_state.compare_list]
    
    features = {
        'Course': 'Mata Kuliah',
        'Program': 'Jurusan',
        'Similarity Score': 'Kecocokan Minat',
        'Difficulty': 'Tingkat Kesulitan (1-5)',
        'Advice': 'Tips Sukses'
    }
    
    comparison_data = []
    
    for key_col, display_name in features.items():
        row_data = {'Fitur': display_name}
        for item in data:
            value = item.get(key_col, '-')
            
            if key_col == 'Similarity Score':
                value = f"{value}%"
            elif key_col == 'Difficulty':
                value = '★' * int(value) + '☆' * (5 - int(value))
            elif key_col == 'Advice':
                value = value[:100] + '...' if len(value) > 100 else value
                
            row_data[item['Course']] = value
        comparison_data.append(row_data)

    st.subheader(f"Perbandingan ({len(data)} dari 3)")
    df_comparison = pd.DataFrame(comparison_data).set_index('Fitur')
    
    # Mencetak tabel terbalik agar Mata Kuliah menjadi kolom (lebih mudah dibaca)
    st.dataframe(df_comparison.T, use_container_width=True) 
    
    # Tombol Analisis AI
    if len(data) >= 2:
        if st.button("🧠 Minta AI Analisis Perbandingan", type="primary"):
            st.session_state.ai_compare_request = True
            st.rerun() 
        
        if st.session_state.get('ai_compare_request'):
            with st.spinner("AI sedang menganalisis perbedaan kunci..."):
                ai_summary = analyze_comparison_with_ai(data)
                st.markdown(f"**Insight AI:**")
                st.info(ai_summary)
                # Reset state setelah analisis selesai
                st.session_state.ai_compare_request = False 

# --- FUNGSI FITUR #3 (SIMULASI DAMPAK) ---

def request_impact_simulation(course):
    """Mengatur mata kuliah mana yang akan disimulasikan."""
    st.session_state.impact_course = course
    st.session_state.impact_result = None # Reset hasil simulasi sebelumnya

def analyze_impact_with_ai(course_data, user_career_query):
    """Meminta Groq menganalisis dampak mata kuliah pada karir yang diminta pengguna.
       Output HANYA menggunakan markdown bold dan unbold.
    """
    try:
        if "GROQ_API_KEY" in st.secrets:
            client = Groq(api_key=st.secrets["GROQ_API_KEY"])
            
            summary = (
                f"Mata Kuliah: {course_data['Course']} (Jurusan: {course_data['Program']}). "
                f"Tingkat Kesulitan: {course_data['Difficulty']}/5. "
                f"Tips Sukses: {course_data['Advice']}. "
            )
            
            prompt = f"""
            Tugas: Analisis bagaimana mata kuliah berikut dapat memengaruhi tujuan karir user.
            
            Data Matkul: {summary}
            Tujuan Karir User: {user_career_query}
            
            Output analisis dalam format:
            **1. Relevansi Inti (Core Relevance):** Jelaskan keterkaitan langsung mata kuliah ini dengan tujuan karir user (poin A).
            **2. Skill yang Diperoleh (Transferable Skills):** Sebutkan minimal 3 soft/hard skill yang didapat dari matkul ini yang bermanfaat untuk karir user (poin B).
            **3. Rekomendasi Tambahan (Next Steps):** Berikan 2-3 saran konkret (misalnya: matkul pendukung, sertifikasi) untuk memaksimalkan dampak (poin C).
            **4. Proyeksi Dampak (Impact Score):** Berikan skor (0-100%) dan jelaskan mengapa.
            
            Gunakan bahasa Indonesia yang menarik dan format list. Pastikan setiap judul poin menggunakan **bold**.
            """
            
            completion = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.6,
                max_tokens=700,
            )
            return completion.choices[0].message.content
    except Exception as e:
        return f"Gagal mendapatkan simulasi dampak AI. Error: {str(e)}"
    return "Tidak ada Simulasi Dampak."

# --- FUNGSI FITUR #4 (ANALISIS JALUR BELAJAR) ---

def request_path_analysis():
    """Memunculkan modal input analisis jalur karir."""
    st.session_state.path_query = "Requesting"
    st.session_state.path_analysis = None
    
def analyze_curriculum_path(user_career_path, bookmarked_courses):
    """Meminta Groq menganalisis dan membandingkan jalur karir.
       Output HANYA menggunakan markdown bold dan unbold.
    """
    try:
        if "GROQ_API_KEY" in st.secrets:
            client = Groq(api_key=st.secrets["GROQ_API_KEY"])
            
            bookmarked_list = ", ".join([b['Course'] for b in bookmarked_courses])
            
            prompt = f"""
            Tugas: Analisis Jalur Karir dan berikan perbandingan dengan mata kuliah yang sudah disimpan pengguna.
            
            Jalur Karir yang Diminta User: **{user_career_path}**
            Mata Kuliah yang Sudah Disimpan User: {bookmarked_list if bookmarked_list else 'Tidak ada'}
            
            Output analisis dalam format:
            **1. Mata Kuliah Wajib (Core Curriculum):** Sebutkan 5-7 mata kuliah esensial (tidak perlu dari data UBM) yang mutlak dibutuhkan untuk karir **{user_career_path}**.
            **2. Analisis Kesenjangan (Gap Analysis):** Bandingkan daftar matkul wajib di atas dengan yang sudah disimpan user ({bookmarked_list}). Tunjukkan matkul yang sudah **Match** dan matkul **Gap** yang harus dicari.
            **3. Rekomendasi Tindakan (Next Step):** Berikan saran konkret bagi user (misalnya: tambahkan matkul X, fokus pada program Y).
            
            Gunakan bahasa Indonesia yang gaul dan format list/poin. Pastikan setiap judul poin menggunakan **bold**.
            """
            
            completion = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.6,
                max_tokens=800,
            )
            return completion.choices[0].message.content
    except Exception as e:
        return f"Gagal mendapatkan analisis jalur AI. Error: {str(e)}"
    return "Tidak ada Analisis Jalur."


# ==========================================
# 3. HALAMAN 1: CARI JURUSAN (REKOMENDASI)
# (Tidak Berubah)
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
            
            df_filter = df.copy()
            if sel_prog != "Semua Jurusan": 
                df_filter = df_filter[df_filter['Program'] == sel_prog]
            
            recs = get_recommendations(clean_text, df_filter, ignored)
            
            if recs.empty:
                with st.spinner("Hmm, mencari hubungan minatmu dengan jurusan yang ada..."):
                    ai_keywords = get_keywords_via_ai(clean_text)
                    st.caption(f"🤖 AI mendeteksi minat terkait: *{ai_keywords}*")
                    recs = get_recommendations(ai_keywords, df_filter, ignored)
            
            if not recs.empty:
                recs['Difficulty'] = recs['Course'].apply(get_course_difficulty)
                recs = recs[(recs['Difficulty'] >= diff_range[0]) & (recs['Difficulty'] <= diff_range[1])]
                
                if not recs.empty:
                    st.success(f"✅ Ditemukan {len(recs)} Mata Kuliah yang pas!")

                    st.header("Visualisasi Kecocokan")
                    create_interest_map(recs.to_dict('records')) 
                    st.markdown("---")
                    
                    st.header("Daftar Detail")
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
                            btn_key = f"bookmark_{i}"
                            st.button("📌 Bookmark", key=btn_key, on_click=bookmark_course, args=(row['Course'], row['Program'], row.get('Similarity Score', None), int(row['Difficulty']) if 'Difficulty' in row else get_course_difficulty(row['Course']), advice))
                else:
                    st.error("Waduh, tidak ada matkul yang cocok dengan filter kesulitanmu.")
            else:
                st.error("Waduh, database kami belum punya matkul yang cocok, meskipun sudah dibantu AI.")
                st.info("Cobalah ngobrol langsung di menu '🤖 Chat Bebas (AI)' untuk saran lebih lanjut.")

# ==========================================
# 4. HALAMAN 2: CHAT AI (FACE-TO-FACE)
# (Tidak Berubah)
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


# ==========================================
# 4.5 HALAMAN: BOOKMARKS (MATA KULIAH TERSIMPAN)
# ==========================================
def page_bookmarks():
    st.title("📜 Bookmark (Mata Kuliah Tersimpan)")
    st.markdown("Daftar mata kuliah yang kamu simpan. Gunakan tombol 'Bandingkan' untuk simulasi, dan 'Simulasi Dampak' untuk proyeksi karir.")

    if not st.session_state.bookmarks:
        st.info("Belum ada bookmark. Simpan mata kuliah dari hasil pencarian menggunakan tombol 📌.")
        st.session_state['confirm_delete'] = None
        return

    
    # --- PENGATURAN AWAL FITUR #4 (JALUR BELAJAR) ---
    col_path, col_clear = st.columns([4, 1])
    with col_path:
        # Tombol untuk memunculkan modal Analisis Jalur Karir
        if st.button("🗺️ Proyeksi Jalur Karir", type="primary"):
            request_path_analysis()
            
    with col_clear:
        # Tombol untuk mereset semua analisis
        if st.button("Bersihkan Analisis"):
            st.session_state.compare_list = []
            st.session_state.ai_compare_request = False
            st.session_state.impact_course = None
            st.session_state.impact_result = None
            st.session_state.path_query = None
            st.session_state.path_analysis = None
            st.rerun()

    st.markdown("---")

    # --- LOGIKA INPUT DAN OUTPUT FITUR #4 ---
    if st.session_state.path_query == "Requesting":
        with st.form(key='path_form'):
            career_path_query = st.text_input(
                "Tuliskan jalur karir spesifik yang kamu inginkan:", 
                placeholder="Contoh: Menjadi UI/UX Designer di E-commerce"
            )
            path_submit = st.form_submit_button("Analisis Jalur 🔍")
            
            if path_submit and career_path_query:
                with st.spinner(f"AI sedang menganalisis jalur untuk '{career_path_query}'..."):
                    analysis_result = analyze_curriculum_path(career_path_query, st.session_state.bookmarks)
                    st.session_state.path_analysis = analysis_result
                    st.session_state.path_query = "Done"
                    st.rerun()
        st.button("❌ Batal Analisis Jalur", on_click=lambda: st.session_state.update(path_query=None, path_analysis=None))
    
    if st.session_state.path_analysis and st.session_state.path_query == "Done":
        st.subheader("📊 Hasil Analisis Jalur Belajar")
        # Menggunakan st.markdown untuk menampilkan hasil AI tanpa kotak berwarna
        st.markdown(st.session_state.path_analysis) 
        st.markdown("---")

    # --- 1. TAMPILKAN TABEL PERBANDINGAN JIKA ADA ITEM (FITUR #2) ---
    if st.session_state.compare_list:
        display_comparison_table()
        st.button("❌ Bersihkan Perbandingan", on_click=lambda: st.session_state.update(compare_list=[], ai_compare_request=False))
        st.markdown("---")


    to_delete = st.session_state.get('confirm_delete')
    sim_course = st.session_state.get('impact_course')

    # --- 2. MEMULAI PERULANGAN UNTUK SETIAP BOOKMARK ---
    for i, b in enumerate(st.session_state.bookmarks):
        
        # 2.1 Tampilkan Kartu Mata Kuliah
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

        
        # 2.2 Tombol Aksi (Hapus, Bandingkan, dan SIMULASI DAMPAK)
        col_del, col_comp, col_sim = st.columns([1, 1, 1]) 

        with col_del:
            btn_key_del = f"request_remove_{i}_{re.sub(r'\\W+', '_', b['Course'])}"
            st.button("🗑️ Hapus", key=btn_key_del, on_click=request_remove_bookmark, args=(b['Course'],))

        with col_comp:
            is_comparing = b['Course'] in st.session_state.compare_list
            label = "✅ Hapus Banding" if is_comparing else "⚖️ Bandingkan"
            btn_type = "secondary" if not is_comparing else "primary"
            
            btn_key_comp = f"toggle_compare_{i}_{re.sub(r'\\W+', '_', b['Course'])}"
            st.button(label, key=btn_key_comp, on_click=toggle_compare, args=(b['Course'],), type=btn_type)

        with col_sim:
            btn_key_sim = f"request_sim_{i}_{re.sub(r'\\W+', '_', b['Course'])}"
            # Cek apakah mata kuliah ini yang sedang menunggu input simulasi
            is_simulating = sim_course == b['Course']
            sim_type = "primary" if is_simulating else "secondary"
            st.button("✨ Simulasi Dampak", key=btn_key_sim, on_click=request_impact_simulation, args=(b['Course'],), type=sim_type)

        
        # 2.3 LOGIKA KONFIRMASI (HAPUS)
        if to_delete and to_delete == b['Course']:
            safe_key_yes = f"inline_yes_{re.sub(r'\\W+', '_', to_delete)}"
            safe_key_no = f"inline_no_{re.sub(r'\\W+', '_', to_delete)}"
            
            st.warning(f"⚠️ **KONFIRMASI PENGHAPUSAN:** Yakin ingin menghapus '{to_delete}'?")
            
            col_y_inline, col_n_inline = st.columns([1,1])
            with col_y_inline:
                st.button("✅ Ya, Hapus", key=safe_key_yes, on_click=confirm_delete_yes, args=(to_delete,), type="primary")
            with col_n_inline:
                st.button("❌ Batal", key=safe_key_no, on_click=confirm_delete_no)
        
        # 2.4 LOGIKA SIMULASI DAMPAK (INPUT) - FITUR #3
        if sim_course and sim_course == b['Course']:
            st.markdown("---")
            st.subheader(f"Proyeksikan Dampak '{sim_course}'")
            
            # Form untuk Input Karir
            with st.form(key=f'sim_form_{i}'):
                career_query = st.text_input(
                    "Ingin tahu dampak mata kuliah ini ke karir apa?", 
                    placeholder="Contoh: Menjadi ahli Data Science",
                    key=f'career_input_{i}'
                )
                
                sim_submit = st.form_submit_button("Luncurkan Analisis 🚀")
                
                if sim_submit and career_query:
                    # Cari data lengkap mata kuliah
                    course_data = next((item for item in st.session_state.bookmarks if item['Course'] == sim_course), None)
                    if course_data:
                        with st.spinner(f"AI sedang memproyeksikan dampak '{sim_course}' ke karir '{career_query}'..."):
                            impact_result = analyze_impact_with_ai(course_data, career_query)
                            st.session_state.impact_result = impact_result
                            st.rerun() 
            
            # Tampilkan Hasil Simulasi
            if st.session_state.impact_result:
                st.success("✨ **Hasil Proyeksi Dampak:**")
                # Menggunakan st.markdown, hasilnya akan berwarna putih (default Streamlit)
                st.markdown(st.session_state.impact_result) 
                
            st.button("❌ Tutup Simulasi", key=f'close_sim_{i}', on_click=lambda: st.session_state.update(impact_course=None, impact_result=None))
            
        st.markdown("<hr style='border: 1px solid #333333;'>", unsafe_allow_html=True)


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
        # Pengecekan Query Params untuk navigasi (tidak berubah)
        try:
            qp = st.query_params
            if qp.get('menu') == ['bookmarks']:
                st.session_state['menu'] = "📜 Bookmark (Mata Kuliah Tersimpan)"
                try:
                    st.query_params = {}
                except Exception:
                    pass
        except Exception:
            pass

        with st.sidebar:
            st.title("Menu Aplikasi")
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
