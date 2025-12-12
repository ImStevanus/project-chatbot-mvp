import streamlit as st
import pandas as pd
import re
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
    
    /* Background & Text Colors */
    .stApp { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); background-attachment: fixed; }
    h1, h2, h3, h4, p, label, .stMarkdown, .stChatInput { color: white !important; }
    .stChatInput textarea { background-color: #2D3748 !important; color: white !important; }
    
    /* Sidebar Text Fix */
    section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] p, section[data-testid="stSidebar"] label, section[data-testid="stSidebar"] div, section[data-testid="stSidebar"] span { color: #2d3748 !important; }
    
    /* Card Style untuk Hasil */
    .result-card { background: #f0f2f6; padding: 20px; border-radius: 15px; margin-bottom: 15px; border-left: 5px solid #667eea; }
    .result-card h3 { color: #31333F !important; margin: 0; }
    .result-card p, .result-card li { color: #31333F !important; }
    
    /* Chat Bubble */
    div[data-testid="stChatMessage"] { background-color: rgba(255, 255, 255, 0.1); border-radius: 10px; padding: 10px; margin-bottom: 10px; border: 1px solid rgba(255,255,255,0.2); }
    </style>
    """, unsafe_allow_html=True)

local_css()

# ==========================================
# 2. LOGIKA DATA & AI (BACKEND)
# ==========================================

if "messages" not in st.session_state:
    st.session_state.messages = []

@st.cache_data
def load_data():
    try:
        df = pd.read_csv('List Mata Kuliah UBM.xlsx - Sheet1.csv')
        df = df.dropna(subset=['Course']) 
        df['combined_features'] = df['Course'].astype(str) + ' ' + df['Program'].astype(str)
        return df
    except FileNotFoundError:
        return pd.DataFrame()

# --- Helpers Lama (Dikembalikan) ---
KEYWORD_MAPPING = {
    "menggambar": "desain visual art seni fotografi kreatif sketsa ilustrasi grafis",
    "jualan": "marketing bisnis manajemen pemasaran retail sales perdagangan kewirausahaan entrepreneur",
    "ngoding": "teknologi informasi sistem komputer data algoritma programming python web software aplikasi digital",
    "hitung": "akuntansi statistika matematika ekonomi keuangan pajak finance analisis",
    "jalan-jalan": "pariwisata hospitality hotel tour travel guide tourism wisata perhotelan",
    "masak": "food beverage tata boga kitchen pastry kuliner makanan minuman chef",
    "game": "game development interactive design programming unity multimedia",
    "duit": "investasi keuangan bisnis entrepreneur kaya",
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
    
    user_input = st.text_area("Ceritakan minatmu:", height=100, placeholder="Contoh: Saya suka bisnis tapi gak mau yang banyak hitungan rumit...")
    
    if st.button("Analisis Minat 🚀"):
        if not user_input:
            st.warning("Isi dulu minat kamu ya!")
        else:
            st.markdown("---")
            clean_text, ignored = process_negation(user_input)
            
            # Filter Logic
            df_filter = df.copy()
            if sel_prog != "Semua Jurusan": 
                df_filter = df_filter[df_filter['Program'] == sel_prog]
            
            recs = get_recommendations(clean_text, df_filter, ignored)
            
            if not recs.empty:
                # Filter Difficulty
                recs['Difficulty'] = recs['Course'].apply(get_course_difficulty)
                recs = recs[(recs['Difficulty'] >= diff_range[0]) & (recs['Difficulty'] <= diff_range[1])]
                
                if not recs.empty:
                    st.success(f"✅ Ditemukan {len(recs)} Mata Kuliah yang pas!")
                    
                    for idx, row in recs.iterrows():
                        stars = '★' * int(row['Difficulty']) + '☆' * (5 - int(row['Difficulty']))
                        advice = get_course_advice(row['Course'])
                        
                        st.markdown(f"""
                        <div class="result-card">
                            <h3>{row['Course']}</h3>
                            <p>🎓 {row['Program']} | ⭐ Kecocokan: {row['Similarity Score']}%</p>
                            <p>Tingkat Kesulitan: <span style="color:#f1c40f; font-size:18px;">{stars}</span></p>
                        </div>
                        """, unsafe_allow_html=True)
                        with st.expander(f"💡 Tips Sukses Mata Kuliah Ini"):
                            st.info(advice)
                    
                    careers = recommend_career_paths(recs.to_dict('records'))
                    if careers:
                        st.markdown("### 💼 Prospek Karir Masa Depan:")
                        st.write(", ".join(careers))
                else:
                    st.warning("Ada yang cocok, tapi tingkat kesulitannya di luar filter kamu.")
            else:
                st.warning("Belum menemukan yang pas. Coba jelaskan lebih detail atau gunakan menu Chat AI!")

# ==========================================
# 4. HALAMAN 2: CHAT AI (FACE-TO-FACE)
# ==========================================
def page_chat_ai():
    st.title("🤖 Ngobrol Bareng AI")
    st.caption("Tanya apa saja seputar kuliah, curhat, atau tips belajar. AI akan menjawab secara real-time!")

    # Tampilkan History
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Input Chat
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
                    
                    # Context AI
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
# 5. NAVIGASI UTAMA
# ==========================================
def main():
    if 'app_started' not in st.session_state:
        st.session_state['app_started'] = False

    if not st.session_state['app_started']:
        # Landing Page
        st.markdown("""
            <div style="text-align: center; padding: 50px; color: white;">
                <div style="font-size: 80px; margin-bottom:20px;">🎓</div>
                <h1 style="color:white !important; font-size: 3em;">AI Course Advisor</h1>
                <p style="font-size: 1.2em;">Konsultan Akademik Pribadimu, 24/7.</p>
            </div>
        """, unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1,1,1])
        with col2:
            if st.button("Mulai Konsultasi 🚀", use_container_width=True):
                st.session_state['app_started'] = True
                st.rerun()
    else:
        # Sidebar Menu
        with st.sidebar:
            st.title("Menu Aplikasi")
            menu = st.radio("Pilih Mode:", ["🔍 Cari Jurusan (Database)", "🤖 Chat Bebas (AI)"])
            st.markdown("---")
            if st.button("🏠 Kembali ke Depan"):
                st.session_state['app_started'] = False
                st.rerun()

        # Routing
        if menu == "🔍 Cari Jurusan (Database)":
            page_recommendation()
        elif menu == "🤖 Chat Bebas (AI)":
            page_chat_ai()

if __name__ == "__main__":
    main()