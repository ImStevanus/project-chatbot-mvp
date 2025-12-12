import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq

# ==========================================
# BAGIAN 1: LOGIKA AI & DATA (BACKEND)
# ==========================================

def ask_groq(query):
    """
    Mengirim pertanyaan ke Groq menggunakan model terbaru (Llama 3.3)
    """
    try:
        if "GROQ_API_KEY" in st.secrets:
            client = Groq(api_key=st.secrets["GROQ_API_KEY"])
            
            system_prompt = """
            Kamu adalah 'AI Course Advisor' untuk Universitas Bunda Mulia (UBM).
            Karaktermu: Ramah, gaul, suportif, dan kekinian. Gunakan emoji.
            
            Tugasmu:
            1. Hubungkan curhatan/hobi user dengan jurusan kuliah yang relevan.
            2. Jawab dengan singkat, padat, dan jelas (maksimal 3-4 kalimat).
            """
            
            chat_completion = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
                ],
                # --- PERBAIKAN DISINI ---
                # Model lama 'llama3-8b-8192' sudah mati. 
                # Kita ganti ke 'llama-3.3-70b-versatile' (Paling baru & Canggih)
                model="llama-3.3-70b-versatile", 
                
                temperature=0.7,
                max_tokens=300,
            )
            
            return chat_completion.choices[0].message.content
        else:
            return "⚠️ Waduh, API Key Groq belum dipasang di secrets.toml."
            
    except Exception as e:
        return f"Maaf, AI lagi pusing (Error: {str(e)})"

# --- MAPPING KEYWORD ---
KEYWORD_MAPPING = {
    "menggambar": "desain visual art seni fotografi kreatif sketsa ilustrasi grafis",
    "jualan": "marketing bisnis manajemen pemasaran retail sales perdagangan kewirausahaan entrepreneur",
    "ngoding": "teknologi informasi sistem komputer data algoritma programming python web software aplikasi digital",
    "hitung": "akuntansi statistika matematika ekonomi keuangan pajak finance analisis",
    "jalan-jalan": "pariwisata hospitality hotel tour travel guide tourism wisata perhotelan",
    "masak": "food beverage tata boga kitchen pastry kuliner makanan minuman chef",
    "game": "game development interactive design programming unity multimedia",
    "tidur": "santai istirahat kesehatan mental psikologi",
    "duit": "investasi keuangan bisnis entrepreneur kaya",
}

PROGRAM_DESCRIPTIONS = {
    "Informatika": "Mempelajari pengembangan software, teknologi jaringan, dan komputasi cerdas.",
    "Sistem Informasi": "Menggabungkan ilmu komputer dengan manajemen bisnis.",
    "Manajemen": "Fokus pada pengelolaan bisnis, strategi pemasaran, dan kepemimpinan.",
    "Akuntansi": "Ahli dalam pencatatan, analisis, dan pelaporan keuangan.",
    "Ilmu Komunikasi": "Strategi penyampaian pesan efektif melalui media digital.",
    "Hospitality dan Pariwisata": "Menyiapkan profesional perhotelan dan kuliner.",
    "Desain Komunikasi Visual": "Solusi komunikasi visual yang kreatif dan inovatif.",
    "Bahasa Inggris": "Komunikasi profesional global melalui bahasa dan budaya.",
    "Bisnis Digital": "Teknologi digital dalam strategi bisnis modern.",
    "Data Science": "Mengolah Big Data menjadi wawasan untuk prediksi.",
    "Psikologi": "Mempelajari perilaku manusia dan proses mental."
}

@st.cache_data
def load_data():
    try:
        df = pd.read_csv('List Mata Kuliah UBM.xlsx - Sheet1.csv')
        df = df.dropna(subset=['Course']) 
        df['combined_features'] = df['Course'].astype(str) + ' ' + df['Program'].astype(str)
        return df
    except FileNotFoundError:
        return pd.DataFrame()

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

def process_negation(user_input):
    negation_patterns = [r'\b(tidak\s+suka|gak\s+suka|benci|anti)\s+(\w+)']
    cleaned_text = user_input.lower()
    words_to_remove = []
    
    for pattern in negation_patterns:
        matches = re.finditer(pattern, cleaned_text)
        for match in matches:
            if len(match.groups()) >= 2:
                negated_word = match.group(2)
                words_to_remove.append(negated_word)
                cleaned_text = cleaned_text.replace(match.group(0), '')
    
    if words_to_remove:
        st.warning(f"⚠️ Sistem mencatat kamu tidak suka: {', '.join(words_to_remove)}. Mata kuliah terkait akan disaring.")
    
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

def recommend_career_paths(courses_list):
    if not courses_list: return []
    careers = set()
    programs = [c['Program'] for c in courses_list]
    
    mapping = {
        'Informatika': ['Software Engineer', 'App Developer'],
        'Data Science': ['Data Scientist', 'AI Engineer'],
        'Bisnis': ['Entrepreneur', 'Manager'],
        'Desain': ['Art Director', 'UI/UX Designer'],
        'Akuntansi': ['Auditor', 'Financial Analyst'],
        'Hospitality': ['Hotel Manager', 'Chef']
    }
    
    for prog in programs:
        for key, vals in mapping.items():
            if key in prog: careers.update(vals)
            
    return list(careers)[:5]

# ==========================================
# BAGIAN 2: UI/UX & TAMPILAN
# ==========================================

def local_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Segoe+UI&display=swap');
    html, body, [class*="css"] { font-family: 'Segoe UI', sans-serif; }
    
    .stApp { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); background-attachment: fixed; }
    
    h1, h2, h3, p, label, .stMarkdown { color: white !important; }
    .result-card h3, .result-card p, .result-card div { color: #31333F !important; }
    section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] p, section[data-testid="stSidebar"] label { color: #2d3748 !important; }
    
    .landing-container { text-align: center; padding: 40px; color: white; }
    .bot-icon { font-size: 80px; background: rgba(255,255,255,0.2); width:140px; height:140px; border-radius:50%; margin: 0 auto 20px; display:flex; align-items:center; justify-content:center; }
    .main-title { font-size: 48px; font-weight: bold; margin-bottom: 10px; }
    
    div.stButton > button { background: white !important; color: #667eea !important; border-radius: 50px !important; font-weight: bold; padding: 10px 30px; }
    
    .ai-chat { background: #2D3748; padding: 15px; border-radius: 15px; border-left: 5px solid #4CAF50; margin-top: 20px; color: white !important; }
    </style>
    """, unsafe_allow_html=True)

def render_landing_page():
    st.markdown("""
        <div class="landing-container">
            <div class="bot-icon">🎓</div>
            <div class="main-title">AI Course Advisor</div>
            <div class="subtitle">Curhat minatmu, temukan masa depanmu.</div>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1,1,1])
    with col2:
        if st.button("Mulai Konsultasi 🚀", use_container_width=True):
            st.session_state['app_started'] = True
            st.rerun()

def main_app():
    df = load_data()
    
    with st.sidebar:
        st.header("⚙️ Filter & Menu")
        if st.button("🏠 Kembali ke Home"):
            st.session_state['app_started'] = False
            st.rerun()
        
        st.markdown("---")
        prog_list = ["Semua Jurusan"] + sorted(df['Program'].unique().tolist()) if not df.empty else []
        sel_prog = st.selectbox("Jurusan", prog_list)
        
        diff_range = st.slider("Tingkat Kesulitan (Bintang)", 1, 5, (1, 5))

    st.title("🎓 Tanya AI Course Advisor")
    st.markdown("Ceritakan minatmu (misal: *'Suka gambar tapi gak suka itungan'*), atau tanya hal bebas!")
    
    user_input = st.text_area("Ketik disini...", height=80, placeholder="Contoh: Saya mau jadi pebisnis yang jago IT...")
    
    if st.button("Kirim 🚀"):
        if not user_input:
            st.warning("Isi dulu ya minat kamu!")
        else:
            st.markdown("---")
            
            clean_text, ignored = process_negation(user_input)
            
            # Filter Data
            df_filter = df.copy()
            if sel_prog != "Semua Jurusan": 
                df_filter = df_filter[df_filter['Program'] == sel_prog]
            
            recs = get_recommendations(clean_text, df_filter, ignored)
            
            # --- SKENARIO A: Ada Matkul yang Cocok ---
            if not recs.empty:
                recs['Difficulty'] = recs['Course'].apply(get_course_difficulty)
                recs = recs[(recs['Difficulty'] >= diff_range[0]) & (recs['Difficulty'] <= diff_range[1])]
                
                if not recs.empty:
                    st.success(f"✅ Ditemukan {len(recs)} Mata Kuliah yang pas buat kamu!")
                    
                    for idx, row in recs.iterrows():
                        stars = '★' * int(row['Difficulty']) + '☆' * (5 - int(row['Difficulty']))
                        advice = get_course_advice(row['Course'])
                        
                        st.markdown(f"""
                        <div class="result-card" style="background: #f0f2f6; padding: 20px; border-radius: 15px; margin-bottom: 10px; border-left: 5px solid #667eea;">
                            <h3 style="margin:0;">{row['Course']}</h3>
                            <p>🎓 {row['Program']} | ⭐ Kesesuaian: {row['Similarity Score']}%</p>
                            <p>Tingkat Kesulitan: {stars}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        with st.expander(f"💡 Tips Sukses: {row['Course']}"):
                            st.info(advice)
                    
                    careers = recommend_career_paths(recs.to_dict('records'))
                    if careers:
                        st.markdown("### 💼 Prospek Karir Masa Depan:")
                        st.write(", ".join(careers))
                    
                    # --- AI COMMENT (GROQ) ---
                    with st.spinner("AI sedang menganalisis pilihanmu..."):
                        ai_comment = ask_groq(f"User minatnya: '{user_input}'. Matkul yang cocok: {recs['Course'].iloc[0]}. Berikan semangat singkat!")
                        st.markdown(f'<div class="ai-chat">🤖 <b>Komentar AI:</b><br>{ai_comment}</div>', unsafe_allow_html=True)
                else:
                    st.warning("Ada matkul yang cocok, tapi tingkat kesulitannya di luar filter kamu.")
            
            # --- SKENARIO B: Tidak Ada Matkul / Tanya Bebas ---
            else:
                st.warning("Database kampus belum menemukan matkul spesifik...")
                with st.spinner("Tanya ke AI Advisor..."):
                    ai_response = ask_groq(user_input)
                    st.markdown(f'<div class="ai-chat">🤖 <b>Jawaban AI:</b><br>{ai_response}</div>', unsafe_allow_html=True)

# --- ENTRY POINT ---
def main():
    st.set_page_config(page_title="AI Course Advisor", page_icon="🎓", layout="wide")
    local_css()
    
    if 'app_started' not in st.session_state:
        st.session_state['app_started'] = False
        
    if not st.session_state['app_started']:
        render_landing_page()
    else:
        main_app()

if __name__ == "__main__":
    main()