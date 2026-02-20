import streamlit as st
import os
import re
import requests
import pandas as pd
from bs4 import BeautifulSoup
from typing import List, Optional, Literal
from pydantic import BaseModel, Field
import instructor
from groq import Groq
from dotenv import load_dotenv

from typing import List, Optional, Literal
from pydantic import BaseModel, Field

# ==============================================================================
# 1. SETUP & SECURITATE
# ==============================================================================
st.set_page_config(page_title="GenAI Headhunter", page_icon="🕵️", layout="wide")

# Încărcăm variabilele din fișierul .env
load_dotenv()

# Încercăm să luăm cheia din OS (local) sau din Streamlit Secrets (cloud)
api_key = os.getenv("GROQ_API_KEY")

# Fallback pentru Streamlit Cloud deployment
if not api_key and "GROQ_API_KEY" in st.secrets:
    api_key = st.secrets["GROQ_API_KEY"]

# Validare critică: Dacă nu avem cheie, oprim aplicația aici.
if not api_key:
    st.error("⛔ EROARE CRITICĂ: Lipsește `GROQ_API_KEY`.")
    st.info("Te rog creează un fișier `.env` în folderul proiectului și adaugă: GROQ_API_KEY=cheia_ta_aici")
    st.stop()

# Configurare Client Groq Global (pentru a nu-l reinițializa constant)
client = instructor.from_groq(Groq(api_key=api_key), mode=instructor.Mode.TOOLS)

# Sidebar Informativ (Fără input de date sensibile)
with st.sidebar:
    st.header("🕵️ GenAI Headhunter")
    st.success("✅ API Key încărcat securizat")
    st.markdown("---")
    st.write("Acest tool demonstrează:")
    st.write("• Web Scraping (BS4)")
    st.write("• Secure Env Variables")
    st.write("• Structured Data (Pydantic)")


# ==============================================================================
# 2. DATA MODELS (PYDANTIC SCHEMAS)
# ==============================================================================
class SalaryRange(BaseModel):
    min: int = Field(..., ge=0, description="Salariul minim")
    max: int = Field(..., ge=0, description="Salariul maxim")
    currency: str = Field(..., description="Moneda (ex: RON, USD, EUR, CHF)")

class Location(BaseModel):
    city: str = Field(..., description="Oraș")
    country: str = Field(..., description="Țară")
    is_remote: bool = Field(..., description="True dacă jobul este remote/hibrid")

class RedFlag(BaseModel):
    severity: Literal["low", "medium", "high"] = Field(..., description="Severitatea semnalului")
    category: Literal["toxicity", "vague", "unrealistic"] = Field(..., description="Categoria semnalului")
    message: str = Field(..., description="Descrierea semnalului de alarmă")

class JobAnalysis(BaseModel):
    role_title: str = Field(..., description="Titlul jobului standardizat")
    company_name: str = Field(..., description="Numele companiei")
    seniority: Literal["Intern", "Junior", "Mid", "Senior", "Lead", "Architect"] = Field(..., description="Nivelul de experiență dedus")
    match_score: int = Field(..., ge=0, le=100, description="Scor 0-100: Calitatea descrierii jobului")
    tech_stack: List[str] = Field(..., description="Listă cu tehnologii specifice (ex: Python, AWS, React)")
    
    red_flags: List[RedFlag] = Field(..., description="Lista de semnale de alarma")

    summary: str = Field(..., description="Un rezumat scurt al rolului (max 2 fraze) în limba română")

    salary_range: Optional[SalaryRange] = Field(None, description="Interval salarial dacă este menționat")

    location: Location = Field(..., description="Locația jobului")

# ==============================================================================
# 3. UTILS - SCRAPER (Colectare Date)
# ==============================================================================

def scrape_clean_job_text(url: str, max_chars: int = 3000) -> str:
    """
    Descarcă pagina și returnează un text curat, optimizat pentru contextul LLM.
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            return f"Error: Status code {response.status_code}"
            
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Eliminăm elementele inutile care consumă tokeni
        for junk in soup(["script", "style", "nav", "footer", "header", "aside", "iframe"]):
            junk.decompose()
            
        # Extragem textul și eliminăm spațiile multiple
        text = soup.get_text(separator=' ', strip=True)
        text = re.sub(r'\s+', ' ', text)
        
        return text[:max_chars] 
        
    except Exception as e:
        return f"Scraping Error: {str(e)}"

# ==============================================================================
# 4. AI SERVICE LAYER (Logica LLM)
# ==============================================================================

def analyze_job_with_ai(text: str) -> JobAnalysis:
    """
    Trimite textul curățat către Groq și returnează obiectul structurat.
    """
    return client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        response_model=JobAnalysis,
        messages=[
            {
                "role": "system", 
                "content": (
                    "Ești un Recruiter Expert în IT. Analizează textul jobului cu obiectivitate. "
                    "Identifică tehnologiile și potențialele probleme (red flags). "
                    "Pentru salary_range: daca nu exista salariul, afiseaza null"
                    "Pentru location: alege city și country din context; dacă nu e clar, pune 'Necunoscut'. "
                    "Pentru red_flags: întoarce o listă de obiecte cu severity (low/medium/high), category (toxicity/vague/unrealistic) și message scurt."
                    "Răspunde strict în formatul cerut."
                )
            },
            {
                "role": "user", 
                "content": f"Analizează acest job description:\n\n{text}"
            }
        ],
        temperature=0.1,
    )

# ==============================================================================
# 5. UI - APLICAȚIA STREAMLIT
# ==============================================================================

st.title("🕵️ GenAI Headhunter Assistant")
st.markdown("Transformă orice Job Description într-o analiză structurată folosind AI.")

# Tab-uri
tab1, tab2 = st.tabs(["🚀 Analiză Job", "📊 Market Scan (Batch)"])

# --- TAB 1: ANALIZA UNUI SINGUR LINK ---
with tab1:
    st.subheader("Analizează un Job URL")
    url_input = st.text_input("Introdu URL-ul:", placeholder="https://...")
    
    if st.button("Analizează Job", key="btn_single"):
        if not url_input:
            st.warning("Te rugăm introdu un URL.")
        else:
            with st.spinner("🕷️ Scraping & 🤖 AI Analysis..."):
                raw_text = scrape_clean_job_text(url_input)
            
            if "Error" in raw_text:
                st.error(raw_text)
            else:
                try:
                    data = analyze_job_with_ai(raw_text)
                    
                    # -- DISPLAY --
                    st.divider()
                    col_h1, col_h2 = st.columns([3, 1])
                    with col_h1:
                        st.markdown(f"### {data.role_title}")
                        st.caption(f"Companie: **{data.company_name}** | Nivel: **{data.seniority}**")
                    with col_h2:
                        color = "normal" if data.match_score > 70 else "inverse"
                        st.metric("Quality Score", f"{data.match_score}/100", delta_color=color)

                    # Detalii
                    c1, c2, c3 = st.columns(3)
                    c1.info(f"**Remote:** {'Da' if data.location.is_remote else 'Nu'}")
                    c2.success(f"**Tehnologii:** {len(data.tech_stack)}")
                    c3.error(f"**Red Flags:** {len(data.red_flags)}")


                    if not data.location.is_remote:
                        st.markdown( f"📍 **Locație:** {data.location.city}, {data.location.country}")
                        
                    st.markdown(f"**📝 Rezumat:** {data.summary}")
                    st.markdown("#### 🛠️ Tech Stack")
                    st.write(", ".join([f"`{tech}`" for tech in data.tech_stack]))

                    if data.red_flags:
                        st.markdown("#### 🚩 Avertismente")
                        for flag in data.red_flags:
                           st.warning(f"**{flag.severity.upper()} / {flag.category}** — {flag.message}")

                except Exception as e:
                    st.error(f"Eroare AI: {str(e)}")

# --- TAB 2: BATCH PROCESSING ---
with tab2:
    st.subheader("📊 Compară mai multe joburi")
    urls_text = st.text_area("Paste URL-uri (unul pe linie):", height=150)
    
    if st.button("Scanează Piața", key="btn_batch"):
        urls = [u.strip() for u in urls_text.split('\n') if u.strip()]
        
        if not urls:
            st.warning("Nu ai introdus link-uri.")
        else:
            results = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, link in enumerate(urls):
                status_text.text(f"Analizez {i+1}/{len(urls)}...")
                text = scrape_clean_job_text(link)
                
                if "Error" not in text:
                    try:
                        res = analyze_job_with_ai(text)
                        results.append({
                         "Role": res.role_title,
                         "Company": res.company_name,
                         "Seniority": res.seniority,
                         "Remote": res.location.is_remote,
                         "TechCount": len(res.tech_stack),
                         "RedFlags": len(res.red_flags),
                         "Score": res.match_score
                        })
                    except:
                        pass # Continuăm chiar dacă unul crapă
                
                progress_bar.progress((i + 1) / len(urls))
            
            status_text.text("Gata!")
            
            if results:
                df = pd.DataFrame(results)
                st.dataframe(df)
                
                # Grafic simplu
                st.bar_chart(df['Seniority'].value_counts())