import streamlit as st
import os
import re
import requests
import pandas as pd
import json
from bs4 import BeautifulSoup
from typing import List, Optional, Literal
from pydantic import BaseModel, Field
import instructor
from groq import Groq
from dotenv import load_dotenv


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

extractor_client = instructor.from_groq(Groq(api_key=api_key),mode=instructor.Mode.TOOLS)

counselor_client = instructor.from_groq(Groq(api_key=api_key),mode=instructor.Mode.TOOLS)

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


class Benefit(BaseModel):
    name: str = Field(..., description="Numele beneficiului")
    details: Optional[str] = Field(None, description="Detalii scurte, dacă există")

class Requirement(BaseModel):
    category: Literal["must_have", "nice_to_have", "other"] = Field(..., description="Tip cerință")
    text: str = Field(..., description="Cerința, formulată concis")

class RawExtraction(BaseModel):
    role_title: Optional[str] = Field(None, description="Titlul rolului")
    company_name: Optional[str] = Field(None, description="Compania")

    tech_stack: List[str] = Field(default_factory=list, description="Tehnologii detectate")
    salary_range: Optional[SalaryRange] = Field(None, description="Interval salarial dacă există")
    benefits: List[Benefit] = Field(default_factory=list, description="Beneficii")
    requirements: List[Requirement] = Field(default_factory=list, description="Cerințe")
    location: Optional[Location] = Field(None, description="Locație ")

    confidence: int = Field(80, ge=0, le=100, description="Încredere în extracție (0-100)")


class StrategicAdvice(BaseModel):
    match_score: int = Field(..., ge=0, le=100, description="Potrivire cu piața/rolul (0-100)")
    market_positioning: str = Field(..., description="Cum se poziționează rolul pe piață (2-5 fraze)")
    interview_questions: List[str] = Field(..., description="Întrebări strategice pentru interviu (5-10)")
    negotiation_tips: List[str] = Field(..., description="Sfaturi de negociere (3-6)")
    red_flags: List[RedFlag] = Field(default_factory=list, description="Red flags deduse din facts")
    summary: str = Field(..., description="Rezumat scurt în limba română (max 2 fraze)")

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

def extract_facts_with_ai(text: str) -> RawExtraction:
    return extractor_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        response_model=RawExtraction,
        messages=[
            {
                "role": "system",
                "content": (
                    "Ești The Extractor. Extragi DOAR fapte brute din textul jobului. "
                    "Nu interpreta, nu da sfaturi, nu inventa. "
                    "Dacă o informație nu e în text, las-o null sau listă goală. "
                    "tech_stack: doar tehnologii explicit menționate. "
                    "salary_range: doar dacă există valori numerice/interval + monedă. "
                    "benefits/requirements: extrage exact ce e menționat."
                ),
            },
            {
             "role": "user", 
             "content": f"Text job:\n\n{text}"
            
            },
        ],
        temperature=0.0,
    )

def generate_advice_with_ai(facts: RawExtraction) -> StrategicAdvice:

    facts_json = json.dumps(facts.model_dump(), ensure_ascii=False)

    return counselor_client.chat.completions.create(
        model="openai/gpt-oss-20b",
        response_model=StrategicAdvice,
        messages=[
            {
                "role": "system",
                "content": (
                    "Ești The Counselor. Primești FACTE extrase despre un job (JSON) și trebuie să returnezi STRICT un obiect "
                    "conform schemei StrategicAdvice. Nu omite niciun câmp.\n\n"

                    "Schema StrategicAdvice are câmpurile OBLIGATORII:\n"
                    "- match_score (int 0-100)\n"
                    "- market_positioning (string)\n"
                    "- interview_questions (list[string])\n"
                    "- negotiation_tips (list[string])\n"
                    "- summary (string, max 2 fraze în română)\n"
                    "- red_flags (list de obiecte RedFlag)\n\n"

                    "Schema RedFlag (OBLIGATORIU pentru fiecare element din red_flags):\n"
                    "- severity: una din ['low','medium','high']\n"
                    "- category: una din ['toxicity','vague','unrealistic']\n"
                    "- message: text scurt\n\n"

                    "Reguli:\n"
                    "- NU folosi chei precum 'description' sau 'details'. Folosește DOAR 'message'.\n"
                    "- Dacă nu ai red flags, red_flags trebuie să fie lista goală [].\n"
                    "- Dacă lipsesc informații (salariu/beneficii/cerințe/locație), adaugă cel puțin 1 red flag "
                    "cu category='vague' și severity='medium'.\n\n"

                    "Exemplu red_flags corect:\n"
                    "[{\"severity\":\"medium\",\"category\":\"vague\",\"message\":\"Lipsesc detalii despre salariu și beneficii.\"}]"
                ),
            },
            {"role": "user", "content": f"FACTS(JSON):\n{facts_json}"},
        ],
        temperature=0.7,
    )

def analyze_job_pipeline(text: str) -> tuple[RawExtraction, StrategicAdvice]:
    facts = extract_facts_with_ai(text)
    advice = generate_advice_with_ai(facts)
    return facts, advice

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

    use_multi_agent = st.toggle("🔁 Level 2: Multi-agent (Extractor + Counselor)", value=False)



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


                    # ==========================
                    # NIVEL 2 (Multi-agent) 
                    # ==========================
                    if use_multi_agent:
                        st.divider()
                        st.markdown("## Multi-agent results")
                        facts, advice = analyze_job_pipeline(raw_text)
                        report = None

                        # 1) Extractor  (facts only)
                    with st.expander("🧾 Extractor (Fapte reale)", expanded=True):
                          # Titlu + companie
                        st.markdown(f"**Rol:** {facts.role_title or 'N/A'}")
                        st.markdown(f"**Companie:** {facts.company_name or 'N/A'}")

                        # Locație
                        if facts.location:
                            if facts.location.is_remote:
                                st.info("Remote / Hybrid")
                            else:
                                st.success(f"📍 {facts.location.city}, {facts.location.country}")

                        # Salariu
                        if facts.salary_range:
                            st.markdown(
                                f"**Salariu:** {facts.salary_range.min} - {facts.salary_range.max} {facts.salary_range.currency}"
                            )
                        else:
                            st.caption("Salariu: nementionat")

                        # Tech stack
                        st.markdown("**Tech stack (facts):**")
                        if facts.tech_stack:
                            st.write(", ".join([f"`{t}`" for t in facts.tech_stack]))
                        else:
                            st.caption("Nu a fost detectată tehnologie explicită.")

                        # Requirements
                        st.markdown("**Cerințe (facts):**")
                        if facts.requirements:
                            must = [r.text for r in facts.requirements if r.category == "must_have"]
                            nice = [r.text for r in facts.requirements if r.category == "nice_to_have"]
                            other = [r.text for r in facts.requirements if r.category == "other"]

                            if must:
                                st.markdown(" **Must-have:**")
                                for x in must:
                                    st.write(f"  - {x}")
                            if nice:
                                st.markdown(" **Nice-to-have:**")
                                for x in nice:
                                    st.write(f"  - {x}")
                            if other:
                                st.markdown("**Altele:**")
                                for x in other:
                                    st.write(f"  - {x}")
                        else:
                            st.caption("Nu sunt cerințe clare în text (sau jobul nu mai e disponibil).")

                        # Benefits
                        st.markdown("**Beneficii (facts):**")
                        if facts.benefits:
                            for b in facts.benefits:
                                if b.details:
                                    st.write(f"- {b.name} — {b.details}")
                                else:
                                    st.write(f"- {b.name}")
                        else:
                            st.caption("Nu sunt beneficii menționate.")


                        # 2) Counselor  (strategic)
                        st.markdown("### 🧠 Consilier (Sfaturi)")
                        st.metric("Match score", f"{advice.match_score}/100")
                        st.markdown(f"**Rezumat:** {advice.summary}")

                        if advice.red_flags:
                            st.markdown("#### 🚩 Red Flags (Consilier)")
                            for rf in advice.red_flags:
                                st.warning(f"**{rf.severity.upper()} / {rf.category}** — {rf.message}")
                        else:
                            st.success("Nu au fost detectate red flags ( Consilier ).")

                        st.markdown("#### 🎤 Întrebări de interviu")
                        for q in advice.interview_questions:
                            st.write(f"- {q}")

                        st.markdown("#### 💬 Sfaturi de negociere")
                        for tip in advice.negotiation_tips:
                            st.write(f"- {tip}")

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