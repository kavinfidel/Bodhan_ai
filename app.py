from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import streamlit as st

from speech_generation import DELIVERY_STYLES, generate, load_tts_model, prepare_text
from word_timestamps import DEFAULT_WHISPER_MODEL, create_timestamp_files


OUTPUT_DIR = Path(__file__).parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

st.set_page_config(page_title="Bodhan Tiny Lessons", page_icon="🌱", layout="wide")
st.markdown(
    """
    <style>
      :root {
        --ink:#17231e;
        --text:#24342d;
        --muted:#4f6258;
        --green:#1f6b4c;
        --green-dark:#18583d;
        --surface:#ffffff;
        --surface-soft:#f5f7f0;
        --line:#cfd9d1;
        --mint:#e8f3eb;
        --gold:#f3c96a;
      }
      .stApp {
        background:#f7f8f2;
        color:var(--text);
        font-family:ui-sans-serif,system-ui,sans-serif;
      }
      h1,h2,h3,
      [data-testid="stMarkdownContainer"] h1,
      [data-testid="stMarkdownContainer"] h2,
      [data-testid="stMarkdownContainer"] h3 {
        font-family:ui-sans-serif,system-ui,sans-serif !important;
        color:var(--ink);
      }
      .block-container { max-width:1120px; padding-top:2.1rem; }
      .hero {
        background:linear-gradient(115deg,#e8f2e9 0%,#f5f0dc 100%);
        border:1px solid #cfded2;
        border-radius:20px;
        padding:30px 34px;
        margin-bottom:22px;
      }
      .eyebrow { color:#315d45; font-size:12px; letter-spacing:.13em; text-transform:uppercase; font-weight:700; }
      .hero h1 { font-size:38px; line-height:1.12; margin:8px 0; }
      .hero p { color:var(--muted); margin:0; font-size:16px; }
      .note {
        background:#fffdf6;
        border:1px solid #e0cf8f;
        border-radius:12px;
        padding:14px 17px;
        color:#4c442d;
        font-size:13px;
      }
      p, label, span, div, textarea, input, button {
        color:var(--text);
      }
      [data-testid="stCaptionContainer"],
      [data-testid="stCaptionContainer"] p,
      [data-testid="stMarkdownContainer"] p {
        color:var(--muted);
      }
      [data-testid="stWidgetLabel"],
      [data-testid="stWidgetLabel"] p,
      [data-testid="stRadio"] label,
      [data-testid="stRadio"] p {
        color:var(--ink) !important;
        font-weight:600;
      }
      .stTextArea textarea,
      div[data-baseweb="select"] > div,
      div[data-baseweb="select"] input,
      div[data-baseweb="slider"],
      [data-testid="stRadio"] label > div:first-child {
        background:var(--surface) !important;
        color:var(--ink) !important;
        border-color:var(--line) !important;
      }
      .stTextArea textarea {
        border:1px solid var(--line);
        border-radius:10px;
      }
      .stTextArea textarea:focus,
      div[data-baseweb="select"] > div:focus-within {
        border-color:var(--green) !important;
        box-shadow:0 0 0 2px rgba(31,107,76,.18) !important;
      }
      div[data-baseweb="popover"],
      ul[role="listbox"] {
        background:var(--surface) !important;
        color:var(--ink) !important;
      }
      li[role="option"],
      div[role="option"] {
        color:var(--ink) !important;
      }
      li[role="option"]:hover,
      div[role="option"]:hover {
        background:var(--mint) !important;
      }
      [data-baseweb="radio"] div[role="radio"] {
        border-color:var(--green) !important;
      }
      [data-baseweb="radio"] div[role="radio"][aria-checked="true"] {
        background:var(--green) !important;
      }
      [data-testid="stSlider"] [role="slider"] {
        background:var(--green) !important;
        border-color:var(--green) !important;
      }
      div[data-testid="stButton"] button[kind="primary"],
      div[data-testid="stDownloadButton"] button {
        background:var(--green);
        border:0;
        border-radius:10px;
        color:#ffffff !important;
        min-height:48px;
        font-weight:700;
      }
      div[data-testid="stButton"] button[kind="primary"]:hover,
      div[data-testid="stDownloadButton"] button:hover {
        background:var(--green-dark);
        color:#ffffff !important;
      }
      div[data-testid="stButton"] button[kind="primary"] p,
      div[data-testid="stDownloadButton"] button p {
        color:#ffffff !important;
      }
      [data-testid="stAlert"] {
        background:var(--surface);
        color:var(--ink);
        border:1px solid var(--line);
      }
      [data-testid="stAudio"] { margin-top:8px; }
      footer { visibility:hidden; }
    </style>
    <div class="hero">
      <div class="eyebrow">Bodhan AI · Multilingual learning demo</div>
      <h1>Little lessons. Lively voices.</h1>
      <p>Turn Tamil, Telugu, Hindi, or English lessons into expressive, child-friendly audio — right on this Mac.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def load_tts():
    """Download the public MLX conversion once and keep its runtime warm."""
    return load_tts_model()


LESSONS = {
    "Tamil": {
        "code": "ta",
        "voices": {"Anitha · Female": "Anitha", "Arun · Male": "Arun"},
        "starter": "வணக்கம் குழந்தைகளே! இன்று நாம் ஒரு புதிய பாடத்தைக் கற்றுக்கொள்வோம்.",
        "samples": {
            "விலங்குகள் · Animals": "வணக்கம் குழந்தைகளே! இன்று நாம் விலங்குகளைப் பற்றிக் கற்றுக்கொள்வோம். பசு நமக்குப் பால் தருகிறது. நாய் வீட்டைக் காக்கிறது. பூனை எலியைப் பிடிக்கிறது. யானைக்கு நீண்ட தும்பிக்கை உள்ளது. விலங்குகளிடம் அன்பாக இருப்போம்.",
            "எண்கள் · Numbers": "ஒன்று, இரண்டு, மூன்று, நான்கு, ஐந்து. ஒரு சூரியன் வானத்தில் ஒளிர்கிறது. இரண்டு கண்களால் நாம் பார்க்கிறோம். மூன்று வண்ணங்களில் நம் தேசியக் கொடி அழகாக இருக்கிறது. எண்ணிப் பழகுவோம், மகிழ்ந்து கற்போம்!",
            "நிறங்கள் · Colours": "சிவப்பு ரோஜா அழகாக மலர்கிறது. பச்சை இலை மரத்தில் ஆடுகிறது. மஞ்சள் சூரியன் ஒளி தருகிறது. நீல வானத்தில் வெள்ளை மேகங்கள் மிதக்கின்றன. உனக்குப் பிடித்த நிறம் எது?",
        },
    },
    "Telugu": {
        "code": "te",
        "voices": {"Sravani · Female": "Sravani", "Vamsi · Male": "Vamsi"},
        "starter": "నమస్కారం పిల్లలూ! ఈ రోజు మనం ఒక కొత్త పాఠం నేర్చుకుందాం.",
        "samples": {
            "జంతువులు · Animals": "నమస్కారం పిల్లలూ! ఈ రోజు మనం జంతువుల గురించి నేర్చుకుందాం. ఆవు మనకు పాలు ఇస్తుంది. కుక్క ఇంటిని కాపాడుతుంది. పిల్లి ఎలుకను పట్టుకుంటుంది. ఏనుగుకు పొడవైన తొండం ఉంటుంది. జంతువులను ప్రేమగా చూసుకుందాం.",
            "సంఖ్యలు · Numbers": "ఒకటి, రెండు, మూడు, నాలుగు, ఐదు. ఆకాశంలో ఒక సూర్యుడు ప్రకాశిస్తాడు. మనకు రెండు కళ్ళు ఉన్నాయి. మన జాతీయ జెండాలో మూడు రంగులు ఉన్నాయి. లెక్కలు నేర్చుకుందాం, ఆనందంగా చదువుకుందాం!",
            "రంగులు · Colours": "ఎర్ర గులాబీ అందంగా వికసిస్తుంది. పచ్చని ఆకు చెట్టుపై ఊగుతుంది. పసుపు సూర్యుడు వెలుగునిస్తాడు. నీలి ఆకాశంలో తెల్లని మేఘాలు తేలుతాయి. నీకు ఇష్టమైన రంగు ఏది?",
        },
    },
    "Hindi": {
        "code": "hi",
        "voices": {"Kavya · Female": "Kavya", "Amit · Male": "Amit"},
        "starter": "नमस्ते बच्चों! आज हम एक नया और मज़ेदार पाठ सीखेंगे।",
        "samples": {
            "जानवर · Animals": "नमस्ते बच्चों! आज हम जानवरों के बारे में सीखेंगे। गाय हमें दूध देती है। कुत्ता घर की रखवाली करता है। बिल्ली चूहे को पकड़ती है। हाथी की लंबी सूँड़ होती है। हमें जानवरों से प्यार करना चाहिए।",
            "गिनती · Numbers": "एक, दो, तीन, चार, पाँच। आकाश में एक सूरज चमकता है। हमारी दो आँखें हैं। हमारी कक्षा में तीन रंगीन गेंदें हैं। आओ, गिनती सीखें और खुशी से खेलें!",
            "रंग · Colours": "लाल गुलाब खिलता है। हरी पत्ती हवा में हिलती है। पीला सूरज रोशनी देता है। नीले आकाश में सफ़ेद बादल तैरते हैं। तुम्हारा पसंदीदा रंग कौन सा है?",
        },
    },
    "English": {
        "code": "en",
        "voices": {
            "Sravani · Expressive female": "Sravani",
            "Vamsi · Warm male": "Vamsi",
        },
        "starter": "Hello, children! Today we are going to learn something new and wonderful.",
        "samples": {
            "Morning rhyme": "Good morning, sun so bright!\nFill our classroom up with light.\nClap your hands and tap your feet.\nLearning together is such a treat!",
            "Numbers rhyme": "One little star shines in the blue.\nAlong comes another, and now there are two!\nThree happy birds sing in a tree.\nFour tiny fish swim in the sea!",
            "Colours": "Red is the rose that blooms in spring. Blue is the sky where little birds sing. Green are the leaves that dance in the breeze. Which happy colour do you see?",
        },
    },
}


def update_language():
    language = st.session_state.lesson_language
    st.session_state.lesson_text = LESSONS[language]["starter"]
    st.session_state.lesson_sample = "Write my own"
    st.session_state.lesson_voice = next(iter(LESSONS[language]["voices"]))


def update_lesson_sample():
    samples = LESSONS[st.session_state.lesson_language]["samples"]
    choice = st.session_state.lesson_sample
    if choice in samples:
        st.session_state.lesson_text = samples[choice]


if "lesson_text" not in st.session_state:
    st.session_state.lesson_text = LESSONS["Tamil"]["starter"]

left, right = st.columns([1.45, 0.85], gap="large")
with left:
    st.subheader("Create a listening lesson")
    language = st.selectbox(
        "Language",
        list(LESSONS),
        key="lesson_language",
        on_change=update_language,
    )
    language_config = LESSONS[language]
    st.selectbox(
        "Quick-start lesson",
        ["Write my own"] + list(language_config["samples"]),
        key="lesson_sample",
        on_change=update_lesson_sample,
    )
    lesson = st.text_area(
        f"{language} lesson text",
        key="lesson_text",
        height=190,
        max_chars=1200,
        help=f"Short passages work best. Write the lesson in {language}, and put each rhyme line on a new line.",
    )
    st.caption(f"{len(lesson)} / 1,200 characters · Short passages are easier for young learners to follow.")

with right:
    st.subheader("Voice & delivery")
    grade = st.selectbox("Learning level", ["LKG – UKG", "Class 1", "Class 2"])
    speaker_label = st.radio(
        f"{language} voice",
        list(language_config["voices"]),
        key="lesson_voice",
        horizontal=False,
    )
    speaker = language_config["voices"][speaker_label]
    delivery = st.selectbox(
        "Delivery",
        list(DELIVERY_STYLES),
        help="Playful rhyme uses the model's trained children's-story expression and adds a breath at each written line.",
    )
    pause_seconds = st.slider(
        "Pause between lines or sentences",
        min_value=0.25,
        max_value=1.5,
        value=0.45,
        step=0.05,
        format="%.2f s",
    )
    st.markdown(
        '<div class="note">For rhymes, put each verse on its own line. Exclamation marks add energy; commas create a short natural breath.</div>',
        unsafe_allow_html=True,
    )

st.write("")
button_col, status_col = st.columns([0.34, 0.66], vertical_alignment="center")
with button_col:
    make_audio = st.button(f"✨  Generate {language} audio", type="primary", use_container_width=True)
with status_col:
    st.caption("First run downloads the model (about 2 GB); later runs use the local cache.")

if make_audio:
    if not lesson.strip():
        st.warning(f"Add a short {language} lesson first.")
    elif not (sys.platform == "darwin" and __import__("platform").machine() == "arm64"):
        st.error("This demo uses MLX and needs Apple silicon (M1/M2/M3/M4) with macOS.")
    else:
        try:
            with st.spinner("Warming up the voice model… first load can take a few minutes."):
                tts = load_tts()
            with st.spinner(f"Making your {language} lesson audio…"):
                started = time.perf_counter()
                wav, details = generate(
                    tts,
                    lesson.strip(),
                    speaker,
                    delivery,
                    pause_seconds,
                )
                elapsed = time.perf_counter() - started
            audio = np.asarray(wav, dtype=np.float32).squeeze()
            if audio.size == 0:
                raise ValueError("The model returned an empty audio clip. Try a shorter passage.")
            filename = f"{language_config['code']}_lesson_{time.strftime('%Y%m%d_%H%M%S')}.wav"

            output_path = OUTPUT_DIR / language_config['code'] / filename
            output_path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(output_path, audio, int(getattr(tts, "sample_rate", 24_000)))
            st.session_state["latest_audio"] = str(output_path)
            st.session_state["latest_details"] = details
            st.session_state["latest_elapsed"] = elapsed
            st.session_state["latest_level"] = grade
            st.session_state["latest_language"] = language
            st.session_state["latest_speaker"] = speaker
            st.session_state["latest_delivery"] = delivery
            st.session_state["latest_text"] = prepare_text(lesson.strip(), delivery)
            st.session_state["latest_language_code"] = language_config["code"]
            st.session_state.pop("latest_timestamps", None)
        except Exception as exc:
            st.error(f"Audio generation failed: {exc}")
            st.info("Check the setup steps in README.md. On first launch, allow the model and SNAC decoder to finish downloading.")

if st.session_state.get("latest_audio"):
    audio_path = Path(st.session_state["latest_audio"])
    if audio_path.exists():
        st.divider()
        st.subheader("Your lesson is ready")
        st.caption(f"{st.session_state.get('latest_language', '')} · {st.session_state.get('latest_level', '')} · {st.session_state.get('latest_speaker', '')} · {st.session_state.get('latest_delivery', 'Natural')} · 24 kHz WAV · generated in {st.session_state.get('latest_elapsed', 0):.1f}s")
        st.audio(audio_path.read_bytes(), format="audio/wav")
        st.download_button("⬇  Download lesson audio", data=audio_path.read_bytes(), file_name=audio_path.name, mime="audio/wav")

        st.subheader("Word timestamps")
        st.caption(
            "Run OpenAI Whisper locally on this Mac. The first run downloads a separate "
            "Whisper model; later runs reuse it. JSON preserves every word's "
            "start/end timing estimate and confidence."
        )
        if st.button("🕒  Find word timestamps", use_container_width=True):
            try:
                with st.spinner("Listening for each word with local Whisper…"):
                    timestamp_bundle = create_timestamp_files(
                        audio_path,
                        language=st.session_state.get("latest_language_code"),
                        prompt=st.session_state.get("latest_text"),
                        model=DEFAULT_WHISPER_MODEL,
                    )
                st.session_state["latest_timestamps"] = timestamp_bundle
            except Exception as exc:
                st.error(f"Timestamp generation failed: {exc}")

        timestamp_bundle = st.session_state.get("latest_timestamps")
        if timestamp_bundle and timestamp_bundle.get("audio") == audio_path.resolve():
            words = timestamp_bundle["words"]
            st.success(f"Found {len(words)} spoken words.")
            st.dataframe(
                [
                    {
                        "#": word.index,
                        "Word": word.word,
                        "Start (s)": word.start,
                        "End (s)": word.end,
                        "Confidence": word.confidence,
                    }
                    for word in words
                ],
                hide_index=True,
                use_container_width=True,
            )
            formats = (
                ("JSON", "json", "application/json"),
            )
            download_columns = st.columns(len(formats))
            for column, (label, kind, mime) in zip(download_columns, formats):
                timestamp_path = timestamp_bundle["paths"][kind]
                with column:
                    st.download_button(
                        f"⬇  {label}",
                        data=timestamp_path.read_bytes(),
                        file_name=timestamp_path.name,
                        mime=mime,
                        use_container_width=True,
                    )
        with st.expander("Generation details"):
            st.write(st.session_state.get("latest_details", ""))

st.markdown(
    "<div style='text-align:center;color:#718078;font-size:12px;margin:34px 0 12px'>Built with Indic-Speak from Bodhan AI / AI4Bharat · Local inference on Apple silicon</div>",
    unsafe_allow_html=True,
)
