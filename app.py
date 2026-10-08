"""Streamlit front end for the music genre classifier.

Analysis runs once per file and its result is kept in st.session_state, so changing any widget
re-renders the results instead of wiping them.
"""
from __future__ import annotations

import hashlib
import os
import tempfile

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from genre_classifier import config
from genre_classifier.audio import AudioError, load_audio
from genre_classifier.infer import GenrePredictor
from genre_classifier.viz import mel_spectrogram_db, waveform_envelope

st.set_page_config(page_title="AI Music Genre Classifier", page_icon="🎵", layout="wide")

MIME = {"mp3": "audio/mpeg", "wav": "audio/wav", "flac": "audio/flac", "ogg": "audio/ogg", "m4a": "audio/mp4"}
ACCENT, PINK = "#8b5cf6", "#ec4899"

st.markdown(
    """
<style>
  .block-container { max-width: 1150px; padding-top: 2rem; }
  .hero { padding: 1.4rem 1.6rem; border-radius: 16px; margin-bottom: 1.2rem;
          background: linear-gradient(120deg, #4c1d95, #be185d); }
  .hero h1 { margin: 0; font-size: 2rem; color: #fff; }
  .hero p  { margin: .3rem 0 0; color: #f3e8ff; opacity: .9; }
  .result-card { padding: 1.4rem; border-radius: 16px; background: #161b26; border: 1px solid #2a3142; }
  .result-card .label { font-size: .8rem; letter-spacing: .08em; text-transform: uppercase; color: #9aa3b5; }
  .result-card .genre { font-size: 2.4rem; font-weight: 700; margin: .1rem 0; }
  .result-card .meta  { color: #c3c9d6; }
  .uncertain .genre { color: #fbbf24; }
  .confident .genre { color: #a78bfa; }
  .note { color: #9aa3b5; font-size: .85rem; }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading model…")
def load_predictor() -> GenrePredictor | None:
    if not config.MODEL_PATH.exists():
        return None
    return GenrePredictor()


def human_size(n_bytes: int) -> str:
    if n_bytes >= 1e6:
        return f"{n_bytes / 1e6:.1f} MB"
    return f"{max(n_bytes, 1) / 1e3:.0f} KB"


def pretty(genre: str) -> str:
    return {"hiphop": "Hip-Hop", "soul_rnb": "Soul / R&B"}.get(genre, genre.capitalize())


def plot_layout(fig: go.Figure, height: int = 380, **kw) -> go.Figure:
    fig.update_layout(template="plotly_dark", height=height, paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=10, r=10, t=30, b=10), **kw)
    return fig


def probability_chart(pred) -> go.Figure:
    order = np.argsort(pred.probabilities)
    names = [pretty(pred.classes[i]) for i in order]
    vals = pred.probabilities[order] * 100
    fig = go.Figure(go.Bar(x=vals, y=names, orientation="h", marker_color=ACCENT,
                           text=[f"{v:.1f}%" for v in vals], textposition="outside", cliponaxis=False))
    fig.update_xaxes(title="Probability (%)", range=[0, max(100, vals.max() * 1.15)])
    return plot_layout(fig, height=max(360, 28 * len(names)))


def spectrogram_chart(result) -> go.Figure:
    t, f, S = result["spec"]
    fig = go.Figure(go.Heatmap(x=t, y=f, z=S, colorscale="Magma", zmin=-80, zmax=0,
                               colorbar=dict(title="dB")))
    fig.update_xaxes(title="Time (s)")
    fig.update_yaxes(title="Frequency (Hz, mel scale)", type="log")
    return plot_layout(fig)


def waveform_chart(result) -> go.Figure:
    t, lo, hi = result["wave"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=hi, line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=t, y=lo, fill="tonexty", line=dict(width=0), fillcolor="rgba(236,72,153,.6)",
                             showlegend=False, hoverinfo="skip"))
    fig.update_xaxes(title="Time (s)")
    fig.update_yaxes(title="Amplitude", range=[-1, 1])
    return plot_layout(fig, height=300)


def timeline_chart(pred) -> go.Figure:
    """Per-section probabilities: shows whether the verdict is stable across the song."""
    keep = [i for i in range(len(pred.classes)) if pred.window_probabilities[:, i].max() > 0.05]
    fig = go.Figure(go.Heatmap(
        x=[f"{t:.0f}s" for t in pred.window_times], y=[pretty(pred.classes[i]) for i in keep],
        z=pred.window_probabilities[:, keep].T * 100, colorscale="Viridis", zmin=0, zmax=100,
        colorbar=dict(title="%")))
    fig.update_xaxes(title="Start of each 10-second section", type="category")
    return plot_layout(fig, height=max(260, 34 * len(keep) + 120))


def analyse(predictor: GenrePredictor, uploaded) -> dict:
    suffix = "." + uploaded.name.rsplit(".", 1)[-1].lower()
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(uploaded.getbuffer())
        path = tmp.name
    try:
        with st.status("Analysing…", expanded=True) as status:
            status.write("Decoding audio")
            y = load_audio(path)
            status.write(f"Classifying the whole song ({len(y) / config.SAMPLE_RATE:.0f} s)")
            pred = predictor.predict(y, progress=status.write)
            status.write("Rendering charts")
            result = {"pred": pred, "duration": len(y) / config.SAMPLE_RATE,
                      "spec": mel_spectrogram_db(y, config.SAMPLE_RATE),
                      "wave": waveform_envelope(y, config.SAMPLE_RATE)}
            status.update(label="Analysis complete", state="complete", expanded=False)
        return result
    finally:
        os.remove(path)


def render_result(result: dict) -> None:
    pred = result["pred"]
    left, right = st.columns([3, 2], gap="large")
    with left:
        if pred.out_of_distribution:
            st.markdown(
                '<div class="result-card uncertain"><div class="label">Verdict</div>'
                '<div class="genre">Not music I know</div>'
                f'<div class="meta">No genre given: {pred.reason}. '
                'This usually means noise, speech, a test tone or an unusual recording.</div></div>',
                unsafe_allow_html=True)
        elif pred.uncertain:
            a, b = pred.top[0], pred.top[1]
            st.markdown(
                f'<div class="result-card uncertain"><div class="label">Verdict</div>'
                f'<div class="genre">Not sure</div>'
                f'<div class="meta">Closest: <b>{pretty(a[0])}</b> ({a[1]:.0%}) or <b>{pretty(b[0])}</b> ({b[1]:.0%}).<br>'
                f'Why: {pred.reason}.</div></div>', unsafe_allow_html=True)
            st.caption("The model only knows the genres listed in the sidebar. Rather than force a confident "
                       "answer on music that does not fit them well, it says so.")
        else:
            st.markdown(
                f'<div class="result-card confident"><div class="label">Predicted genre</div>'
                f'<div class="genre">{pretty(pred.label)}</div>'
                f'<div class="meta">Confidence {pred.confidence:.0%} · '
                f'{pred.agreement:.0%} of the song\'s sections agree</div></div>', unsafe_allow_html=True)
    with right:
        if pred.out_of_distribution:
            st.markdown("**Why no genre?**")
            st.metric("Music score", f"{pred.music_prob:.0%}",
                      help="How strongly the pretrained encoder recognises this as music (AudioSet 'Music' class).")
            st.caption("Genre scores are hidden because they would be meaningless for audio that is not music.")
        else:
            st.markdown("**Top 3**")
            for g, p in pred.top:
                st.progress(min(max(p, 0.0), 1.0), text=f"{pretty(g)} — {p:.1%}")

    if pred.out_of_distribution:
        tab_spec, tab_wave = st.tabs(["Spectrogram", "Waveform"])
        tab_prob = tab_time = None
    else:
        tab_prob, tab_spec, tab_wave, tab_time = st.tabs(["Probabilities", "Spectrogram", "Waveform", "Over time"])
    if tab_prob is not None:
        with tab_prob:
            st.plotly_chart(probability_chart(pred), use_container_width=True)
    with tab_spec:
        st.plotly_chart(spectrogram_chart(result), use_container_width=True)
        st.caption("Log-mel spectrogram of the full upload (brighter = louder).")
    with tab_wave:
        st.plotly_chart(waveform_chart(result), use_container_width=True)
    if tab_time is not None:
        with tab_time:
            st.plotly_chart(timeline_chart(pred), use_container_width=True)
            st.caption("Each column is one 10-second section. A song that changes style shows up here.")


# ------------------------------------------------------------------------------ page
st.markdown(
    '<div class="hero"><h1>🎵 AI Music Genre Classifier</h1>'
    "<p>Analyses the whole song with a pretrained audio transformer and says when it is unsure.</p></div>",
    unsafe_allow_html=True,
)

predictor = load_predictor()

with st.sidebar:
    st.header("About this model")
    if predictor is None:
        st.error("Model file missing.")
        st.caption("Train it first: see the README, section “Train it yourself”.")
    else:
        m = predictor.metrics
        if m:
            st.metric("Test accuracy", f"{m['accuracy']:.0%}", help=f"Track-level, {m['test_tracks']} held-out tracks")
            st.caption(f"Macro F1 {m['macro_f1']:.2f} · calibration error {m['ece']:.2f}")
        st.markdown("**Genres it knows**")
        st.write(", ".join(pretty(g) for g in predictor.classes))
        st.caption("Anything outside this list (or hard to place) is reported as “Not sure” instead of guessed.")

if predictor is None:
    st.stop()

uploaded = st.file_uploader("Upload a song", type=list(MIME), help="MP3, WAV, FLAC, OGG or M4A, up to 50 MB.")

if uploaded is None:
    st.session_state.pop("result", None)
    c = st.columns(3)
    for col, (icon, title, text) in zip(c, [
        ("🎧", "1 · Upload", "Any song, any length. The whole track is used, not just the intro."),
        ("🧠", "2 · Analyse", "A pretrained audio transformer listens to evenly spaced 10-second sections."),
        ("📊", "3 · Review", "See the verdict, the evidence over time, and a warning when it is unsure."),
    ]):
        col.markdown(f"**{icon} {title}**")
        col.caption(text)
else:
    key = hashlib.sha1(uploaded.getbuffer()).hexdigest()
    ext = uploaded.name.rsplit(".", 1)[-1].lower()
    info, player = st.columns([2, 3], gap="large")
    with info:
        st.markdown(f"**{uploaded.name}**")
        st.caption(human_size(uploaded.size))
    with player:
        st.audio(uploaded, format=MIME.get(ext, "audio/mpeg"))

    if st.session_state.get("result_key") != key:
        st.session_state.pop("result", None)
    if st.button("Analyse genre", type="primary"):
        try:
            st.session_state["result"] = analyse(predictor, uploaded)
            st.session_state["result_key"] = key
        except AudioError as exc:
            st.error(str(exc))
    if "result" in st.session_state and st.session_state.get("result_key") == key:
        render_result(st.session_state["result"])

st.markdown("---")
st.caption("Built with Streamlit, PyTorch and an AudioSet-pretrained Audio Spectrogram Transformer. "
           "Predictions are statistical guesses; see the README for limitations.")
