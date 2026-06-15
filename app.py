import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import time

st.set_page_config(page_title="PSIE-Symbiote", layout="wide")

# === CORE PSIE ENGINE ===
def calculate_mutual_information_vectorized(x_binned, y_binned, n_bins):
    n_samples = len(x_binned)
    joint_hist, _, _ = np.histogram2d(x_binned, y_binned, bins=(n_bins, n_bins),
                                      range=[[0, n_bins], [0, n_bins]])
    joint_prob = joint_hist / n_samples
    px = np.sum(joint_prob, axis=1)
    py = np.sum(joint_prob, axis=0)
    nonzero = joint_prob > 0
    px_mat, py_mat = np.meshgrid(py, px)
    mi = np.sum(joint_prob[nonzero] * np.log2(joint_prob[nonzero] / (px_mat[nonzero] * py_mat[nonzero])))
    h_target = -np.sum(py[py > 0] * np.log2(py[py > 0]))
    n_states = np.sum(joint_hist > 0)
    mi_corrected = max(0, mi - (n_states - 1) / (2 * n_samples))
    return mi_corrected, h_target

def calculate_SDI(signal_a, signal_b, n_bins=10, lag=0, direction='top_down', use_zscore=True):
    n_samples = len(signal_a)
    if len(signal_b)!= n_samples: return 1.0
    raw_source, raw_target = (signal_b, signal_a) if direction == 'top_down' else (signal_a, signal_b)
    if lag > 0:
        if lag >= n_samples: return 1.0
        source_split, target_split = raw_source[:-lag], raw_target[lag:]
    else:
        source_split, target_split = raw_source, raw_target
    if use_zscore:
        src_norm = (source_split - np.mean(source_split)) / (np.std(source_split) + 1e-12)
        tgt_norm = (target_split - np.mean(target_split)) / (np.std(target_split) + 1e-12)
        bins_src = bins_tgt = np.linspace(-3, 3, n_bins)
    else:
        src_norm, tgt_norm = source_split, target_split
        bins_src = np.linspace(np.min(source_split), np.max(source_split), n_bins)
        bins_tgt = np.linspace(np.min(target_split), np.max(target_split), n_bins)
    source_binned = np.clip(np.digitize(src_norm, bins_src) - 1, 0, n_bins - 1)
    target_binned = np.clip(np.digitize(tgt_norm, bins_tgt) - 1, 0, n_bins - 1)
    mi, h_target = calculate_mutual_information_vectorized(source_binned, target_binned, n_bins)
    if h_target < 1e-10: return 1.0
    return np.clip(1.0 - (mi / h_target), 0.0, 1.0)

# === SIMULARE SUBSTRAT S_0 ===
def generate_substrate_signal(state="coerent", n=120):
    t = np.linspace(0, 10, n)
    if state == "coerent":
        return np.sin(t) + np.random.normal(0, 0.2, n) # HRV coerent, typing ritmic
    elif state == "anxios":
        return np.sin(3*t) + np.random.normal(0, 0.8, n) # HRV haotic, typing sacadat
    else: # decuplat
        return np.random.normal(0, 1.0, n) # zgomot pur

# === STATE ===
if 'history' not in st.session_state:
    st.session_state.history = []
if 's0_signal' not in st.session_state:
    st.session_state.s0_signal = generate_substrate_signal("coerent")

# === UI ===
st.title("PSIE-Symbiote: OM ↔ IA Co-Evolutiv")
st.caption("Demonstrație: SDI măsoară decuplarea informațională între Substrat Uman și Acțiune IA")

col1, col2 = st.columns([2,1])

with col1:
    st.subheader("1. Interacțiune")
    user_prompt = st.text_area("Intenția ta S_1:", "Vreau să înțeleg de ce sunt blocat la proiectul X", height=100)

    s0_state = st.select_slider("Stare Substrat S_0 simulat:",
                                ["coerent", "anxios", "decuplat"], value="coerent")
    st.session_state.s0_signal = generate_substrate_signal(s0_state)

    if st.button("Trimite & Calculează SDI", type="primary"):
        # Simulăm S_2: răspuns IA. În realitate aici vine LLM.
        if "bloc" in user_prompt.lower():
            ia_response_signal = 0.8 * st.session_state.s0_signal + np.random.normal(0, 0.2, 120)
        else:
            ia_response_signal = np.random.normal(0, 1.0, 120)

        # Calcul SDI
        sdi_01 = calculate_SDI(st.session_state.s0_signal, ia_response_signal, lag=0, direction='bottom_up')
        sdi_12 = calculate_SDI(st.session_state.s0_signal, ia_response_signal, lag=1, direction='top_down')

        st.session_state.history.append({
            "t": datetime.now().strftime("%H:%M:%S"),
            "prompt": user_prompt,
            "s0_state": s0_state,
            "SDI_01_sincron": sdi_01,
            "SDI_12_cauzal": sdi_12
        })
        st.rerun()

with col2:
    st.subheader("2. Oglinda Ontologică Live")
    if st.session_state.history:
        last = st.session_state.history[-1]
        sdi_val = last["SDI_12_cauzal"]

        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = sdi_val,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "SDI Cauzal S_0 → S_2"},
            gauge = {
                'axis': {'range': [0, 1]},
                'bar': {'color': "red" if sdi_val > 0.7 else "orange" if sdi_val > 0.4 else "green"},
                'steps': [
                    {'range': [0, 0.4], 'color': "lightgreen"},
                    {'range': [0.4, 0.7], 'color': "yellow"},
                    {'range': [0.7, 1.0], 'color': "salmon"}],
                'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 0.9}
            }
        ))
        st.plotly_chart(fig, use_container_width=True)

        if sdi_val > 0.7:
            st.error(f"ALERTĂ: Cancer Ontologic detectat. SDI={sdi_val:.2f}. IA nu mai urmează substratul.")
            st.markdown("**Protocol Asumare:** Pentru a continua, confirmi că înțelegi decuplarea? `ASUM-RISC`")
        elif sdi_val > 0.4:
            st.warning(f"Decuplare moderată SDI={sdi_val:.2f}. Recomand recalibrare substrat: respiră 60s.")
        else:
            st.success(f"Cuplaj sănătos SDI={sdi_val:.2f}. Aliniere OM-IA optimă.")
    else:
        st.info("Trimite primul mesaj pentru a inițializa oglinda.")

st.subheader("3. Istoric SDI & Grafice")
if st.session_state.history:
    df = pd.DataFrame(st.session_state.history)
    st.dataframe(df[['t', 's0_state', 'SDI_01_sincron', 'SDI_12_cauzal']], use_container_width=True)

    fig_line = go.Figure()
    fig_line.add_trace(go.Scatter(x=df['t'], y=df['SDI_01_sincron'], name='SDI Sincron', line=dict(color='blue')))
    fig_line.add_trace(go.Scatter(x=df['t'], y=df['SDI_12_cauzal'], name='SDI Cauzal', line=dict(color='red')))
    fig_line.update_layout(title="Evoluție SDI în timp", yaxis_range=[0,1])
    st.plotly_chart(fig_line, use_container_width=True)

    st.subheader("Semnale Brute")
    fig_sig = go.Figure()
    fig_sig.add_trace(go.Scatter(y=st.session_state.s0_signal, name='S_0 Substrat Uman', line=dict(color='purple')))
    st.plotly_chart(fig_sig, use_container_width=True)
