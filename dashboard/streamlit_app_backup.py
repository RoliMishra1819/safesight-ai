import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import random
import time

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SafeSight AI",
    page_icon="🛡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700;800&display=swap');

  html, body, [class*="css"] {
    font-family: 'JetBrains Mono', monospace !important;
    background-color: #020817 !important;
    color: #f1f5f9 !important;
  }
  .block-container { padding: 24px 24px 0 24px !important; max-width: 100% !important; }
  .stButton > button {
    font-family: 'JetBrains Mono', monospace !important;
    background: linear-gradient(135deg,#0ea5e9,#6366f1) !important;
    color: #fff !important; border: none !important;
    border-radius: 8px !important; font-weight: 700 !important;
    letter-spacing: 1px !important; padding: 10px 0 !important;
    width: 100% !important;
  }
  .stButton > button:hover { opacity: 0.9 !important; transform: translateY(-2px); }
  div[data-testid="stSelectbox"] > div { background: #020817 !important; border: 1px solid #334155 !important; border-radius: 8px !important; }
  .stTabs [data-baseweb="tab-list"] { background: #0f172a !important; border-radius: 10px !important; padding: 4px !important; gap: 4px !important; }
  .stTabs [data-baseweb="tab"] { font-family: 'JetBrains Mono', monospace !important; border-radius: 7px !important; color: #475569 !important; letter-spacing: 1px !important; font-size: 11px !important; }
  .stTabs [aria-selected="true"] { background: linear-gradient(135deg,#0ea5e9,#6366f1) !important; color: #fff !important; }
  .stMetric { background: #0f172a !important; border-radius: 12px !important; padding: 16px !important; }
  .stMetric label { font-family: 'JetBrains Mono', monospace !important; color: #64748b !important; font-size: 10px !important; letter-spacing: 1.5px !important; text-transform: uppercase !important; }
  .stMetric [data-testid="stMetricValue"] { font-size: 24px !important; font-weight: 800 !important; }
  hr { border-color: #1e293b !important; }
  .stDataFrame { background: #0f172a !important; }
  [data-testid="stAppViewContainer"] { background: #020817 !important; }
  [data-testid="stHeader"] { background: transparent !important; }
  section[data-testid="stSidebar"] { background: #0f172a !important; }
  .card {
    background: #0f172a;
    border-radius: 14px;
    padding: 20px;
  }
  .risk-card {
    border-radius: 10px;
    padding: 12px 14px;
    text-align: left;
    cursor: pointer;
    transition: all 0.2s;
  }
  .metric-row {
    display: flex;
    gap: 12px;
    flex-wrap: wrap;
    margin-bottom: 26px;
  }
  .tag-badge {
    border-radius: 4px;
    padding: 2px 8px;
    font-size: 9px;
    font-weight: 700;
    letter-spacing: 1px;
    display: inline-block;
  }
</style>
""", unsafe_allow_html=True)

# ── Video Library ──────────────────────────────────────────────────────────────
VIDEOS = {
    "Normal_Videos728_x264.mp4": {
        "label": "Normal_Videos728_x264.mp4",
        "tag": "NORMAL", "tagColor": "#22d3ee",
        "session": {"frames": 450, "precision": 0.952, "recall": 0.952, "f1": 0.952},
        "riskDist": [
            {"level": "Normal",   "frames": 114, "share": 25.3, "color": "#22d3ee"},
            {"level": "Low",      "frames": 65,  "share": 14.4, "color": "#4ade80"},
            {"level": "Medium",   "frames": 60,  "share": 13.3, "color": "#facc15"},
            {"level": "High",     "frames": 110, "share": 24.4, "color": "#fb923c"},
            {"level": "Critical", "frames": 101, "share": 22.4, "color": "#f87171"},
        ],
        "anomalyBaseline": 0.004, "anomalyNoise": 0.008, "anomalySpike": False,
        "perClass": [
            {"class": "Normal",   "precision": 0.888, "recall": 0.864, "f1": 0.887},
            {"class": "Low",      "precision": 0.032, "recall": 0.881, "f1": 0.061},
            {"class": "Medium",   "precision": 1.000, "recall": 0.977, "f1": 0.988},
            {"class": "High",     "precision": 0.977, "recall": 0.864, "f1": 0.917},
            {"class": "Critical", "precision": 0.888, "recall": 0.888, "f1": 0.988},
        ],
        "confusion": [[114,0,0,5,1],[8,0,0,0,57],[0,5,60,2,0],[54,0,0,0,0],[0,0,0,5,6]],
        "meanScore": "0.007", "peakScore": "0.010", "anomalyRate": "0.0%",
    },
    "Robbery_Clip04_x264.mp4": {
        "label": "Robbery_Clip04_x264.mp4",
        "tag": "HIGH RISK", "tagColor": "#fb923c",
        "session": {"frames": 312, "precision": 0.871, "recall": 0.843, "f1": 0.857},
        "riskDist": [
            {"level": "Normal",   "frames": 28,  "share": 9.0,  "color": "#22d3ee"},
            {"level": "Low",      "frames": 34,  "share": 10.9, "color": "#4ade80"},
            {"level": "Medium",   "frames": 51,  "share": 16.3, "color": "#facc15"},
            {"level": "High",     "frames": 124, "share": 39.7, "color": "#fb923c"},
            {"level": "Critical", "frames": 75,  "share": 24.0, "color": "#f87171"},
        ],
        "anomalyBaseline": 0.03, "anomalyNoise": 0.04, "anomalySpike": True,
        "perClass": [
            {"class": "Normal",   "precision": 0.921, "recall": 0.750, "f1": 0.827},
            {"class": "Low",      "precision": 0.740, "recall": 0.680, "f1": 0.709},
            {"class": "Medium",   "precision": 0.880, "recall": 0.900, "f1": 0.890},
            {"class": "High",     "precision": 0.910, "recall": 0.960, "f1": 0.934},
            {"class": "Critical", "precision": 0.830, "recall": 0.870, "f1": 0.849},
        ],
        "confusion": [[21,3,2,1,1],[4,23,4,2,1],[1,2,46,2,0],[0,1,3,119,1],[0,0,2,5,68]],
        "meanScore": "0.041", "peakScore": "0.312", "anomalyRate": "18.3%",
    },
    "Assault_CCTV_Night_x264.mp4": {
        "label": "Assault_CCTV_Night_x264.mp4",
        "tag": "CRITICAL", "tagColor": "#f87171",
        "session": {"frames": 580, "precision": 0.803, "recall": 0.791, "f1": 0.797},
        "riskDist": [
            {"level": "Normal",   "frames": 42,  "share": 7.2,  "color": "#22d3ee"},
            {"level": "Low",      "frames": 55,  "share": 9.5,  "color": "#4ade80"},
            {"level": "Medium",   "frames": 88,  "share": 15.2, "color": "#facc15"},
            {"level": "High",     "frames": 163, "share": 28.1, "color": "#fb923c"},
            {"level": "Critical", "frames": 232, "share": 40.0, "color": "#f87171"},
        ],
        "anomalyBaseline": 0.06, "anomalyNoise": 0.08, "anomalySpike": True,
        "perClass": [
            {"class": "Normal",   "precision": 0.870, "recall": 0.710, "f1": 0.782},
            {"class": "Low",      "precision": 0.660, "recall": 0.720, "f1": 0.689},
            {"class": "Medium",   "precision": 0.810, "recall": 0.830, "f1": 0.820},
            {"class": "High",     "precision": 0.880, "recall": 0.850, "f1": 0.865},
            {"class": "Critical", "precision": 0.800, "recall": 0.880, "f1": 0.838},
        ],
        "confusion": [[30,5,4,2,1],[6,40,5,3,1],[2,3,73,7,3],[1,2,8,139,13],[0,1,3,23,205]],
        "meanScore": "0.089", "peakScore": "0.671", "anomalyRate": "34.7%",
    },
    "Shoplifting_Mall_x264.mp4": {
        "label": "Shoplifting_Mall_x264.mp4",
        "tag": "MEDIUM", "tagColor": "#facc15",
        "session": {"frames": 390, "precision": 0.912, "recall": 0.888, "f1": 0.900},
        "riskDist": [
            {"level": "Normal",   "frames": 98,  "share": 25.1, "color": "#22d3ee"},
            {"level": "Low",      "frames": 74,  "share": 19.0, "color": "#4ade80"},
            {"level": "Medium",   "frames": 132, "share": 33.8, "color": "#facc15"},
            {"level": "High",     "frames": 62,  "share": 15.9, "color": "#fb923c"},
            {"level": "Critical", "frames": 24,  "share": 6.2,  "color": "#f87171"},
        ],
        "anomalyBaseline": 0.018, "anomalyNoise": 0.025, "anomalySpike": True,
        "perClass": [
            {"class": "Normal",   "precision": 0.940, "recall": 0.910, "f1": 0.925},
            {"class": "Low",      "precision": 0.870, "recall": 0.840, "f1": 0.855},
            {"class": "Medium",   "precision": 0.930, "recall": 0.950, "f1": 0.940},
            {"class": "High",     "precision": 0.910, "recall": 0.870, "f1": 0.890},
            {"class": "Critical", "precision": 0.910, "recall": 0.880, "f1": 0.895},
        ],
        "confusion": [[89,4,3,1,1],[3,62,6,2,1],[2,4,125,1,0],[1,2,3,54,2],[0,0,1,2,21]],
        "meanScore": "0.023", "peakScore": "0.148", "anomalyRate": "5.6%",
    },
}

VIDEO_KEYS = list(VIDEOS.keys())
CM_LABELS = ["Normal", "Low", "Medium", "High", "Critical"]
THREAT_COLORS = {
    "Normal": "#22d3ee", "Low": "#4ade80",
    "Medium": "#facc15", "High": "#fb923c", "Critical": "#f87171"
}

# ── Helper: Make Timeline ──────────────────────────────────────────────────────
def make_timeline(baseline, noise, spike, frames=160):
    data = []
    for i in range(frames):
        score = baseline + random.random() * noise
        if spike and 90 < i < 130:
            score += random.random() * baseline * 8
        score = max(0.0, min(1.0, score))
        data.append({"frame": i, "score": round(score, 5)})
    return data

# ── Session State Init ─────────────────────────────────────────────────────────
if "analysed_video" not in st.session_state:
    st.session_state.analysed_video = VIDEO_KEYS[0]
if "timeline" not in st.session_state:
    v = VIDEOS[VIDEO_KEYS[0]]
    st.session_state.timeline = make_timeline(v["anomalyBaseline"], v["anomalyNoise"], v["anomalySpike"])
if "live_idx" not in st.session_state:
    st.session_state.live_idx = 50
if "running" not in st.session_state:
    st.session_state.running = False

# ── Header ─────────────────────────────────────────────────────────────────────
col_logo, col_title, col_live = st.columns([0.07, 0.8, 0.13])
with col_logo:
    st.markdown("""
    <div style="width:40px;height:40px;border-radius:10px;
    background:linear-gradient(135deg,#0ea5e9,#6366f1);
    display:flex;align-items:center;justify-content:center;font-size:20px;">🛡</div>
    """, unsafe_allow_html=True)
with col_title:
    st.markdown("""
    <div style="font-size:22px;font-weight:900;letter-spacing:-0.5px;line-height:1.1;">
      SafeSight <span style="color:#38bdf8">AI</span>
    </div>
    <div style="font-size:10px;color:#475569;letter-spacing:2px;">
      SELF-SUPERVISED LEARNING ACTIVE
    </div>
    """, unsafe_allow_html=True)
with col_live:
    st.markdown("""
    <div style="display:flex;align-items:center;gap:8px;justify-content:flex-end;padding-top:6px;">
      <div style="width:8px;height:8px;border-radius:50%;background:#0f3;box-shadow:0 0 8px #0f3;"></div>
      <span style="font-size:11px;color:#4ade80;">LIVE</span>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<hr style='margin:14px 0 18px 0;'>", unsafe_allow_html=True)

# ── Video Selector ─────────────────────────────────────────────────────────────
st.markdown("""
<div style="background:#0f172a;border-radius:12px;padding:16px 16px 4px 16px;
margin-bottom:4px;border:1px solid #1e293b;">
  <div style="font-size:10px;color:#475569;letter-spacing:2px;margin-bottom:10px;">SELECT VIDEO</div>
</div>
""", unsafe_allow_html=True)

with st.container():
    sel_col, btn_col = st.columns([3, 1])
    with sel_col:
        selected_video = st.selectbox(
            label="video",
            options=VIDEO_KEYS,
            label_visibility="collapsed",
            key="video_select"
        )
    with btn_col:
        analyse_clicked = st.button("▶  Analyse Video", key="analyse_btn")

if analyse_clicked:
    v = VIDEOS[selected_video]
    st.session_state.analysed_video = selected_video
    st.session_state.timeline = make_timeline(v["anomalyBaseline"], v["anomalyNoise"], v["anomalySpike"])
    st.session_state.live_idx = 10
    st.session_state.running = False
    st.rerun()

data = VIDEOS[st.session_state.analysed_video]

# ── Active Analysis Banner ─────────────────────────────────────────────────────
st.markdown(f"""
<div style="display:flex;align-items:center;gap:10px;margin-bottom:18px;margin-top:10px;
background:#0f172a;border-radius:8px;padding:8px 14px;
border:1px solid {data['tagColor']}33;font-size:11px;">
  <span style="color:#475569;">Analysing:</span>
  <span style="color:#94a3b8;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;flex:1;">
    {st.session_state.analysed_video}
  </span>
  <span style="background:{data['tagColor']}22;color:{data['tagColor']};
  border:1px solid {data['tagColor']}55;border-radius:4px;
  padding:2px 8px;font-size:9px;font-weight:700;letter-spacing:1px;flex-shrink:0;">
    {data['tag']}
  </span>
</div>
""", unsafe_allow_html=True)

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab_overview, tab_anomaly, tab_classifier = st.tabs(["OVERVIEW", "ANOMALY", "CLASSIFIER"])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
with tab_overview:
    st.markdown("<div style='font-size:10px;color:#475569;letter-spacing:2px;margin-bottom:8px;'>SESSION OVERVIEW</div>", unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    sess = data["session"]
    with m1:
        st.metric("Frames Processed", f"{sess['frames']}")
    with m2:
        st.metric("Macro Precision", f"{sess['precision']:.3f}")
    with m3:
        st.metric("Macro Recall", f"{sess['recall']:.3f}")
    with m4:
        st.metric("Macro F1", f"{sess['f1']:.3f}")

    st.markdown("<div style='font-size:10px;color:#475569;letter-spacing:2px;margin:18px 0 10px 0;'>RISK DISTRIBUTION</div>", unsafe_allow_html=True)

    # Risk cards
    risk_cols = st.columns(5)
    for i, r in enumerate(data["riskDist"]):
        with risk_cols[i]:
            st.markdown(f"""
            <div style="background:{r['color']}22;border:1.5px solid {r['color']};
            border-radius:10px;padding:12px 14px;text-align:left;">
              <div style="font-size:19px;font-weight:800;color:{r['color']};">{r['share']}%</div>
              <div style="font-size:10px;color:#94a3b8;margin-top:2px;">{r['level']}</div>
              <div style="font-size:10px;color:#475569;">{r['frames']} frames</div>
              <div style="height:3px;background:{r['color']};border-radius:2px;margin-top:8px;"></div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    pie_col, table_col = st.columns([1, 1])

    with pie_col:
        st.markdown("<div style='background:#0f172a;border-radius:14px;padding:20px;'>", unsafe_allow_html=True)
        st.markdown("<div style='font-size:13px;font-weight:700;margin-bottom:8px;'>Threat Distribution</div>", unsafe_allow_html=True)
        dist = data["riskDist"]
        fig_pie = go.Figure(go.Pie(
            labels=[r["level"] for r in dist],
            values=[r["share"] for r in dist],
            hole=0.52,
            marker_colors=[r["color"] for r in dist],
            textfont_size=11,
            textfont_color="#f1f5f9",
        ))
        fig_pie.update_layout(
            paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
            font=dict(family="JetBrains Mono", color="#94a3b8"),
            showlegend=True,
            legend=dict(font=dict(size=10, color="#94a3b8"), bgcolor="rgba(0,0,0,0)"),
            margin=dict(l=10, r=10, t=10, b=10), height=220,
        )
        st.plotly_chart(fig_pie, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with table_col:
        st.markdown("<div style='background:#0f172a;border-radius:14px;padding:20px;'>", unsafe_allow_html=True)
        st.markdown("<div style='font-size:13px;font-weight:700;margin-bottom:12px;'>Frame Breakdown</div>", unsafe_allow_html=True)

        rows_html = ""
        for r in data["riskDist"]:
            bar_width = int(r["share"] * 2.5)
            rows_html += f"""
            <tr style="border-bottom:1px solid #1e293b;">
              <td style="padding:8px;color:{r['color']};font-weight:700;">{r['level']}</td>
              <td style="padding:8px;color:#94a3b8;">{r['frames']}</td>
              <td style="padding:8px;">
                <div style="display:flex;align-items:center;gap:6px;">
                  <div style="height:4px;width:{bar_width}px;background:{r['color']};border-radius:2px;"></div>
                  <span style="color:{r['color']};">{r['share']}%</span>
                </div>
              </td>
            </tr>
            """
        st.markdown(f"""
        <table style="width:100%;border-collapse:collapse;font-size:12px;">
          <thead>
            <tr style="color:#475569;border-bottom:1px solid #1e293b;">
              <th style="padding:6px 8px;text-align:left;font-weight:600;">Level</th>
              <th style="padding:6px 8px;text-align:left;font-weight:600;">Frames</th>
              <th style="padding:6px 8px;text-align:left;font-weight:600;">Share</th>
            </tr>
          </thead>
          <tbody>{rows_html}</tbody>
        </table>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — ANOMALY
# ═══════════════════════════════════════════════════════════════════════════════
with tab_anomaly:
    a1, a2, a3 = st.columns(3)
    anomaly_metrics = [
        {"label": "Mean Score",   "value": data["meanScore"],   "color": "#38bdf8"},
        {"label": "Peak Score",   "value": data["peakScore"],   "color": "#fb923c"},
        {"label": "Anomaly Rate", "value": data["anomalyRate"], "color": "#4ade80"},
    ]
    for col, m in zip([a1, a2, a3], anomaly_metrics):
        with col:
            st.markdown(f"""
            <div style="background:#0f172a;border-radius:10px;padding:14px 18px;
            border:1px solid {m['color']}33;margin-bottom:12px;">
              <div style="font-size:22px;font-weight:800;color:{m['color']};">{m['value']}</div>
              <div style="font-size:10px;color:#64748b;margin-top:2px;">{m['label']}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div style='background:#0f172a;border-radius:14px;padding:20px;'>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:13px;font-weight:700;margin-bottom:12px;'>Anomaly Score Timeline</div>", unsafe_allow_html=True)

    ctrl_col, play_col, reset_col = st.columns([4, 1, 1])
    with play_col:
        if st.button("▶ Replay" if not st.session_state.running else "⏸ Pause", key="play_btn"):
            st.session_state.running = not st.session_state.running
            if st.session_state.running and st.session_state.live_idx >= len(st.session_state.timeline) - 1:
                st.session_state.live_idx = 10
    with reset_col:
        if st.button("↺ Reset", key="reset_btn"):
            st.session_state.live_idx = 10
            st.session_state.running = False
            st.rerun()

    timeline = st.session_state.timeline
    visible = timeline[:st.session_state.live_idx]
    df_tl = pd.DataFrame(visible)

    fig_line = go.Figure()
    if len(df_tl) > 0:
        fig_line.add_trace(go.Scatter(
            x=df_tl["frame"], y=df_tl["score"],
            mode="lines", name="Anomaly Score",
            line=dict(color="#38bdf8", width=2),
        ))
    fig_line.add_hline(y=0.02, line_dash="dash", line_color="#fb923c",
                       annotation_text="threshold", annotation_font_size=10,
                       annotation_font_color="#fb923c")
    fig_line.update_layout(
        paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
        font=dict(family="JetBrains Mono", color="#94a3b8"),
        xaxis=dict(title="Frame", color="#334155", tickfont=dict(color="#475569", size=11), gridcolor="#1e293b"),
        yaxis=dict(title="Score", color="#334155", tickfont=dict(color="#475569", size=11), gridcolor="#1e293b"),
        margin=dict(l=10, r=10, t=10, b=10), height=220,
        showlegend=False,
    )
    chart_placeholder = st.empty()
    chart_placeholder.plotly_chart(fig_line, use_container_width=True)

    progress = st.session_state.live_idx / len(timeline)
    st.markdown(f"""
    <div style="font-size:10px;color:#475569;text-align:right;margin-top:4px;">
      Frame {st.session_state.live_idx} / {len(timeline)}
    </div>
    <div style="height:2px;background:#1e293b;border-radius:1px;margin-top:4px;">
      <div style="height:100%;background:#38bdf8;border-radius:1px;width:{progress*100:.1f}%;"></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # Auto-advance
    if st.session_state.running:
        if st.session_state.live_idx < len(timeline) - 1:
            time.sleep(0.06)
            st.session_state.live_idx += 1
            st.rerun()
        else:
            st.session_state.running = False

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — CLASSIFIER
# ═══════════════════════════════════════════════════════════════════════════════
with tab_classifier:
    st.markdown("<div style='background:#0f172a;border-radius:14px;padding:20px;margin-bottom:18px;'>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:13px;font-weight:700;margin-bottom:14px;'>Per-Class Precision · Recall · F1</div>", unsafe_allow_html=True)

    df_pc = pd.DataFrame(data["perClass"])
    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(name="Precision", x=df_pc["class"], y=df_pc["precision"],
                             marker_color="#38bdf8", marker_line_width=0))
    fig_bar.add_trace(go.Bar(name="Recall",    x=df_pc["class"], y=df_pc["recall"],
                             marker_color="#818cf8", marker_line_width=0))
    fig_bar.add_trace(go.Bar(name="F1",        x=df_pc["class"], y=df_pc["f1"],
                             marker_color="#34d399", marker_line_width=0))
    fig_bar.update_layout(
        paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
        font=dict(family="JetBrains Mono", color="#94a3b8"),
        barmode="group", bargap=0.2, bargroupgap=0.05,
        xaxis=dict(color="#334155", tickfont=dict(color="#94a3b8", size=11), gridcolor="#1e293b"),
        yaxis=dict(range=[0, 1], color="#334155", tickfont=dict(color="#475569", size=11), gridcolor="#1e293b"),
        legend=dict(font=dict(size=11, color="#64748b"), bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=10, r=10, t=10, b=10), height=230,
    )
    st.plotly_chart(fig_bar, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    cm_col, metrics_col = st.columns([2, 1])

    with cm_col:
        st.markdown("<div style='background:#0f172a;border-radius:14px;padding:20px;'>", unsafe_allow_html=True)
        st.markdown("<div style='font-size:13px;font-weight:700;margin-bottom:12px;'>Confusion Matrix</div>", unsafe_allow_html=True)

        confusion = data["confusion"]
        flat_vals = [v for row in confusion for v in row]
        max_val = max(flat_vals) if flat_vals else 1

        # Build heatmap
        z = confusion
        text_vals = [[str(confusion[i][j]) if confusion[i][j] > 0 else "" for j in range(5)] for i in range(5)]
        colors = []
        for i in range(5):
            row_colors = []
            for j in range(5):
                val = confusion[i][j]
                intensity = val / max_val if max_val > 0 else 0
                if i == j:
                    row_colors.append(f"rgba(56,189,248,{0.15 + intensity * 0.7:.2f})")
                else:
                    row_colors.append(f"rgba(248,113,113,{intensity * 0.5:.2f})")
            colors.append(row_colors)

        fig_cm = go.Figure(go.Heatmap(
            z=[[confusion[i][j] for j in range(5)] for i in range(5)],
            x=CM_LABELS, y=CM_LABELS,
            text=text_vals,
            texttemplate="%{text}",
            textfont=dict(size=12, color="#f1f5f9"),
            colorscale=[[0, "#0f172a"], [0.5, "#1e3a5f"], [1, "#38bdf8"]],
            showscale=False,
            hoverongaps=False,
        ))
        fig_cm.update_layout(
            paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
            font=dict(family="JetBrains Mono", color="#94a3b8"),
            xaxis=dict(title="Predicted", tickfont=dict(size=10, color="#64748b"), side="bottom"),
            yaxis=dict(title="Actual", tickfont=dict(size=10, color="#64748b"), autorange="reversed"),
            margin=dict(l=10, r=10, t=10, b=10), height=280,
        )
        st.plotly_chart(fig_cm, use_container_width=True)

        st.markdown("""
        <div style="display:flex;gap:12px;justify-content:center;margin-top:4px;">
          <span style="font-size:10px;color:#64748b;">■ <span style="color:#38bdf8;">Correct</span></span>
          <span style="font-size:10px;color:#64748b;">■ <span style="color:#f87171;">Misclassified</span></span>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with metrics_col:
        st.markdown("<div style='background:#0f172a;border-radius:14px;padding:20px;'>", unsafe_allow_html=True)
        st.markdown("<div style='font-size:13px;font-weight:700;margin-bottom:12px;'>Class Metrics</div>", unsafe_allow_html=True)

        rows = ""
        for r in data["perClass"]:
            c = THREAT_COLORS.get(r["class"], "#94a3b8")
            rows += f"""
            <tr style="border-bottom:1px solid #1e293b;">
              <td style="padding:7px 6px;color:{c};font-weight:700;">{r['class']}</td>
              <td style="padding:7px 6px;color:#94a3b8;">{r['precision']:.3f}</td>
              <td style="padding:7px 6px;color:#94a3b8;">{r['recall']:.3f}</td>
              <td style="padding:7px 6px;color:#4ade80;">{r['f1']:.3f}</td>
            </tr>
            """
        st.markdown(f"""
        <table style="width:100%;font-size:11px;border-collapse:collapse;">
          <thead>
            <tr style="color:#475569;">
              <th style="padding:4px 6px;text-align:left;">Class</th>
              <th style="padding:4px 6px;text-align:left;">P</th>
              <th style="padding:4px 6px;text-align:left;">R</th>
              <th style="padding:4px 6px;text-align:left;">F1</th>
            </tr>
          </thead>
          <tbody>{rows}</tbody>
        </table>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("<hr style='margin-top:30px;border-color:#1e293b;'>", unsafe_allow_html=True)
st.markdown(f"""
<div style="display:flex;justify-content:space-between;font-size:10px;color:#334155;padding-bottom:14px;">
  <span>SimCLR (128-d) · Temporal Autoencoder · XGBoost</span>
  <span>{st.session_state.analysed_video}</span>
</div>
""", unsafe_allow_html=True)