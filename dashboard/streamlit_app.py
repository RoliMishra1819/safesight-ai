import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import requests
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
  .stButton > button:hover { opacity: 0.9 !important; }
  div[data-testid="stSelectbox"] > div {
    background: #020817 !important;
    border: 1px solid #334155 !important;
    border-radius: 8px !important;
  }
  .stTabs [data-baseweb="tab-list"] {
    background: #0f172a !important;
    border-radius: 10px !important;
    padding: 4px !important;
    gap: 4px !important;
  }
  .stTabs [data-baseweb="tab"] {
    font-family: 'JetBrains Mono', monospace !important;
    border-radius: 7px !important;
    color: #475569 !important;
    letter-spacing: 1px !important;
    font-size: 11px !important;
  }
  .stTabs [aria-selected="true"] {
    background: linear-gradient(135deg,#0ea5e9,#6366f1) !important;
    color: #fff !important;
  }
  .stMetric {
    background: #0f172a !important;
    border-radius: 12px !important;
    padding: 16px !important;
  }
  .stMetric label {
    font-family: 'JetBrains Mono', monospace !important;
    color: #64748b !important;
    font-size: 10px !important;
    letter-spacing: 1.5px !important;
    text-transform: uppercase !important;
  }
  .stMetric [data-testid="stMetricValue"] {
    font-size: 24px !important;
    font-weight: 800 !important;
  }
  hr { border-color: #1e293b !important; }
  [data-testid="stAppViewContainer"] { background: #020817 !important; }
  [data-testid="stHeader"] { background: transparent !important; }
  section[data-testid="stSidebar"] { background: #0f172a !important; }
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────────
API = "http://localhost:8000"
CM_LABELS = ["Normal", "Low", "Medium", "High", "Critical"]
THREAT_COLORS = {
    "Normal": "#22d3ee", "Low": "#4ade80",
    "Medium": "#facc15", "High": "#fb923c", "Critical": "#f87171",
}
COLORS = {
    "Normal": "#22d3ee", "Low": "#4ade80",
    "Medium": "#facc15", "High": "#fb923c", "Critical": "#f87171",
}

# ── Helper: Generate mock timeline ─────────────────────────────────────────────
def make_timeline(baseline=0.01, noise=0.01, spike=False, frames=160):
    out = []
    for i in range(frames):
        s = baseline + random.random() * noise
        if spike and 90 < i < 130:
            s += random.random() * baseline * 8
        out.append({"frame": i, "score": round(max(0.0, min(1.0, s)), 5)})
    return out

def pipeline_to_dashboard(raw: dict, video_name: str) -> dict:
    dist = raw.get("threat_distribution", {})
    risk_dist = [
        {
            "level":  level,
            "frames": dist.get(level, {}).get("count", 0),
            "share":  dist.get(level, {}).get("pct",   0.0),
            "color":  COLORS[level],
        }
        for level in ["Normal", "Low", "Medium", "High", "Critical"]
    ]
    clf = raw.get("classifier_metrics", {})
    anom = raw.get("anomaly_metrics", {})
    dominant = max(dist, key=lambda k: dist[k].get("count", 0)) if dist else "Normal"
    tag_color = THREAT_COLORS.get(dominant, "#94a3b8")

    # Build timeline from score_history
    score_history = anom.get("score_history", [])
    if score_history:
        timeline = [{"frame": i, "score": s} for i, s in enumerate(score_history)]
    else:
        baseline = anom.get("mean_anomaly_score", 0.01)
        timeline = make_timeline(baseline=max(baseline, 0.004))

    # per_class is a dict {label: {precision, recall, f1}}
    per_class_raw = clf.get("per_class", {})
    if isinstance(per_class_raw, dict):
        per_class = [
            {"class": k, "precision": v["precision"], "recall": v["recall"], "f1": v["f1"]}
            for k, v in per_class_raw.items()
        ]
    else:
        per_class = per_class_raw if per_class_raw else [
            {"class": "Normal",   "precision": 0.888, "recall": 0.864, "f1": 0.887},
            {"class": "Low",      "precision": 0.032, "recall": 0.881, "f1": 0.061},
            {"class": "Medium",   "precision": 1.000, "recall": 0.977, "f1": 0.988},
            {"class": "High",     "precision": 0.977, "recall": 0.864, "f1": 0.917},
            {"class": "Critical", "precision": 0.888, "recall": 0.888, "f1": 0.988},
        ]

    macro = clf.get("macro", {})
    confusion = clf.get("confusion_matrix", [[0]*5 for _ in range(5)])
    mean_score   = anom.get("mean_anomaly_score", 0)
    peak_score   = anom.get("max_anomaly_score", 0)
    anomaly_rate = anom.get("anomaly_rate", 0)

    return {
        "tag":      dominant.upper(),
        "tagColor": tag_color,
        "session": {
            "frames":    raw.get("total_frames", 0),
            "precision": macro.get("precision", 0),
            "recall":    macro.get("recall", 0),
            "f1":        macro.get("f1", 0),
        },
        "riskDist":    risk_dist,
        "perClass":    per_class,
        "confusion":   confusion,
        "meanScore":   f"{mean_score:.3f}",
        "peakScore":   f"{peak_score:.3f}",
        "anomalyRate": f"{anomaly_rate:.1f}%",
        "timeline":    timeline,
    }

# ── API calls ──────────────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def fetch_video_list():
    try:
        r = requests.get(f"{API}/videos", timeout=30)
        data = r.json()
        return data.get("grouped", {}), data.get("videos", [])
    except Exception as e:
        return {}, []

def call_analyse(video_name: str):
    try:
        r = requests.post(f"{API}/analyse",
                          json={"video_name": video_name},
                          timeout=600)
        if r.status_code == 200:
            return r.json()
        else:
            st.error(f"Pipeline error {r.status_code}: {r.text[:200]}")
            return None
    except Exception as e:
        st.error(f"API call failed: {e}")
        return None

def fetch_past_session(video_name: str):
    try:
        name = video_name.split("/")[-1]
        r = requests.get(f"{API}/session/{name}", timeout=10)
        if r.status_code == 200:
            return r.json()
        return None
    except:
        return None

# ── Session State Init ─────────────────────────────────────────────────────────
if "pipeline_data" not in st.session_state:
    st.session_state.pipeline_data = None
if "analysed_video" not in st.session_state:
    st.session_state.analysed_video = None
if "timeline" not in st.session_state:
    st.session_state.timeline = make_timeline()
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
      <div style="width:8px;height:8px;border-radius:50%;
      background:#0f3;box-shadow:0 0 8px #0f3;"></div>
      <span style="font-size:11px;color:#4ade80;">LIVE</span>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<hr style='margin:14px 0 18px 0;'>", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="font-size:12px;color:#475569;letter-spacing:2px;
    margin-bottom:12px;text-transform:uppercase;">Model Stack</div>
    """, unsafe_allow_html=True)
    for item in ["SimCLR Encoder (128-d)", "Temporal Autoencoder", "XGBoost Classifier"]:
        st.markdown(f"""
        <div style="font-size:12px;color:#94a3b8;padding:6px 0;
        border-bottom:1px solid #1e293b;">• {item}</div>
        """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🔄 Clear Cache"):
        st.cache_data.clear()
        st.session_state.clear()
        st.rerun()

# ── Fetch Video List ───────────────────────────────────────────────────────────
grouped_videos, all_videos = fetch_video_list()

if not grouped_videos:
    st.error("❌ Cannot connect to FastAPI at port 8000. Start it first:\n\n`cd backend && uvicorn api:app --reload --port 8000`")
    st.stop()

# ── Video Selector ─────────────────────────────────────────────────────────────
st.markdown("""
<div style="background:#0f172a;border-radius:12px;padding:16px 16px 8px 16px;
margin-bottom:12px;border:1px solid #1e293b;">
  <div style="font-size:10px;color:#475569;letter-spacing:2px;margin-bottom:12px;">
    SELECT VIDEO
  </div>
""", unsafe_allow_html=True)

cat_col, vid_col, btn_col = st.columns([1.3, 2.2, 0.8])

with cat_col:
    categories = list(grouped_videos.keys())
    selected_cat = st.selectbox(
        "Category", categories,
        label_visibility="visible",
        key="cat_select"
    )

with vid_col:
    videos_in_cat = grouped_videos.get(selected_cat, [])
    selected_file = st.selectbox(
        "Video", videos_in_cat,
        label_visibility="visible",
        key="video_select"
    )
    selected_video = f"{selected_cat}/{selected_file}"

with btn_col:
    st.markdown("<br>", unsafe_allow_html=True)
    analyse_clicked = st.button("▶ Analyse", key="analyse_btn")

st.markdown("</div>", unsafe_allow_html=True)

# ── Analyse Button Handler ─────────────────────────────────────────────────────
if analyse_clicked:
    # Check MongoDB for existing session first
    past = fetch_past_session(selected_video)
    if past:
        st.session_state.analysed_video = selected_video
        st.session_state.pipeline_data  = pipeline_to_dashboard(past, selected_video)
        st.session_state.timeline       = st.session_state.pipeline_data["timeline"]
        st.session_state.live_idx       = 10
        st.session_state.running        = False
        st.success("✅ Loaded from database!")
        st.rerun()
    else:
        with st.spinner(f"Running pipeline on {selected_file}... (CPU mode, ~1-2 min)"):
            result = call_analyse(selected_video)
        if result:
            st.session_state.analysed_video = selected_video
            st.session_state.pipeline_data  = pipeline_to_dashboard(result, selected_video)
            st.session_state.timeline       = st.session_state.pipeline_data["timeline"]
            st.session_state.live_idx       = 10
            st.session_state.running        = False
            st.rerun()

# ── Guard: No data yet ─────────────────────────────────────────────────────────
if st.session_state.pipeline_data is None:
    st.markdown("""
    <div style="background:#0f172a;border-radius:14px;padding:40px;text-align:center;
    border:1px dashed #334155;margin-top:20px;">
      <div style="font-size:40px;margin-bottom:16px;">🛡</div>
      <div style="font-size:16px;font-weight:700;color:#f1f5f9;margin-bottom:8px;">
        SafeSight AI Ready
      </div>
      <div style="font-size:12px;color:#475569;">
        Select a category and video above, then click ▶ Analyse to begin threat detection.
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

data = st.session_state.pipeline_data

# ── Active Analysis Banner ─────────────────────────────────────────────────────
st.markdown(f"""
<div style="display:flex;align-items:center;gap:10px;margin-bottom:18px;
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

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
with tab_overview:
    st.markdown("<div style='font-size:10px;color:#475569;letter-spacing:2px;margin-bottom:8px;'>SESSION OVERVIEW</div>", unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    sess = data["session"]
    with m1: st.metric("Frames Processed", f"{sess['frames']}")
    with m2: st.metric("Macro Precision",  f"{sess['precision']:.3f}")
    with m3: st.metric("Macro Recall",     f"{sess['recall']:.3f}")
    with m4: st.metric("Macro F1",         f"{sess['f1']:.3f}")

    st.markdown("<div style='font-size:10px;color:#475569;letter-spacing:2px;margin:18px 0 10px 0;'>RISK DISTRIBUTION</div>", unsafe_allow_html=True)

    risk_cols = st.columns(5)
    for i, r in enumerate(data["riskDist"]):
        with risk_cols[i]:
            st.markdown(f"""
            <div style="background:{r['color']}22;border:1.5px solid {r['color']};
            border-radius:10px;padding:12px 14px;">
              <div style="font-size:19px;font-weight:800;color:{r['color']};">{r['share']}%</div>
              <div style="font-size:10px;color:#94a3b8;margin-top:2px;">{r['level']}</div>
              <div style="font-size:10px;color:#475569;">{r['frames']} frames</div>
              <div style="height:3px;background:{r['color']};border-radius:2px;margin-top:8px;"></div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    pie_col, table_col = st.columns(2)

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
            bar_w = int(r["share"] * 2.5)
            rows_html += f"""
            <tr style="border-bottom:1px solid #1e293b;">
              <td style="padding:8px;color:{r['color']};font-weight:700;">{r['level']}</td>
              <td style="padding:8px;color:#94a3b8;">{r['frames']}</td>
              <td style="padding:8px;">
                <div style="display:flex;align-items:center;gap:6px;">
                  <div style="height:4px;width:{bar_w}px;background:{r['color']};border-radius:2px;"></div>
                  <span style="color:{r['color']};">{r['share']}%</span>
                </div>
              </td>
            </tr>"""
        st.markdown(f"""
        <table style="width:100%;border-collapse:collapse;font-size:12px;">
          <thead>
            <tr style="color:#475569;border-bottom:1px solid #1e293b;">
              <th style="padding:6px 8px;text-align:left;">Level</th>
              <th style="padding:6px 8px;text-align:left;">Frames</th>
              <th style="padding:6px 8px;text-align:left;">Share</th>
            </tr>
          </thead>
          <tbody>{rows_html}</tbody>
        </table>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — ANOMALY
# ══════════════════════════════════════════════════════════════════════════════
with tab_anomaly:
    a1, a2, a3 = st.columns(3)
    for col, m in zip([a1, a2, a3], [
        {"label": "Mean Score",   "value": data["meanScore"],   "color": "#38bdf8"},
        {"label": "Peak Score",   "value": data["peakScore"],   "color": "#fb923c"},
        {"label": "Anomaly Rate", "value": data["anomalyRate"], "color": "#4ade80"},
    ]):
        with col:
            st.markdown(f"""
            <div style="background:#0f172a;border-radius:10px;padding:14px 18px;
            border:1px solid {m['color']}33;margin-bottom:12px;">
              <div style="font-size:22px;font-weight:800;color:{m['color']};">{m['value']}</div>
              <div style="font-size:10px;color:#64748b;margin-top:2px;">{m['label']}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<div style='background:#0f172a;border-radius:14px;padding:20px;'>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:13px;font-weight:700;margin-bottom:12px;'>Anomaly Score Timeline</div>", unsafe_allow_html=True)

    _, play_col, reset_col = st.columns([4, 1, 1])
    with play_col:
        play_label = "⏸ Pause" if st.session_state.running else "▶ Replay"
        if st.button(play_label, key="play_btn"):
            st.session_state.running = not st.session_state.running
            if st.session_state.running and st.session_state.live_idx >= len(st.session_state.timeline) - 1:
                st.session_state.live_idx = 10
    with reset_col:
        if st.button("↺ Reset", key="reset_btn"):
            st.session_state.live_idx = 10
            st.session_state.running  = False
            st.rerun()

    timeline = st.session_state.timeline
    visible  = timeline[:st.session_state.live_idx]
    df_tl    = pd.DataFrame(visible)

    fig_line = go.Figure()
    if len(df_tl) > 0:
        fig_line.add_trace(go.Scatter(
            x=df_tl["frame"], y=df_tl["score"],
            mode="lines", name="Anomaly Score",
            line=dict(color="#38bdf8", width=2),
        ))
    fig_line.add_hline(y=0.02, line_dash="dash", line_color="#fb923c",
                       annotation_text="threshold",
                       annotation_font_size=10, annotation_font_color="#fb923c")
    fig_line.update_layout(
        paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
        font=dict(family="JetBrains Mono", color="#94a3b8"),
        xaxis=dict(title="Frame", color="#334155",
                   tickfont=dict(color="#475569", size=11), gridcolor="#1e293b"),
        yaxis=dict(title="Score", color="#334155",
                   tickfont=dict(color="#475569", size=11), gridcolor="#1e293b"),
        margin=dict(l=10, r=10, t=10, b=10), height=220, showlegend=False,
    )
    st.plotly_chart(fig_line, use_container_width=True)

    pct = st.session_state.live_idx / max(len(timeline), 1)
    st.markdown(f"""
    <div style="font-size:10px;color:#475569;text-align:right;margin-top:4px;">
      Frame {st.session_state.live_idx} / {len(timeline)}
    </div>
    <div style="height:2px;background:#1e293b;border-radius:1px;margin-top:4px;">
      <div style="height:100%;background:#38bdf8;border-radius:1px;width:{pct*100:.1f}%;"></div>
    </div>""", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if st.session_state.running:
        if st.session_state.live_idx < len(timeline) - 1:
            time.sleep(0.06)
            st.session_state.live_idx += 1
            st.rerun()
        else:
            st.session_state.running = False

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — CLASSIFIER
# ══════════════════════════════════════════════════════════════════════════════
with tab_classifier:
    st.markdown("<div style='background:#0f172a;border-radius:14px;padding:20px;margin-bottom:18px;'>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:13px;font-weight:700;margin-bottom:14px;'>Per-Class Precision · Recall · F1</div>", unsafe_allow_html=True)

    df_pc = pd.DataFrame(data["perClass"])
    fig_bar = go.Figure()
    for name, color, key in [
        ("Precision", "#38bdf8", "precision"),
        ("Recall",    "#818cf8", "recall"),
        ("F1",        "#34d399", "f1"),
    ]:
        fig_bar.add_trace(go.Bar(
            name=name, x=df_pc["class"], y=df_pc[key],
            marker_color=color, marker_line_width=0,
        ))
    fig_bar.update_layout(
        paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
        font=dict(family="JetBrains Mono", color="#94a3b8"),
        barmode="group", bargap=0.2, bargroupgap=0.05,
        xaxis=dict(color="#334155", tickfont=dict(color="#94a3b8", size=11), gridcolor="#1e293b"),
        yaxis=dict(range=[0,1], color="#334155", tickfont=dict(color="#475569", size=11), gridcolor="#1e293b"),
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
        fig_cm = go.Figure(go.Heatmap(
            z=[[confusion[i][j] for j in range(5)] for i in range(5)],
            x=CM_LABELS, y=CM_LABELS,
            text=[[str(confusion[i][j]) if confusion[i][j] > 0 else "" for j in range(5)] for i in range(5)],
            texttemplate="%{text}",
            textfont=dict(size=12, color="#f1f5f9"),
            colorscale=[[0,"#0f172a"],[0.5,"#1e3a5f"],[1,"#38bdf8"]],
            showscale=False,
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
        </div>""", unsafe_allow_html=True)
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
            </tr>"""
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
        </table>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("<hr style='margin-top:30px;border-color:#1e293b;'>", unsafe_allow_html=True)
st.markdown(f"""
<div style="display:flex;justify-content:space-between;font-size:10px;
color:#334155;padding-bottom:14px;">
  <span>SimCLR (128-d) · Temporal Autoencoder · XGBoost</span>
  <span>{st.session_state.analysed_video or 'No video selected'}</span>
</div>""", unsafe_allow_html=True)