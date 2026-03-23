import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.integrate import odeint

# ─── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Wave Energy Converter Sim",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');

  html, body, [class*="css"] {
      font-family: 'DM Sans', sans-serif;
      background-color: #0b1120;
      color: #e2e8f0;
  }
  .stApp { background-color: #0b1120; }

  h1, h2, h3 {
      font-family: 'Space Mono', monospace;
      letter-spacing: -0.02em;
  }

  /* Sidebar */
  section[data-testid="stSidebar"] {
      background: #111827;
      border-right: 1px solid #1e3a5f;
  }
  section[data-testid="stSidebar"] label {
      font-size: 0.78rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: #64b5f6;
  }

  /* Metric cards */
  [data-testid="metric-container"] {
      background: #111827;
      border: 1px solid #1e3a5f;
      border-radius: 10px;
      padding: 14px 18px;
  }
  [data-testid="metric-container"] label {
      color: #64b5f6 !important;
      font-size: 0.72rem !important;
      text-transform: uppercase;
      letter-spacing: 0.1em;
  }
  [data-testid="metric-container"] [data-testid="stMetricValue"] {
      font-family: 'Space Mono', monospace;
      font-size: 1.4rem;
      color: #e2e8f0;
  }

  /* Divider */
  hr { border-color: #1e3a5f; }

  /* Headers */
  .block-container { padding-top: 1.8rem; }
  .section-title {
      font-family: 'Space Mono', monospace;
      font-size: 0.7rem;
      text-transform: uppercase;
      letter-spacing: 0.15em;
      color: #3b82f6;
      margin-bottom: 0.3rem;
      border-bottom: 1px solid #1e3a5f;
      padding-bottom: 4px;
  }
  .main-title {
      font-family: 'Space Mono', monospace;
      font-size: 1.7rem;
      font-weight: 700;
      color: #e2e8f0;
      line-height: 1.15;
  }
  .sub-title {
      font-size: 0.9rem;
      color: #64748b;
      margin-top: 2px;
  }
  .badge {
      display: inline-block;
      font-size: 0.65rem;
      font-family: 'Space Mono', monospace;
      background: #1e3a5f;
      color: #64b5f6;
      border-radius: 4px;
      padding: 2px 8px;
      margin-right: 4px;
      letter-spacing: 0.05em;
  }
</style>
""", unsafe_allow_html=True)


# ─── Physics Model ─────────────────────────────────────────────────────────────
def wave_excitation_force(t, H_wave, T_wave, F_exc_coeff):
    """Sinusoidal wave excitation force."""
    omega = 2 * np.pi / T_wave
    return F_exc_coeff * H_wave * np.sin(omega * t)

def wec_ode(y, t, params):
    """
    2-DOF WEC model:
      State: [z, z_dot]  (buoy displacement & velocity)
    Forces:
      - Wave excitation
      - Hydrostatic restoring (spring)
      - Radiation damping
      - PTO (spring + damper)
      - Mooring (linear spring)
    """
    z, z_dot = y
    p = params

    omega = 2 * np.pi / p['T_wave']
    F_wave = p['F_exc'] * p['H_wave'] * np.sin(omega * t)

    F_hydrostatic = -p['k_hydro'] * z
    F_radiation   = -p['b_rad'] * z_dot
    F_pto         = -(p['k_pto'] * z + p['b_pto'] * z_dot)
    F_mooring     = -p['k_moor'] * z

    m_total = p['mass'] + p['m_added']
    z_ddot  = (F_wave + F_hydrostatic + F_radiation + F_pto + F_mooring) / m_total

    return [z_dot, z_ddot]

def run_simulation(params, t_end=60.0, dt=0.02):
    t = np.arange(0, t_end, dt)
    y0 = [0.0, 0.0]
    sol = odeint(wec_ode, y0, t, args=(params,))

    z     = sol[:, 0]
    z_dot = sol[:, 1]

    # PTO power
    P_pto = params['b_pto'] * z_dot**2

    # Spring cylinder force
    F_spring = params['k_pto'] * z

    # Damping cylinder force
    F_damp = params['b_pto'] * z_dot

    # Accumulator pressure proxy (normalised)
    P_acc = np.abs(F_spring) / (params['k_pto'] + 1e-9) * 10

    return t, z, z_dot, P_pto, F_spring, F_damp, P_acc


# ─── Sidebar Parameters ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="section-title">🌊 Wave Input</div>', unsafe_allow_html=True)
    H_wave   = st.slider("Wave Height H (m)",    0.5, 5.0, 2.0, 0.1)
    T_wave   = st.slider("Wave Period T (s)",     4.0, 20.0, 8.0, 0.5)
    F_exc    = st.slider("Excitation Coeff",      500.0, 5000.0, 2000.0, 100.0)

    st.markdown('<div class="section-title" style="margin-top:1rem">⚙️ Buoy / Mass</div>', unsafe_allow_html=True)
    mass     = st.slider("Buoy Mass (kg)",        500.0, 5000.0, 1500.0, 100.0)
    m_added  = st.slider("Added Mass (kg)",       100.0, 2000.0, 500.0, 50.0)
    b_rad    = st.slider("Radiation Damping",     100.0, 2000.0, 400.0, 50.0)
    k_hydro  = st.slider("Hydrostatic Stiffness", 1000.0, 20000.0, 8000.0, 500.0)

    st.markdown('<div class="section-title" style="margin-top:1rem">🔴 Spring Cylinder</div>', unsafe_allow_html=True)
    k_pto    = st.slider("Spring Stiffness k_PTO", 500.0, 15000.0, 4000.0, 250.0)

    st.markdown('<div class="section-title" style="margin-top:1rem">🟡 Damping Cylinder</div>', unsafe_allow_html=True)
    b_pto    = st.slider("Damping Coeff b_PTO",   100.0, 5000.0, 1200.0, 100.0)

    st.markdown('<div class="section-title" style="margin-top:1rem">⚓ Mooring</div>', unsafe_allow_html=True)
    k_moor   = st.slider("Mooring Stiffness",     0.0, 2000.0, 200.0, 50.0)

    st.markdown('<div class="section-title" style="margin-top:1rem">⏱ Simulation</div>', unsafe_allow_html=True)
    t_end    = st.slider("Duration (s)",          20, 120, 60, 10)

params = dict(
    H_wave=H_wave, T_wave=T_wave, F_exc=F_exc,
    mass=mass, m_added=m_added, b_rad=b_rad, k_hydro=k_hydro,
    k_pto=k_pto, b_pto=b_pto, k_moor=k_moor,
)

# ─── Run ───────────────────────────────────────────────────────────────────────
t, z, z_dot, P_pto, F_spring, F_damp, P_acc = run_simulation(params, t_end=t_end)

# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-bottom:1.2rem">
  <div class="main-title">Wave Energy Converter</div>
  <div class="sub-title">Hydraulic PTO Simulation — Spring &amp; Damping Cylinder Circuit</div>
  <div style="margin-top:8px">
    <span class="badge">PTO</span>
    <span class="badge">SPRING CYL</span>
    <span class="badge">DAMP CYL</span>
    <span class="badge">ACCUMULATOR</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ─── KPI Metrics ──────────────────────────────────────────────────────────────
avg_power  = np.mean(P_pto)
peak_power = np.max(P_pto)
max_disp   = np.max(np.abs(z))
max_vel    = np.max(np.abs(z_dot))

c1, c2, c3, c4 = st.columns(4)
c1.metric("Avg PTO Power",   f"{avg_power:.1f} W")
c2.metric("Peak PTO Power",  f"{peak_power:.1f} W")
c3.metric("Max Displacement",f"{max_disp:.3f} m")
c4.metric("Max Velocity",    f"{max_vel:.3f} m/s")

st.markdown("<hr>", unsafe_allow_html=True)

# ─── Plot colours ─────────────────────────────────────────────────────────────
CLR_WAVE   = "#64b5f6"   # blue  — wave / displacement
CLR_VEL    = "#4dd0e1"   # cyan  — velocity
CLR_SPRING = "#ef5350"   # red   — spring force  (matches Simulink pink)
CLR_DAMP   = "#ffd54f"   # amber — damping force (matches Simulink yellow)
CLR_POWER  = "#66bb6a"   # green — power
CLR_ACC    = "#ab47bc"   # purple — accumulator
BG         = "#0b1120"
GRID       = "#1e3a5f"
PAPER      = "#111827"

def style_fig(fig):
    fig.update_layout(
        paper_bgcolor=PAPER, plot_bgcolor=BG,
        font=dict(family="DM Sans", color="#94a3b8", size=11),
        margin=dict(l=10, r=10, t=36, b=10),
        legend=dict(bgcolor="rgba(0,0,0,0)", font_size=10),
        hovermode="x unified",
    )
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID)
    return fig

# ─── Row 1: Displacement + Velocity ──────────────────────────────────────────
st.markdown('<div class="section-title">Buoy Motion — PTO_Velocity / WT_PTO</div>', unsafe_allow_html=True)
fig1 = make_subplots(specs=[[{"secondary_y": True}]])
fig1.add_trace(go.Scatter(x=t, y=z,     name="Displacement z (m)",  line=dict(color=CLR_WAVE, width=2)), secondary_y=False)
fig1.add_trace(go.Scatter(x=t, y=z_dot, name="Velocity ż (m/s)",    line=dict(color=CLR_VEL,  width=1.5, dash="dot")), secondary_y=True)
fig1.update_yaxes(title_text="z  [m]",    secondary_y=False, title_font_color=CLR_WAVE)
fig1.update_yaxes(title_text="ż  [m/s]", secondary_y=True,  title_font_color=CLR_VEL)
fig1.update_xaxes(title_text="Time [s]")
fig1.update_layout(title="Buoy Displacement & Velocity", height=280)
style_fig(fig1)
st.plotly_chart(fig1, use_container_width=True)

# ─── Row 2: Spring + Damping forces side by side ──────────────────────────────
col_l, col_r = st.columns(2)

with col_l:
    st.markdown('<div class="section-title">Spring Cylinder Circuit — PT_Spring / PT_Blind</div>', unsafe_allow_html=True)
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=t, y=F_spring, name="Spring Force (N)",
                              line=dict(color=CLR_SPRING, width=2),
                              fill='tozeroy', fillcolor='rgba(239,83,80,0.08)'))
    fig2.update_layout(title="Spring Cylinder Force",
                       xaxis_title="Time [s]", yaxis_title="Force [N]", height=260)
    style_fig(fig2)
    st.plotly_chart(fig2, use_container_width=True)

with col_r:
    st.markdown('<div class="section-title">Damping Cylinder Circuit — PT_Damp_Rod / PT_Damp_LP</div>', unsafe_allow_html=True)
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=t, y=F_damp, name="Damping Force (N)",
                              line=dict(color=CLR_DAMP, width=2),
                              fill='tozeroy', fillcolor='rgba(255,213,79,0.07)'))
    fig3.update_layout(title="Damping Cylinder Force",
                       xaxis_title="Time [s]", yaxis_title="Force [N]", height=260)
    style_fig(fig3)
    st.plotly_chart(fig3, use_container_width=True)

# ─── Row 3: PTO Power + Phase Portrait ────────────────────────────────────────
col3, col4 = st.columns(2)

with col3:
    st.markdown('<div class="section-title">PTO Power Output — ZT_PTO / ST_01</div>', unsafe_allow_html=True)
    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(x=t, y=P_pto, name="P_PTO (W)",
                              line=dict(color=CLR_POWER, width=2),
                              fill='tozeroy', fillcolor='rgba(102,187,106,0.1)'))
    fig4.add_hline(y=avg_power, line_dash="dash", line_color="#a5d6a7",
                   annotation_text=f"Avg: {avg_power:.1f} W", annotation_font_color="#a5d6a7")
    fig4.update_layout(title="Instantaneous PTO Power",
                       xaxis_title="Time [s]", yaxis_title="Power [W]", height=280)
    style_fig(fig4)
    st.plotly_chart(fig4, use_container_width=True)

with col4:
    st.markdown('<div class="section-title">Phase Portrait — z vs ż</div>', unsafe_allow_html=True)
    fig5 = go.Figure()
    fig5.add_trace(go.Scatter(
        x=z, y=z_dot, mode='lines',
        line=dict(color=CLR_VEL, width=1.2),
        name="Phase trajectory",
    ))
    # colour by time
    n_seg = 200
    idx = np.linspace(0, len(t)-2, n_seg, dtype=int)
    for i in idx:
        alpha = i / len(t)
        r = int(100 + 155*alpha)
        fig5.add_trace(go.Scatter(x=z[i:i+2], y=z_dot[i:i+2], mode='lines',
                                  line=dict(color=f'rgba({r},180,{255-r},0.6)', width=1.5),
                                  showlegend=False))
    fig5.update_layout(title="Phase Portrait (z vs ż)",
                       xaxis_title="Displacement z [m]", yaxis_title="Velocity ż [m/s]", height=280)
    style_fig(fig5)
    st.plotly_chart(fig5, use_container_width=True)

# ─── Row 4: Accumulator pressure proxy ────────────────────────────────────────
st.markdown('<div class="section-title">Spring Accumulator Pressure Proxy — ULS_Accumulator</div>', unsafe_allow_html=True)
fig6 = go.Figure()
fig6.add_trace(go.Scatter(x=t, y=P_acc, name="Acc. Pressure (bar equiv.)",
                          line=dict(color=CLR_ACC, width=1.8),
                          fill='tozeroy', fillcolor='rgba(171,71,188,0.08)'))
fig6.update_layout(title="Accumulator Pressure (Proportional to Spring Force)",
                   xaxis_title="Time [s]", yaxis_title="Pressure [bar equiv.]", height=220)
style_fig(fig6)
st.plotly_chart(fig6, use_container_width=True)

# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("""
<hr>
<div style="font-family:'Space Mono',monospace; font-size:0.65rem; color:#334155; text-align:center; padding:8px 0">
  WEC Hydraulic PTO Simulation &nbsp;·&nbsp; Spring Cylinder + Damping Cylinder + Accumulator &nbsp;·&nbsp; 2-DOF Linear Model
</div>
""", unsafe_allow_html=True)
