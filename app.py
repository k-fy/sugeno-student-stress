import io
import time

import numpy as np
import streamlit as st
import matplotlib
matplotlib.use("Agg")  # backend non-GUI, lebih ringan; harus sebelum import pyplot
import matplotlib.pyplot as plt

from fuzzy_core import (
    MF_SLEEP, MF_STUDY, RULE_BASE, CONSEQUENT, T1, T2, DOMAIN, TRAIN_RANGE, trapmf, infer,
)

st.set_page_config(page_title="Know Your Academic Stress", page_icon="💌", layout="centered")

# ------------------------------------------------------------
# Color palette (from the design inspiration)
# ------------------------------------------------------------
BLACK, UMBER = "#252525", "#443A35"
DUNE, VANILLA, TAUPE = "#E4DDCC", "#F8F4EE", "#C5B49D"
CATEGORY_COLOR = {"Low": "#8FAE8B", "Moderate": "#D4B062", "High": "#C4695D"}

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Figtree:ital,wght@0,400;0,600;1,400;1,600&display=swap');

html, body, [data-testid="stApp"] {{ background:{BLACK}; color:{VANILLA}; }}
header[data-testid="stHeader"], footer, #MainMenu, section[data-testid="stSidebar"] {{ display:none; }}
.block-container {{ max-width:1000px; padding-top:1.2rem; padding-bottom:60vh; }}

/* ---------- running text (ticker) ---------- */
.ticker {{ overflow:hidden; white-space:nowrap; width:100%; }}
.ticker-track {{ display:flex; width:max-content; animation:slide 30s linear infinite; }}
.ticker:hover .ticker-track {{ animation-play-state:paused; }}
.group {{ display:flex; flex-shrink:0; }}
.group span {{ padding-right:90px; font-family:'Times New Roman',Times,serif;
              font-style:italic; font-size:14px; color:{VANILLA}; }}
.group b {{ font-weight:700; }}
@keyframes slide {{ from {{ transform:translateX(0); }} to {{ transform:translateX(-50%); }} }}

/* ---------- hero ---------- */
.hero-sub {{ text-align:center; font-family:'Times New Roman',Times,serif; font-style:italic;
            font-size:24px; margin:90px 0 50px; color:{VANILLA}; }}
.hero-title {{ text-align:center; font-family:'Times New Roman',Times,serif; font-style:italic;
              font-size:clamp(38px,6.5vw,64px); line-height:1.1; margin:0 0 34px; color:{VANILLA}; }}
.hero-title .acc {{ color:{TAUPE}; text-decoration:underline; text-underline-offset:6px; }}

/* ---------- form ---------- */
[data-testid="stForm"] {{ border:none; padding:0; background:transparent; }}
[data-testid="stWidgetLabel"] p {{ font-family:'Figtree',sans-serif; font-style:italic;
                                  font-size:15px; color:{VANILLA}; }}
div[data-baseweb="input"], div[data-baseweb="base-input"] {{
    background:{VANILLA} !important; border-radius:999px !important; border:none !important; }}
div[data-baseweb="input"] input {{ color:{BLACK} !important; text-align:center; font-family:'Figtree',sans-serif;
                                  -webkit-text-fill-color:{BLACK}; }}
[data-testid="stNumberInputStepUp"], [data-testid="stNumberInputStepDown"] {{ display:none; }}
[data-testid="stFormSubmitButton"] button {{
    background:{TAUPE}; color:{UMBER}; border:none; border-radius:999px; height:46px;
    font-family:'Figtree',sans-serif; font-weight:600; font-style:italic; }}
[data-testid="stFormSubmitButton"] button:hover {{ background:{DUNE}; color:{BLACK}; }}

/* ---------- result ---------- */
.score-card {{ background:{UMBER}; border-radius:22px; padding:26px 20px; text-align:center; margin:10px 0 6px; }}
.score-card .lab {{ font-family:'Figtree',sans-serif; font-style:italic; color:{DUNE}; font-size:15px; }}
.score-card .z {{ font-family:'Times New Roman',Times,serif; font-style:italic; font-size:72px; line-height:1.1; color:{VANILLA}; }}
.badge {{ display:inline-block; padding:8px 28px; border-radius:999px; color:{BLACK};
         font-family:'Figtree',sans-serif; font-weight:600; letter-spacing:.08em; font-size:18px; }}
.sec-title {{ font-family:'Times New Roman',Times,serif; font-style:italic; font-size:30px; margin:34px 0 4px; }}

/* ---------- alerts (st.info / st.warning): brown instead of blue/yellow ---------- */
[data-testid="stAlert"] {{ background:transparent !important; }}
[data-testid="stAlertContainer"] {{ background:{UMBER} !important; border:1px solid {TAUPE}; border-radius:14px; }}
[data-testid="stAlertContainer"], [data-testid="stAlertContainer"] p {{ color:{VANILLA} !important; font-family:'Figtree',sans-serif; }}
[data-testid="stAlertContainer"] svg {{ fill:{TAUPE} !important; color:{TAUPE} !important; }}

/* ---------- tabs: taupe/brown instead of red ---------- */
[data-testid="stTab"] p {{ font-family:'Figtree',sans-serif; }}
[data-testid="stTab"][data-hovered] {{ color:{DUNE} !important; }}
[data-testid="stTab"][data-selected],
[data-testid="stTab"][data-focus-visible] {{ color:{TAUPE} !important; }}
[data-testid="stTab"][data-selected] .react-aria-SelectionIndicator {{ background-color:{TAUPE} !important; }}
[data-testid="stTab"][data-focus-visible] {{ box-shadow:none !important; }}
[data-testid="stTabs"] [role="tablist"]::after {{ background-color:{UMBER} !important; }}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# Top bar & hero
# ------------------------------------------------------------
item = (
    "<span><b>Kelompok 6 Soft Computing</b></span>"
    "<span>Anggita Kirana Puspa (24060124130064)</span>"
    "<span>Puti Shasta Khafiyani (24060124140132)</span>"
    "<span>Varissa Nabila Kifli (24060124140125)</span>"
)
group = f'<div class="group">{item * 2}</div>'   # repeated 2x so there is no gap on wide screens

st.markdown(f"""
<div class="ticker"><div class="ticker-track">{group}{group}</div></div>
<div class="hero-sub">we present you our project ...</div>
<div class="hero-title">Know <span class="acc">your</span> Academic Stress</div>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# Input form
# ------------------------------------------------------------
_, center, _ = st.columns([1, 6, 1])
with center:
    with st.form("input_form", border=False):
        c1, c2 = st.columns(2)
        sleep = c1.number_input("Sleep Hours", min_value=0.0, max_value=24.0, value=None,
                                step=0.1, format="%.1f", placeholder="0 – 24")
        study = c2.number_input("Study Hours", min_value=0.0, max_value=24.0, value=None,
                                step=0.1, format="%.1f", placeholder="0 – 24")
        submitted = st.form_submit_button("Submit", width="stretch")

if submitted:
    if sleep is None or study is None:
        with center:
            st.warning("Please fill in both Sleep Hours and Study Hours (range 0–24 hours).")
    elif sleep + study > 24:
        with center:
            st.warning("Sleep Hours + Study Hours cannot exceed 24 hours per day.")
    else:
        st.session_state["input"] = (float(sleep), float(study))
        st.session_state["scroll"] = True

# ------------------------------------------------------------
# Plot helpers
# ------------------------------------------------------------
def style_axes(ax):
    ax.set_facecolor("none")
    for s in ax.spines.values():
        s.set_color(TAUPE)
    ax.tick_params(colors=VANILLA)
    ax.xaxis.label.set_color(VANILLA)
    ax.yaxis.label.set_color(VANILLA)
    ax.title.set_color(VANILLA)
    ax.grid(alpha=0.18, color=TAUPE)


def plot_mf(sets, x_input, mu, title, xlabel):
    colors = [TAUPE, DUNE, "#B7826B"]
    xs = np.linspace(DOMAIN[0], DOMAIN[1], 600)
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    fig.patch.set_alpha(0)
    for (name, p), c in zip(sets.items(), colors):
        ax.plot(xs, [trapmf(x, p) for x in xs], label=name, lw=2.2, color=c)
        ax.plot([x_input], [mu[name]], "o", color=VANILLA, ms=7, zorder=5)
    ax.axvspan(*TRAIN_RANGE, color=TAUPE, alpha=0.10)
    ax.text(sum(TRAIN_RANGE) / 2, 1.07, "training range", ha="center", fontsize=8, color=DUNE)
    ax.axvline(x_input, color=VANILLA, ls="--", lw=1)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Membership degree (µ)")
    ax.set_xlim(*DOMAIN)
    ax.set_ylim(-0.02, 1.15)
    ax.set_xticks(range(0, 25, 4))
    style_axes(ax)
    legend = ax.legend(loc="center right", facecolor=UMBER, edgecolor=TAUPE)
    for t in legend.get_texts():
        t.set_color(VANILLA)
    return fig


def plot_score_bar(z):
    fig, ax = plt.subplots(figsize=(9, 1.5))
    fig.patch.set_alpha(0)
    for x0, width, cat in [(0, T1, "Low"), (T1, T2 - T1, "Moderate"), (T2, 100 - T2, "High")]:
        ax.barh([0], [width], left=x0, color=CATEGORY_COLOR[cat], alpha=0.55)
        ax.text(x0 + width / 2, 0, cat, ha="center", va="center", color=BLACK, fontweight="bold")
    ax.axvline(z, color=VANILLA, lw=3)
    ax.text(z, 0.58, f"z* = {z:.2f}", ha="center", va="bottom", color=VANILLA, fontweight="bold")
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.5, 1.1)
    ax.set_yticks([])
    ax.set_xticks([0, T1, T2, 100])
    ax.set_facecolor("none")
    ax.tick_params(colors=VANILLA)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(TAUPE)
    return fig


def fig_to_png(fig, dpi=110):
    """Render figure ke PNG (bytes) lalu tutup figure supaya memori tidak menumpuk."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, transparent=True, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


@st.cache_data(show_spinner=False)
def get_pngs(sleep_v, study_v):
    """Gambar dibuat sekali per kombinasi input; input yang sama langsung dari cache."""
    r = infer(sleep_v, study_v)
    return (
        fig_to_png(plot_score_bar(r["z"])),
        fig_to_png(plot_mf(MF_SLEEP, sleep_v, r["mu_sleep"], "Sleep Hours", "Hours of sleep per day")),
        fig_to_png(plot_mf(MF_STUDY, study_v, r["mu_study"], "Study Hours", "Hours of study per day")),
    )


# ------------------------------------------------------------
# Result (appears after Submit)
# ------------------------------------------------------------
if "input" in st.session_state:
    sleep_v, study_v = st.session_state["input"]
    result = infer(sleep_v, study_v)
    z, cat = result["z"], result["category"]

    st.markdown('<div id="result"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-title">Your result</div>', unsafe_allow_html=True)
    st.markdown(
        f"""<div class="score-card">
              <div class="lab">Sleep {sleep_v:.1f} h &nbsp;·&nbsp; Study {study_v:.1f} h &nbsp;·&nbsp; stress score (z*)</div>
              <div class="z">{z:.2f}</div>
              <span class="badge" style="background:{CATEGORY_COLOR[cat]}">{cat.upper()}</span>
            </div>""", unsafe_allow_html=True)

    # Scroll langsung di sini, sebelum grafik & tab (bagian berat) dirender
    if st.session_state.pop("scroll", False):
        # nonce unik tiap submit: kalau isi iframe sama persis dengan sebelumnya,
        # Streamlit tidak me-render ulang dan script-nya tidak jalan lagi
        nonce = time.time_ns()
        st.iframe(
            f"<!-- {nonce} --><script>window.parent.document.getElementById('result')"
            ".scrollIntoView({behavior:'smooth'});</script>", height=1)

    png_bar, png_sleep, png_study = get_pngs(sleep_v, study_v)
    st.image(png_bar, width="stretch")

    lo, hi = TRAIN_RANGE
    if not (lo <= sleep_v <= hi and lo <= study_v <= hi):
        st.info(f"Your input is outside the {lo:.0f}–{hi:.0f} h range of the training data. "
                "The lowest and highest fuzzy sets are held at full membership beyond that range, "
                "so treat this result as an extrapolation.")

    tab1, tab2, tab3 = st.tabs(["1. Fuzzification", "2. Active Rules & Result", "3. About the Model"])

    with tab1:
        st.write("Input values are converted into membership degrees (µ). "
                 "Dashed line = input, dot = µ value.")
        g1, g2 = st.columns(2)
        g1.image(png_sleep, width="stretch")
        g2.image(png_study, width="stretch")
        t1, t2 = st.columns(2)
        t1.write("**µ Sleep Hours**")
        t1.dataframe({k: [round(v, 4)] for k, v in result["mu_sleep"].items()}, hide_index=True)
        t2.write("**µ Study Hours**")
        t2.dataframe({k: [round(v, 4)] for k, v in result["mu_study"].items()}, hide_index=True)

    with tab2:
        st.write("Each rule's firing strength is **w = min(µ Study, µ Sleep)**. "
                 "Highlighted rows are the active rules (w > 0).")

        def highlight(row):
            color = "background-color:#6b5a4e" if row["w (firing)"] > 0 else ""
            return [color] * len(row)

        st.dataframe(
            result["table"].style.apply(highlight, axis=1).format(
                {"µ Study": "{:.4f}", "µ Sleep": "{:.4f}", "w (firing)": "{:.4f}",
                 "z_i": "{:.0f}", "w × z_i": "{:.4f}"}),
            hide_index=True, width="stretch")

        st.subheader("Weighted Average")
        st.latex(r"z^* = \frac{\sum w_i z_i}{\sum w_i} = "
                 rf"\frac{{{result['numerator']:.4f}}}{{{result['denominator']:.4f}}} = {z:.4f}")
        limit = "≤ 32.5" if cat == "Low" else "≤ 67.5" if cat == "Moderate" else "> 67.5"
        st.write(f"Since z* = **{z:.2f}** ({limit}), the stress category is **{cat}**.")

    with tab3:
        st.markdown(
            "- **Model**: Zero-order Sugeno fuzzy, AND = MIN, output = weighted average.\n"
            "- **Fuzzy sets**: trapezoidal, 3 sets per input, optimized with Differential Evolution.\n"
            f"- **Consequents**: Low = {CONSEQUENT['Low']}, Moderate = {CONSEQUENT['Moderate']}, "
            f"High = {CONSEQUENT['High']}.\n"
            f"- **Classification**: Low ≤ {T1} | Moderate ≤ {T2} | High > {T2}.\n"
            "- **Performance (test data, 600 rows)**: accuracy 0.9867 and macro-F1 0.9837."
        )
        st.write("**Rule base**")
        st.dataframe(
            {"Rule": [f"R{n}" for n, *_ in RULE_BASE],
             "Study": [r[1] for r in RULE_BASE],
             "Sleep": [r[2] for r in RULE_BASE],
             "Output": [r[3] for r in RULE_BASE]},
            hide_index=True, width="stretch")