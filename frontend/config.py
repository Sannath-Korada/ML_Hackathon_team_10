"""
Central configuration: constants, defaults and the app's visual theme.

Keeping these in one place means re-skinning the UI, changing the default
backend URL, or tweaking a layer colour never requires touching page logic
in app.py - and the map (map_view.py) and the legend (components.py) always
agree on what each colour means, because they both import it from here.
"""

APP_NAME = "PathSnap"
APP_TAGLINE = "Distributed Map-Matching & Route Reconstruction"

DEFAULT_API_URL = "http://127.0.0.1:8000"
DEFAULT_TIMEOUT = 180   # seconds - model + HMM/Viterbi inference can be slow on a laptop
HEALTH_TIMEOUT = 4      # seconds - a connection test must feel near-instant
MAX_HISTORY = 25        # how many past requests the session activity log keeps

# Layer colours shared by the map and the legend.
COLOR_GPS = "#2f8fe0"
COLOR_ROAD = "#e6a23c"
COLOR_PREDICTED = "#19c37d"
COLOR_CLASSIFIER_ONLY = "#c4524e"
COLOR_REFERENCE = "#b98bf0"

THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
:root{--muted:#91a4bf;--line:rgba(151,174,207,.16)}
html,body,[class*="css"]{font-family:'DM Sans',sans-serif}
.stApp{background:radial-gradient(ellipse at 80% 0%,#12304a 0%,#081321 46%,#081321 100%);color:#eaf2ff}
[data-testid="stHeader"]{background:transparent}
.block-container{max-width:1480px;padding:1.4rem 2rem 3rem}
section[data-testid="stSidebar"]{background:#0b1929;border-right:1px solid var(--line)}
section[data-testid="stSidebar"] *{color:#dce8f8}
section[data-testid="stSidebar"] input{background:#101f33!important;color:#eaf2ff!important}
h1,h2,h3{font-family:'Space Grotesk',sans-serif;letter-spacing:-.03em}
.hero{background:linear-gradient(115deg,#18334f,#104b5b);border:1px solid #39718a;border-radius:22px;padding:26px 30px;margin-bottom:22px;box-shadow:0 18px 55px #0003}
.kicker{color:#8edddc;font-size:.74rem;font-weight:700;letter-spacing:.16em;text-transform:uppercase}
.hero-title{font:700 2.35rem 'Space Grotesk';color:#f5f9ff;margin:5px 0}
.hero-sub{color:#c0d2e7;margin:0}
.pill{display:inline-block;padding:5px 10px;border-radius:99px;border:1px solid #ffffff35;color:#c8e7f5;font-size:.75rem;margin:12px 6px 0 0}
.section{color:#80a5c9;font-size:.75rem;font-weight:700;letter-spacing:.13em;text-transform:uppercase;margin:1.2rem 0 .6rem}
.panel{background:linear-gradient(145deg,#112139,#0e1b2d);border:1px solid var(--line);border-radius:17px;padding:18px}
.kpi{background:linear-gradient(145deg,#142942,#102036);border:1px solid var(--line);border-radius:15px;padding:14px;margin-bottom:12px;min-height:92px}
.kpi-label{color:#91a8c3;font-size:.76rem}
.kpi-value{color:#f3f8ff;font:600 1.35rem 'Space Grotesk';margin-top:5px;word-break:break-word}
.kpi-note{color:#7dd9c5;font-size:.72rem;margin-top:2px}
div.stButton>button[kind="primary"]{background:linear-gradient(100deg,#2878d6,#1b9ca7);border:0;border-radius:11px;color:white;font-weight:700;min-height:44px}
div.stButton>button[kind="secondary"]{background:#14263d;color:#dcecff;border:1px solid #2a4564;border-radius:10px}
div[data-testid="stMetric"]{background:#112139;border:1px solid var(--line);border-radius:13px;padding:12px}
hr{border-color:var(--line)}
footer{visibility:hidden}
.status-chip{display:inline-block;padding:6px 10px;border-radius:9px;font-size:.82rem;margin-top:6px}
.status-ok{background:#0f3324;color:#7fe3b4;border:1px solid #1f6b48}
.status-fail{background:#3a1a1a;color:#f29a9a;border:1px solid #6b2a2a}
.status-unknown{background:#1b2b3d;color:#9fb6cc;border:1px solid #2a4564}
.pipe-step{padding:7px 0;color:#b8c9dc;font-size:.87rem}
.pipe-num{color:#66d5d1;font-weight:700;margin-right:10px}
.legend-row{display:flex;align-items:center;gap:8px;padding:5px 0;color:#dce8f8;font-size:.85rem}
.legend-dot{width:11px;height:11px;border-radius:50%;display:inline-block;flex-shrink:0}
.log-table{width:100%;border-collapse:collapse;font-size:.76rem;margin-bottom:10px}
.log-table th{color:#7f9ab8;text-align:left;font-weight:600;padding:4px 6px;border-bottom:1px solid var(--line)}
.log-table td{padding:4px 6px;color:#dce8f8;border-bottom:1px solid rgba(151,174,207,.08)}
.log-table td.ok{color:#7fe3b4}
.log-table td.fail{color:#f29a9a}
.footer-note{text-align:center;color:#617994;font-size:.74rem;padding:30px 0}
</style>
"""
