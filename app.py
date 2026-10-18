"""
Movie recommender frontend (Streamlit).

Put these next to this file (or set MOVIE_DATA_DIR):
    movies_data.joblib, tfidf_matrix.joblib, movies_metadata.csv
Run:  streamlit run app.py
"""
import ast
import html
import os
import random
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

DATA_DIR = Path(os.environ.get("MOVIE_DATA_DIR", Path(__file__).parent))
POSTER_HOSTS = [
    "https://image.tmdb.org/t/p/w342",
    "https://media.themoviedb.org/t/p/w342",
    # Public image proxy: fetches the TMDB image for you, helps when TMDB itself is blocked
    "https://wsrv.nl/?url=image.tmdb.org/t/p/w342",
]

st.set_page_config(page_title="CineMind-AI", page_icon="🎬", layout="wide")

# ----------------------------------------------------------------------------
# Styling
# ----------------------------------------------------------------------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@400;500;700&display=swap');

:root { --bg:#141414; --surface:#1f1f1f; --line:#333; --text:#fff; --muted:#b3b3b3;
        --red:#e50914; --green:#46d369; --star:#f5c518; }
.stApp { background: var(--bg); color: var(--text); font-family: 'Inter', sans-serif; }
[data-testid="stSidebar"] { background: #0a0a0a; border-right: 1px solid #222; }
[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer { visibility: hidden; }
.block-container { padding-top: 2rem; max-width: 1400px; }
.stButton > button { background: var(--red); color: #fff; border: 0; font-weight: 700; border-radius: 4px; }
.stButton > button:hover { background: #f6121d; color: #fff; }
h2, h3, .display { font-family: 'Bebas Neue', sans-serif !important; letter-spacing: .03em; font-weight: 400 !important; }

.brand { font-family: 'Bebas Neue', sans-serif; color: var(--red); font-size: 2.6rem; letter-spacing: .05em; line-height: 1; margin-bottom: 1rem; }

/* hero with blurred poster backdrop */
.hero { position: relative; overflow: hidden; border-radius: 8px; margin-bottom: 2.2rem; background: linear-gradient(120deg, #2b0a0c, #141414); }
.hero .bd { position: absolute; inset: 0; width: 100%; height: 100%; object-fit: cover;
            filter: blur(38px) brightness(.45); transform: scale(1.4); }
.hero::after { content: ""; position: absolute; inset: 0; background: linear-gradient(90deg, rgba(20,20,20,.85), rgba(20,20,20,.1)), linear-gradient(0deg, #141414, transparent 40%); }
.hero-in { position: relative; z-index: 1; display: flex; gap: 32px; padding: 32px; }
.hero .poster { width: 210px; min-width: 210px; aspect-ratio: 2/3; border-radius: 6px; box-shadow: 0 12px 40px rgba(0,0,0,.6); }
.hero h2 { font-size: 3.6rem; margin: 0 0 .4rem; line-height: 1; }
.hero .tag { color: var(--muted); font-style: italic; margin: .5rem 0 .8rem; }
.hero .ov { color: #e5e5e5; line-height: 1.6; max-width: 68ch; margin-top: 1rem; }
.meta { display: flex; gap: 14px; align-items: center; color: var(--muted); font-weight: 500; flex-wrap: wrap; }
.meta .star { color: var(--star); font-weight: 700; }

/* poster, with the title showing underneath if the image can't load */
.poster { position: relative; overflow: hidden; background: linear-gradient(160deg, #3a0d10, #1a1a1a);
          display: flex; align-items: center; justify-content: center; text-align: center;
          padding: 12px; color: var(--muted); font-family: 'Bebas Neue', sans-serif; font-size: 1.3rem; }
.poster img { position: absolute; inset: 0; width: 100%; height: 100%; object-fit: cover; }

/* recommendation grid */
.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(185px, 1fr)); gap: 22px 16px; }
.card { background: var(--surface); border-radius: 6px; overflow: hidden; transition: transform .25s, box-shadow .25s; }
.card:hover { transform: scale(1.06); box-shadow: 0 14px 34px rgba(0,0,0,.7); z-index: 2; position: relative; }
.card .poster { aspect-ratio: 2/3; width: 100%; }
.card .ov { position: absolute; inset: 0; z-index: 2; display: flex; align-items: flex-end; padding: 12px;
            background: linear-gradient(0deg, rgba(0,0,0,.95) 15%, rgba(0,0,0,.1)); opacity: 0; transition: opacity .25s;
            font-family: 'Inter', sans-serif; font-size: .78rem; line-height: 1.45; color: #e5e5e5; text-align: left; }
.card:hover .ov { opacity: 1; }
.card .body { padding: 11px 12px 13px; display: flex; flex-direction: column; gap: 6px; }
.card .title { font-weight: 700; line-height: 1.25; font-size: .95rem; }
.row { display: flex; justify-content: space-between; align-items: center; font-size: .82rem; }
.match { color: var(--green); font-weight: 700; }
.rate { color: var(--star); font-weight: 700; }
.yr { color: var(--muted); }
.chips { display: flex; flex-wrap: wrap; gap: 5px; }
.chip { font-size: .7rem; padding: 2px 8px; border-radius: 3px; background: #2f2f2f; color: var(--muted); }
.chip.shared { background: rgba(229,9,20,.18); color: #ff6b73; }
@media (max-width: 700px) { .hero-in { flex-direction: column; } .hero h2 { font-size: 2.6rem; } }
</style>
""",
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def parse_genres(raw):
    try:
        return [g["name"] for g in ast.literal_eval(raw)]
    except Exception:
        return []


def find_file(name):
    """Look next to app.py, then in model_artifacts/ and data/."""
    for d in (DATA_DIR, DATA_DIR / "model_artifacts", DATA_DIR / "data"):
        if (d / name).exists():
            return d / name
    return None


@st.cache_resource(show_spinner="Loading movies…")
def load_assets():
    model_path = find_file("movies_data.joblib")
    tfidf_path = find_file("tfidf_matrix.joblib")
    if model_path is None or tfidf_path is None:
        raise FileNotFoundError(
            "movies_data.joblib / tfidf_matrix.joblib not found. Put them next to app.py "
            "or inside a model_artifacts folder."
        )
    movies = joblib.load(model_path).reset_index(drop=True)
    tfidf = joblib.load(tfidf_path)

    meta_path = find_file("movies_metadata.csv")
    has_meta = meta_path is not None
    if has_meta:
        meta = pd.read_csv(
            meta_path,
            low_memory=False,
            usecols=["title", "poster_path", "release_date", "runtime", "vote_count", "genres", "id"],
        )
        # The model has fewer rows than the CSV (rows dropped while cleaning).
        # Rows kept their original order, so match them back by walking both lists.
        csv_titles = meta["title"].fillna("").tolist()
        positions, j = [], 0
        for t in movies["title"].tolist():
            while j < len(csv_titles) and csv_titles[j] != t:
                j += 1
            if j == len(csv_titles):
                raise RuntimeError(f"Could not match '{t}' to movies_metadata.csv")
            positions.append(j)
            j += 1
        extra = meta.iloc[positions].reset_index(drop=True)
        movies["poster_path"] = extra["poster_path"]
        movies["tmdb_id"] = pd.to_numeric(extra["id"], errors="coerce")
        movies["year"] = pd.to_datetime(extra["release_date"], errors="coerce").dt.year
        movies["runtime"] = pd.to_numeric(extra["runtime"], errors="coerce")
        movies["vote_count"] = pd.to_numeric(extra["vote_count"], errors="coerce").fillna(0)
        movies["genre_list"] = extra["genres"].map(parse_genres)
    else:
        # No CSV: no posters, year, runtime or vote counts.
        movies["poster_path"] = None
        movies["tmdb_id"] = np.nan
        movies["year"] = np.nan
        movies["runtime"] = np.nan
        movies["vote_count"] = 0
        movies["genre_list"] = movies["genres"].fillna("").astype(str).str.split()
    movies.attrs["has_meta"] = has_meta
    movies["popularity"] = pd.to_numeric(movies["popularity"], errors="coerce").fillna(0)

    # Dropdown labels: "Title (year)", made unique when needed
    year_txt = movies["year"].map(lambda y: f" ({int(y)})" if pd.notna(y) else "")
    labels = (movies["title"].astype(str) + year_txt).tolist()
    seen = {}
    for i, lab in enumerate(labels):
        seen[lab] = seen.get(lab, 0) + 1
        if seen[lab] > 1:
            labels[i] = f"{lab} #{seen[lab]}"
    movies["label"] = labels
    return movies, tfidf


def recommend(movies, tfidf, idx, n, min_rating, genres, sort_by):
    sims = cosine_similarity(tfidf[idx], tfidf).ravel()
    sims[idx] = -1
    ok = (movies["vote_average"].values >= min_rating)
    if genres:
        ok &= movies["genre_list"].map(lambda g: bool(set(g) & set(genres))).values
    sims_f = np.where(ok, sims, -1)

    order = np.argsort(-sims_f)[: n * 6]
    seen, picks = set(), []
    for i in order:
        if sims_f[i] <= 0:
            break
        key = (movies.at[i, "title"], movies.at[i, "year"])
        if key in seen or key == (movies.at[idx, "title"], movies.at[idx, "year"]):
            continue
        seen.add(key)
        picks.append(i)
        if len(picks) == n:
            break

    out = movies.loc[picks].copy()
    out["similarity"] = sims[picks]
    if sort_by == "Rating":
        out = out.sort_values("vote_average", ascending=False)
    return out


# ----------------------------------------------------------------------------
# HTML helpers
# ----------------------------------------------------------------------------
@st.cache_data(ttl=120, show_spinner=False)
def pick_poster_host(paths):
    """Return the first poster server that serves any of the sample posters, else None."""
    import requests
    from concurrent.futures import ThreadPoolExecutor

    if not paths:
        return None

    def ok(url):
        try:
            r = requests.get(url, timeout=4, stream=True)
            r.close()
            return r.status_code == 200
        except Exception:
            return False

    for base in POSTER_HOSTS:
        with ThreadPoolExecutor(len(paths)) as ex:
            if any(ex.map(lambda p: ok(base + p), paths)):
                return base
    return None


def flat(s):
    """One line, so Markdown never mistakes indented HTML for a code block."""
    return "".join(line.strip() for line in s.splitlines())


USE_FRESH = False  # True once a TMDB key is set: then only current posters are used
FRESH = {}  # row index -> current poster path from the TMDB API (rebuilt every run)


@st.cache_resource
def poster_cache():
    return {}  # tmdb id -> current poster path ("" = movie has no poster)


def fetch_fresh_paths(rows, key, cache):
    """Ask TMDB for the current poster path of each movie. Returns a list of failure codes."""
    import requests
    from concurrent.futures import ThreadPoolExecutor

    headers = {"Authorization": f"Bearer {key}"} if key.startswith("eyJ") else {}
    params = {} if headers else {"api_key": key}
    todo = sorted({int(r["tmdb_id"]) for _, r in rows if pd.notna(r["tmdb_id"]) and int(r["tmdb_id"]) not in cache})

    def get(i):
        code = 0
        for attempt in range(3):  # retry timeouts, rate limits and server errors
            try:
                r = requests.get(f"https://api.themoviedb.org/3/movie/{i}", params=params, headers=headers, timeout=8)
                code = r.status_code
                if code == 200:
                    return i, (r.json().get("poster_path") or ""), 200
                if code == 404:
                    return i, "", 404
                if code in (401, 403):
                    break
            except Exception:
                code = 0
            time.sleep(0.6 * (attempt + 1))
        return i, None, code

    failures = []
    with ThreadPoolExecutor(8) as ex:
        for i, path, code in ex.map(get, todo):
            if path is None:
                failures.append(code)
            else:
                cache[i] = path
    return failures


def poster_url(row, base):
    p = FRESH.get(row.name, "") if USE_FRESH else row["poster_path"]
    return f"{base}{p}" if base and isinstance(p, str) and p else ""


def poster_html(row, base, overlay=""):
    url = poster_url(row, base)
    img = f'<img src="{url}" alt="" loading="lazy" referrerpolicy="no-referrer">' if url else ""
    return f'<div class="poster"><span>{html.escape(str(row["title"]))}</span>{img}{overlay}</div>'


def chips_html(genres, shared=()):
    return '<div class="chips">' + "".join(
        f'<span class="chip{" shared" if g in shared else ""}">{html.escape(g)}</span>' for g in genres[:4]
    ) + "</div>"


def year_txt(row):
    return f"{int(row['year'])}" if pd.notna(row["year"]) else ""


def hero_html(row, base):
    runtime = f"<span>{int(row['runtime'])} min</span>" if pd.notna(row["runtime"]) and row["runtime"] > 0 else ""
    yr = f"<span>{year_txt(row)}</span>" if year_txt(row) else ""
    tagline = row["tagline"] if isinstance(row["tagline"], str) and row["tagline"] else ""
    tag = f'<div class="tag">{html.escape(tagline)}</div>' if tagline else ""
    url = poster_url(row, base)
    bd = f'<img class="bd" src="{url}" alt="" referrerpolicy="no-referrer">' if url else ""
    return flat(f"""
<div class="hero">{bd}
  <div class="hero-in">
    {poster_html(row, base)}
    <div>
      <h2>{html.escape(str(row["title"]))}</h2>
      <div class="meta">{yr}{runtime}<span class="star">★ {row["vote_average"]:.1f}</span></div>
      {tag}
      {chips_html(row["genre_list"], row["genre_list"])}
      <p class="ov">{html.escape(str(row["overview"]))}</p>
    </div>
  </div>
</div>""")


def card_html(row, base, base_genres):
    shared = set(row["genre_list"]) & set(base_genres)
    plot = str(row["overview"]) if isinstance(row["overview"], str) else ""
    plot = html.escape(plot[:170] + ("…" if len(plot) > 170 else ""))
    yr = year_txt(row)
    return flat(f"""
<div class="card">
  {poster_html(row, base, f'<div class="ov">{plot}</div>')}
  <div class="body">
    <div class="title">{html.escape(str(row["title"]))}</div>
    <div class="row"><span class="match">{row["similarity"] * 100:.0f}% Match</span><span class="rate">★ {row["vote_average"]:.1f}</span></div>
    <div class="row"><span class="yr">{yr}</span></div>
    {chips_html(row["genre_list"], shared)}
  </div>
</div>""")


# ----------------------------------------------------------------------------
# App
# ----------------------------------------------------------------------------
try:
    movies, tfidf = load_assets()
except Exception as e:  # missing files, version mismatch, failed row match
    st.error(f"Could not load the model files from {DATA_DIR}.\n\n{e}")
    st.stop()

labels = movies.sort_values("popularity", ascending=False)["label"].tolist()
label_to_idx = dict(zip(movies["label"], movies.index))
all_genres = sorted({g for gl in movies["genre_list"] for g in gl})

if "pick" not in st.session_state:
    st.session_state["pick"] = labels[0]


def surprise():
    st.session_state["pick"] = random.choice(labels[:5000])


with st.sidebar:
    st.markdown('<div class="brand">CineMind-AI</div>', unsafe_allow_html=True)
    st.selectbox("Pick a movie you like", labels, key="pick")
    st.button("Surprise me", on_click=surprise, use_container_width=True)
    st.divider()
    n = st.slider("How many recommendations", 4, 24, 12, step=4)
    min_rating = st.slider("Minimum rating", 0.0, 9.0, 0.0, step=0.5)
    genres = st.multiselect("Only these genres", all_genres)
    sort_by = st.radio("Sort by", ["Match", "Rating"], horizontal=True)

if not movies.attrs.get("has_meta", True):
    st.sidebar.warning("movies_metadata.csv not found, so posters and years are hidden. "
                       "Copy it into this folder to show them.")

idx = label_to_idx[st.session_state["pick"]]
base = movies.loc[idx]

recs = recommend(movies, tfidf, idx, n, min_rating, genres, sort_by)

poster_base = None
if movies.attrs.get("has_meta", True):
    try:
        default_key = os.environ.get("TMDB_API_KEY") or st.secrets.get("TMDB_API_KEY", "")
    except Exception:
        default_key = os.environ.get("TMDB_API_KEY", "")
    if default_key:
        key = default_key.strip()  # saved key: never sent to the page, so it can't be seen in the UI
    else:
        with st.sidebar.expander("Poster settings", expanded=True):
            st.caption("The saved poster links in the dataset are from 2017 and many are gone. "
                       "A free TMDB API key lets the app fetch current posters. "
                       "To save it permanently, put it in .streamlit/secrets.toml.")
            key = st.text_input("TMDB API key", type="password").strip()

    sample = tuple(movies.sort_values("popularity", ascending=False)["poster_path"].dropna().head(4))
    if key:
        rows = [(idx, base)] + list(recs.iterrows())
        cache = poster_cache()
        USE_FRESH = True
        failures = fetch_fresh_paths(rows, key, cache)
        for i, r in rows:
            if pd.notna(r["tmdb_id"]) and int(r["tmdb_id"]) in cache:
                FRESH[i] = cache[int(r["tmdb_id"])]
        if failures and not FRESH:
            if 401 in failures or 403 in failures:
                st.sidebar.error("TMDB rejected this key. Copy the API Key (v3) or the Read Access Token again.")
            else:
                st.sidebar.error("Couldn't reach api.themoviedb.org from this network.")
        have = sum(1 for p in FRESH.values() if p)
        if FRESH:
            msg = f"Posters found for {have} of {len(rows)} movies on this page."
            if failures:
                msg += f" {len(failures)} lookups failed, so refresh the page to retry."
            elif have < len(rows):
                msg += " TMDB has no poster for the rest."
            st.sidebar.caption(msg)
        sample = tuple(p for p in FRESH.values() if p)[:4] or sample

    poster_base = pick_poster_host(sample)
    if poster_base is None:
        st.sidebar.warning("Couldn't confirm a poster server from this app. If your browser can open "
                           "image.tmdb.org, tick the box below. Otherwise try a hotspot or a VPN.")
        if st.sidebar.checkbox("Try loading posters anyway"):
            poster_base = POSTER_HOSTS[0]

st.markdown(hero_html(base, poster_base), unsafe_allow_html=True)

st.subheader(f"Because you picked {base['title']}")
if recs.empty:
    st.info("No movies match these filters. Lower the minimum rating or clear the genre filter.")
else:
    cards = "".join(card_html(r, poster_base, base["genre_list"]) for _, r in recs.iterrows())
    st.markdown(f'<div class="grid">{cards}</div>', unsafe_allow_html=True)
    st.caption("Red genre tags are shared with your pick. Match % is how close the plot, genres and tagline are. Hover a poster for the plot.")
st.caption("Poster data from TMDB. This product uses the TMDB API but is not endorsed or certified by TMDB.")