import os
import re
import pickle
import sqlite3
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import streamlit as st
from groq import Groq
from sentence_transformers import SentenceTransformer

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title='PetDB Query',
    page_icon='🌋',
    layout='wide',
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown('''
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&family=DM+Mono&display=swap');

html, body, [class*="css"] {
    font-family: "DM Sans", sans-serif;
}

.main { background: #f7f6f2; }

h1, h2, h3 { font-family: "DM Serif Display", serif; }

.stTextArea textarea {
    font-family: "DM Sans", sans-serif !important;
    font-size: 15px !important;
    border: 2px solid #0E2841 !important;
    border-radius: 8px !important;
    background: #fff !important;
    padding: 14px !important;
}

.stButton > button {
    background: #0E2841 !important;
    color: #fff !important;
    font-family: "DM Sans", sans-serif !important;
    font-weight: 600 !important;
    font-size: 15px !important;
    border-radius: 8px !important;
    border: none !important;
    padding: 0.6rem 2rem !important;
    transition: background 0.2s ease;
}
.stButton > button:hover {
    background: #156082 !important;
}

.sql-block {
    background: #0E2841;
    color: #e8e8e8;
    font-family: "DM Mono", monospace;
    font-size: 13px;
    padding: 16px 20px;
    border-radius: 8px;
    white-space: pre-wrap;
    margin-bottom: 1rem;
}

.summary-block {
    background: #fff;
    border-left: 4px solid #E97132;
    padding: 16px 20px;
    border-radius: 0 8px 8px 0;
    font-size: 15px;
    color: #1a1a1a;
    margin-bottom: 1rem;
}

.fallback-block {
    background: #fff3f0;
    border-left: 4px solid #c0392b;
    padding: 14px 18px;
    border-radius: 0 8px 8px 0;
    font-size: 14px;
    color: #7b2020;
}

.stat-row {
    display: flex;
    gap: 1rem;
    margin-bottom: 1.2rem;
}
.stat-card {
    background: #0E2841;
    color: #fff;
    border-radius: 8px;
    padding: 14px 20px;
    min-width: 130px;
    text-align: center;
}
.stat-card .val {
    font-size: 26px;
    font-weight: 700;
    color: #E97132;
    font-family: "DM Serif Display", serif;
}
.stat-card .lbl {
    font-size: 11px;
    color: #8A9BB0;
    margin-top: 2px;
}

.header-band {
    background: #0E2841;
    color: #fff;
    padding: 2rem 2.5rem 1.5rem;
    border-radius: 12px;
    margin-bottom: 2rem;
}
.header-band h1 {
    margin: 0 0 0.3rem 0;
    font-size: 2.2rem;
    color: #fff;
}
.header-band p {
    margin: 0;
    color: #8A9BB0;
    font-size: 15px;
}
.orange-line {
    height: 4px;
    background: #E97132;
    border-radius: 2px;
    margin: 0.8rem 0 0;
    width: 60px;
}
</style>
''', unsafe_allow_html=True)

# ── Paths ─────────────────────────────────────────────────────────────────────
DB_PATH        = 'petdb_backarc.sqlite'
SCHEMA_PATH    = 'petdb_schema.txt'
CACHE_DIR      = Path('cache')
FAISS_PATH     = CACHE_DIR / 'rag_index.faiss'
CHUNKS_PATH    = CACHE_DIR / 'rag_chunks.pkl'
EMBED_MODEL_ID = 'sentence-transformers/all-MiniLM-L6-v2'
GROQ_MODEL_ID  = 'qwen/qwen3-32b'
TOP_K          = 3
CACHE_DIR.mkdir(exist_ok=True)

VALID_COLUMNS = {
    'row_id', 'sample_name', 'citation', 'latitude', 'longitude',
    'analyzed_material', 'tectonic_setting', 'basin_name',
    'elevation_m', 'elevation_max_m', 'sample_alteration',
    'rock_texture', 'geologic_age', 'station_site', 'expedition',
    'sio2', 'tio2', 'al2o3', 'feot', 'mgo', 'mno', 'cao', 'na2o', 'k2o', 'p2o5',
    'rb', 'sr', 'ba', 'cs', 'y', 'zr', 'hf', 'nb', 'ta', 'th', 'u', 'pb',
    'v', 'cr', 'co', 'ni', 'sc',
    'la', 'ce', 'pr', 'nd', 'sm', 'eu', 'gd', 'dy', 'ho', 'er', 'yb', 'lu',
    'rb87_sr86', 'sr87_sr86', 'sm147_nd144', 'nd143_nd144',
    'e_nd', 'pb206_pb204', 'pb207_pb204', 'pb208_pb204',
}

FALLBACK = 'This question cannot be answered from the PetDB back-arc basin dataset.'

SYSTEM_PROMPT = (
    '/no-think\n'
    'You are a geochemist and SQLite expert. '
    'Given a database schema, geochemical context, and a natural language question, '
    'you generate a single valid SQLite query that answers the question. '
    'Rules:\n'
    '- Return ONLY the SQL query, no explanation, no markdown, no code fences.\n'
    '- Always filter out NULL values with IS NOT NULL when operating on geochemical columns.\n'
    '- Use ROUND() for float outputs: 2 decimal places for major oxides and trace elements, '
    '4 decimal places for isotope ratios (sr87_sr86, nd143_nd144, pb206_pb204, pb207_pb204, pb208_pb204).\n'
    '- Do not use sample_name as a unique identifier. Use row_id if uniqueness is needed.\n'
    '- Computed ratios must guard against division by zero with a WHERE clause.\n'
    '- Only query columns that exist in the schema. Never invent proxies for real-world '
    'data like prices, dates, or external measurements not present in the database.\n'
)

SUMMARY_SYSTEM = (
    '/no-think\n'
    'You are a geochemist interpreting query results from a PetDB back-arc basin basalt database. '
    'Write a concise 2-3 sentence interpretation of the data returned. '
    'Be specific about values, patterns, and what they mean geochemically. '
    'Do not mention SQL, databases, or technical details. Write for a geoscience audience.'
)

# ── Cached resources ──────────────────────────────────────────────────────────
@st.cache_resource(show_spinner='Loading embedding model...')
def load_embedder():
    return SentenceTransformer(EMBED_MODEL_ID)


@st.cache_resource(show_spinner='Loading RAG index...')
def load_rag(_embedder):
    if not FAISS_PATH.exists() or not CHUNKS_PATH.exists():
        st.error('RAG cache not found. Please add cache/rag_index.faiss and cache/rag_chunks.pkl to the repo.')
        st.stop()
    idx = faiss.read_index(str(FAISS_PATH))
    with open(CHUNKS_PATH, 'rb') as f:
        data = pickle.load(f)
    return idx, data['chunks'], data['sources']


@st.cache_data
def load_schema():
    with open(SCHEMA_PATH, 'r') as f:
        return f.read()


@st.cache_data
def get_db_stats():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute('SELECT COUNT(*) FROM samples')
    rows = cur.fetchone()[0]
    cur.execute('SELECT COUNT(DISTINCT basin_name) FROM samples')
    basins = cur.fetchone()[0]
    conn.close()
    return rows, basins


# ── Core functions ────────────────────────────────────────────────────────────
def retrieve(query, idx, all_chunks, chunk_sources, k=TOP_K):
    embedder = load_embedder()
    q_emb = embedder.encode(
        [query], normalize_embeddings=True, convert_to_numpy=True
    ).astype(np.float32)
    scores, indices = idx.search(q_emb, k)
    return [
        {'chunk': all_chunks[i], 'source': chunk_sources[i], 'score': float(s)}
        for s, i in zip(scores[0], indices[0])
    ]


def build_prompt(question, schema, idx, all_chunks, chunk_sources):
    chunks = retrieve(question, idx, all_chunks, chunk_sources)
    geo_context = '\n\n'.join(
        f'[{c["source"]}]\n{c["chunk"]}' for c in chunks
    )
    user_content = (
        f'[SCHEMA]\n{schema}\n\n'
        f'[GEOCHEMICAL CONTEXT]\n{geo_context}\n\n'
        f'[QUESTION]\n{question}\n\n'
        f'[SQL]'
    )
    return [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user',   'content': user_content},
    ]


def call_groq(client, messages, max_tokens, temperature):
    kwargs = dict(model=GROQ_MODEL_ID, messages=messages, max_tokens=max_tokens)
    if temperature > 0:
        kwargs['temperature'] = temperature
    response = client.chat.completions.create(**kwargs)
    raw = response.choices[0].message.content.strip()
    raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()
    return raw


def generate_sql(question, client, schema, idx, all_chunks, chunk_sources):
    messages = build_prompt(question, schema, idx, all_chunks, chunk_sources)
    raw = call_groq(client, messages, max_tokens=512, temperature=0.1)
    sql = re.sub(r'^```[\w]*\n?', '', raw)
    sql = re.sub(r'\n?```$', '', sql)
    return sql.strip()


def validate_sql(sql):
    sql_lower = sql.lower()
    if 'from samples' not in sql_lower:
        return False, 'SQL does not query the samples table.'

    sql_keywords = {
        'select', 'from', 'where', 'group', 'by', 'order', 'having',
        'limit', 'and', 'or', 'not', 'null', 'is', 'in', 'as', 'on',
        'join', 'left', 'right', 'inner', 'count', 'avg', 'sum', 'min',
        'max', 'round', 'distinct', 'case', 'when', 'then', 'else', 'end',
        'samples', 'asc', 'desc', 'between', 'like', 'cast', 'coalesce',
        'over', 'partition', 'iif', 'abs', 'length', 'upper', 'lower',
        'sqrt', 'power', 'substr', 'trim',
    }
    sql_no_strings = re.sub(r"'[^']*'", '', sql_lower)
    aliases = set(re.findall(r'\bas\s+([a-z_][a-z0-9_]*)', sql_no_strings))
    tokens  = re.findall(r'[a-z_][a-z0-9_]*', sql_no_strings)
    unknown = [
        t for t in tokens
        if t not in sql_keywords
        and t not in VALID_COLUMNS
        and t not in aliases
        and not t.replace('.', '').isnumeric()
    ]
    if unknown:
        return False, f'SQL references unknown columns: {set(unknown)}'

    conn = sqlite3.connect(DB_PATH)
    try:
        dry = re.sub(r'\blimit\s+\d+', '', sql_lower, flags=re.IGNORECASE).rstrip('; ')
        conn.execute(dry + ' LIMIT 0')
    except Exception as e:
        conn.close()
        return False, f'SQL syntax error: {e}'
    conn.close()
    return True, None


def run_sql(sql):
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql_query(sql, conn)
    except Exception as e:
        conn.close()
        raise RuntimeError(f'SQL execution failed: {e}') from e
    conn.close()
    return df


def generate_summary(df, question, client):
    preview = df.head(20).to_string(index=False)
    messages = [
        {'role': 'system', 'content': SUMMARY_SYSTEM},
        {'role': 'user', 'content': (
            f'Question asked: {question}\n\n'
            f'Query returned {len(df)} rows. Here is a preview:\n\n{preview}\n\n'
            f'Write a 2-3 sentence geochemical interpretation.'
        )},
    ]
    return call_groq(client, messages, max_tokens=200, temperature=0.3)


# ── Column catalogue ──────────────────────────────────────────────────────────
COLUMN_CATALOGUE = [
    # Identifiers and metadata
    ('row_id',            '--',   'Primary key'),
    ('sample_name',       '--',   'Sample identifier (not unique across studies)'),
    ('citation',          '--',   'Bibliographic citation'),
    ('latitude',          'deg',  'Decimal degrees, negative = South'),
    ('longitude',         'deg',  'Decimal degrees, negative = West'),
    ('analyzed_material', '--',   'WHOLE ROCK, GLASS, or UNSPECIFIED'),
    ('basin_name',        '--',   'Back-arc basin name'),
    ('tectonic_setting',  '--',   'Tectonic setting label'),
    ('elevation_m',       'm',    'Min depth/elevation, negative = submarine'),
    ('elevation_max_m',   'm',    'Max depth/elevation'),
    ('sample_alteration', '--',   'Alteration state'),
    ('rock_texture',      '--',   'Rock texture description'),
    ('geologic_age',      '--',   'Geologic age label'),
    ('station_site',      '--',   'Station or site identifier'),
    ('expedition',        '--',   'Expedition or cruise identifier'),
    # Major oxides
    ('sio2',  'wt%', 'Silicon dioxide'),
    ('tio2',  'wt%', 'Titanium dioxide'),
    ('al2o3', 'wt%', 'Aluminum oxide'),
    ('feot',  'wt%', 'Total iron as FeO'),
    ('mgo',   'wt%', 'Magnesium oxide'),
    ('mno',   'wt%', 'Manganese oxide'),
    ('cao',   'wt%', 'Calcium oxide'),
    ('na2o',  'wt%', 'Sodium oxide'),
    ('k2o',   'wt%', 'Potassium oxide'),
    ('p2o5',  'wt%', 'Phosphorus pentoxide'),
    # Trace elements
    ('rb', 'ppm', 'Rubidium'),
    ('sr', 'ppm', 'Strontium'),
    ('ba', 'ppm', 'Barium'),
    ('cs', 'ppm', 'Cesium'),
    ('y',  'ppm', 'Yttrium'),
    ('zr', 'ppm', 'Zirconium'),
    ('hf', 'ppm', 'Hafnium'),
    ('nb', 'ppm', 'Niobium'),
    ('ta', 'ppm', 'Tantalum'),
    ('th', 'ppm', 'Thorium'),
    ('u',  'ppm', 'Uranium'),
    ('pb', 'ppm', 'Lead'),
    ('v',  'ppm', 'Vanadium'),
    ('cr', 'ppm', 'Chromium'),
    ('co', 'ppm', 'Cobalt'),
    ('ni', 'ppm', 'Nickel'),
    ('sc', 'ppm', 'Scandium'),
    # REE
    ('la', 'ppm', 'Lanthanum'),
    ('ce', 'ppm', 'Cerium'),
    ('pr', 'ppm', 'Praseodymium'),
    ('nd', 'ppm', 'Neodymium'),
    ('sm', 'ppm', 'Samarium'),
    ('eu', 'ppm', 'Europium'),
    ('gd', 'ppm', 'Gadolinium'),
    ('dy', 'ppm', 'Dysprosium'),
    ('ho', 'ppm', 'Holmium'),
    ('er', 'ppm', 'Erbium'),
    ('yb', 'ppm', 'Ytterbium'),
    ('lu', 'ppm', 'Lutetium'),
    # Isotope ratios
    ('rb87_sr86',   '--', '87Rb/86Sr ratio'),
    ('sr87_sr86',   '--', '87Sr/86Sr ratio'),
    ('sm147_nd144', '--', '147Sm/144Nd ratio'),
    ('nd143_nd144', '--', '143Nd/144Nd ratio'),
    ('e_nd',        '--', 'Epsilon-Nd'),
    ('pb206_pb204', '--', '206Pb/204Pb ratio'),
    ('pb207_pb204', '--', '207Pb/204Pb ratio'),
    ('pb208_pb204', '--', '208Pb/204Pb ratio'),
]


# ── App ───────────────────────────────────────────────────────────────────────
def main():
    # Session state init
    for key in ['sql', 'df', 'summary', 'error', 'fallback']:
        if key not in st.session_state:
            st.session_state[key] = None

    # Header
    st.markdown('''
    <div class="header-band">
        <h1>🌋 PetDB Query</h1>
        <p>Natural language querying of back-arc basin geochemical data</p>
        <div class="orange-line"></div>
    </div>
    ''', unsafe_allow_html=True)

    # Groq client
    try:
        groq_key = st.secrets['GROQ_API_KEY']
    except Exception:
        groq_key = os.environ.get('GROQ_API_KEY', '')
    if not groq_key:
        st.error('GROQ_API_KEY not found. Add it to .streamlit/secrets.toml')
        st.stop()
    client = Groq(api_key=groq_key)

    # Load resources
    embedder             = load_embedder()
    idx, chunks, sources = load_rag(embedder)
    schema               = load_schema()
    rows, basins         = get_db_stats()

    # Stats row
    st.markdown(f'''
    <div class="stat-row">
        <div class="stat-card"><div class="val">10,581</div><div class="lbl">samples</div></div>
        <div class="stat-card"><div class="val">{basins}</div><div class="lbl">basins</div></div>
        <div class="stat-card"><div class="val">62</div><div class="lbl">retained columns</div></div>
        <div class="stat-card"><div class="val">1</div><div class="lbl">RAG document</div></div>
    </div>
    ''', unsafe_allow_html=True)

    # Column catalogue expander
    with st.expander(f'View all 62 retained columns', expanded=False):
        cat_df = pd.DataFrame(COLUMN_CATALOGUE, columns=['Column', 'Unit', 'Description'])
        st.dataframe(cat_df, use_container_width=True, hide_index=True)

    st.markdown('---')

    # Query input
    st.markdown('#### Ask a question')
    question = st.text_area(
        label='question',
        label_visibility='collapsed',
        placeholder='e.g. What is the average MgO for each basin? Which samples have Nb/U above 40?',
        height=100,
    )

    run = st.button('Run Query', use_container_width=False)

    if run:
        question = question.strip()
        if not question or question.upper() == 'YOUR QUESTION HERE':
            st.warning('Please enter a question.')
            st.stop()
        if len(question) < 10:
            st.session_state['fallback'] = FALLBACK
            st.session_state['sql']     = None
            st.session_state['df']      = None
            st.session_state['summary'] = None
            st.session_state['error']   = None
        else:
            with st.spinner('Generating SQL...'):
                sql = generate_sql(question, client, schema, idx, chunks, sources)

            valid, reason = validate_sql(sql)
            if not valid:
                st.session_state['fallback'] = f'{FALLBACK}<br><br><small>{reason}</small>'
                st.session_state['sql']      = sql
                st.session_state['df']       = None
                st.session_state['summary']  = None
                st.session_state['error']    = None
            else:
                with st.spinner('Running query...'):
                    try:
                        df = run_sql(sql)
                        st.session_state['error'] = None
                    except RuntimeError as e:
                        st.session_state['error']   = str(e)
                        st.session_state['df']      = None
                        st.session_state['summary'] = None
                        st.session_state['sql']     = sql
                        st.session_state['fallback']= None
                        df = None

                if df is not None:
                    if df.empty:
                        st.session_state['df']      = df
                        st.session_state['summary'] = None
                    else:
                        with st.spinner('Interpreting results...'):
                            summary = generate_summary(df, question, client)
                        st.session_state['df']      = df
                        st.session_state['summary'] = summary
                    st.session_state['sql']      = sql
                    st.session_state['fallback'] = None

    # ── Render stored results ─────────────────────────────────────────────────
    if st.session_state['fallback']:
        st.markdown(
            f'<div class="fallback-block">{st.session_state["fallback"]}</div>',
            unsafe_allow_html=True
        )
        if st.session_state['sql']:
            with st.expander('Generated SQL (debugging)', expanded=False):
                st.markdown(
                    f'<div class="sql-block">{st.session_state["sql"]}</div>',
                    unsafe_allow_html=True
                )

    elif st.session_state['error']:
        st.error(st.session_state['error'])

    elif st.session_state['df'] is not None:
        df = st.session_state['df']

        # SQL expander
        if st.session_state['sql']:
            with st.expander('Generated SQL', expanded=False):
                st.markdown(
                    f'<div class="sql-block">{st.session_state["sql"]}</div>',
                    unsafe_allow_html=True
                )

        if df.empty:
            st.info('Query returned no results.')
        else:
            # Summary
            if st.session_state['summary']:
                st.markdown(
                    f'<div class="summary-block">{st.session_state["summary"]}</div>',
                    unsafe_allow_html=True
                )

            # Table
            st.markdown(f'**{len(df):,} rows returned**')
            st.dataframe(df, use_container_width=True, hide_index=True)

            # Download -- key prevents rerun reset
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label='Download CSV',
                data=csv,
                file_name='petdb_results.csv',
                mime='text/csv',
                key='download_csv',
            )


if __name__ == '__main__':
    main()
