import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import streamlit as st
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Music Genre Recommender",
    page_icon="🎵",
    layout="centered"
)

@st.cache_resource(show_spinner=False)
def load_and_train():
    url = "https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset/resolve/main/dataset.csv"

    # Only load the columns we actually need — saves significant memory
    usecols = [
        'track_id', 'track_name', 'artists', 'track_genre', 'popularity',
        'danceability', 'energy', 'valence', 'tempo',
        'acousticness', 'instrumentalness', 'speechiness',
        'liveness', 'loudness', 'mode', 'explicit'
    ]

    df = pd.read_csv(url, usecols=usecols, low_memory=False)
    df.columns = df.columns.str.strip().str.lower()
    df = df.drop_duplicates(subset=['track_id'])
    df = df.dropna()

    # Fix types
    df['explicit'] = df['explicit'].astype(str).str.strip().str.lower()
    df['explicit'] = df['explicit'].map(
        {'true': 1, 'false': 0, '1': 1, '0': 0}
    ).fillna(0).astype(np.int8)

    df['mode'] = pd.to_numeric(df['mode'], errors='coerce').fillna(0).astype(np.int8)
    df['popularity'] = pd.to_numeric(df['popularity'], errors='coerce').fillna(0).astype(np.float32)
    df['tempo'] = pd.to_numeric(df['tempo'], errors='coerce').fillna(120.0).astype(np.float32)

    FLOAT_COLS = ['danceability', 'energy', 'valence', 'acousticness',
                  'instrumentalness', 'speechiness', 'liveness', 'loudness']
    for col in FLOAT_COLS:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0).astype(np.float32)

    FEATURES = [
        'danceability', 'energy', 'valence', 'tempo',
        'acousticness', 'instrumentalness', 'speechiness',
        'liveness', 'loudness', 'mode', 'explicit', 'popularity'
    ]

    X = df[FEATURES].astype(np.float32)
    y = df['track_genre'].astype(str)

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Use only 30% of data for training to stay within memory limits
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42
    )

    # Smaller forest to reduce memory
    model = RandomForestClassifier(
        n_estimators=50,
        max_depth=15,
        random_state=42,
        n_jobs=1  # single core to avoid memory spikes
    )
    model.fit(X_train, y_train)

    accuracy = accuracy_score(y_test, model.predict(X_test))

    return model, le, df, FEATURES, accuracy


def get_example_tracks(df, genre, n=3):
    genre_tracks = df[df['track_genre'] == genre].sort_values(
        'popularity', ascending=False
    ).drop_duplicates(subset=['track_name'])
    return genre_tracks[['track_name', 'artists', 'popularity']].head(n)


# ── UI ────────────────────────────────────────────────────────────
st.title("🎵 Music Genre Recommender")
st.caption("Tell us what kind of music you're in the mood for — powered by 114,000 real Spotify tracks.")

try:
    with st.spinner("Loading dataset and training model... about 60 seconds on first load."):
        model, le, df, FEATURES, accuracy = load_and_train()
    st.success(f"Model ready. Trained on {len(df):,} tracks across {df['track_genre'].nunique()} genres.")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

st.divider()

st.subheader("🎛️ What are you in the mood for?")
st.caption("Drag the sliders to describe the music you want right now.")

col1, col2 = st.columns(2)

with col1:
    danceability = st.slider("💃 Danceability", 0.0, 1.0, 0.5, 0.01,
        help="0 = not for dancing, 1 = pure dance floor")
    energy = st.slider("⚡ Energy", 0.0, 1.0, 0.5, 0.01,
        help="0 = calm and quiet, 1 = intense and loud")
    valence = st.slider("😊 Mood", 0.0, 1.0, 0.5, 0.01,
        help="0 = sad or dark, 1 = happy and euphoric")
    acousticness = st.slider("🎸 Acoustic vs Electronic", 0.0, 1.0, 0.5, 0.01,
        help="0 = fully electronic, 1 = fully acoustic")

with col2:
    instrumentalness = st.slider("🎹 Instrumentalness", 0.0, 1.0, 0.1, 0.01,
        help="0 = lots of vocals, 1 = mostly instrumental")
    tempo = st.slider("🥁 Tempo (BPM)", 40, 220, 120, 1,
        help="60 = slow ballad, 120 = pop, 160 = fast EDM")
    popularity = st.slider("📈 Popularity", 0, 100, 50, 1,
        help="0 = underground, 100 = mainstream chart-toppers")
    explicit = st.selectbox("🔞 Explicit content", options=[0, 1],
        format_func=lambda x: "No explicit content" if x == 0 else "Explicit okay",
        index=0)

speechiness = 0.3 if danceability > 0.6 else 0.05
liveness = 0.15
loudness = -5.0 + (energy * 10)
mode = 1 if valence > 0.5 else 0

st.divider()

if st.button("🎧 Get My Genre Recommendations", use_container_width=True):
    user_input = pd.DataFrame([[
        float(danceability), float(energy), float(valence), float(tempo),
        float(acousticness), float(instrumentalness), float(speechiness),
        float(liveness), float(loudness), int(mode), int(explicit), float(popularity)
    ]], columns=FEATURES)

    probabilities = model.predict_proba(user_input)[0]
    top3_indices = np.argsort(probabilities)[::-1][:3]
    top3_genres = le.inverse_transform(top3_indices)
    top3_probs = probabilities[top3_indices]

    st.subheader("🎯 Your Personalized Genre Recommendations")

    medals = ["🥇", "🥈", "🥉"]

    for i, (genre, prob) in enumerate(zip(top3_genres, top3_probs)):
        with st.container():
            st.markdown(f"### {medals[i]} {genre.title()} &nbsp; `{prob:.0%} match`")
            examples = get_example_tracks(df, genre)
            for _, row in examples.iterrows():
                st.write(f"• **{row['track_name']}** — {row['artists']}")
            st.divider()

    st.subheader("📊 Your Taste Profile")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Energy", f"{energy:.0%}")
    c2.metric("Danceability", f"{danceability:.0%}")
    c3.metric("Mood", f"{valence:.0%}")
    c4.metric("Tempo", f"{tempo} BPM")
