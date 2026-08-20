import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import streamlit as st
import warnings
warnings.filterwarnings('ignore')

# ── Page config ───────────────────────────────────────────────────
st.set_page_config(
    page_title="Music Genre Recommender",
    page_icon="🎵",
    layout="centered"
)

# ── Load and train model (cached so it only runs once) ────────────
@st.cache_resource(show_spinner=False)
def load_and_train():
    url = "https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset/resolve/main/dataset.csv"
    df = pd.read_csv(url)

    df.columns = df.columns.str.strip().str.lower()
    df = df.drop_duplicates(subset=['track_id'])
    df = df.dropna(subset=[
        'track_genre', 'danceability', 'energy', 'valence',
        'tempo', 'acousticness', 'instrumentalness',
        'speechiness', 'liveness', 'loudness', 'mode',
        'explicit', 'popularity'
    ])
    df['explicit'] = df['explicit'].astype(int)

    FEATURES = [
        'danceability', 'energy', 'valence', 'tempo',
        'acousticness', 'instrumentalness', 'speechiness',
        'liveness', 'loudness', 'mode', 'explicit', 'popularity'
    ]

    X = df[FEATURES]
    y = df['track_genre']

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        random_state=42,
        n_jobs=-1
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
st.caption("Tell us what kind of music you're in the mood for and we'll recommend the perfect genre — powered by 114,000 real Spotify tracks.")

with st.spinner("Loading 114,000 Spotify tracks and training model... this takes about 30 seconds on first load."):
    model, le, df, FEATURES, accuracy = load_and_train()

st.success(f"Model ready. Trained on {len(df):,} tracks across {df['track_genre'].nunique()} genres. Test accuracy: {accuracy:.1%}")
st.divider()

# ── User inputs ───────────────────────────────────────────────────
st.subheader("🎛️ What are you in the mood for?")
st.caption("Drag the sliders to describe the music you want right now.")

col1, col2 = st.columns(2)

with col1:
    danceability = st.slider(
        "💃 Danceability",
        min_value=0.0, max_value=1.0, value=0.5, step=0.01,
        help="0 = not for dancing, 1 = pure dance floor"
    )
    energy = st.slider(
        "⚡ Energy",
        min_value=0.0, max_value=1.0, value=0.5, step=0.01,
        help="0 = calm and quiet, 1 = intense and loud"
    )
    valence = st.slider(
        "😊 Mood",
        min_value=0.0, max_value=1.0, value=0.5, step=0.01,
        help="0 = sad or dark, 1 = happy and euphoric"
    )
    acousticness = st.slider(
        "🎸 Acoustic vs Electronic",
        min_value=0.0, max_value=1.0, value=0.5, step=0.01,
        help="0 = fully electronic, 1 = fully acoustic"
    )

with col2:
    instrumentalness = st.slider(
        "🎹 Instrumentalness",
        min_value=0.0, max_value=1.0, value=0.1, step=0.01,
        help="0 = lots of vocals, 1 = mostly instrumental"
    )
    tempo = st.slider(
        "🥁 Tempo (BPM)",
        min_value=40, max_value=220, value=120, step=1,
        help="60 = slow ballad, 120 = pop, 160 = fast EDM"
    )
    popularity = st.slider(
        "📈 Popularity",
        min_value=0, max_value=100, value=50, step=1,
        help="0 = underground deep cuts, 100 = mainstream chart-toppers"
    )
    explicit = st.selectbox(
        "🔞 Explicit content",
        options=[0, 1],
        format_func=lambda x: "No explicit content" if x == 0 else "Explicit content okay",
        index=0
    )

# Infer remaining features from user inputs
speechiness = 0.3 if danceability > 0.6 else 0.05
liveness = 0.15
loudness = -5.0 + (energy * 10)
mode = 1 if valence > 0.5 else 0

# ── Predict ───────────────────────────────────────────────────────
st.divider()

if st.button("🎧 Get My Genre Recommendations", use_container_width=True):
    user_input = pd.DataFrame([[
        danceability, energy, valence, float(tempo),
        acousticness, instrumentalness, speechiness,
        liveness, loudness, mode, explicit, float(popularity)
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
