import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ─── PAGE CONFIG & DARK THEME CSS ─────────────────────────────────────────────
st.set_page_config(
    page_title="🎬 Movie Recommender",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
      /* Full‑page dark background */
      .reportview-container, .main, .block-container {
          background-color: #121212;
          color: #FFFFFF;
      }

      /* Dropdown */
      .stSelectbox > div[data-baseweb="select"] > div {
          background-color: #1f1f1f;
          color: #FFFFFF;
          border: 1px solid #333333;
      }
      .stSelectbox > div[data-baseweb="select"] input {
          background-color: #1f1f1f;
          color: #FFFFFF;
      }

      /* Button */
      .stButton > button {
          background-color: #1f1f1f;
          color: #FFFFFF;
          border: 1px solid #BB86FC;
          border-radius: 4px;
          padding: 0.5em 1em;
      }
      .stButton > button:hover {
          background-color: #BB86FC;
          color: #121212;
      }

      /* Text input / search field */
      .stTextInput > div > input {
          background-color: #1f1f1f;
          color: #FFFFFF;
          border: 1px solid #333333;
      }

      /* Headings */
      h1, .css-1d391kg {
          color: #FFFFFF;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─── LOAD & PREPARE DATA ────────────────────────────────────────────────────────
df = pd.read_csv("movies.csv")
for col in ["genres","keywords","overview","cast","director"]:
    df[col] = df[col].fillna("")
df["combined"] = (
    df["genres"] + " "
  + df["keywords"] + " "
  + df["overview"] + " "
  + df["cast"] + " "
  + df["director"]
)

tfidf       = TfidfVectorizer(stop_words="english")
tfidf_matrix= tfidf.fit_transform(df["combined"])
cosine_sim  = cosine_similarity(tfidf_matrix, tfidf_matrix)
indices     = pd.Series(df.index, index=df["title"]).drop_duplicates()

# ─── UI ─────────────────────────────────────────────────────────────────────────
st.title("🎬 Movie Recommender")

movie = st.selectbox(
    "Enter a movie title:",
    options=df["title"].tolist(),
)

def recommend_movies(title, num=5):
    idx = indices.get(title)
    if idx is None:
        return ["❌ Movie not found."]
    sims = list(enumerate(cosine_sim[idx]))
    sims = sorted(sims, key=lambda x: x[1], reverse=True)[1 : num+1]
    return df["title"].iloc[[i[0] for i in sims]].tolist()

if st.button("Recommend"):
    recs = recommend_movies(movie)
    if recs and recs[0].startswith("❌"):
        st.error(recs[0])
    else:
        st.success("Here’s what we think you’ll like:")
        for m in recs:
            st.write("• " + m)
