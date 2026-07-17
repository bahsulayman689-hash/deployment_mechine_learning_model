import streamlit as st
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


#---------------------------------------------------------
#search engine suggestion for movies remmandation systems
#---------------------------------------------------------
st.markdown(
"""
<h2 style='text-align:center'>
🔍 Find Your Next Movie
</h2>
""",
unsafe_allow_html=True
)


movie_name = st.text_input(
    "",
    placeholder="Search movies like Avatar, Titanic, Inception..."
)
if "history" not in st.session_state:
    st.session_state.history = []


if movie_name:
    st.session_state.history.append(movie_name)
st.sidebar.write("Recent Searches")

for item in st.session_state.history[-5:]:
    st.sidebar.write("🎬", item)
#---------------------------------------------------

#---------------------------------------------------
# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="🎬 Movie Recommendation System",
    page_icon="🎥",
    layout="wide"
)

# --------------------------------------------------
# Load Dataset
# --------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("movies (1).csv")

movies_data = load_data()

# --------------------------------------------------
# Fill Missing Values
# --------------------------------------------------
selected_features = [
    "genres",
    "keywords",
    "tagline",
    "cast",
    "director"
]

for feature in selected_features:
    movies_data[feature] = movies_data[feature].fillna("")

# --------------------------------------------------
# Combine Features
# --------------------------------------------------
combined_features = (
    movies_data["genres"] + " " +
    movies_data["keywords"] + " " +
    movies_data["tagline"] + " " +
    movies_data["cast"] + " " +
    movies_data["director"]
)

# --------------------------------------------------
# TF-IDF
# --------------------------------------------------
@st.cache_resource
def create_similarity():
    vectorizer = TfidfVectorizer()
    feature_vectors = vectorizer.fit_transform(combined_features)
    similarity = cosine_similarity(feature_vectors)
    return similarity

similarity = create_similarity()

# --------------------------------------------------
# Recommendation Function
# --------------------------------------------------
def recommend_movies(movie_name):

    movie_titles = movies_data["title"].tolist()

    close_matches = difflib.get_close_matches(
        movie_name,
        movie_titles,
        n=1,
        cutoff=0.5
    )

    if len(close_matches) == 0:
        return None

    close_match = close_matches[0]

    movie_index = movies_data[
        movies_data.title == close_match
    ].index[0]

    similarity_score = list(enumerate(similarity[movie_index]))

    sorted_movies = sorted(
        similarity_score,
        key=lambda x: x[1],
        reverse=True
    )

    recommendations = []

    for movie in sorted_movies[1:11]:

        index = movie[0]

        title = movies_data.iloc[index]["title"]

        genre = movies_data.iloc[index]["genres"]

        vote = movies_data.iloc[index]["vote_average"]


        

        

        recommendations.append(
            {
                "Title": title,
                "Genre": genre,
                "Rating": vote,
                
            }
        )

    return close_match, recommendations


# --------------------------------------------------
# Title
# --------------------------------------------------
st.title("🎬 Movie Recommendation System")

st.write(
    """
Select or type your favourite movie, and the system
will recommend similar movies using **TF-IDF** and
**Cosine Similarity**.
"""
)

# --------------------------------------------------
# Input
# --------------------------------------------------
movie_name = st.text_input(
    "Enter Movie Name",
    placeholder="Example: Avatar"
)

# --------------------------------------------------
# Button
# --------------------------------------------------
if st.button("🎥 Recommend Movies"):

    if movie_name.strip() == "":
        st.warning("Please enter a movie name.")

    else:

        result = recommend_movies(movie_name)

        if result is None:
            st.error("Movie not found.")
        else:

            matched_movie, recommendations = result

        

            st.success(f"You searched for: **{matched_movie}**")

            st.subheader("Recommended Movies")

            for i, movie in enumerate(recommendations, start=1):

                with st.container(border=True):

                    st.markdown(f"### {i}. {movie['Title']}")

                    st.write(f"**Genre:** {movie['Genre']}")

                    st.write(f"⭐ Rating: {movie['Rating']}")
        
#----------------------------------------------------------------------
#poster
#--------------------------------------------------------------------------
st.balloons()

# --------------------------------------------------
# Sidebar
# --------------------------------------------------
with st.sidebar:

    st.header("About")

    st.write(
        """
This recommendation system uses:

- TF-IDF Vectorizer
- Cosine Similarity
- Content-Based Filtering

Dataset:
Movie Metadata
"""
    )

    st.divider()

    st.write("Developed by Sulayman Bah, mechine learning enginner")
    st.write("the master mind of mechine learning and deep learning ")
import streamlit as st

with st.sidebar:

    st.title("🎬 Movie AI")

    st.markdown(
        """
        ### About

        This application recommends movies
        using:

        🤖 **Machine Learning**
        
        - TF-IDF Vectorizer
        - Cosine Similarity
        - Content-Based Filtering
        """
    )

    st.divider()

    st.subheader("✨ Features")

    st.write(
        """
        ✅ Movie recommendations  
        ✅ Search suggestions  
        ✅ Movie posters  
        ✅ Release dates  
        ✅ Ratings  
        """
    )

    st.divider()

    st.subheader("🛠 Technologies")

    st.write(
        """
        🐍 Python  
        📊 Pandas  
        🤖 Scikit-learn  
        🎨 Streamlit  
        🎥 TMDb API  
        """
    )

    st.divider()

    st.subheader("👨‍💻 Developer")

    st.write(
        """
        Built by **Sulayman Bah**

        Machine Learning Developer
        """
    )

    st.caption("© 2026 Movie AI")
page = st.sidebar.radio(
    "Navigate",
    [
        "🏠 Home",
        "🎬 Recommendations",
        "ℹ️ About"
    ]
)
if page == "🏠 Home":
    st.title("Welcome")

elif page == "🎬 Recommendations":
    st.title("Find Your Movie")

else:
    st.title("About")
st.markdown(
"""
<style>

[data-testid="stSidebar"] {
    background-color: #111827;
}

[data-testid="stSidebar"] h1 {
    color: white;
}

</style>
""",
unsafe_allow_html=True
)