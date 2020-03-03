"""
Cluster artists based on the words in their lyrics
Data from kaggle
Format data with create_data_kaggle.py
"""

__author__ = 'don.tuggener@zhaw.ch'

import json
import re

from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer


def plot_dendrogram(clustered, artists):
    """ Plot a dendrogram from the hierarchical clustering of the artist lyrics """
    # plt.figure(figsize=(25, 10))   # for orientation = 'bottom'|'top'
    plt.figure(figsize=(10, 25))
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Distance')  # this' but the label of the whole axis!
    plt.ylabel('Artists')
    plt.tight_layout()
    dendrogram(clustered,
               # leaf_rotation=90.,  # rotates the x axis labels
               leaf_font_size=8.,  # font size for the x axis labels
               labels=artists,
               orientation='left',
               )
    # plt.show() # Instead pf saving
    plt.savefig('example_dendrogram.svg', bbox_inches='tight')


def words_per_artist(artist_lyrics, lyrics_tfidf_matrix, ix2word, n=10):
    """
    For each artist, print the most highly weighted words acc. to TF IDF
    Print n words that are above the mean weight

    :param ix2word: Map mit ID und Wort -> ID an Wort zuweisen
    """
    for i, artist in enumerate(artist_lyrics):
        word_with_tfidf = list(zip(ix2word, lyrics_tfidf_matrix[i]))
        word_with_tfidf_ranked = sorted(word_with_tfidf, key=lambda x: x[1], reverse=True)

        words_per_artist = [word_with_tfidf_ranked[j][0] for j in range(n)]
        print(artist + str(words_per_artist))


def plot_pca(tfidf_matrix):
    pca = PCA(n_components=2)
    X = pca.fit_transform(tfidf_matrix)  # -> "Position" bestimmen in zwei Dimensionen
    plt.figure(figsize=(20, 20))

    artists_by_genre = {}  # Key = Genre, Value = Liste mit allen Artisten + Position
    for genre, x, y, artist in zip(artist2genre.values(), X[:, 0], X[:, 1], artist2genre.keys()):
        if genre not in artists_by_genre:
            artists_by_genre[genre] = []  # Liste erzeugen
        artists_by_genre[genre].append((artist, x, y))

    # Pro Genre -> Alle Artisten einzeln plotten
    for genre, values in artists_by_genre.items():
        artists, xs, ys = list(zip(*values))
        plt.scatter(xs, ys, label=genre)

    plt.legend()
    plt.show()


if __name__ == '__main__':
    print('Loading data')
    artist2genre = json.load(open('data/artist2genre_kaggle.json', 'r', encoding='utf-8'))
    artist_lyrics = json.load(open('data/artist_lyrics_kaggle.json', 'r', encoding='utf-8'))
    # Custom tokenization to remove numbers etc.
    lyrics = [' '.join(re.findall('[A-Za-z]+', l)) for l in artist_lyrics.values()]

    print('Vectorizing with TF IDF')
    # TF IDF -> wie häufig kommt Wort in Lyrics vor
    tfidf_vectorizer = TfidfVectorizer()  # siehe https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
    tfidf_matrix = tfidf_vectorizer.fit_transform(lyrics).toarray()
    ix2word = tfidf_vectorizer.get_feature_names()  # Dict, für Zuweisung ID -> Wort

    plot_pca(tfidf_matrix)

    print('Distinct words per artist')
    words_per_artist(artist_lyrics, tfidf_matrix, ix2word)

    print('Clustering')
    # TODO call SciPy's hierarchical clustering
    kmeans = KMeans(n_clusters=10)
    clustered = kmeans.fit_predict(tfidf_matrix)
    print(clustered)

    print('Plotting')
    artist_names = [a + ': ' + artist2genre[a].upper() for a in list(artist_lyrics.keys())]
    plot_dendrogram(linkage(tfidf_matrix, metric='cosine'), artist_names)
