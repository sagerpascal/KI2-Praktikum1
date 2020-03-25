"""
Cluster artists based on the words in their lyrics
Data from kaggle
Format data with create_data_kaggle.py
"""

__author__ = 'don.tuggener@zhaw.ch'

import json
import re
from sklearn.datasets import load_breast_cancer
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


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
    plt.savefig('dendrogram.svg', bbox_inches='tight')


def print_distinct_words(artist_lyrics, lyrics_tfidf_matrix, ix2word, n=5):
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


def plot_pca(tfidf_matrix, artist_lyrics):
    n_components = 2
    X_scaled = StandardScaler().fit_transform(tfidf_matrix)
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    plot_cumulative_variance(pca.explained_variance_ratio_, n_components=n_components)

    artists_by_genre = {} # Key = Genre, Value = Liste mit allen Artisten + Position
    for x, y, artist in zip(X_pca[:, 0], X_pca[:, 1], artist_lyrics.keys()):
        genre = artist2genre[artist]
        if genre not in artists_by_genre:
            artists_by_genre[genre] = []  # Liste erzeugen
        artists_by_genre[genre].append((artist, x, y))

    # Pro Genre -> Alle Artisten einzeln plotten
    with plt.style.context('seaborn-whitegrid'):
        plt.figure(figsize=(6, 4))
        for genre, values in artists_by_genre.items():
            artists, xs, ys = list(zip(*values))
            plt.scatter(xs, ys, label=genre)

        plt.xlim(-30,20)
        plt.ylim(-20, 20)
        plt.xlabel('Principal Component {}'.format(1))
        plt.ylabel('Principal Component {}'.format(2))
        plt.tight_layout()
        plt.title('Principlal component analysis')
        plt.legend()
        plt.show()


def plot_cumulative_variance(pca_explained_variance_ratio_, n_components):
    plt.figure(figsize=(5, 5))
    # explained_variance_ratio_ is an array in which we see that chosing X components explains Y %
    # of the variance within our data set
    plt.plot(np.cumsum(pca_explained_variance_ratio_))
    plt.xlabel('Number of Components')
    plt.xlim(0, n_components)
    plt.ylabel('Cumulative Proportion of Variance Explained')
    plt.ylim(0, 1)
    plt.show()


def get_genre_lyrics(artist_lyrics):
    genre_lyrics = {}
    for artist, lyric in artist_lyrics.items():
        genre = artist2genre[artist]
        if genre in genre_lyrics:
            genre_lyrics[genre] = lyric + " " + genre_lyrics[genre]
        else:
            genre_lyrics[genre] = lyric
    return genre_lyrics


def print_clusters(kmeans_labels, feature_name):
    clusters = {}
    for i in range(len(kmeans_labels)):
        name = feature_name[i]
        if kmeans_labels[i] in clusters:
            clusters[kmeans_labels[i]] = name + ", " + clusters[kmeans_labels[i]]
        else:
            clusters[kmeans_labels[i]] = name
    [print (x, y) for x, y in clusters.items()]


def print_clusters_lyrics(ix2word_stop_words, order_centroids, n_clusters):
    clusters = {}
    for i in range(n_clusters):
        clusters[i] = []
        for ind in order_centroids[i, : 15]:
            clusters[i].append(ix2word_stop_words[ind] +  ", ")
    [print(x, *y) for x, y in clusters.items()]


if __name__ == '__main__':
    print('Loading data')
    artist2genre = json.load(open('data/artist2genre_kaggle.json', 'r', encoding='utf-8'))
    artist_lyrics = json.load(open('data/artist_lyrics_kaggle.json', 'r', encoding='utf-8'))
    genre_lyrics = get_genre_lyrics(artist_lyrics)


    # Custom tokenization to remove numbers etc.
    lyrics = [' '.join(re.findall('[A-Za-z]+', l)) for l in artist_lyrics.values()]
    genres = [' '.join(re.findall('[A-Za-z]+', l)) for l in genre_lyrics.values()]

    ##### TF IDF #####
    print('Vectorizing with TF IDF')
    tfidf_vectorizer = TfidfVectorizer()  # siehe https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
    tfidf_matrix = tfidf_vectorizer.fit_transform(artist_lyrics).toarray()
    ix2word = tfidf_vectorizer.get_feature_names()  # Dict, für Zuweisung ID -> Wort

    # sublinear_tf=True addresses the problem that 20 occurrences of a word is probably not 20 times more important than 1 occurrence
    # terms which occur 10 times would have tf=2, words which occur 1000 times would have tf = 4
    tfidf_vectorizer_stop_words = TfidfVectorizer(stop_words='english', sublinear_tf=True)
    tfidf_matrix_stop_words = tfidf_vectorizer_stop_words.fit_transform(lyrics).toarray()
    ix2word_stop_words = tfidf_vectorizer_stop_words.get_feature_names()  # Dict, für Zuweisung ID -> Wort

    ##### PCA #####
    plot_pca(tfidf_matrix_stop_words, artist_lyrics)

    ##### Distinct words #####
    print('\n ##### Distinct words per artist #####')
    print_distinct_words(artist_lyrics, tfidf_matrix_stop_words, ix2word_stop_words)

    print('\n ##### Distinct words per genre #####')
    tfidf_vectorizer_genre = TfidfVectorizer(stop_words='english', sublinear_tf=True)
    tfidf_matrix_genre = tfidf_vectorizer_genre.fit_transform(genres).toarray()
    ix2word_genre = tfidf_vectorizer_genre.get_feature_names()  # Dict, für Zuweisung ID -> Wort
    print_distinct_words(genre_lyrics, tfidf_matrix_genre, ix2word_genre)

    #### Hierarchical Clustering #####
    print('\n #### Hierarchical clustering artists #####')
    kmeans_artist = KMeans(n_clusters=10).fit(tfidf_matrix)
    print_clusters(kmeans_artist.labels_, ix2word)

    print('\n #### Hierarchical clustering genre_lyrics #####')
    tfidf_vectorizer_genre = TfidfVectorizer(stop_words='english', sublinear_tf=True)
    tfidf_matrix_genre = tfidf_vectorizer_genre.fit_transform(genre_lyrics).toarray()
    ix2word_genre = tfidf_vectorizer_genre.get_feature_names()
    kmeans_genre = KMeans(n_clusters=5).fit(tfidf_matrix_genre)
    print_clusters(kmeans_genre.labels_, ix2word_genre)

    print('\n #### Hierarchical clustering lyric #####')
    n_clusters = 10
    kmeans_lyrics = KMeans(n_clusters=n_clusters).fit(tfidf_matrix_stop_words)
    order_centroids = kmeans_lyrics.cluster_centers_.argsort()[:, ::-1]
    print_clusters_lyrics(ix2word_stop_words, order_centroids, n_clusters)


    #### Dendogram #####
    print('\n  #### Dendogram #####')
    artist_names = [a + ': ' + artist2genre[a].upper() for a in list(artist_lyrics.keys())]
    clustered_artist = linkage(tfidf_matrix_stop_words, 'ward')
    plot_dendrogram(clustered_artist, artist_names)
