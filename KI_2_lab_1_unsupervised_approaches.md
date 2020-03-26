# KI Lab **1**: **Unsupervised** approaches

**Authoren**: Pascal Sager, Luca Stähli

**Code**: https://github.com/sagerpascal/KI2-Praktikum1

## Ziel

Ziel ist es aus den zur Verfügung gestellten <a href="https://www.kaggle.com/gyani95/380000-lyrics-from-metrolyrics" target="__blank">Datensätzen</a>, welche über 380'000 lyrics von verschiedenen Künstler in unterschiedlichen Genres enthalten zu clustern. Die Datensätze sind im csv wie folgt strukturiert:

```
index,  song,  year,  artist,  genre,  lyrics
```

## Ansatz

In einem ersten Schritt werden die Files **artist2genre_kaggle.json** und **artist_lyrics_kaggle.json** generiert. 

Das erste File enthält pro Genre die Top 10 Künstler mit den meisten Tracks in diesem Genre.

```Json
artist2genre = { "artist": "genre", ...}
```

Letzteres File enthält sämtliche Lyrics dieser Künstler zusammengefasst.

```json
artist_lyrics = { "artist": "lyric 1 lyric 2 ...", ... ]
```

#### TF-IDF (Term Frequency – Inverse Document Frequency) 

Anschliessend wird Mithilfe der sklearn Library die TF-IDF Methode angewendet. Dabei ist folgende Unterscheidung wichtig:

- **Term Frequency**: Fasst zusammen, wie oft ein bestimmtes Wort in einem Dokument vorkommt.
- **Inverse Document Frequency**: Hiermit wird die Wichtigkeit von Wörter, welche in verschiedenen Dokumenten häufig vorkommen, verkleinert (z.B. the, a, is).

Die TF-IDF Methode bewertet also, wie wichtig ein Wort im Vergleich zu dessen Kontext ist, indem es für jedes Word einen TF-IDF score berechnet. Bei der Initialisierung könnte als Parameter zusätzlich noch stop_words='english' mitgegeben werden um die Stoppwörter zu filtern oder sublinear_tf=True um Wörter welcher öfter vorkommen in einem artist_lyric nicht automatisch mehr Gewichtung zu geben.

```python
tfidf_vectorizer = TfidfVectorizer() # sublinear_tf=True, stop_words='english'
# compute the word counts, idf and tf-idf values of the lyrics
tfidf_matrix = tfidf_vectorizer.fit_transform(lyrics).toarray()
ix2word = tfidf_vectorizer.get_feature_names()
print(tfidf_matrix.shape)
```

Das Resultat tfidf_matrix ist eine 100x72302 Matrix (100 artist_lyrics, 7'3202 einzigartige Wörter). Pro artist_lyric wird also jedem einzigartigen Wort ein tfidf-score zugewiesen. ix2word ist eine Liste mit den feature namen (einzigartigen Wörter). Folgendes Beispiel verdeutlicht dies:

ix2word = ['between', 'me', 'to', 'is', 'one', 'second', ...]

tfidf_matrix  = [ [ 0.		0.0.46979139		0.58028582 ... ], [ 0.		0.		  0.45624415 ...], ...]

|                | between | me         | to         |
| -------------- | ------- | ---------- | ---------- |
| artist_lyric 1 | 0.      | 0.46979139 | 0.58028582 |
| artist_lyric 2 | 0.      | 0.         | 0.45624415 |

#### Principle Components (PCA)

Um das Ergebnis analysieren zu können, wird die tfidf_matrix mittels dem PCA Verfahren auf eine Ebene mit 2 Dimensionen projiziert. Zuerst müssen die Daten jedoch noch mit dem StandardScaler standardisiert werden (vgl. [Importance of Feature Scaling](https://scikit-learn.org/stable/auto_examples/preprocessing/plot_scaling_importance.html)). Wie gross die Abweichung gegenüber den ursprünglichen  Daten ist, kann zusätzlich über die cumulative explained variance berechnet und analysiert werden.

```python
X_scaled = StandardScaler().fit_transform(tfidf_matrix)
pca = PCA(n_components=2)
Y_pca = pca.fit_transform(X_scaled)
explained_variance = pca.explained_variance_ratio_
```

#### K-Means Clustering

In einem nächsten Schritt werden die Cluster Center und der anliegendste Cluster für jedes Sample der tfidf_matrix mit dem K-Means Algorithmus berechnet.

```python
# 10 Genres
kmeans = KMeans(n_clusters=10)
clustered = kmeans.fit(tfidf_matrix_stop_words_filtered)
```

#### Hierarchical clustering

Ebenfalls wird basierend auf der tfidf_matrix ein Dendogramm mit "Ward" als Cluster-Linkage-Kriterium erstellt. 

```Python
clustered_artist = linkage(tfidf_matrix_stop_words_filtered, 'ward')
lot_dendrogram(clustered_artist, artist_names)
```



## Resultat

Cluster the song lyrics into the numer of genres or artists. Do the clusters correspond to the genres/artists? (kMeans)

Das selbe Verhalten wie beim PCA-Plot ist auch beim Clustern der Samples ersichtlich. Die Genre Hip-Hop und die Sängerin Celine Dion können einem Cluster eindeutig zugewiesen werden. Durch die Aufgabe Distinct Words  lassen sich allenfalls einige weitere Genres bestimmen (Wort moody -> Indie?, Wort nymphetamine, lilith, rites -> cradle-of-filth?)

| Cluster | Wörter                                                       | Genre            |
| ------- | ------------------------------------------------------------ | ---------------- |
| 0       | love,  just,  don,  know,  oh,  ll,  like,  let,  yeah,  wanna,  make,  way,  little,  ve,  want, ... | ?                |
| 1       | like,  don,  just,  ll,  love,  know,  ve,  oh,  time,  got,  want,  say,  life,  let,  come, ... |                  |
| 2       | niggas,  nigga,  hoes,  homie,  gon,  pussy,  tryna,  haters,  holla,  gangsta,  mma,  mayne,  shawty,  imma,  rapper, ... | Hip-Hop          |
| 3       | mutilation,  kasso,  walkaway,  infection,  anodized,  saturates,  undercurrents,  demise,  janssens,  impact,  elimination,  breed,  beings,  desecrate,  forced, ... |                  |
| 4       | ridi,  des,  comme,  bois,  dans,  novus,  totus,  les,  qui,  questa,  terre,  et,  une,  sont,  pas, ... | Celine Dion      |
| 5       | love,  ll,  don,  know,  just,  like,  ve,  time,  way,  got,  oh,  say,  day,  let,  baby, ... |                  |
| 6       | love,  embraceable,  spoonful,  ll,  just,  don,  got,  know,  sookie,  ve,  baby,  oh,  darn,  let,  like, ... |                  |
| 7       | agus,  nl,  tir,  ina,  dlamn,  shule,  abhaile,  bh,  liom,  maidin,  gaelach,  ar,  bheidh,  fol,  fhearr, ... |                  |
| 8       | nymphetamine,  lilith,  rites,  seraphim,  danse,  ebon,  succulent,  countess,  gabrielle,  midian,  ravens,  nemesis,  succubus,  gilles,  benighted, ... | cradle-of-filth? |
| 9       | mooday,  lonelily,  sera,  prague,  gamete,  unplayed,  boned,  kitsch,  groovin,  moody,  weatherman,  amie,  quand,  yoga,  grower, ... | Indie?           |



**Generate a dendrogram of the songs. Do songs from the same genre/artists appear in the same branches? Do higher level nodes group similar artists/genres as their children? (hierarchical clustering)**

Die Klassifizierung beim Dendogramm mit "Ward" funktioniert sehr gut bis auf einzelne Ausnahmen. Selbst bei der Gruppierung auf höherer Ebene ist es noch nachvollziehbar, da die Musikstile der Gruppen in bestimmten Punkten ähnlich sind:

- Hip-Hop
- R&B, Jazz, Country, Pop
- Metal, Electronic, Rock, Indie, Folk

Link: https://github.com/sagerpascal/KI2-Praktikum1/blob/master/dendrogram.svg



**What are the most distinct terms within a genre or per artist? (TF IDF)**

- TfidfVectorizer() initialisiert mit sublinear_tf=True und stop_words='english'
- lyrics Liste gefiltert nach Wörter welche nur Buchstaben enthalten

Distinct words per artist:

```python
devendra-banhart['angelika', 'bandera', 'melo', 'naa', 'seena']
the-blood-brothers['johnstone', 'lyons', 'linda', 'peacock', 'edward']
frank-turner['armadillo', 'wessex', 'mittens', 'josephine', 'recovery']
damien-rice['mooday', 'lonelily', 'sera', 'gamete', 'prague']
dar-williams['summerday', 'buzzer', 'teenagers', 'yoko', 'flinty']
david-guetta['cranks', 'woohoo', 'toyfriend', 'crank', 'ge']
bjrthrk['hann', 'hn', 'thad', 'ekki', 'svo']
crduan-xshadows['gambit', 'swims', 'valkyrie', 'matchstick', 'sangrael']
armin-van-buuren['dominator', 'unforgivable', 'rougher', 'dom', 'tougher']
everything-but-the-girl['neglecting', 'anytown', 'deserts', 'hooch', 'salisbury']
fall['bournemouth', 'spinetrak', 'muzorewi', 'shiftwork', 'oranj']
fear-factory['kasso', 'walkaway', 'anodized', 'undercurrents', 'janssens']
cannibal-corpse['innards', 'mutilate', 'cadaver', 'gore', 'pus']
cradle-of-filth['nymphetamine', 'lilith', 'rites', 'seraphim', 'danse']
anthrax['friggin', 'riggin', 'cupajoe', 'schism', 'martyrs']
bill-anderson['publications', 'anderson', 'wariner', 'saginaw', 'jan']
emmylou-harris['evangeline', 'bobbie', 'caffeine', 'babyion', 'loora']
eddy-arnold['ea', 'irene', 'didli', 'drea', 'eam']
dolly-parton['limozeen', 'parton', 'applejack', 'sug', 'eee']
buck-owens['beup', 'liggy', 'bakersfield', 'abilene', 'beum']
frank-sinatra['eydie', 'baubles', 'linga', 'bangles', 'venite']
bing-crosby['constantinople', 'partridge', 'macnamara', 'dinah', 'hens']
ella-fitzgerald['tain', 'paganini', 'choppity', 'flam', 'baaaaaaaaaaimp']
billie-holiday['pom', 'pigfoot', 'moonglow', 'porgy', 'affraid']
dean-martin['jl', 'chee', 'bimba', 'dm', 'babababoo']
celtic-woman['siuil', 'shule', 'nl', 'bheidh', 'agus']
clannad['agus', 'mise', 'gur', 'bhi', 'droichead']
arrogant-worms['winnebago', 'chomp', 'ignace', 'doot', 'manly']
gordon-lightfoot['borderstone', 'katy', 'alberta', 'cherokee', 'lavender']
ataraxia['ridi', 'novus', 'totus', 'questa', 'pereo']
david-bowie['buh', 'tvc', 'suffragette', 'whop', 'helden']
bob-dylan['quinn', 'suckling', 'tweedle', 'brownsville', 'jokerman']
b-b-king['lucille', 'vengo', 'boogies', 'fuyer', 'hody']
elvis-costello['moderns', 'shabby', 'sulky', 'clubland', 'uncomplicated']
elton-john['dorado', 'charan', 'jatee', 'mairi', 'chloe']
50-cent['niggas', 'nigga', 'homie', 'hoes', 'niggaz']
chris-brown['niggas', 'womp', 'shabba', 'nigga', 'beh']
drake['niggas', 'houstatlantavegas', 'nigga', 'jumpman', 'drizzy']
chamillionaire['chamillionaire', 'koopa', 'chamillitary', 'rasaq', 'chamillion']
eminem['hailie', 'eminem', 'dre', 'mathers', 'obie']
britney-spears['womanizer', 'tik', 'danja', 'phonography', 'prerogative']
bee-gees['braff', 'spicks', 'coalman', 'specks', 'blamin']
celine-dion['aux', 'ami', 'autres', 'tes', 'veux']
barbra-streisand['barbra', 'marmelstein', 'soun', 'tekel', 'brice']
american-idol['alejandro', 'dda', 'juda', 'jolene', 'ihr']
anti-flag['brandenburg', 'emo', 'sux', 'submitted', 'exodus']
aretha-franklin['wholy', 'soulville', 'feelgood', 'matty', 'dwee']
etta-james['spoonful', 'sookie', 'dickory', 'pushover', 'ouh']
babyface['babyface', 'daryl', 'shoop', 'roni', 'chow']
brian-mcknight['youporn', 'iyo', 'nasaktan', 'aaaah', 'mcknight']
```

Distinct words per genre:

```python
Indie['johnstone', 'mooday', 'lyons', 'angelika', 'summerday']
Electronic['hann', 'hn', 'thad', 'ekki', 'svo']
Metal['nymphetamine', 'lilith', 'bournemouth', 'rites', 'spinetrak']
Country['parton', 'beup', 'limozeen', 'mhm', 'wagoner']
Jazz['jl', 'chee', 'bimba', 'dinah', 'babababoo']
Folk['agus', 'ridi', 'liom', 'gan', 'mise']
Rock['buh', 'tvc', 'ramona', 'suffragette', 'helden']
Hip-Hop['niggas', 'chamillionaire', 'homie', 'dre', 'nigga']
Pop['monde', 'womanizer', 'comme', 'jamais', 'autre']
R&B['babyface', 'wholy', 'spoonful', 'daryl', 'shoop']
```



**Create a 2D plot that shows the similarity of the artists/genres. (PCA)**

- TfidfVectorizer() initialisiert mit stop_words='english'
- lyrics Liste gefiltert nach Wörter welche nur Buchstaben enthalten

Eine Analyse der TF-IDF scores  mit dem PCA-Plot zwischen den Künstlern oder den Genres ist schwierig, da sich viele Samples sehr ähnlich sind.  Gut erkennbar sind einige Ausreisser vorallem aus dem Genre Hip-Hop wie eminem, chamillionaire, 50-Cent oder Drake. Weitere Ausreisser sind Celine-Dion (aufgrund französischer Songs) und einige Künstler aus dem Metal-Genre. Das Resultat ist nachvollziehbar, denn es bestätigt die Annahme, dass die Wortwahl aus den Bereichen Hip-Hop und Metal gegenüber den Songs aus den restlichen Genres abweicht.

![image-20200325221647120](C:\Users\lucas\AppData\Roaming\Typora\typora-user-images\image-20200325221647120.png)

![image-20200325221902029](C:\Users\lucas\AppData\Roaming\Typora\typora-user-images\image-20200325221902029.png)

Explained Variance:

![image-20200326000106854](C:\Users\lucas\AppData\Roaming\Typora\typora-user-images\image-20200326000106854.png)

## Quellen

- C. Müller, Andreas und Guido, Sarah 2017: Introduction to Machine Learning with Python: A Guide for Data Scientists
- scikit-learn2019: Importance of Feature Scaling. URL: https://scikit-learn.org/stable/auto_examples/preprocessing/plot_scaling_importance.html