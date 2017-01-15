# genreclassifier
## Extract feature
```
python2 feature.py [FILE]
```
## Train and Eval
- Use features from `feature.json`.
```
python2 train.py
```
## Dataset
The dataset used for training the model is the GTZAN dataset. A brief of the data set: 

* This dataset was used for the well known paper in genre classification " Musical genre classification of audio signals " by G. Tzanetakis and P. Cook in IEEE Transactions on Audio and Speech Processing 2002.
* The dataset consists of 1000 audio tracks each 30 seconds long. It contains 10 genres, each represented by 100 tracks. The tracks are all 22050Hz Mono 16-bit audio files in .wav format.
* Official web-page: [marsyas.info](http://marsyas.info/download/data_sets)
* Download size: Approximately 1.2GB
* Download link: [Download the GTZAN genre collection](http://opihi.cs.uvic.ca/sound/genres.tar.gz)

