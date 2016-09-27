# AUVSI ADLC Training Samples Creation.

This package creates target samples for training the ADLC algortihm.
The targets are created accoording to the
[AUVSI SUAS competition](http://www.auvsi-suas.org/).
The sofware was developed and used by the TAS team during the
[2015](http://www.auvsi-suas.org/competitions/2015/) and 
[2016](http://www.auvsi-suas.org/competitions/2016/) competitions.

### Installation

#### Linux
    > git clone https://github.com/amitibo/auvsi-targets.git
    > cd auvsi_targets
    > sudo apt-get install python-opencv virtualenvwrapper ttf-dejavu
    > mkvirtual --system-site-packages auvsi # we need access to opencv from the virtualenv
    (auvsi) > pip install -r requirements.txt
    (auvsi) > ./install_aggdraw

#### Windows
* **Python** - Recommended to install using a distibution like
  [Anaconda](https://www.continuum.io/downloads).
* **opencv** - Install from [Unofficial python binaries](http://www.lfd.uci.edu/~gohlke/pythonlibs/).
  Tested with version >= 2.4.10
* **aggdraw** - (used for the image processing project):
  Install from [Unofficial python binaries](http://www.lfd.uci.edu/~gohlke/pythonlibs/).
    > git clone https://github.com/amitibo/auvsi-targets.git
    > cd auvsi_targets
    > pip install -r requirements.txt

## How to Use

* To create shape trainig samples run ```./create_patches.py```
* To create letter training samples run ```./create_letter_samples.py```

Input images are under ```DATA/resized_images``` and ```DATA/renamed_images```, flight data is under ```DATA/flight_data```.

The shape samples and letter samples are created under ```DATA/train_images```
and ```DATA/train_letter``` respectively. Each sample is made of an image
file and a corresponding label file.

## License

see `LICENSE` file.















