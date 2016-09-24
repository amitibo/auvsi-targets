# AUVSI ADLC Training Samples Creation.

This package creates target samples for training the ADLC algortihm.
The targets are created accoording to the
[AUVSI SUAS competition](http://www.auvsi-suas.org/).
The sofware was developed and used by the TAS team during the
[2015](http://www.auvsi-suas.org/competitions/2015/) and 
[2016](http://www.auvsi-suas.org/competitions/2016/) competitions.

### Prerequisits

* **Python** - Recommended to install using a distibution like
  [Anaconda](https://www.continuum.io/downloads).
* **opencv** - Install from [Unofficial python binaries](http://www.lfd.uci.edu/~gohlke/pythonlibs/).
  Tested with version >= 2.4.10
* **aggdraw** - (used for the image processing project):
  Install from [Unofficial python binaries](http://www.lfd.uci.edu/~gohlke/pythonlibs/).
* **exifread** - Install with ```pip install exifread```.
* **pyqrcode** - Install with ```pip install pyqrcode```.
* **pypng** - Install with ```pip install pypng```.

### Installation

    > git clone 
    > cd auvsi_targets
    > python setup.py develop

## How to Use

* Put True-Type fonts inside ```AUVSItargets/resources/fonts``` (The package
  does not include fonts.)
* To create shape trainig samples run ```scripts/create_patches.py```
* To create letter training samples run ```scripts/create_letter_samples.py```

The shape samples and letter samples are created under ```DATA/train_images```
and ```DATA/train_letter``` respectively. Each sample is made of an image
file and corresponding label file.

## License

see `LICENSE` file.