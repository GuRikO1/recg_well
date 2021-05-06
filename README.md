# Decoding for Identifiable Wells

## Set up
* manage python libs using pipenv
```
$ pipenv install
```

## Run
* Test your algorithm
```
$ pipenv run python3 main.py -p ./pictures/Image_00001_CH4.jpg -d True
```

* Verify the accuracy of the mark presence/absence discrimination algorithm
```
$ pipenv tun python3 test_main.py
```


* Verify that the answer labels are correct
```
$ pipenv run python3 test_marker_labels.py
```

*

Created by Yuriko Ezaki
