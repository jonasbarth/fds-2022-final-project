# Data For Final Report
The purpose of this package is to gather notebooks and scripts that generate some data to be used in the final report.

# channels
The `channels.py` is a script that takes an image as input, extracts specified channels, and saves images with the
extracted channels.

It has the following flags:
* `-i` the path to the image.
* `-c` the channels to extract. Each channel is separated by `,` and each set of channels is separated by `:`.
* `-o` the output directory. Will be created if it doesn't exist.

Running from the root of the project:
```bash
python -m report.channels -i data/image/train/slice_12_img_136.npy -c 1,2,3:6,7,8:11,12,13 -o ./doc
```

# hog
The `hog.py` script that takes an image as input, extracts specific channels and saves the histogram of gradient for 
channel.

It has the following flags:
* `-i` the path to the image.
* `-c` the channels to extract. Each channel is separated by `,` and each set of channels is separated by `:`.
* `-o` the output directory. Will be created if it doesn't exist.

Running from the root of the project:
```bash
python -m report.hog -i data/image/train/slice_12_img_136.npy -c 1,2,3:6,7,8:11,12,13 -o ./doc
```


# mask
The `masks.py` is a script that takes a mask as input, converts it to grayscale, and saves it as a `.jpg` file.

It has the following flags:
* `-m` the path to the mask.
* `-o` the output directory. Will be created if it doesn't exist.

Running from the root of the project:
```bash
python -m report.mask -i data/image/train/slice_12_mask_136.npy -o ./doc
```