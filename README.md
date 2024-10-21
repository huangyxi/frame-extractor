# Extract Key Frames Frome Video Based on IoU Score

## Installation

### Dependencies

```bash
vcpkg install cxxopts
vcpkg install indicators
```

OpenCV is required to be installed on your system. You can install it using the following command:

Linux:
```bash
sudo apt-get install libopencv-dev
```

Mac:
```bash
brew install opencv
```

### Build

```bash
mkdir build
cd build
cmake ..
cmake --build .
```

## Usage

```bash
./frame-extract [-f <frame step>] [-o <output directory>] <video file>
```
More options can be found by running `./frame-extract --help`.


## Post-Processing

Use `ImageMagick` to convert the extracted frames to a PDF file.
```bash
convert *.png output.pdf
```

Use `ps2pdf` to reduce the size of the PDF file. [^1]
```bash
ps2pdf -dPDFSETTING=/ebook output.pdf output-compressed.pdf
```

[^1]: https://askubuntu.com/a/243753
