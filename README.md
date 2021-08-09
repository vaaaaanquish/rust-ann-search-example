# rust-ann-search-example
Image search example by approximate nearest-neighbor library In Rust

use
```
 - tensorflow 0.17.0
 - pretrain ResNet50
 - hora (Rust ANN library) 0.1.1
```

image -> resize -> resnet embedding -> ANN indexing -> search


# Usage

Plese put image files in `./img` directory. If If you don't have a handy file, there is a script to create a dataset in docker image.

```sh
docker build -t ann .
docker run -it ann

# if make dataset by food101
[docker]$ ./make_food101_dataset.sh

# running
[docker]$ cargo run
```
