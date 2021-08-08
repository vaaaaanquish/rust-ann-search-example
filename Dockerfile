FROM rust

WORKDIR /app
RUN apt update &&\
    rm -rf ~/.cache &&\
    apt clean all &&\
    apt install -y cmake &&\
    apt install -y clang
RUN apt install -y build-essential libffi-dev libssl-dev zlib1g-dev liblzma-dev libbz2-dev libreadline-dev libsqlite3-dev libopencv-dev tk-dev git


# install python
ENV HOME="/app"
ENV PYENV_ROOT="$HOME/.pyenv"
ENV PATH="${PYENV_ROOT}/shims:${PYENV_ROOT}/bin:${PATH}"
RUN git clone https://github.com/pyenv/pyenv.git $HOME/.pyenv
RUN echo 'eval "$(pyenv init -)"' >> ~/.bashrc
RUN eval "$(pyenv init -)"
RUN pyenv install 3.7.7
RUN pyenv global 3.7.7
RUN pip install tensorflow==2.5.0

# dump pretrain resnet model
RUN mkdir /app/model
COPY ./make_model_file.py /app
RUN python make_model_file.py

# file
COPY ./Cargo.toml /app/Cargo.toml
COPY ./src /app/src
COPY ./img /app/img

ENTRYPOINT [ "/bin/bash" ]
