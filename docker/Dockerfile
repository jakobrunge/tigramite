FROM debian:bookworm-slim
WORKDIR /opt/tigramite
COPY ./docker/dependencies.txt ./docker/install_packages.sh ./
RUN DEPFILE=./dependencies.txt ./install_packages.sh
COPY . .
RUN make install_docker
RUN make install 
USER appuser
