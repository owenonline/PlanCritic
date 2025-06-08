FROM ubuntu:22.04

# Set environment variables to avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Install build dependencies for VAL and OPTIC
RUN apt-get update && apt-get install -y \
    build-essential git cmake flex bison perl wget curl \
    coinor-libcbc-dev coinor-libclp-dev coinor-libcoinutils-dev \
    libbz2-dev libgsl-dev coinor-libosi-dev coinor-libcgl-dev \
    python3 python3-pip python3-venv \
    && \
    rm -rf /var/lib/apt/lists/*

# Clone and build VAL using the proper build script with required parameters
RUN git clone https://github.com/KCL-Planning/VAL.git && \
    cd VAL && \
    chmod +x scripts/linux/build_linux64.sh && \
    ./scripts/linux/build_linux64.sh all Release

RUN git clone https://github.com/Dongbox/optic-clp-release.git

# Create workspace directory, and copy the files needed to run plan critic. since we need to modify what we're copying during building, we can't have it as a bind mount (however, we preservea adherence model training, the domains folder, and the experiment config in that way so they can be edited without rebuilding the contianer)
WORKDIR /workspace 
COPY plan_critic /workspace/plan_critic
COPY pyproject.toml /workspace/pyproject.toml
COPY cai_ui.png /workspace/cai_ui.png

# Copy the built tool binaries to the correct location
RUN cp /VAL/build/linux64/Release/bin/Validate /workspace/plan_critic/tools/Validate
RUN cp /VAL/build/linux64/Release/bin/libVAL.so /workspace/plan_critic/tools/libVAL.dylib
RUN cp /optic-clp-release/optic-clp /workspace/plan_critic/tools/optic-cplex

# Install Python dependencies and PlanCritic in editable mode
RUN cd /workspace && \
    pip3 install --upgrade pip && \
    pip3 install .

# Set up networking to connect to host CouchDB
# The application should use host.docker.internal:5984 when running in Docker
ENV COUCHDB_URL=http://admin:password@host.docker.internal:5984/

CMD ["bash"]
