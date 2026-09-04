# Credits to https://github.com/amrsa1/Android-Emulator-image

FROM eclipse-temurin:17-jdk-jammy

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /
#=============================
# Install Dependencies
#=============================
SHELL ["/bin/bash", "-c"]
RUN apt update && apt install -y curl sudo wget unzip bzip2 libdrm-dev libxkbcommon-dev libgbm-dev libasound-dev libnss3 libxcursor1 libpulse-dev libxshmfence-dev xauth xvfb x11vnc fluxbox wmctrl libdbus-glib-1-2 ffmpeg

#==============================
# Android SDK ARGS
#==============================
ARG ARCH="x86_64"
ARG TARGET="google_apis"
ARG API_LEVEL="33"
ARG BUILD_TOOLS="33.0.0"
ARG ANDROID_ARCH="x86_64"
ARG ANDROID_API_LEVEL="android-${API_LEVEL}"
ARG ANDROID_APIS="${TARGET};${ARCH}"
ARG EMULATOR_PACKAGE="system-images;${ANDROID_API_LEVEL};${ANDROID_APIS}"
ARG PLATFORM_VERSION="platforms;${ANDROID_API_LEVEL}"
ARG BUILD_TOOL="build-tools;${BUILD_TOOLS}"
ARG ANDROID_CMD="commandlinetools-linux-11076708_latest.zip"
ARG ANDROID_SDK_PACKAGES="${EMULATOR_PACKAGE} ${PLATFORM_VERSION} ${BUILD_TOOL} platform-tools emulator"
ARG EMULATOR_BUILD_ID="15042444"

#==============================
# Set JAVA_HOME - SDK
#==============================
ENV ANDROID_SDK_ROOT=/opt/android
ENV PATH="$PATH:$ANDROID_SDK_ROOT/cmdline-tools/tools:$ANDROID_SDK_ROOT/cmdline-tools/tools/bin:$ANDROID_SDK_ROOT/emulator:$ANDROID_SDK_ROOT/tools/bin:$ANDROID_SDK_ROOT/platform-tools:$ANDROID_SDK_ROOT/build-tools/${BUILD_TOOLS}"
ENV DOCKER="true"
# Stable in-container defaults for nested virtualization hosts.
ENV EMULATOR_GPU_MODE="lavapipe"
ENV EMULATOR_SAFE_GPU_MODE="lavapipe"
ENV EMULATOR_SEGFAULT_GPU_FALLBACK="lavapipe"

#============================================
# Install required Android CMD-line tools
#============================================
RUN wget https://dl.google.com/android/repository/${ANDROID_CMD} -P /tmp && \
    unzip -d $ANDROID_SDK_ROOT /tmp/$ANDROID_CMD && \
    mkdir -p $ANDROID_SDK_ROOT/cmdline-tools/tools && cd $ANDROID_SDK_ROOT/cmdline-tools &&  mv NOTICE.txt source.properties bin lib tools/  && \
    cd $ANDROID_SDK_ROOT/cmdline-tools/tools && ls

#============================================
# Install required packages using SDK manager
#============================================
RUN yes Y | sdkmanager --licenses
RUN set -eux; \
        for i in 1 2 3; do \
            if yes Y | sdkmanager --verbose --sdk_root="${ANDROID_SDK_ROOT}" ${ANDROID_SDK_PACKAGES}; then \
                break; \
            fi; \
            if [ "$i" -eq 3 ]; then \
                echo "sdkmanager failed after 3 attempts" >&2; \
                exit 1; \
            fi; \
            echo "sdkmanager attempt $i failed, retrying..." >&2; \
            sleep $((i * 20)); \
        done

# Overlay a pinned emulator build to avoid host-specific crashes in some revisions.
RUN wget "https://dl.google.com/android/repository/emulator-linux_x64-${EMULATOR_BUILD_ID}.zip" -P /tmp && \
    unzip -qo "/tmp/emulator-linux_x64-${EMULATOR_BUILD_ID}.zip" -d "$ANDROID_SDK_ROOT" && \
    rm -f "/tmp/emulator-linux_x64-${EMULATOR_BUILD_ID}.zip"
#============================================
# Create required emulator
#============================================
ARG EMULATOR_NAME="Pixel_6_API_33"
ARG EMULATOR_DEVICE="pixel_6"
ENV EMULATOR_NAME=$EMULATOR_NAME
ENV DEVICE_NAME=$EMULATOR_DEVICE
RUN echo "no" | avdmanager --verbose create avd --force --name "${EMULATOR_NAME}" --device "${EMULATOR_DEVICE}" --package "${EMULATOR_PACKAGE}"


#====================================
# Install Python 3.11 from source
#====================================
RUN apt-get update && \
    apt-get install -y build-essential zlib1g-dev libncurses5-dev libgdbm-dev \
    libnss3-dev libssl-dev libreadline-dev libffi-dev libsqlite3-dev wget libbz2-dev && \
    wget https://www.python.org/ftp/python/3.11.3/Python-3.11.3.tgz && \
    tar -xvf Python-3.11.3.tgz && \
    cd Python-3.11.3 && \
    ./configure --enable-optimizations && \
    make -j $(nproc) && \
    make altinstall && \
    cd .. && \
    rm -rf Python-3.11.3* && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Create symlinks for python3.11
RUN ln -sf /usr/local/bin/python3.11 /usr/local/bin/python3 && \
    ln -sf /usr/local/bin/python3.11 /usr/local/bin/python

# install pip
RUN apt-get update && apt-get install python3-pip -y

#====================================
# Install uv
#====================================
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ARG PATH="$PATH:/root/.local/bin"
ENV PATH="$PATH:/root/.local/bin"

WORKDIR /app

#=========================
# Copying project to /app
#=========================
COPY . /app

RUN chmod a+x docker_setup/start_emu.sh && \
    chmod a+x docker_setup/start_emu_headless.sh && \
    chmod a+x docker_setup/entrypoint.sh

#====================================
# Install dependencies
#====================================
RUN python3.11 -m pip install --upgrade "setuptools<82" wheel
RUN uv pip install -v . --system
# Keep runtime dependencies aligned with host workflow scripts.
RUN python3.11 -m pip install -r requirements.txt pysqlite3-binary

#=======================
# framework entry point
#=======================
ENTRYPOINT [ "./docker_setup/entrypoint.sh" ]