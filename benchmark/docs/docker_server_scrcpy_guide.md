# Docker Server + scrcpy Viewing Guide

This guide explains exactly how to:

1. Start AndroidWorld server with Docker.
2. Verify the emulator and server are healthy.
3. View/control the emulator from your local desktop using `scrcpy`.

## A. Prerequisites

On the remote server:

```bash
docker --version
ls -l /dev/kvm
```

You should see `/dev/kvm` present for the recommended in-container emulator mode.

On your local desktop:

```bash
adb version
scrcpy --version
ssh -V
```

## B. First-Time Setup (Remote Server)

Run from repository `benchmark` directory:

```bash
cd $HOME/AnyAppBench/benchmark

# 1) Build image
docker build -t android_world:latest .

# 2) First-time app/emulator provisioning run (can take several minutes)
docker run --rm --name android_world_container --network host \
  --privileged --device /dev/kvm \
  -e AW_PERFORM_EMULATOR_SETUP=1 \
  android_world:latest
```

After first-time setup completes, stop the container if still running with `Ctrl+C`.

## C. Every Time You Start the Server (Remote Server)

```bash
cd $HOME/AnyAppBench/benchmark

# 1) Clean up old container from previous run
docker rm -f android_world_container 2>/dev/null || true

# 2) Start server + in-container emulator
docker run --rm --name android_world_container --network host \
  --privileged --device /dev/kvm \
  -e AW_PERFORM_EMULATOR_SETUP=0 \
  android_world:latest
```

Keep this terminal open while the server is running.

## D. Health Checks (Remote Server)

Open a second terminal on the same remote server:

```bash
# Server health endpoint
curl -i http://127.0.0.1:5000/health

# Emulator visible to adb
adb devices -l

# ADB TCP endpoint used for scrcpy tunnel
ss -ltn | grep 5555
```

Expected:

1. `/health` returns HTTP `200`.
2. `adb devices -l` shows `emulator-5554` in `device` state.
3. Port `127.0.0.1:5555` is listening.

## E. Connect from Local Desktop with scrcpy

### Step 1: Create SSH tunnel (Local Desktop)

Keep this running in one local terminal:

```bash
ssh -N -L 5555:127.0.0.1:5555 <remote_user>@<remote_host>
```

Example:

```bash
ssh -N -L 5555:127.0.0.1:5555 ttran@203.0.113.10
```

### Step 2: Connect adb locally (Local Desktop)

In a second local terminal:

```bash
adb kill-server
adb start-server
adb connect 127.0.0.1:5555
adb devices -l
```

Expected output includes `127.0.0.1:5555` with state `device`.

### Step 3: Launch scrcpy (Local Desktop)

```bash
scrcpy -s 127.0.0.1:5555
```

You should now see and control the remote emulator on your local desktop.

## F. If Local Port 5555 Is Already Used

Use another local port, for example `5560`:

```bash
# Terminal 1 (local): tunnel 5560 -> remote 5555
ssh -N -L 5560:127.0.0.1:5555 <remote_user>@<remote_host>

# Terminal 2 (local): adb + scrcpy via 5560
adb connect 127.0.0.1:5560
scrcpy -s 127.0.0.1:5560
```

## G. Common Problems and Fixes

### 1) `failed to connect to '127.0.0.1:5555': Connection refused`

Cause: emulator adb TCP endpoint is not up yet.

Fix on remote server:

```bash
adb devices -l
ss -ltn | grep 5555
docker logs --tail 100 android_world_container
```

Wait for emulator boot to complete and retry local `adb connect`.

### 2) `offline` device in `adb devices`

Fix on remote server:

```bash
adb kill-server
adb start-server
adb devices -l
```

Then reconnect from local desktop.

### 3) Docker container not running

Fix on remote server:

```bash
docker ps
docker logs --tail 100 android_world_container
```

Restart with section C commands.

## H. Stop Everything

On remote server:

```bash
docker stop android_world_container
```

On local desktop:

1. Close `scrcpy` window.
2. Stop SSH tunnel with `Ctrl+C`.
