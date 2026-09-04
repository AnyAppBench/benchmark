# Docker AVD snapshot-clone hook

`benchmark/scripts/docker_avd_snapshot_hook.py` implements the exact
`clone_activate` / `release` receipt contract consumed by
`consume_catbench_frozen_schedule.py`. It performs real Docker volume and
container operations. It does not call a model, choose an episode, generate a
plan, or authorize a CATBench run.

## Guarantees and failure behavior

For every `clone_activate` request, the hook:

1. requires a content-attested base volume with exact frozen-base, release,
   cohort, snapshot-ID, and snapshot-hash labels;
2. refuses a base volume referenced by any running or stopped container;
3. fingerprints the complete AVD tree without following symlinks;
4. creates a new volume whose name is deterministically bound to the frozen
   `snapshot_clone_id`, and refuses to reuse a pre-existing volume;
5. copies the base through a network-disabled, read-only helper container,
   using reflink/sparse copying when the Docker storage filesystem supports it;
6. re-fingerprints both base and clone and requires exact equality;
7. replaces only the configured pool worker, waits for the exact emulator
   serial to report boot completion and root-ADB readiness, and validates the
   worker mount, labels, serial, and isolated ADB-server port; and
8. writes the consumer receipt only after every check passes.

For `release`, it requires the same clone labels and exact worker mount, stops
and removes the worker container, fingerprints the post-episode AVD, removes
the clone volume, verifies that it is gone, and then writes the release
receipt. A stale clone, wrong worker, changed hook, changed base, duplicate JSON
key, unexpected request field, receipt collision, or cleanup ambiguity fails
closed. Base and worker locks serialize competing consumers.

The tree fingerprint covers sorted relative paths, entry type, permissions,
UID/GID, extended attributes, symlink targets, file size, and file content.
Wall-clock timestamps and sparse-block layout are intentionally excluded;
neither changes Android-visible snapshot contents. Special files are rejected.

## Create an observational frozen base

Provision and validate all pinned applications on an ordinary pool volume
first. Stop and detach its container while preserving the volume. The detach
action refuses an active episode clone.

```bash
export NUM_EMULATORS=2
export FIRST_CONSOLE_PORT=5576
export FIRST_GRPC_PORT=8576
export FIRST_ADB_SERVER_PORT=5051

benchmark/scripts/manage_catbench_docker_pool.sh detach-worker 0
```

Seal a new, previously nonexistent base volume from that offline source. The
command below copies real AVD bytes and records their computed fingerprint; it
does not invent the emulator-binary, system-image, AVD-config, app, or approval
hashes required by the release manifest.

```bash
export CATBENCH_DOCKER_WORKER_INDEX=0
export CATBENCH_DOCKER_FIRST_CONSOLE_PORT=5576
export CATBENCH_DOCKER_FIRST_GRPC_PORT=8576
export CATBENCH_DOCKER_FIRST_ADB_SERVER_PORT=5051

benchmark/scripts/docker_avd_snapshot_hook.py \
  --seal-base \
  --source-volume catbench-docker-avd-0 \
  --base-volume catbench-primary-api33-frozen-v1 \
  --snapshot-id catbench-primary-api33-frozen-v1 \
  --release-id catbench_acl_revision_5cat_v1 \
  --cohort-sha256 0646ac7c1b15e45be6988ff593611073c14b90a78da53eca3e190151a454aeae \
  --evidence /absolute/release/path/base_volume_seal_evidence.json
```

Keep the base volume unattached. `base_snapshot.json` must use the evidence's
`snapshot_id` and `snapshot_sha256` and must independently supply the other
fields checked by the consumer. The installed-app attestation must bind the
same snapshot ID and hash. Sealing is observation, not approval.

## Configure a consumer worker

The hook receives only request and receipt paths from the consumer. Its Docker
binding is therefore explicit environment configuration. For worker 0 in the
two-worker pool above:

```bash
export CATBENCH_DOCKER_BASE_VOLUME=catbench-primary-api33-frozen-v1
export CATBENCH_DOCKER_WORKER_INDEX=0
export CATBENCH_DOCKER_NUM_EMULATORS=2
export CATBENCH_DOCKER_FIRST_CONSOLE_PORT=5576
export CATBENCH_DOCKER_FIRST_GRPC_PORT=8576
export CATBENCH_DOCKER_FIRST_ADB_SERVER_PORT=5051
export ANDROID_ADB_SERVER_PORT=5051
export CATBENCH_DOCKER_POOL_MANAGER="$PWD/benchmark/scripts/manage_catbench_docker_pool.sh"
export CATBENCH_DOCKER_START_SCRIPT="$PWD/benchmark/docker_setup/start_catbench_emu_headless.sh"
export CATBENCH_DOCKER_BASE_START_SCRIPT="$PWD/benchmark/docker_setup/start_emu_headless.sh"
export CATBENCH_EMULATOR_MEMORY_MB=4096
export CATBENCH_EMULATOR_CORES=2
export CATBENCH_DOCKER_HELPER_IMAGE='android_world@sha256:6d8b2c148aebd3a1fe626768efe22c01a7a62cdbd2cbbe7d3f973adc57c7dd2f'
export CATBENCH_DOCKER_EMULATOR_IMAGE="$CATBENCH_DOCKER_HELPER_IMAGE"
export CATBENCH_ADB="${ANDROID_SDK_ROOT:?}/platform-tools/adb"
```

`ANDROID_ADB_SERVER_PORT` is deliberately mandatory and must match the selected
worker. The consumer and its episode runner inherit it, so their `adb -s`
commands cannot silently connect to another ADB server. Worker 1 uses serial
`emulator-5578` and ADB server `5052` with the same first-port values and
`CATBENCH_DOCKER_WORKER_INDEX=1`.

The consumer hashes the hook. The hook in turn embeds the exact accepted
SHA-256 values for the pool manager, CATBench resource wrapper, and legacy base
launcher; rejects a symlink or byte mismatch; rechecks all three before each
worker transition; and records their hashes plus the pinned 4096-MiB/2-core
contract in every receipt. A deliberate manager/launcher/resource update
therefore also requires a hook update, a new live smoke, and a new independent
release approval. Production should set `CATBENCH_DOCKER_LOCK_DIR` to a
host-local directory outside the source checkout. The hook caps its internal
command timeout below the consumer's 300-second hook timeout.

## Verification without episodes

The focused tests use an in-memory Docker boundary and local filesystem trees;
they do not emit benchmark results:

```bash
python benchmark/scripts/docker_avd_snapshot_hook_test.py -v
bash -n benchmark/scripts/manage_catbench_docker_pool.sh
```

An operational release still needs a release-bound seal and clone/boot/release
smoke, base-snapshot manifest, installed-app attestation, plan approvals, and
model endpoint attestations. This implementation alone does not make G4, G5,
or G6 green and does not authorize removing `--preflight_only`.

### Historical non-release storage smoke on 2026-07-10

The implementation was additionally exercised against the real API-33 AVD
volume of worker 1. The source was detached, sealed under the explicit release
ID `infrastructure_smoke_not_release`, cloned, booted as `emulator-5578`,
released, fingerprinted, and deleted. Worker 0 stayed healthy; worker 1 was
then restored to its original `catbench-docker-avd-1` volume. All temporary
volumes were removed. The final dependency-pinned revision used the idle,
app-provisioned worker-1 candidate as its source without interacting with any
app. Its activation parent/clone hash was
`88cc50149f22d7928a3604fd8854fc208555810199c2b4c0e3b3d751d59a5d21`;
the post-boot released-volume hash was
`f62efab5f217324f085288c160b6dce7ac1761d1589ac75a1cfc6c2457a55127`.

The exact receipt hashes and restoration state are recorded in
`benchmark/docs/audits/docker_avd_snapshot_hook_live_smoke_20260710.json`.
The byte-exact base-seal evidence, activation receipt, and release receipt are
stored beside that summary with the `docker_avd_snapshot_hook_*_20260710.json`
names; their checked-in SHA-256 values equal the hashes captured at runtime.
This was storage-and-boot validation only: it initialized no task, took no
agent action, called no model, and produced no benchmark result. An earlier
same-day infrastructure smoke observed a copied-AVD duplicate-lock error before
the launcher's bounded safe retry succeeded. That observation remains a
production base-hygiene item rather than being hidden or counted as a clean
release pass.

This historical storage smoke used an all-zero cohort label under the explicit
release ID `infrastructure_smoke_not_release`. It tested strict SHA formatting
and Docker mechanics only and is not primary-cohort evidence. Its launcher and
hook hashes were superseded after the real-clone audit exposed the missing
root-ADB readiness gate.

### Current real-cohort candidate validation

The current hook (SHA-256
`5f47587591446a5155a655adf11f2d4741b2cb68d2f69bd59611428ce9b17ec2`)
pins launcher SHA-256
`09173b2eb6e2e9929ddbc1981005492f74487973c8809d0f91dbf870dce0ef12`.
The launcher restarts `adbd` as root and refuses readiness unless the selected
serial reports uid 0.

Candidate v2 is bound to the actual primary cohort SHA-256
`0646ac7c1b15e45be6988ff593611073c14b90a78da53eca3e190151a454aeae`.
Its immutable content fingerprint is
`20bef329440fa8ab593e7633b43f8dde4b9d43d11792ed6930ca4a02732a3495`.
A fresh clone booted with automatic root ADB, re-attested every active APK set
(23/23), passed all 42 Clock You deterministic state-fixture cases, and was
then fingerprinted and deleted. Those cases are narrower than the protocol's
human action/replay/signature conformance record. The first diagnostic pass exposed and retained a
missing-root failure and a one-pixel out-of-bounds audit tap; both were fixed
before this clean rerun. Exact paths and hashes are in
`benchmark/docs/audits/docker_primary_base_candidate_v2_validation_summary_20260710.json`.

This is still observational candidate evidence: it is not independent release
approval, does not qualify the remaining app-task adapters, and contains no
agent action, model call, or C1/C2 result.

### Candidate-v3 Maps-resource follow-up (2026-07-11)

A later check-only preflight found that both persistent workers, and therefore
worker-0-derived candidate v2, lacked AndroidWorld's required OsmAnd
`Liechtenstein_europe.obf`. Candidate v2 was not modified. The resource was
installed on mutable worker 0 together with exact Organic Maps and CoMaps
offline resources. This repair exposed a stale CoMaps `260421` path; the pinned
APK native library and an exact official source revision identify series
`2026.04.05`, so the official `260405` resources were pinned by size and
SHA-256. All 21 external/internal device copies across 11 logical resources
match their pins.

The repaired mutable volume re-attested 23/23 apps and was sealed as
`catbench-primary-api33-candidate-v3-20260711`, fingerprint
`1b7a9832c3addd700f28449094663de98e4dadd1677e72a872541ecc98816990`.
Unlike the earlier schedule-shaped diagnostic request, the v3 request is
explicitly `analysis_eligible=false`, conformance-only, and bound to a matching
Maps/OsmAnd identity. Its fresh clone proved root ADB, re-attested 23/23,
re-matched every resource, and passed GPX/KML/link exact-versus-reversed or
wrong-place storage-helper checks for OsmAnd, Organic Maps, and CoMaps. The
clone was then fingerprinted and deleted and both baseline workers restored.

The composite report is
`benchmark/docs/audits/docker_primary_base_candidate_v3_validation_summary_20260711.json`.
This is not full Maps UI/primitive-action conformance, independent resource or
base approval, a model episode, or permission to launch C1/C2.

### Candidate-v3 Files storage-fixture follow-up (2026-07-11)

A later r3 request was separately labeled `analysis_eligible=false` and
conformance-only. Before the Files audit, a fresh attestation re-pulled all 23
active app sets from that boot; the audit then rehashed the five Files APK sets
again and matched them to the pins. It also inspected the live Docker worker
and required the receipt's exact clone volume, snapshot-clone label, image ID
and digest, serial, ADB server, console port, and gRPC port before any case ran.

All 40 launched rows for five real file managers and eight durable semantics
passed 185 independently reseeded no-op/negative/exact-positive storage-
predicate fixtures. The clone was fingerprinted, deleted, and worker 0 was
restored to `catbench-docker-avd-0`. The separate machine record is
`benchmark/docs/audits/docker_primary_base_candidate_v3_files_storage_conformance_r3_20260711.json`.
It explicitly records that transitions were injected through ADB, not app UI;
the app classes share predicates; ViewInfo/Share were excluded; and no human
trajectory, raw native-state trace, or per-adapter snapshot reset/replay was
captured. It is not Files G3/G4 qualification or permission to launch C1/C2.
