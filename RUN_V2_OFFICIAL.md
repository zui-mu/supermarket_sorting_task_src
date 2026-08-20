# DG-202606 V2.0 Official Runtime Notes

This local source tree is now configured for the official V2.0 server/client split.

## What Changed

- Official tasks are read from `/supermarket_sorting/task`.
- The default order mode is `SUPERMARKET_ORDER=official`.
- Runtime layout truth is disabled by default: `SUPERMARKET_ALLOW_RUNTIME_LAYOUT=0`.
- The official-style helper now limits the published task list with `SUPERMARKET_TASK_COUNT=5`.
- Set `SUPERMARKET_TASK_ANONYMOUS=1` to emit anonymous task ids like the formal V2.0 message.
- Full-scene `SUPERMARKET_TASKS=all` is only for local stress tests, not formal 5-order scoring.

## Image Names

After loading the official tar files, tag them as:

```bash
docker tag crpi-1pzq998p9m7w0auy.cn-hangzhou.personal.cr.aliyuncs.com/challengecup/supermarket_sorting_final:server supermarket_sorting:server
docker tag crpi-1pzq998p9m7w0auy.cn-hangzhou.personal.cr.aliyuncs.com/challengecup/supermarket_sorting_final:client supermarket_sorting:client
```

## Client Scripts

Run inside the official client container:

```bash
cd /workspace/baseline
./scripts/run_v2_perception.sh
./scripts/run_v2_decision_client.sh
```

## Formal Test Helper

To run the official score-enabled flow with one command, use:

```bash
./scripts/run_v2_official_test.sh
```

This starts the score-enabled server, then the perception and decision clients,
and writes logs under `logs_official/`.

For development-only truth-assisted tests, explicitly set:

```bash
export SUPERMARKET_ALLOW_RUNTIME_LAYOUT=1
```

Do not use that flag for official-style validation.

The official helper waits for a real `/supermarket_sorting/task` message before
starting the client. This avoids mistaking slow Server initialization for a
navigation or grasp failure. `run_v2_smoke_test.sh` is separate and explicitly
uses the development `gt` backend for a deterministic single-target check.
The official helper refuses `gt`, runtime-layout truth, and `/referee/state`
oracle flags so a local score cannot be mistaken for a compliant result.
