# CUDA copy results

The four-GPU runner writes one timestamped directory below `doc/results/4gpu/`.
Each run contains:

- `environment.txt`: GPU topology, NVLink status, CUDA version, repository revision, and command line.
- `summary.csv`: one row per D2D pattern/background-direction pair.
- `<pattern>_<direction>/command.txt`: exact D2D and background commands.
- `<pattern>_<direction>/d2d.log` and `d2d.json`: D2D measurements.
- `<pattern>_<direction>/background.log` and `background.json`: background measurements when enabled.

Run a quick validation matrix with:

```bash
./scripts/run_4gpu_copy_matrix.sh --size=1M --repeats=1
```

The default research pass uses `--size=255M --repeats=20`. The D2D warmup is
fixed at 10 iterations. `ring` issues four directed copies for four GPUs;
`allpairs` issues all twelve directed GPU pairs. The reported aggregate D2D
bandwidth uses the maximum elapsed time across the source-GPU streams.
