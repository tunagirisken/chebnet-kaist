# KAIST Car-Hacking Dataset (external — not in git)

This dataset is **not stored in the repository**. The files total ~900 MB and
exceed GitHub file size limits. Clone the project, then download data locally.

## Quick setup

```bash
pip install -e .

# Show download instructions
python3 -m chebnet_kaist.cli.setup_data --instructions

# After placing files, verify layout
python3 -m chebnet_kaist.cli.setup_data --verify

# If you have normal_run_data.7z but not the txt file
python3 -m chebnet_kaist.cli.setup_data --extract-normal
```

## Official download

**Source:** [HCRL Car-Hacking Dataset](https://ocslab.hksecurity.net/Datasets/car-hacking-dataset)
(Korea University, OTIDS project)

Download all attack CSV files and the normal-run archive from the official page.
Academic use — please cite the original papers listed on that site.

## Expected directory layout

```
data/kaist/
  Car-Hacking Dataset/
    DoS_dataset.csv
    Fuzzy_dataset.csv
    RPM_dataset.csv
    gear_dataset.csv
    normal_run_data/
      normal_run_data.txt
    normal_run_data.7z          # optional archive; extract with setup_data
```

## Copy from an existing local copy

If you already have the dataset elsewhere (old clone, USB, lab server):

```bash
python3 -m chebnet_kaist.cli.setup_data --copy-from /path/to/folder --verify
```

The source folder may contain `Car-Hacking Dataset/` or the CSV files directly.

## File formats

Attack CSV (`DoS_dataset.csv`, etc.):

```
Timestamp, CAN_ID, DLC, D0, D1, D2, D3, D4, D5, D6, D7, Flag
1478198376.389427, 0316, 8, 05, 21, 68, 09, 21, 21, 00, 6f, R
1478198377.185137, 0000, 8, 00, 00, 00, 00, 00, 00, 00, 00, T
```

Flag: `R` = normal frame in log, `T` = injected attack frame.

Normal TXT (`normal_run_data.txt`):

```
Timestamp: 1479121434.850202  ID: 0350  000  DLC: 8  05 28 84 66 6d 00 00 a2
```

## Dataset statistics (reference)

| Class  | File               | Total frames | R (normal) | T (attack) |
|--------|--------------------|-------------:|-----------:|-----------:|
| normal | normal_run_data.txt|      988,871 |    988,871 |          0 |
| dos    | DoS_dataset.csv    |    3,634,583 |  3,047,062 |    587,521 |
| fuzzy  | Fuzzy_dataset.csv  |    3,751,024 |  3,259,177 |    491,847 |
| rpm    | RPM_dataset.csv    |    4,580,226 |  3,925,329 |    654,897 |
| gear   | gear_dataset.csv   |    4,402,977 |  3,805,725 |    597,252 |

## Graph cache

After data is in place, training builds a graph cache at
`data/kaist/graphs_cache.pt` (also gitignored). Use `--ignore-cache` to rebuild.
