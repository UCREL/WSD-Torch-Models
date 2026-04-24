# Resource Benchmarking


This directory contains all of the scripts required to run all the different taggers to benchmark them for;
* Memory usage
    * RAM
    * GPU - when used
* Speed - Tokens per second

We benchmark the following taggers:
* `ucrelnlp/PyMUSAS-Neural-English-Small-BEM`
* `ucrelnlp/PyMUSAS-Neural-English-Base-BEM`
* `ucrelnlp/PyMUSAS-Neural-Multilingual-Small-BEM`
* `ucrelnlp/PyMUSAS-Neural-Multilingual-Base-BEM`

These benchmarks can be used through the [results section](#current-results) to determine if any changes to the code base have improved speed and memory performance on either CPU or GPU.

## How to run the benchmark

**NOTE** currently this is setup to use `torch` with `cuda=128`, if you would like to use a different `torch` change how `uv` is used in [./run_benchmark.sh](./run_benchmark.sh) and [./download_models.sh](./download_models.sh).

We assume you have the development setup installed following these [instructions in the main README](../../README.md#setup).

And then you will need to download the neural tagger models like so (this can take a bit of time and will download up to 5GB of models to your disk via `huggingface` so it is likely to be installed to your `$HF_HOME` which is normally `$HOME/.cache/huggingface`);

``` bash
bash ./download_models.sh
```

To run all of the benchmarks on CPU (this will take between 5-30 minutes to run);

``` bash
bash ./run_benchmark.sh --device cpu
```

This will then produce on stdout the following MarkDown table;

<details>
<summary>Example Benchmark Markdown table</summary>

| Language | Tagger | Load Model Memory Requirements | Average Memory Requirements | Large Text Memory Requirements | Tokens Per Second | Number of Tokens Processed | Large Text Tokens Processed | Load Model GPU Memory Requirements | Average GPU Memory Requirements | Large Text GPU Memory Requirements |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| English| Neural-E-17M| 198.89| 227.82| 402.59| 350.96| 1,423| 1,154| 0.00| 0.00| 0.00 |
| English| Neural-E-68M| 381.00| 210.71| 10.80| 3,361.17| 1,423| 1,154| 0.00| 0.00| 0.00 |
| Multilingual| Neural-M-140M| 1,020.21| 1,032.24| 1,284.75| 2,999.71| 1,423| 1,154| 0.00| 0.00| 0.00 |
| Multilingual| Neural-M-304M| 1,575.93| 1,613.13| 1,630.09| 638.62| 1,423| 1,154| 0.00| 0.00| 0.00 |
</details>

To run on GPU (you will need a Nvidia GPU), this first requires building the following docker container (this is best done in a terminal outside of your editor/IDE);

``` bash
docker build -t wsd-torch-gpu-benchmarking:0.1.0 -f ./Dockerfile ..
```

And then you can run the benchmarks on GPU like so (you can also use this to run the CPU benchmarks as well)

``` bash
docker run --rm --gpus all --shm-size 4g wsd-torch-gpu-benchmarking:0.1.0 ARGUMENTS TO THE `run_benchmarks.sh` SCRIPT
```

Example;

``` bash
docker run --rm --gpus all --shm-size 4g wsd-torch-gpu-benchmarking:0.1.0 --device cuda --token-limit 100000 --large-text-token-limit 1500
```

### Current results

These are the current results on `cuda` and `cpu` respectively when running on a `Intel Core Ultra 7 265F with 20 cores and threads`, `64GB` DDR5 RAM, `2TB` of PCIe Gen 5 SSD, and `16GB` `Nvidia 5070ti` GPU using `CUDA 12.8`:

CPU:
``` bash
bash run_benchmark.sh -d cuda -t 100000 -l 1500

| Language | Tagger | Load Model Memory Requirements | Average Memory Requirements | Large Text Memory Requirements | Tokens Per Second | Number of Tokens Processed | Large Text Tokens Processed | Load Model GPU Memory Requirements | Average GPU Memory Requirements | Large Text GPU Memory Requirements |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| English| Neural-E-17M| 205.54| 1,008.28| 190.11| 5,718.34| 100,511| 1,620| 65.11| 235.93| 3,556.12 |
| English| Neural-E-68M| 413.31| 1,231.05| 388.45| 2,676.74| 100,511| 1,620| 264.02| 594.56| 7,196.18 |
| Multilingual| Neural-M-140M| 1,033.38| 1,222.97| 997.85| 2,235.65| 100,511| 1,620| 537.28| 827.67| 5,840.12 |
| Multilingual| Neural-M-304M| 942.51| 1,298.80| 894.18| 2,345.30| 100,511| 1,620| 1,190.79| 1,761.80| 11,745.87 |
```

GPU:
``` bash
docker run --rm --gpus all --shm-size 4g wsd-torch-gpu-benchmarking:0.1.0 --device cuda --token-limit 100000 --large-text-token-limit 1500

| Language | Tagger | Load Model Memory Requirements | Average Memory Requirements | Large Text Memory Requirements | Tokens Per Second | Number of Tokens Processed | Large Text Tokens Processed | Load Model GPU Memory Requirements | Average GPU Memory Requirements | Large Text GPU Memory Requirements |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| English| Neural-E-17M| 190.52| 1,134.17| 174.12| 5,396.30| 100,511| 1,620| 65.11| 235.93| 3,556.12 |
| English| Neural-E-68M| 2.65| 711.94| -16.82| 2,696.52| 100,511| 1,620| 264.02| 594.56| 7,196.18 |
| Multilingual| Neural-M-140M| 697.93| 1,809.50| 642.09| 2,215.45| 100,511| 1,620| 537.28| 827.67| 5,840.12 |
| Multilingual| Neural-M-304M| 476.82| 822.03| 455.48| 2,313.53| 100,511| 1,620| 1,190.79| 1,761.80| 11,745.87 |
```

## How to interpret the benchmark results

All of the benchmarking uses Wikipedia texts, specifically from the [HuggingFaceFW/finewiki dataset repository](https://huggingface.co/datasets/HuggingFaceFW/finewiki), we used Wikipedia as it has an open license and covers a lot of languages that our taggers support. All of the memory statistics are in Mega Bytes (MB).

Here we detail the meaning of each header in the markdown table;

* Language - The language of the Wikipedia data the tagger was processing.
* Tagger - The name of tagger type.
* Load Model Memory Requirements - The RAM/memory requirements to load the model.
* Average Memory Requirements - The RAM/memory requirements to load and run the model on the Wikipedia texts separated by new lines (not sentences therefore will be made up of multiple sentences, typically these represent paragraphs).
* Large Text Memory Requirements - The RAM/memory requirements to load and run the model on a single Wikipedia text that has been joined together from multiple Wikipedia texts to ensure the text is at least `--large-text-token-limit`.
* Tokens Per Second - Number of tokens the tagger processed per second.
* Number of Tokens Processed - Number of tokens processed to generate the `Tokens Per Second` metric, these tokens are from processing the Wikipedia texts separated by new lines, of which this is the same data that is used for `Average Memory Requirements`.
* Large Text Tokens Processed - The length in tokens of the large text that was processed to generate `Large Text Memory Requirements` metric.
* Load Model GPU Memory Requirements - The VRAM/GPU memory requirements to load the model.
* Average GPU Memory Requirements - The VRAM/GPU memory requirements to load and run the model on the Wikipedia texts separated by new lines (not sentences therefore will be made up of multiple sentences, typically these represent paragraphs).
* Large Text GPU Memory Requirements - The VRAM/GPU memory requirements to load and run the model on a single Wikipedia text that has been joined together from multiple Wikipedia texts to ensure the text is at least `--large-text-token-limit`.

**Note** the RAM/memory requirements are only estimates, but are a good guide. The reason they are only estimates as we cannot get the peak memory usage but rather the memory usage before and after a process has been completed, to get memory usage during the tagging process this would require running an external memory profiler, like [Scalene](https://github.com/plasma-umass/scalene) which we did not do here as it is difficult to get the memory requirement programmatically. For more accurate estimates you could run the [Scalene](https://github.com/plasma-umass/scalene) profile on an individual tagger benchmarking script, e.g. `scalene run benchmark_rule_based_tagger.py` (once you have installed `scalene`).

## Brief description of the benchmarking scripts

All of the scripts come with a `--help` guide if you want to know more about a specific script;

* `benchmark_model.py` -- Used to benchmark the neural tagger.
* `benchmarking_utils.py` -- NOT A SCRIPT but a module used by `benchmark_model.py`.
* `format_benchmarking_data.py` -- Formats the output generated from benchmarking script (`benchmark_model.py`) into a markdown table that is used to display the benchmarking results.
* `run_benchmarks.sh` -- A BASH script that calls the `benchmark_model.py` Python script to benchmark all of the neural taggers, and then calls the `format_benchmarking_data.py` script to format the generated benchmarking results.
* `download_models.sh` -- downloads the neural tagger models required for benchmarking.