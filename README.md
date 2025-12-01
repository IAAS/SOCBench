# SOCBench

The SOCBench contains the artifacts of SOCBench-D and SOCBench-SC.
SOCBench-D is a service discovery benchmark for natural language queries and OpenAPI specifications to expected endpoints.
SOCBench-SC is a static code analysis tool to extract invoked endpoints from a given Python code.
Together, they can be used to benchmark the Python code generation capabilities for automated service composition of LLMs by inputting the SOCBench-D benchmark files and analyzing the resulting Python code using SOCBench-SC.

## SOCBench-D

Source code and benchmark files of the SOCBench-D service discovery benchmark.

`socbench-d/code`: Source code to generate benchmark files.

`socbench-d/benchmark`: Benchmark files.

Reproducibility dataset: <https://dx.doi.org/10.21227/vdm4-k186>

Citation:
```
R. D. Pesl, J. G. Mathew, M. Mecella, and M. Aiello, “Retrieval-augmented generation for service discovery: Chunking strategies and benchmarking,” 2025. [Online]. Available: https://arxiv.org/abs/2505.19310
```

## SOCBench-SC

`socbenchsc/`: Python package for the static code analysis.

Reproducibility dataset: <https://dx.doi.org/10.21227/kzhg-ss64>

Citation
```
TBD
```
