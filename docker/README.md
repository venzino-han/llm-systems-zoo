# docker/

`Dockerfile` builds on the official `vllm/vllm-openai` image (which pins a working torch/CUDA/vLLM
combination) and adds the training, Ray and application extras.

**Status: not validated.** The published pilot results were produced in a native environment
(`docs/hardware.md`). Building and running the full GPU test suite inside this image is on the roadmap;
until then, treat the Dockerfile as a starting point, not a reproducibility guarantee.

Record the image digest in the run manifest (`software.container_digest`) when a run is executed in
a container.
