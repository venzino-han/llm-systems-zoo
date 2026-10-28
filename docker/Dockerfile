# Reference container for the GPU labs.
# Status: NOT yet validated end-to-end (see docker/README.md). CPU tests run in CI without it.
#
# Build:  docker build -t llm-systems-zoo -f docker/Dockerfile .
# Run:    docker run --gpus all -it --rm -v $PWD:/workspace -v ~/.cache/huggingface:/root/.cache/huggingface llm-systems-zoo
FROM vllm/vllm-openai:latest

WORKDIR /workspace
COPY pyproject.toml README.md ./
COPY llm_systems_zoo ./llm_systems_zoo
# Track extras that do not conflict with the vLLM image's pinned torch.
RUN pip install --no-cache-dir -e ".[dev]" peft trl accelerate "ray[default,data,train,serve]" \
        sentence-transformers faiss-cpu datasets
COPY . .
ENTRYPOINT ["/bin/bash"]
