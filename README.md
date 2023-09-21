# MiniRAG

A proof-of-concept Retrieval Augmented Generation (RAG) bot using Llama2.

*Work in progress*

- [Setup](#setup)
  - [CPU only](#cpu-only)
  - [GPU (with Cuda)](#gpu)
    - [GPU Issues](#gpu-issues)
  - [Modify `constants`](#modify-constants)
    - [Download LLM](#download-llm)
- [Usage](#usage)
- [Roadmap](#roadmap)


## Setup

- create and activate `virtual environment`
  ```
  # Windows
  python -m venv ragenv
  .\ragenv\Scripts\activate
  
  # Linux
  sudo apt install python3.10-venv
  python -m venv ragenv
  source ragenv/bin/activate
  ``` 

#### CPU only

- install `requirements`
  ```
  pip install -r requirements.txt
  ```

#### GPU

- install `CUDA` on your system according to [Official Docs](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)
  ```
  sudo apt install nvidia-cuda-toolkit
  ```

- install `LlamaCpp-Python` with GPU support
  ```
  # Windows
  set CMAKE_ARGS="-DLLAMA_CUBLAS=on"
  set FORCE_CMAKE=1
  
  # Linux
  CMAKE_ARGS="-DLLAMA_CUBLAS=on"
  FORCE_CMAKE=1

  pip install llama-cpp-python --no-cache-dir --verbose
  ```

- install `requirements`
  ```
  pip install -r requirements.txt
  ```
  
#### GPU Issues
Note: when setting up `LlamaCpp-Python` check `--verbose` flag to see if 
`BLAS=1` and GPU support is in fact enabled.</br> 
Setup may succeed even if there are errors with Cuda - then only CPU support will be enabled

[Failed GPU Setup, only CPU available](https://github.com/imartinez/privateGPT/issues/885#issuecomment-1646752174)
[Instructions for GPU Support in llama.cpp](https://github.com/oobabooga/text-generation-webui/discussions/1984)

### Modify `constants`

tl;dr: specify `model_path` (**required**), optional: adjust context length, 
gpu parameters

#### Download LLM

- you need access to [HuggingFace](https://huggingface.co/)
- we will use a *quantized* model in `GGUF` format
  - select a model repository from [TheBloke](https://huggingface.co/TheBloke)
  - select a model file, for example from [Llama-2-7b-Chat-GGUF](https://huggingface.co/TheBloke/Llama-2-7b-Chat-GGUF):
    - `llama-2-7b-chat.Q4_K_M.gguf` *medium, balanced quality - recommended*
    - different model files for different use cases (and hardware requirements)

- place the model file in the repo and **specify the path** to the model 
  file in `constants.py`

## Usage

- every time you run the file, the raw document gets chunked and embedded 
  first, which may take some time
- run the `main.py` and you will be prompted for input
  ```
  python main.py
  ```
- ideally, input a question which can be answered from the context of the 
  given documents


## Roadmap

**ToDo**s

- stream output
  - only print the relevant messages, not verbose
- separate embedding from query function
  - **do not load model twice per run!**
  - embed once, save & load `chromadb` instance
  - keep persistent memory
    - do not embed same documents multiple times
    - only new documents / changes in existing documents
- add other document formats and loaders (PDF, csv, images?)
  - use different `text_splitter`
    - try different `chunk_size`, `chunk_overlap`
- keep chat history as context
- test performance with different settings
  - different models
    - parameter size 7b/13b, base/chat/uncensored
  - different prompts
    - different models require different templates
      - see corresponding HuggingFace page for model
