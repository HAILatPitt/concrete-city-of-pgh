# ConCReTE: Forecasting the City of Pittsburgh's Revenue

A [marimo](https://marimo.io) notebook in which students forecast the city's quarterly revenue from figures that were extracted from PDF reports six different ways: by hand, Docling, ChatGPT, Claude chat, the Claude API and pypdf. It follows ConCReTE's Predict → Observe → Explain → Evaluate structure:

- **Predict:** commit to which extraction to trust, before seeing the data.
- **Observe:** dig into the data in five code cells, build the forecast, and ask an AI helper.
- **Explain:** write a reflection and a note to the city's Chief Data Officer.
- **Evaluate:** see which choices hold up and what the data was hiding.

The fastest way in is the browser version, which runs entirely in your browser:

https://hailatpitt.github.io/concrete-city-of-pgh/

It takes a few seconds to load.

## The AI helper runs on your own model on Pitt's CRCD cluster

The helper uses an open-weight model (Qwen2.5-32B-Instruct, 4-bit) that each student starts on CRCD with llama.cpp. The notebook shows every step at the top. In short:

You use two terminal windows on your laptop: **Terminal A** logs in to CRCD and runs the model, **Terminal B** holds the tunnel.

1. You need a CRCD account with an allocation on the `gpu` cluster, and Wireless-PittNet or the Pitt VPN.
2. **Terminal A:** `ssh <pittID>@h2p.crc.pitt.edu`, then start a GPU session: `srun -M gpu -p l40s -n1 -c16 --gres=gpu:1 -t 2:00:00 --pty bash`. The prompt changes to a node such as `gpu-n80`; steps 3 and 4 run there.
3. **Terminal A:** set `LLAMACPP_MODEL="bartowski/Qwen2.5-32B-Instruct-GGUF:Q4_K_M"`, `LLAMACPP_DOWNLOAD_DIR="/vast/<group>/$USER/llamacpp"` and `LLAMACPP_EXTRA_ARGS="-c 16384"`, then `module load llamacpp/0.5.0`. Wait for `llamacpp-status` to print `READY`.
4. **Terminal A:** print your tunnel command with `echo "ssh -N -L 8080:${LLAMACPP_BASE_URL#http://} $USER@h2p.crc.pitt.edu"` and copy the line it prints. Typing it yourself? The node name includes `gpu-` (`gpu-n80`, not `n80`).
5. **Terminal B** (on your laptop, not logged in to CRCD): paste that line and leave the window open.
6. In the notebook, keep `http://localhost:8080`, click **Test connection**, and allow the browser's local-network prompt.

CRC's guides: [llama.cpp](https://crc-pages.pitt.edu/user-manual/applications/ai-ml/llamacpp/) · [Inference servers on the GPU cluster](https://crc-pages.pitt.edu/user-manual/applications/ai-ml/inference-server/)

You can do the whole data task without the helper.

## Running on your own machine

This uses [`uv`](https://docs.astral.sh/uv/). Running locally is also the fallback if your browser refuses to let the web version reach the tunnel.

- Run mode (as students see it): `uv run marimo run concrete-model-building.py`
- Edit mode: `uv run marimo edit concrete-model-building.py`

## Publishing the web version

GitHub Pages serves the `docs/` folder on `main`. After changing the notebook, rebuild `docs/` and commit it:

```
uv run marimo export html-wasm concrete-model-building.py -o docs --mode run --no-show-code
cp -r public docs/
touch docs/.nojekyll
```
