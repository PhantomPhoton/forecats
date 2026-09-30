Forecats
=========
Wake up every morning (more) excited to look at the forecast! This is a custom Home Assistant integration which uses forecast data, pictures of your pets, and AI image providers to generate and serve weather-themed drawings of your precious babies every morning.

Blog post [here](https://secondthoughts.my/posts/projects/forecats/).

![A picture of two cats and a snowy, cloudy forecast combine to make a drawing of two cats in scarves looking out a window](static/img/diagram.png)

**Features**
- Home Assistant integration with a config UI for providers and credentials
- Multiple Forecats instances (e.g. cloud Gemini + local InvokeAI) with separate output folders
- Per-stage provider selection (activity text vs image generation)
- Gemini, OpenAI-compatible chat, InvokeAI, and Unsloth Studio backends
- Customizable automation via the `forecats.generate_pet_picture` action (schedule + weather entity + pets/styles)
- Optional optimization for E-ink screens (currently limited to spectra6 panels)

## Examples

You can display it on an e-ink screen

<img src="static/img/e-ink-screen.jpg" alt="A cat in front of an e-ink screen" width="400">

Or on your HA Dashboard

<img src="static/img/screenshot.png" alt="Home Assistant app screenshot" width="400">


Works in better weather

<img src="static/img/example4.png" alt="Two cats with sunglasses" width="400">

## Getting Started

> [!WARNING]
> Cloud Gemini usage costs money after free credits. Local OpenAI-compatible chat (Ollama / LM Studio / …), InvokeAI, or Unsloth Studio avoids API cost but needs a machine that can run the models (CPU works, but image jobs can take a long time).

### Requirements
- A [home assistant server](https://www.home-assistant.io/installation/) with the following add-ons
  - [File editor add-on](https://www.home-assistant.io/common-tasks/os/#installing-and-using-the-file-editor-add-on)
  - [Terminal and ssh add-on](https://www.home-assistant.io/common-tasks/os/#installing-and-using-the-ssh-add-on)
  - [Meteorologisk institutt (Met.no)](https://www.home-assistant.io/integrations/met/) integration
- For **Gemini**: a Google AI Studio [API key](https://aistudio.google.com/api-keys)
- For **OpenAI-compatible** activity: a reachable chat server (`http://host:11434/v1` Ollama, `http://host:1234` LM Studio, Unsloth, vLLM, …) exposing `/v1/chat/completions` and `/v1/models`
- For **InvokeAI**: a reachable InvokeAI server (`http://host:9090`) with a **Flux Kontext** (or FLUX.2) main model plus T5/CLIP/VAE companions (auto-discovered when installed); optional Text LLM for activity; CPU-supported; expect long image timeouts
- For **Unsloth Studio**: a reachable Studio server (`http://host:8888`) with an API key from Settings → API; chat model for activity and a **FLUX.2** reference diffusion model for images (up to 4 photos). The config UI lists hub-cached / recommended models (GGUF included)

### Setup

#### Option A: Install via HACS (recommended)

1. Install [HACS](https://hacs.xyz/) if you haven't already.
2. In HACS, click **Custom repositories** and add `https://github.com/jwardbond/forecats` with category **Integration**.
3. Search for "Forecats" in HACS and install it.
4. Restart Home Assistant, then continue from **Configure Forecats** below.

#### Option B: Manual install

*Do the following in your HA server, using the Terminal & SSH addon, or `docker exec` if you are running a container on a host system*

1. **Create the necessary directory structure** in your Home Assistant server:

  ```bash
  mkdir -p /config/custom_components && mkdir -p /config/forecats_data/input_images
  ```

2. **Download and copy the integration files**

  ```bash
  cd /tmp && git clone https://github.com/jwardbond/forecats.git && cp -r forecats/custom_components/forecats /config/custom_components/
  ```

#### Configure Forecats (config flow)

1. Go to **Settings → Devices & Services → Add Integration → Forecats**.
2. Create an instance:
   - **Display name** (e.g. `Cloud Gemini` or `Local CPU`)
3. Pick **activity** (`gemini`, `openai`, `invokeai`, or `unsloth`) and **image** (`gemini`, `invokeai`, or `unsloth`) providers independently.
4. Enter only the credentials those providers need (Gemini API key, OpenAI-compatible URL/key, InvokeAI URL/token, Unsloth URL/API key).
5. Set model IDs and **per-provider timeouts** (seconds). InvokeAI image default is 3600s for CPU; Unsloth image default is 1800s.
6. Repeat **Add Integration** to create a second instance (e.g. local) while keeping cloud running.

You no longer put API keys in the automation.

#### Upload pet images

**Option 1 — Media Browser (recommended)**
- In HA, go to **Media > Local Media** and create a folder called `forecats`.
- Upload your pet photos directly from the browser.
- In the `forecats.generate_pet_picture` action, use **Input images** to pick that folder and/or individual photos (multiple allowed).

**Option 2 — Explicit media / paths**
- You can still pass a filesystem path string for `input_image_dir` in YAML, or mix media items and paths.
- Choose clear photos of your pets and rename the files so the pets' names are in the filenames.

#### Automation

Create an automation (UI or YAML) that runs on a schedule and calls **forecats.generate_pet_picture**.

**UI**
1. **Settings → Automations & Scenes → Create automation**
2. Trigger: time (e.g. `05:00:00`)
3. Action: **Forecats: Generate Pet Picture**
4. Select the Forecats instance, weather entity, location, pets, input images, art styles, final size, **output directory**, and display profile

**YAML**
- Copy the [automation template](config_examples/automation_fragment.yaml) into `config/automations.yaml` (or create via UI and switch to YAML).
- Fill out or remove any `<>` placeholders.
- `config_entry_id` is required (find it under the Forecats integration device/entry).

The action fetches today's forecast from the required `weather_entity` (no separate `weather.get_forecasts` step).

Restart Home Assistant after install.

**Outputs** (set via **Output directory** on the action; default `/config/www/daily_forecats`):
- `forecats_original.png`
- `forecats_optimized.png`

Reachable at e.g. `<YOUR HA URL>/local/daily_forecats/forecats_optimized.png`.

>[!Note]
> Cloud Gemini often finishes in 10–30 seconds. Local InvokeAI on CPU can take much longer — schedule e-ink / follow-up automations after your configured image timeout.

> [!Note]
> To test: **Developer tools → Actions → forecats.generate_pet_picture**.

### Providers

| Stage | Gemini | OpenAI-compatible | InvokeAI | Unsloth Studio |
| --- | --- | --- | --- | --- |
| Activity text | Gemini LLM | `/v1/chat/completions` (Ollama, LM Studio, …) | Text LLM / expand-prompt | OpenAI-compatible `/v1/chat/completions` |
| Image | Nano Banana (strong reference-image likeness) | — (activity only) | Flux / Flux Kontext graph + uploaded refs (CPU OK, slow) | FLUX.2 reference workflow (up to 4 refs) via Studio API |

**Gemini (cloud)** — Best pet likeness today. Image models show as Nano Banana names in the UI:
- `gemini-3-pro-image-preview` — Nano Banana Pro
- `gemini-3.1-flash-image` — Nano Banana 2
- `gemini-2.5-flash-image` — Nano Banana

**OpenAI-compatible (activity only)** — Any server that speaks OpenAI chat: Ollama (`http://host:11434/v1`), LM Studio, Unsloth chat, vLLM, etc. Forecats lists models from `GET /v1/models` and generates activity via `POST /v1/chat/completions`. If you previously used the dedicated Ollama provider, reconfigure the instance to **openai** (old `ollama_*` fields are ignored).

**InvokeAI (local)** — Flux Kontext / FLUX.2 graphs with uploaded reference photos. Good on CPU with long timeouts; companion T5/CLIP/VAE models are auto-selected when installed.

**Unsloth Studio (local)** — Activity uses Studio’s OpenAI-compatible chat endpoint. Images use Studio’s diffusion API with up to **4** reference photos (`init_image` + extras). The config UI model dropdown mirrors Studio’s Images picker: hub inventory (local/cached models, including GGUF) plus a curated recommended list. Prefer a **FLUX.2** reference model (e.g. `unsloth/FLUX.2-klein-4B`) for multi-photo likeness; edit-style models (Kontext / Qwen-Image-Edit) typically take one input. For GGUF quants, pick a loaded entry or set `repo_id::gguf_filename`.

Tip while tuning local: run **two instances** (Gemini + InvokeAI/Unsloth, or OpenAI-compatible activity + local image) and point each automation at a different **output directory**. Local reference models won’t match Gemini’s restyle/text fidelity — use them for offline / no-API-cost runs.

## Local Testing
I got annoyed testing out new prompts on HA, so I made a folder to experiment locally. If you would like to use it:

1. Clone the repo and enter the testing folder:
  ```bash
  git clone https://github.com/jwardbond/forecats.git && cd forecats/local_testing
  ```
2. Add your cat images to the `forecats_data/input_images` folder.
3. Copy `.env.example` to `.env` and set providers / keys / URLs / timeouts.
4. Copy the data from your automation into `test.py` if needed.
5. Run:
  ```bash
  uv run test.py
  ```

Outputs go to the directory set by `OUTPUT_DIR` (default `www/daily_forecats`).

## (Optional) Sending to an e-ink screen
You will need a screen controllable with ESPHOME. I used seeed studio's [e10002 spectra6 display](https://www.seeedstudio.com/reTerminal-E1002-p-6533.html). I've included the esphome config I use [here](https://github.com/jwardbond/forecats/blob/ha_integration/config_examples/seeede1002.yaml). The basic idea is to set the automation to run every day at a 5:00 am, and have the screen wake up every day slightly before that, download the picture after generation finishes (leave headroom for local CPU jobs), and then go into deep sleep until the next day.


## TODO
- [x] Enrol in HACS for easier install
- [ ] Option to save images to dir
- [x] Action-based automation (weather entity + pets/styles on `forecats.generate_pet_picture`)
- [ ] Separate e-ink instructions
- [x] Configurable providers via GUI (config flow)
- [ ] Support for multiple cities
