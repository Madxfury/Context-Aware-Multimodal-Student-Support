# Context-Aware Multimodal Student Support

**Samvya** is a student wellness research prototype combining camera-based facial emotion cues, conversation context and optional wearable features with an AI counselling interface.

## Current features

- Profile setup and returning-user selection.
- Camera check-in with an optional Swin facial-emotion checkpoint and attention visualisation.
- Groq-generated replies, ElevenLabs speech output and Sarvam speech transcription.
- SQLite session history, recent-session context and shareable session report pages with browser print/PDF output.
- Rule-based NLP, intervention selection and response guardrails.
- Optional physiological BiLSTM inference with per-session calibration, plus explicitly labelled simulated wearable data for development.

## Run locally

Use Python 3.10 or 3.11. The runtime dependency versions below were checked with Python 3.10.11.

```bash
git clone https://github.com/Madxfury/Context-Aware-Multimodal-Student-Support.git
cd Context-Aware-Multimodal-Student-Support
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
python app.py
```

Open http://127.0.0.1:5002. On Windows, activate with `.venv\Scripts\activate`.

Add your own `GROQ_API_KEY`, `ELEVENLABS_API_KEY` and `SARVAM_API_KEY` to `.env`. Chat generation, spoken playback and transcription respectively require these providers. Never commit keys. Without a Groq key, chat returns a configuration message rather than a generated reply.

### Optional trained models

Model checkpoints and training datasets are not distributed in this repository. Obtain the team's trained checkpoints separately and configure their paths in `.env`:

```dotenv
EMOTION_MODEL_PATH=/absolute/path/to/Swin_FANE_Best_Model.pth
PHYSIOLOGICAL_MODEL_PATH=/absolute/path/to/best_bilstm.pth
```

The default locations are `Models/Swin_FANE_Best_Model.pth` and `Models/Physiological_V2/best_bilstm.pth`. Use checkpoints matching the architectures in `app.py` and `services/physiological_model.py`.

The website, profiles and report routes start without the weights. Camera inference returns HTTP 503 when its checkpoint is missing. Physiological inference reports `model_available: false` instead of producing a prediction. `/api/health` reports which models are available. Do not substitute randomly initialised weights for trained checkpoints.

The local database is created automatically at `data/counsellor.db`. Set `DATABASE_PATH` to use another location. Session databases and exported personal records are excluded from Git.

## Verification

```bash
python -m pip install -r requirements-dev.txt
python -m pytest tests/test_observation.py tests/test_step6_7.py tests/test_step8.py tests/test_step10a_simulator.py tests/test_step10c_debug.py tests/test_step11_nlp.py tests/test_step12_guardrails.py tests/test_session_reports.py
node --check static/js/main.js
```

Additional inference/calibration tests (`test_step5.py`, `test_step8_inference.py`, `test_step9_pipeline.py`, `test_step10b_end_to_end.py`) require the trained physiological checkpoint. Live provider calls and model accuracy need separate validation with the corresponding keys and checkpoints.

## Repository layout

- `app.py`: Flask routes, vision inference and provider integration.
- `services/`, `schemas/`: persistence, contextual state, NLP and inference services.
- `templates/`, `static/`: website and printable session reports.
- `tests/`: service and application checks.
- `scripts/`, `utilities/`: research/training utilities. These may require extra packages listed in `requirements-training.txt`, datasets and local paths. That file preserves the upstream historical experiment environment and is separate from the app runtime.

## Scope

This is a local research prototype, not a diagnostic instrument or a substitute for professional care. Emotion labels and wellness scores are experimental outputs. The history and report routes currently have no authentication or access controls, so keep the server local and use demonstration profiles. Simulated wearable readings must not be presented as measurements from the physical hardware.

## Source and attribution

Application snapshot imported from [Taherhotel/Context-aware-Adaptive-counsellor](https://github.com/Taherhotel/Context-aware-Adaptive-counsellor), commit `690b140f0b62f00f91dd678f7856c6e0857866b4`, retaining its MIT license. The import excludes committed credentials, participant databases, generated logs, notebook checkpoints and the facial-image training dataset. Runtime/setup corrections and report regression checks are maintained in this repository. The earlier upstream README described a different elderly-care project and is replaced here with instructions matching Samvya.
