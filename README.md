# Option Pricing Toolkit GUI

This package contains only a Streamlit GUI. It is designed to import your original, unchanged code.

## Files

- `app.py`: the Streamlit interface
- `requirements.txt`: required packages
- `option_pricing.py`: Implementations of used classes, methods and functions

## Preview locally

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Then run:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open `http://localhost:8501`. Running locally does not publish the application online.
