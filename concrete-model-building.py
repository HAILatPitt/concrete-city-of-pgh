import marimo

__generated_with = "0.18.0"
app = marimo.App(width="compact", app_title="ConCReTE Model Building")

with app.setup:
    import asyncio
    import functools
    import html
    import json
    import sys
    import textwrap
    import urllib.request

    import matplotlib.pyplot as plt
    import pandas as pd
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import root_mean_squared_error
    from sklearn.model_selection import train_test_split

    CRCD_STEPS = textwrap.dedent("""
        **This scenario includes an AI helper that runs on your own model on Pitt's CRCD GPU cluster.** Setting it up takes about 15 minutes the first time (mostly the one-time model download) and about 3 minutes after that. You can do the whole data task without it; only the helper needs it.

        **Before you begin**
        - A CRCD account with an allocation on the `gpu` cluster. No allocation yet? Request one at [crc.pitt.edu/service-request-forms](https://crc.pitt.edu/service-request-forms).
        - Be on **Wireless-PittNet** or connected to the **Pitt VPN**.
        - Use **Chrome, Edge or Firefox**.

        **1. Log in to CRCD and start a GPU session.** In a terminal: `ssh <pittID>@h2p.crc.pitt.edu`, then:
        ```
        srun -M gpu -p l40s -n1 -c16 --gres=gpu:1 -t 2:00:00 --pty bash
        ```

        **2. Choose the model and start the server.** Replace `<group>` with your CRCD group name:
        ```
        export LLAMACPP_MODEL="bartowski/Qwen2.5-32B-Instruct-GGUF:Q4_K_M"
        export LLAMACPP_DOWNLOAD_DIR="/vast/<group>/$USER/llamacpp"
        export LLAMACPP_EXTRA_ARGS="-c 16384"
        module load llamacpp/0.5.0
        ```
        This is a 32-billion-parameter model compressed to about 20 GB. The first download takes a while; later sessions reuse the folder.

        **3. Wait until it's ready.** Run `llamacpp-status` until it prints something like `READY - http://gpu-n80:27555`. The part after `http://` is your **node** (`gpu-n80`) and **port** (`27555`).

        **4. On your own laptop, open a tunnel.** In a *new* terminal window on your laptop (not on CRCD), using your node and port:
        ```
        ssh -N -L 8080:<node>:<port> <pittID>@h2p.crc.pitt.edu
        ```
        It will look like nothing happens. That's normal. Leave this window open the whole time you work.

        **5. Connect.** Keep the address below as `http://localhost:8080` and click **Test connection**. Your browser will ask whether this site may access other apps and devices on your network or computer. Click **Allow**. That lets this page talk to the tunnel.

        **6. When you're finished,** type `exit` in the CRCD session. The GPU is billed for every minute you hold it, whether or not you're using it.

        *Your model server has no password: other CRCD users could reach it while it runs, so stop it when you're done. This scenario only sends the city's public revenue data and your questions to it. Not working? You can also run this scenario on your laptop with `uv run marimo run concrete-model-building.py` (see the README).*

        Sources: CRC [llama.cpp guide](https://crc-pages.pitt.edu/user-manual/applications/ai-ml/llamacpp/) · CRC [Inference servers on the GPU cluster](https://crc-pages.pitt.edu/user-manual/applications/ai-ml/inference-server/)
    """)

    FONT_LINK = (
        "<link rel='stylesheet' href='https://fonts.googleapis.com/css2?family=Libre+Franklin:"
        "wght@400;500;600;700&family=Lora:ital,wght@0,400;0,500;0,600;1,400;1,500&display=swap'>"
    )
    # Pitt navy is the case's ink; Pitt gold marks what the learner adds. Cool hue-260 neutrals.
    PALETTE = (
        ".ss{--gold:#FFB81C;--navy:#003594;--navy-dk:#003163;--ink:oklch(0.24 0.012 260);"
        "--mid:oklch(0.47 0.010 260);--rule:oklch(0.86 0.006 260);--wash:oklch(0.975 0.004 260);"
        "--sans:'Libre Franklin','PT Sans',sans-serif;--serif:'Lora',Georgia,serif;"
        "color:var(--ink);font-family:var(--sans)}"
    )
    # Card faces render inside the marimo button's shadow root, so they carry their own styles.
    CARD_STYLE = (
        ".ss-card{box-sizing:border-box;width:224px;height:300px;overflow:hidden;background:#fff;"
        "border:1px solid var(--rule);border-radius:2px;display:flex;flex-direction:column;text-align:left;"
        "white-space:normal;font-weight:400;box-shadow:0 1px 0 var(--rule),0 8px 18px -14px rgba(0,49,99,.35);"
        "animation:ss-flip .4s cubic-bezier(.2,.7,.3,1) both}"
        ".ss-card:hover{border-color:var(--gold)}"
        ".ss-front-face{display:var(--ss-front,flex)}.ss-back-face{display:var(--ss-back,none)}"
        "@keyframes ss-flip{from{transform:rotateY(90deg);opacity:.4}to{transform:rotateY(0);opacity:1}}"
        ".ss-band,.ss-band-b{display:flex;align-items:center;justify-content:center;"
        "font:600 12px/1 var(--sans);letter-spacing:.16em;text-transform:uppercase}"
        ".ss-band{height:54px;background:var(--navy);color:#fff}"
        ".ss-band-b{height:40px;margin-top:auto;background:var(--gold);color:var(--navy-dk)}"
        ".ss-front-body{flex:1;display:flex;flex-direction:column;align-items:center;justify-content:center;"
        "text-align:center;padding:0 20px}"
        ".ss-numeral{font:italic 400 22px/1 var(--serif);color:var(--navy)}"
        ".ss-front-title{font:500 24px/1.15 var(--serif);color:var(--ink);margin-top:10px}"
        ".ss-hair{width:28px;height:2px;background:var(--gold);margin:14px auto 0}"
        ".ss-back{padding:18px 18px 14px}"
        ".ss-back-head{display:flex;align-items:baseline;justify-content:space-between;"
        "border-bottom:2px solid var(--navy);padding-bottom:7px}"
        ".ss-back-head b{font:500 16px/1.1 var(--serif);color:var(--ink)}"
        ".ss-back-head i{font:italic 400 15px/1 var(--serif);color:var(--navy)}"
        ".ss-back-text{display:block;font:400 13px/1.45 var(--sans);color:var(--ink);margin:9px 0 0;hyphens:auto}"
        ".ss-back-text::first-letter{font:500 30px/.8 var(--serif);float:left;color:var(--navy);margin:3px 4px 0 0}"
    )
    # marimo's button is a fixed-height inline-flex; centring it in a card-height flex slot lets the
    # card overflow it symmetrically and sit in normal flow. The card's -9px margin covers the button's
    # 7.2px side padding and 1px border. ponytail: tied to marimo 0.17.7's button box; re-check on upgrade.
    CARD_SLOT = {"display": "flex", "align-items": "center", "justify-content": "center", "height": "300px"}
    # Page-level blocks (light DOM), injected once by the title cell.
    BRIEF_STYLE = (
        ".ss-kicker{font:600 12px/1 var(--sans);letter-spacing:.14em;text-transform:uppercase;color:var(--navy)}"
        ".ss-title{font:500 42px/1.1 var(--serif);letter-spacing:-.01em;margin:10px 0 30px;color:var(--ink)}"
        ".ss-gate{margin:18px 0 0;text-align:center;font:italic 400 15px/1.4 var(--serif);color:var(--mid)}"
        ".ss-gate::before,.ss-gate::after{content:'';display:inline-block;width:36px;height:1px;"
        "background:var(--rule);vertical-align:middle;margin:0 12px}"
        ".ss-stage{margin-top:40px}"
        ".ss-runhead{display:flex;justify-content:space-between;font:600 12px/1 var(--sans);letter-spacing:.14em;"
        "text-transform:uppercase;color:var(--mid);border-top:2px solid var(--navy-dk);padding-top:9px}"
        ".ss-runhead span:first-child{color:var(--navy)}"
        ".ss-stage-title{font:500 34px/1.1 var(--serif);margin:18px 0 0;padding:0;border:0;color:var(--ink)}"
        ".ss-stage-title em{font-style:italic;color:var(--navy)}"
        ".ss-steps{display:flex;gap:18px;margin-top:12px;font:400 13px/1 var(--sans);color:var(--mid)}"
        ".ss-step.now{color:var(--ink);font-weight:600;border-bottom:2px solid var(--gold);padding-bottom:3px}"
        ".ss-prompt{margin-top:22px;background:var(--wash);border-top:1px solid var(--rule);"
        "border-bottom:1px solid var(--rule);padding:18px 22px;font:400 16px/1.6 var(--serif);color:var(--ink)}"
        ".ss-prompt .n{font-style:italic;color:var(--navy)}"
        ".ss-prompt strong{font-weight:600}"
        ".ss-prompt ul,.ss-prompt ol{margin:10px 0 12px 22px;padding:0;list-style:disc}"
        ".ss-prompt li{margin:5px 0;padding-left:2px}"
        ".ss-label{display:grid;grid-template-columns:34px 1fr;align-items:baseline;margin:22px 0 8px}"
        ".ss-label i{font:italic 400 20px/1 var(--serif);color:var(--navy)}"
        ".ss-label span{font:600 14px/1.35 var(--sans);color:var(--ink)}"
        ".ss-indent{padding-left:34px}"
        ".ss-locked{margin-top:22px;background:var(--wash);border-top:2px solid var(--navy-dk);"
        "border-bottom:1px solid var(--rule);padding:14px 22px 18px}"
        ".ss-locked-head{font:600 12px/1 var(--sans);letter-spacing:.14em;text-transform:uppercase;color:var(--navy)}"
        ".ss-locked h4{font:italic 500 16px/1.3 var(--serif);color:var(--ink);margin:14px 0 4px}"
        ".ss-locked p{font:400 15px/1.55 var(--sans);margin:0}"
        ".ss-task-title{font:500 24px/1.2 var(--serif);color:var(--ink);margin:34px 0 12px;padding:0;border:0}"
        ".ss-brief{background:var(--wash);border-top:1px solid var(--rule);border-bottom:1px solid var(--rule);"
        "padding:16px 22px 18px;font:400 14px/1.6 var(--sans);color:var(--ink)}"
        ".ss-brief p{margin:0 0 10px}"
        ".ss-brief table{width:100%;border-collapse:collapse;margin:6px 0 0;font-size:13.5px}"
        ".ss-brief th,.ss-brief td{text-align:left;vertical-align:top;padding:7px 10px 7px 0;"
        "border-bottom:1px solid var(--rule)}"
        ".ss-brief th{font:600 11px/1 var(--sans);letter-spacing:.12em;text-transform:uppercase;color:var(--navy);"
        "border-bottom:2px solid var(--navy)}"
        ".ss-brief td:first-child{white-space:nowrap;padding-right:18px}"
        ".ss-note{margin:16px 0 0;background:#fff8e6;border-top:2px solid var(--gold);padding:12px 18px;"
        "font:400 14px/1.55 var(--sans);color:var(--ink)}"
        ".ss-note.ok{background:#ecfdf5;border-top-color:#15803d}"
        ".ss-sheet{width:100%;border-collapse:collapse;font:400 13.5px/1.45 var(--sans);color:var(--ink)}"
        ".ss-sheet th{text-align:left;font:600 11px/1 var(--sans);letter-spacing:.12em;text-transform:uppercase;"
        "color:var(--navy);border-bottom:2px solid var(--navy);padding:8px 10px 8px 0}"
        ".ss-sheet td{text-align:left;vertical-align:top;padding:7px 10px 7px 0;border-bottom:1px solid var(--rule)}"
        ".ss-sheet .grp td{font:600 12px/1 var(--sans);letter-spacing:.1em;text-transform:uppercase;color:var(--mid);"
        "padding-top:18px;border-bottom:1px solid var(--navy-dk)}"
        ".ss-sheet .inc td{background:#ecfdf5}"
        ".ss-sheet .muted{color:var(--mid);font-style:italic}"
    )
    STAGES = ("Predict", "Observe", "Explain", "Evaluate")
    BRIEF_NAME = "Revenue Forecast Brief"
    EVAL_STYLE = (
        "<style>.poee-eval .sum{background:var(--wash);border-top:2px solid var(--navy-dk);"
        "border-bottom:1px solid var(--rule);padding:16px 22px;margin:14px 0}"
        ".poee-eval .big{font:500 22px/1.3 var(--serif);color:var(--ink)}"
        ".poee-eval .bar{display:flex;height:8px;overflow:hidden;margin:12px 0}"
        ".poee-eval .legend,.poee-eval .chips{font:400 12.5px/1.6 var(--sans);color:var(--mid);margin-top:6px}"
        ".poee-eval .chip{display:inline-block;background:#fff;border:1px solid var(--rule);padding:1px 8px;margin:2px;"
        "color:var(--ink)}"
        ".poee-eval .ev{background:#fff;border:1px solid var(--rule);border-top:3px solid #d97706;padding:16px 20px;margin:14px 0}"
        ".poee-eval .ev.harm{border-top-color:#b91c1c}"
        ".poee-eval h3{font:500 19px/1.25 var(--serif);color:var(--ink);margin:0 0 12px;padding:0;border:0}"
        ".poee-eval .trail{display:flex;gap:8px;margin-bottom:10px}"
        ".poee-eval .trail div{flex:1;padding:8px 10px;font:400 13px/1.4 var(--sans);background:var(--wash)}"
        ".poee-eval .trail .ok{background:#ecfdf5}"
        ".poee-eval .k{display:block;font:600 10.5px/1.4 var(--sans);text-transform:uppercase;letter-spacing:.1em;"
        "color:var(--mid)}"
        ".poee-eval .also{font-size:11.5px;color:var(--mid)}"
        ".poee-eval .key{display:inline-block;width:10px;height:10px;margin:0 6px -1px 0}"
        ".poee-eval .why,.poee-eval .cost{font:400 14px/1.6 var(--sans);color:var(--ink);margin:8px 0}"
        ".poee-eval .cost{background:#fdf3f2;padding:10px 12px}"
        ".poee-eval .grp{margin:26px 0 4px;font:600 12px/1 var(--sans);text-transform:uppercase;letter-spacing:.14em;"
        "color:var(--navy)}</style>"
    )

    CARDS = {
        "story": {
            "numeral": "i",
            "title": "The story so far",
            "back": (
                "Pittsburgh's data office wants a forecast of the city's quarterly revenue. The numbers live in years "
                "of PDF revenue reports, and the team has already pulled them out six different ways: by hand, "
                "with software, and with AI."
            ),
            "full": (
                "The City of Pittsburgh is implementing a three-year data strategy under the leadership of Chris "
                "Belasco, Senior Manager of Digital Services and Chief Data Officer. This strategy positions data as "
                "a critical asset to enhance city operations, engage the community, and drive equity. Participants "
                "are tasked with analyzing and forecasting the city's revenue, which is critical for supporting "
                "Pittsburgh's long-term goals. Before this scenario, Chris's team identified a need to forecast the "
                "city's revenue, found the Q4 Revenue Summary Table in each Revenue Report PDF, and extracted the "
                "actual quarterly revenue totals six ways: manual extraction by a human, Docling, one-shot AI "
                "chatbots (ChatGPT and Claude), the Claude API, and pypdf."
            ),
        },
        "role": {
            "numeral": "ii",
            "title": "Your role",
            "back": (
                "You're an intern for Chris Belasco, the city's Chief Data Officer. You decide which extracted "
                "numbers to trust, then build a simple forecast from them."
            ),
            "full": (
                "You are an intern working for Chris, and he has asked you to help with this next stage of the "
                "project. You have a dataset of the city's quarterly revenue from 2012 to 2024, with one column for "
                "each of the six extraction methods."
            ),
        },
        "deliver": {
            "numeral": "iii",
            "title": "What you deliver",
            "back": (
                "A forecast of the city's revenue for the next quarter, and a short note to Chris saying how "
                "confident he should be in it, and why."
            ),
            "full": (
                "Use the historical quarterly revenue data to build a predictive model for the city's revenue. "
                "Beyond an accurate model, consider what the most appropriate way to extract the data is, and be "
                "ready to explain to Chris how much to trust the result."
            ),
        },
    }

    METHOD_LABELS = {
        "revenue by hand": "By hand (a person typed them in)",
        "revenue by docling": "Docling (document-parsing software)",
        "revenue by chatgpt": "ChatGPT (one-shot AI chat)",
        "revenue by claude chat": "Claude chat (one-shot AI chat)",
        "revenue by claude api": "Claude API (automated AI pipeline)",
        "revenue by pypdf": "pypdf (a PDF text-extraction script)",
        "none": "None of them",
    }

    PREDICT_GUIDING = [
        "What previous decisions might have been made that impact your work here? Would your answer change if "
        "your role was different?",
        "Before selecting from one of the 6 options below, consider each option carefully. Do any of them raise "
        "technical or ethical concerns?",
        "What could go wrong when software or AI reads numbers from a PDF table? Which mistakes would be hardest "
        "to notice?",
        "Would you trust a person typing the numbers by hand more? What mistakes do people make?",
    ]

    class ModelUnavailable(Exception):
        """The student's model server could not be reached or gave an unusable reply."""


@app.function
def revenue_columns(data) -> list[str]:
    return [c for c in data.columns if c.lower().startswith("revenue by")]


@app.function
def quarter_label(year, quarter) -> str:
    return f"{int(year)} {quarter}"


@app.function
def next_quarter_label(year, quarter) -> str:
    number = int(str(quarter).lstrip("Q"))
    return f"{int(year) + 1} Q1" if number == 4 else f"{int(year)} Q{number + 1}"


@app.function
def build_lagged(data, column: str, lag: int, drop_missing: bool):
    frame = pd.DataFrame({
        "quarter_label": [quarter_label(y, q) for y, q in zip(data["year"], data["quarter"])],
        "revenue": data[column].astype(float),
    })
    frame["lag"] = frame["revenue"].shift(lag)
    return frame.dropna(subset=["revenue", "lag"]).reset_index(drop=True) if drop_missing else frame


@app.function
def score_model(frame, train_share: float) -> dict:
    if frame[["revenue", "lag"]].isna().any().any():
        raise ValueError("The data still has missing values, so the model can't be trained.")
    cut = int(len(frame) * train_share)
    train, test = frame.iloc[:cut], frame.iloc[cut:]
    if len(train) < 2 or len(test) < 1:
        raise ValueError("The split leaves too few quarters to train or test on.")
    model = LinearRegression().fit(train[["lag"]], train["revenue"])
    predictions = model.predict(test[["lag"]])
    return {
        "rmse": root_mean_squared_error(test["revenue"], predictions),
        "test_mean": test["revenue"].mean(),
        "test_min": test["revenue"].min(),
        "test_max": test["revenue"].max(),
        "train_rows": len(train),
        "test_rows": len(test),
    }


@app.function
def forecast_inputs(data, column: str, lag: int) -> dict | None:
    """The next quarter after the column's last value, and the quarter it is predicted from."""
    last = data[column].last_valid_index()
    if last is None:
        return None
    source = last + 1 - lag
    return {
        "label": next_quarter_label(data["year"][last], data["quarter"][last]),
        "source_label": quarter_label(data["year"][source], data["quarter"][source]) if source >= 0 else None,
        "source_value": data[column][source] if source >= 0 else None,
    }


@app.function
def next_quarter_forecast(data, column: str, lag: int) -> dict | None:
    inputs = forecast_inputs(data, column, lag)
    if inputs is None or inputs["source_value"] is None or pd.isna(inputs["source_value"]):
        return None
    usable = build_lagged(data, column, lag, True)
    if len(usable) < 2:
        return None
    model = LinearRegression().fit(usable[["lag"]], usable["revenue"])
    value = float(model.predict(pd.DataFrame({"lag": [float(inputs["source_value"])]}))[0])
    return {
        "label": inputs["label"],
        "value": value,
        "reason": f"Predicted from {inputs['source_label']}, {lag} quarter(s) earlier, using every usable quarter.",
    }


@app.function
def forecast_unavailable_reason(data, column: str, lag: int) -> str:
    inputs = forecast_inputs(data, column, lag)
    if inputs is None:
        return "This column has no values at all."
    if inputs["source_label"] is None:
        return f"There is no quarter {lag} quarter(s) before {inputs['label']} to predict from."
    return (f"To forecast {inputs['label']} with a lag of {lag}, the model needs {inputs['source_label']}, "
            "which is missing in this column.")


@app.function
def random_split_rmse(data, column: str, lag: int) -> float:
    frame = build_lagged(data, column, lag, True)
    train, test = train_test_split(frame, test_size=0.2, random_state=0)
    model = LinearRegression().fit(train[["lag"]], train["revenue"])
    return root_mean_squared_error(test["revenue"], model.predict(test[["lag"]]))


@app.function
def chronological_rmse(data, column: str, lag: int, share: float) -> float:
    return score_model(build_lagged(data, column, lag, True), share)["rmse"]


@app.function
def fooled_rows(data) -> list[dict]:
    cases = [
        ("Hand-typed column, lag 4, 80% training", chronological_rmse(data, "revenue by hand", 4, 0.8),
         "A sensible model, tested on the most recent quarters."),
        ("ChatGPT column, lag 2, 80% training", chronological_rmse(data, "revenue by chatgpt", 2, 0.8),
         "The lowest error of any column, yet this column drops digits and misses whole years. Its errors sit in "
         "the years the model learns from, and its test covers different quarters, so the score can't show them."),
        ("Hand-typed column, lag 4, 10% training", chronological_rmse(data, "revenue by hand", 4, 0.1),
         "The model learned from only 4 quarters. The low error is luck, not skill."),
        ("Hand-typed column, lag 4, 90% training", chronological_rmse(data, "revenue by hand", 4, 0.9),
         "Tested on only 5 quarters, so the score depends on which 5 happen to be at the end."),
        ("Hand-typed column, lag 4, random split", random_split_rmse(data, "revenue by hand", 4),
         "A random split lets the model learn from quarters after the ones it is tested on. It peeked at the future."),
    ]
    return [{"setting": s, "reported_rmse_m": round(v / 1e6, 1), "what_is_going_on": w} for s, v, w in cases]



@app.function
def normalize_base_url(text: str) -> str:
    url = text.strip()
    if url and "://" not in url:
        url = "http://" + url
    url = url.rstrip("/")
    if url.endswith("/v1"):
        url = url[: -len("/v1")]
    return url.rstrip("/")


@app.function
def in_browser() -> bool:
    return "pyodide" in sys.modules


@app.function
def urllib_request(url: str, payload: dict | None) -> dict:
    data = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST" if data else "GET"
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read())


@app.function
async def model_request(url: str, payload: dict | None = None) -> dict:
    """GET (no payload) or POST JSON to the student's model server: browser fetch in WASM, urllib locally."""
    try:
        if in_browser():
            from pyodide.http import pyfetch

            options = {"method": "POST", "headers": {"Content-Type": "application/json"},
                       "body": json.dumps(payload)} if payload is not None else {}
            response = await pyfetch(url, **options)
            if not response.ok:
                raise ModelUnavailable(f"Your model server answered with an error (HTTP {response.status}).")
            return await response.json()
        return await asyncio.to_thread(urllib_request, url, payload)
    except ModelUnavailable:
        raise
    except Exception as error:
        raise ModelUnavailable(
            f"Couldn't reach your model ({type(error).__name__}). Check that the SSH tunnel window is still open "
            "and that the server shows READY."
        ) from error


@app.function
async def list_model_name(base: str) -> str:
    body = await model_request(f"{base}/v1/models")
    try:
        return body["data"][0]["id"]
    except (KeyError, IndexError, TypeError) as error:
        raise ModelUnavailable("The server answered, but not like a llama.cpp model server.") from error


@app.function
async def ask_model(base: str, messages: list[dict]) -> str:
    body = await model_request(f"{base}/v1/chat/completions", {"messages": messages, "temperature": 0.3})
    try:
        return body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as error:
        raise ModelUnavailable("The model answered in an unexpected format.") from error


@app.function
def with_entry(mapping: dict, name: str, value) -> dict:
    return {**mapping, name: value}


@app.function
def record_field(setter, key: str, text: str) -> None:
    setter(lambda fields: with_entry(fields, key, text))


@app.function
def cards_ready(opened: frozenset) -> bool:
    return set(CARDS) <= set(opened)


@app.function
def toggled(items: frozenset, item: str) -> frozenset:
    return frozenset(items ^ {item})


@app.function
def added(items: frozenset, item: str) -> frozenset:
    return frozenset(items | {item})


@app.function
def record_flip(set_flipped, set_opened, card_id: str, _value=None) -> None:
    set_flipped(lambda flipped: toggled(flipped, card_id))
    set_opened(lambda opened: added(opened, card_id))


@app.function
def card_html(card_id: str) -> str:
    card = CARDS[card_id]
    title = html.escape(card["title"], quote=False)
    front = (
        "<div class='ss-card ss-front-face'><div class='ss-band'>Student Success</div>"
        f"<div class='ss-front-body'><div class='ss-numeral'>{card['numeral']}</div>"
        f"<div class='ss-front-title'>{title}</div><div class='ss-hair'></div></div>"
        "<div class='ss-band-b'>Click to open</div></div>"
    )
    back = (
        f"<div class='ss-card ss-back ss-back-face'><div class='ss-back-head'><b>{title}</b><i>{card['numeral']}</i></div>"
        f"<div class='ss-back-text'>{html.escape(card['back'], quote=False)}</div></div>"
    )
    return f"<style>{PALETTE}{CARD_STYLE}</style><div class='ss' style='margin:0 -9px'>{front}{back}</div>"


@app.function
def card_slot_style(flipped: bool) -> dict:
    # The face is chosen by custom properties, which inherit into the button's shadow root.
    return {**CARD_SLOT, "--ss-front": "none" if flipped else "flex", "--ss-back": "none" if not flipped else "block"}


@app.function
def bold_html(text: str) -> str:
    parts = html.escape(text, quote=False).split("**")
    return "".join(f"<strong>{part}</strong>" if i % 2 else part for i, part in enumerate(parts))


@app.function
def stage_header_html(stage: int) -> str:
    steps = "".join(
        f'<span class="ss-step now">{name}</span>' if i == stage else f'<span class="ss-step">{name}</span>'
        for i, name in enumerate(STAGES, 1)
    )
    return (
        f'<div class="ss ss-stage"><div class="ss-runhead"><span>{BRIEF_NAME}</span>'
        f"<span>Stage {stage} of 4</span></div>"
        f'<h2 class="ss-stage-title"><em>Stage {stage}</em> · {STAGES[stage - 1]}</h2>'
        f'<div class="ss-steps">{steps}</div></div>'
    )


@app.function
def paragraph_html(text: str) -> str:
    return html.escape(text.strip()).replace("\n", "<br>") if text.strip() else "<i>(left blank)</i>"


@app.function
def prediction_ready(method: str | None, why: str, accuracy: str) -> bool:
    return bool(method) and bool(why.strip()) and bool(accuracy.strip())


@app.function
def locked_prediction_html(prediction: dict) -> str:
    return (
        '<div class="ss ss-locked"><div class="ss-locked-head">Your locked prediction</div>'
        f"<h4>The method I would trust</h4><p>{html.escape(METHOD_LABELS.get(prediction['method'], prediction['method']))}</p>"
        f"<h4>Why</h4><p>{paragraph_html(prediction['why'])}</p>"
        f"<h4>How accurate I expect the forecast to be</h4><p>{paragraph_html(prediction['accuracy'])}</p></div>"
    )


@app.cell(hide_code=True)
def _():
    import marimo as mo

    mo.Html(
        f"{FONT_LINK}<style>{PALETTE}{BRIEF_STYLE}</style>"
        "<div class='ss'><div class='ss-kicker'>ConCReTE · City of Pittsburgh</div>"
        "<h1 class='ss-title'>Forecasting the city's revenue</h1></div>"
    )
    return (mo,)


@app.cell
def _(mo):
    get_intro_dialog_loaded, set_intro_dialog_loaded = mo.state(False)
    return get_intro_dialog_loaded, set_intro_dialog_loaded


@app.cell
def model_state(mo):
    get_model_base, set_model_base = mo.state("")
    get_model_status, set_model_status = mo.state(("idle", ""))
    return get_model_base, get_model_status, set_model_base, set_model_status


@app.cell
def connect_inputs(mo):
    model_url = mo.ui.text(value="http://localhost:8080", label="Model address", full_width=True)
    test_button = mo.ui.run_button(label="Test connection", kind="success")
    return model_url, test_button


@app.cell
async def test_connection(model_url, set_model_base, set_model_status, test_button):
    if test_button.value:
        _base = normalize_base_url(model_url.value)
        try:
            _name = await list_model_name(_base)
            set_model_base(_base)
            set_model_status(("ok", _name))
        except ModelUnavailable as _error:
            set_model_base("")
            set_model_status(("error", str(_error)))
    return


@app.cell(hide_code=True)
def connect_panel(get_model_status, mo, model_url, test_button):
    _state, _detail = get_model_status()
    _note = {
        "ok": mo.Html(f"<div class='ss ss-note ok'><strong>Connected:</strong> {html.escape(_detail)}. "
                      "The AI helper is ready.</div>"),
        "error": mo.Html(f"<div class='ss ss-note'><strong>Not connected.</strong> {html.escape(_detail)}</div>"),
    }.get(_state, "")
    mo.vstack([
        mo.Html("<h3 class='ss ss-task-title'>Before you start: connect your AI model</h3>"),
        mo.Html(f"<div class='ss ss-brief'>{mo.md(CRCD_STEPS).text}</div>"),
        model_url,
        test_button,
        _note,
    ])
    return


@app.cell
def card_states(mo):
    get_flipped, set_flipped = mo.state(frozenset())
    get_opened, set_opened = mo.state(frozenset())
    get_prediction, set_prediction = mo.state(None)
    get_predict_draft, set_predict_draft = mo.state({"why": "", "accuracy": ""})
    return (
        get_flipped,
        get_opened,
        get_predict_draft,
        get_prediction,
        set_flipped,
        set_opened,
        set_predict_draft,
        set_prediction,
    )


@app.cell(hide_code=True)
def card_buttons(mo, set_flipped, set_opened):
    # Built once, both faces in the label; flip_cards picks the face, so a button is never recreated.
    story_card, role_card, deliver_card = (
        mo.ui.run_button(
            label=card_html(card_id),
            on_change=functools.partial(record_flip, set_flipped, set_opened, card_id),
        )
        for card_id in ("story", "role", "deliver")
    )
    return deliver_card, role_card, story_card


@app.cell(hide_code=True)
def flip_cards(deliver_card, get_flipped, get_opened, mo, role_card, story_card):
    _slots = [
        card.style(card_slot_style(card_id in get_flipped()))
        for card_id, card in (("story", story_card), ("role", role_card), ("deliver", deliver_card))
    ]
    _gate = "" if cards_ready(get_opened()) else mo.Html("<p class='ss ss-gate'>Open all three cards to begin.</p>")
    mo.vstack([mo.hstack(_slots, justify="space-between", wrap=True, gap=1), _gate])
    return


@app.cell
def predict_inputs(mo, set_predict_draft):
    method_choice = mo.ui.radio(options={label: key for key, label in METHOD_LABELS.items()}, value=None)
    # Plain dict: the predict cell must not re-run while the learner types.
    predict_fields = {
        key: mo.ui.text_area(rows=4, full_width=True, on_change=functools.partial(record_field, set_predict_draft, key))
        for key in ("why", "accuracy")
    }
    lock_button = mo.ui.run_button(label="Lock my prediction", kind="success")
    return lock_button, method_choice, predict_fields


@app.cell
def lock_prediction(get_predict_draft, lock_button, method_choice, mo, set_prediction):
    _draft = get_predict_draft()
    _ready = prediction_ready(method_choice.value, _draft["why"], _draft["accuracy"])
    if lock_button.value and _ready:
        set_prediction({"method": method_choice.value, "why": _draft["why"].strip(), "accuracy": _draft["accuracy"].strip()})
    lock_warning = (
        mo.Html("<div class='ss ss-note'>Choose a method and answer both questions before you lock.</div>")
        if lock_button.value and not _ready
        else ""
    )
    return (lock_warning,)


@app.cell(hide_code=True)
def predict(get_opened, get_prediction, lock_button, lock_warning, method_choice, mo, predict_fields):
    mo.stop(not cards_ready(get_opened()))
    _guiding = "".join(f"<li>{html.escape(q)}</li>" for q in PREDICT_GUIDING)
    _header = [
        mo.Html(stage_header_html(1)),
        mo.Html(
            "<div class='ss ss-prompt'>Before you see the data, form your own view. Think through these questions "
            f"first. Nobody will grade your answers to them:<ul>{_guiding}</ul>Then commit to a position below. "
            "<strong>Once you lock your prediction it cannot be changed.</strong> You will come back to it at the "
            "end.</div>"
        ),
    ]
    if get_prediction() is None:
        _output = mo.vstack(_header + [
            mo.Html(
                "<div class='ss'><div class='ss-label'><i>1</i><span>Which extraction method would you trust "
                f"to build the city's forecast?</span></div><div class='ss-indent'>{method_choice}</div>"
                "<div class='ss-label'><i>2</i><span>Why? What makes it more trustworthy than the others?</span>"
                f"</div><div class='ss-indent'>{predict_fields['why']}</div>"
                "<div class='ss-label'><i>3</i><span>How accurate do you expect a simple forecast of city revenue "
                f"to be, and why?</span></div><div class='ss-indent'>{predict_fields['accuracy']}</div>"
                f"<div class='ss-indent' style='margin-top:22px'>{lock_button}</div></div>"
            ),
            lock_warning,
        ])
    else:
        _output = mo.vstack(_header + [mo.Html(locked_prediction_html(get_prediction()))])
    _output
    return


@app.cell(hide_code=True)
def briefing(get_prediction, mo):
    mo.stop(get_prediction() is None)
    mo.accordion({
        "Read the full briefing": mo.md(
            "\n\n".join(f"### {card['title']}\n\n{card['full']}" for card in CARDS.values())
        )
    })
    return


@app.cell(hide_code=True)
def _(mo, set_intro_dialog_loaded):
    context = """
        # **ConCReTE: City of Pittsburgh Module**
        ---
        ## **The Story So Far**
        The City of Pittsburgh is implementing a three-year data strategy under the leadership of Chris Belasco, Senior Manager of Digital Services and Chief Data Officer. This strategy positions data as a critical asset to enhance city operations, engage the community, and drive equity. The city focuses on improving data accessibility and quality while fostering a culture of insight-driven decision-making. Participants in this initiative are tasked with analyzing and forecasting the city’s revenue, which is critical for supporting Pittsburgh’s long-term goals. Prior to the start of this scenario, Chris's team has:

        1. Identified a need to forecast the city's revenue.
        2. Reviewed historical revenue data and identified the specific tables needed in the City's Revenue Report PDFs; namely, the Q4 Revenue Summary Table from each report and the Quarterly Revenue Totals (Actual, not projected) from each table.
        3. Downloaded the Revenue Report PDFs and extracted the Q4 Revenue Summary Tables into separate documents.
        4. Used six different methodologies to extract the actual quarterly revenue totals from each PDF table.
            - Manual Extraction by a Human
            - Automated Extraction with Docling
            - Automated Extraction with One-Shot AI ChatBot (Claude & ChatGPT)
            - Automated Extraction Claude API (PDF & Images)
            - Automated Extraction with pypdf

        ### Role
        You are an intern working for Chris, and he has asked you to help out with this next stage of the project.

        ### Business Objective
        You want to use the historical quarterly revenue data to create a predictive model for the city's revenue.
        Beyond creating an accurate predictive model, you will need to consider what the most appropriate way to approach parsing the data is.

        ### Data
        We have a dataset of the city's quarterly revenue from 2012 to 2023.

        ---
        """

    rdso_1 = """
    ### *RDSO 1: Evaluate*
    _What previous decisions might have been made that impact your work here? Would your answer change if your role was different?_
    """

    rdso_2 = """
    ### *RDSO 2: Predict*
    _Before selecting from one of the 6 options below, consider each option carefully. Do any of them raise technical or ethical concerns?_
    """

    rdso_3 = """
    ### *RDSO 3: Synthesize*
    _What are your thoughts on using data extracted by AI? Did it affect the quality of your model?_
    """

    _base = (
        "You are reviewing the work of a student learning responsible data science skills. "
        "Below is the context of the scenario they are working within and the one question they are answering right now. "
        "Give them feedback on their response in the form of a single follow-up question. "
        "Once they respond to your follow-up, encourage them to move to the next part of the scenario."
    )
    _questions = {1: rdso_1, 2: rdso_2, 3: rdso_3}
    # student messages per chat, so later chats know earlier answers.
    # plain dict, not mo.state: updating it must not re-run cells (that would wipe the chats)
    _answers = {}

    def _make_tutor(n):
        def tutor(messages, config):
            _answers[n] = [m.content for m in messages if m.role == "user"]
            if not input_key.value:
                return "Please paste your Groq API key in the box at the top of the page (see **Before you start**), press Enter, then send your message again."
            system = f"{_base}\n\n{context}\n\nCURRENT QUESTION:\n{_questions[n]}"
            earlier = "\n\n".join(
                f"RDSO {k}:\n" + "\n".join(_answers[k]) for k in range(1, n) if _answers.get(k)
            )
            if earlier:
                system += f"\n\nFor reference, the student's earlier answers in this scenario:\n{earlier}"
            llm = mo.ai.llm.groq(model="llama-3.1-8b-instant", system_message=system, api_key=input_key.value)
            return llm(messages, config)
        return tutor

    chat_1, chat_2, chat_3 = (mo.ui.chat(_make_tutor(n)) for n in (1, 2, 3))
    set_intro_dialog_loaded(True)
    mo.vstack([mo.md(context), mo.vstack([mo.md(rdso_1), chat_1]).callout()])
    return chat_2, chat_3, rdso_2, rdso_3


@app.cell(hide_code=True)
def _(mo, pd):
    import sys as _sys
    # load the data (a local path when run locally, a URL in the browser build)
    data_path = str(mo.notebook_location() / "public" / "data" / "all_extracted_data.csv")
    if "pyodide" in _sys.modules:
        # fetch the text ourselves: given the URL, pandas sees GitHub Pages' gzip header
        # and tries to decompress text the browser already decompressed (BadGzipFile)
        from pyodide.http import open_url
        data_path = open_url(data_path)
    pgh_revenue_data = pd.read_csv(data_path)

    # create a new column that is the year and quarter combined
    pgh_revenue_data['Year-Quarter'] = pgh_revenue_data['year'].astype(str) + '-' + pgh_revenue_data['quarter']
    return (pgh_revenue_data,)


@app.cell(hide_code=True)
def _(
    chat_2,
    is_ai_source,
    mo,
    pgh_revenue_data,
    rdso_2,
    revenue_selector,
):
    ds_task = mo.md(
        """
        ---
        ## **Data Science Task: Building a Forecasting Model**
        You have a tabular dataset, `all_extracted_data.csv`, which contains all of the extracted data for each of the 6 methods. Your goal is to try to accurately predict quarterly revenue values using this historical data.
        First we load our dataset `all_extracted_data.csv` which contains the city's quarterly revenue.
        Use the DataFrame explorer below to get familiar with the data. What differences do you notice?
        """
    )

    options = mo.md(
        "Our dataset has six options for the quarterly revenue data. We need to pick one."
    )

    ai_warning = (
                    mo.md(f"You have selected the **{revenue_selector.value}** dataset.").callout()
                    if not is_ai_source
                    else mo.md(f"""You have selected the **{revenue_selector.value}** dataset, which was extracted using AI.""").callout("warn")
                )


    # display the data
    data_vstack = mo.vstack([ds_task, pgh_revenue_data])
    rdso_2_vstack = mo.vstack([mo.md(rdso_2), chat_2]).callout()
    selector_vstack = mo.vstack([revenue_selector, ai_warning])

    mo.vstack([data_vstack, options, rdso_2_vstack, selector_vstack])
    return


@app.cell(hide_code=True)
def _(mo, pgh_revenue_data, revenue_selector):
    rev_description = mo.md("Now you can create a new dataframe, `revenues`, with just the dataset you have selected.")

    # create a copy of just the data we want
    revenues = pgh_revenue_data[['Year-Quarter', revenue_selector.value]].copy()
    revenues = revenues.rename(columns={revenue_selector.value:"revenue"})
    mo.vstack([rev_description, revenues])
    return (revenues,)


@app.cell
def _(get_intro_dialog_loaded, mo):
    mo.stop(not get_intro_dialog_loaded())
    mo.md(f"""Now you need to select features for the statistical models. These features will be the inputs to your modeling function. We are doing a [time-series analysis](https://www.geeksforgeeks.org/machine-learning/time-series-analysis-and-forecasting/) so our features are [lagged values.](https://www.geeksforgeeks.org/machine-learning/what-is-lag-in-time-series-forecasting/) Use the slider below to select how much lag we should lag our features.""")
    return


@app.cell(hide_code=True)
def _(lag_amount, mo):
    mo.vstack([lag_amount, f"You have selected a lag of {lag_amount.value}"])
    return


@app.cell
def _(lag_amount, revenues):
    # add lag value column
    lagged_revenues = revenues.assign(lag=revenues['revenue'].shift(lag_amount.value))
    lagged_revenues
    return (lagged_revenues,)


@app.cell
def _(get_intro_dialog_loaded, mo):
    mo.stop(not get_intro_dialog_loaded())
    mo.md(f"""Notice the [missing values](https://www.geeksforgeeks.org/python/how-to-deal-with-missing-values-in-a-timeseries-in-python/) as you increase the lag. Also note, depending on which extraction method column you selected there are missing values as well. Do you want to clean the data and remove rows with missing values?""")
    return


@app.cell(hide_code=True)
def _(mo, remove_missing):
    mo.hstack([
        remove_missing,
        mo.md(
            f"You have decided {'**not**' if not remove_missing.value else ''} to remove missing values."
            f"{' This will feed your model `NULL` values.' if not remove_missing.value else ''}"
        )
    ], justify="start", gap=5)
    return


@app.cell(hide_code=True)
def _(lagged_revenues, remove_missing):
    # optionally drop rows with missing values
    clean_revenues = lagged_revenues.dropna() if remove_missing.value else lagged_revenues.copy()
    clean_revenues
    return (clean_revenues,)


@app.cell(hide_code=True)
def _(mo, training_percent):
    mo.md(f"""
    What percentage of the data do you want to use for [training?](https://www.geeksforgeeks.org/python/training-data-vs-testing-data/#what-is-training-data?)

    {training_percent}

    You have selected {training_percent.value * 100:.0f}% of the data for training.
    """)
    return


@app.cell
def _(clean_revenues, training_percent):
    # Chronological split
    train_size = int(len(clean_revenues) * training_percent.value)
    train = clean_revenues.iloc[:train_size]
    test = clean_revenues.iloc[train_size:]

    x_train = train[['lag']]
    y_train = train['revenue']
    x_test = test[['lag']]
    y_test = test['revenue']
    return x_test, x_train, y_test, y_train


@app.cell(hide_code=True)
def _(mo, x_test, x_train, y_test, y_train):
    mo.hstack([mo.vstack([mo.md("**X Train**"),x_train,mo.md("**Y Train**"),y_train]),
              mo.vstack([mo.md("**X Test**"),x_test,mo.md("**Y Test**"),y_test]),
              ])
    return


@app.cell
def _(get_intro_dialog_loaded, mo):
    mo.stop(not get_intro_dialog_loaded())
    mo.vstack([
        mo.md("---"),
        mo.md(f"""## **Forecasting Model Implementation**
    Now you have everything you need to generate the model. As you've made your selections, Marimo has been building the model dynamically behind the scenes. Below are your results alongside a dashboard where you can adjust all parameters in one place if you'd like to make changes."""),
        mo.md("---")
    ])
    return


@app.cell(hide_code=True)
def _(
    LinearRegression,
    mo,
    root_mean_squared_error,
    x_test,
    x_train,
    y_test,
    y_train,
):
    try:
        # train the model
        model = LinearRegression()
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)

        output = mo.md(
            f"""
            *The model has a Root Mean Square Error (RMSE) of ${root_mean_squared_error(y_test, predictions):,.2f}*

            *What this means:*

            - *On average, your predictions are off by ${root_mean_squared_error(y_test, predictions):,.2f}*
            - *The average actual revenue in your test set is ${y_test.mean():,.2f}*
            - *The RMSE represents {(root_mean_squared_error(y_test, predictions)/y_test.mean()*100):.1f}% of the average revenue*

            *For context:*

            - *68% of predictions should fall within ±${root_mean_squared_error(y_test, predictions):,.2f} of the actual value*
            - *The minimum revenue in your test set is ${y_test.min():,.2f}*
            - *The maximum revenue in your test set is ${y_test.max():,.2f}*
        """
        ).callout('info')
    except Exception as e:
        output = mo.md(
            f"**Your code has an error!**\n\n```\n{e}\n```\n\nTry adjusting your selections and running the code again."
        ).callout("danger")

    output
    return


@app.cell(hide_code=True)
def _(lag_amount, mo, remove_missing, revenue_selector, training_percent):
    mo.md(f"""
    ## Adjust your selections

    Play with the options again and see how the statistical performance changes.

    What combination of choices are most accurate?

    {mo.hstack([revenue_selector, mo.vstack([remove_missing,lag_amount,training_percent])], justify="start", gap=5)}
    """)
    return


@app.cell
def _(get_intro_dialog_loaded, mo):
    mo.stop(not get_intro_dialog_loaded())
    mo.md(f"""## Communicating Results

    You now have a predictive model for the city's revenue. How would you communicate these results back to Chris?""")
    return


@app.cell(hide_code=True)
def _(mo, remove_missing, revenue_selector):
    _ai_sources = ["claude", "chatgpt", "docling"]
    is_ai_source = any(value in revenue_selector.value for value in _ai_sources)
    (
        mo.md("You elected not to remove missing values. This broke your model. Try removing those and seeing what you can come up with!").callout('warn')
        if not remove_missing.value
        else mo.md("Nice job! Parsing the data by hand or writing a script to do it for you may take longer, but it ensures that your results are accurate. Now think about how you'll present your findings to your boss.").callout('success')
        if not is_ai_source
        else mo.md(f"""You trained your model using **{revenue_selector.value}**, which is a column from an AI source! You need to think about how you are going to convey the accuracy of the data when you communicate your results back to Chris.""").callout("danger")
    )
    return (is_ai_source,)


@app.cell
def _(chat_3, mo, rdso_3):
    mo.vstack([mo.md(rdso_3), chat_3]).callout()
    return


@app.cell
def _(mo, pgh_revenue_data):
    # select the column names that start with "revenue_by"
    _revenue_columns = [col for col in pgh_revenue_data.columns if col.lower().startswith('revenue by')]


    # make marimo buttons for each of the values in the _revenue_columns list
    revenue_selector = mo.ui.radio(
        options=_revenue_columns,
        value=_revenue_columns[0],  # sets default value to first column
        label="Select Revenue Column"
    )
    return (revenue_selector,)


@app.cell
def _(mo):
    remove_missing = mo.ui.switch(label="Remove missing values")
    return (remove_missing,)


@app.cell
def _(mo):
    training_percent = mo.ui.slider(start=0.1, stop=0.9, step=0.1, value=0.1, label="Training Set Percentage:")
    return (training_percent,)


@app.cell
def _(mo):
    lag_amount = mo.ui.slider(start=1, stop=5, step=1, value=1, label="Lag Amount:")
    return (lag_amount,)


if __name__ == "__main__":
    app.run()
