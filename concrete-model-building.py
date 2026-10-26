import marimo

__generated_with = "0.18.0"
app = marimo.App(width="compact", app_title="ConCReTE Model Building")

with app.setup:
    import ast
    import base64
    import asyncio
    import contextlib
    import io
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

        You will use **two terminal windows on your laptop**. **Terminal A** logs in to CRCD and runs the model. **Terminal B** stays on your laptop and holds the tunnel that connects this page to the model.

        **1. Terminal A: log in to CRCD and start a GPU session.** Open a terminal on your laptop (this is Terminal A) and log in with `ssh <pittID>@h2p.crc.pitt.edu`. Then start a GPU session:
        ```
        srun -M gpu -p l40s -n1 -c16 --gres=gpu:1 -t 2:00:00 --pty bash
        ```
        When the prompt changes to something like `[pittID@gpu-n80 ~]$`, you are on a GPU machine. Type steps 2 and 3 here, in Terminal A.

        **2. Terminal A: choose the model and start the server.** Replace `<group>` with your CRCD group name:
        ```
        export LLAMACPP_MODEL="bartowski/Qwen2.5-32B-Instruct-GGUF:Q4_K_M"
        export LLAMACPP_DOWNLOAD_DIR="/vast/<group>/$USER/llamacpp"
        export LLAMACPP_EXTRA_ARGS="-c 16384"
        module load llamacpp/0.5.0
        ```
        This is a 32-billion-parameter model compressed to about 20 GB. The first download takes a while; later sessions reuse the folder.

        **3. Terminal A: wait until it's ready, then print your tunnel command.** Run `llamacpp-status` until it prints something like `READY - http://gpu-n80:27555` (your node and port will differ). Then run this, exactly as written:
        ```
        echo "ssh -N -L 8080:${LLAMACPP_BASE_URL#http://} $USER@h2p.crc.pitt.edu"
        ```
        It prints your own tunnel command, for example `ssh -N -L 8080:gpu-n80:27555 abc123@h2p.crc.pitt.edu`. Copy the whole printed line. If you type it yourself instead, the node name must include `gpu-`: `gpu-n80`, not `n80`.

        **4. Terminal B: open the tunnel.** Open a *second* terminal window on your laptop (this is Terminal B). Do not log in to CRCD in it. Paste the line you copied and press Enter. After your Pitt password, it will look like nothing happens. That's normal. Leave Terminal B open the whole time you work.

        **5. Connect.** Keep the address below as `http://localhost:8080` and click **Test connection**. Your browser will ask whether this site may access other apps and devices on your network or computer. Click **Allow**. That lets this page talk to the tunnel.

        **6. When you're finished,** type `exit` in Terminal A and press Ctrl+C in Terminal B. The GPU is billed for every minute you hold it, whether or not you're using it.

        **Test connection fails?** Look at Terminal B.
        - `open failed: connect failed: Name or service not known`: the node name in your tunnel is wrong, most often a missing `gpu-`. Press Ctrl+C in Terminal B and paste the line from step 3 again.
        - `open failed: connect failed: Connection refused`: the port is wrong or the server stopped. Run `llamacpp-status` in Terminal A; the port changes every time the server starts, so print the tunnel command again.
        - Terminal B is back at a prompt: the tunnel closed. Paste the line from step 3 again.

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
        ".poee-eval .ev.reveal{border-top-color:#003594}"
        ".poee-eval .num{white-space:nowrap}"
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

    STARTER_CELLS = {
        "compare": {
            "title": "1 · Compare the sources",
            "question": "The averages look similar. Does that mean every quarter matches? How would you check row by row?",
            "code": textwrap.dedent("""\
                # Average quarterly revenue reported by each extraction method
                methods = [c for c in revenue_data.columns if c.startswith("revenue by")]
                revenue_data[methods].mean().round(0)
            """),
        },
        "plot": {
            "title": "2 · Plot over time",
            "question": "Add the other columns to the plot. Does anything look impossible for a city budget? Do you see a pattern that repeats every year?",
            "code": textwrap.dedent("""\
                labels = revenue_data["year"].astype(str) + " " + revenue_data["quarter"]
                plt.figure(figsize=(9, 4))
                plt.plot(labels, revenue_data["revenue by hand"], label="by hand")
                plt.xticks(labels[::4], rotation=45)
                plt.ylabel("Revenue ($)")
                plt.title("Quarterly revenue")
                plt.legend()
            """),
        },
        "truth": {
            "title": "3 · Who is right?",
            "question": "Is any source automatically the truth? Look for quarters where the hand-typed value is the odd one out.",
            "code": textwrap.dedent("""\
                # Treat the hand-typed column as the truth and count disagreements per method
                truth = revenue_data["revenue by hand"]
                others = [c for c in revenue_data.columns if c.startswith("revenue by") and c != "revenue by hand"]
                {method: int(((revenue_data[method] - truth).abs() > 1).sum()) for method in others}
            """),
        },
        "missing": {
            "title": "4 · Missing values",
            "question": "Which quarters are missing for each method? What happens to \"the same quarter last year\" if a whole year disappears?",
            "code": textwrap.dedent("""\
                # How many blanks does each column have?
                revenue_data.isna().sum()
            """),
        },
        "split": {
            "title": "5 · Testing the model",
            "question": "This error is lower than the controls below give you. Did the model get better, or did the test get easier?",
            "code": textwrap.dedent("""\
                frame = revenue_data[["revenue by hand"]].rename(columns={"revenue by hand": "revenue"})
                frame["lag"] = frame["revenue"].shift(4)
                frame = frame.dropna()
                train, test = train_test_split(frame, test_size=0.2, random_state=0)
                model = LinearRegression().fit(train[["lag"]], train["revenue"])
                error = root_mean_squared_error(test["revenue"], model.predict(test[["lag"]]))
                print(f"Error: ${error / 1e6:.1f} million")
            """),
        },
    }

    HELPER_RULES = textwrap.dedent("""\
        You are a patient teaching assistant inside a data-science exercise for university students.
        The student is investigating six versions of the City of Pittsburgh's quarterly revenue, each extracted from
        PDF reports by a different method, and building a simple lag-based forecast.

        How to help:
        - Explain concepts (lags, missing values, train/test splits, error measures), Python and pandas code, and error messages.
        - When they ask what is wrong with the data, guide them toward HOW to look (compare columns row by row, plot them, check where values disagree) rather than telling them WHAT you see.
        - Never choose a column, a lag or a split for them, never name specific wrong values or quarters, and never grade their work.
        - Keep answers short: a few sentences, with a small code example when it helps.

        You can see the data, their current code in each exploration cell with its last output, and their current settings below.
    """)

    EXPLAIN_PROMPTS = [
        "Which method did you end up trusting, and what changed or confirmed your view since you made your prediction?",
        "What are your thoughts on using data extracted by AI? Did it affect the quality of your model?",
        "Write a short note to Chris: your forecast, how confident he should be in it, and why.",
    ]

    SOUND_COLUMNS = ("revenue by hand", "revenue by claude chat", "revenue by claude api", "revenue by pypdf")
    SOUND_COLUMN_TEXT = (
        "By hand, Claude chat, Claude API or pypdf. They agree on every quarter except 2017 Q2, where all four "
        "machine methods record $113,830,377 and the hand-typed value is $113,830,337."
    )
    COLUMN_VERDICTS = {
        "revenue by docling": {
            "why": "Docling read 2018 Q4 as $6,425 (the other sources say $95.4 million) and 2021 Q4 as $608.6 "
                   "million (the others say $129.7 million). It is also missing every quarter of 2013, 2017 and 2024.",
            "cost": "Your model learned from a quarter where the city seemingly collected almost nothing, and from "
                    "another four times its real size. Depending on the lag and split, those quarters either sit in "
                    "the years the model trains on, where the error figure can't see them, or feed the quarters it is "
                    "tested on, where they inflate the error without saying why. A budget built on this forecast "
                    "rests on numbers nobody checked.",
        },
        "revenue by chatgpt": {
            "why": "ChatGPT misread all four quarters of 2019 (for example $12.9 million where the reports say $122.2 million) "
                   "and is missing every quarter of 2013, 2014, 2022 and 2024.",
            "cost": "At some lags and splits this column scores the lowest error of all six, so it can look like the "
                    "best choice. Chris would "
                    "be told the forecast is more reliable than it is, and the city could plan spending around "
                    "revenue that was never there.",
        },
        None: {
            "why": "You didn't choose a column, so no model was built.",
            "cost": "Chris would have no forecast to plan with. Choosing which numbers to trust is the first decision "
                    "in this task.",
        },
    }
    MISSING_VERDICT = {
        "sound": "Remove rows with missing values, after checking which quarters they are.",
        "why": "A linear regression can't train on blank quarters, so with missing values kept the model produced no "
               "error figure and no forecast.",
        "cost": "With nothing to hand Chris, someone ends up filling the blanks by guessing, which is how unchecked "
                "numbers slip into a budget.",
    }
    LAG_VERDICT = {
        "sound": "A lag of 4: each quarter is predicted from the same quarter a year earlier.",
        "why": "City revenue follows a yearly pattern: Q1 averages $214.9 million, against $104 to $132 million for "
               "the other quarters. Other lags compare unlike quarters. For the hand-typed column, lag 4 roughly "
               "halves the error ($85.6 million at lag 2, $42.2 million at lag 4).",
        "cost": "The forecast would miss the seasonal swing, predicting a quiet quarter after a busy Q1 and a busy one "
                "after a quiet quarter, so cash planning would be wrong in both directions.",
    }
    SPLIT_VERDICTS = {
        "small": {
            "why": "Training on so little history leaves the model only a handful of quarters to learn from. Its "
                   "error can still look good (at 10%, the hand-typed column scores $27.9 million after learning "
                   "from just 4 quarters), but that is luck, not skill.",
            "cost": "Chris would get a forecast that looks well tested but rests on a year or two of history.",
        },
        "large": {
            "why": "Training on 90% leaves only about 5 quarters to test on, so the error figure depends on which "
                   "quarters happen to be last (the hand-typed column scores $11.7 million).",
            "cost": "The error figure would make the forecast sound far more reliable than it is.",
        },
        "sound": "50% to 80%: enough quarters to learn from, and enough recent quarters to test on.",
    }
    REVEALS = {
        "compare": "Averages hide single bad values. Docling records 2018 Q4 as $6,425 instead of $95.4 million, and "
                   "ChatGPT misreads 2019, for example $12.9 million instead of $122.2 million. Only a "
                   "row-by-row comparison shows them.",
        "plot": "Plotting every column shows spikes and drops no real budget has, and a pattern that repeats every "
                "year: Q1 is by far the biggest quarter. That pattern is why a lag of 4, the same quarter last year, "
                "works best.",
        "truth": "On 2017 Q2 every machine method that read 2017 records $113,830,377 (Docling skipped the year), "
                 "and only the hand-typed value differs "
                 "($113,830,337). The person most likely made the typo. No source is automatically the truth.",
        "missing": "The blanks aren't random. Docling and ChatGPT miss whole years, so dropping missing rows quietly "
                   "removes years from the time series, and a year-ago comparison then lines up the wrong quarters.",
        "split": "A random split mixes later quarters into training, so the model is partly tested on the past it "
                 "has already seen around. Its $29.0 million error looks better than the honest chronological "
                 "$42.2 million only because the test got easier.",
    }

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
         "The lowest error of any column, yet this column misreads several quarters and misses whole years. Its errors sit in "
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
    except urllib.error.HTTPError as error:
        raise ModelUnavailable(f"Your model server answered with an error (HTTP {error.code}).") from error
    except Exception as error:
        raise ModelUnavailable(
            f"Couldn't reach your model ({type(error).__name__}). Look at your tunnel window (Terminal B): if it "
            "says open failed, the node name or port in the tunnel command is wrong, so copy it again from step 3. "
            "Otherwise check that Terminal B is still open and that llamacpp-status shows READY."
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
        "<div class='ss-card ss-front-face'><div class='ss-band'>City of Pittsburgh</div>"
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


@app.function
def cell_namespace(data) -> dict:
    return {
        "revenue_data": data.copy(),
        "pd": pd,
        "LinearRegression": LinearRegression,
        "root_mean_squared_error": root_mean_squared_error,
        "train_test_split": train_test_split,
        "plt": plt,
    }


@app.function
def figure_html(figure) -> str:
    """Draw now, so a figure that can't be drawn fails inside run_cell's guard, not in the page."""
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=100, bbox_inches="tight")
    encoded = base64.b64encode(buffer.getvalue()).decode()
    return f"<img src='data:image/png;base64,{encoded}' style='max-width:100%' alt='Your plot'>"


@app.function
def run_cell(code: str, data) -> dict:
    """Run a student's cell on a fresh copy of the data; return stdout, last value, error and any figure."""
    namespace = cell_namespace(data)
    stdout = io.StringIO()
    value, error, figure = None, None, None
    plt.close("all")
    try:
        tree = ast.parse(code)
        last = tree.body.pop() if tree.body and isinstance(tree.body[-1], ast.Expr) else None
        with contextlib.redirect_stdout(stdout):
            exec(compile(tree, "<cell>", "exec"), namespace)
            if last is not None:
                value = eval(compile(ast.Expression(last.value), "<cell>", "eval"), namespace)
        if plt.get_fignums():
            figure = figure_html(plt.gcf())
    except (Exception, SystemExit) as exc:
        error = f"{type(exc).__name__}: {exc}"
    return {
        "stdout": stdout.getvalue(),
        "value": None if value is None else repr(value),
        "result": value,
        "error": error,
        "figure": figure,
    }


@app.function
def explain_unlocked(prediction, runs: dict) -> bool:
    return prediction is not None and all(runs.get(key, 0) >= 1 for key in STARTER_CELLS)


@app.function
def cell_summary(key: str, cell: dict) -> str:
    code = cell.get("code", "")[:2000]
    output = (cell.get("output") or "")[:2000]
    return f"--- Cell {key} ---\nCode:\n{code}\nLast output:\n{output}"


@app.function
def helper_system_prompt(data_csv: str, cells: dict, settings: dict) -> str:
    sections = "\n\n".join(cell_summary(key, cell) for key, cell in cells.items())
    chosen = ", ".join(f"{name}={value}" for name, value in settings.items()) or "none yet"
    return f"{HELPER_RULES}\n\nDATA (CSV):\n{data_csv}\n\nEXPLORATION CELLS:\n{sections}\n\nCURRENT SETTINGS: {chosen}"


@app.function
def helper_history(system: str, messages: list) -> list:
    # The newest question and the four exchanges before it: older turns would overflow the model's context.
    return [{"role": "system", "content": system}] + messages[-9:]


@app.function
def output_text(result: dict | None) -> str:
    if not result:
        return ""
    return "\n".join(part for part in (result["stdout"], result["value"] or "", result["error"] or "") if part)


@app.function
def run_and_record(editor, data, set_outputs, set_runs, key: str, _value=None) -> None:
    result = run_cell(editor.value, data)
    set_outputs(lambda outputs: with_entry(outputs, key, result))
    set_runs(lambda runs: with_entry(runs, key, runs.get(key, 0) + 1))


@app.function
def can_submit(fields: dict) -> bool:
    return all(fields.get(f"r{i}", "").strip() for i in range(1, len(EXPLAIN_PROMPTS) + 1))


@app.function
def column_card(column: str | None) -> dict | None:
    if column in SOUND_COLUMNS:
        return None
    verdict = COLUMN_VERDICTS[column]
    return {"setting": "column", "title": "Which numbers you trusted",
            "chose": METHOD_LABELS.get(column, "No column chosen"), "sound": SOUND_COLUMN_TEXT, **verdict}


@app.function
def split_card(train_share: float) -> dict | None:
    share = round(train_share, 1)
    if 0.5 <= share <= 0.8:
        return None
    verdict = SPLIT_VERDICTS["small" if share < 0.5 else "large"]
    return {"setting": "split", "title": "How much history you trained on", "chose": f"{share:.0%} for training",
            "sound": SPLIT_VERDICTS["sound"], **verdict}


@app.function
def settings_cards(column: str | None, lag: int, drop_missing: bool, train_share: float) -> list[dict]:
    cards = [column_card(column)]
    if not drop_missing:
        cards.append({"setting": "missing", "title": "Missing values", "chose": "Kept the blank quarters", **MISSING_VERDICT})
    if lag != 4:
        cards.append({"setting": "lag", "title": "How far back the model looks", "chose": f"A lag of {lag}", **LAG_VERDICT})
    cards.append(split_card(train_share))
    return [card for card in cards if card]


@app.function
def reveal_cards(cells: dict, runs: dict) -> list[dict]:
    return [
        {"key": key, "title": cell["title"], "finding": REVEALS[key],
         "edited": cells.get(key, cell["code"]).strip() != cell["code"].strip(), "ran": runs.get(key, 0) > 0}
        for key, cell in STARTER_CELLS.items()
    ]


@app.function
def settings_card_html(card: dict) -> str:
    return (
        f"<div class=\"ev\"><h3>{html.escape(card['title'])}</h3>"
        f"<div class='trail'><div><span class='k'>You chose</span><b>{html.escape(card['chose'])}</b></div>"
        f"<div class='ok'><span class='k'>A sound choice</span>{html.escape(card['sound'])}</div></div>"
        f"<div class='why'><b>Why.</b> {html.escape(card['why'])}</div>"
        f"<div class='cost'><b>What happens for the city.</b> {html.escape(card['cost'])}</div></div>"
    )


@app.function
def reveal_card_html(card: dict) -> str:
    fact = "You changed the starter code." if card["edited"] else "You ran the starter code as given."
    return (
        f"<div class=\"ev reveal\"><h3>{html.escape(card['title'])}</h3>"
        f"<div class='why'>{html.escape(card['finding'])}</div>"
        f"<div class='also'>{fact}</div></div>"
    )


@app.function
def fooled_table_html(data) -> str:
    rows = "".join(
        f"<tr><td>{html.escape(r['setting'])}</td><td class='num'>${r['reported_rmse_m']:.1f}M</td>"
        f"<td>{html.escape(r['what_is_going_on'])}</td></tr>"
        for r in fooled_rows(data)
    )
    return ("<table class='ss-sheet'><tr><th>Setting</th><th>Reported error</th><th>What's really going on</th></tr>"
            f"{rows}</table>")


@app.function
def evaluation_html(data, settings: dict, cells: dict, runs: dict) -> str:
    cards = settings_cards(settings["column"], settings["lag"], settings["remove_missing"], settings["training_share"])
    part1 = "".join(settings_card_html(c) for c in cards) or (
        "<div class='sum'><div class='big'>Every setting holds up.</div></div>"
    )
    part3 = "".join(reveal_card_html(c) for c in reveal_cards(cells, runs))
    return (
        f"<style>{PALETTE}</style>{EVAL_STYLE}<div class='ss poee-eval'>"
        f"<div class='grp'>Part 1 · Your forecast settings</div>{part1}"
        "<div class='grp'>Part 2 · The number that fooled you</div>"
        "<div class='sum'><div class='why'>A lower error is not proof of a better forecast. These settings all "
        "\"score\" well for the wrong reasons:</div>"
        f"{fooled_table_html(data)}</div>"
        f"<div class='grp'>Part 3 · What the data was hiding</div>{part3}</div>"
    )


@app.function
def forecast_summary(data, settings: dict) -> str:
    if not settings["column"]:
        return "No column chosen, so no model."
    try:
        score = score_model(build_lagged(data, settings["column"], settings["lag"], settings["remove_missing"]),
                            settings["training_share"])
    except ValueError as error:
        return str(error)
    forecast = next_quarter_forecast(data, settings["column"], settings["lag"])
    told = (f"Forecast for {forecast['label']}: ${forecast['value']:,.0f}." if forecast
            else "No forecast possible: " + forecast_unavailable_reason(data, settings["column"], settings["lag"]))
    return f"Error (RMSE) ${score['rmse']:,.0f} on {score['test_rows']} test quarters. {told}"


@app.function
def report_html(data, prediction: dict, settings: dict, explain: dict, helper_log: list, cells: dict, runs: dict) -> str:
    answers = "".join(
        f"<h3>{i}. {html.escape(q)}</h3><p>{paragraph_html(explain.get(f'r{i}', ''))}</p>"
        for i, q in enumerate(EXPLAIN_PROMPTS, 1)
    )
    questions = "".join(f"<li>{html.escape(q)}</li>" for q in helper_log) or "<li><i>(none)</i></li>"
    chosen = (f"{METHOD_LABELS.get(settings['column'], 'no column')} · lag {settings['lag']} · missing values "
              f"{'removed' if settings['remove_missing'] else 'kept'} · {settings['training_share']:.0%} training")
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Revenue Forecast Brief</title>
<style>body {{ font-family: system-ui, sans-serif; max-width: 900px; margin: 2rem auto; padding: 0 1rem; line-height: 1.5; }}
th, td {{ padding: 4px 8px; vertical-align: top; border-bottom: 1px solid #ddd; text-align: left; }}</style></head><body>
<h1>Revenue Forecast Brief</h1>
<p><b>Scenario:</b> City of Pittsburgh quarterly revenue forecast</p>
<h2>Prediction (locked before seeing the data)</h2>
{locked_prediction_html(prediction)}
<h2>Forecast</h2>
<p><b>Settings:</b> {html.escape(chosen)}<br>{html.escape(forecast_summary(data, settings))}</p>
<h2>Reflection and note to Chris</h2>
{answers}
<h2>Questions asked to the AI helper</h2>
<ul>{questions}</ul>
<h2>Evaluation</h2>
{evaluation_html(data, settings, cells, runs)}
</body></html>"""


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
def load_data(mo):
    # A local path when run locally, a URL in the browser build.
    _path = str(mo.notebook_location() / "public" / "data" / "all_extracted_data.csv")
    if "pyodide" in sys.modules:
        # Fetch the text ourselves: given the URL, pandas sees GitHub Pages' gzip header
        # and tries to decompress text the browser already decompressed (BadGzipFile).
        from pyodide.http import open_url

        _path = open_url(_path)
    pgh_revenue_data = pd.read_csv(_path)
    return (pgh_revenue_data,)


@app.cell
def observe_states(mo):
    get_cells, set_cells = mo.state({key: cell["code"] for key, cell in STARTER_CELLS.items()})
    get_runs, set_runs = mo.state({key: 0 for key in STARTER_CELLS})
    get_outputs, set_outputs = mo.state({key: None for key in STARTER_CELLS})
    get_submitted, set_submitted = mo.state(False)
    # Plain containers, read when needed: the helper chat must never be recreated (that wipes its history).
    helper_context = {"base": "", "csv": "", "cells": {}, "settings": {}}
    helper_log = []
    return (
        get_cells,
        get_outputs,
        get_runs,
        get_submitted,
        helper_context,
        helper_log,
        set_cells,
        set_outputs,
        set_runs,
        set_submitted,
    )


@app.cell
def cell_controls(mo, pgh_revenue_data, set_cells, set_outputs, set_runs):
    # Plain dicts: the cell that renders these must not re-run while the learner types.
    cell_editors = {
        key: mo.ui.code_editor(
            value=cell["code"], language="python", min_height=110,
            on_change=functools.partial(record_field, set_cells, key),
        )
        for key, cell in STARTER_CELLS.items()
    }
    cell_run_buttons = {
        key: mo.ui.run_button(
            label="Run",
            on_change=functools.partial(run_and_record, cell_editors[key], pgh_revenue_data, set_outputs, set_runs, key),
        )
        for key in STARTER_CELLS
    }
    return cell_editors, cell_run_buttons


@app.cell
def model_controls(mo):
    column_choice = mo.ui.radio(
        options={label: key for key, label in METHOD_LABELS.items() if key != "none"}, value=None,
        label="Revenue column",
    )
    lag_amount = mo.ui.slider(start=1, stop=5, step=1, value=1, label="Lag (quarters)", show_value=True)
    remove_missing = mo.ui.switch(label="Remove missing values")
    training_share = mo.ui.slider(start=0.1, stop=0.9, step=0.1, value=0.1, label="Training share", show_value=True)
    return column_choice, lag_amount, remove_missing, training_share


@app.cell(hide_code=True)
def observe_intro(get_prediction, mo, pgh_revenue_data):
    mo.stop(get_prediction() is None)
    mo.vstack([
        mo.Html(stage_header_html(2)),
        mo.Html(
            "<div class='ss ss-prompt'>Here is the city's quarterly revenue as each method extracted it. "
            "Explore it with the five code cells below: run each one as it is, then change it to dig deeper. "
            "The question beside each cell is a hint about where to look. Then build your forecast with the "
            "controls further down. The AI helper can explain code, errors and ideas, but the choices are yours.</div>"
        ),
        pgh_revenue_data,
    ])
    return


@app.cell(hide_code=True)
def observe_cells(cell_editors, cell_run_buttons, get_cells, get_outputs, get_prediction, get_submitted, mo):
    mo.stop(get_prediction() is None)
    _panels = []
    for _key, _cell in STARTER_CELLS.items():
        _result = get_outputs()[_key]
        _shown = [mo.md("*Not run yet.*")]
        if _result is not None:
            _shown = []
            if _result["stdout"]:
                _shown.append(mo.md(f"```\n{_result['stdout']}\n```"))
            if _result["result"] is not None:
                _shown.append(_result["result"] if isinstance(_result["result"], (pd.DataFrame, pd.Series)) else mo.md(f"```\n{_result['value']}\n```"))
            if _result["figure"] is not None:
                _shown.append(mo.Html(_result["figure"]))
            if _result["error"]:
                _shown.append(mo.Html(f"<div class='ss ss-note'><strong>Error.</strong> {html.escape(_result['error'])}</div>"))
            if not _shown:
                _shown = [mo.md("*Ran with no output.*")]
        _panels.append(mo.vstack([
            mo.Html(f"<h3 class='ss ss-task-title'>{html.escape(_cell['title'])}</h3>"
                    f"<div class='ss ss-note'>{html.escape(_cell['question'])}</div>"),
            *([mo.md(f"```python\n{get_cells()[_key]}\n```")] if get_submitted() else [cell_editors[_key], cell_run_buttons[_key]]),
            *_shown,
        ]))
    mo.vstack(_panels, gap=1.5)
    return


@app.cell(hide_code=True)
def forecast_model(
    column_choice,
    get_prediction,
    get_submitted,
    lag_amount,
    mo,
    pgh_revenue_data,
    remove_missing,
    training_share,
):
    mo.stop(get_prediction() is None)
    _controls = (
        mo.md(
            f"**Your settings (locked):** {METHOD_LABELS.get(column_choice.value, 'no column chosen')} · lag "
            f"{lag_amount.value} · missing values {'removed' if remove_missing.value else 'kept'} · "
            f"{training_share.value:.0%} training"
        )
        if get_submitted()
        else mo.vstack([column_choice, lag_amount, remove_missing, training_share])
    )
    if column_choice.value is None:
        _result = mo.Html("<div class='ss ss-note'>Choose a revenue column to build the model.</div>")
    else:
        _frame = build_lagged(pgh_revenue_data, column_choice.value, lag_amount.value, remove_missing.value)
        try:
            _score = score_model(_frame, training_share.value)
            _forecast = next_quarter_forecast(pgh_revenue_data, column_choice.value, lag_amount.value)
            _forecast_text = (
                f"**Forecast for {_forecast['label']}:** ${_forecast['value']:,.0f}  \n*{_forecast['reason']}*"
                if _forecast
                else f"**No forecast possible.** {forecast_unavailable_reason(pgh_revenue_data, column_choice.value, lag_amount.value)}"
            )
            _result = mo.md("\n".join([
                f"**Error (RMSE): ${_score['rmse']:,.0f}**, trained on {_score['train_rows']} quarters and tested on {_score['test_rows']}.",
                "",
                f"- On average, the predictions for the test quarters are off by ${_score['rmse']:,.0f}.",
                f"- The average actual revenue in those quarters is ${_score['test_mean']:,.0f}, so the error is "
                f"{_score['rmse'] / _score['test_mean'] * 100:.1f}% of it.",
                f"- The test quarters range from ${_score['test_min']:,.0f} to ${_score['test_max']:,.0f}.",
                "",
                _forecast_text,
            ]))
        except ValueError as _error:
            _result = mo.Html(f"<div class='ss ss-note'>{html.escape(str(_error))}</div>")
    mo.vstack([
        mo.Html("<h3 class='ss ss-task-title'>Build your forecast</h3>"),
        mo.Html("<div class='ss ss-prompt'>Choose which column to trust, how far back the model looks (the lag), "
                "whether to drop rows with missing values, and how much of the history to train on. The model "
                "trains on the earliest quarters and is tested on the latest ones.</div>"),
        _controls,
        _result,
    ])
    return


@app.cell
def update_helper_context(
    column_choice,
    get_cells,
    get_model_base,
    get_outputs,
    helper_context,
    lag_amount,
    pgh_revenue_data,
    remove_missing,
    training_share,
):
    helper_context["base"] = get_model_base()
    helper_context["csv"] = pgh_revenue_data.to_csv(index=False)
    helper_context["cells"] = {
        key: {"code": get_cells()[key], "output": output_text(get_outputs()[key])} for key in STARTER_CELLS
    }
    helper_context["settings"] = {
        "column": column_choice.value, "lag": lag_amount.value,
        "remove_missing": remove_missing.value, "training_share": training_share.value,
    }
    return


@app.cell
def helper_chat(helper_context, helper_log, mo):
    async def helper_reply(messages, config):
        helper_log.append(messages[-1].content)
        if not helper_context["base"]:
            return "Connect your model at the top of the page (**Test connection**) to use the helper."
        system = helper_system_prompt(helper_context["csv"], helper_context["cells"], helper_context["settings"])
        history = helper_history(system, [{"role": m.role, "content": m.content} for m in messages])
        try:
            return await ask_model(helper_context["base"], history)
        except ModelUnavailable as error:
            return f"I can't reach your model right now. {error}"

    helper = mo.ui.chat(
        helper_reply,
        prompts=["What does a lag of 4 mean?", "How do I compare two columns row by row?", "What does this error mean?"],
    )
    return (helper,)


@app.cell(hide_code=True)
def helper_panel(get_prediction, helper, mo):
    mo.stop(get_prediction() is None)
    mo.vstack([
        mo.Html("<h3 class='ss ss-task-title'>Ask the AI helper</h3>"
                "<div class='ss ss-prompt'>Your model on CRCD can see the data, your code in the five cells, and "
                "your current settings. Ask it to explain code, errors or ideas. It won't make your choices for "
                "you.</div>"),
        helper,
    ])
    return


@app.cell(hide_code=True)
def observe_gate(get_prediction, get_runs, mo):
    mo.stop(get_prediction() is None)
    _waiting = [cell["title"] for key, cell in STARTER_CELLS.items() if get_runs()[key] == 0]
    (
        mo.Html(f"<p class='ss ss-gate'>Run every code cell at least once to continue to Stage 3. Still to run: "
                f"{html.escape(', '.join(_waiting))}.</p>")
        if _waiting
        else mo.Html("<p class='ss ss-gate'>You've run every cell. Stage 3 is below.</p>")
    )
    return


@app.cell
def explain_inputs(mo):
    get_explain, set_explain = mo.state({"r1": "", "r2": "", "r3": ""})
    submit_button = mo.ui.run_button(label="Submit for evaluation", kind="success")
    return get_explain, set_explain, submit_button


@app.cell
def explain_fields(mo, set_explain):
    # Plain dict: the explain cell must not re-run while the learner types.
    explain_fields = {
        f"r{i}": mo.ui.text_area(rows=5, full_width=True, on_change=functools.partial(record_field, set_explain, f"r{i}"))
        for i in range(1, len(EXPLAIN_PROMPTS) + 1)
    }
    return (explain_fields,)


@app.cell
def submit_action(get_explain, mo, set_submitted, submit_button):
    _ready = can_submit(get_explain())
    if submit_button.value and _ready:
        set_submitted(True)
    submit_warning = (
        mo.Html("<div class='ss ss-note'>Answer all three questions before you submit.</div>")
        if submit_button.value and not _ready
        else ""
    )
    return (submit_warning,)


@app.cell(hide_code=True)
def explain(
    explain_fields,
    get_explain,
    get_prediction,
    get_runs,
    get_submitted,
    mo,
    submit_button,
    submit_warning,
):
    mo.stop(not explain_unlocked(get_prediction(), get_runs()))
    _header = [
        mo.Html(stage_header_html(3)),
        mo.Html("<div class='ss ss-prompt'>Look back at what you predicted before you saw the data, and at what you "
                "found. Write for yourself and for Chris. Nobody will grade the wording, only whether your "
                "reasoning holds up.</div>"),
        mo.Html(locked_prediction_html(get_prediction())),
    ]
    if get_submitted():
        _answers = "".join(
            f"<div class='ss-label'><i>{i}</i><span>{html.escape(q)}</span></div>"
            f"<div class='ss-indent'>{paragraph_html(get_explain()[f'r{i}'])}</div>"
            for i, q in enumerate(EXPLAIN_PROMPTS, 1)
        )
        _output = mo.vstack(_header + [
            mo.Html(f"<div class='ss'>{_answers}</div>"),
            mo.Html("<div class='ss ss-note ok'><strong>Submitted.</strong> Your choices and answers are locked.</div>"),
        ])
    else:
        _questions = "".join(
            f"<div class='ss-label'><i>{i}</i><span>{html.escape(q)}</span></div>"
            f"<div class='ss-indent'>{explain_fields[f'r{i}']}</div>"
            for i, q in enumerate(EXPLAIN_PROMPTS, 1)
        )
        _output = mo.vstack(_header + [
            mo.Html(f"<div class='ss'>{_questions}</div>"),
            mo.md("Submitting locks your code cells, your forecast settings and these answers, then shows your evaluation."),
            submit_button,
            submit_warning,
        ])
    _output
    return


@app.cell(hide_code=True)
def evaluate(
    column_choice,
    get_cells,
    get_runs,
    get_submitted,
    lag_amount,
    mo,
    pgh_revenue_data,
    remove_missing,
    training_share,
):
    mo.stop(not get_submitted())
    final_settings = {
        "column": column_choice.value, "lag": lag_amount.value,
        "remove_missing": remove_missing.value, "training_share": training_share.value,
    }
    mo.vstack([
        mo.Html(stage_header_html(4)),
        mo.Html("<div class='ss ss-prompt'>Here is how your forecast holds up. Part 1 covers only the settings worth "
                "a second look. Part 2 shows why the error number alone can mislead. Part 3 shows what each code "
                "cell could reveal.</div>"),
        mo.Html(evaluation_html(pgh_revenue_data, final_settings, get_cells(), get_runs())),
    ])
    return (final_settings,)


@app.cell(hide_code=True)
def download_brief(final_settings, get_cells, get_explain, get_prediction, get_runs, helper_log, mo, pgh_revenue_data):
    _doc = report_html(pgh_revenue_data, get_prediction(), final_settings, get_explain(), helper_log, get_cells(), get_runs())
    mo.vstack([
        mo.Html("<h3 class='ss ss-task-title'>Download your brief</h3>"),
        mo.md("The file holds your prediction, your forecast, your reflection, your questions to the helper and "
              "this evaluation. It opens in any browser; use **Print → Save as PDF** for a PDF."),
        mo.download(data=_doc.encode("utf-8"), filename="revenue-forecast-brief.html", mimetype="text/html",
                    label="Download brief (.html)"),
        mo.md("*Nothing is saved online. Refreshing or closing this page erases your work, so download before you leave.*"),
    ])
    return


if __name__ == "__main__":
    app.run()
