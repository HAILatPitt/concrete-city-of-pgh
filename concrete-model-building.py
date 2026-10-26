import marimo

__generated_with = "0.18.0"
app = marimo.App(width="medium", app_title="ConCReTE Model Building")

with app.setup:
    import ast
    import base64
    import codecs
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
        **This scenario includes an AI helper that runs on your own model on Pitt's CRCD GPU cluster.** Setting it up takes about 15 minutes the first time, mostly for the one-time model download, and about 3 minutes after that. You can do the whole data task without it.

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
        "<link rel='stylesheet' href='https://fonts.googleapis.com/css2?family=Hanken+Grotesk:wght@400;500;600;700"
        "&family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,500;0,6..72,600;1,6..72,400&display=swap'>"
    )
    # An investigation's palette: navy ink, pink only for what deserves a second look, tan for metadata.
    PALETTE = (
        ".ss{--ink:#242a49;--pink:#ff335f;--pink-lt:#fbd7de;--crimson:#801930;--tan:#66615b;--tan-lt:#f3f2f1;"
        "--rule:#c2c0bd;--slate:#7b89a1;--serif:'Newsreader',Georgia,serif;"
        "--sans:'Hanken Grotesk',system-ui,sans-serif;"
        "--dots:radial-gradient(circle,#242a49 .9px,transparent 1.2px) 0 0/3px 3px;"
        "color:var(--ink);font-family:var(--serif)}"
    )
    # The cards render inside the marimo button's shadow root, so they carry their own styles, and the
    # button rule below reaches only that one button. Its open/closed look comes from custom properties
    # set on the slot outside (card_slot_style), which inherit into the shadow root.
    CARD_STYLE = (
        "button{display:block!important;width:100%!important;height:auto!important;padding:0!important;"
        "border:0!important;background:none!important;box-shadow:none!important;text-align:left!important;"
        "white-space:normal!important;font-weight:400!important}"
        ".ss-ex{position:relative;isolation:isolate;padding:0 12px 12px 0}"
        ".ss-ex::before{content:'';position:absolute;left:12px;top:12px;right:0;bottom:0;background:var(--dots);"
        "opacity:calc(var(--ss-open,0) * .55);z-index:-1;transition:opacity .15s}"
        ".ss-ex-card{display:grid;grid-template-columns:64px 1fr 150px;column-gap:24px;align-items:start;"
        "background:#fff;border:1px solid var(--ink);padding:24px 28px;transition:transform .15s;"
        "transform:translate(calc(var(--ss-open,0) * -4px),calc(var(--ss-open,0) * -4px))}"
        "button:hover .ss-ex-card{transform:translate(-4px,-4px)}button:hover .ss-ex::before{opacity:.55}"
        ".ss-ex-num{font:italic 400 40px/1 var(--serif);color:var(--ink)}"
        ".ss-ex-title{font:600 25px/1.2 var(--serif);color:var(--ink);margin:0 0 6px}"
        ".ss-ex-short{font:400 17px/1.55 var(--serif);color:var(--ink)}"
        ".ss-ex-cue{justify-self:end;align-self:center;display:flex;align-items:center;gap:10px;white-space:nowrap;"
        "font:600 14px/1 var(--sans);color:var(--ink)}"
        ".ss-cue-closed{display:var(--ss-cue-closed,inline)}.ss-cue-open{display:var(--ss-cue-open,none)}"
        ".ss-ex-cue i{width:20px;height:20px;border:1.5px solid var(--ink);border-radius:50%;"
        "background:linear-gradient(var(--ink),var(--ink)) center/9px 1.5px no-repeat,"
        "linear-gradient(var(--ink),var(--ink)) center/1.5px var(--ss-plus,9px) no-repeat}"
        ".ss-ex-full{grid-column:2 / 4;display:var(--ss-full,none);margin-top:18px;padding-top:18px;"
        "border-top:1px solid var(--rule);max-width:760px;font:400 17px/1.6 var(--serif);color:var(--ink)}"
        "@media (max-width:700px){.ss-ex-card{grid-template-columns:36px 1fr;padding:18px}"
        ".ss-ex-cue{grid-column:2;justify-self:start;margin-top:12px}.ss-ex-full{grid-column:1 / 3}}"
    )
    # Page-level blocks (light DOM), injected once by the title cell and again in the downloaded brief.
    BRIEF_STYLE = (
        ".ss-lab{font:600 13px/1.3 var(--sans);letter-spacing:.06em;text-transform:uppercase;color:var(--tan);margin:0}"
        ".ss-lab.cr{color:var(--crimson)}"
        ".ss-swy{display:flex;align-items:center;gap:8px}"
        ".ss-swy::before{content:'';width:22px;height:4px;background:var(--pink)}"
        ".ss-hero{display:grid;grid-template-columns:1fr 300px;column-gap:56px;align-items:end;margin-bottom:28px}"
        ".ss-kicker{display:flex;align-items:center;gap:10px;font:600 15px/1 var(--sans);color:var(--crimson);"
        "margin:0 0 18px}"
        ".ss-kicker::before{content:'';width:10px;height:10px;background:var(--pink)}"
        ".ss-title{font:500 66px/1.02 var(--serif);letter-spacing:-.015em;margin:0;padding:0;border:0;color:var(--ink)}"
        ".ss-byline{font:400 15px/1.5 var(--sans);color:var(--tan);border-top:1px solid var(--ink);"
        "padding-top:14px;margin:0}"
        ".ss-byline b{color:var(--ink);font-weight:600}"
        ".ss-sources{list-style:none;margin:10px 0 0;padding:0;display:grid;grid-template-columns:1fr 1fr;"
        "gap:4px 18px;font:400 15px/1.5 var(--sans);color:var(--ink)}"
        ".ss-sources li::before{content:'';display:inline-block;width:6px;height:6px;border-radius:50%;"
        "background:var(--slate);margin-right:8px;vertical-align:2px}"
        ".ss-gate{margin:20px 0 0;text-align:center;font:500 15px/1.4 var(--sans);color:var(--tan)}"
        ".ss-gate::before{content:'\\2022\\2002\\2022\\2002\\2022\\2002\\2022';display:block;"
        "letter-spacing:.2em;color:var(--ink);font-size:20px;margin-bottom:10px}"
        ".ss-stage{border-top:4px solid var(--ink);padding-top:20px;margin-top:64px}"
        ".ss-stage-top{display:flex;justify-content:space-between;align-items:baseline}"
        ".ss-stage-title{font:500 52px/1.05 var(--serif);letter-spacing:-.01em;margin:10px 0 0;padding:0;border:0;"
        "color:var(--ink)}"
        ".ss-progress{list-style:none;margin:28px 0 0;padding:0;display:grid;grid-template-columns:repeat(4,1fr);"
        "position:relative}"
        ".ss-progress::before{content:'';position:absolute;left:6px;right:calc(25% - 6px);top:6px;height:1px;"
        "background:var(--rule)}"
        ".ss-progress li{position:relative;padding-top:26px;font:400 15px/1.3 var(--sans);color:var(--tan);margin:0}"
        ".ss-progress li::before{content:'';position:absolute;left:0;top:0;width:11px;height:11px;border-radius:50%;"
        "border:1.5px solid var(--tan);background:#fff}"
        ".ss-progress li b{display:block;font:600 13px/1.3 var(--sans);letter-spacing:.06em;text-transform:uppercase}"
        ".ss-progress .done{color:var(--ink)}"
        ".ss-progress .done::before{background:var(--ink);border-color:var(--ink)}"
        ".ss-progress .now{color:var(--ink);font-weight:600}"
        ".ss-progress .now::before{background:var(--pink);border-color:var(--pink);width:15px;height:15px;"
        "top:-2px;left:-2px}"
        ".ss-prompt{margin-top:28px;max-width:800px;font:400 19px/1.55 var(--serif);color:var(--ink)}"
        ".ss-prompt strong{font-weight:600}"
        ".ss-prompt ul{list-style:none;counter-reset:q;margin:20px 0;padding:26px 30px;background:var(--tan-lt);"
        "display:grid;grid-template-columns:1fr 1fr;gap:18px 40px}"
        ".ss-prompt li{counter-increment:q;display:grid;grid-template-columns:34px 1fr;font-size:17px;"
        "line-height:1.5;margin:0}"
        ".ss-prompt li::before,.ss-label i{content:counter(q);display:grid;place-items:center;width:24px;height:24px;"
        "border:1.5px solid var(--ink);border-radius:50%;font:700 13px/1 var(--sans);font-style:normal;margin-top:2px}"
        ".ss-label{display:grid;grid-template-columns:40px 1fr;align-items:start;margin:28px 0 10px}"
        ".ss-label span{font:600 18px/1.4 var(--serif);color:var(--ink)}"
        ".ss-indent{padding-left:40px}"
        ".ss-exhibit{position:relative;isolation:isolate;margin-top:32px;padding:0 12px 12px 0}"
        ".ss-exhibit::before{content:'';position:absolute;left:12px;top:12px;right:0;bottom:0;"
        "background:var(--dots);opacity:.5;z-index:-1}"
        ".ss-exhibit-in{background:#fff;border:1px solid var(--ink);padding:20px 26px 22px}"
        ".ss-exhibit h4{font:italic 500 17px/1.3 var(--serif);color:var(--ink);margin:16px 0 4px}"
        ".ss-exhibit p:not(.ss-lab){font:400 16px/1.55 var(--sans);color:var(--ink);margin:0}"
        ".ss-task-title{font:600 30px/1.15 var(--serif);color:var(--ink);margin:6px 0 8px;padding:0;border:0}"
        ".ss-head{margin-top:56px}"
        ".ss-desc{font:400 17px/1.55 var(--serif);color:var(--ink);max-width:760px;margin:0 0 14px}"
        ".ss-hint{font:500 16px/1.5 var(--sans);color:var(--ink);max-width:820px;margin:6px 0 14px}"
        ".ss-hint b{display:block;font:600 13px/1.3 var(--sans);letter-spacing:.06em;text-transform:uppercase;"
        "color:var(--tan);margin-bottom:4px}"
        ".ss-brief{background:var(--tan-lt);padding:22px 28px 24px;font:400 16px/1.6 var(--sans);color:var(--ink)}"
        ".ss-brief p{margin:0 0 12px}"
        ".ss-brief code{background:#fff;padding:1px 5px}"
        ".ss-note{margin:14px 0 0;background:var(--pink-lt);padding:12px 16px;font:400 15px/1.55 var(--sans);"
        "color:var(--ink)}"
        ".ss-note strong{color:var(--crimson)}"
        ".ss-note.ok{background:var(--tan-lt)}.ss-note.ok strong{color:var(--ink)}"
        ".ss-result{display:grid;grid-template-columns:1fr 1fr;border-top:1px solid var(--ink);"
        "border-bottom:1px solid var(--ink);margin-top:20px}"
        ".ss-result > div{padding:26px 32px 28px 0}"
        ".ss-result > div + div{border-left:1px solid var(--rule);padding-left:32px}"
        ".ss-big{font:500 52px/1 var(--serif);letter-spacing:-.01em;font-variant-numeric:tabular-nums lining-nums;"
        "color:var(--ink);margin:12px 0 14px}"
        ".ss-big span{background:linear-gradient(transparent 62%,var(--pink-lt) 62%)}"
        ".ss-result p:not(.ss-lab):not(.ss-big){font:400 16px/1.55 var(--sans);color:var(--ink);margin:0 0 6px}"
        ".ss-result .ss-meta{color:var(--tan)}"
        "@media (max-width:700px){.ss-hero{grid-template-columns:1fr;row-gap:24px}"
        ".ss-title{font-size:44px}.ss-stage-title{font-size:40px}.ss-prompt ul{grid-template-columns:1fr}"
        ".ss-result{grid-template-columns:1fr}.ss-result > div + div{border-left:0;border-top:1px solid var(--rule);"
        "padding-left:0}.ss-big{font-size:40px}}"
    )
    STAGES = ("Predict", "Observe", "Explain", "Evaluate")
    EVAL_STYLE = (
        "<style>.poee-eval .grp{font:600 13px/1.3 var(--sans);letter-spacing:.06em;text-transform:uppercase;"
        "color:var(--crimson);margin:48px 0 14px}"
        ".poee-eval .sum{position:relative;isolation:isolate;padding:0 12px 12px 0}"
        ".poee-eval .sum::before,.poee-eval .verdict::before{content:'';position:absolute;left:12px;top:12px;right:0;"
        "bottom:0;background:var(--dots);opacity:.55;z-index:-1}"
        ".poee-eval .sum > div{background:#fff;border:1px solid var(--ink);padding:22px 28px}"
        ".poee-eval .big{font:500 30px/1.2 var(--serif);color:var(--ink)}"
        ".poee-eval .verdict{position:relative;isolation:isolate;padding:0 12px 12px 0;margin:0 0 22px}"
        ".poee-eval .verdict-in{background:#fff;border:1px solid var(--ink)}"
        ".poee-eval .verdict-top{display:flex;justify-content:space-between;align-items:center;gap:20px;"
        "padding:16px 26px;border-bottom:1px solid var(--ink)}"
        ".poee-eval .lab{font:600 13px/1.3 var(--sans);letter-spacing:.06em;text-transform:uppercase;"
        "color:var(--tan);margin:0}"
        ".poee-eval h4{font:600 24px/1.25 var(--serif);color:var(--ink);margin:4px 0 0}"
        ".poee-eval .stamp{font:700 13px/1 var(--sans);letter-spacing:.08em;text-transform:uppercase;"
        "color:var(--crimson);border:1.5px solid var(--crimson);padding:6px 10px;white-space:nowrap}"
        ".poee-eval .rows{display:grid;grid-template-columns:repeat(3,1fr)}"
        ".poee-eval .rows > div{padding:20px 26px 24px}"
        ".poee-eval .rows > div + div{border-left:1px solid var(--rule)}"
        ".poee-eval .rows p:not(.lab){font:400 16px/1.55 var(--serif);color:var(--ink);margin:10px 0 0}"
        ".poee-eval .rows .sound{background:var(--tan-lt)}"
        ".poee-eval .why{font:400 18px/1.5 var(--serif);color:var(--ink);margin:0 0 12px;max-width:800px}"
        ".poee-eval table{width:100%;border-collapse:collapse}"
        ".poee-eval th{font:600 13px/1.3 var(--sans);letter-spacing:.06em;text-transform:uppercase;color:var(--tan);"
        "text-align:left;padding:10px 16px 10px 0;border-bottom:1px solid var(--ink)}"
        ".poee-eval td{padding:14px 16px 14px 0;border-bottom:1px solid var(--rule);vertical-align:top;"
        "font:400 16px/1.5 var(--serif);color:var(--ink)}"
        ".poee-eval td.set{font:600 15px/1.4 var(--sans);width:24%}"
        ".poee-eval td.num{font:500 22px/1.1 var(--serif);white-space:nowrap;text-align:right;width:9%;"
        "font-variant-numeric:tabular-nums lining-nums}"
        ".poee-eval td.barcell{width:22%;padding-top:20px}"
        ".poee-eval .bar{height:12px;background:var(--pink)}"
        ".poee-eval tr.base td{background:var(--tan-lt)}.poee-eval tr.base .bar{background:var(--ink)}"
        ".poee-eval .legend{font:400 14px/1.5 var(--sans);color:var(--tan);margin:10px 0 0}"
        ".poee-eval .rev{display:grid;grid-template-columns:260px 1fr;column-gap:40px;padding:22px 0;"
        "border-bottom:1px solid var(--rule)}"
        ".poee-eval .rev h3{font:600 20px/1.3 var(--serif);color:var(--ink);margin:0;padding:0;border:0}"
        ".poee-eval .rev p{font:400 18px/1.55 var(--serif);color:var(--ink);margin:0}"
        ".poee-eval .rev .fact{font:400 14px/1.5 var(--sans);color:var(--tan);margin-top:8px}"
        ".poee-eval .num{white-space:nowrap}"
        "@media (max-width:700px){.poee-eval .rows,.poee-eval .rev{grid-template-columns:1fr}"
        ".poee-eval .rows > div + div{border-left:0;border-top:1px solid var(--rule)}"
        ".poee-eval td.barcell,.poee-eval th:nth-child(3){display:none}}</style>"
    )

    CARDS = {
        "story": {
            "numeral": "i",
            "title": "The story so far",
            "back": (
                "Pittsburgh's data office wants to forecast the city's revenue for each quarter. The figures sit in "
                "years of PDF revenue reports, and the team has already copied them out six different ways."
            ),
            "full": (
                "This is part of the city's three-year data strategy, led by Chris Belasco, Senior Manager of Digital "
                "Services and Chief Data Officer. The strategy uses data to improve city services, involve residents "
                "and promote equity. For this task his team took the Q4 Revenue Summary Table from each Revenue "
                "Report PDF. One person typed the totals in by hand, and five tools read them automatically: Docling, "
                "ChatGPT, Claude chat, the Claude API and pypdf."
            ),
        },
        "role": {
            "numeral": "ii",
            "title": "Your role",
            "back": "You're an intern for Chris Belasco, the city's Chief Data Officer.",
            "full": (
                "You get a table of the city's revenue for every quarter from 2012 to 2024. It has six revenue "
                "columns, one per extraction method, and they don't always agree. You decide which column to trust "
                "and build a simple model from it that predicts next quarter's revenue."
            ),
        },
        "deliver": {
            "numeral": "iii",
            "title": "What you deliver",
            "back": "A forecast of next quarter's revenue, and a short note telling Chris how far to trust it.",
            "full": (
                "Chris will act on your note, so he needs to know which numbers your forecast rests on and what "
                "might be wrong with them. Be ready to explain why you trusted the column you chose."
            ),
        },
    }
    METHOD_LABELS = {
        "revenue by hand": "By hand (a person typed them in)",
        "revenue by docling": "Docling (software that reads documents)",
        "revenue by chatgpt": "ChatGPT (asked once in a chat)",
        "revenue by claude chat": "Claude chat (asked once in a chat)",
        "revenue by claude api": "Claude API (a program that calls Claude)",
        "revenue by pypdf": "pypdf (a script that pulls text out of PDFs)",
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
            "question": "Is the hand-typed column always right? Look for quarters where it disagrees with all the others.",
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
            "question": "This error is lower than the one the controls below give you. Did the model improve, or did the test get easier?",
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
        "Which method did you end up trusting? What in the data changed or confirmed your prediction?",
        "What are your thoughts on using data extracted by AI? Did it affect the quality of your model?",
        "Write a short note to Chris: your forecast, how confident he should be in it, and why.",
    ]

    SOUND_COLUMNS = ("revenue by hand", "revenue by claude chat", "revenue by claude api", "revenue by pypdf")
    SOUND_COLUMN_TEXT = (
        "By hand, Claude chat, Claude API or pypdf. These four columns match on every quarter except 2017 Q2, where "
        "the hand-typed value is $113,830,337 and the machine-read columns show $113,830,377."
    )
    COLUMN_VERDICTS = {
        "revenue by docling": {
            "why": "Docling read 2018 Q4 as $6,425 (the other sources say $95.4 million) and 2021 Q4 as $608.6 "
                   "million (the others say $129.7 million). It is also missing every quarter of 2013, 2017 and 2024.",
            "cost": "Your model learned from one quarter where the city seemed to collect almost nothing and another "
                    "at four times its real size. Depending on the lag and split, those quarters land either in the "
                    "training years, where the error figure can't see them, or in the test, where they push the "
                    "error up with no sign of why. A budget built on this forecast would rest on numbers nobody checked.",
        },
        "revenue by chatgpt": {
            "why": "ChatGPT misread all four quarters of 2019 (for example $12.9 million where the reports say $122.2 million) "
                   "and is missing every quarter of 2013, 2014, 2022 and 2024.",
            "cost": "At some lags and splits this column has the lowest error of the six, so it can look like the best "
                    "choice. Chris would think the forecast is more reliable than it is, and the city could plan "
                    "spending around revenue it never collected.",
        },
        None: {
            "why": "You didn't choose a column, so no model was built.",
            "cost": "Chris would have no forecast to plan with. Choosing which numbers to trust is the first decision "
                    "in this task.",
        },
    }
    MISSING_VERDICT = {
        "sound": "Remove rows with missing values, after checking which quarters they are.",
        "why": "A linear regression can't train on blank quarters, so with the blanks kept the model produced no "
               "error figure and no forecast.",
        "cost": "With nothing to give Chris, someone ends up filling the blanks by guessing, and that is how "
                "unchecked numbers get into a budget.",
    }
    LAG_VERDICT = {
        "sound": "A lag of 4, which predicts each quarter from the same quarter a year earlier.",
        "why": "City revenue follows a yearly pattern: Q1 averages $214.9 million, against $104 to $132 million for "
               "the other quarters. Other lags compare quarters that don't match. For the hand-typed column, lag 4 "
               "roughly halves the error ($85.6 million at lag 2, $42.2 million at lag 4).",
        "cost": "The forecast would miss the seasonal swing. It would predict a quiet quarter after a busy Q1 and a "
                "busy one after a quiet quarter, so cash planning would be wrong in both directions.",
    }
    SPLIT_VERDICTS = {
        "small": {
            "why": "Training on so little history gives the model only a handful of quarters to learn from. The error "
                   "can still look good (at 10%, the hand-typed column scores $27.9 million after learning from just 4 "
                   "quarters), but that score comes from luck.",
            "cost": "Chris would get a forecast that looks well tested and rests on only a year or two of history.",
        },
        "large": {
            "why": "Training on 90% leaves only about 5 quarters to test on, so the error figure depends on which "
                   "quarters happen to come last (the hand-typed column scores $11.7 million).",
            "cost": "The error figure would make the forecast sound far more reliable than it is.",
        },
        "sound": "50% to 80%, which leaves enough quarters to learn from and enough recent quarters to test on.",
    }
    REVEALS = {
        "compare": "An average can hide a single bad value. Docling records 2018 Q4 as $6,425 instead of $95.4 "
                   "million, and ChatGPT misreads 2019 (for example $12.9 million instead of $122.2 million). You "
                   "only see these by comparing the columns row by row.",
        "plot": "Plotting every column shows spikes and drops no real budget has, and a pattern that repeats every "
                "year: Q1 is by far the biggest quarter. That pattern is why a lag of 4, the same quarter last year, "
                "works best.",
        "truth": "In 2017 Q2 every machine method that read 2017 records $113,830,377 (Docling skipped the year), and "
                 "only the hand-typed value differs ($113,830,337). The person most likely made a typo, so the "
                 "hand-typed column needs checking too.",
        "missing": "The blanks aren't random. Docling and ChatGPT miss whole years, so dropping rows with missing "
                   "values removes those years from the series, and a comparison with the year before then lines up "
                   "the wrong quarters.",
        "split": "A random split puts later quarters into training, so the model is tested on quarters surrounded by "
                 "ones it has already seen. Its $29.0 million error looks better than the $42.2 million you get by "
                 "training on earlier quarters and testing on later ones, because the test got easier.",
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
        "reason": f"Based on {inputs['source_label']}, {quarters_text(lag)} earlier. For this forecast the model is "
                  "refit on every usable quarter.",
    }


@app.function
def quarters_text(count: int) -> str:
    return "1 quarter" if count == 1 else f"{count} quarters"


@app.function
def forecast_unavailable_reason(data, column: str, lag: int) -> str:
    inputs = forecast_inputs(data, column, lag)
    if inputs is None:
        return "This column has no values at all."
    if inputs["source_label"] is None:
        return f"The data has no quarter {quarters_text(lag)} before {inputs['label']} to predict from."
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
         "The lowest error of any column at these settings, even though this column misreads four quarters and misses "
         "whole years. The bad values are in the training years and the test covers other quarters, so the score "
         "can't show them."),
        ("Hand-typed column, lag 4, 10% training", chronological_rmse(data, "revenue by hand", 4, 0.1),
         "The model learned from only 4 quarters, so the low error is luck."),
        ("Hand-typed column, lag 4, 90% training", chronological_rmse(data, "revenue by hand", 4, 0.9),
         "Tested on only 5 quarters, so the score depends on which 5 happen to be at the end."),
        ("Hand-typed column, lag 4, random split", random_split_rmse(data, "revenue by hand", 4),
         "A random split lets the model learn from quarters that come after the ones it is tested on, so it has "
         "already seen the future."),
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
def model_unavailable(error: Exception) -> ModelUnavailable:
    if isinstance(error, urllib.error.HTTPError):
        return ModelUnavailable(f"Your model server answered with an error (HTTP {error.code}).")
    return ModelUnavailable(
        f"Couldn't reach your model ({type(error).__name__}). Look at your tunnel window (Terminal B): if it "
        "says open failed, the node name or port in the tunnel command is wrong, so copy it again from step 3. "
        "Otherwise check that Terminal B is still open and that llamacpp-status shows READY."
    )


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
        raise model_unavailable(error) from error


@app.function
async def list_model_name(base: str) -> str:
    body = await model_request(f"{base}/v1/models")
    try:
        return body["data"][0]["id"]
    except (KeyError, IndexError, TypeError) as error:
        raise ModelUnavailable("The server answered, but not like a llama.cpp model server.") from error


@app.function
def sse_text(line: str) -> str:
    """The text one streamed chunk adds; empty for keep-alives, role-only chunks and [DONE]."""
    data = line.strip()[len("data:"):].strip() if line.strip().startswith("data:") else ""
    if not data or data == "[DONE]":
        return ""
    choices = json.loads(data).get("choices") or [{}]
    return choices[0].get("delta", {}).get("content") or ""


@app.function
def urllib_open_stream(url: str, payload: dict):
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}, method="POST"
    )
    return urllib.request.urlopen(request, timeout=60)


@app.function
async def browser_stream_lines(url: str, payload: dict):
    from pyodide.http import pyfetch

    response = await pyfetch(
        url, method="POST", headers={"Content-Type": "application/json"}, body=json.dumps(payload)
    )
    if not response.ok:
        raise ModelUnavailable(f"Your model server answered with an error (HTTP {response.status}).")
    reader = response.js_response.body.getReader()
    decoder = codecs.getincrementaldecoder("utf-8")()
    pending = ""
    while True:
        chunk = await reader.read()
        if chunk.done:
            break
        pending += decoder.decode(chunk.value.to_bytes())
        *lines, pending = pending.split("\n")
        for line in lines:
            yield line
    yield pending


@app.function
async def local_stream_lines(url: str, payload: dict):
    response = await asyncio.to_thread(urllib_open_stream, url, payload)
    with response:
        while line := await asyncio.to_thread(response.readline):
            yield line.decode("utf-8", errors="replace")


@app.function
async def stream_model(base: str, messages: list[dict]):
    """Yield the reply so far, each time the model adds to it."""
    url = f"{base}/v1/chat/completions"
    payload = {"messages": messages, "temperature": 0.3, "stream": True}
    lines = browser_stream_lines(url, payload) if in_browser() else local_stream_lines(url, payload)
    reply = ""
    try:
        async for line in lines:
            piece = sse_text(line)
            if piece:
                reply += piece
                yield reply
    except ModelUnavailable:
        raise
    except Exception as error:
        raise model_unavailable(error) from error


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
    return (
        f"<style>{PALETTE}{CARD_STYLE}</style><div class='ss ss-ex'><div class='ss-ex-card'>"
        f"<span class='ss-ex-num'>{card['numeral']}</span>"
        f"<div><div class='ss-ex-title'>{html.escape(card['title'], quote=False)}</div>"
        f"<div class='ss-ex-short'>{html.escape(card['back'], quote=False)}</div></div>"
        "<span class='ss-ex-cue'><span class='ss-cue-closed'>Read in full</span>"
        "<span class='ss-cue-open'>Close</span><i></i></span>"
        f"<div class='ss-ex-full'>{html.escape(card['full'], quote=False)}</div></div></div>"
    )


@app.function
def card_slot_style(opened: bool) -> dict:
    # The open look is chosen by custom properties, which inherit into the button's shadow root.
    if opened:
        return {"width": "100%", "--ss-open": "1", "--ss-full": "block", "--ss-plus": "0px",
                "--ss-cue-open": "inline", "--ss-cue-closed": "none"}
    return {"width": "100%", "--ss-open": "0", "--ss-full": "none", "--ss-plus": "9px",
            "--ss-cue-open": "none", "--ss-cue-closed": "inline"}


@app.function
def bold_html(text: str) -> str:
    parts = html.escape(text, quote=False).split("**")
    return "".join(f"<strong>{part}</strong>" if i % 2 else part for i, part in enumerate(parts))


@app.function
def stage_header_html(stage: int) -> str:
    steps = "".join(
        f'<li class="{"now" if i == stage else "done" if i < stage else "next"}"><b>Stage {i}</b>{name}</li>'
        for i, name in enumerate(STAGES, 1)
    )
    return (
        f'<div class="ss ss-stage"><div class="ss-stage-top"><p class="ss-lab cr">Stage {stage} of 4</p></div>'
        f'<h2 class="ss-stage-title">{STAGES[stage - 1]}</h2><ol class="ss-progress">{steps}</ol></div>'
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
        '<div class="ss ss-exhibit"><div class="ss-exhibit-in"><p class="ss-lab cr">Your locked prediction</p>'
        f"<h4>The method I would trust</h4><p>{html.escape(METHOD_LABELS.get(prediction['method'], prediction['method']))}</p>"
        f"<h4>Why</h4><p>{paragraph_html(prediction['why'])}</p>"
        f"<h4>How accurate I expect the forecast to be</h4><p>{paragraph_html(prediction['accuracy'])}</p></div></div>"
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
def table_ready(value):
    """marimo 0.18 can't draw a pandas 3 table whose text index has no name (without pyarrow), so name it."""
    if isinstance(value, (pd.DataFrame, pd.Series)) and value.index.nlevels == 1 and value.index.name is None:
        return value.rename_axis("index")
    return value


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
        "result": table_ready(value),
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
        "<div class='verdict'><div class='verdict-in'><div class='verdict-top'>"
        f"<div><p class='lab'>{html.escape(card['title'])} · You chose</p><h4>{html.escape(card['chose'])}</h4></div>"
        "<span class='stamp'>Worth a second look</span></div><div class='rows'>"
        f"<div><p class='lab'>Why</p><p>{html.escape(card['why'])}</p></div>"
        f"<div><p class='lab'>What happens for the city</p><p>{html.escape(card['cost'])}</p></div>"
        f"<div class='sound'><p class='lab'>A sound choice</p><p>{html.escape(card['sound'])}</p></div>"
        "</div></div></div>"
    )


@app.function
def reveal_card_html(card: dict) -> str:
    fact = "You changed the starter code." if card["edited"] else "You ran the starter code as given."
    return (
        f"<div class='rev'><h3>{html.escape(card['title'])}</h3>"
        f"<div><p>{html.escape(card['finding'])}</p><p class='fact'>{fact}</p></div></div>"
    )


@app.function
def fooled_table_html(data) -> str:
    cases = fooled_rows(data)
    largest = max(r["reported_rmse_m"] for r in cases)
    rows = "".join(
        f"<tr class='{'base' if i == 0 else ''}'><td class='set'>{html.escape(r['setting'])}</td>"
        f"<td class='num'>${r['reported_rmse_m']:.1f}M</td>"
        f"<td class='barcell'><div class='bar' style='width:{r['reported_rmse_m'] / largest * 100:.1f}%'></div></td>"
        f"<td>{html.escape(r['what_is_going_on'])}</td></tr>"
        for i, r in enumerate(cases)
    )
    return (
        "<table><tr><th>Setting</th><th>Reported error</th><th></th><th>What's really going on</th></tr>"
        f"{rows}</table><p class='legend'>Dark bar: a sensible model, tested on the latest quarters. "
        "Pink bars: settings that score better for the wrong reason.</p>"
    )


@app.function
def evaluation_html(data, settings: dict, cells: dict, runs: dict) -> str:
    cards = settings_cards(settings["column"], settings["lag"], settings["remove_missing"], settings["training_share"])
    part1 = "".join(settings_card_html(c) for c in cards) or (
        "<div class='sum'><div><div class='big'>Every setting holds up.</div></div></div>"
    )
    part3 = "".join(reveal_card_html(c) for c in reveal_cards(cells, runs))
    return (
        f"<style>{PALETTE}</style>{EVAL_STYLE}<div class='ss poee-eval'>"
        f"<div class='grp'>Part 1 · Your forecast settings</div>{part1}"
        "<div class='grp'>Part 2 · The number that fooled you</div>"
        "<p class='why'>A lower error is not proof of a better forecast. These settings all "
        "\"score\" well for the wrong reasons:</p>"
        f"{fooled_table_html(data)}"
        f"<div class='grp'>Part 3 · What the data was hiding</div>{part3}</div>"
    )


@app.function
def forecast_result_html(score: dict, forecast: dict | None, unavailable: str) -> str:
    share = score["rmse"] / score["test_mean"] * 100
    left = (
        "<div><p class='ss-lab'>Error (RMSE)</p>"
        f"<p class='ss-big'>${score['rmse']:,.0f}</p>"
        f"<p>On average, the predictions for the test quarters are off by ${score['rmse']:,.0f}.</p>"
        f"<p>The average actual revenue in those quarters is ${score['test_mean']:,.0f}, so the error is "
        f"{share:.1f}% of it.</p>"
        f"<p>The test quarters range from ${score['test_min']:,.0f} to ${score['test_max']:,.0f}.</p>"
        f"<p class='ss-meta'>Trained on {score['train_rows']} quarters and tested on {score['test_rows']}.</p></div>"
    )
    right = (
        f"<div><p class='ss-lab cr'>Forecast for {forecast['label']}</p>"
        f"<p class='ss-big'><span>${forecast['value']:,.0f}</span></p><p><i>{html.escape(forecast['reason'])}</i></p></div>"
        if forecast
        else f"<div><p class='ss-lab cr'>No forecast possible</p><p>{html.escape(unavailable)}</p></div>"
    )
    return f"<div class='ss ss-result'>{left}{right}</div>"


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
{FONT_LINK}<style>{PALETTE}{BRIEF_STYLE}
body {{ margin: 0; background: #fff; }}
.ss-doc {{ max-width: 1000px; margin: 48px auto; padding: 0 24px; }}
.ss-doc > h2 {{ font: 500 30px/1.2 var(--serif); border-top: 4px solid var(--ink); padding-top: 14px; margin: 48px 0 12px; }}
.ss-doc > h3 {{ font: 600 18px/1.35 var(--serif); margin: 20px 0 6px; }}
.ss-doc > p, .ss-doc > ul {{ font: 400 16px/1.6 var(--sans); }}</style></head><body><div class="ss ss-doc">
<p class="ss-kicker">ConCReTE · City of Pittsburgh</p>
<h1 class="ss-title">Revenue Forecast Brief</h1>
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
</div></body></html>"""


@app.cell(hide_code=True)
def _():
    import marimo as mo

    mo.Html(
        f"{FONT_LINK}<style>{PALETTE}{BRIEF_STYLE}</style><div class='ss' style='margin-top:24px'>"
        "<div class='ss-hero'><div><p class='ss-kicker'>ConCReTE · City of Pittsburgh</p>"
        "<h1 class='ss-title'>Forecasting the city's revenue</h1></div>"
        "<div><p class='ss-byline'><b>The data</b><br>The city's revenue for every quarter from 2012 to 2024, "
        "copied out of its PDF reports in six ways:</p>"
        "<ul class='ss-sources'><li>By hand</li><li>Docling</li><li>ChatGPT</li><li>Claude chat</li>"
        "<li>Claude API</li><li>pypdf</li></ul></div></div></div>"
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
        mo.Html("<div class='ss'><p class='ss-lab cr'>Before you start</p>"
                "<h3 class='ss-task-title'>Connect your AI model</h3></div>"),
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
    mo.vstack([*_slots, _gate], gap=0.75)
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
            "<div class='ss ss-prompt'>Before you look at the data, decide what you think. These questions can help "
            f"you think it through, and nobody grades your answers to them:<ul>{_guiding}</ul>Then answer the three "
            "questions below. <strong>Once you lock your prediction you can't change it.</strong> You'll come back "
            "to it at the end.</div>"
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
            "<div class='ss ss-prompt'>This is the city's quarterly revenue, with one column for each extraction "
            "method. Below are five code cells. Run each one as it is, then change the code to look further; the "
            "hint above each cell suggests where. After that, build your forecast with the controls further down.</div>"
        ),
        pgh_revenue_data,
    ])
    return


@app.cell(hide_code=True)
def helper_panel(get_prediction, helper, mo):
    mo.stop(get_prediction() is None)
    mo.vstack([
        mo.Html("<div class='ss ss-head'><p class='ss-lab'>AI helper</p><h3 class='ss-task-title'>Ask the AI helper</h3>"
                "<p class='ss-desc'>The helper runs on your model on CRCD. It can see the data, the code in the five "
                "cells below and your current settings. Ask it about code, error messages or ideas; it won't make "
                "your choices for you.</p></div>"),
        helper,
    ])
    return


@app.cell(hide_code=True)
def observe_cells(cell_editors, cell_run_buttons, get_cells, get_outputs, get_prediction, get_submitted, mo):
    mo.stop(get_prediction() is None)
    _panels = []
    for _number, (_key, _cell) in enumerate(STARTER_CELLS.items(), 1):
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
            mo.Html(f"<div class='ss ss-head'><p class='ss-lab ss-swy'>Code cell {_number} of "
                    f"{len(STARTER_CELLS)}</p><h3 class='ss-task-title'>{html.escape(_cell['title'])}</h3>"
                    f"<div class='ss-hint'><b>Hint</b>{html.escape(_cell['question'])}</div></div>"),
            *([mo.md(f"```python\n{get_cells()[_key]}\n```")] if get_submitted() else [cell_editors[_key], cell_run_buttons[_key]]),
            *_shown,
        ]))
    mo.vstack(_panels, gap=0.5)
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
            _unavailable = "" if _forecast else forecast_unavailable_reason(
                pgh_revenue_data, column_choice.value, lag_amount.value
            )
            _result = mo.Html(forecast_result_html(_score, _forecast, _unavailable))
        except ValueError as _error:
            _result = mo.Html(f"<div class='ss ss-note'>{html.escape(str(_error))}</div>")
    mo.vstack([
        mo.Html("<div class='ss ss-head'><p class='ss-lab'>Your forecast</p>"
                "<h3 class='ss-task-title'>Build your forecast</h3></div>"),
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
            yield "Connect your model at the top of the page (**Test connection**) to use the helper."
            return
        system = helper_system_prompt(helper_context["csv"], helper_context["cells"], helper_context["settings"])
        history = helper_history(system, [{"role": m.role, "content": m.content} for m in messages])
        reply = ""
        try:
            async for reply in stream_model(helper_context["base"], history):
                yield reply
        except ModelUnavailable as error:
            yield f"{reply}\n\n*I can't reach your model right now. {error}*".strip()

    helper = mo.ui.chat(
        helper_reply,
        prompts=["What does a lag of 4 mean?", "How do I compare two columns row by row?", "What does this error mean?"],
    )
    return (helper,)


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
        mo.Html("<div class='ss ss-prompt'>Compare what you predicted with what you found in the data. Nobody "
                "grades the wording, only whether your reasoning holds up.</div>"),
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
        mo.Html("<div class='ss ss-prompt'>Part 1 lists only the settings worth a second look. Part 2 shows how "
                "the error number can mislead. Part 3 shows what each code cell could have revealed.</div>"),
        mo.Html(evaluation_html(pgh_revenue_data, final_settings, get_cells(), get_runs())),
    ])
    return (final_settings,)


@app.cell(hide_code=True)
def download_brief(final_settings, get_cells, get_explain, get_prediction, get_runs, helper_log, mo, pgh_revenue_data):
    _doc = report_html(pgh_revenue_data, get_prediction(), final_settings, get_explain(), helper_log, get_cells(), get_runs())
    mo.vstack([
        mo.Html("<div class='ss ss-head'><p class='ss-lab'>Before you leave</p>"
                "<h3 class='ss-task-title'>Download your brief</h3></div>"),
        mo.md("The file holds your prediction, your forecast, your reflection, your questions to the helper and "
              "this evaluation. It opens in any browser; use **Print → Save as PDF** for a PDF."),
        mo.download(data=_doc.encode("utf-8"), filename="revenue-forecast-brief.html", mimetype="text/html",
                    label="Download brief (.html)"),
        mo.md("*Nothing is saved online. Refreshing or closing this page erases your work, so download before you leave.*"),
    ])
    return


if __name__ == "__main__":
    app.run()
