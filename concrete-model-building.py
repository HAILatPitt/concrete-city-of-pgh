import marimo

__generated_with = "0.18.0"
app = marimo.App(width="medium", app_title="ConCReTE Model Building")

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


@app.cell
def _():
    import marimo as mo
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
        "ok": mo.md(f"**Connected:** `{_detail}`. The AI helper is ready.").callout("success"),
        "error": mo.md(f"**Not connected.** {_detail}").callout("warn"),
    }.get(_state, "")
    mo.vstack([
        mo.md("## Before you start: connect your AI model"),
        mo.md(CRCD_STEPS),
        model_url,
        test_button,
        _note,
    ])
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
