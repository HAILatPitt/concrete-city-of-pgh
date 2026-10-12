import marimo

__generated_with = "0.18.0"
app = marimo.App(width="medium", app_title="ConCReTE Model Building")


@app.cell
async def _():
    import sys
    if "pyodide" in sys.modules:
        # browser (WASM) build: the groq client isn't bundled, install it at load time
        import micropip
        await micropip.install("groq")
    return


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import root_mean_squared_error
    return LinearRegression, mo, pd, root_mean_squared_error


@app.cell
def _(mo):
    get_intro_dialog_loaded, set_intro_dialog_loaded = mo.state(False)
    return get_intro_dialog_loaded, set_intro_dialog_loaded


@app.cell(hide_code=True)
def _(mo):
    input_key = mo.ui.text(label="Groq AI API key", kind="password")
    mo.vstack([
        mo.md("""
        ### Before you start: get your free Groq API key
        This scenario includes an AI tutor that asks you follow-up questions. To use it, you need your own free key from Groq, an AI service. It takes about 2 minutes.

        1. Go to **[console.groq.com/keys](https://console.groq.com/keys)** and sign up. A Google or GitHub account works.
        2. Click **Create API Key**, give it any name (e.g. "ConCReTE"), and click **Submit**.
        3. **Copy the key right away.** Groq shows it only once. It starts with `gsk_`.
        4. Paste it into the **Groq AI API key** box below and press **Enter**.

        **Keep your key private.** Don't share it or post it anywhere. Your key stays in your browser and is only sent to Groq. What you type in the chat is sent to Groq to generate replies, so don't include personal information.

        *No key? You can still do the whole data science task. Only the AI tutor won't respond.*
        """).callout("info"),
        input_key,
    ])
    return (input_key,)


@app.cell(hide_code=True)
def _(input_key, mo, set_intro_dialog_loaded):
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
    revenue_columns = [col for col in pgh_revenue_data.columns if col.lower().startswith('revenue by')]


    # make marimo buttons for each of the values in the revenue_columns list
    revenue_selector = mo.ui.radio(
        options=revenue_columns,
        value=revenue_columns[0],  # sets default value to first column
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
