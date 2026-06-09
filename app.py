            test_data[col] = np.nan

    test_data = test_data[features]
    probabilities = model.predict_proba(test_data)[:, 1]

    return pd.DataFrame(
        {
            "Id": np.arange(1, len(test_data) + 1),
            "Probability": probabilities,
        }
    )


inject_dark_theme()
render_hero()

if "use_sample_data" not in st.session_state:
    st.session_state.use_sample_data = False

with st.sidebar:
    st.markdown("## EVOASTRA")
    st.caption("Synthetic credit-risk AI lab")
    st.divider()

    if st.button("Load Sample Data", use_container_width=True):
        st.session_state.use_sample_data = True

    uploaded_train = st.file_uploader("Upload training CSV", type=["csv"])
    uploaded_test = st.file_uploader("Optional test CSV", type=["csv"])

    st.divider()
    st.markdown("### Training Settings")
    sample_limit = st.slider("Training sample size", 200, 2000, 500, 100)
    ctgan_epochs = st.slider("CTGAN epochs", 1, 10, 1, 1)
    synthetic_ratio = st.slider("Target default share", 0.10, 0.30, 0.15, 0.05)

    train_button = st.button("Train CTGAN + Model", type="primary", use_container_width=True)

    st.divider()
    st.caption("Tip: Streamlit Cloud works best with small CTGAN settings.")

train_source = uploaded_train

if train_source is None and st.session_state.use_sample_data:
    if DEFAULT_TRAIN_PATH.exists():
        train_source = DEFAULT_TRAIN_PATH
    else:
        st.warning("Sample data was not found. Upload `cs-training.csv` to continue.")

if train_source is None and DEFAULT_TRAIN_PATH.exists():
    st.info("Click **Load Sample Data** in the sidebar, or upload your own training CSV.")
    st.stop()

if train_source is None:
    st.info("Upload `cs-training.csv` in the sidebar to start.")
    st.stop()

try:
    df = clean_data(load_credit_data(train_source))
except Exception as error:
    st.error("Could not load the training data.")
    st.exception(error)
    st.stop()

st.markdown("## Portfolio Overview")
cols = st.columns(4)
cols[0].metric("Credit Records", f"{len(df):,}")
cols[1].metric("Model Features", len([c for c in df.columns if c != TARGET]))
cols[2].metric("Observed Default Rate", f"{df[TARGET].mean():.2%}")
cols[3].metric("Missing Values", f"{int(df.isna().sum().sum()):,}")

left, right = st.columns([1.1, 1])
with left:
    st.markdown("### Dataset Preview")
    st.dataframe(df.head(20), use_container_width=True, height=360)

with right:
    st.markdown("### Target Distribution")
    distribution = df[TARGET].value_counts().sort_index()
    chart_df = pd.DataFrame(
        {
            "Risk Class": ["No Serious Delinquency", "Serious Delinquency"],
            "Records": [int(distribution.get(0, 0)), int(distribution.get(1, 0))],
        }
    )
    st.bar_chart(chart_df, x="Risk Class", y="Records", color="#25e0c4")

if not train_button:
    st.markdown(
        """
        <div class="panel">
            <h3>Ready to Train</h3>
            <p style="color:#b9adc9;">
                Use the sidebar controls to tune sample size, CTGAN epochs, and target default share.
                Click <b>Train CTGAN + Model</b> when you are ready.
            </p>
            <p class="status-ok">Cloud-safe defaults are already selected.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

with st.spinner("Training CTGAN and credit risk model. Please wait..."):
    try:
        (
            model,
            features,
            synthetic_df,
            metrics,
            matrix,
            report,
            synthetic_status,
        ) = train_credit_model(df, sample_limit, ctgan_epochs, synthetic_ratio)
    except Exception as error:
        st.error("Training failed. Full error:")
        st.exception(error)
        st.stop()

st.success("Training complete.")
st.info(synthetic_status)

st.markdown("## Executive Metrics")
metric_cols = st.columns(6)
metric_cols[0].metric("ROC AUC", f"{metrics['ROC AUC']:.3f}")
metric_cols[1].metric("Avg Precision", f"{metrics['Average Precision']:.3f}")
metric_cols[2].metric("Accuracy", f"{metrics['Accuracy']:.3f}")
metric_cols[3].metric("Precision", f"{metrics['Precision']:.3f}")
metric_cols[4].metric("Recall", f"{metrics['Recall']:.3f}")
metric_cols[5].metric("F1", f"{metrics['F1']:.3f}")

st.metric("Synthetic Rows Generated", f"{metrics['Synthetic Rows']:,}")

matrix_col, synthetic_col = st.columns([.9, 1.1])
with matrix_col:
    st.markdown("### Confusion Matrix")
    st.dataframe(
        pd.DataFrame(
            matrix,
            index=["Actual 0", "Actual 1"],
            columns=["Predicted 0", "Predicted 1"],
        ),
        use_container_width=True,
    )

with synthetic_col:
    st.markdown("### Synthetic Default Examples")
    if synthetic_df.empty:
        st.info("No synthetic rows were generated.")
    else:
        st.dataframe(synthetic_df.head(12), use_container_width=True, height=300)
        st.download_button(
            "Download Synthetic Data",
            synthetic_df.to_csv(index=False),
            "synthetic_ctgan_defaults.csv",
            "text/csv",
            use_container_width=True,
        )

with st.expander("Classification Report"):
    st.code(report)

test_source = uploaded_test
if test_source is None and st.session_state.use_sample_data and DEFAULT_TEST_PATH.exists():
    test_source = DEFAULT_TEST_PATH

if test_source is not None:
    try:
        test_df = pd.read_csv(test_source)
        submission = make_submission(model, features, test_df)

        st.markdown("## Prediction Submission")
        st.dataframe(submission.head(25), use_container_width=True)

        st.download_button(
            "Download Predictions",
            submission.to_csv(index=False),
            "ctgan_credit_risk_submission.csv",
            "text/csv",
            use_container_width=True,
        )
    except Exception as error:
        st.error("Could not create test predictions.")
        st.exception(error)
    
