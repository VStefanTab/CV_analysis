import streamlit as st
from extractor import extract_text
from model import load_model, predict

st.set_page_config(
    page_title='CV-JD Fit Predictor',
    page_icon='📄',
    layout='wide'
)

@st.cache_resource
def get_model():
    with st.spinner('Loading model...'):
        return load_model(assets_dir='../assets')

model, tokenizer, config, device = get_model()


st.title('CV - Job Description Fit Predictor')
st.caption(
    'Upload a CV and a Job Description (PDF or DOCX)'
    'The model predicts whether the candidate is a fit for the role'
)
st.divider()

col_cv, col_jd = st.columns(2)

with col_cv:
    st.subheader('Candidate CV')
    cv_file = st.file_uploader(
        'Upload CV', type=['pdf', 'docx'], key='cv',
        label_visibility='collapsed'
    )
    cv_text = ''
    if cv_file:
        cv_text, err = extract_text(cv_file)
        if err:
            st.error(err)
        else:
            st.success(f'Extracted {len(cv_text):,} characters')
            with st.expander('Preview extracted text'):
                st.text(cv_text[:1500] + ('...' if len(cv_text) > 1500 else ''))

with col_jd:
    st.subheader('Job Description')
    jd_file = st.file_uploader(
        'Upload JD', type=['pdf', 'docx'], key='jd',
        label_visibility='collapsed'
    )
    jd_text = ''
    if jd_file:
        jd_text, err = extract_text(jd_file)
        if err:
            st.error(err)
        else:
            st.success(f'Extracted {len(jd_text):,} characters')
            with st.expander('Preview extracted text'):
                st.text(jd_text[:1500] + ('...' if len(jd_text) > 1500 else ''))

st.divider()

predict_btn = st.button(
    'Predict fit',
    type='primary',
    disabled=(not cv_text or not jd_text)
)

if predict_btn:
    with st.spinner('Running model inference...'):
        result = predict(cv_text, jd_text, model, tokenizer, config, device)

    st.divider()

    if result['label'] == 1:
        st.success(f"## Verdict: {result['verdict']}")
    else:
        st.error(f"## Verdict: {result['verdict']}")

    m1, m2, m3 = st.columns(3)

    with m1:
        st.metric(
            label='Fit probability',
            value=f"{result['probability']*100:.1f}%",
            help='P(accepted) from the sigmoid output. Threshold = 0.50'
        )
    with m2:
        st.metric(
            label='Model confidence',
            value=f"{result['confidence']*100:.1f}%",
            help='How far the prediction is from the 0.5 decision boundary. 100% = maximally certain.'
        )
    with m3:
        st.metric(
            label='Semantic similarity',
            value=f"{result['cosine_sim']*100:.1f}%",
            help='Cosine similarity between the BERT embeddings of the CV and JD.'
        )

    st.write('')
    st.write('**Fit probability**')
    st.progress(result['probability'])

    st.write('**Semantic similarity**')
    st.progress(max(0.0, result['cosine_sim']))

    st.divider()
    with st.expander('How to interpret these results'):
        st.markdown("""
        | Metric | What it means |
        |---|---|
        | **Fit probability** | The model's raw confidence that this candidate fits the role. Above 50% = predicted fit. |
        | **Model confidence** | How decisive the prediction is. Low confidence = the pair is near the decision boundary. |
        | **Semantic similarity** | How much the CV and JD share in meaning according to the BERT encoder. High similarity doesn't guarantee a fit — it means the texts talk about similar things. |

        **Important:** This model was trained on tech job postings and tech CVs.
        Predictions on other domains may be less reliable.
        """)

with st.sidebar:
    st.header('Model info')
    st.markdown(f"""
    - **Architecture:** Cross-Encoder
    - **Base model:** `cross-encoder/ms-marco-MiniLM-L-12-v2`
    - **Max tokens:** {config['max_len']} per text
    - **Threshold:** {config['threshold']}
    - **Device:** `{device}`
    """)
    st.divider()
    st.header('Test metrics')
    st.markdown("""
    | Metric | Score |
    |---|---|
    | Accuracy | 0.991 |
    | F1 Score | 0.991 |
    | ROC-AUC  | 0.997 |

    *Evaluated on held-out test set*
    """)
    st.divider()
    st.caption(
        'Deep Learning course project — '
        'Cross-Encoder fine-tuned on tech CV-JD pairs'
    )