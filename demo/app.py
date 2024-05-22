import pandas as pd
import streamlit as st
from spacy import displacy

import edsnlp
from edsnlp.utils.filter import filter_spans

DEFAULT_TEXT = "Brice Denice a déménagé à Hossegor en 2006."


@st.cache_resource()
def load_model():
    nlp = edsnlp.load("AP-HP/eds-pseudo-public")
    return nlp


st.set_page_config(
    page_title="EDS-Pseudo Demo",
    page_icon="📄",
)

st.title("EDS-Pseudo")

st.warning(
    "You should **not** put sensitive data in the example, as this application "
    "**is not secure**."
)

st.sidebar.header("About")
st.sidebar.markdown(
    "EDS-Pseudo is a contributive effort maintained by AP-HP's Data Science team. "
    "Have a look at the "
    "[documentation](https://aphp.github.io/eds-pseudo/) for "
    "more information on the available components."
)

model_load_state = st.info("Loading model...")

nlp = load_model()

model_load_state.empty()

st.header("Enter a text to analyse:")
text = st.text_area(
    "Modify the following text and see the pipeline react :",
    DEFAULT_TEXT,
    height=50,
)

doc = nlp(text)
doc.ents = filter_spans(
    (*doc.ents, *doc.spans.get("dates", []), *doc.spans.get("measurements", []))
)

st.header("Visualisation")

st.markdown(
    "The pipeline extracts simple entities using a dictionnary of RegEx (see the "
    "[Export the pipeline section](#export-the-pipeline) for more information)."
)

category20 = [
    "#1f77b4",
    "#aec7e8",
    "#ff7f0e",
    "#ffbb78",
    "#2ca02c",
    "#98df8a",
    "#d62728",
    "#ff9896",
    "#9467bd",
    "#c5b0d5",
    "#8c564b",
    "#c49c94",
    "#e377c2",
    "#f7b6d2",
    "#7f7f7f",
    "#c7c7c7",
    "#bcbd22",
    "#dbdb8d",
    "#17becf",
    "#9edae5",
]

labels = [
    "date",
    "covid",
    "drug",
    "cim10",
    "emergency_priority",
    "sofa",
    "charlson",
    "size",
    "weight",
    "adicap",
]

colors = {label: cat for label, cat in zip(labels, category20)}
colors["custom"] = "linear-gradient(90deg, #aa9cfc, #fc9ce7)"
options = {
    "colors": colors,
}

html = displacy.render(doc, style="ent")
html = html.replace("line-height: 2.5;", "line-height: 2.25;")
html = (
    '<div style="padding: 10px; border: solid 2px; border-radius: 10px; '
    f'border-color: #afc6e0;">{html}</div>'
)
st.write(html, unsafe_allow_html=True)

data = []
for ent in doc.ents:
    d = dict(
        start=ent.start_char,
        end=ent.end_char,
        text=ent.text,
        label=ent.label_,
        normalized_value=str(ent._.value or ""),
    )

    data.append(d)

st.header("Entity qualification")

if data:
    df = pd.DataFrame.from_records(data)
    df.normalized_value = df.normalized_value.replace({"None": ""})
    st.dataframe(df)

else:
    st.markdown("You pipeline did not match any entity...")
