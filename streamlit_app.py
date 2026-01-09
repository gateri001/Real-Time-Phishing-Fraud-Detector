import streamlit as st
from inference import predict_message

st.set_page_config(page_title="PhishDetect Demo", layout="centered")

st.title("Real-Time Phishing & Fraud Detector — Demo")
st.write(
    "Paste a message (SMS / Email / Text) and get a threat score, label, "
    "evidence, and recommended action."
)

# Default text
default_text = (
    "Your bank account has been locked. "
    "Click http://bit.ly/verify-now to verify your details."
)

text = st.text_area("Message", height=200, value=default_text)

if st.button("Analyze"):
    res = predict_message(text)

    st.metric("Threat Score", f"{res['score']:.3f}")
    st.markdown(f"### Label: **{res['label']}**")
    st.markdown(f"### Recommended Action: **{res['action']}**")

    st.subheader("Evidence")
    st.json(res["evidence"])

st.markdown("---")
st.subheader("Quick examples to try:")
st.write("- `URGENT: Verify your account at http://malicious.example.com`")
st.write("- `Happy birthday! Let's meet for lunch tomorrow.`")
