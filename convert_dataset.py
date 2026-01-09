import pandas as pd

src = "spam.csv"
dst = "sample_data.csv"

print("Loading:", src)
df = pd.read_csv(src, encoding="latin1", low_memory=False)
print("Columns detected:", list(df.columns))

# Handle common Kaggle SMS Spam dataset (v1 = label, v2 = text)
if "v2" in df.columns and "v1" in df.columns:
    out = df[["v2", "v1"]].rename(columns={"v2": "text", "v1": "label"})
else:
    raise SystemExit("Unsupported format — tell me your column names and I will fix it.")

# Normalize label values
out["label"] = out["label"].astype(str).str.lower().map(
    lambda x: "benign" if x == "ham"
              else "phishing" if "phish" in x
              else "fraud" if "spam" in x
              else x
)

out.to_csv(dst, index=False)
print("Wrote:", dst)
print(out.head())
