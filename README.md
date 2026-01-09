# Real-Time Phishing & Fraud Detector (Regenerated Hackathon Prototype)

This regenerated repository contains a compact end-to-end prototype for a Real-Time Phishing & Fraud Detector:
- Data prep script
- Training script (Hugging Face Trainer)
- Inference wrapper
- Streamlit demo app
- FastAPI inference app + Dockerfile
- requirements.txt and sample data

Quick start:
1. Create Python 3.9+ virtualenv and install deps:
   ```
   pip install -r requirements.txt
   ```
2. Prepare synthetic sample data:
   ```
   python data_prep.py --out sample_data.csv --n 500
   ```
3. Train (toy):
   ```
   python train.py --data sample_data.csv --output_dir ./model_out --epochs 1
   ```
4. Run demo:
   ```
   streamlit run streamlit_app.py
   ```
5. Run API:
   ```
   uvicorn app_fastapi:app --reload --port 8000
   ```
