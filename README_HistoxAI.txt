# HistoxAI — redesigned black & red website

## What's changed
- Three-page navigation: Overview, Our Approach, Prediction.
- Dark charcoal/black interface with red accents, subtle orbit/particle visual motif, and spacious editorial layout inspired by the screenshot supplied.
- Overview explains colorectal cancer and the project.
- Our Approach explains the dataset, InceptionV3 workflow, XAI methods, and previously reported metrics.
- Prediction retains InceptionV3 prediction, class probabilities, Grad-CAM, high-activation region highlights, Integrated Gradients, LIME, and JSON export.

## Run locally (Windows PowerShell)
Open PowerShell in this extracted folder, then run:

```powershell
# Activate your existing virtual environment if it is at C:\Users\N\venv:
& "C:\Users\N\venv\Scripts\Activate.ps1"

pip install -r requirements.txt
streamlit run app.py
```

If you have not created a virtual environment:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

## Model checkpoint
Place your trained checkpoint at:
`colorectal_output/colorectal_inceptionv3_final.pth`

The `colorectal_output` folder is included as a placeholder. Copy your downloaded `.pth` file into it. Alternatively, on the Prediction page, upload the `.pth` checkpoint in the sidebar.

## Important
- Region outlines are attribution regions, not tumor segmentation masks.
- Grad-CAM, Integrated Gradients and LIME explain model influence; they do not establish biological meaning or a clinical diagnosis.
- Softmax scores are not calibrated clinical probabilities.
- The reported test split is tile-level, not patient/slide-level. This prototype is not validated for clinical use.
