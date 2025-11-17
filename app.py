import gradio as gr
import pickle
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Load objects
def load(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

scaler = load('models/scaler.pkl')
model = load('models/model.pkl')
feature_names = load('models/features.pkl')

# For visualization, let's reload your dataset (optional)
import pandas as pd
df = pd.read_csv('data/synthetic_houses.csv')

# Visualizations (plots saved to /plots)
os.makedirs('plots', exist_ok=True)
def plot_feature_hist(feature):
    plt.figure(figsize=(6,4))
    sns.histplot(df[feature], kde=True, bins=20, color='blue', edgecolor='black')
    plt.title(f'Distribution of {feature}')
    plt.tight_layout()
    fname = f'plots/{feature}_hist.png'
    plt.savefig(fname)
    plt.close()
    return fname

def plot_corr_heatmap():
    plt.figure(figsize=(7,5))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.tight_layout()
    fname = 'plots/corr_heatmap.png'
    plt.savefig(fname)
    plt.close()
    return fname

def plot_pred_vs_actual():
    # Use part of the real test split
    X = df[feature_names]
    y_true = df['price']
    y_pred = model.predict(scaler.transform(X))
    plt.figure(figsize=(6,6))
    plt.scatter(y_true, y_pred, s=15, alpha=0.6)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'k--', lw=2)
    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')
    plt.title('Predicted vs Actual')
    plt.tight_layout()
    fname = 'plots/pred_vs_actual.png'
    plt.savefig(fname)
    plt.close()
    return fname

# Main inference function
def predict(area, bedrooms, bathrooms, stories, parking, mainroad, guestroom):
    vals = np.array([[area, bedrooms, bathrooms, stories, parking, mainroad, guestroom]])
    X_scaled = scaler.transform(vals)
    preds = np.array([tree.predict(X_scaled) for tree in model.estimators_])
    mean_pred = preds.mean()
    std_pred = preds.std()
    lower = mean_pred - 2*std_pred  # ~95% conf. interval
    upper = mean_pred + 2*std_pred
    return (
        f"Predicted Price: ${mean_pred:,.0f} (95% CI: ${lower:,.0f} - ${upper:,.0f})",
        plot_feature_hist('area'),
        plot_corr_heatmap(),
        plot_pred_vs_actual()
    )

# Build Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("# House Price Predictor 📊\nMove the sliders and click Predict!")
    with gr.Row():
        with gr.Column(scale=2):
            area = gr.Slider(500, 3500, step=10, value=1500, label='Area (sq. ft)')
            bedrooms = gr.Slider(1, 4, step=1, value=2, label='Bedrooms')
            bathrooms = gr.Slider(1, 3, step=1, value=2, label='Bathrooms')
            stories = gr.Slider(1, 3, step=1, value=2, label='Stories')
            parking = gr.Slider(0, 3, step=1, value=1, label='Parking Spaces')
            mainroad = gr.Slider(0, 1, step=1, value=1, label='Main Road Access (1=Yes, 0=No)')
            guestroom = gr.Slider(0, 1, step=1, value=0, label='Guest Room (1=Yes, 0=No)')
        with gr.Column(scale=1):
            output = gr.Textbox(label="Predicted Price")
            out1 = gr.Image(label="Area Histogram")
            out2 = gr.Image(label="Correlation Heatmap")
            out3 = gr.Image(label="Predicted vs Actual")
    gr.Button("Predict").click(
        predict, [area, bedrooms, bathrooms, stories, parking, mainroad, guestroom], [output, out1, out2, out3]
    )

demo.launch()
