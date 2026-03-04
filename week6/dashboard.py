import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from pathlib import Path
import sys
from sklearn.metrics import f1_score, recall_score, roc_auc_score

# ----------------------------------
# Project Setup
# ----------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from models.directed_gcn_v1 import DirectedGCN

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_PATH = PROJECT_ROOT / "data" / "processed"
RESULTS_PATH = PROJECT_ROOT / "results"

st.set_page_config(page_title="Directed GCN Fraud Dashboard", layout="wide")

st.title("🚀 Directed GCN Fraud Detection Dashboard")

# ----------------------------------
# Load Graph
# ----------------------------------

@st.cache_data
def load_graph(name):
    path = DATA_PATH / f"{name}_final.pt"
    if not path.exists():
        st.error(f"{path} not found.")
        st.stop()
    data = torch.load(path, weights_only=False)
    return data.to(DEVICE)

# ----------------------------------
# Structural Feature Enrichment
# ----------------------------------

def compute_structural_features(data):
    num_nodes = data.num_nodes
    src, dst = data.edge_index

    in_deg = torch.zeros(num_nodes, device=DEVICE)
    in_deg.scatter_add_(0, dst, torch.ones_like(dst, dtype=torch.float))

    out_deg = torch.zeros(num_nodes, device=DEVICE)
    out_deg.scatter_add_(0, src, torch.ones_like(src, dtype=torch.float))

    total_deg = in_deg + out_deg
    total_deg[total_deg == 0] = 1

    degree_imbalance = torch.abs(in_deg - out_deg) / total_deg
    flow_asymmetry = (out_deg - in_deg) / total_deg

    fraud_labels = data.y.float()

    first_hop = torch.zeros(num_nodes, device=DEVICE)
    first_hop.index_add_(0, dst, fraud_labels[src])

    deg_in = torch.zeros(num_nodes, device=DEVICE)
    deg_in.scatter_add_(0, dst, torch.ones_like(dst, dtype=torch.float))
    deg_in[deg_in == 0] = 1

    first_hop = first_hop / deg_in

    second_hop = torch.zeros(num_nodes, device=DEVICE)
    second_hop.index_add_(0, dst, first_hop[src])
    second_hop = second_hop / deg_in

    structural_feats = torch.stack([
        in_deg,
        out_deg,
        degree_imbalance,
        flow_asymmetry,
        second_hop
    ], dim=1)

    return structural_feats

# ----------------------------------
# Load Hyperparameter Tuning Results
# ----------------------------------

@st.cache_data
def load_tuning_results():
    path = RESULTS_PATH / "hyperparameter_tuning_results.csv"
    if not path.exists():
        st.error("hyperparameter_tuning_results.csv not found.")
        st.stop()
    return pd.read_csv(path)

# ----------------------------------
# Train Tuned Model for Selected Dataset
# ----------------------------------

@st.cache_resource
def train_best_model(dataset_name):

    tuning_df = load_tuning_results()

    best_row = (
        tuning_df[tuning_df["Dataset"] == dataset_name]
        .sort_values("F1", ascending=False)
        .iloc[0]
    )

    hidden_dim = int(best_row["Hidden_Dim"])
    lr = float(best_row["LR"])
    weight_decay = float(best_row["Weight_Decay"])

    data = load_graph(dataset_name)

    structural_feats = compute_structural_features(data)
    data_enriched = data.clone()
    data_enriched.x = torch.cat([data.x, structural_feats], dim=1)

    model = DirectedGCN(
        input_dim=data_enriched.x.shape[1],
        hidden_dim=hidden_dim
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )

    criterion = nn.CrossEntropyLoss()

    model.train()
    for _ in range(25):  # fast training
        optimizer.zero_grad()
        logits = model(data_enriched)
        loss = criterion(
            logits[data_enriched.train_mask],
            data_enriched.y[data_enriched.train_mask]
        )
        loss.backward()
        optimizer.step()

    model.eval()

    return model, data_enriched, hidden_dim, lr, weight_decay

# ----------------------------------
# Dataset Selection
# ----------------------------------

st.sidebar.header("Dataset Selection")

dataset_name = st.sidebar.selectbox(
    "Choose Dataset",
    ["elliptic", "paysim"]
)

model, data_enriched, hidden_dim, lr, weight_decay = train_best_model(dataset_name)

st.sidebar.header("Model Information")
st.sidebar.write(f"Dataset: **{dataset_name.upper()}**")
st.sidebar.write(f"Hidden Dimension: **{hidden_dim}**")
st.sidebar.write(f"Learning Rate: **{lr}**")
st.sidebar.write(f"Weight Decay: **{weight_decay}**")

# ----------------------------------
# Run Prediction
# ----------------------------------

with torch.no_grad():
    logits = model(data_enriched)
    probs = F.softmax(logits, dim=1)[:, 1]
    preds = logits.argmax(dim=1)

test_mask = data_enriched.test_mask
y_true = data_enriched.y

f1 = f1_score(y_true[test_mask].cpu(), preds[test_mask].cpu(), zero_division=0)
recall = recall_score(y_true[test_mask].cpu(), preds[test_mask].cpu(), zero_division=0)
auc = roc_auc_score(y_true[test_mask].cpu(), probs[test_mask].cpu())

col1, col2, col3 = st.columns(3)
col1.metric("F1 Score", f"{f1:.4f}")
col2.metric("Recall", f"{recall:.4f}")
col3.metric("AUC", f"{auc:.4f}")

st.divider()

# ----------------------------------
# Node-Level Explorer
# ----------------------------------

st.subheader("🔍 Node-Level Fraud Probability Explorer")

node_id = st.slider("Select Node ID", 0, data_enriched.num_nodes - 1, 0)

st.write(f"**True Label:** {int(y_true[node_id].item())}")
st.write(f"**Predicted Label:** {int(preds[node_id].item())}")
st.write(f"**Fraud Probability:** {probs[node_id].item():.4f}")

# ----------------------------------
# Distribution Plot
# ----------------------------------

st.subheader("📊 Fraud Probability Distribution")

prob_df = pd.DataFrame({
    "Fraud_Probability": probs.cpu().numpy(),
    "True_Label": y_true.cpu().numpy()
})

st.bar_chart(
    prob_df.groupby("True_Label")["Fraud_Probability"].mean()
)

# ----------------------------------
# Load Result Tables
# ----------------------------------

st.subheader("📈 Experiment Results")

if (RESULTS_PATH / "final_model_comparison.csv").exists():
    final_df = pd.read_csv(RESULTS_PATH / "final_model_comparison.csv")
    st.write("### Final Model Comparison")
    st.dataframe(final_df)

if (RESULTS_PATH / "robustness_results.csv").exists():
    robustness_df = pd.read_csv(RESULTS_PATH / "robustness_results.csv")
    st.write("### Robustness Study Results")
    st.dataframe(robustness_df)

if (RESULTS_PATH / "subsampling_results.csv").exists():
    subsample_df = pd.read_csv(RESULTS_PATH / "subsampling_results.csv")
    st.write("### Subsampling Study Results")
    st.dataframe(subsample_df)