import pandas as pd
import joblib
import os

from retrain.train import train_model
from retrain.evaluate import compare_models
from retrain.promote import promote_model


df_old = pd.read_csv("data/train_v1.csv")
df_new = pd.read_csv("data/train_v2.csv")

# Train old model 
old_model, old_acc = train_model(df_old)

# Save initial champion if not exists
if not os.path.exists("models/champion.pkl"):
    promote_model(old_model)

# Train challenger
challenger_model, new_acc = train_model(df_new)

# Compare
if compare_models(old_acc, new_acc):
    promote_model(challenger_model)
else:
    print("Keeping existing champion")
