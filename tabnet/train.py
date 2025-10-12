import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from pytorch_tabnet.pretraining import TabNetPretrainer
from pytorch_tabnet.tab_model import TabNetClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import os

np.random.seed(42)
torch.manual_seed(42)

TARGET = 'Churn'
PRETRAIN_EPOCHS = 15
FINETUNE_EPOCHS = 50
PATIENCE = 15
BASE_BATCH_SIZE = 2048
MODEL_FILENAME = "tabnet_churn_model_with_pretraining"

train_df = pd.read_csv('data-clean/train.csv')
val_df = pd.read_csv('data-clean/val.csv')
test_df = pd.read_csv('data-clean/test.csv')

categorical_features = [
    'has_children',
    'marital_status',
    'home_owner',
    'college_degree',
    'good_credit'
]
all_features = [col for col in train_df.columns if col != TARGET]

cat_idxs = [i for i, feature in enumerate(all_features) if feature in categorical_features]


combined_df_for_dims = pd.concat([train_df, val_df, test_df], ignore_index=True)
cat_dims = [int(combined_df_for_dims[col].max() + 1) for col in categorical_features]

print(f"ALL FEATURES: {all_features}")

X_train = train_df[all_features].values
y_train = train_df[TARGET].values
X_val = val_df[all_features].values
y_val = val_df[TARGET].values
X_test = test_df[all_features].values
y_test = test_df[TARGET].values

print("\nSTARTING SELF-SUPERVISED PRE-TRAINING")

tabnet_params = dict(
    n_d=16, 
    n_a=16, 
    n_steps=5, 
    gamma=1.5,
    lambda_sparse=1e-4, 
    mask_type='entmax'
)

pretrainer = TabNetPretrainer(
    **tabnet_params, 
    optimizer_fn=torch.optim.Adam,
    optimizer_params=dict(lr=2e-2),
    verbose=1
)

pretrainer.fit(
    X_train=X_train,
    eval_set=[X_val],
    max_epochs=PRETRAIN_EPOCHS,
    patience=PATIENCE,
    batch_size=BASE_BATCH_SIZE,
    pretraining_ratio=0.7 
)

print("\nSTARTING SUPERVISED FINE-TUNING")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_gpus = torch.cuda.device_count()
print(f"Using device: {device}")
if device == 'cuda':
    print(f"Found {n_gpus} GPUs.")

batch_size = BASE_BATCH_SIZE * n_gpus if n_gpus > 1 else BASE_BATCH_SIZE

model = TabNetClassifier(
    **tabnet_params,    
    cat_idxs=cat_idxs,  
    cat_dims=cat_dims,

    optimizer_fn=torch.optim.AdamW,
    optimizer_params=dict(lr=3e-4, weight_decay=1e-5),
    scheduler_params={"step_size":10, "gamma":0.9},
    scheduler_fn=torch.optim.lr_scheduler.StepLR,
    device_name=device,
    verbose=1
)

if n_gpus > 1:
    print(f"WRAPPING MODEL IN TORCH.NN.DATAPARALLEL USING BATCH SIZE: {batch_size}")
    model.network = nn.DataParallel(model.network)

model.fit(
    X_train=X_train, y_train=y_train,
    eval_set=[(X_val, y_val)],
    eval_name=['valid'],
    eval_metric=['auc', 'accuracy', 'logloss'],
    max_epochs=FINETUNE_EPOCHS,
    patience=PATIENCE,
    batch_size=batch_size,
    from_unsupervised=pretrainer 
)
history = model.history
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history['loss'], label='Training Loss')
plt.plot(history['valid_logloss'], label='Validation Loss')
plt.title('Training and Validation Loss'); plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history['valid_auc'], label='Validation AUC', color='orange')
plt.title('Validation AUC'); plt.xlabel('Epoch'); plt.ylabel('AUC'); plt.legend()
plt.tight_layout()
plt.savefig('learning_curves.png')
plt.close()

model.save_model(MODEL_FILENAME)

y_pred = model.predict(X_test)

report = classification_report(y_test, y_pred, target_names=['Not Churn', 'Churn'])
print(report)

accuracy = accuracy_score(y_test, y_pred)
print(f"TEST ACCURACY: {accuracy:.4f}")

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Not Churn', 'Churn'],
            yticklabels=['Not Churn', 'Churn'])
plt.title('Confusion Matrix')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.savefig('confusion_matrix.png')
plt.close()