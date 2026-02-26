<<<<<<< HEAD
import tensorflow as tf  
import tensorflow_datasets as tfds  
import numpy as np  
import matplotlib.pyplot as plt  
from tensorflow.keras.applications.efficientnet import preprocess_input  
from tensorflow.keras.models import load_model  
from sklearn.metrics import confusion_matrix, classification_report, roc_curve  
import os  
from medmnist import PneumoniaMNIST
  
#Make directory for save the result file  
os.makedirs("outputs", exist_ok=True)  
  
#Load the data and preprare training, valuation and test dataset for the evaluation  
best_model = load_model("models/pneumonia_efficientNet_finetuned.keras")  
  
def preprocess_for_model(x):  
    x = tf.cast(x, tf.float32)  
    x = tf.ensure_shape(x,[28,28])  
    x = tf.expand_dims(x,-1) #(28,28,1)  
    x = tf.image.resize(x, (224, 224))  
    x = tf.image.grayscale_to_rgb(x)  
    x = preprocess_input(x)  
    x = tf.ensure_shape(x, (224, 224, 3))  
    return x  

#Load the data and preprare training, valuation and test dataset for the evaluation  
test_dataset = PneumoniaMNIST(split="test", download=True)

#Numpy array  
x_np = test_dataset.imgs #(N,28,28) or (N,28,28,1)  
y_np = test_dataset.labels #(N,1)  

#Convert numpy array to tensorflow dataset format  
#→ Able to make data pipeline that take out each samples  
x_np = x_np.squeeze()  
y_np = y_np.squeeze()  
ds_test = tf.data.Dataset.from_tensor_slices((x_np, y_np))  
  
ds_triplet = (ds_test  
              .map(lambda x, y: (x, preprocess_for_model(x), tf.cast(y, tf.int32)),  
                   num_parallel_calls=tf.data.AUTOTUNE)  
              .batch(32)  
              .prefetch(tf.data.AUTOTUNE))  
  
ds_eval = (ds_triplet  
           .map(lambda x_raw, x_proc, y: (x_proc, tf.cast(tf.expand_dims(y, -1), tf.float32)),  
                num_parallel_calls=tf.data.AUTOTUNE)  
           .prefetch(tf.data.AUTOTUNE))  
  
raw_imgs, y_true, p_prob = [], [], []  
  
for x_raw, x_proc, y in ds_triplet:  
    p = best_model.predict(x_proc, verbose=0).reshape(-1)  # sigmoid (N,)  
    raw_imgs.append(x_raw.numpy())                         # (B,28,28,1)  
    y_true.append(y.numpy().reshape(-1))                   # (B,)  
    p_prob.append(p)  
  
raw_imgs = np.concatenate(raw_imgs, axis=0)  
y_true = np.concatenate(y_true, axis=0).astype(int)  
p_prob = np.concatenate(p_prob, axis=0)  
y_pred = (p_prob >= 0.5).astype(int)  
  
# Make index  
mis_idx = np.where(y_pred != y_true)[0]  
cor_idx = np.where(y_pred == y_true)[0]  
fp_idx = np.where((y_true ==0) & (y_pred==1))[0]  
fn_idx = np.where((y_true==1) & (y_pred ==0))[0]  
  
print("Correct:", len(cor_idx), "Misclassified:", len(mis_idx))  
print("Misclassified normal lung as pneumonia: ", len(fp_idx))  
print("Misclassified pneumonia as normal lung: ", len(fn_idx))  
  
#Misclassification with pic  
n_show = 10  
show_mis = mis_idx[:n_show]  
show_cor = cor_idx[:n_show]  
  
fig, axes = plt.subplots(2, n_show, figsize=(3*n_show, 10))  
  
for i, idx in enumerate(show_cor):  
    axes[0, i].imshow(raw_imgs[idx].squeeze(), cmap="gray")  
    axes[0, i].set_title(f"Correct\nT:{y_true[idx]} P:{y_pred[idx]} Prob:{p_prob[idx]:.2f}")  
    axes[0, i].axis("off")  
  
for i, idx in enumerate(show_mis):  
    axes[1, i].imshow(raw_imgs[idx].squeeze(), cmap="gray")  
    axes[1, i].set_title(f"Misclassified\nT:{y_true[idx]} P:{y_pred[idx]} Prob:{p_prob[idx]:.2f}")  
    axes[1, i].axis("off")  
plt.tight_layout()  
plt.savefig("outputs/misclassified_examples.png", dpi=200, bbox_inches="tight")  
print("Saved:outputs/misclassified_examples.png")  
  
#Loss function, Accuracy, AUC(Area under curve)  
test_loss, test_acc, test_auc = best_model.evaluate(ds_eval, verbose=0)  
print(f"Test loss: {test_loss:.4f}")  
print(f"Test accuracy: {test_acc:.4f}")  
print(f"Test AUC: {test_auc:.4f}")  
  
with open("outputs/loss_acc_auc.txt", "w") as f:  
    f.write(f"Test Loss: {test_loss:.4f}\n")  
    f.write(f"Test Accuracy: {test_acc:.4f}\n")  
    f.write(f"Test AUC: {test_auc:.4f}\n")  
  
#Confusion Matrix  
p_test = best_model.predict(ds_eval, verbose=0).reshape(-1)  
  
#Convert 0/1 label  
#y_test = np.array([y.numpy() for _, y in ds_test]).reshape(-1).astype(int)  
y_test = y_true  
  
y_pred = (p_test >= 0.5).astype(int)  
print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred, digits=4))  
  
cm = confusion_matrix(y_test, y_pred)  
plt.figure()  
plt.imshow(cm)  
plt.title("Confusion Matrix")  
plt.colorbar()  
plt.xlabel("Predicted")  
plt.ylabel("True")  
for i in range(2):  
    for j in range(2):  
        plt.text(j, i, cm[i, j], ha="center", va="center")  
plt.tight_layout()  
plt.savefig("outputs/confusion_matrix.png", dpi=200, bbox_inches="tight")  
print("Saved:outputs/confusion_matrix.png")  
  
#ROC curve  
fpr, tpr, thresholds = roc_curve(y_test, p_test)  
auc = test_auc  
  
plt.figure()  
plt.plot(fpr, tpr, label=f"AUC={auc:.4f}")  
plt.plot([0,1], [0,1], linestyle="--", label="Chance")  
plt.xlabel("False Positive Rate")  
plt.ylabel("True Positive Rate")  
plt.legend()  
plt.tight_layout()  
plt.savefig("outputs/roc_curve.png", dpi=200, bbox_inches="tight")  
=======
import tensorflow as tf  
import tensorflow_datasets as tfds  
import numpy as np  
import matplotlib.pyplot as plt  
from tensorflow.keras.applications.efficientnet import preprocess_input  
from tensorflow.keras.models import load_model  
from sklearn.metrics import confusion_matrix, classification_report, roc_curve  
import os  
from medmnist import PneumoniaMNIST
  
#Make directory for save the result file  
os.makedirs("outputs", exist_ok=True)  
  
#Load the data and preprare training, valuation and test dataset for the evaluation  
best_model = load_model("models/pneumonia_efficientNet_finetuned.keras")  
  
def preprocess_for_model(x):  
    x = tf.cast(x, tf.float32)  
    x = tf.ensure_shape(x,[28,28])  
    x = tf.expand_dims(x,-1) #(28,28,1)  
    x = tf.image.resize(x, (224, 224))  
    x = tf.image.grayscale_to_rgb(x)  
    x = preprocess_input(x)  
    x = tf.ensure_shape(x, (224, 224, 3))  
    return x  

#Load the data and preprare training, valuation and test dataset for the evaluation  
test_dataset = PneumoniaMNIST(split="test", download=True)

#Numpy array  
x_np = test_dataset.imgs #(N,28,28) or (N,28,28,1)  
y_np = test_dataset.labels #(N,1)  

#Convert numpy array to tensorflow dataset format  
#→ Able to make data pipeline that take out each samples  
x_np = x_np.squeeze()  
y_np = y_np.squeeze()  
ds_test = tf.data.Dataset.from_tensor_slices((x_np, y_np))  
  
ds_triplet = (ds_test  
              .map(lambda x, y: (x, preprocess_for_model(x), tf.cast(y, tf.int32)),  
                   num_parallel_calls=tf.data.AUTOTUNE)  
              .batch(32)  
              .prefetch(tf.data.AUTOTUNE))  
  
ds_eval = (ds_triplet  
           .map(lambda x_raw, x_proc, y: (x_proc, tf.cast(tf.expand_dims(y, -1), tf.float32)),  
                num_parallel_calls=tf.data.AUTOTUNE)  
           .prefetch(tf.data.AUTOTUNE))  
  
raw_imgs, y_true, p_prob = [], [], []  
  
for x_raw, x_proc, y in ds_triplet:  
    p = best_model.predict(x_proc, verbose=0).reshape(-1)  # sigmoid (N,)  
    raw_imgs.append(x_raw.numpy())                         # (B,28,28,1)  
    y_true.append(y.numpy().reshape(-1))                   # (B,)  
    p_prob.append(p)  
  
raw_imgs = np.concatenate(raw_imgs, axis=0)  
y_true = np.concatenate(y_true, axis=0).astype(int)  
p_prob = np.concatenate(p_prob, axis=0)  
y_pred = (p_prob >= 0.5).astype(int)  
  
# Make index  
mis_idx = np.where(y_pred != y_true)[0]  
cor_idx = np.where(y_pred == y_true)[0]  
fp_idx = np.where((y_true ==0) & (y_pred==1))[0]  
fn_idx = np.where((y_true==1) & (y_pred ==0))[0]  
  
print("Correct:", len(cor_idx), "Misclassified:", len(mis_idx))  
print("Misclassified normal lung as pneumonia: ", len(fp_idx))  
print("Misclassified pneumonia as normal lung: ", len(fn_idx))  
  
#Misclassification with pic  
n_show = 10  
show_mis = mis_idx[:n_show]  
show_cor = cor_idx[:n_show]  
  
fig, axes = plt.subplots(2, n_show, figsize=(3*n_show, 10))  
  
for i, idx in enumerate(show_cor):  
    axes[0, i].imshow(raw_imgs[idx].squeeze(), cmap="gray")  
    axes[0, i].set_title(f"Correct\nT:{y_true[idx]} P:{y_pred[idx]} Prob:{p_prob[idx]:.2f}")  
    axes[0, i].axis("off")  
  
for i, idx in enumerate(show_mis):  
    axes[1, i].imshow(raw_imgs[idx].squeeze(), cmap="gray")  
    axes[1, i].set_title(f"Misclassified\nT:{y_true[idx]} P:{y_pred[idx]} Prob:{p_prob[idx]:.2f}")  
    axes[1, i].axis("off")  
plt.tight_layout()  
plt.savefig("outputs/misclassified_examples.png", dpi=200, bbox_inches="tight")  
print("Saved:outputs/misclassified_examples.png")  
  
#Loss function, Accuracy, AUC(Area under curve)  
test_loss, test_acc, test_auc = best_model.evaluate(ds_eval, verbose=0)  
print(f"Test loss: {test_loss:.4f}")  
print(f"Test accuracy: {test_acc:.4f}")  
print(f"Test AUC: {test_auc:.4f}")  
  
with open("outputs/loss_acc_auc.txt", "w") as f:  
    f.write(f"Test Loss: {test_loss:.4f}\n")  
    f.write(f"Test Accuracy: {test_acc:.4f}\n")  
    f.write(f"Test AUC: {test_auc:.4f}\n")  
  
#Confusion Matrix  
p_test = best_model.predict(ds_eval, verbose=0).reshape(-1)  
  
#Convert 0/1 label  
#y_test = np.array([y.numpy() for _, y in ds_test]).reshape(-1).astype(int)  
y_test = y_true  
  
y_pred = (p_test >= 0.5).astype(int)  
print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred, digits=4))  
  
cm = confusion_matrix(y_test, y_pred)  
plt.figure()  
plt.imshow(cm)  
plt.title("Confusion Matrix")  
plt.colorbar()  
plt.xlabel("Predicted")  
plt.ylabel("True")  
for i in range(2):  
    for j in range(2):  
        plt.text(j, i, cm[i, j], ha="center", va="center")  
plt.tight_layout()  
plt.savefig("outputs/confusion_matrix.png", dpi=200, bbox_inches="tight")  
print("Saved:outputs/confusion_matrix.png")  
  
#ROC curve  
fpr, tpr, thresholds = roc_curve(y_test, p_test)  
auc = test_auc  
  
plt.figure()  
plt.plot(fpr, tpr, label=f"AUC={auc:.4f}")  
plt.plot([0,1], [0,1], linestyle="--", label="Chance")  
plt.xlabel("False Positive Rate")  
plt.ylabel("True Positive Rate")  
plt.legend()  
plt.tight_layout()  
plt.savefig("outputs/roc_curve.png", dpi=200, bbox_inches="tight")  
>>>>>>> 16f0156 (Initial commit - EfficientNet Pneumonia Classification with Docker)
print("Saved:outputs/roc_curve.png")