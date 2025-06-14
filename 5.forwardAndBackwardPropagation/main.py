import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from neural_network import NeuralNetwork
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Veri okuma
df = pd.read_csv("iris.csv")
X = df.drop(["species"], axis=1).values
y = df["species"].values

# Etiketleri sayısal verilere çevir
encoder = LabelEncoder()
y = encoder.fit_transform(y)

# Veri ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modeli oluştur
input_size = X.shape[1]  # 4
hidden_size = 10
output_size = len(np.unique(y)) # 3 sınıfımız var
model = NeuralNetwork(input_size, hidden_size, output_size)

# Eğitim kısmı
model.train(X_train, y_train, epochs=1000, learning_rate=0.1)

# Test verisiyle tahmin
y_pred = model.predict(X_test)

# Başarı oranı
acc = accuracy_score(y_test, y_pred)
print(f"Test doğruluğu: {acc:.2f}")

# Loss grafiği çizimi
plt.plot(model.losses)
plt.title("Loss vs Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.tight_layout()
plt.savefig("loss_graph.png")
plt.show()

# Karışıklık matrisi
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=encoder.classes_)
disp.plot(cmap="Blues", values_format="d")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()