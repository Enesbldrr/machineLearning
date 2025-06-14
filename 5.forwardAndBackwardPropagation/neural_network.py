import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.W1 = np.random.randn(input_size, hidden_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size)
        self.b2 = np.zeros((1, output_size))

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def sigmoid_derivative(self, z):
        sig = self.sigmoid(z)
        return sig * (1 - sig)

    def forward(self, X):
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = self.sigmoid(self.Z1)
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = self.sigmoid(self.Z2)   # Çıkış katmanı
        return self.A2

    def backward(self, X, Y, output, learning_rate):
        # 1. Çıkış katmanının hatası
        m = Y.shape[0] # örnek sayısı
        y_one_hot = np.zeros_like(output)
        y_one_hot[np.arange(m), Y] = 1  # one-hot vektöre çevir

        dZ2 = output - y_one_hot  # çıktı - gerçek
        dW2 = np.dot(self.A1.T, dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m

        # 2. Gizli katmanın hatası
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * self.sigmoid_derivative(self.Z1)
        dW1 = np.dot(X.T, dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m

        # 3. Ağırlıkları ve biasları güncelle
        self.W2 -= learning_rate * dW2
        self.W1 -= learning_rate * dW1
        self.b2 -= learning_rate * db2
        self.b1 -= learning_rate * db1

    def train(self, X, Y, epochs, learning_rate):
        self.losses = []

        for epoch in range(epochs):
            # 1. İleri yayılım
            output = self.forward(X)

            # 2. Loss hseabı
            m = Y.shape[0]
            y_one_hot = np.zeros_like(output)
            y_one_hot[np.arange(m), Y] = 1
            loss = -np.mean(np.sum(y_one_hot * np.log(output + 1e-8), axis=1))
            self.losses.append(loss)

            # 3. Geri yayılım
            self.backward(X, Y, output, learning_rate)

            # 4. 100 epoch başında yazdır
            if (epoch + 1) % 100 == 0 or epoch == 0:
                print(f"Epoch {epoch+1}/{epochs} - Loss: {loss:.4f}")

    def predict(self, X):
        output = self.forward(X)
        return np.argmax(output, axis=1)