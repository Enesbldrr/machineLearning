import numpy as np
import time

# ==============================================
# 1. ORTAK MATRİS TANIMI
# ==============================================

A = np.array([[4, 2],
              [1, 3]])
print("Matris A:\n", A)

# ==============================================
# 2. NumPy Kullanarak Özdeğer ve Özvektör Hesabı
# ==============================================

start_numpy = time.time()

eigenvalues_np, eigenvectors_np = np.linalg.eig(A)

end_numpy = time.time()

print("\n[NumPy ile Hesaplama]")
print("Özdeğerler:\n", eigenvalues_np)
print("Özvektörler (sütunlar):\n", eigenvectors_np)
print("Süre: {:.6f} saniye".format(end_numpy - start_numpy))

# ==============================================
# 3. Manuel Özdeğer ve Özvektör Hesabı (2x2 için)
# ==============================================

# Özdeğerler için karakteristik polinom çözümü: det(A - λI) = 0
# | 4-λ   2   |
# | 1     3-λ | = (4-λ)(3-λ) - 2*1 = λ^2 - 7λ + 10

coefficients = [1, -7, 10]  # λ^2 - 7λ + 10
eigenvalues_manual = np.roots(coefficients)

print("\n[Manuel Hesaplama - Özdeğerler]")
print("Özdeğerler:\n", eigenvalues_manual)

# Özvektörleri elle bulmak için: (A - λI) v = 0 çözülür
def find_eigenvector(A, eigenvalue):
    B = A - eigenvalue * np.identity(A.shape[0])
    # Homojen denklem çözümü için herhangi bir çözüm v
    # np.linalg.svd ile en küçük tekil değere karşılık gelen vektörü alabiliriz
    u, s, vh = np.linalg.svd(B)
    return vh[-1]

eigenvectors_manual = []
for val in eigenvalues_manual:
    vec = find_eigenvector(A, val)
    eigenvectors_manual.append(vec / np.linalg.norm(vec))  # normalize et

eigenvectors_manual = np.array(eigenvectors_manual).T  # sütun olarak düzenle

print("\n[Manuel Hesaplama - Özvektörler]")
print("Özvektörler (sütunlar):\n", eigenvectors_manual)

# ==============================================
# 4. Karşılaştırma
# ==============================================

print("\n[Karşılaştırma]")
print("Özdeğerler aynı mı?:", np.allclose(np.sort(eigenvalues_np), np.sort(eigenvalues_manual)))
print("Özvektörler aynı mı (yön olarak)?:", np.allclose(np.abs(eigenvectors_np), np.abs(eigenvectors_manual)))

# Av = λv denklemi test (NumPy özvektörleriyle)
for i in range(len(eigenvalues_np)):
    left = A @ eigenvectors_np[:, i]
    right = eigenvalues_np[i] * eigenvectors_np[:, i]
    print(f"\nλ{i+1} için Av = λv testi (NumPy): {np.allclose(left, right)}")

# Süre bilgisi yukarıda yazıldı

