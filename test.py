import pandas as pd

# Baca file CSV untuk data uji
test_data = pd.read_csv(r'uts/emnist-bymerge-test.csv')

# Kolom label, ganti jika nama kolom berbeda
label_column = '24'

# Hitung jumlah kelas dan tentukan jumlah sampel per kelas untuk data uji
num_classes = test_data[label_column].nunique()
samples_per_class_test = 3000 // num_classes  # Contoh: set 750 sampel untuk data uji

# Ambil sampel merata per kelas
balanced_test_data = test_data.groupby(label_column, group_keys=False).apply(lambda x: x.sample(samples_per_class_test))

# Pastikan jumlah total sampel sesuai dengan yang diinginkan, misal 750 sampel
if len(balanced_test_data) > 3000:
    balanced_test_data = balanced_test_data.sample(3000)

# Tampilkan distribusi kelas untuk verifikasi
print(balanced_test_data[label_column].value_counts())

# Simpan hasilnya ke file CSV baru jika diperlukan
balanced_test_data.to_csv('balanced_emnist_test_subset.csv', index=False)
