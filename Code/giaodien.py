import pandas as pd
import random
from sklearn.model_selection import train_test_split
import numpy as np
from tkinter import *
from tkinter import messagebox
import tkinter as tk
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression # Thêm Logistic Regression
from sklearn.neighbors import KNeighborsClassifier # Thêm KNN

# --- 1. Tải và Chia Dữ Liệu ---
data = pd.read_csv('../Dataset/heart.csv')

# Tách features (X) và target (y)
X = data.drop(['target'], axis=1)
y = data['target']

# Chia dữ liệu: Train/Test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, shuffle=True, random_state=42
)

# Khởi tạo và Huấn luyện Standard Scaler trên dữ liệu TRAIN
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test) # Áp dụng scaler lên dữ liệu test

y_test_array = np.array(y_test) # Chuyển y_test thành numpy array

# --- 2. Huấn Luyện Các Mô Hình VỚI THAM SỐ TỐI ƯU ---

# Tham số tối ưu từ Grid Search
BEST_C_LOGREG = 0.1
BEST_K_KNN = 1

# Hồi quy Logistic (LogReg) - Áp dụng C tối ưu
logreg = LogisticRegression(C=BEST_C_LOGREG, solver='liblinear', random_state=42)
logreg.fit(X_train_scaled, y_train)

# K-Láng giềng Gần nhất (KNN) - Áp dụng K tối ưu
knn = KNeighborsClassifier(n_neighbors=BEST_K_KNN)
knn.fit(X_train_scaled, y_train)


# --- 3. Cấu Hình Giao Diện Tkinter ---

# FORM
root = Tk()
root.geometry("950x750") # Điều chỉnh kích thước để chứa thông tin tham số
root.title("Dự đoán bệnh tim của bệnh nhân (LogReg & KNN - Tối ưu)")

# Tạo danh sách các feature và Entry Boxes
features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
textboxes = {}

lable_ten = Label(root, text="Nhập thông tin bệnh nhân!", font=("Arial Bold", 12), fg="green")
lable_ten.grid(row=0, column=1, padx=40, pady=10, columnspan=2)

# Hiển thị tham số tối ưu
lbl_best_params = Label(root, text=f"Tham số Tối ưu:\n"
                                   f"LogReg: C = {BEST_C_LOGREG}\n"
                                   f"KNN: K = {BEST_K_KNN}",
                        font=("Arial", 10), fg="purple", justify=LEFT)
lbl_best_params.grid(row=0, column=4, padx=10, pady=10, columnspan=2, sticky='w')


# Thiết lập layout cho các trường nhập liệu
for i, feature in enumerate(features):
    row = i % 7 + 1
    col_label = (i // 7) * 3 + 1
    col_textbox = (i // 7) * 3 + 2

    label = Label(root, text=f" {feature.capitalize()}:")
    label.grid(row=row, column=col_label, padx=10, pady=5, sticky='w')

    textbox = Entry(root, width=15)
    textbox.grid(row=row, column=col_textbox, padx=10, pady=5, sticky='w')
    textboxes[feature] = textbox


# Hàm lấy ngẫu nhiên một bộ dữ liệu từ X_test (dữ liệu gốc chưa scale)
def diendulieu():
    # Sử dụng X_test (chưa scale) để lấy dữ liệu thô
    index = random.randint(0, len(X_test) - 1)
    test_data = X_test.iloc[index]

    for i, feature in enumerate(features):
        textboxes[feature].delete(0, tk.END)
        textboxes[feature].insert(tk.END, test_data[i])

    # Hiển thị nhãn thực tế
    actual_target = y_test_array[index]
    lbl_actual.config(text=f"Nhãn thực tế (Target): {actual_target} ({'Có bệnh' if actual_target == 1 else 'Không bệnh'})")

button_Random = Button(root, text='Điền dữ liệu Ngẫu nhiên', command=diendulieu, bg='lightblue')
button_Random.grid(row=9, column=3, padx=20, pady=10)

lbl_actual = Label(root, text="Nhãn thực tế (Target): ...", fg="blue", font=("Arial", 10))
lbl_actual.grid(row=10, column=3, pady=5)


# --- 4. Hàm Chức Năng Chung ---

# Hàm lấy dữ liệu nhập và chuẩn hóa
def get_and_scale_input():
    # Lấy dữ liệu thô từ các ô nhập liệu
    input_data = [textboxes[f].get() for f in features]
    if any(data == '' for data in input_data):
        messagebox.showinfo("Thông báo", "Bạn cần nhập đầy đủ thông tin!")
        return None
    try:
        # Chuyển đổi thành số thực (float)
        x_test_raw = np.array(input_data, dtype=float).reshape(1, -1)
        # Chuẩn hóa dữ liệu đầu vào bằng scaler đã huấn luyện
        x_test_scaled = scaler.transform(x_test_raw)
        return x_test_scaled
    except ValueError:
        messagebox.showerror("Lỗi", "Dữ liệu nhập vào phải là số!")
        return None

# --- 5. Hồi quy Logistic (LogReg - C=0.1) ---
y_logreg = logreg.predict(X_test_scaled)
metrics_logreg = {
    "Accuracy": accuracy_score(y_test_array, y_logreg),
    "Precision": precision_score(y_test_array, y_logreg),
    "Recall": recall_score(y_test_array, y_logreg),
    "F1-score": f1_score(y_test_array, y_logreg),
}

lbl_logreg_metrics = Label(root, justify=LEFT, font=("Arial", 10))
lbl_logreg_metrics.grid(row=15, column=1, columnspan=2, padx=10, pady=5, sticky='w')
lbl_logreg_metrics.configure(text=
    f"Tỉ lệ dự đoán đúng của **LogReg (C={BEST_C_LOGREG})**:\n" +
    "\n".join([f"{k}: {v:.4f}" for k, v in metrics_logreg.items()])
)

def dudoan_logreg():
    x_test_scaled = get_and_scale_input()
    if x_test_scaled is not None:
        # Sử dụng mô hình đã tối ưu để dự đoán
        prediction = logreg.predict(x_test_scaled)[0]
        result = "Có bệnh tim (1)" if prediction == 1 else "Không có bệnh tim (0)"
        lbl_logreg_result.config(text=result, fg='red' if prediction == 1 else 'green')

button_logreg = Button(root, text=f'Kết quả dự đoán LogReg (C={BEST_C_LOGREG})', command=dudoan_logreg, bg='yellow')
button_logreg.grid(row=16, column=1, padx=5, pady=10, sticky='w')
lbl_logreg_result = Label(root, text="...", font=("Arial Bold", 10))
lbl_logreg_result.grid(row=16, column=2, padx=5, sticky='w')

# --- 6. KNN (K=1) ---
y_knn = knn.predict(X_test_scaled)
metrics_knn = {
    "Accuracy": accuracy_score(y_test_array, y_knn),
    "Precision": precision_score(y_test_array, y_knn),
    "Recall": recall_score(y_test_array, y_knn),
    "F1-score": f1_score(y_test_array, y_knn),
}

lbl_knn_metrics = Label(root, justify=LEFT, font=("Arial", 10))
lbl_knn_metrics.grid(row=15, column=3, columnspan=2, padx=10, pady=5, sticky='w')
lbl_knn_metrics.configure(text=
    f"Tỉ lệ dự đoán đúng của **KNN (K={BEST_K_KNN})**:\n" +
    "\n".join([f"{k}: {v:.4f}" for k, v in metrics_knn.items()])
)

def dudoan_knn():
    x_test_scaled = get_and_scale_input()
    if x_test_scaled is not None:
        # Sử dụng mô hình đã tối ưu để dự đoán
        prediction = knn.predict(x_test_scaled)[0]
        result = "Có bệnh tim (1)" if prediction == 1 else "Không có bệnh tim (0)"
        lbl_knn_result.config(text=result, fg='red' if prediction == 1 else 'green')

button_knn = Button(root, text=f'Kết quả dự đoán KNN (K={BEST_K_KNN})', command=dudoan_knn, bg='lightgreen')
button_knn.grid(row=16, column=3, padx=5, pady=10, sticky='w')
lbl_knn_result = Label(root, text="...", font=("Arial Bold", 10))
lbl_knn_result.grid(row=16, column=4, padx=5, sticky='w')


root.mainloop()