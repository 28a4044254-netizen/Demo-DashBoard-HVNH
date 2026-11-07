import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib

# 1. Sinh bộ dữ liệu mẫu 200 sinh viên chuẩn quy chế tín chỉ HVNH
np.random.seed(42)
n = 200

df = pd.DataFrame({
    'Ma_SV': [f'SV24{str(i).zfill(4)}' for i in range(1, n + 1)],
    'Ho_Ten': [f'Nguyễn Văn Sinh Viên {i}' for i in range(1, n + 1)],
    'Lop_Nien_Che': np.random.choice(['K25-HTTTQL', 'K26-NH', 'K27-TC', 'K27-KT'], n),
    'GPA_Ky_Truoc': np.round(np.random.uniform(0.6, 3.9, n), 2),
    'GPA_Tich_Luy': np.round(np.random.uniform(0.8, 3.8, n), 2),
    'Tin_Chi_Dang_Ky': np.random.choice([12, 15, 18, 21, 24], n),
    'Tin_Chi_No': np.random.choice([0, 0, 0, 3, 6, 9, 12, 15, 18], n),
    'Ty_Le_Vang_Mat': np.random.randint(0, 40, n),
    'Diem_Ren_Luyen': np.random.randint(45, 100, n)
})

# Quy tắc gán nhãn thực tế để dạy AI (1: Nguy cơ cảnh báo học vụ, 0: An toàn)
dieukien_canhbao = (
    (df['GPA_Ky_Truoc'] < 1.2) | 
    (df['Tin_Chi_No'] >= 12) | 
    ((df['GPA_Ky_Truoc'] < 1.6) & (df['Ty_Le_Vang_Mat'] >= 20))
)
df['Nhan_Canh_Bao'] = np.where(dieukien_canhbao, 1, 0)

# Xuất ra file CSV mẫu để dùng thử chức năng Upload trên Dashboard
df.to_csv('data_hvnh_mau.csv', index=False, encoding='utf-8-sig')
print("1. Đã tạo thành công file dữ liệu mẫu: data_hvnh_mau.csv")

# 2. Huấn luyện mô hình Random Forest
features = ['GPA_Ky_Truoc', 'GPA_Tich_Luy', 'Tin_Chi_Dang_Ky', 'Tin_Chi_No', 'Ty_Le_Vang_Mat', 'Diem_Ren_Luyen']
X = df[features]
y = df['Nhan_Canh_Bao']

model = RandomForestClassifier(n_estimators=150, max_depth=6, random_state=42)
model.fit(X, y)

# 3. Lưu mô hình AI ra file .pkl
joblib.dump(model, 'model.pkl')
print("2. Đã huấn luyện và lưu mô hình AI thành công vào file: model.pkl")