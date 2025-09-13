# Dữ liệu mẫu - thay bằng kết quả thực tế
horizons = [
    ("short_term", 60,  6,  1),
    ("medium_term",180, 12,  5),
    ("long_term", 360, 12, 30),
    ("CNN_LSTM", 120, 24, 5)
]
model_types = ["LSTM", "biLSTM", "GRU"]
results = {
    ("short_term",  "LSTM"):     {"mae": 2.46, "rmse": 2.74, "r2": 0.75},
    ("short_term",  "biLSTM"):   {"mae": 2.85, "rmse": 3.67, "r2": 0.67},
    ("short_term",  "GRU"):      {"mae": 2.74, "rmse": 3.64, "r2": 0.68},
    ("medium_term", "LSTM"):     {"mae": 2.74, "rmse": 3.72, "r2": 0.67},
    ("medium_term", "biLSTM"):   {"mae": 2.84, "rmse": 3.90, "r2": 0.63},
    ("medium_term", "GRU"):      {"mae": 2.20, "rmse": 2.60, "r2": 0.72},
    ("long_term",   "GRU"):     {"mae": 3.10, "rmse": 4.18, "r2": 0.58},
    ("long_term",   "biLSTM"):   {"mae": 3.61, "rmse": 4.69, "r2": 0.47},
    ("long_term",   "LSTM"):      {"mae": 3.66, "rmse": 4.86, "r2": 0.43},
    ("CNN_LSTM", "GRU"):     {"mae": 3.48, "rmse": 4.33, "r2": 0.54},
    ("CNN_LSTM", "LSTM"):    {"mae": 3.15, "rmse": 4.28, "r2": 0.56},
    ("CNN_LSTM", "biLSTM"):  {"mae": 1318002345, "rmse": 5740202022, "r2": -7.99},
}

import tkinter as tk

root = tk.Tk()
root.title("So sánh Các Model Theo Time Horizon")

# Tiêu đề cột
headers = ["Horizon", "Model", "In", "Out", "Gr", "MAE", "RMSE", "R2"]
for j, h in enumerate(headers):
    lbl = tk.Label(root, text=h, font=("Arial", 10, "bold"), borderwidth=1, relief="solid", padx=5, pady=5)
    lbl.grid(row=0, column=j, sticky="nsew")

# Dữ liệu bảng
row_idx = 1
for horizon, n_in, n_out, gran in horizons:
    for mt in model_types:
        vals = results.get((horizon, mt), {"mae":0,"rmse":0,"r2":0})
        row = [
            horizon, mt,
            str(n_in), str(n_out), str(gran),
            f"{vals['mae']:.2f}",
            f"{vals['rmse']:.2f}",
            f"{vals['r2']:.2f}"
        ]
        for col_idx, text in enumerate(row):
            lbl = tk.Label(root, text=text, font=("Arial", 10),
                           borderwidth=1, relief="solid", padx=5, pady=5)
            lbl.grid(row=row_idx, column=col_idx, sticky="nsew")
        row_idx += 1

# Điều chỉnh kích thước cột
for i in range(len(headers)):
    root.grid_columnconfigure(i, weight=1)

root.mainloop()