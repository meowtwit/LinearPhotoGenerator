#!/usr/bin/env python3
# vector_warp_gui.py
# -----------------------------------------------
# 画像を選択し、ρ (内積) と |v|, |w|, θ のうち2つ + 出力サイズを入力 → 残りを補完 → 変形＆プレビュー
# -----------------------------------------------

import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path
import numpy as np
import math
import cv2
from PIL import Image, ImageTk
import matplotlib.pyplot as plt

# ---------------------- 変形ロジック ---------------------- #
def warp(img, e1, e2):
    H = np.array([[e1[0], e2[0], 0],
                  [e1[1], e2[1], 0],
                  [0,      0,     1]], dtype=np.float32)
    h0, w0 = img.shape[:2]
    corners = np.array([[0,0,1],[w0,0,1],[w0,h0,1],[0,h0,1]], dtype=np.float32).T
    warped_corners = H @ corners
    xs, ys = warped_corners[0], warped_corners[1]
    min_x, max_x = xs.min(), xs.max()
    min_y, max_y = ys.min(), ys.max()
    translate = np.array([[1,0,-min_x],
                          [0,1,-min_y],
                          [0,0,     1]], dtype=np.float32)
    Htot = translate @ H
    out_w, out_h = int(np.ceil(max_x-min_x)), int(np.ceil(max_y-min_y))
    warped = cv2.warpPerspective(img, Htot, (out_w, out_h))
    return warped, Htot

# ---------------------- GUI セットアップ ---------------------- #
root = tk.Tk()
root.title("Vector Warp GUI")
root.geometry("640x650")

# 変数定義
img_path     = tk.StringVar()
mag_v_input  = tk.StringVar()
mag_w_input  = tk.StringVar()
theta_input  = tk.StringVar()
rho_input    = tk.StringVar()
out_w_input  = tk.StringVar()
out_h_input  = tk.StringVar()
preview_label = tk.Label(root)

# ファイル選択
def choose_file():
    f = filedialog.askopenfilename(
        title="画像を選択",
        filetypes=[("Image files","*.png *.jpg *.jpeg"),("All files","*.*")])
    if f: img_path.set(f)

# 補完ロジック
def fill_missing():
    try:
        rho = float(rho_input.get())
    except:
        messagebox.showerror("入力エラー","内積ρを数値で入力してください")
        return
    mag_v = None; mag_w = None; theta = None
    try: mag_v = float(mag_v_input.get())
    except: pass
    try: mag_w = float(mag_w_input.get())
    except: pass
    try: theta = float(theta_input.get())
    except: pass
    cnt = sum([mag_v is not None, mag_w is not None, theta is not None])
    if cnt != 2:
        messagebox.showerror("入力エラー","|v|, |w|, θ のうち2つを入力してください")
        return
    # θ 未入力 → 計算
    if mag_v is not None and mag_w is not None:
        val = rho / (mag_v * mag_w)
        th = math.degrees(math.acos(max(-1.0, min(1.0, val))))
        theta_input.set(f"{th:.2f}")
    # |w| 未入力 → 計算
    elif mag_v is not None and theta is not None:
        rad = math.radians(theta)
        if abs(math.cos(rad)) < 1e-6:
            messagebox.showerror("計算エラー","cosθが0に近いため |w| を計算できません")
            return
        mag_w = rho / (mag_v * math.cos(rad))
        out_h_input.set(out_h_input.get())  # パス
        mag_w_input.set(f"{mag_w:.3f}")
    # |v| 未入力 → 計算
    elif mag_w is not None and theta is not None:
        rad = math.radians(theta)
        if abs(math.cos(rad)) < 1e-6:
            messagebox.showerror("計算エラー","cosθが0に近いため |v| を計算できません")
            return
        mag_v = rho / (mag_w * math.cos(rad))
        mag_v_input.set(f"{mag_v:.3f}")

# リセット
def reset_fields():
    img_path.set("")
    mag_v_input.set("")
    mag_w_input.set("")
    theta_input.set("")
    rho_input.set("")
    out_w_input.set("")
    out_h_input.set("")
    preview_label.config(image="")
    preview_label.image = None

# 実行
def run_warp():
    try:
        mag_v = float(mag_v_input.get())
        mag_w = float(mag_w_input.get())
        theta = float(theta_input.get())
        rho   = float(rho_input.get())
    except:
        messagebox.showerror("入力エラー","|v|, |w|, θ, ρ をすべて正しく入力してください")
        return
    # input サイズ
    out_w = None; out_h = None
    try: out_w = int(out_w_input.get()); out_h = int(out_h_input.get())
    except: pass
    # ベクトル構成
    v = np.array([mag_v, 0.])
    rad = math.radians(theta)
    w = np.array([mag_w * math.cos(rad), mag_w * math.sin(rad)])
    # 画像読み込み＆変形
    path = img_path.get()
    img  = cv2.imread(path, cv2.IMREAD_COLOR)
    warped, H = warp(img, v, w)
    # 出力サイズ指定があればリサイズ
    if out_w and out_h and out_w>0 and out_h>0:
        warped = cv2.resize(warped, (out_w, out_h), interpolation=cv2.INTER_AREA)
    # 保存
    out = str(Path(path).with_suffix("")) + "_warp.png"
    plt.figure(figsize=(6,6))
    origin = H[:2,2]
    def draw(vec, clr, lbl):
        plt.arrow(*origin, *(vec*100), width=2,
                  head_width=12, head_length=18,
                  color=clr, length_includes_head=True)
        plt.text(origin[0]+vec[0]*100,
                 origin[1]+vec[1]*100,
                 f" {lbl}", color=clr, fontsize=12, weight="bold")
    draw((v/np.linalg.norm(v)), "red",  "v")
    draw((w/np.linalg.norm(w)), "blue", "w")
    plt.imshow(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
    plt.axis("off"); plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    # プレビュー表示
    pil = Image.open(out).resize((240,240))
    imgtk = ImageTk.PhotoImage(pil)
    preview_label.config(image=imgtk)
    preview_label.image = imgtk
    messagebox.showinfo("完了", f"保存: {out}\n出力サイズ: {warped.shape[1]}x{warped.shape[0]}")

# UI 配置
tk.Label(root, text="画像ファイル:").grid(row=0, column=0, sticky="e")
tk.Entry(root, textvariable=img_path, width=40).grid(row=0, column=1)
tk.Button(root, text="参照", command=choose_file).grid(row=0, column=2)

tk.Label(root, text="|v|:").grid(row=1, column=0, sticky="e")
tk.Entry(root, textvariable=mag_v_input, width=40).grid(row=1, column=1)

tk.Label(root, text="|w|:").grid(row=2, column=0, sticky="e")
tk.Entry(root, textvariable=mag_w_input, width=40).grid(row=2, column=1)

tk.Label(root, text="θ (°):").grid(row=3, column=0, sticky="e")
tk.Entry(root, textvariable=theta_input, width=40).grid(row=3, column=1)

tk.Label(root, text="ρ (内積):").grid(row=4, column=0, sticky="e")
tk.Entry(root, textvariable=rho_input, width=40).grid(row=4, column=1)

tk.Label(root, text="出力 幅(px):").grid(row=5, column=0, sticky="e")
tk.Entry(root, textvariable=out_w_input, width=40).grid(row=5, column=1)

tk.Label(root, text="出力 高さ(px):").grid(row=6, column=0, sticky="e")
tk.Entry(root, textvariable=out_h_input, width=40).grid(row=6, column=1)

# 操作ボタン
tk.Button(root, text="補完", command=fill_missing).grid(row=7, column=1, sticky="w", pady=8)
tk.Button(root, text="変換", command=run_warp, bg="#4CAF50", fg="white").grid(row=7, column=1, sticky="e", pady=8)
tk.Button(root, text="リセット", command=reset_fields).grid(row=7, column=2)

# プレビュー領域
preview_label.grid(row=8, column=0, columnspan=3, pady=10)
root.mainloop()
