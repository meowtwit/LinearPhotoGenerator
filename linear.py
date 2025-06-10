#!/usr/bin/env python3
# tiny_warp_gui.py
# -----------------------------------------------
# 画像を選択して ρ（内積）を入力 → 変形＆保存
# -----------------------------------------------

import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path
import numpy as np
import cv2
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from warp_by_vectors import warp, auto_vectors  # ← 前回スクリプトを import

# Tk セットアップ
root = tk.Tk()
root.title("Vector Warp GUI")
root.geometry("1420x1000")

# -- 変数
img_path = tk.StringVar()
rho_value = tk.StringVar(value="0.6")  # デフォルト

# -- コールバック
def choose_file():
    fname = filedialog.askopenfilename(
        title="画像を選択",
        filetypes=[("Image files", "*.png *.jpg *.jpeg"), ("All files", "*.*")])
    if fname:
        img_path.set(fname)

def run_warp():
    path = img_path.get()
    try:
        rho = float(rho_value.get())
    except ValueError:
        messagebox.showerror("入力エラー", "ρ（内積）は数値で入力してください")
        return
    if abs(rho) >= 1 or abs(rho) < 1e-9:
        messagebox.showerror("入力エラー", "ρ には 0, ±1 を除く -1<ρ<1 を指定してください")
        return
    if not Path(path).is_file():
        messagebox.showerror("入力エラー", "画像ファイルを選択してください")
        return

    # 変形処理
    try:
        v, w = auto_vectors(rho)          # |v|=|w|=1 で希望の内積を生成
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        warped, H = warp(img, v, w)       # 既存関数を使って線形変換

        # 矢印を matplotlib で描画
        plt.figure(figsize=(6,6))
        origin = H[:2,2]
        def draw(vec, col, label):
            plt.arrow(*origin, *(vec*100), width=2,
                      head_width=12, head_length=18,
                      color=col, length_includes_head=True)
            plt.text(origin[0]+vec[0]*100, origin[1]+vec[1]*100,
                     f" {label}", color=col, fontsize=12, weight="bold")
        draw(v, "red", "v"); draw(w, "blue", "w")
        plt.imshow(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.tight_layout()

        # 保存先（元ファイル名＋_warp.png）
        out_path = str(Path(path).with_suffix("")) + "_warp.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()

        messagebox.showinfo("完了",
            f"変換完了！\n保存先: {out_path}\n\nv·w = {np.dot(v,w):.4f}")
    except Exception as e:
        messagebox.showerror("エラー", str(e))
        # --- 画像プレビュー表示
    pil_img = Image.open(out_path)                    # 保存した画像を Pillow で開く
    pil_img = pil_img.resize((200, 200))              # 必要ならリサイズ
    img_tk = ImageTk.PhotoImage(pil_img)              # Tkinter用に変換
    preview_label.config(image=img_tk)
    preview_label.image = img_tk  # 参照保持が必要！これがないと画像が消える


# -- UI パーツ
tk.Label(root, text="画像ファイル:").grid(row=0, column=0, sticky="e", pady=6)
tk.Entry(root, textvariable=img_path, width=36).grid(row=0, column=1, padx=4)
tk.Button(root, text="参照...", command=choose_file).grid(row=0, column=2)

tk.Label(root, text="ρ (内積):").grid(row=1, column=0, sticky="e", pady=6)
tk.Entry(root, textvariable=rho_value, width=10).grid(row=1, column=1, sticky="w", padx=4)

tk.Button(root, text="変換して保存", command=run_warp,
          bg="#4CAF50", fg="white", width=18).grid(row=3, column=1, pady=18)
# --- プレビュー用のラベル
preview_label = tk.Label(root)
preview_label.grid(row=4, column=0, columnspan=3, pady=10)



root.mainloop()
