import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import stft, resample
from ultralytics import YOLO

# ===== 学習時と同じ前処理パラメータに合わせる =====
OUT_W = 1000
OUT_H = 1000

NPERSEG = 256          # stftのデフォルト窓長 
NOVERLAP = 0           # hop=256にしたいなら0
NFFT = 4096

FREQ_MAX_HZ = 2000     # あなたの指定（0〜3000Hz表示）
TRAIN_MAX_DURATION_SEC = 0.888  # ★ここを学習時に使った「最長秒」に置き換えてください

def read_wav_mono_float(path: str):
    fs, x = wavfile.read(path) 
    if x.dtype.kind in "iu":
        x = x.astype(np.float32) / np.iinfo(x.dtype).max
    else:
        x = x.astype(np.float32)
    if x.ndim == 2:
        x = x.mean(axis=1)
    return fs, x

def wav_to_spec_image(wav_path: str, out_png_path: str):
    fs, x = read_wav_mono_float(wav_path)

    f, t, Zxx = stft(
        x, fs=fs,
        nperseg=NPERSEG,
        noverlap=NOVERLAP,
        nfft=NFFT,
        boundary=None, padded=False,
        return_onesided=True
    )

    S_db = 20.0 * np.log10(np.abs(Zxx) + 1e-10)

    # 0〜3000Hzに制限（上限を超える周波数ビンを捨てる）
    f_mask = f <= FREQ_MAX_HZ
    f = f[f_mask]
    S_db = S_db[f_mask, :]

    # 学習時と同じ「最長秒」基準で時間を0〜1に正規化（※ここが重要）
    t_norm = t / TRAIN_MAX_DURATION_SEC

    # 2000×2000に等間隔サンプル化（時間=OUT_W, 周波数=OUT_H）
    S_time = resample(S_db, OUT_W, axis=1)   # time方向
    S_img  = resample(S_time, OUT_H, axis=0) # freq方向 

    # 画像保存（学習時と同じ描画方式に合わせること）
    plt.figure(figsize=(OUT_W/200, OUT_H/200), dpi=200)
    plt.imshow(S_img, origin="lower", aspect="auto", extent=[0, 1, 0, FREQ_MAX_HZ])
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(out_png_path, bbox_inches="tight", pad_inches=0)
    plt.close()

    return out_png_path

def main():
    # 1) wav -> スペクトログラムpng
    wav_path = r"path/to/input.wav"
    tmp_png = r"tmp_spec.png"
    wav_to_spec_image(wav_path, tmp_png)

    # 2) YOLO分類推論
    model = YOLO(r"runs/classify/train/weights/best.pt")
    results = model.predict(source=tmp_png, imgsz=2000)
    print(results[0])

if __name__ == "__main__":
    main()
