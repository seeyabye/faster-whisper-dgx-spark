# whisper-test (faster-whisper on DGX Spark/ThinkStation PGX)

NVIDIA DGX Spark 相当環境（Lenovo ThinkStation PGX）で、`faster-whisper` を CUDA で確実に動かすための最小構成をまとめたリポジトリです。ホスト側 CUDA 13 が既に入っており変更しづらい研究用マシンでも、Docker 上で CUDA 12.4 + cuDNN 9 でビルドした CTranslate2 を用いて GPU 推論を再現できます。

公式 faster-whisper リポジトリ: https://github.com/SYSTRAN/faster-whisper

## なぜこの構成が必要か
- `faster-whisper` は CTranslate2 を利用し、ビルド時の CUDA/cuDNN と実行環境が一致していないと GPU 版が動作しない。
- ThinkStation PGX ではホストに CUDA 13 が入っており、`pip install ctranslate2` すると CPU 版や CUDA 非対応ビルドが入ることがある。
- 公式手順だけでは CUDA 版が入らず、`ValueError: This CTranslate2 package was not compiled with CUDA support` となるケースがあった。
- そこで CUDA 12.4 + cuDNN 9 を明示した Docker で CTranslate2 をソースビルドし、Python バインディングを明示的に再インストールすることで再現性を確保。

## 提供物
- `docker-compose.yml` と `docker/Dockerfile` による GPU 対応コンテナ構成
  - ベース: `nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04`
  - CTranslate2 v4.6.2 を CUDA 有効でソースビルド
  - `pip` で入る CPU 版 `ctranslate2` を一度アンインストールし、ビルド済み Python バインディングを再インストール
- FastAPI + Uvicorn の簡易サーバ (`docker/server.py`)
  - `POST /transcribe` で音声を受け取り、Whisper 推論結果を返却
  - `GET /health` でモデル名・デバイス・compute_type を返却
- サンプル音声とテストスクリプト（`test/`）

## 仕組みのポイント
- CUDA 12.x + cuDNN 9 で CTranslate2 をビルドし、実行時も同一バージョンを保証
- `entrypoint.sh` で `LD_LIBRARY_PATH` を明示設定し、`nvidia.cublas.lib` / `nvidia.cudnn.lib` のパスを動的に追加
- ホスト CUDA 13 と切り離すため Docker を必須とし、NVIDIA Container Toolkit を前提に GPU を渡す

## 前提条件
- NVIDIA Container Toolkit 導入済み
- GPU ドライバがホストに正しく入っていること
- ホスト側 CUDA は 13 系のままで可（変更不要）

### ローカルCPU実行（任意）
Docker を使わず手元の Python (.venv) で CPU 推論だけ試す場合の簡易手順:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r docker/requirements.txt

# CPU でサンプル音声を実行
FW_DEVICE=cpu FW_COMPUTE_TYPE=int8 python3 test/transcribe_cpu.py
```
※ CPU のみの確認用。GPU を使う場合は Docker 構成を推奨。

## セットアップ
```bash
# ビルド
docker compose build

# 起動（GPU 必須）
docker compose up -d

# ヘルスチェック
curl http://localhost:8002/health
```

## 推論APIの使い方
```bash
curl -X POST http://localhost:8002/transcribe \
  -F "file=@test/003.wav" \
  -F "language=ja" \
  -F "beam_size=5" \
  -F "vad_filter=true"
```
レスポンスには `text` と `segments`（start/end/text）、`language`, `language_probability` が含まれます。

環境変数でモデルや実行設定を変更できます:
- `FW_MODEL` (デフォルト `large-v3`)
- `FW_DEVICE` (デフォルト `cuda`)
- `FW_COMPUTE_TYPE` (デフォルト `float16`)

### Docker コンテナ内で `test/` のスクリプトを実行する例
ホストに Python を入れずに検証したい場合、起動済みコンテナ内でテストスクリプトを実行できます。
```bash
# 1) コンテナに入る（compose が起動している前提）
docker compose exec whisper bash

# 2) GPU 版サンプル（デフォルトの large-v3 / float16）
python3 test/transcribe.py

# 3) CPU で試したい場合（compute_type を切り替え）
FW_DEVICE=cpu FW_COMPUTE_TYPE=int8 python3 test/transcribe_cpu.py
```
いずれも `test/003.wav` を入力にし、標準出力へテキストと各セグメントを表示します。自分の音声で試す場合は `python3 test/transcribe.py /path/to/your.wav` のようにパスを指定してください。

## ディレクトリ構成（抜粋）
- `docker-compose.yml` — サービス定義（GPU リクエスト含む）
- `docker/Dockerfile` — CUDA 12.4 + cuDNN 9 ベースで CTranslate2 をソースビルド
- `docker/entrypoint.sh` — CUDA ライブラリパスを動的設定し uvicorn 起動
- `docker/server.py` — FastAPI サーバ（/health, /transcribe）
- `docker/requirements.txt` — 最小依存: `faster-whisper`, `fastapi`, `uvicorn[standard]`, `python-multipart`
- `test/` — サンプル音声とローカル実行スクリプト

## 再現のための注意点
- `pip install faster-whisper` だけでは CUDA 版 `ctranslate2` は保証されない
- CUDA バージョン不一致だとインストール成功しても実行時に GPU を掴まない
- 本 Dockerfile は一度 CPU 版をアンインストールし、ビルド済み CUDA 版を再インストールして整合性を取る

