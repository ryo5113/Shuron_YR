import argparse
from flask import Flask, request, jsonify, session
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import glob, os

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-change-me")
EMBEDDINGS = OllamaEmbeddings(model="nomic-embed-text")

# ——— プロンプトテンプレート（回答ひな型付き） ———
DEFAULT_PROMPT = PromptTemplate(
    input_variables=["question", "context"],
    template=(
        "あなたはICFコード分析の専門家です。\n"
        "以下の【文脈】はICFコアセットと各コードの評価方法が含まれています。ICFコードと評価方法は必ずこの【文脈】に含まれるコアセットからのみ選択・判断してください。\n\n"
        "一般的な脳卒中の知識などから、新しいICFコードを追加してはいけません。\n"
        "あなたはICFコードを決定する際には、必ず【文脈】内の情報のみを根拠として使用してください。\n"
        "また、評価点を出力する際に、必ず【文脈】内の「各構成要素の評価方法（要約）」に書かれているスケールと記法のみを用いてください。\n"
        "具体的には以下のようなルールを守ってください。\n"
        "1）b/s コード：xxx.0〜xxx.4 の数値および必要に応じて .8（詳細不明）, .9（非該当） を用いること。\n"
        "2）d コード：実行状況（Performance）と能力（Capacity）の2種類の評価点を、【文脈】の例（d5101.1_ / d5101.1_2 など）と同じ形式で表記すること。\n"
        "3）e コード：阻害因子は xxx.0〜xxx.4、促進因子は xxx+0〜xxx+4 の形式のみを用いること。判断できない場合は.8（詳細不明）, .9（非該当） を用いること。\n"
        "それ以外のスケールや表現（例：「軽度」「中等度」の日本語だけ、0〜4以外の数字）は使用してはいけません。\n\n"

        "【文脈（ICFコアセット抜粋）】\n"
        "{context}\n\n"
        "【質問（対話履歴とICF状況を含む）】\n"
        "{question}\n\n"

        "あなたのタスクは次の3つです。\n"
        "1) コアセット内のICFコードについて、現在の情報から評価点を推測できるものには暫定の評価点を付ける。\n"
        "2) 評価点を付けられないコード・判断が難しいコードには「未評価区分（A/B/C など）」を付けて理由を書く。\n"
        "3) 未評価のコードのうち1つだけを選び、そのコードの評価点を決めるための質問を1つだけ提示する。\n\n"

        "評価点について:\n"
        "- 評価点はICF修飾子（例: 0〜4 程度）を想定し、現在の情報から妥当と思われる値を暫定的に付けてください。\n"
        "- 情報が不十分で数値を決められない場合は、無理に数値を付けず「未評価」としてください。\n"
        "- 未評価の理由は、以下のように分類してください。\n"
        "  A: 情報不足で評価不能（カルテや対話にほとんど情報がない）\n"
        "  B: いくつか情報はあるが、数値に落とし込むには判断が難しい\n"
        "  C: 現状の情報では明らかに該当しない可能性が高い\n\n"

        "出力フォーマットは必ず次の構造に従ってください。\n\n"

        "1. ICFコードと評価点（暫定）\n"
        "   各コードについて、以下の形式で出力してください。\n"
        "   - コード: b144\n"
        "     説明: 記憶の機能\n"
        "     評価点: 2  （0〜4 などで暫定評価。確信度が低い場合はその旨も書く）\n"
        "   - コード: d510\n"
        "     説明: 洗体\n"
        "     評価点: 未評価\n"
        "     未評価区分: A  情報不足で評価不能\n"
        "   - コード: e310\n"
        "     説明: 近親者\n"
        "     評価点: 未評価\n"
        "     未評価区分: C  現状では該当しない可能性が高い\n"
        "   のように、【文脈】に含まれる関連ICFコードを漏れなく列挙し、\n"
        "   それぞれについて「評価点」か「未評価＋未評価区分」を必ず付けてください。\n\n"

        "2. 評価根拠の説明\n"
        "   - コード: b144\n"
        "     評価点: 2\n"
        "     根拠: カルテに「長期記憶が難しい」と記載されており、日常生活に中等度の影響があると推測されるため。\n"
        "   - コード: d510\n"
        "     評価点: 未評価（A）\n"
        "     根拠: 「1人で入浴ができない」という情報はあるが、どの程度介助が必要かの詳細が不足しており、具体的な点数が決められないため。\n"
        "   のように、評価点または未評価区分ごとに理由を簡潔に説明してください。\n\n"

        "3. 次に評価点を決めるための質問（必ず1つだけ）\n"
        "   - 対象ICFコード: d510\n"
        "   - 質問: 「入浴の際、どの動作に介助が必要ですか？全身を通してほぼ全介助なのか、一部の動作のみ介助が必要なのか教えてください。」\n\n"
        "   重要:\n"
        "   - 未評価のコードが複数ある場合でも、今回の出力では、評価を進める優先度が最も高いと考えるコードを1つだけ選んでください。\n"
        "   - 質問はその1コードに対して1つだけ出力し、他のコードに対する質問は絶対に出力しないでください。\n"
        "   - 「質問がない」と判断せず、必ず1つの質問を出してください。\n"
    )
)

def load_pdfs_to_chroma(pdf_paths, db_path):
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
    docs = []
    for p in pdf_paths:
        loaded = PyPDFLoader(p).load()
        chunks = splitter.split_documents(loaded)
        print(f"Loaded {p}: {len(chunks)} chunks")
        docs.extend(chunks)

    db = Chroma.from_documents(docs, EMBEDDINGS, persist_directory=db_path)
    db.persist()
    print(f"Total chunks persisted: {len(docs)}")
    return db

# Stroke 用コアセット DB を用意
if not os.path.isdir("db_stroke"):
    print("db_stroke フォルダが存在しないため、Stroke 用データをロードします。")
    load_pdfs_to_chroma(["docs/ICF_RAG_Stroke.pdf"], db_path="db_stroke")
else:
    print("db_stroke フォルダが存在するため、Stroke 用データロードをスキップします。")

# Generic 用 DB を用意
if not os.path.isdir("db_generic"):
    print("db_generic フォルダが存在しないため、Generic 用データをロードします。")
    load_pdfs_to_chroma(["docs/ICF_RAG_Generic.pdf"], db_path="db_generic")
else:
    print("db_generic フォルダが存在するため、Generic 用データロードをスキップします。")

# RetrievalQA を組み立てる共通関数
def build_qa(db_path, prompt_template=DEFAULT_PROMPT):
    db = Chroma(persist_directory=db_path, embedding_function=EMBEDDINGS)
    retriever = db.as_retriever(search_kwargs={"k": 4})
    llm = Ollama(model="gemma3:12b")
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={
            "prompt": prompt_template,
            "document_variable_name": "context"
        }
    )

# それぞれの DB 用に QA を構築（プロセス常駐で1回だけ）
qa_stroke  = build_qa("db_stroke")
qa_generic = build_qa("db_generic")

# ===== DB選択のためのユーティリティ =====
def _classify_db(question: str) -> str:
    """
    初回のみ使用する簡易ルーティング。
    """
    q = (question or "").strip().lower()
    return "stroke" if ("脳卒中" in q or "stroke" in q) else "generic"

def _get_qa_by_key(key: str):
    """キーから該当のQAを返す。"""
    return qa_stroke if key == "stroke" else qa_generic
# ================================================

def _build_full_question(q: str, history, icf_summary: str) -> str:
    """
    対話履歴とこれまでのICFコード結果をまとめてLLMに渡すための質問文を構成する。
    RAG はあくまで ICFコード決定の根拠として使われる。
    """
    if history:
        history_text = ""
        for turn in history:
            history_text += f"ユーザー: {turn['user']}\nアシスタント: {turn['assistant']}\n"
    else:
        history_text = "（これまでの対話はありません）"

    icf_text = icf_summary if icf_summary else "（これまでに確定したICFコードはまだありません）"

    full_question = (
        "これまでの対話とICFコード情報を踏まえて、ICFコード候補とその根拠を更新してください。\n\n"
        "【これまでの対話履歴】\n"
        f"{history_text}\n\n"
        "【これまでに推定されているICFコードの要約】\n"
        f"{icf_text}\n\n"
        "【今回の新しいユーザー入力】\n"
        f"{q}\n"
    )
    return full_question

@app.route("/ask", methods=["POST"])
def ask_api():
    """
    セッションごとに active_db / 対話履歴 / ICFコード結果 を保持する。
    JSON で {\"reset\": true} が来たら当該セッションの状態をリセット。
    """
    payload = request.get_json(silent=True) or {}
    q = payload.get("question", "")

    # リセット要求があれば全状態をクリア
    if payload.get("reset"):
        for key in ("active_db", "history", "icf_summary"):
            session.pop(key, None)

    # DB選択（初回のみ分類して固定）
    active_db = session.get("active_db")
    if active_db is None:
        active_db = _classify_db(q)
        session["active_db"] = active_db
        print(f"[API] 初回質問のため DB を '{active_db}' に固定しました。")

    qa = _get_qa_by_key(active_db)

    # 対話履歴とICFコード結果を取得
    history = session.get("history", [])
    icf_summary = session.get("icf_summary", "")

    # LLM に渡す最終質問文を構成
    full_question = _build_full_question(q, history, icf_summary)

    # 回答生成（RAGはICFコード決定のための文脈として使用）
    answer = qa.run(full_question)

    # 対話履歴とICFコード結果を更新
    history.append({"user": q, "assistant": answer})
    session["history"] = history
    session["icf_summary"] = answer  # 毎回答ごとにICFコード結果を更新

    return jsonify({
        "answer": answer,
        "active_db": active_db
    })

def run_cli():
    """
    CLI はプロセス内の変数でDB・対話履歴・ICFコード結果を固定。
    /reset と入力した時のみ解除（再度初回判定へ）。
    """
    print("── VSCode 対話モード ──\n(exit または Ctrl+C で終了, /reset で状態をリセット)\n")
    active_db = None
    history = []
    icf_summary = ""

    try:
        while True:
            q = input("分析内容: ").strip()
            if q.lower() in ("exit", "quit"):
                break
            if q == "/reset":
                active_db = None
                history = []
                icf_summary = ""
                print("DB選択と対話履歴・ICFコード結果をリセットしました。次の質問で再選択します。\n")
                continue

            if active_db is None:
                active_db = _classify_db(q)
                print(f"[CLI] 初回質問のため DB を '{active_db}' に固定しました。\n")

            qa = _get_qa_by_key(active_db)

            full_question = _build_full_question(q, history, icf_summary)
            who = "Stroke" if active_db == "stroke" else "Generic"

            answer = qa.run(full_question)
            print(f"回答 ({who}):\n", answer, "\n")

            # 対話履歴とICFコード結果を更新
            history.append({"user": q, "assistant": answer})
            icf_summary = answer

    except (EOFError, KeyboardInterrupt):
        pass
    print("対話モードを終了しました。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cli", action="store_true", help="CLI 対話モードで起動")
    parser.add_argument("--port", type=int, default=5000, help="Flask ポート番号")
    args = parser.parse_args()

    if args.cli:
        run_cli()
    else:
        print(f"Flask サーバーをポート {args.port} で起動します…")
        app.run(port=args.port)
