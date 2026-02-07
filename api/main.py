# api.py
"""
评论过滤 API

运行方式:
    python api.py

测试:
    curl -X POST http://localhost:5000/filter -H "Content-Type: application/json" -d '{"text": "去死吧垃圾"}'
"""

import os
import logging
import time
from flask import Flask, request, jsonify
from filter_engine.engine import create_engine

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
gemini_api_key = os.getenv("GEMINI_API_KEY")

logger.info("Initializing filter engine...")
logger.info(f"GEMINI_API_KEY configured: {'yes' if gemini_api_key else 'no'}")

# 初始化过滤引擎
engine = create_engine(
        strictness="medium",
        enable_rag=False,
        enable_llm=True,
        llm_api_key=gemini_api_key
    )
logger.info("Filter engine initialized successfully.")

@app.route("/filter", methods=["POST"])
def filter_comment():
    """
    过滤单条评论

    请求体: {"text": "评论内容"}
    返回: 过滤结果
    """
    start_time = time.time()
    data = request.get_json()

    if not data or "text" not in data:
        logger.warning(f"Bad request: missing 'text' field, body={data}")
        return jsonify({"error": "缺少 text 字段"}), 400

    text = data["text"]
    comment_id = data.get("id", "demo")
    logger.info(f"[{comment_id}] Received filter request, text length={len(text)}, text={text[:80]}{'...' if len(text) > 80 else ''}")

    try:
        result = engine.filter_comment(comment_id=comment_id, text=text, enable_llm=True)
        elapsed = time.time() - start_time
        logger.info(
            f"[{comment_id}] Filter complete in {elapsed:.3f}s | "
            f"category={result.category.value} confidence={result.confidence:.2f} "
            f"action={result.action} severity={result.severity} "
            f"exempted={result.is_exempted} path={result.processing_path}"
        )
        return jsonify(result.to_dict())
    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"[{comment_id}] Filter failed after {elapsed:.3f}s: {e}", exc_info=True)
        return jsonify({"error": "Internal server error"}), 500


if __name__ == "__main__":
    logger.info("Starting comment filter API: http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=True)