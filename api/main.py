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
import random
from flask import Flask, request, jsonify
from filter_engine.engine import create_engine
from comment_generator import NegativeCommentGenerator
from config.settings import Platform, CommentCategory

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

# 初始化评论生成器
comment_generator = NegativeCommentGenerator()
logger.info("Comment generator initialized.")

# 随机用户名池
TROLL_NAMES = [
    ("DarkShadow99", "@darkshadow99"), ("ToxicAvenger", "@toxic_avenger"),
    ("RageBot", "@ragebot_x"), ("HateWatcher", "@hatewatcher"),
    ("TrollKing", "@trollking420"), ("AngryCritic", "@angrycritic"),
    ("BullyMaster", "@bullymaster"), ("VenomUser", "@venomuser_"),
    ("NightmareX", "@nightmare_x"), ("黑粉001", "@heifen001"),
    ("键盘侠", "@keyboard_warrior"), ("SpamLord", "@spamlord9000"),
    ("FakeNews24", "@fakenews24"), ("CancelCulture", "@cancel_them"),
    ("RatioKing", "@ratio_king"),
]
NORMAL_NAMES = [
    ("Luna Park", "@lunapark"), ("Sam Carter", "@samcarter_"),
    ("Mia Zhang", "@miazhang"), ("Chris Dev", "@chrisdev"),
    ("Taylor Kim", "@taylorkim"), ("Jordan Lee", "@jordanlee"),
    ("Riley Cooper", "@rileycooper"), ("Morgan Fisher", "@morganfisher"),
    ("小明同学", "@xiaoming_"), ("快乐星球", "@happystar"),
    ("Avery James", "@averyjames"), ("Quinn Harper", "@quinnharper"),
]
_gen_id = 0

@app.route("/health")
def health():
    return jsonify({"status": "ok"})


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


@app.route("/generate", methods=["POST"])
def generate_comments():
    """
    生成模拟评论

    请求体:
      count    - 生成数量 (默认 1)
      mode     - "normal" | "attack" (默认 "normal")
      language - "zh" | "en" | null (随机)

    返回: { comments: [...] }
    """
    global _gen_id
    start_time = time.time()
    data = request.get_json() or {}
    count = min(data.get("count", 1), 20)  # 上限 20
    mode = data.get("mode", "normal")
    language = data.get("language", None)

    toxic_ratio = 0.75 if mode == "attack" else 0.4
    logger.info(
        f"[generate] Received request: count={count} mode={mode} "
        f"language={language or 'random'} toxic_ratio={toxic_ratio}"
    )

    comments = []
    category_counts = {}
    toxic_count = 0

    try:
        for _ in range(count):
            is_toxic = random.random() < toxic_ratio
            _gen_id += 1

            if is_toxic:
                mc = comment_generator.generate_single(
                    platform=Platform.TWITTER,
                    language=language,
                )
                name, handle = random.choice(TROLL_NAMES)
                toxic_count += 1
            else:
                mc = comment_generator.generate_single(
                    category=CommentCategory.SAFE,
                    platform=Platform.TWITTER,
                    language=language,
                )
                name, handle = random.choice(NORMAL_NAMES)

            cat = mc.category.value
            category_counts[cat] = category_counts.get(cat, 0) + 1

            comments.append({
                "id": f"gen_{_gen_id}",
                "author": name,
                "handle": handle,
                "avatar": f"https://api.dicebear.com/7.x/avataaars/svg?seed={handle}",
                "text": mc.text,
                "category": cat,
                "language": mc.language,
                "timestamp": "just now",
                "likes": random.randint(0, 100) if not is_toxic else random.randint(0, 5),
                "retweets": random.randint(0, 30) if not is_toxic else random.randint(0, 2),
                "replies": random.randint(0, 15),
            })

        elapsed = time.time() - start_time
        cat_summary = " ".join(f"{k}={v}" for k, v in sorted(category_counts.items()))
        logger.info(
            f"[generate] Complete in {elapsed:.3f}s | "
            f"count={count} toxic={toxic_count} safe={count - toxic_count} | "
            f"categories: {cat_summary}"
        )
        return jsonify({"comments": comments})

    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"[generate] Failed after {elapsed:.3f}s: {e}", exc_info=True)
        return jsonify({"error": "Internal server error"}), 500


if __name__ == "__main__":
    logger.info("Starting comment filter API: http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=True)