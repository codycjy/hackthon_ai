# api.py
"""
评论过滤 API

运行方式:
    python api.py

测试:
    curl -X POST http://localhost:5000/filter -H "Content-Type: application/json" -d '{"text": "去死吧垃圾"}'
"""

from flask import Flask, request, jsonify
from filter_engine.engine import create_engine

app = Flask(__name__)
kimi_api_key = "sk-gF7hiU9IuKttaT2Q77YovdVUaXwENXNK3cNtWVneIrRjDIqU"

# 初始化过滤引擎
engine = create_engine(
        strictness="medium",
        enable_rag=False,
        enable_llm=True,
        llm_api_key=kimi_api_key
    )

@app.route("/filter", methods=["POST"])
def filter_comment():
    """
    过滤单条评论
    
    请求体: {"text": "评论内容"}
    返回: 过滤结果
    """
    data = request.get_json()
    
    if not data or "text" not in data:
        return jsonify({"error": "缺少 text 字段"}), 400
    
    text = data["text"]
    comment_id = data.get("id", "demo")
    
    result = engine.filter_comment(comment_id=comment_id, text=text, enable_llm=True)
    print(result)
    
    return jsonify(result.to_dict())


if __name__ == "__main__":
    print("启动评论过滤 API: http://localhost:5000")
    print("测试: curl -X POST http://localhost:5000/filter -H 'Content-Type: application/json' -d '{\"text\": \"去死吧垃圾\"}'")
    app.run(host="0.0.0.0", port=5000, debug=True)