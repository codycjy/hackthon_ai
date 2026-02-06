# test_api.py
"""
API 测试脚本

运行方式:
    1. 先启动 API: python api.py
    2. 运行测试: python test_api.py
"""

import requests

API_URL = "http://localhost:5000/filter"


def test_filter(text):
    """测试单条评论"""
    response = requests.post(API_URL, json={"text": text})
    result = response.json()
    print(f"文本: {text}")
    print(f"分类: {result['category']}")
    print(f"动作: {result['action']}")
    print(f"置信度: {result['confidence']:.2%}")
    print(f"原因: {result['reason']}")
    print("-" * 50)
    return result


if __name__ == "__main__":
    test_cases = [
        "去死吧垃圾",
        "今天天气真好",
        "太美了我恨你怎么这么好看",
        "加我微信领红包",
        "You're such an idiot",
        "这个视频太棒了！",
    ]
    
    print("=" * 50)
    print("评论过滤 API 测试")
    print("=" * 50 + "\n")
    
    for text in test_cases:
        test_filter(text)