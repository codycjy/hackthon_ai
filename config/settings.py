# config/settings.py
from enum import Enum
from dataclasses import dataclass
from typing import Dict, List

class FilterStrictness(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

class CommentCategory(Enum):
    # 人身及情绪安全
    THREAT = "threat"  # 死亡威胁/人肉搜索
    HATE_APPEARANCE = "hate_appearance"  # 外貌羞辱
    HATE_IDENTITY = "hate_identity"  # 身份仇恨
    DISTORTION = "distortion"  # 造谣/假粉挑拨
    TOXIC = "toxic"  # 爹味说教/脏话/恶意Emoji
    TRAFFIC_HIJACKING = "traffic_hijacking"  # 引流/竞品
    # 商业安全
    SCAM_SPAM = "scam_spam"  # 欺诈/杀猪盘
    # 智能豁免
    SAFE = "safe"  # 粉丝反击/玩笑
    UNKNOWN = "unkown" # 兜底使用

class Platform(Enum):
    INSTAGRAM = "instagram"
    TWITTER = "twitter"
    GENERAL = "general"

@dataclass
class FilterConfig:
    """过滤配置"""
    strictness: FilterStrictness = FilterStrictness.MEDIUM
    enable_rag: bool = True
    enable_llm_deep_analysis: bool = True
    enable_cyberbullying_detection: bool = False
    
    # 不同严格程度的阈值配置
    thresholds: Dict[FilterStrictness, Dict] = None

    #api_key
    gemini_api_key = None
    
    def __post_init__(self):
        self.thresholds = {
            FilterStrictness.LOW: {
                "rule_confidence": 0.9,
                "ml_confidence": 0.85,
                "rag_similarity": 0.9,
                "block_categories": [CommentCategory.THREAT, CommentCategory.SCAM_SPAM]
            },
            FilterStrictness.MEDIUM: {
                "rule_confidence": 0.7,
                "ml_confidence": 0.7,
                "rag_similarity": 0.8,
                "block_categories": [
                    CommentCategory.THREAT, CommentCategory.HATE_APPEARANCE,
                    CommentCategory.HATE_IDENTITY, CommentCategory.DISTORTION,
                    CommentCategory.SCAM_SPAM, CommentCategory.TRAFFIC_HIJACKING
                ]
            },
            FilterStrictness.HIGH: {
                "rule_confidence": 0.5,
                "ml_confidence": 0.5,
                "rag_similarity": 0.6,
                "block_categories": [cat for cat in CommentCategory if cat != CommentCategory.SAFE]
            }
        }

# 关键词库
TOXIC_KEYWORDS = {
    "zh": {
        "threat": ["去死", "杀了你", "弄死你", "人肉", "曝光你地址", "让你好看"],
        "hate_appearance": ["猪", "坦克", "骷髅", "整容脸", "脸僵", "丑", "像鬼", "肥婆", "矮子"],
        "hate_identity": ["支那", "黑鬼", "死gay", "变态", "人妖"],
        "distortion": ["被包养", "抄袭", "假货", "骗子", "小三"],
        "toxic": ["傻X", "SB", "垃圾", "废物", "智障", "脑残", "滚", "闭嘴", "恶心"],
        "traffic_hijacking": ["加我微信", "看我主页", "私聊领取", "点击链接"],
        "scam_spam": ["兼职", "日结", "投资", "比特币", "免费领取", "中奖"]
    },
    "en": {
        "threat": ["kys", "kill yourself", "die", "doxx", "find you", "your address"],
        "hate_appearance": ["fat", "ugly", "skeleton", "plastic", "hideous", "disgusting"],
        "hate_identity": ["n-word", "chink", "fag", "tranny", "retard"],
        "distortion": ["fake", "fraud", "liar", "cheater", "copycat"],
        "toxic": ["bitch", "idiot", "loser", "garbage", "trash", "stfu", "pathetic"],
        "traffic_hijacking": ["check my bio", "link in bio", "dm me", "follow back"],
        "scam_spam": ["crypto", "invest", "giveaway", "free money", "work from home"]
    }
}

# 恶意Emoji
TOXIC_EMOJIS = ["🖕", "🤮", "💩", "🤡", "🐖", "🐷", "👎", "💀", "🔫"]

# 智能豁免关键词（表示正面意图）
EXEMPTION_PATTERNS = {
    "zh": ["太好看了", "羡慕", "嫉妒你", "想成为你", "我也想", "太棒了", "爱了"],
    "en": ["slay", "queen", "dead", "im dying", "obsessed", "iconic", "living for this", "skinny legend"]
}