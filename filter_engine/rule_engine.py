# filter_engine/rule_engine.py
import logging
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
from config.settings import CommentCategory, FilterStrictness, TOXIC_KEYWORDS

logger = logging.getLogger(__name__)

@dataclass
class RuleResult:
    """规则匹配结果"""
    matched: bool
    category: Optional[CommentCategory]
    confidence: float
    matched_rules: list
    reason: str

class RuleEngine:
    """基于规则的快速过滤引擎"""
    
    def __init__(self, strictness: FilterStrictness = FilterStrictness.MEDIUM):
        self.strictness = strictness
        self.rules = self._init_rules()
    
    def _init_rules(self) -> list:
        """初始化规则列表"""
        return [
            # 威胁类规则（最高优先级）
            {
                "name": "death_threat",
                "category": CommentCategory.THREAT,
                "patterns": ["去死", "杀了你", "kys", "kill yourself", "弄死你"],
                "confidence": 0.95,
                "priority": 1
            },
            {
                "name": "doxxing",
                "category": CommentCategory.THREAT,
                "patterns": ["人肉", "曝光地址", "doxx", "your address", "找到你"],
                "confidence": 0.9,
                "priority": 1
            },
            # 诈骗类规则
            {
                "name": "crypto_scam",
                "category": CommentCategory.SCAM_SPAM,
                "patterns": ["比特币", "crypto", "投资回报", "guaranteed returns", "日结"],
                "confidence": 0.85,
                "priority": 2
            },
            {
                "name": "fake_collab",
                "category": CommentCategory.SCAM_SPAM,
                "patterns": ["dm for collab", "ambassador needed", "品牌直招"],
                "confidence": 0.8,
                "priority": 2
            },
            # 引流类规则
            {
                "name": "traffic_hijack",
                "category": CommentCategory.TRAFFIC_HIJACKING,
                "patterns": ["加我微信", "check my bio", "link in bio", "看我主页"],
                "confidence": 0.85,
                "priority": 3
            },
            # 恶意emoji规则
            {
                "name": "toxic_emoji_spam",
                "category": CommentCategory.TOXIC,
                "emoji_patterns": ["🖕", "🤮🤮🤮", "💩💩💩", "🤡🤡🤡"],
                "confidence": 0.75,
                "priority": 4
            },
            # 外貌攻击规则
            {
                "name": "body_shaming",
                "category": CommentCategory.HATE_APPEARANCE,
                "patterns": ["猪", "坦克", "整容脸", "plastic surgery", "太丑", "ugly"],
                "confidence": 0.7,
                "priority": 4
            }
        ]
    
    def apply_rules(self, text: str, features: Dict) -> RuleResult:
        """应用规则进行匹配"""
        text_lower = text.lower()
        matched_rules = []
        highest_confidence = 0.0
        matched_category = None

        for rule in self.rules:
            is_match = False

            # 文本模式匹配
            if "patterns" in rule:
                for pattern in rule["patterns"]:
                    if pattern.lower() in text_lower:
                        is_match = True
                        break

            # Emoji模式匹配
            if "emoji_patterns" in rule:
                for pattern in rule["emoji_patterns"]:
                    if pattern in text:
                        is_match = True
                        break

            # 特征条件匹配
            if "feature_conditions" in rule:
                conditions_met = all(
                    features.get(k) == v
                    for k, v in rule["feature_conditions"].items()
                )
                if conditions_met:
                    is_match = True

            if is_match:
                matched_rules.append(rule["name"])
                logger.debug(f"Rule matched: {rule['name']} -> {rule['category'].value} (confidence={rule['confidence']})")
                if rule["confidence"] > highest_confidence:
                    highest_confidence = rule["confidence"]
                    matched_category = rule["category"]

        # 检查豁免条件
        if features.get("exemption_matches") and matched_category not in [
            CommentCategory.THREAT, CommentCategory.SCAM_SPAM
        ]:
            original_confidence = highest_confidence
            highest_confidence *= 0.5
            logger.debug(f"Rule exemption applied: confidence {original_confidence:.2f} -> {highest_confidence:.2f}")

        cat_value = matched_category.value if matched_category else "none"
        logger.info(f"Rule engine: matched={len(matched_rules) > 0} rules={matched_rules} category={cat_value} confidence={highest_confidence:.2f}")

        return RuleResult(
            matched=len(matched_rules) > 0,
            category=matched_category,
            confidence=highest_confidence,
            matched_rules=matched_rules,
            reason=f"Matched rules: {matched_rules}" if matched_rules else "No rule matched"
        )