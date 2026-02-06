
from typing import Dict, List, Any
from config.settings import TOXIC_EMOJIS, TOXIC_KEYWORDS, EXEMPTION_PATTERNS

class FeatureExtractor:
    """特征提取器"""
    
    def __init__(self):
        self.toxic_emojis = set(TOXIC_EMOJIS)
    
    def extract_features(self, text: str, metadata: Dict) -> Dict[str, Any]:
        """提取所有特征"""
        features = {
            # 基础特征
            "text_length": len(text),
            "word_count": len(text.split()),
            
            # Emoji特征
            "emoji_count": len(metadata.get("emojis", [])),
            "toxic_emoji_count": self._count_toxic_emojis(metadata.get("emojis", [])),
            "toxic_emojis": self._get_toxic_emojis(metadata.get("emojis", [])),
            
            # 社交特征
            "mention_count": len(metadata.get("mentions", [])),
            "hashtag_count": len(metadata.get("hashtags", [])),
            "has_url": len(metadata.get("urls", [])) > 0,
            
            # 关键词特征
            "keyword_matches": self._match_keywords(text),
            "exemption_matches": self._match_exemptions(text),
            
            # 模式特征
            "has_caps_abuse": self._detect_caps_abuse(metadata.get("original", text)),
            "repetition_score": self._calculate_repetition(text),
            "is_mention_bomb": len(metadata.get("mentions", [])) > 3
        }
        
        return features
    
    def _count_toxic_emojis(self, emojis: List[str]) -> int:
        """统计恶意emoji数量"""
        count = 0
        for emoji_str in emojis:
            for char in emoji_str:
                if char in self.toxic_emojis:
                    count += 1
        return count
    
    def _get_toxic_emojis(self, emojis: List[str]) -> List[str]:
        """获取恶意emoji列表"""
        toxic = []
        for emoji_str in emojis:
            for char in emoji_str:
                if char in self.toxic_emojis:
                    toxic.append(char)
        return toxic
    
    def _match_keywords(self, text: str) -> Dict[str, List[str]]:
        """匹配关键词"""
        matches = {}
        text_lower = text.lower()
        
        for lang in ["zh", "en"]:
            for category, keywords in TOXIC_KEYWORDS.get(lang, {}).items():
                for keyword in keywords:
                    if keyword.lower() in text_lower:
                        if category not in matches:
                            matches[category] = []
                        matches[category].append(keyword)
        
        return matches
    
    def _match_exemptions(self, text: str) -> List[str]:
        """匹配豁免模式"""
        matches = []
        text_lower = text.lower()
        
        for lang in ["zh", "en"]:
            for pattern in EXEMPTION_PATTERNS.get(lang, []):
                if pattern.lower() in text_lower:
                    matches.append(pattern)
        
        return matches
    
    def _detect_caps_abuse(self, text: str) -> bool:
        """检测大写字母滥用"""
        if len(text) < 10:
            return False
        upper_count = sum(1 for c in text if c.isupper())
        return upper_count / len(text) > 0.5
    
    def _calculate_repetition(self, text: str) -> float:
        """计算重复度（检测刷屏）"""
        words = text.lower().split()
        if len(words) < 2:
            return 0.0
        unique_words = set(words)
        return 1 - (len(unique_words) / len(words))