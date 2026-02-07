# preprocessor/text_cleaner.py
import re
import logging
from typing import List, Dict, Tuple
import unicodedata

logger = logging.getLogger(__name__)

class TextCleaner:
    
    def __init__(self):
        # 变体字符映射（用于对抗规避检测）
        self.char_variants = {
            'a': ['@', '4', 'α', 'а'],
            'e': ['3', 'ε', 'е'],
            'i': ['1', '!', 'і', 'l'],
            'o': ['0', 'ο', 'о'],
            's': ['$', '5', 'ѕ'],
            'b': ['8', 'в'],
            't': ['7', '+'],
        }
        self.reverse_variants = {}
        for char, variants in self.char_variants.items():
            for v in variants:
                self.reverse_variants[v] = char
    
    def clean(self, text: str) -> str:
        # 转小写
        text = text.lower()
        # Unicode标准化
        text = unicodedata.normalize('NFKC', text)
        # 去除多余空格
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def normalize_variants(self, text: str) -> str:
        """标准化变体字符（对抗l33t speak等）"""
        result = []
        for char in text:
            result.append(self.reverse_variants.get(char, char))
        return ''.join(result)
    
    def extract_clean_text(self, text: str) -> Tuple[str, Dict]:
        """提取并返回清洗后的文本和提取的元数据"""
        logger.debug(f"Cleaning text: {text[:80]}{'...' if len(text) > 80 else ''}")

        metadata = {
            "original": text,
            "mentions": [],
            "hashtags": [],
            "urls": [],
            "emojis": []
        }

        # 提取@提及
        metadata["mentions"] = re.findall(r'@[\w]+', text)

        # 提取hashtags
        metadata["hashtags"] = re.findall(r'#[\w]+', text)

        # 提取URLs
        metadata["urls"] = re.findall(r'https?://\S+', text)

        # 提取emojis
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F1E0-\U0001F1FF"  # flags
            "\U00002702-\U000027B0"
            "\U0001f926-\U0001f937"
            "\U00010000-\U0010ffff"
            "]+",
            flags=re.UNICODE
        )
        metadata["emojis"] = emoji_pattern.findall(text)

        # 清洗文本
        cleaned = self.clean(text)
        cleaned = self.normalize_variants(cleaned)

        logger.debug(
            f"Text cleaned: mentions={len(metadata['mentions'])} hashtags={len(metadata['hashtags'])} "
            f"urls={len(metadata['urls'])} emojis={len(metadata['emojis'])}"
        )

        return cleaned, metadata


