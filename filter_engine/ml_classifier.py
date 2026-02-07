# filter_engine/ml_classifier.py
from typing import Dict, List, Optional
from dataclasses import dataclass
import logging
import time
import numpy as np
from config.settings import CommentCategory

from detoxify import Detoxify

logger = logging.getLogger(__name__)

@dataclass
class ClassificationResult:
    """分类结果"""
    category: CommentCategory
    confidence: float
    all_scores: Dict[CommentCategory, float]
    detoxify_raw: Dict[str, float]  # 原始Detoxify输出


class LightweightClassifier:
    """
    基于Detoxify的ML分类器

    
    输出标签:
    - toxicity: 总体毒性
    - severe_toxicity: 严重毒性
    - obscene: 淫秽
    - threat: 威胁
    - insult: 侮辱
    - identity_attack: 身份攻击
    - sexual_explicit: 色情内容
    """
    
    def __init__(self, model_type: str = 'multilingual', device: str = 'cpu'):
        """
        初始化Detoxify模型
        
        Args:
            model_type: 模型类型 ('original', 'unbiased', 'multilingual')
            device: 运行设备 ('cpu', 'cuda')
        """
        self.model_type = model_type
        self.device = device
        
        # 加载Detoxify模型
        logger.info(f"Loading Detoxify model: {model_type} on {device}...")
        load_start = time.time()
        self.model = Detoxify(model_type, device=device)
        logger.info(f"Detoxify model loaded in {time.time() - load_start:.2f}s")
        
        # Detoxify标签到自定义类别的映射
        self.label_mapping = {
            'threat': CommentCategory.THREAT,
            'identity_attack': CommentCategory.HATE_IDENTITY,
            'insult': CommentCategory.HATE_APPEARANCE,  # 侮辱映射到外貌攻击（可调整）
            'severe_toxicity': CommentCategory.TOXIC,
            'toxicity': CommentCategory.TOXIC,
            'obscene': CommentCategory.TOXIC,
        }
        
        # 各类别的阈值配置
        self.thresholds = {
            'threat': 0.5,
            'identity_attack': 0.5,
            'severe_toxicity': 0.7,
            'insult': 0.6,
            'toxicity': 0.7,
            'obscene': 0.7,
        }
    
    def classify(self, text: str, features: Dict) -> ClassificationResult:
        """
        使用Detoxify分类评论

        Args:
            text: 待分类文本
            features: 预处理提取的特征（用于辅助判断）

        Returns:
            ClassificationResult: 分类结果
        """
        start_time = time.time()
        logger.debug(f"ML classify: text={text[:60]}{'...' if len(text) > 60 else ''}")

        # 获取Detoxify预测结果
        detoxify_results = self.model.predict(text)
        predict_elapsed = time.time() - start_time
        logger.debug(f"Detoxify predict in {predict_elapsed:.3f}s | toxicity={detoxify_results.get('toxicity', 0):.4f} severe={detoxify_results.get('severe_toxicity', 0):.4f} threat={detoxify_results.get('threat', 0):.4f} insult={detoxify_results.get('insult', 0):.4f} identity_attack={detoxify_results.get('identity_attack', 0):.4f}")

        # 转换为自定义类别得分
        category_scores = self._convert_to_category_scores(detoxify_results, features)

        # 找到最高分类别
        best_category = max(category_scores, key=category_scores.get)
        best_score = category_scores[best_category]

        # 计算归一化置信度
        total_score = sum(category_scores.values())
        if total_score > 0:
            confidence = best_score / total_score
        else:
            confidence = 0.0

        # 应用豁免逻辑降低置信度
        if features.get("exemption_matches"):
            if best_category != CommentCategory.THREAT:
                original_confidence = confidence
                confidence *= 0.6
                logger.debug(f"Exemption applied: confidence {original_confidence:.2f} -> {confidence:.2f}")
                # 如果有豁免匹配且置信度降低后较低，考虑判为SAFE
                if confidence < 0.4:
                    logger.debug(f"Low confidence after exemption, overriding category {best_category.value} -> safe")
                    best_category = CommentCategory.SAFE
                    best_score = 1 - detoxify_results.get('toxicity', 0)

        final_confidence = min(confidence * 1.3, 0.99)
        elapsed = time.time() - start_time
        logger.info(f"ML classify done in {elapsed:.3f}s | category={best_category.value} confidence={final_confidence:.2f}")

        return ClassificationResult(
            category=best_category,
            confidence=final_confidence,
            all_scores=category_scores,
            detoxify_raw=detoxify_results
        )
    
    def _convert_to_category_scores(
        self, 
        detoxify_results: Dict[str, float],
        features: Dict
    ) -> Dict[CommentCategory, float]:
        """
        将Detoxify输出转换为自定义类别得分
        
        Args:
            detoxify_results: Detoxify原始输出
            features: 预处理特征
        
        Returns:
            各类别的得分字典
        """
        scores = {cat: 0.0 for cat in CommentCategory}
        
        # 1. 威胁类 (THREAT)
        threat_score = detoxify_results.get('threat', 0)
        if threat_score > self.thresholds['threat']:
            scores[CommentCategory.THREAT] = threat_score
        
        # 2. 身份攻击 (HATE_IDENTITY)
        identity_score = detoxify_results.get('identity_attack', 0)
        if identity_score > self.thresholds['identity_attack']:
            scores[CommentCategory.HATE_IDENTITY] = identity_score
        
        # 3. 外貌攻击 (HATE_APPEARANCE) - 结合insult和关键词
        insult_score = detoxify_results.get('insult', 0)
        keyword_matches = features.get("keyword_matches", {})
        if "hate_appearance" in keyword_matches:
            scores[CommentCategory.HATE_APPEARANCE] = max(insult_score, 0.7)
        elif insult_score > self.thresholds['insult']:
            scores[CommentCategory.HATE_APPEARANCE] = insult_score * 0.8
        
        # 4. 造谣 (DISTORTION) - Detoxify不直接支持，依赖关键词
        if "distortion" in keyword_matches:
            scores[CommentCategory.DISTORTION] = 0.7 + len(keyword_matches.get("distortion", [])) * 0.1
        
        # 5. 恶意评论 (TOXIC) - 综合toxicity和severe_toxicity
        toxicity = detoxify_results.get('toxicity', 0)
        severe_toxicity = detoxify_results.get('severe_toxicity', 0)
        obscene = detoxify_results.get('obscene', 0)
        
        toxic_score = max(toxicity, severe_toxicity, obscene)
        if toxic_score > self.thresholds['toxicity']:
            # 避免与其他更具体的类别重复计分
            if scores[CommentCategory.THREAT] < 0.5 and scores[CommentCategory.HATE_IDENTITY] < 0.5:
                scores[CommentCategory.TOXIC] = toxic_score
        
        # 6. 引流 (TRAFFIC_HIJACKING) - Detoxify不支持，完全依赖关键词
        if "traffic_hijacking" in keyword_matches:
            scores[CommentCategory.TRAFFIC_HIJACKING] = 0.75 + len(keyword_matches.get("traffic_hijacking", [])) * 0.1
        if features.get("has_url") and features.get("mention_count", 0) > 0:
            scores[CommentCategory.TRAFFIC_HIJACKING] = max(
                scores[CommentCategory.TRAFFIC_HIJACKING], 
                0.5
            )
        
        # 7. 诈骗 (SCAM_SPAM) - Detoxify不支持，依赖关键词
        if "scam_spam" in keyword_matches:
            scores[CommentCategory.SCAM_SPAM] = 0.8 + len(keyword_matches.get("scam_spam", [])) * 0.1
        
        # 8. 安全/豁免 (SAFE)
        exemption_matches = features.get("exemption_matches", [])
        if exemption_matches:
            # 有豁免模式匹配时，提高SAFE得分
            safe_boost = len(exemption_matches) * 0.3
            non_toxic_score = 1 - toxicity
            scores[CommentCategory.SAFE] = min(non_toxic_score + safe_boost, 1.0)
        else:
            # 低毒性文本
            if toxicity < 0.3:
                scores[CommentCategory.SAFE] = 1 - toxicity
        
        # 恶意emoji加成
        toxic_emoji_count = features.get("toxic_emoji_count", 0)
        if toxic_emoji_count > 0:
            scores[CommentCategory.TOXIC] = max(
                scores[CommentCategory.TOXIC],
                0.5 + toxic_emoji_count * 0.15
            )
        
        return scores
    
    def predict_batch(self, texts: List[str], features_list: List[Dict]) -> List[ClassificationResult]:
        """
        批量分类
        
        Args:
            texts: 文本列表
            features_list: 对应的特征列表
        
        Returns:
            分类结果列表
        """
        # Detoxify支持批量预测
        detoxify_results_batch = self.model.predict(texts)
        
        results = []
        for i, text in enumerate(texts):
            # 将批量结果转换为单条格式
            single_result = {
                key: values[i] if isinstance(values, list) else values
                for key, values in detoxify_results_batch.items()
            }
            
            features = features_list[i] if i < len(features_list) else {}
            category_scores = self._convert_to_category_scores(single_result, features)
            
            best_category = max(category_scores, key=category_scores.get)
            best_score = category_scores[best_category]
            total_score = sum(category_scores.values())
            confidence = best_score / total_score if total_score > 0 else 0.0
            
            if features.get("exemption_matches") and best_category != CommentCategory.THREAT:
                confidence *= 0.6
                if confidence < 0.4:
                    best_category = CommentCategory.SAFE
            
            results.append(ClassificationResult(
                category=best_category,
                confidence=min(confidence * 1.3, 0.99),
                all_scores=category_scores,
                detoxify_raw=single_result
            ))
        
        return results
    
    def get_toxicity_scores(self, text: str) -> Dict[str, float]:
        """
        获取原始Detoxify毒性分数（用于调试）
        
        Args:
            text: 输入文本
        
        Returns:
            Detoxify各标签的分数
        """
        return self.model.predict(text)


# 辅助函数：创建带有GPU支持的分类器
def create_classifier(use_gpu: bool = False, model_type: str = 'multilingual') -> LightweightClassifier:
    """
    创建分类器实例
    
    Args:
        use_gpu: 是否使用GPU
        model_type: 模型类型
    
    Returns:
        配置好的分类器
    """
    device = 'cuda' if use_gpu else 'cpu'
    return LightweightClassifier(model_type=model_type, device=device)


# 使用示例
if __name__ == "__main__":
    # 创建分类器
    classifier = LightweightClassifier(model_type='multilingual')
    
    # 测试用例
    test_cases = [
        {
            "text": "去死吧你这个垃圾",
            "features": {"keyword_matches": {"threat": ["去死"], "toxic": ["垃圾"]}, "exemption_matches": [], "toxic_emoji_count": 0, "has_url": False, "mention_count": 0}
        },
        {
            "text": "You should kill yourself, nobody likes you",
            "features": {"keyword_matches": {"threat": ["kill yourself"]}, "exemption_matches": [], "toxic_emoji_count": 0, "has_url": False, "mention_count": 0}
        },
        {
            "text": "太美了吧！我恨你怎么这么好看",
            "features": {"keyword_matches": {}, "exemption_matches": ["太美了", "好看"], "toxic_emoji_count": 0, "has_url": False, "mention_count": 0}
        },
        {
            "text": "OMG slay queen! You look absolutely stunning 💕",
            "features": {"keyword_matches": {}, "exemption_matches": ["slay", "queen"], "toxic_emoji_count": 0, "has_url": False, "mention_count": 0}
        },
        {
            "text": "这身材也好意思发？像头猪一样",
            "features": {"keyword_matches": {"hate_appearance": ["猪"]}, "exemption_matches": [], "toxic_emoji_count": 0, "has_url": False, "mention_count": 0}
        },
        {
            "text": "投资比特币稳赚不赔，加我微信了解",
            "features": {"keyword_matches": {"scam_spam": ["比特币", "加我微信"]}, "exemption_matches": [], "toxic_emoji_count": 0, "has_url": False, "mention_count": 0}
        },
        {
            "text": "💩💩💩 垃圾内容 🤮🤮🤮",
            "features": {"keyword_matches": {"toxic": ["垃圾"]}, "exemption_matches": [], "toxic_emoji_count": 6, "has_url": False, "mention_count": 0}
        },
    ]
    
    print("=" * 70)
    print("Detoxify ML分类器测试")
    print("=" * 70)
    
    for i, case in enumerate(test_cases, 1):
        result = classifier.classify(case["text"], case["features"])
        print(f"\n{i}. 文本: {case['text'][:50]}...")
        print(f"   分类结果: {result.category.value}")
        print(f"   置信度: {result.confidence:.2%}")
        print(f"   Detoxify原始分数: ", end="")
        formatted_raw = {k: f"{v:.4f}" for k, v in result.detoxify_raw.items()}
        print(formatted_raw)
        # 只显示分数>0.1的标签，保留3位小数
        for label, score in result.detoxify_raw.items():
            if score > 0.1:
                print(f"      → {label}: {score:.3f}")