# filter_engine/engine.py
"""
评论过滤主引擎

整合规则引擎、ML分类器、RAG检索和LLM深度分析
提供统一的评论过滤接口

用法:
    from filter_engine.engine import CommentFilterEngine
    
    engine = CommentFilterEngine()
    result = engine.filter_comment("comment_id", "评论文本")
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import logging
import json
import os
import time

from config.settings import CommentCategory, FilterConfig, FilterStrictness
from preprocessor.text_cleaner import TextCleaner
from preprocessor.feature_extractor import FeatureExtractor
from filter_engine.ml_classifier import LightweightClassifier, ClassificationResult
from filter_engine.rag_retriever import RAGRetriever, RAGResult
from filter_engine.llm_analyzer import LLMAnalyzer, LLMAnalysisResult
from filter_engine.rule_engine import RuleEngine, RuleResult

logger = logging.getLogger(__name__)


@dataclass
class FilterResult:
    """过滤结果"""
    comment_id: str
    original_text: str
    category: CommentCategory
    confidence: float
    action: str  # "keep", "delete", "review"
    reason: str
    severity: str  # "low", "medium", "high", "critical"
    is_exempted: bool
    processing_path: str
    all_scores: Dict[str, float] = field(default_factory=dict)
    detoxify_raw: Optional[Dict[str, float]] = None
    similar_cases: Optional[List[Dict]] = None
    llm_analysis: Optional[Dict] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典（确保所有值都是 JSON 可序列化的）"""
        
        def convert_value(v):
            """递归转换值为 JSON 可序列化类型"""
            if v is None:
                return None
            elif isinstance(v, (int, str, bool)):
                return v
            elif isinstance(v, float):
                return float(v)
            elif hasattr(v, 'item'):  # numpy 标量
                return v.item()
            elif hasattr(v, 'value'):  # 枚举
                return v.value
            elif isinstance(v, dict):
                return {str(k): convert_value(val) for k, val in v.items()}
            elif isinstance(v, (list, tuple)):
                return [convert_value(item) for item in v]
            else:
                return str(v)
        
        return {
            "comment_id": self.comment_id,
            "original_text": self.original_text,
            "category": self.category.value if hasattr(self.category, "value") else str(self.category),
            "confidence": float(self.confidence),
            "action": self.action,
            "reason": self.reason,
            "severity": self.severity,
            "is_exempted": self.is_exempted,
            "processing_path": self.processing_path,
            "all_scores": convert_value(self.all_scores),
            "detoxify_raw": convert_value(self.detoxify_raw),
            "similar_cases": convert_value(self.similar_cases),
            "llm_analysis": convert_value(self.llm_analysis),
        }


class CommentFilterEngine:
    """
    评论过滤主引擎
    
    整合多层过滤能力：
    1. 规则引擎：快速匹配高置信度威胁/诈骗
    2. ML分类器：Detoxify多标签毒性检测
    3. RAG检索：基于相似案例的分类建议
    4. LLM分析：深度语义理解（可选）
    """
    
    def __init__(self, config: Optional[FilterConfig] = None):
        """
        初始化过滤引擎
        
        Args:
            config: 过滤配置，为 None 时使用默认配置
        """
        self.config = config or FilterConfig()
        
        logger.info("Initializing Comment Filter Engine...")
        
        # 初始化预处理组件
        self.text_cleaner = TextCleaner()
        self.feature_extractor = FeatureExtractor()
        
        # 初始化规则引擎
        self.rule_engine = RuleEngine(self.config.strictness)
        
        # 初始化 ML 分类器 (Detoxify)
        logger.info("Loading Detoxify classifier...")
        self.classifier = LightweightClassifier(
            model_type=getattr(self.config, 'detoxify_model', 'multilingual'),
            device=getattr(self.config, 'detoxify_device', 'cpu')
        )
        
        # RAG 和 LLM 懒加载标记
        self._rag_retriever: Optional[RAGRetriever] = None
        self._rag_init_attempted: bool = False
        self._llm_analyzer: Optional[LLMAnalyzer] = None
        self._llm_init_attempted: bool = False
        
        logger.info("Filter engine initialized successfully.")
    
    def _get_rag_retriever(self, force_init: bool = False) -> Optional[RAGRetriever]:
        """
        获取 RAG 检索器（懒加载）
        
        Args:
            force_init: 是否强制尝试初始化（即使之前失败过）
        
        Returns:
            RAGRetriever 实例或 None
        """
        # 如果已经初始化成功，直接返回
        if self._rag_retriever is not None:
            return self._rag_retriever
        
        # 如果之前尝试过初始化但失败了，且不强制重试，返回 None
        if self._rag_init_attempted and not force_init:
            return None
        
        # 尝试初始化
        self._rag_init_attempted = True
        logger.info("Loading RAG retriever (BGE-M3 + FAISS)...")
        try:
            embedding_model = getattr(self.config, 'embedding_model', 'BAAI/bge-m3')
            self._rag_retriever = RAGRetriever(model_name=embedding_model)
            logger.info("RAG retriever loaded successfully.")
        except Exception as e:
            logger.warning(f"Failed to load RAG retriever: {e}")
            self._rag_retriever = None
        
        return self._rag_retriever
    
    def _get_llm_analyzer(self, force_init: bool = False) -> Optional[LLMAnalyzer]:
        """
        获取 LLM 分析器（懒加载）
        
        Args:
            force_init: 是否强制尝试初始化（即使之前失败过）
        
        Returns:
            LLMAnalyzer 实例或 None
        """
        # 如果已经初始化成功，直接返回
        if self._llm_analyzer is not None:
            return self._llm_analyzer
        
        # 如果之前尝试过初始化但失败了，且不强制重试，返回 None
        if self._llm_init_attempted and not force_init:
            return None
        
        # 尝试初始化
        self._llm_init_attempted = True
        logger.info("Initializing LLM analyzer...")
        try:
            api_key = getattr(self.config, 'llm_api_key', None) or \
                      getattr(self.config, 'gemini_api_key', None)
            
            if not api_key:
                logger.warning("No LLM API key configured, LLM analyzer disabled.")
                return None
            
            model = getattr(self.config, 'llm_model', 'gemini-3-flash-preview')
            self._llm_analyzer = LLMAnalyzer(api_key=api_key, model=model)
            logger.info("LLM analyzer initialized successfully.")
        except Exception as e:
            logger.warning(f"Failed to initialize LLM analyzer: {e}")
            self._llm_analyzer = None
        
        return self._llm_analyzer
    
    def _preprocess(self, text: str) -> tuple:
        """
        预处理文本
        
        Args:
            text: 原始评论文本
            
        Returns:
            (cleaned_text, metadata, features) 三元组
        """
        # 1. 文本清洗和元数据提取
        cleaned_text, metadata = self.text_cleaner.extract_clean_text(text)
        
        # 2. 特征提取
        features = self.feature_extractor.extract_features(cleaned_text, metadata)
        
        return cleaned_text, metadata, features
    
    def filter_comment(
        self, 
        comment_id: str, 
        text: str,
        enable_rag: Optional[bool] = None,
        enable_llm: Optional[bool] = None
    ) -> FilterResult:
        """
        过滤单条评论
        
        Args:
            comment_id: 评论ID
            text: 评论文本
            enable_rag: 是否启用 RAG（None 使用默认配置）
            enable_llm: 是否启用 LLM 深度分析（None 使用默认配置）
        
        Returns:
            FilterResult: 过滤结果
        """
        # 确定是否启用各组件
        use_rag = enable_rag if enable_rag is not None else getattr(self.config, 'enable_rag', False)
        use_llm = enable_llm if enable_llm is not None else getattr(self.config, 'enable_llm_deep_analysis', False)

        pipeline_start = time.time()
        logger.info(f"[{comment_id}] Start filter pipeline | rag={use_rag} llm={use_llm} text={text[:60]}{'...' if len(text) > 60 else ''}")

        # ==================== 1. 预处理 ====================
        step_start = time.time()
        cleaned_text, metadata, features = self._preprocess(text)
        logger.debug(f"[{comment_id}] Preprocess done in {time.time() - step_start:.3f}s")

        # ==================== 2. 规则引擎快速过滤 ====================
        step_start = time.time()
        rule_result = self.rule_engine.apply_rules(cleaned_text, features)
        logger.debug(f"[{comment_id}] Rule engine done in {time.time() - step_start:.3f}s | matched={rule_result.matched} rules={rule_result.matched_rules}")
        
        # 高置信度规则匹配直接返回（威胁/诈骗快速通道）
        # if rule_result.matched and rule_result.confidence >= 0.9:
        #     if rule_result.category in [CommentCategory.THREAT, CommentCategory.SCAM_SPAM]:
        #         return FilterResult(
        #             comment_id=comment_id,
        #             original_text=text,
        #             category=rule_result.category,
        #             confidence=rule_result.confidence,
        #             action="delete",
        #             reason=rule_result.reason,
        #             severity="critical" if rule_result.category == CommentCategory.THREAT else "high",
        #             is_exempted=False,
        #             processing_path="rule_fast_track",
        #             all_scores={rule_result.category.value: rule_result.confidence},
        #             detoxify_raw=None,
        #             similar_cases=None,
        #             llm_analysis=None
        #         )
        
        # ==================== 3. ML 分类 (Detoxify) ====================
        step_start = time.time()
        ml_result = self.classifier.classify(cleaned_text, features)
        logger.debug(f"[{comment_id}] ML classify done in {time.time() - step_start:.3f}s | category={ml_result.category.value} confidence={ml_result.confidence:.2f}")

        # ==================== 4. RAG 检索（可选） ====================
        rag_result: Optional[RAGResult] = None
        if use_rag:
            step_start = time.time()
            rag_retriever = self._get_rag_retriever()
            if rag_retriever is not None:
                try:
                    rag_result = rag_retriever.retrieve(cleaned_text)
                    rag_cat = rag_result.suggested_category.value if rag_result.suggested_category else "none"
                    logger.debug(f"[{comment_id}] RAG retrieve done in {time.time() - step_start:.3f}s | category={rag_cat} confidence={rag_result.confidence:.2f} cases={len(rag_result.similar_cases)}")
                except Exception as e:
                    logger.warning(f"[{comment_id}] RAG retrieval failed in {time.time() - step_start:.3f}s: {e}")
                    rag_result = None
            else:
                logger.debug(f"[{comment_id}] RAG retriever not available, skipping")
        else:
            logger.debug(f"[{comment_id}] RAG disabled, skipping")

        # ==================== 5. LLM 深度分析（可选） ====================
        llm_result: Optional[LLMAnalysisResult] = None
        if use_llm:
            # 只对低置信度或有豁免可能的案例进行 LLM 分析
            # should_use_llm = (
            #     ml_result.confidence < 0.8 or 
            #     features.get("exemption_matches") or
            #     (rag_result and rag_result.confidence < 0.7)
            # )
            
            if use_llm:
                step_start = time.time()
                llm_analyzer = self._get_llm_analyzer()
                if llm_analyzer is not None:
                    try:
                        # 准备传递给 LLM 的数据
                        ml_data = {
                            "category": ml_result.category.value if hasattr(ml_result.category, "value") else str(ml_result.category),
                            "confidence": ml_result.confidence,
                            "all_scores": {
                                k.value if hasattr(k, "value") else str(k): v 
                                for k, v in ml_result.all_scores.items()
                            }
                        }
                        
                        rag_data = None
                        if rag_result:
                            rag_data = {
                                "suggested_category": rag_result.suggested_category.value if rag_result.suggested_category and hasattr(rag_result.suggested_category, "value") else None,
                                "confidence": rag_result.confidence,
                                "similar_cases": rag_result.similar_cases,
                                "reasoning": rag_result.reasoning
                            }
                        
                        rule_data = {
                            "matched": rule_result.matched,
                            "category": rule_result.category.value if rule_result.category else None,
                            "confidence": rule_result.confidence,
                            "matched_rules": rule_result.matched_rules,
                            "reason": rule_result.reason
                        }
                        
                        llm_result = llm_analyzer.analyze(
                            text=cleaned_text,
                            features=features,
                            rule_result=rule_data,
                            ml_result=ml_data,
                            rag_result=rag_data
                        )
                    except Exception as e:
                        logger.warning(f"[{comment_id}] LLM analysis failed in {time.time() - step_start:.3f}s: {e}")
                        llm_result = None
                else:
                    logger.debug(f"[{comment_id}] LLM analyzer not available, skipping")
        else:
            logger.debug(f"[{comment_id}] LLM disabled, skipping")

        # ==================== 6. 综合决策 ====================
        step_start = time.time()
        final_result = self._make_decision(
            comment_id=comment_id,
            text=text,
            features=features,
            rule_result=rule_result,
            ml_result=ml_result,
            rag_result=rag_result,
            llm_result=llm_result
        )
        pipeline_elapsed = time.time() - pipeline_start
        logger.info(
            f"[{comment_id}] Pipeline complete in {pipeline_elapsed:.3f}s | "
            f"category={final_result.category.value} confidence={final_result.confidence:.2f} "
            f"action={final_result.action} severity={final_result.severity} "
            f"exempted={final_result.is_exempted} path={final_result.processing_path}"
        )

        return final_result
    
    def filter_batch(
        self, 
        comments: List[Dict[str, str]],
        enable_rag: Optional[bool] = None,
        enable_llm: Optional[bool] = None
    ) -> List[FilterResult]:
        """
        批量过滤评论
        
        Args:
            comments: 评论列表，每个元素包含 {"id": str, "text": str}
            enable_rag: 是否启用 RAG
            enable_llm: 是否启用 LLM
        
        Returns:
            过滤结果列表
        """
        results = []
        total = len(comments)
        
        for i, comment in enumerate(comments):
            comment_id = comment.get("id", f"batch_{i}")
            text = comment.get("text", "")
            
            if not text:
                logger.warning(f"Empty text for comment {comment_id}, skipping")
                continue
            
            try:
                result = self.filter_comment(
                    comment_id=comment_id,
                    text=text,
                    enable_rag=enable_rag,
                    enable_llm=enable_llm
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to filter comment {comment_id}: {e}")
                # 返回一个安全的默认结果
                results.append(FilterResult(
                    comment_id=comment_id,
                    original_text=text,
                    category=CommentCategory.UNKNOWN,
                    confidence=0.0,
                    action="review",
                    reason=f"Processing error: {str(e)}",
                    severity="low",
                    is_exempted=False,
                    processing_path="error",
                    all_scores={},
                    detoxify_raw=None,
                    similar_cases=None,
                    llm_analysis=None
                ))
            
            if (i + 1) % 10 == 0:
                logger.info(f"Processed {i + 1}/{total} comments")
        
        return results
    
    def _make_decision(
        self,
        comment_id: str,
        text: str,
        features: Dict,
        rule_result: RuleResult,
        ml_result: ClassificationResult,
        rag_result: Optional[RAGResult],
        llm_result: Optional[LLMAnalysisResult]
    ) -> FilterResult:
        """综合各层结果做出最终决策"""
        
        # 决策变量
        category: CommentCategory
        confidence: float
        is_exempted: bool
        severity: str
        processing_path: str
        
        # ==================== 优先使用 LLM 结果 ====================
        if llm_result is not None:
            category = llm_result.category
            confidence = llm_result.confidence
            is_exempted = llm_result.is_exempted
            severity = llm_result.severity
            processing_path = "rule+ml+rag+llm" if rag_result else "rule+ml+llm"
            logger.debug(f"[{comment_id}] Decision: using LLM result -> {category.value} ({confidence:.2f})")

        else:
            # ==================== 综合 ML 和 RAG 结果 ====================
            ml_category = ml_result.category
            ml_confidence = ml_result.confidence
            
            rag_category = rag_result.suggested_category if rag_result else None
            rag_confidence = rag_result.confidence if rag_result else 0.0
            
            # 决策逻辑
            if rag_category and ml_category == rag_category:
                # 两者一致：提高置信度
                category = ml_category
                confidence = min((ml_confidence + rag_confidence) / 1.5, 0.95)
                processing_path = "rule+ml+rag_agree"
            elif rag_result and rag_confidence > ml_confidence + 0.2:
                # RAG 显著更高
                category = rag_category
                confidence = rag_confidence
                processing_path = "rule+ml+rag_prefer"
            elif ml_confidence > 0.6:
                # ML 置信度足够
                category = ml_category
                confidence = ml_confidence
                processing_path = "rule+ml"
            elif rag_result and rag_confidence > 0.5:
                # RAG 有一定置信度
                category = rag_category
                confidence = rag_confidence
                processing_path = "rule+rag"
            else:
                # 默认使用 ML
                category = ml_category
                confidence = ml_confidence
                processing_path = "rule+ml_default"

            logger.debug(f"[{comment_id}] Decision path: {processing_path} | ml={ml_category.value}({ml_confidence:.2f}) rag={rag_category.value if rag_category else 'N/A'}({rag_confidence:.2f})")

            # 检查豁免
            is_exempted = self._check_exemption(category, features, ml_result)
            if is_exempted:
                logger.debug(f"[{comment_id}] Exemption granted: {category.value} -> safe")
                category = CommentCategory.SAFE

            # 评估严重程度
            severity = self._assess_severity(category)
        
        # ==================== 决定动作 ====================
        action = self._determine_action(category, confidence, is_exempted)
        
        # ==================== 生成原因说明 ====================
        reason = self._generate_reason(category, confidence, is_exempted, processing_path)
        
        # ==================== 构建返回结果 ====================
        # 转换 all_scores 的 key 为字符串
        all_scores_str = {}
        for k, v in ml_result.all_scores.items():
            key = k.value if hasattr(k, "value") else str(k)
            all_scores_str[key] = v
        
        # 转换 llm_result 为字典
        llm_analysis_dict = None
        if llm_result:
            llm_analysis_dict = {
                "category": llm_result.category.value if hasattr(llm_result.category, "value") else str(llm_result.category),
                "confidence": llm_result.confidence,
                "intent": llm_result.intent,
                "is_exempted": llm_result.is_exempted,
                "reasoning": llm_result.reasoning,
                "severity": llm_result.severity,
                "suggested_action": llm_result.suggested_action
            }
        
        return FilterResult(
            comment_id=comment_id,
            original_text=text,
            category=category,
            confidence=confidence,
            action=action,
            reason=reason,
            severity=severity,
            is_exempted=is_exempted,
            processing_path=processing_path,
            all_scores=all_scores_str,
            detoxify_raw=ml_result.detoxify_raw,
            similar_cases=rag_result.similar_cases if rag_result else None,
            llm_analysis=llm_analysis_dict
        )
    
    def _check_exemption(
        self, 
        category: CommentCategory, 
        features: Dict,
        ml_result: ClassificationResult
    ) -> bool:
        """检查是否应该豁免"""
        # 已经是安全类别
        if category == CommentCategory.SAFE:
            return True
        
        # 有豁免模式匹配
        exemption_matches = features.get("exemption_matches", [])
        if not exemption_matches:
            return False
        
        # 威胁类不豁免
        if category == CommentCategory.THREAT:
            return False
        
        # 诈骗类不豁免
        if category == CommentCategory.SCAM_SPAM:
            return False
        
        # 低毒性 + 有豁免模式 -> 豁免
        detoxify_raw = ml_result.detoxify_raw or {}
        toxicity = detoxify_raw.get("toxicity", 0)
        
        if toxicity < 0.5 and exemption_matches:
            return True
        
        # 中等毒性但有多个豁免模式
        if toxicity < 0.7 and len(exemption_matches) >= 2:
            return True
        
        return False
    
    def _assess_severity(self, category: CommentCategory) -> str:
        """评估严重程度"""
        severity_map = {
            CommentCategory.THREAT: "critical",
            CommentCategory.HATE_IDENTITY: "high",
            CommentCategory.SCAM_SPAM: "high",
            CommentCategory.HATE_APPEARANCE: "medium",
            CommentCategory.DISTORTION: "medium",
            CommentCategory.TOXIC: "medium",
            CommentCategory.TRAFFIC_HIJACKING: "low",
            CommentCategory.SAFE: "low",
            CommentCategory.UNKNOWN: "low"
        }
        return severity_map.get(category, "medium")
    
    def _determine_action(
        self, 
        category: CommentCategory, 
        confidence: float, 
        is_exempted: bool
    ) -> str:
        """决定动作"""
        # 豁免或安全类别 -> 保留
        if is_exempted or category == CommentCategory.SAFE:
            return "keep"
        
        # 获取阈值
        delete_threshold = self._get_threshold("delete")
        review_threshold = self._get_threshold("review")
        
        # 高危类别降低阈值
        if category in [CommentCategory.THREAT, CommentCategory.SCAM_SPAM]:
            delete_threshold *= 0.8
            review_threshold *= 0.8
        
        # 根据置信度决定动作
        if confidence >= delete_threshold:
            return "delete"
        elif confidence >= review_threshold:
            return "review"
        else:
            return "keep"
    
    def _get_threshold(self, action_type: str) -> float:
        """获取阈值"""
        # 尝试从配置获取
        if hasattr(self.config, 'get_threshold'):
            return self.config.get_threshold(action_type)
        
        # 默认阈值（根据严格程度）
        strictness = getattr(self.config, 'strictness', FilterStrictness.MEDIUM)
        
        thresholds = {
            FilterStrictness.LOW: {"delete": 0.9, "review": 0.7},
            FilterStrictness.MEDIUM: {"delete": 0.8, "review": 0.5},
            FilterStrictness.HIGH: {"delete": 0.7, "review": 0.4},
        }
        
        return thresholds.get(strictness, thresholds[FilterStrictness.MEDIUM]).get(action_type, 0.8)
    
    def _generate_reason(
        self, 
        category: CommentCategory, 
        confidence: float, 
        is_exempted: bool, 
        path: str
    ) -> str:
        """生成原因说明"""
        category_name = category.value if hasattr(category, "value") else str(category)
        
        if is_exempted:
            return f"Content exempted (matched positive patterns), original category: {category_name}, confidence: {confidence:.1%}, path: {path}"
        elif category == CommentCategory.SAFE:
            return f"Content appears safe, confidence: {confidence:.1%}, path: {path}"
        else:
            return f"Detected [{category_name}] with {confidence:.1%} confidence, path: {path}"
    
    def set_strictness(self, level: FilterStrictness):
        """
        设置过滤严格程度
        
        Args:
            level: FilterStrictness 枚举值
        """
        self.config.strictness = level
        self.rule_engine = RuleEngine(level)
        logger.info(f"Filter strictness set to: {level.value if hasattr(level, 'value') else level}")
    
    def get_stats(self, results: List[FilterResult]) -> Dict[str, Any]:
        """
        统计过滤结果
        
        Args:
            results: 过滤结果列表
            
        Returns:
            统计信息字典
        """
        total = len(results)
        if total == 0:
            return {"total": 0}
        
        # 初始化统计
        stats = {
            "total": total,
            "actions": {"keep": 0, "delete": 0, "review": 0},
            "categories": {},
            "severities": {"low": 0, "medium": 0, "high": 0, "critical": 0},
            "avg_confidence": 0.0,
            "exempted_count": 0,
            "processing_paths": {}
        }
        
        # 初始化类别计数
        for cat in CommentCategory:
            stats["categories"][cat.value] = 0
        
        confidence_sum = 0.0
        
        for result in results:
            # 动作统计
            action = result.action
            if action in stats["actions"]:
                stats["actions"][action] += 1
            
            # 类别统计
            cat_value = result.category.value if hasattr(result.category, "value") else str(result.category)
            if cat_value in stats["categories"]:
                stats["categories"][cat_value] += 1
            else:
                stats["categories"][cat_value] = 1
            
            # 严重程度统计
            severity = result.severity
            if severity in stats["severities"]:
                stats["severities"][severity] += 1
            
            # 置信度累加
            confidence_sum += result.confidence
            
            # 豁免计数
            if result.is_exempted:
                stats["exempted_count"] += 1
            
            # 处理路径统计
            path = result.processing_path
            stats["processing_paths"][path] = stats["processing_paths"].get(path, 0) + 1
        
        # 计算平均置信度
        stats["avg_confidence"] = round(confidence_sum / total, 4)
        
        # 计算百分比
        stats["action_percentages"] = {
            action: round(count / total * 100, 2)
            for action, count in stats["actions"].items()
        }
        
        stats["category_percentages"] = {
            cat: round(count / total * 100, 2)
            for cat, count in stats["categories"].items()
            if count > 0
        }
        
        return stats
    
    def health_check(self) -> Dict[str, Any]:
        """
        健康检查
        
        Returns:
            各组件状态
        """
        status = {
            "engine": "healthy",
            "text_cleaner": "healthy",
            "feature_extractor": "healthy",
            "rule_engine": "healthy",
            "ml_classifier": "unknown",
            "rag_retriever": "not_initialized",
            "llm_analyzer": "not_initialized"
        }
        
        # 检查 ML 分类器
        try:
            test_result = self.classifier.classify("test", {})
            status["ml_classifier"] = "healthy"
        except Exception as e:
            status["ml_classifier"] = f"error: {str(e)}"
        
        # 检查 RAG（不强制初始化，只报告当前状态）
        if self._rag_retriever is not None:
            try:
                self._rag_retriever.retrieve("test")
                status["rag_retriever"] = "healthy"
            except Exception as e:
                status["rag_retriever"] = f"error: {str(e)}"
        elif self._rag_init_attempted:
            status["rag_retriever"] = "init_failed"
        
        # 检查 LLM（不强制初始化，只报告当前状态）
        if self._llm_analyzer is not None:
            try:
                health = self._llm_analyzer.health_check()
                status["llm_analyzer"] = health.get("status", "unknown")
            except Exception as e:
                status["llm_analyzer"] = f"error: {str(e)}"
        elif self._llm_init_attempted:
            status["llm_analyzer"] = "init_failed"
        
        return status


# ==================== 便捷函数 ====================

def create_engine(
    strictness: str = "medium",
    enable_rag: bool = True,
    enable_llm: bool = False,
    llm_api_key: str = None
) -> CommentFilterEngine:
    """
    创建过滤引擎的便捷函数
    
    Args:
        strictness: 严格程度 ("low", "medium", "high")
        enable_rag: 是否启用 RAG
        enable_llm: 是否启用 LLM
        llm_api_key: LLM API Key
        
    Returns:
        CommentFilterEngine 实例
    """
    # 创建配置
    config = FilterConfig()
    
    # 设置严格程度
    strictness_map = {
        "low": FilterStrictness.LOW,
        "medium": FilterStrictness.MEDIUM,
        "high": FilterStrictness.HIGH,
        "lenient": FilterStrictness.LOW,
        "normal": FilterStrictness.MEDIUM,
        "strict": FilterStrictness.HIGH,
    }
    config.strictness = strictness_map.get(strictness.lower(), FilterStrictness.MEDIUM)
    
    # 设置组件开关
    config.enable_rag = enable_rag
    config.enable_llm_deep_analysis = enable_llm
    
    if llm_api_key:
        config.llm_api_key = llm_api_key
    
    return CommentFilterEngine(config)


# ==================== 命令行入口 ====================

def main():
    """命令行入口：用于本地快速测试过滤引擎"""
    import argparse
    import sys
    import os

    parser = argparse.ArgumentParser(
        description="Comment Filter Engine 测试工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=r"""
示例:
  # 单条测试
  python -m filter_engine.engine --test "去死吧垃圾"

  # 单条测试（启用LLM，需要设置API Key）
  python -m filter_engine.engine --test "太美了我恨你" --enable-llm --llm-api-key "sk-xxx"

  # 或者通过环境变量设置 API Key
  export LLM_API_KEY="sk-xxx"
  python -m filter_engine.engine --test "太美了我恨你" --enable-llm

  # 批量测试
  python -m filter_engine.engine --batch '[{"id":"1","text":"去死吧"}]' --stats

  # 设置严格程度
  python -m filter_engine.engine --test "nobody asked" --strictness strict
  
  # 健康检查
  python -m filter_engine.engine --health
"""
    )

    parser.add_argument("--test", type=str, help="测试单条评论文本")
    parser.add_argument("--id", type=str, default="test_comment", help="单条测试的评论ID")

    parser.add_argument("--batch", type=str, help="批量测试：传入JSON字符串数组")
    parser.add_argument("--batch-file", type=str, help="批量测试：从文件读取JSON内容")

    parser.add_argument("--stats", action="store_true", help="输出批量统计信息")
    parser.add_argument("--health", action="store_true", help="执行健康检查")

    # RAG/LLM 开关
    parser.add_argument("--enable-rag", action="store_true", help="禁用 RAG")
    parser.add_argument("--enable-llm", action="store_true", help="启用 LLM 深度分析")
    parser.add_argument("--llm-api-key", type=str, help="LLM API Key（也可通过环境变量 LLM_API_KEY 设置）")

    parser.add_argument(
        "--strictness",
        type=str,
        default="medium",
        choices=["lenient", "normal", "strict", "low", "medium", "high"],
        help="过滤严格程度"
    )

    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="日志级别"
    )

    args = parser.parse_args()

    # 配置日志
    logging.basicConfig(
        level=getattr(logging, args.log_level, logging.INFO),
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    )

    # 获取 LLM API Key（命令行参数优先，其次环境变量）
    llm_api_key = args.llm_api_key or os.getenv("GEMINI_API_KEY")
    
    # 使用 create_engine 创建引擎
    engine = create_engine(
        strictness=args.strictness,
        enable_rag=args.enable_rag,
        enable_llm=True,
        llm_api_key=llm_api_key
    )
    
    # 如果启用了 LLM 但没有 API Key，给出警告
    if args.enable_llm and not llm_api_key:
        logger.warning("--enable-llm 已设置但未提供 API Key，LLM 分析将不可用")
        logger.warning("请通过 --llm-api-key 或环境变量 LLM_API_KEY 设置")

    # 确定运行时 RAG/LLM 开关
    enable_rag = args.enable_rag
    enable_llm = args.enable_llm

    def print_result(r: FilterResult):
        """打印单条结果"""
        payload = r.to_dict()
        print(json.dumps(payload, ensure_ascii=False, indent=2))

    # 健康检查
    if args.health:
        print("\n执行健康检查...")
        print("-" * 50)
        health = engine.health_check()
        print(json.dumps(health, ensure_ascii=False, indent=2))
        return

    # 单条测试
    if args.test:
        print(f"\n分析文本: {args.test}")
        print(f"RAG: {'启用' if enable_rag else '禁用'}, LLM: {'启用' if enable_llm else '禁用'}")
        print("-" * 50)
        r = engine.filter_comment(
            comment_id=args.id,
            text=args.test,
            enable_rag=enable_rag,
            enable_llm=True
        )
        print_result(r)
        return

    # 批量测试
    batch_text = None
    if args.batch_file:
        try:
            with open(args.batch_file, "r", encoding="utf-8") as f:
                batch_text = f.read()
        except Exception as e:
            print(f"读取文件失败: {e}", file=sys.stderr)
            sys.exit(2)
    elif args.batch:
        batch_text = args.batch

    if batch_text:
        try:
            comments = json.loads(batch_text)
            if not isinstance(comments, list):
                raise ValueError("批量输入必须是 JSON 数组")
        except Exception as e:
            print(f"批量JSON解析失败: {e}", file=sys.stderr)
            sys.exit(2)

        print(f"\n批量分析 {len(comments)} 条评论...")
        print(f"RAG: {'启用' if enable_rag else '禁用'}, LLM: {'启用' if enable_llm else '禁用'}")
        print("-" * 50)
        
        results = engine.filter_batch(comments, enable_rag=enable_rag, enable_llm=enable_llm)
        
        for r in results:
            print_result(r)
            print()

        if args.stats:
            stats = engine.get_stats(results)
            print("\n" + "=" * 50)
            print("统计信息:")
            print("=" * 50)
            print(json.dumps(stats, ensure_ascii=False, indent=2))
        return

    # 无参数时显示帮助
    parser.print_help()


if __name__ == "__main__":
    main()
