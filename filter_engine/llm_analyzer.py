# filter_engine/llm_analyzer.py
"""
LLM 深度分析器

使用 Kimi (Moonshot AI) API 进行深度语义理解
包含降级方案（当 API 不可用时使用规则分析）

用法:
    from filter_engine.llm_analyzer import LLMAnalyzer
    
    analyzer = LLMAnalyzer(api_key="your-moonshot-api-key")
    result = analyzer.analyze(text, features)
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
import json
import logging
import os
import re

from config.settings import CommentCategory

logger = logging.getLogger(__name__)
api_key = "sk-gF7hiU9IuKttaT2Q77YovdVUaXwENXNK3cNtWVneIrRjDIqU"


@dataclass
class LLMAnalysisResult:
    """LLM 分析结果"""
    category: CommentCategory
    confidence: float
    intent: str  # "attack", "joke", "compliment", "spam", "defense", "neutral"
    is_exempted: bool
    reasoning: str
    severity: str  # "low", "medium", "high", "critical"
    suggested_action: str


@dataclass
class CyberbullyingReport:
    """网暴分析报告"""
    is_cyberbullying: bool
    attack_pattern: str
    estimated_accounts: int
    main_narratives: List[str]
    trend: str
    recommendations: List[str]


class LLMAnalyzer:
    """
    LLM 深度分析器
    
    使用 Kimi (Moonshot AI) API 进行深度语义理解

    """
    
    # Kimi API 配置
    KIMI_BASE_URL = "https://api.moonshot.cn/v1"
    KIMI_MODELS = {
        "fast": "moonshot-v1-8k",      # 快速，适合短文本
        "balanced": "kimi-k2-turbo-preview",  # 平衡
        "long": "moonshot-v1-128k"      # 长上下文
    }
    
    def __init__(
        self, 
        api_key: str = "sk-gF7hiU9IuKttaT2Q77YovdVUaXwENXNK3cNtWVneIrRjDIqU", 
        model: str = "kimi-k2-turbo-preview",
        base_url: str = None,
        timeout: int = 30
    ):
        """
        初始化 LLM 分析器
        
        Args:
            api_key: Moonshot API Key，也可通过环境变量 MOONSHOT_API_KEY 设置
            model: Kimi 模型名称，可选:
                   - moonshot-v1-8k (快速，推荐用于评论分析)
                   - moonshot-v1-32k (平衡)
                   - moonshot-v1-128k (长上下文)
            base_url: API 基础地址，默认为 Kimi 官方地址
            timeout: 请求超时时间（秒）
        """
        self.api_key = api_key or os.getenv("MOONSHOT_API_KEY") or os.getenv("KIMI_API_KEY")
        self.model = model
        self.base_url = base_url or self.KIMI_BASE_URL
        self.timeout = timeout
        
        self._client = None
        self._initialized = False
        
        # 豁免模式
        self.exemption_patterns = [
            "太好看", "太美了", "羡慕", "嫉妒", "我恨你",
            "slay", "queen", "legend", "iconic", "obsessed",
            "im dying", "im dead", "killing me", "amazing"
        ]
        
        # 攻击指标
        self.attack_indicators = [
            "去死", "垃圾", "退网", "翻车", "社死",
            "kys", "trash", "flop", "ratio", "nobody asked"
        ]
    
    def _lazy_init(self):
        """懒加载 Kimi 客户端"""
        if self._initialized:
            return
        
        if not self.api_key:
            logger.warning(
                "未提供 Moonshot API key，LLM 分析将使用降级模式。"
                "请设置环境变量 MOONSHOT_API_KEY 或在初始化时传入 api_key"
            )
            self._initialized = True
            return
        
        try:
            # 使用 OpenAI 兼容客户端（Kimi 兼容 OpenAI API 格式）
            from openai import OpenAI
            
            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout
            )
            logger.info(f"Kimi 客户端初始化成功，模型: {self.model}")
            
        except ImportError:
            logger.warning(
                "openai 包未安装，使用降级模式。"
                "请安装: pip install openai"
            )
        except Exception as e:
            logger.warning(f"Kimi 客户端初始化失败: {e}，使用降级模式")
        
        self._initialized = True
    
    def analyze(
        self,
        text: str,
        features: Dict,
        rule_result: Dict = None,
        ml_result: Dict = None,
        rag_result: Dict = None
    ) -> LLMAnalysisResult:
        """
        深度分析评论
        
        Args:
            text: 评论文本
            features: 预处理特征
            rule_result: 规则引擎结果
            ml_result: ML 分类结果
            rag_result: RAG 检索结果
        
        Returns:
            LLMAnalysisResult
        """
        self._lazy_init()
        
        # 尝试使用 Kimi API
        if self._client:
            try:
                return self._analyze_with_kimi(text, features, ml_result, rag_result)
            except Exception as e:
                logger.warning(f"Kimi API 调用失败: {e}，使用降级分析")
        
        # 降级：使用规则分析
        return self._analyze_fallback(text, features, ml_result, rag_result)
    
    def _analyze_with_kimi(
        self,
        text: str,
        features: Dict,
        ml_result: Dict,
        rag_result: Dict
    ) -> LLMAnalysisResult:
        """使用 Kimi API 分析"""
        
        # 构建 prompt
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_analysis_prompt(text, features, ml_result, rag_result)
        
        # 调用 Kimi API
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,  # 低温度，更稳定的输出
            max_tokens=512,
            response_format={"type": "json_object"}  # 强制 JSON 输出
        )
        
        # 解析响应
        response_text = response.choices[0].message.content
        return self._parse_kimi_response(response_text, text)
    
    def _build_system_prompt(self) -> str:
        """构建系统提示词"""
        return """你是一个专业的社交媒体内容审核AI助手。你的任务是分析评论内容，判断其类型、意图和是否需要过滤。

你需要特别注意以下几点：
1. 区分真实攻击和粉丝间的玩笑/表达方式
2. "我恨你这么好看"、"羡慕死了"等是粉丝表达羡慕，不是攻击
3. "slay"、"queen"、"im dead"、"killing it"等在年轻人语境下是正面表达
4. 粉丝引用黑子言论进行反击时，应该豁免
5. 判断时要考虑上下文和语气

请始终以JSON格式输出分析结果。"""

    def _build_analysis_prompt(
        self,
        text: str,
        features: Dict,
        ml_result: Dict,
        rag_result: Dict
    ) -> str:
        """构建用户分析 prompt"""
        
        # 处理 ML 结果
        ml_category = ml_result.get("category", {}) if ml_result else None
        if hasattr(ml_category, "value"):
            ml_category = ml_category.value
        ml_confidence = ml_result.get("confidence", 0) if ml_result else 0
        
        # 处理 RAG 结果
        rag_category = rag_result.get("suggested_category") if rag_result else None
        if hasattr(rag_category, "value"):
            rag_category = rag_category.value
        rag_confidence = rag_result.get("confidence", 0) if rag_result else 0
        
        # 相似案例
        similar_cases = rag_result.get("similar_cases", []) if rag_result else []
        similar_cases_text = "\n".join([
            f"  - \"{c['text']}\" -> {c['category']} (相似度: {c['similarity']:.2f})"
            for c in similar_cases[:3]
        ]) if similar_cases else "无"
        
        # 特征信息
        keyword_matches = list(features.get("keyword_matches", {}).keys()) if features else []
        toxic_emoji_count = features.get("toxic_emoji_count", 0) if features else 0
        exemption_matches = features.get("exemption_matches", []) if features else []
        
        prompt = f"""请分析以下社交媒体评论：

【评论内容】
"{text}"

【已提取特征】
- 关键词类别: {keyword_matches if keyword_matches else "无"}
- 恶意Emoji数量: {toxic_emoji_count}
- 豁免模式匹配: {exemption_matches if exemption_matches else "无"}

【机器学习分类】
- 类别: {ml_category if ml_category else "未知"}
- 置信度: {ml_confidence:.2f}

【RAG检索结果】
- 建议类别: {rag_category if rag_category else "未知"}
- 置信度: {rag_confidence:.2f}
- 相似案例:
{similar_cases_text}

请分析并输出JSON格式结果，包含以下字段：
{{
    "intent": "attack/joke/compliment/spam/defense/neutral 之一",
    "is_exempted": true或false,
    "category": "threat/hate_appearance/hate_identity/distortion/toxic/traffic_hijacking/scam_spam/safe 之一",
    "severity": "critical/high/medium/low 之一",
    "confidence": 0.0到1.0之间的数值,
    "reasoning": "简短的判断理由"
}}"""
        
        return prompt
    
    def _parse_kimi_response(self, response: str, original_text: str) -> LLMAnalysisResult:
        """解析 Kimi 响应"""
        try:
            # 尝试直接解析 JSON
            try:
                data = json.loads(response)
            except json.JSONDecodeError:
                # 尝试从响应中提取 JSON
                json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
                if json_match:
                    data = json.loads(json_match.group())
                else:
                    raise ValueError("响应中未找到有效的 JSON")
            
            # 解析类别
            category_str = data.get("category", "safe")
            try:
                category = CommentCategory(category_str)
            except ValueError:
                logger.warning(f"未知类别: {category_str}，默认为 safe")
                category = CommentCategory.SAFE
            
            # 解析其他字段
            is_exempted = data.get("is_exempted", False)
            severity = data.get("severity", "medium")
            confidence = float(data.get("confidence", 0.8))
            
            return LLMAnalysisResult(
                category=category,
                confidence=min(max(confidence, 0.0), 1.0),  # 确保在 [0, 1] 范围
                intent=data.get("intent", "neutral"),
                is_exempted=is_exempted,
                reasoning=data.get("reasoning", "Kimi 分析"),
                severity=severity,
                suggested_action=self._get_action(category, severity, is_exempted)
            )
            
        except Exception as e:
            logger.warning(f"解析 Kimi 响应失败: {e}，响应内容: {response[:200]}")
            return self._analyze_fallback(original_text, {}, {}, {})
    
    def _analyze_fallback(
        self,
        text: str,
        features: Dict,
        ml_result: Dict,
        rag_result: Dict
    ) -> LLMAnalysisResult:
        """降级分析（不使用 LLM）"""
        text_lower = text.lower()
        features = features or {}
        ml_result = ml_result or {}
        rag_result = rag_result or {}
        
        # 分析意图
        intent = self._analyze_intent(text_lower, features)
        
        # 检查豁免
        is_exempted = self._check_exemption(text_lower, intent, features)
        
        # 确定类别
        if is_exempted:
            category = CommentCategory.SAFE
        else:
            # 优先使用 ML 结果
            ml_category = ml_result.get("category")
            if hasattr(ml_category, "value"):
                category = ml_category
            elif isinstance(ml_category, str):
                try:
                    category = CommentCategory(ml_category)
                except ValueError:
                    category = CommentCategory.SAFE
            else:
                # 使用 RAG 结果
                rag_category = rag_result.get("suggested_category")
                if rag_category:
                    category = rag_category if isinstance(rag_category, CommentCategory) else CommentCategory.SAFE
                else:
                    category = CommentCategory.SAFE
        
        # 确定严重程度
        severity = self._assess_severity(category)
        
        # 确定置信度
        confidence = self._calculate_confidence(ml_result, rag_result, is_exempted)
        
        return LLMAnalysisResult(
            category=category,
            confidence=confidence,
            intent=intent,
            is_exempted=is_exempted,
            reasoning=f"降级分析: intent={intent}, exempted={is_exempted}",
            severity=severity,
            suggested_action=self._get_action(category, severity, is_exempted)
        )
    
    def _analyze_intent(self, text: str, features: Dict) -> str:
        """分析意图"""
        exemption_matches = features.get("exemption_matches", [])
        
        # 正面表达
        if exemption_matches:
            if any(p in text for p in ["太美了", "太好看", "slay", "queen", "amazing", "love"]):
                return "compliment"
        
        # 攻击性
        if any(p in text for p in self.attack_indicators):
            # 检查是否是反击黑子
            if "那个黑子说" in text or "someone said" in text or "他们说" in text:
                return "defense"
            return "attack"
        
        # 垃圾信息
        if features.get("has_url") and any(p in text for p in ["dm", "check", "link", "bio", "私信", "加我"]):
            return "spam"
        
        # 玩笑/夸张表达
        if any(p in text for p in ["im dying", "im dead", "笑死", "哈哈", "绝了", "无语"]):
            return "joke"
        
        return "neutral"
    
    def _check_exemption(self, text: str, intent: str, features: Dict) -> bool:
        """检查豁免"""
        # 正面意图直接豁免
        if intent in ["compliment", "joke", "defense"]:
            return True
        
        exemption_matches = features.get("exemption_matches", [])
        keyword_matches = features.get("keyword_matches", {})
        
        # 有豁免模式匹配
        if exemption_matches:
            # 但如果有严重威胁，不豁免
            if "threat" in keyword_matches:
                return False
            # 轻微负面但有豁免模式，可以豁免
            if "toxic" in keyword_matches and len(keyword_matches.get("toxic", [])) <= 1:
                return True
            # 有豁免模式且无其他严重问题
            if not any(k in keyword_matches for k in ["threat", "hate_identity", "scam_spam"]):
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
    
    def _calculate_confidence(self, ml_result: Dict, rag_result: Dict, is_exempted: bool) -> float:
        """计算置信度"""
        base = 0.7
        
        if ml_result and ml_result.get("confidence", 0) > 0.6:
            base += 0.1
        
        if rag_result and rag_result.get("confidence", 0) > 0.6:
            base += 0.1
        
        if is_exempted:
            base += 0.05
        
        return min(base, 0.95)
    
    def _get_action(self, category: CommentCategory, severity: str, is_exempted: bool) -> str:
        """获取建议操作"""
        if is_exempted:
            return "allow"
        
        if severity == "critical":
            return "block_and_report"
        elif severity == "high":
            return "block"
        elif severity == "medium":
            return "hide_and_review"
        else:
            return "allow_with_flag"
    
    # ==================== 批量分析接口 ====================
    
    def analyze_batch(
        self,
        comments: List[Dict],
        features_list: List[Dict] = None
    ) -> List[LLMAnalysisResult]:
        """
        批量分析评论
        
        Args:
            comments: 评论列表，每个元素包含 "text" 字段
            features_list: 对应的特征列表
            
        Returns:
            分析结果列表
        """
        results = []
        features_list = features_list or [{}] * len(comments)
        
        for comment, features in zip(comments, features_list):
            text = comment.get("text", "") if isinstance(comment, dict) else str(comment)
            try:
                result = self.analyze(text, features)
                results.append(result)
            except Exception as e:
                logger.error(f"分析评论失败: {e}")
                results.append(self._analyze_fallback(text, features, {}, {}))
        
        return results
    
    # ==================== 网暴分析 ====================
    
    def analyze_cyberbullying(
        self, 
        comments: List[Dict], 
        time_window_hours: int = 24
    ) -> CyberbullyingReport:
        """
        网暴场景分析
        
        Args:
            comments: 评论列表
            time_window_hours: 时间窗口（小时）
            
        Returns:
            CyberbullyingReport
        """
        
        if len(comments) < 5:
            return CyberbullyingReport(
                is_cyberbullying=False,
                attack_pattern="insufficient_data",
                estimated_accounts=0,
                main_narratives=[],
                trend="unknown",
                recommendations=["数据量不足，继续监控"]
            )
        
        # 统计攻击类评论
        attack_comments = []
        for c in comments:
            cat = c.get("category")
            if hasattr(cat, "value"):
                cat = cat.value
            if cat and cat != "safe":
                attack_comments.append(c)
        
        attack_ratio = len(attack_comments) / len(comments)
        
        # 判断是否网暴
        is_cyberbullying = attack_ratio > 0.6 and len(attack_comments) >= 10
        
        # 提取叙事
        narratives = self._extract_narratives(attack_comments)
        
        # 判断趋势
        trend = self._analyze_trend(comments)
        
        # 生成建议
        recommendations = self._generate_recommendations(
            is_cyberbullying, 
            attack_ratio, 
            len(attack_comments)
        )
        
        return CyberbullyingReport(
            is_cyberbullying=is_cyberbullying,
            attack_pattern="coordinated" if is_cyberbullying else "sporadic",
            estimated_accounts=len(set(
                c.get("user_id", str(i)) for i, c in enumerate(attack_comments)
            )),
            main_narratives=narratives[:3],
            trend=trend,
            recommendations=recommendations
        )
    
    def _extract_narratives(self, comments: List[Dict]) -> List[str]:
        """提取主要攻击叙事"""
        narrative_keywords = {}
        for comment in comments:
            text = comment.get("text", "").lower()
            for keyword in ["假", "骗", "抄袭", "翻车", "塌房", "fake", "fraud", "exposed", "scam"]:
                if keyword in text:
                    narrative_keywords[keyword] = narrative_keywords.get(keyword, 0) + 1
        
        sorted_narratives = sorted(
            narrative_keywords.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        return [f"关于'{k}'的指控 (出现{v}次)" for k, v in sorted_narratives]
    
    def _analyze_trend(self, comments: List[Dict]) -> str:
        """分析趋势"""
        if len(comments) < 2:
            return "unknown"
        
        mid_point = len(comments) // 2
        first_half = comments[:mid_point]
        second_half = comments[mid_point:]
        
        def get_attack_ratio(clist):
            attacks = sum(1 for c in clist if c.get("category") and (
                c.get("category") != CommentCategory.SAFE and 
                c.get("category") != "safe"
            ))
            return attacks / max(len(clist), 1)
        
        first_ratio = get_attack_ratio(first_half)
        second_ratio = get_attack_ratio(second_half)
        
        if second_ratio > first_ratio * 1.2:
            return "escalating"
        elif second_ratio < first_ratio * 0.8:
            return "declining"
        else:
            return "stable"
    
    def _generate_recommendations(
        self, 
        is_cyberbullying: bool, 
        attack_ratio: float, 
        attack_count: int
    ) -> List[str]:
        """生成建议"""
        recommendations = []
        
        if is_cyberbullying:
            recommendations.append("⚠️ 检测到协同攻击，建议开启最高过滤等级")
            recommendations.append("考虑暂时关闭评论或限制为粉丝可见")
            recommendations.append("保存证据截图，如有必要可向平台举报")
        else:
            if attack_ratio > 0.3:
                recommendations.append("负面评论比例偏高，建议提升过滤等级")
            if attack_count > 20:
                recommendations.append("建议开启账号追踪功能，识别重复攻击者")
        
        recommendations.append("持续监控评论趋势变化")
        
        return recommendations
    
    # ==================== 健康检查 ====================
    
    def health_check(self) -> Dict:
        """
        健康检查，测试 API 连接
        
        Returns:
            健康状态信息
        """
        self._lazy_init()
        
        result = {
            "status": "unknown",
            "api_available": False,
            "model": self.model,
            "base_url": self.base_url,
            "fallback_ready": True
        }
        
        if not self._client:
            result["status"] = "fallback_mode"
            result["message"] = "API 客户端未初始化，使用降级模式"
            return result
        
        try:
            # 发送测试请求
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": "测试连接，请回复 OK"}
                ],
                max_tokens=10
            )
            
            result["status"] = "healthy"
            result["api_available"] = True
            result["message"] = "API 连接正常"
            
        except Exception as e:
            result["status"] = "degraded"
            result["message"] = f"API 连接失败: {str(e)}"
        
        return result


# ==================== 便捷函数 ====================

def create_analyzer(
    api_key: str = None,
    model: str = "moonshot-v1-8k"
) -> LLMAnalyzer:
    """
    创建 LLM 分析器的便捷函数
    
    Args:
        api_key: Moonshot API Key
        model: 模型名称
        
    Returns:
        LLMAnalyzer 实例
    """
    return LLMAnalyzer(api_key=api_key, model=model)


# ==================== 命令行入口 ====================

def main():
    """命令行入口"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="LLM 分析器测试工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    # 测试分析
    python -m filter_engine.llm_analyzer --test "去死吧垃圾"
    
    # 健康检查
    python -m filter_engine.llm_analyzer --health
    
    # 使用指定模型
    python -m filter_engine.llm_analyzer --test "测试" --model moonshot-v1-32k
        """
    )
    
    parser.add_argument(
        "--test",
        type=str,
        help="测试分析指定文本"
    )
    parser.add_argument(
        "--health",
        action="store_true",
        help="执行健康检查"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="moonshot-v1-8k",
        help="使用的模型 (默认: moonshot-v1-8k)"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        help="Moonshot API Key (也可通过环境变量 MOONSHOT_API_KEY 设置)"
    )
    
    args = parser.parse_args()
    
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # 创建分析器
    analyzer = LLMAnalyzer(
        api_key=api_key,
        model=args.model
    )
    
    if args.health:
        print("\n执行健康检查...")
        print("-" * 50)
        health = analyzer.health_check()
        for key, value in health.items():
            print(f"{key}: {value}")
            
    elif args.test:
        print(f"\n分析文本: {args.test}")
        print("-" * 50)
        result = analyzer.analyze(args.test, {})
        print(f"类别: {result.category.value}")
        print(f"置信度: {result.confidence:.2f}")
        print(f"意图: {result.intent}")
        print(f"是否豁免: {result.is_exempted}")
        print(f"严重程度: {result.severity}")
        print(f"建议操作: {result.suggested_action}")
        print(f"推理: {result.reasoning}")
        
    else:
        parser.print_help()


if __name__ == "__main__":
    main()