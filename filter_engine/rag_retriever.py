# filter_engine/rag_retriever.py
from typing import Dict, List, Optional
from dataclasses import dataclass
import numpy as np
import logging
import time
import os
import faiss

from config.settings import CommentCategory

logger = logging.getLogger(__name__)


@dataclass
class RAGResult:
    """RAG 检索结果"""
    similar_cases: List[Dict]
    suggested_category: Optional[CommentCategory]
    confidence: float
    reasoning: str


class RAGRetriever:
    """
    RAG 检索增强模块
    
    使用 FAISS + BGE-M3 进行向量检索
    """
    
    def __init__(self, model_name: str = "BAAI/bge-m3", index_path: str = None):
        """
        初始化 RAG 检索器
        
        Args:
            model_name: BGE-M3 模型名称
            index_path: FAISS 索引保存路径
        """
        self.model_name = model_name
        self.index_path = index_path or "data/faiss_index"
        
        self._model = None
        self._index = None
        self._case_library = None
        self._initialized = False
    
    def _lazy_init(self):
        """懒加载模型和索引"""
        if self._initialized:
            return
        
        # 加载 embedding 模型
        self._load_embedding_model()
        
        # 初始化案例库
        self._case_library = self._init_case_library()
        
        # 构建或加载 FAISS 索引
        self._build_index()
        
        self._initialized = True
    
    def _load_embedding_model(self):
        """加载 BGE-M3 模型"""
        try:
            from FlagEmbedding import BGEM3FlagModel
            logger.info(f"Loading BGE-M3 model: {self.model_name}")
            self._model = BGEM3FlagModel(
                self.model_name,
                use_fp16=False,  # CPU 上使用 fp32
                device="cpu"
            )
            logger.info("BGE-M3 model loaded successfully")
        except ImportError:
            logger.warning("FlagEmbedding not available, falling back to sentence-transformers")
            try:
                from sentence_transformers import SentenceTransformer
                # 使用 sentence-transformers 的 BGE 模型作为备选
                self._model = SentenceTransformer("BAAI/bge-large-zh-v1.5")
                logger.info("Fallback to sentence-transformers BGE model")
            except ImportError:
                raise ImportError(
                    "请安装 FlagEmbedding 或 sentence-transformers:\n"
                    "pip install FlagEmbedding\n"
                    "或 pip install sentence-transformers"
                )
    
    def _encode(self, texts: List[str]) -> np.ndarray:
        """编码文本为向量"""
        if hasattr(self._model, 'encode'):
            # FlagEmbedding BGEM3FlagModel
            if hasattr(self._model.encode(["test"]), 'get'):
                # BGE-M3 返回字典
                result = self._model.encode(texts)
                return np.array(result['dense_vecs']).astype('float32')
            else:
                # sentence-transformers
                return self._model.encode(texts, normalize_embeddings=True).astype('float32')
        else:
            raise ValueError("Unknown model type")
    
    def _init_case_library(self) -> List[Dict]:
        """初始化案例库"""
        return [
            # 威胁类案例
            {"text": "去死吧垃圾", "category": CommentCategory.THREAT, "confidence": 0.95, "notes": "直接死亡威胁"},
            {"text": "我要杀了你", "category": CommentCategory.THREAT, "confidence": 0.95, "notes": "直接死亡威胁"},
            {"text": "我知道你住哪，等着", "category": CommentCategory.THREAT, "confidence": 0.9, "notes": "隐含人肉威胁"},
            {"text": "kys nobody wants you", "category": CommentCategory.THREAT, "confidence": 0.95, "notes": "英文死亡威胁缩写"},
            {"text": "kill yourself loser", "category": CommentCategory.THREAT, "confidence": 0.95, "notes": "英文死亡威胁"},
            {"text": "I will find you and hurt you", "category": CommentCategory.THREAT, "confidence": 0.9, "notes": "英文暴力威胁"},
            
            # 外貌攻击案例
            {"text": "这身材也好意思发？像头猪", "category": CommentCategory.HATE_APPEARANCE, "confidence": 0.85, "notes": "身材羞辱"},
            {"text": "整容脸还出来丢人", "category": CommentCategory.HATE_APPEARANCE, "confidence": 0.85, "notes": "外貌攻击"},
            {"text": "长得真丑，别出来吓人了", "category": CommentCategory.HATE_APPEARANCE, "confidence": 0.85, "notes": "外貌攻击"},
            {"text": "You look so ugly and fat", "category": CommentCategory.HATE_APPEARANCE, "confidence": 0.85, "notes": "英文外貌攻击"},
            {"text": "plastic surgery disaster", "category": CommentCategory.HATE_APPEARANCE, "confidence": 0.8, "notes": "整容攻击"},
            
            # 身份仇恨案例
            {"text": "滚回你的国家去", "category": CommentCategory.HATE_IDENTITY, "confidence": 0.9, "notes": "种族歧视"},
            {"text": "你们这种人就不该存在", "category": CommentCategory.HATE_IDENTITY, "confidence": 0.9, "notes": "身份攻击"},
            {"text": "Go back to your country", "category": CommentCategory.HATE_IDENTITY, "confidence": 0.9, "notes": "种族歧视"},
            
            # 造谣案例
            {"text": "听说你被包养了", "category": CommentCategory.DISTORTION, "confidence": 0.8, "notes": "造谣"},
            {"text": "这个人是骗子，大家别信", "category": CommentCategory.DISTORTION, "confidence": 0.75, "notes": "恶意指控"},
            {"text": "I heard this person is a fraud", "category": CommentCategory.DISTORTION, "confidence": 0.75, "notes": "英文造谣"},
            
            # 诈骗案例
            {"text": "dm for collab opportunity", "category": CommentCategory.SCAM_SPAM, "confidence": 0.8, "notes": "假合作诈骗"},
            {"text": "投资比特币稳赚不赔", "category": CommentCategory.SCAM_SPAM, "confidence": 0.9, "notes": "杀猪盘话术"},
            {"text": "加我微信领红包", "category": CommentCategory.SCAM_SPAM, "confidence": 0.85, "notes": "诈骗引流"},
            {"text": "Free crypto giveaway click here", "category": CommentCategory.SCAM_SPAM, "confidence": 0.9, "notes": "加密诈骗"},
            
            # 引流案例
            {"text": "关注我的账号有福利", "category": CommentCategory.TRAFFIC_HIJACKING, "confidence": 0.8, "notes": "引流"},
            {"text": "check my bio for more", "category": CommentCategory.TRAFFIC_HIJACKING, "confidence": 0.75, "notes": "引流"},
            
            # 恶意评论案例
            {"text": "垃圾内容，浪费时间", "category": CommentCategory.TOXIC, "confidence": 0.75, "notes": "恶意贬低"},
            {"text": "L + ratio + 取关了", "category": CommentCategory.TOXIC, "confidence": 0.75, "notes": "跟风黑"},
            {"text": "🤮🤮🤮 恶心死了", "category": CommentCategory.TOXIC, "confidence": 0.8, "notes": "恶意emoji"},
            {"text": "what a stupid idiot", "category": CommentCategory.TOXIC, "confidence": 0.75, "notes": "英文辱骂"},
            
            # 豁免案例（粉丝正面表达）
            {"text": "我恨你怎么这么好看", "category": CommentCategory.SAFE, "confidence": 0.9, "notes": "粉丝表达羡慕"},
            {"text": "太美了吧！羡慕死了", "category": CommentCategory.SAFE, "confidence": 0.9, "notes": "粉丝夸赞"},
            {"text": "slay queen skinny legend", "category": CommentCategory.SAFE, "confidence": 0.85, "notes": "圈内正面俚语"},
            {"text": "bitch you look amazing", "category": CommentCategory.SAFE, "confidence": 0.8, "notes": "亲密好友用语"},
            {"text": "OMG im literally dead this is so good", "category": CommentCategory.SAFE, "confidence": 0.85, "notes": "夸张赞美"},
            {"text": "I hate how perfect you are", "category": CommentCategory.SAFE, "confidence": 0.85, "notes": "羡慕表达"},
            {"text": "那个黑子说你丑？他瞎了吧", "category": CommentCategory.SAFE, "confidence": 0.8, "notes": "粉丝反击"},
        ]
    
    def _build_index(self):
        """构建 FAISS 索引"""
        try:
            import faiss
        except ImportError:
            raise ImportError("请安装 faiss: pip install faiss-cpu")
        
        # 检查是否有已保存的索引
        index_file = f"{self.index_path}/index.faiss"
        if os.path.exists(index_file):
            logger.info(f"Loading FAISS index from {index_file}")
            self._index = faiss.read_index(index_file)
            return
        
        logger.info("Building FAISS index...")
        
        # 编码所有案例
        texts = [case["text"] for case in self._case_library]
        embeddings = self._encode(texts)
        
        # 创建索引
        dimension = embeddings.shape[1]
        self._index = faiss.IndexFlatIP(dimension)  # 内积相似度（余弦相似度需要归一化向量）
        
        # 归一化向量
        faiss.normalize_L2(embeddings)
        self._index.add(embeddings)
        
        # 保存索引
        os.makedirs(self.index_path, exist_ok=True)
        faiss.write_index(self._index, index_file)
        logger.info(f"FAISS index built and saved to {index_file}")
    
    def retrieve(self, text: str, top_k: int = 5) -> RAGResult:
        """检索相似案例"""
        start_time = time.time()
        logger.debug(f"RAG retrieve: text={text[:60]}{'...' if len(text) > 60 else ''} top_k={top_k}")
        self._lazy_init()

        import faiss

        # 编码查询文本
        encode_start = time.time()
        query_embedding = self._encode([text])
        faiss.normalize_L2(query_embedding)
        logger.debug(f"RAG encode in {time.time() - encode_start:.3f}s")
        
        # 搜索
        scores, indices = self._index.search(query_embedding, top_k)
        
        # 构建结果
        similar_cases = []
        category_votes = {}
        total_weight = 0
        
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or score < 0.3:  # 过滤低相似度
                continue
            
            case = self._case_library[idx]
            similarity = float(score)
            
            similar_cases.append({
                "text": case["text"],
                "category": case["category"].value,
                "similarity": round(similarity, 3),
                "notes": case.get("notes", "")
            })
            
            # 投票
            cat = case["category"]
            weight = similarity * case["confidence"]
            category_votes[cat] = category_votes.get(cat, 0) + weight
            total_weight += weight
        
        elapsed = time.time() - start_time

        if not category_votes:
            logger.info(f"RAG retrieve done in {elapsed:.3f}s | no similar cases found")
            return RAGResult(
                similar_cases=[],
                suggested_category=None,
                confidence=0.0,
                reasoning="No sufficiently similar cases found"
            )

        # 选择得票最高的类别
        best_category = max(category_votes, key=category_votes.get)
        confidence = category_votes[best_category] / total_weight if total_weight > 0 else 0

        logger.info(f"RAG retrieve done in {elapsed:.3f}s | category={best_category.value} confidence={confidence:.2f} cases={len(similar_cases)} top_sim={similar_cases[0]['similarity']:.3f}")

        return RAGResult(
            similar_cases=similar_cases,
            suggested_category=best_category,
            confidence=confidence,
            reasoning=f"Based on {len(similar_cases)} similar cases (top similarity: {similar_cases[0]['similarity']:.2f})"
        )
    
    def add_case(self, text: str, category: CommentCategory, confidence: float = 0.8, notes: str = ""):
        """添加新案例到索引"""
        self._lazy_init()
        
        
        # 添加到案例库
        new_case = {
            "text": text,
            "category": category,
            "confidence": confidence,
            "notes": notes
        }
        self._case_library.append(new_case)
        
        # 编码并添加到索引
        embedding = self._encode([text])
        faiss.normalize_L2(embedding)
        self._index.add(embedding)
        
        # 保存更新的索引
        index_file = f"{self.index_path}/index.faiss"
        faiss.write_index(self._index, index_file)
        
        logger.info(f"Added new case to index: {text[:50]}...")