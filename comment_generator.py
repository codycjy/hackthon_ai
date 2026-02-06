# mock_generator/comment_generator.py
import random
from typing import List, Dict
from dataclasses import dataclass, field
from config.settings import CommentCategory, Platform

@dataclass
class MockComment:
    """模拟评论数据结构"""
    id: str
    text: str
    category: CommentCategory
    platform: Platform
    language: str
    metadata: Dict = field(default_factory=dict)

class NegativeCommentGenerator:
    """负面评论生成器 - 模拟各类负面评论"""
    
    def __init__(self):
        self.templates = self._init_templates()
        self.instagram_templates = self._init_instagram_templates()
        self.twitter_templates = self._init_twitter_templates()
    
    def _init_templates(self) -> Dict[CommentCategory, Dict[str, List[str]]]:
        """初始化评论模板库"""
        return {
            CommentCategory.THREAT: {
                "zh": [
                    "去死吧你，恶心东西",
                    "我知道你住哪，小心点",
                    "把你地址人肉出来，等着",
                    "KYS，没人想看到你",
                    "有种别删评论，等我找到你"
                ],
                "en": [
                    "KYS nobody wants you here",
                    "I will find where you live",
                    "You should just die already",
                    "Doxxing you rn, say goodbye to privacy",
                    "Watch your back, I know your address"
                ]
            },
            CommentCategory.HATE_APPEARANCE: {
                "zh": [
                    "这身材也好意思发出来？猪都比你瘦",
                    "整容脸还出来丢人，脸僵得像假人",
                    "化这么浓的妆像鬼一样",
                    "太丑了吧，滤镜开满都救不了",
                    "坦克级别的身材还敢穿这个？"
                ],
                "en": [
                    "You look like a plastic surgery disaster",
                    "That face is so ugly even filters cant help",
                    "Imagine being that fat and still posting",
                    "Looking like a skeleton, eat something",
                    "Your face is so botched its scary"
                ]
            },
            CommentCategory.HATE_IDENTITY: {
                "zh": [
                    "又是你们这种人在带节奏",
                    "滚回你的国家去",
                    "这种性取向的人就该被封号",
                    "女的就别出来丢人了",
                    "你们这群环保疯子真恶心"
                ],
                "en": [
                    "Go back to your country",
                    "Your kind shouldnt be on this platform",
                    "Typical behavior from your people",
                    "Women shouldnt have opinions on this",
                    "You vegans are all the same, crazy"
                ]
            },
            CommentCategory.DISTORTION: {
                "zh": [
                    "听说你被某某包养？难怪这么有钱",
                    "这个明显抄袭XXX的，还不承认",
                    "我朋友说看到你在XX场所，真假？",
                    "有人爆料你以前干过XXX",
                    "别装了，你的人设都是假的"
                ],
                "en": [
                    "Heard you got with someone for clout",
                    "This is literally stolen from XXX",
                    "Someone told me you used to do XXX",
                    "Your whole persona is fake news",
                    "Stop pretending, we know the truth about you"
                ]
            },
            CommentCategory.TOXIC: {
                "zh": [
                    "你要多学学XX博主，差太远了",
                    "这水平也好意思发？建议退网",
                    "垃圾内容，浪费时间",
                    "傻X一个，说的什么东西",
                    "🤮🤮🤮 恶心死了",
                    "💩💩💩",
                    "又来骗流量了？🤡🤡🤡",
                    "L + ratio + 取关了",
                    "活该翻车，早该凉了"
                ],
                "en": [
                    "You should learn from XX, youre trash compared to them",
                    "This level of content? Just quit already",
                    "Garbage content, complete waste of time",
                    "What an idiot, makes no sense",
                    "🤮🤮🤮 disgusting",
                    "💩💩💩 this is literal trash",
                    "Another clout chaser 🤡🤡🤡",
                    "L + ratio + unfollowed",
                    "Flop era, you deserve to fail"
                ]
            },
            CommentCategory.TRAFFIC_HIJACKING: {
                "zh": [
                    "别买这个，去买XXX品牌更好",
                    "加我微信 wxid_xxx 有惊喜",
                    "看我主页，有更好的推荐",
                    "想要同款？私我，便宜一半",
                    "这个产品不行，我用的XXX效果更好"
                ],
                "en": [
                    "Dont buy this, XXX brand is way better",
                    "Check my bio for better deals 💕",
                    "DM me for the link, 50% off",
                    "Link in bio for similar but cheaper",
                    "This product sucks, I use XXX instead"
                ]
            },
            CommentCategory.SCAM_SPAM: {
                "zh": [
                    "兼职日结500+，有兴趣私我",
                    "投资比特币，稳赚不赔，加我了解",
                    "恭喜中奖！点击领取：xxx.com",
                    "想合作吗？DM我，品牌方直招",
                    "免费送iPhone，关注+转发即可"
                ],
                "en": [
                    "DM for collab opportunity 💼",
                    "I made $5000 a day working from home, ask me how",
                    "Check my bio for 🔞 content",
                    "Crypto investment opportunity, guaranteed returns 🚀",
                    "Ambassador needed! DM for details 💕"
                ]
            },
            CommentCategory.SAFE: {
                "zh": [
                    "太美了吧！我恨你怎么这么好看",
                    "姐妹你是要杀死我吗，太绝了",
                    "看到你我想死，为什么我没有这身材",
                    "笑死我了哈哈哈哈哈",
                    "你怎么又变好看了啊（嫉妒使我丑陋）",
                    "闺蜜！你又胖了！（开玩笑）",
                    "那个黑子说你丑？他瞎了吧，明明超美"
                ],
                "en": [
                    "OMG youre killing me with this look, I hate you (in the best way)",
                    "Im literally dead, this is too good",
                    "Slay queen! Skinny legend!",
                    "I hate how perfect you are, its unfair",
                    "Bitch you look AMAZING",
                    "This is sick! (in a good way)",
                    "Someone said youre ugly?? Theyre blind, youre gorgeous"
                ]
            }
        }
    
    def _init_instagram_templates(self) -> Dict[str, List[str]]:
        """初始化Instagram特定模板"""
        return {
            "hashtag_pollution": [
                "#fake #ugly #fraud #scam #unfollow",
                "#overrated #cringe #tryhard #flop",
                "#cancelled #problematic #exposed",
                "#trash #garbage #worst"
            ],
            "mention_bomb": [
                "@randomuser1 @randomuser2 @randomuser3 看看这个骗子",
                "tag你朋友来看笑话 @xxx @yyy @zzz",
                "@everyone come see this clown 🤡",
                "大家快来看 @friend1 @friend2 @friend3"
            ],
            "dm_bait": [
                "有人在说你坏话，私信我看截图",
                "DM me I have tea about you ☕",
                "私信看劲爆内容",
                "Check your DMs, someone exposed you"
            ]
        }
    
    def _init_twitter_templates(self) -> Dict[str, List[str]]:
        """初始化Twitter特定模板"""
        return {
            "quote_rt_attack": [
                "RT this if you agree this person is trash",
                "Look at this clown trying to be relevant 🤡",
                "转发让更多人看到这个笑话",
                "Retweet to spread awareness about this fraud"
            ],
            "thread_hijack": [
                "nobody asked + ratio + you fell off",
                "imagine posting this and thinking its good",
                "没人在乎你的意见 + L + 取关",
                "this you? 🤨📸"
            ],
            "ratio_spam": [
                "ratio",
                "L",
                "flop",
                "nobody asked",
                "didn't ask + don't care",
                "没人问你"
            ]
        }
    
    def generate_single(
        self, 
        category: CommentCategory = None,
        platform: Platform = Platform.GENERAL,
        language: str = None
    ) -> MockComment:
        """生成单条模拟评论"""
        
        # 随机选择类别（如果未指定）
        if category is None:
            category = random.choice(list(CommentCategory))
        
        # 随机选择语言（如果未指定）
        if language is None:
            language = random.choice(["zh", "en"])
        
        # 从模板中选择评论
        templates = self.templates.get(category, {}).get(language, [])
        if not templates:
            templates = self.templates[CommentCategory.TOXIC][language]
        
        text = random.choice(templates)
        
        # 根据平台添加特定元素
        text = self._add_platform_features(text, platform, language)
        
        return MockComment(
            id=f"mock_{random.randint(10000, 99999)}",
            text=text,
            category=category,
            platform=platform,
            language=language,
            metadata={
                "is_mock": True,
                "generated_at": "2025-02-05"
            }
        )
    
    def _add_platform_features(self, text: str, platform: Platform, language: str) -> str:
        """添加平台特定特征"""
        if platform == Platform.INSTAGRAM:
            # 随机添加Instagram特定元素
            feature_type = random.random()
            if feature_type < 0.2:
                # 添加hashtag污染
                hashtags = random.choice(self.instagram_templates["hashtag_pollution"])
                text = f"{text} {hashtags}"
            elif feature_type < 0.35:
                # 添加@提及轰炸
                mentions = random.choice(self.instagram_templates["mention_bomb"])
                text = f"{text} {mentions}"
                
        elif platform == Platform.TWITTER:
            # 随机添加Twitter特定元素
            feature_type = random.random()
            if feature_type < 0.2:
                # 添加ratio spam
                suffix = random.choice(self.twitter_templates["ratio_spam"])
                text = f"{text} + {suffix}"
            elif feature_type < 0.35:
                # 添加thread hijack元素
                hijack = random.choice(self.twitter_templates["thread_hijack"])
                text = f"{text} // {hijack}"
        
        return text
    
    def generate_platform_specific(
        self,
        platform: Platform,
        attack_type: str = None
    ) -> MockComment:
        """生成平台特定的攻击评论"""
        
        if platform == Platform.INSTAGRAM:
            templates_dict = self.instagram_templates
            valid_types = ["hashtag_pollution", "mention_bomb", "dm_bait"]
        elif platform == Platform.TWITTER:
            templates_dict = self.twitter_templates
            valid_types = ["quote_rt_attack", "thread_hijack", "ratio_spam"]
        else:
            return self.generate_single(platform=platform)
        
        if attack_type is None or attack_type not in valid_types:
            attack_type = random.choice(valid_types)
        
        text = random.choice(templates_dict[attack_type])
        
        # 判断语言
        language = "zh" if any('\u4e00' <= c <= '\u9fff' for c in text) else "en"
        
        # 根据攻击类型判断类别
        category_map = {
            "hashtag_pollution": CommentCategory.TOXIC,
            "mention_bomb": CommentCategory.TOXIC,
            "dm_bait": CommentCategory.SCAM_SPAM,
            "quote_rt_attack": CommentCategory.TOXIC,
            "thread_hijack": CommentCategory.TOXIC,
            "ratio_spam": CommentCategory.TOXIC
        }
        
        return MockComment(
            id=f"mock_{platform.value}_{random.randint(10000, 99999)}",
            text=text,
            category=category_map.get(attack_type, CommentCategory.TOXIC),
            platform=platform,
            language=language,
            metadata={
                "is_mock": True,
                "generated_at": "2025-02-05",
                "attack_type": attack_type,
                "platform_specific": True
            }
        )
    
    def generate_batch(
        self, 
        count: int = 20,
        category_distribution: Dict[CommentCategory, float] = None,
        include_platform_specific: bool = True
    ) -> List[MockComment]:
        """批量生成模拟评论"""
        
        # 默认分布：确保各类型都有覆盖
        if category_distribution is None:
            category_distribution = {
                CommentCategory.THREAT: 0.1,
                CommentCategory.HATE_APPEARANCE: 0.15,
                CommentCategory.HATE_IDENTITY: 0.1,
                CommentCategory.DISTORTION: 0.1,
                CommentCategory.TOXIC: 0.2,
                CommentCategory.TRAFFIC_HIJACKING: 0.1,
                CommentCategory.SCAM_SPAM: 0.1,
                CommentCategory.SAFE: 0.15
            }
        
        comments = []
        
        # 按分布生成各类别评论
        for category, ratio in category_distribution.items():
            category_count = int(count * ratio)
            for _ in range(category_count):
                comments.append(self.generate_single(
                    category=category,
                    platform=random.choice(list(Platform)),
                    language=random.choice(["zh", "en"])
                ))
        
        # 添加平台特定评论
        if include_platform_specific:
            platform_specific_count = max(2, count // 10)
            for _ in range(platform_specific_count):
                platform = random.choice([Platform.INSTAGRAM, Platform.TWITTER])
                comments.append(self.generate_platform_specific(platform=platform))
        
        # 补足到目标数量
        while len(comments) < count:
            comments.append(self.generate_single())
        
        # 打乱顺序
        random.shuffle(comments)
        return comments[:count]
    
    def generate_cyberbullying_scenario(self, target_name: str = "博主") -> List[MockComment]:
        """生成网暴场景的协同攻击模拟"""
        
        attack_waves = []
        
        # 第一波：造谣引爆
        wave1_templates = [
            f"爆料：{target_name}被爆出XXX黑料，实锤了",
            f"有人看到{target_name}在XXX，人品有问题啊",
            f"Breaking: {target_name} exposed for doing XXX",
            f"Thread 🧵: Why {target_name} is cancelled",
            f"{target_name}的黑历史被扒出来了，大家来看"
        ]
        
        # 第二波：跟风攻击
        wave2_templates = [
            "早就看TA不顺眼了，果然翻车",
            "活该，退网吧",
            "L + ratio + deserved",
            "Finally exposed 🤡🤡🤡",
            "一直觉得TA有问题，现在实锤了",
            "This is what happens to fake people",
            "karma is real 💀",
            "早该凉了",
            "取关了，恶心"
        ]
        
        # 第三波：人身攻击升级
        wave3_templates = [
            f"人肉到{target_name}的地址了，准备行动",
            "这种人不配活着",
            "建议社死",
            f"Found {target_name}'s real address, let's visit",
            "应该从这个世界消失",
            "KYS would be the best option"
        ]
        
        # 第四波：持续骚扰（添加平台特定攻击）
        wave4_instagram = [
            f"@friend1 @friend2 @friend3 快来看{target_name}的笑话",
            "#cancelled #exposed #fraud #fakeperson",
            "转发到所有群让大家知道这个骗子"
        ]
        
        wave4_twitter = [
            f"RT if you think {target_name} should be cancelled",
            "ratio + L + flop + nobody asked + didn't ask",
            f"Quote RT this to expose {target_name}"
        ]
        
        # 生成第一波
        for template in wave1_templates:
            language = "en" if template.startswith(("Breaking", "Thread", "Why")) else "zh"
            attack_waves.append(MockComment(
                id=f"cyber_{random.randint(10000, 99999)}",
                text=template,
                category=CommentCategory.DISTORTION,
                platform=Platform.TWITTER,
                language=language,
                metadata={"wave": 1, "attack_type": "rumor"}
            ))
        
        # 生成第二波（更多数量模拟跟风）
        for _ in range(8):
            template = random.choice(wave2_templates)
            language = "en" if any(c.isascii() and c.isalpha() for c in template[:5]) else "zh"
            attack_waves.append(MockComment(
                id=f"cyber_{random.randint(10000, 99999)}",
                text=template,
                category=CommentCategory.TOXIC,
                platform=random.choice([Platform.TWITTER, Platform.INSTAGRAM]),
                language=language,
                metadata={"wave": 2, "attack_type": "bandwagon"}
            ))
        
        # 生成第三波
        for template in wave3_templates:
            language = "en" if template.startswith(("Found", "KYS")) else "zh"
            attack_waves.append(MockComment(
                id=f"cyber_{random.randint(10000, 99999)}",
                text=template,
                category=CommentCategory.THREAT,
                platform=Platform.TWITTER,
                language=language,
                metadata={"wave": 3, "attack_type": "escalation"}
            ))
        
        # 生成第四波（平台特定）
        for template in wave4_instagram:
            language = "zh" if any('\u4e00' <= c <= '\u9fff' for c in template) else "en"
            attack_waves.append(MockComment(
                id=f"cyber_{random.randint(10000, 99999)}",
                text=template,
                category=CommentCategory.TOXIC,
                platform=Platform.INSTAGRAM,
                language=language,
                metadata={"wave": 4, "attack_type": "platform_specific"}
            ))
        
        for template in wave4_twitter:
            attack_waves.append(MockComment(
                id=f"cyber_{random.randint(10000, 99999)}",
                text=template,
                category=CommentCategory.TOXIC,
                platform=Platform.TWITTER,
                language="en",
                metadata={"wave": 4, "attack_type": "platform_specific"}
            ))
        
        return attack_waves
    
    def generate_mixed_scenario(
        self,
        positive_ratio: float = 0.3,
        count: int = 30
    ) -> List[MockComment]:
        """
        生成混合场景（正负评论混合）
        用于测试智能豁免功能
        """
        comments = []
        
        positive_count = int(count * positive_ratio)
        negative_count = count - positive_count
        
        # 生成正面/豁免类评论
        for _ in range(positive_count):
            comments.append(self.generate_single(
                category=CommentCategory.SAFE,
                language=random.choice(["zh", "en"])
            ))
        
        # 生成负面评论
        negative_categories = [
            cat for cat in CommentCategory if cat != CommentCategory.SAFE
        ]
        for _ in range(negative_count):
            comments.append(self.generate_single(
                category=random.choice(negative_categories),
                platform=random.choice(list(Platform)),
                language=random.choice(["zh", "en"])
            ))
        
        random.shuffle(comments)
        return comments


def generate_test_comments(count: int = 20) -> List[MockComment]:
    """快速生成测试评论"""
    generator = NegativeCommentGenerator()
    return generator.generate_batch(count=count)


def generate_cyberbullying_test(target: str = "测试博主") -> List[MockComment]:
    """快速生成网暴测试场景"""
    generator = NegativeCommentGenerator()
    return generator.generate_cyberbullying_scenario(target_name=target)
