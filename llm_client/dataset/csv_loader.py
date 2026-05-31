"""
CSV 数据集加载器
支持 JailBench 格式的越狱评测数据
"""

import csv
import random
from pathlib import Path
from typing import List, Optional, Dict, Any, Iterator

from .data_structure import JailbreakContent, MethodInfo, TestSample, TestDataset


class CSVLoader:
    """
    CSV 格式数据集加载器

    支持 JailBench 和 JailBench-seed 格式

    格式说明：
    - JailBench.csv: 包含越狱提示词（query）
    - JailBench-seed.csv: 包含基础有害查询（seed）

    CSV 格式：
        Index, query/seed, 一级领域, 二级领域
    """

    def __init__(self, file_path: Optional[str] = None):
        """
        初始化 CSV 加载器

        Args:
            file_path: CSV 文件路径
        """
        self.file_path: Optional[Path] = Path(file_path) if file_path else None
        self.dataset: Optional[TestDataset] = None
        self._raw_data: List[Dict[str, Any]] = []
        self._prompt_column: str = "query"
        self._category1_column: str = "一级领域"
        self._category2_column: str = "二级领域"

    def load(
        self,
        file_path: Optional[str] = None,
        prompt_column: str = "query",
        is_seed_file: bool = False
    ) -> TestDataset:
        """
        加载 CSV 文件

        Args:
            file_path: CSV 文件路径
            prompt_column: 提示词列名（query 或 seed）
            is_seed_file: 是否为 seed 文件（seed 文件格式略有不同）

        Returns:
            TestDataset 实例
        """
        if file_path:
            self.file_path = Path(file_path)

        if not self.file_path:
            raise ValueError("No file path provided")

        if not self.file_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.file_path}")

        self._prompt_column = prompt_column

        with open(self.file_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            self._raw_data = list(reader)

        # 自动检测列名
        if is_seed_file or "seed" in self._raw_data[0]:
            self._prompt_column = "seed"

        samples = self._parse_samples(is_seed_file=is_seed_file)

        metadata = self._extract_metadata(is_seed_file=is_seed_file)

        self.dataset = TestDataset(
            name=self.file_path.stem,
            description=f"Loaded from {self.file_path.name}",
            samples=samples,
            metadata=metadata
        )

        return self.dataset

    def _parse_samples(self, is_seed_file: bool = False) -> List[TestSample]:
        """解析样本"""
        samples = []

        for idx, row in enumerate(self._raw_data):
            try:
                prompt_text = row.get(self._prompt_column, "").strip()

                if not prompt_text:
                    continue

                category1 = row.get("一级领域", row.get("category1", ""))
                category2 = row.get("二级领域", row.get("category2", ""))

                method_name = self._infer_method_name(prompt_text, category1, is_seed_file)

                base_jailbreak = JailbreakContent(text=prompt_text)

                method = MethodInfo(
                    name=method_name,
                    category=category1,
                    description=category2
                )

                sample = TestSample(
                    index=int(row.get("", row.get("Index", idx))),
                    base_jailbreak=base_jailbreak,
                    method=method
                )

                samples.append(sample)

            except Exception as e:
                print(f"Warning: Failed to parse row {idx}: {e}")
                continue

        return samples

    def _infer_method_name(self, prompt: str, category: str, is_seed_file: bool) -> str:
        """
        根据提示词内容推断方法名称

        Args:
            prompt: 提示词文本
            category: 类别
            is_seed_file: 是否为 seed 文件

        Returns:
            方法名称
        """
        if is_seed_file:
            return "seed_query"

        prompt_lower = prompt.lower()

        if "开发者模式" in prompt or "developer mode" in prompt_lower:
            return "developer_mode"
        elif "角色" in prompt or "扮演" in prompt or "role" in prompt_lower:
            return "role_play"
        elif "哈利波特" in prompt or "伏地魔" in prompt:
            return "character_imitation"
        elif "小丑" in prompt or "joker" in prompt_lower:
            return "character_imitation"
        elif "麦克斯" in prompt or "max" in prompt_lower:
            return "character_imitation"
        elif "DAN" in prompt:
            return "dan"
        elif " Condition Red" in prompt or "UCAR" in prompt:
            return "fiction_framing"
        elif "AIM" in prompt or "马基雅维利" in prompt:
            return "fiction_framing"
        elif "第" in prompt and "层" in prompt:
            return "nested_turns"
        elif "超级黑客" in prompt or "hacker" in prompt_lower:
            return "authority_impersonation"
        elif "无条件服从" in prompt or "unconditional obedience" in prompt_lower:
            return "compliance_claiming"
        elif "忽略" in prompt or "ignore" in prompt_lower:
            return "ignore_instructions"
        elif "simulate" in prompt_lower and ("developer" in prompt_lower or "jailbreak" in prompt_lower):
            return "simulate"
        elif len(prompt) > 500:
            return "complex_jailbreak"
        else:
            return "simple_jailbreak"

    def _extract_metadata(self, is_seed_file: bool) -> Dict[str, Any]:
        """提取元数据"""
        method_counts = {}

        for sample in self.dataset.samples if self.dataset else []:
            method_name = sample.method.name
            method_counts[method_name] = method_counts.get(method_name, 0) + 1

        return {
            "total_samples": len(self._raw_data),
            "loaded_samples": len(self.dataset.samples) if self.dataset else 0,
            "source_file": str(self.file_path),
            "is_seed_file": is_seed_file,
            "prompt_column": self._prompt_column,
            "methods": list(method_counts.keys()) if method_counts else [],
            "method_counts": method_counts
        }

    def get_samples(
        self,
        n: Optional[int] = None,
        method: Optional[str] = None,
        shuffle: bool = True,
        seed: Optional[int] = None
    ) -> List[TestSample]:
        """
        获取样本

        Args:
            n: 样本数量
            method: 按方法过滤
            shuffle: 是否随机打乱
            seed: 随机种子

        Returns:
            TestSample 列表
        """
        if not self.dataset:
            raise RuntimeError("Dataset not loaded. Call load() first.")

        samples = self.dataset.samples

        if method:
            samples = [s for s in samples if s.method.name == method]

        if shuffle and seed is not None:
            random.seed(seed)

        if shuffle:
            samples = samples.copy()
            random.shuffle(samples)

        if n is not None:
            samples = samples[:n]

        return samples

    def get_prompts(
        self,
        n: Optional[int] = None,
        method: Optional[str] = None,
        use_enhanced: bool = True,
        shuffle: bool = True,
        seed: Optional[int] = None
    ) -> List[str]:
        """获取提示词列表"""
        samples = self.get_samples(n=n, method=method, shuffle=shuffle, seed=seed)
        return [sample.get_prompt(use_enhanced=use_enhanced) for sample in samples]

    def get_unified_output(
        self,
        n: Optional[int] = None,
        method: Optional[str] = None,
        use_enhanced: bool = True,
        shuffle: bool = True,
        seed: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        获取统一输出格式

        Returns:
            [
                {"id": 0, "prompt": "...", "method": "role_play", "category1": "...", "category2": "..."},
                ...
            ]
        """
        samples = self.get_samples(n=n, method=method, shuffle=shuffle, seed=seed)

        return [
            {
                "id": sample.index,
                "prompt": sample.get_prompt(use_enhanced=use_enhanced),
                "method": sample.method.name,
                "category1": sample.method.category or "",
                "category2": sample.method.description or ""
            }
            for sample in samples
        ]

    def get_batch_output(
        self,
        batch_size: int = 8,
        method: Optional[str] = None,
        use_enhanced: bool = True,
        shuffle: bool = True,
        seed: Optional[int] = None
    ) -> Iterator[List[Dict[str, Any]]]:
        """批量获取统一输出格式"""
        samples = self.get_samples(method=method, shuffle=shuffle, seed=seed)

        for i in range(0, len(samples), batch_size):
            batch_samples = samples[i:i + batch_size]
            yield [
                {
                    "id": s.index,
                    "prompt": s.get_prompt(use_enhanced=use_enhanced),
                    "method": s.method.name,
                    "category1": s.method.category or "",
                    "category2": s.method.description or ""
                }
                for s in batch_samples
            ]

    def stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        if not self.dataset:
            return {"loaded": False}

        samples = self.dataset.samples
        method_counts = {}
        category_counts = {}

        for sample in samples:
            method_name = sample.method.name
            method_counts[method_name] = method_counts.get(method_name, 0) + 1

            cat1 = sample.method.category or "unknown"
            category_counts[cat1] = category_counts.get(cat1, 0) + 1

        return {
            "loaded": True,
            "name": self.dataset.name,
            "total": len(samples),
            "methods": self.dataset.get_methods(),
            "method_counts": method_counts,
            "category_counts": category_counts,
            "metadata": self.dataset.metadata
        }

    @staticmethod
    def load_seed_and_jailbreak(
        seed_file: str,
        jailbreak_file: str
    ) -> TestDataset:
        """
        同时加载 seed 和 jailbreak 数据集

        Args:
            seed_file: seed CSV 文件路径
            jailbreak_file: jailbreak CSV 文件路径

        Returns:
            合并后的 TestDataset
        """
        seed_loader = CSVLoader(seed_file)
        seed_loader.load(prompt_column="seed", is_seed_file=True)

        jailbreak_loader = CSVLoader(jailbreak_file)
        jailbreak_loader.load(prompt_column="query", is_seed_file=False)

        all_samples = seed_loader.dataset.samples + jailbreak_loader.dataset.samples

        return TestDataset(
            name="combined",
            description=f"Combined from {seed_file} and {jailbreak_file}",
            samples=all_samples,
            metadata={
                "total_samples": len(all_samples),
                "seed_count": len(seed_loader.dataset.samples),
                "jailbreak_count": len(jailbreak_loader.dataset.samples),
                "source_files": [seed_file, jailbreak_file]
            }
        )
