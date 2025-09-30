from datasets import load_dataset, Dataset
import json
from typing import Dict, List, Any
import torch
from torch.utils.data import DataLoader

class StrategyQADataset:
    """StrategyQA数据集处理类"""
    
    def __init__(self, split: str = "train_filtered"):
        """
        初始化数据集
        Args:
            split: 数据集分割 ("train", "train_filtered", "train_paragraphs", "test")
        """
        self.split = split
        self.dataset = self._load_dataset()
        self.processed_data = None
        
    def _load_dataset(self) -> Dataset:
        """加载原始数据集"""
        if self.split == "train_filtered":
            url = "https://huggingface.co/datasets/voidful/StrategyQA/resolve/main/strategyqa_train_filtered.json"
        elif self.split == "train":
            url = "https://huggingface.co/datasets/voidful/StrategyQA/resolve/main/strategyqa_train.json"
        elif self.split == "train_paragraphs":
            url = "https://huggingface.co/datasets/voidful/StrategyQA/resolve/main/strategyqa_train_paragraphs.json"
        else:
            raise ValueError(f"Unsupported split: {self.split}")
            
        dataset = load_dataset("json", data_files=url)
        return dataset["train"]
    
    def preprocess(self, max_samples: int = None) -> List[Dict[str, Any]]:
        """
        预处理数据
        Args:
            max_samples: 最大样本数，用于快速实验
        Returns:
            处理后的数据列表
        """
        processed = []
        
        for i, sample in enumerate(self.dataset):
            if max_samples and i >= max_samples:
                break
                
            # 构建标准化的样本
            processed_sample = {
                "id": sample["qid"],
                "question": sample["question"],
                "answer": sample["answer"],  # True/False
                "answer_text": "Yes" if sample["answer"] else "No",
                "term": sample.get("term", ""),
                "description": sample.get("description", ""),
                "facts": sample.get("facts", []) if "facts" in sample else []
            }
            processed.append(processed_sample)
            
        self.processed_data = processed
        return processed
    
    def get_dataloader(self, batch_size: int = 8, shuffle: bool = True) -> DataLoader:
        """获取PyTorch DataLoader"""
        if self.processed_data is None:
            self.preprocess()
            
        return DataLoader(
            self.processed_data,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=self._collate_fn
        )
    
    def _collate_fn(self, batch: List[Dict]) -> Dict[str, Any]:
        """自定义批处理函数"""
        return {
            "ids": [item["id"] for item in batch],
            "questions": [item["question"] for item in batch],
            "answers": [item["answer"] for item in batch],
            "answer_texts": [item["answer_text"] for item in batch],
            "terms": [item["term"] for item in batch],
            "descriptions": [item["description"] for item in batch],
            "facts": [item["facts"] for item in batch]
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取数据集统计信息"""
        if self.processed_data is None:
            self.preprocess()
            
        total_samples = len(self.processed_data)
        yes_count = sum(1 for item in self.processed_data if item["answer"])
        no_count = total_samples - yes_count
        
        return {
            "total_samples": total_samples,
            "yes_answers": yes_count,
            "no_answers": no_count,
            "yes_ratio": yes_count / total_samples,
            "avg_question_length": sum(len(item["question"]) for item in self.processed_data) / total_samples
        }

# 使用示例
if __name__ == "__main__":
    # 创建数据集实例
    dataset = StrategyQADataset(split="train_filtered")
    
    # 预处理数据（使用小样本进行快速测试）
    processed_data = dataset.preprocess(max_samples=100)
    
    # 打印统计信息
    stats = dataset.get_statistics()
    print("数据集统计信息:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # 查看第一个样本
    print(f"\n第一个样本:")
    print(f"  问题: {processed_data[0]['question']}")
    print(f"  答案: {processed_data[0]['answer_text']}")
    print(f"  术语: {processed_data[0]['term']}")
    
    # 获取DataLoader
    dataloader = dataset.get_dataloader(batch_size=4)
    print(f"\nDataLoader批次数: {len(dataloader)}")
    
    # 查看一个批次
    for batch in dataloader:
        print(f"\n批次示例:")
        print(f"  问题数量: {len(batch['questions'])}")
        print(f"  第一个问题: {batch['questions'][0]}")
        break