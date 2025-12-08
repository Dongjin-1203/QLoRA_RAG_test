import json
from pathlib import Path
from typing import Dict, List, Any

class EvalDataset:
    def __init__(self, dataset_path: str = "data/eval_dataset.json"):
        self.dataset_path = Path(dataset_path)
        self.data = self._load_dataset()
    
    def _load_dataset(self) -> Dict[str, Any]:
        with open(self.dataset_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def get_in_distribution(self) -> List[Dict[str, Any]]:
        return self.data.get('in_distribution', [])
    
    def get_out_distribution(self) -> List[Dict[str, Any]]:
        return self.data.get('out_distribution', [])
    
    def get_all_queries(self) -> Dict[str, List[Dict[str, Any]]]:
        return {
            'in_distribution': self.get_in_distribution(),
            'out_distribution': self.get_out_distribution()
        }
    
    def print_summary(self):
        print(f"In-Distribution: {len(self.get_in_distribution())}개")
        print(f"Out-Distribution: {len(self.get_out_distribution())}개")
    
    def print_samples(self, n: int = 3):
        print("\n[In-Distribution 샘플]")
        for item in self.get_in_distribution()[:n]:
            print(f"  - {item['query']}")