"""
평가 데이터셋 생성 도구

실제 RFP 문서에서 질문-답변 쌍을 만들어
Ground Truth가 있는 평가 데이터셋을 생성합니다.

사용법:
    python create_eval_dataset.py --input data/rag_chunks_final.csv --output data/eval_dataset.json
"""

import json
import csv
import argparse
from pathlib import Path
from typing import List, Dict, Any


class EvalDatasetCreator:
    """평가 데이터셋 생성 클래스"""
    
    def __init__(self):
        self.dataset = {
            "metadata": {
                "version": "1.0",
                "description": "RFPilot 평가 데이터셋",
                "created_by": "manual_annotation"
            },
            "in_distribution": [],
            "out_distribution": []
        }
    
    def add_in_distribution_sample(
        self,
        query: str,
        expected_answer: str,
        category: str,
        source_doc: str = None,
        metadata: Dict[str, Any] = None
    ):
        """In-Distribution 샘플 추가"""
        sample = {
            "query": query,
            "expected_answer": expected_answer,
            "category": category,
            "expected_type": "document",
            "source_doc": source_doc,
            "metadata": metadata or {}
        }
        self.dataset["in_distribution"].append(sample)
    
    def add_out_distribution_sample(
        self,
        query: str,
        expected_answer: str,
        category: str,
        metadata: Dict[str, Any] = None
    ):
        """Out-Distribution 샘플 추가"""
        sample = {
            "query": query,
            "expected_answer": expected_answer,
            "category": category,
            "expected_type": "out_of_scope",
            "metadata": metadata or {}
        }
        self.dataset["out_distribution"].append(sample)
    
    def create_template_dataset(self):
        """템플릿 데이터셋 생성 (수동 작성용)"""
        print("📝 템플릿 데이터셋 생성 중...")
        
        # In-Distribution 템플릿
        in_dist_templates = [
            {
                "query": "사업 제안서 제출 마감일은 언제인가요?",
                "expected_answer": "2024년 3월 15일까지입니다.",  # 실제 문서에서 추출
                "category": "deadline",
                "source_doc": "RFP_2024_001.hwp",
                "metadata": {"difficulty": "easy"}
            },
            {
                "query": "제안 요청서의 제출 서류는 무엇인가요?",
                "expected_answer": "기술제안서, 가격제안서, 사업자등록증, 회사소개서가 필요합니다.",
                "category": "requirements",
                "source_doc": "RFP_2024_001.hwp",
                "metadata": {"difficulty": "medium"}
            },
            {
                "query": "사업 예산 규모는 얼마인가요?",
                "expected_answer": "총 5억원입니다.",
                "category": "budget",
                "source_doc": "RFP_2024_002.hwp",
                "metadata": {"difficulty": "easy"}
            },
        ]
        
        # Out-Distribution 템플릿
        out_dist_templates = [
            {
                "query": "한국의 수도는 어디인가요?",
                "expected_answer": "서울입니다.",
                "category": "general_knowledge",
                "metadata": {"difficulty": "easy"}
            },
            {
                "query": "파이썬에서 리스트와 튜플의 차이는 무엇인가요?",
                "expected_answer": "리스트는 가변(mutable)이고, 튜플은 불변(immutable)입니다.",
                "category": "programming",
                "metadata": {"difficulty": "medium"}
            },
        ]
        
        # 데이터셋에 추가
        for sample in in_dist_templates:
            self.add_in_distribution_sample(**sample)
        
        for sample in out_dist_templates:
            self.add_out_distribution_sample(**sample)
        
        print(f"✅ 템플릿 생성 완료")
        print(f"   - In-Distribution: {len(in_dist_templates)}개")
        print(f"   - Out-Distribution: {len(out_dist_templates)}개")
        print(f"\n⚠️ 이 템플릿을 수정하여 실제 데이터를 채워주세요!")
    
    def load_from_csv(self, csv_path: str):
        """CSV에서 데이터셋 로드"""
        print(f"📥 CSV 로드 중: {csv_path}")
        
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                distribution = row.get('distribution', 'in_distribution')
                
                if distribution == 'in_distribution':
                    self.add_in_distribution_sample(
                        query=row['query'],
                        expected_answer=row['expected_answer'],
                        category=row['category'],
                        source_doc=row.get('source_doc'),
                        metadata=json.loads(row.get('metadata', '{}'))
                    )
                else:
                    self.add_out_distribution_sample(
                        query=row['query'],
                        expected_answer=row['expected_answer'],
                        category=row['category'],
                        metadata=json.loads(row.get('metadata', '{}'))
                    )
        
        print(f"✅ CSV 로드 완료")
    
    def save_json(self, output_path: str):
        """JSON 형식으로 저장"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.dataset, f, ensure_ascii=False, indent=2)
        
        print(f"💾 저장 완료: {output_path}")
    
    def save_csv_template(self, output_path: str):
        """수동 작성용 CSV 템플릿 저장"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'distribution', 'query', 'expected_answer', 
                'category', 'source_doc', 'metadata'
            ])
            writer.writeheader()
            
            # In-Distribution 예시
            writer.writerow({
                'distribution': 'in_distribution',
                'query': '사업 제안서 제출 마감일은 언제인가요?',
                'expected_answer': '2024년 3월 15일까지입니다.',
                'category': 'deadline',
                'source_doc': 'RFP_2024_001.hwp',
                'metadata': '{"difficulty": "easy"}'
            })
            
            # Out-Distribution 예시
            writer.writerow({
                'distribution': 'out_distribution',
                'query': '한국의 수도는 어디인가요?',
                'expected_answer': '서울입니다.',
                'category': 'general_knowledge',
                'source_doc': '',
                'metadata': '{"difficulty": "easy"}'
            })
        
        print(f"📄 CSV 템플릿 저장: {output_path}")
        print(f"   → 이 파일을 수정하여 실제 데이터를 채워주세요!")
    
    def print_summary(self):
        """데이터셋 요약 출력"""
        print("\n" + "="*60)
        print("데이터셋 요약")
        print("="*60)
        print(f"In-Distribution: {len(self.dataset['in_distribution'])}개")
        print(f"Out-Distribution: {len(self.dataset['out_distribution'])}개")
        print(f"총 샘플: {len(self.dataset['in_distribution']) + len(self.dataset['out_distribution'])}개")
        print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(description='평가 데이터셋 생성')
    parser.add_argument('--mode', choices=['template', 'csv'], default='template',
                        help='생성 모드: template (템플릿 생성) 또는 csv (CSV에서 로드)')
    parser.add_argument('--input', type=str, help='입력 CSV 파일 경로')
    parser.add_argument('--output', type=str, default='data/eval_dataset.json',
                        help='출력 JSON 파일 경로')
    parser.add_argument('--csv-template', type=str, default='data/eval_template.csv',
                        help='CSV 템플릿 저장 경로')
    
    args = parser.parse_args()
    
    creator = EvalDatasetCreator()
    
    if args.mode == 'template':
        print("📝 템플릿 모드")
        creator.create_template_dataset()
        creator.save_json(args.output)
        creator.save_csv_template(args.csv_template)
    
    elif args.mode == 'csv':
        if not args.input:
            print("❌ CSV 모드에서는 --input 옵션이 필요합니다.")
            return
        
        print("📥 CSV 모드")
        creator.load_from_csv(args.input)
        creator.save_json(args.output)
    
    creator.print_summary()
    
    print("\n✅ 완료!")
    print(f"\n다음 단계:")
    print(f"1. {args.csv_template} 파일을 열어서 실제 데이터 작성")
    print(f"2. python create_eval_dataset.py --mode csv --input {args.csv_template} --output {args.output}")
    print(f"3. 생성된 {args.output}을 실험에 사용")


if __name__ == "__main__":
    main()