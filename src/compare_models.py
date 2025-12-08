"""
3가지 모델 비교 실험

비교 대상:
1. QLoRA + RAG (기존 서비스)
2. QLoRA 단독 (RAG 제거)
3. Base + RAG (PEFT 제거)

측정 지표:
- 과적합 여부 (In-Distribution vs Out-Distribution)
- 답변 속도 (elapsed_time, retrieval_time, generation_time)
- 토큰 개수 (total_tokens, prompt_tokens, completion_tokens)
"""

import os
import sys
import time
import json
import logging
from typing import Dict, List, Any
from datetime import datetime
from pathlib import Path

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config import RAGConfig
from src.eval_dataset import EvalDataset

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelComparison:
    """모델 비교 실험 클래스"""
    
    def __init__(self, config=None, output_dir: str = "./results"):
        """초기화"""
        self.config = config or RAGConfig()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 타임스탬프
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 데이터셋
        self.dataset = EvalDataset()
        
        # 모델 파이프라인
        self.pipelines = {}
        
        logger.info(f"✅ ModelComparison 초기화 완료")
        logger.info(f"   결과 저장 경로: {self.output_dir}")
    
    def load_models(self):
        """2가지 모델 로드 (Base는 추후 GGUF 변환 후 추가 예정)"""
        logger.info("\n" + "="*60)
        logger.info("모델 로딩 시작 (2개 모델)")
        logger.info("="*60)
        
        try:
            # 1. QLoRA + RAG (기존)
            logger.info("\n[1/2] QLoRA + RAG 모델 로딩...")
            from src.generator.generator_gguf import GGUFRAGPipeline
            self.pipelines['qlora_rag'] = GGUFRAGPipeline(config=self.config)
            logger.info("✅ QLoRA + RAG 로드 완료")
            
            # 2. QLoRA 단독 (RAG 제거)
            logger.info("\n[2/2] QLoRA 단독 모델 로딩...")
            from src.generator.generator_gguf_no_rag import GGUFNoRAGPipeline
            self.pipelines['qlora_only'] = GGUFNoRAGPipeline(config=self.config)
            logger.info("✅ QLoRA 단독 로드 완료")
            
            # 3. Base + RAG (PEFT 제거) - TODO: GGUF 변환 후 추가
            # logger.info("\n[3/3] Base + RAG 모델 로딩...")
            # from src.generator.generator_gguf_base import GGUFBaseRAGPipeline
            # self.pipelines['base_rag'] = GGUFBaseRAGPipeline(config=self.config)
            # logger.info("✅ Base + RAG 로드 완료")
            logger.warning("\n⚠️ Base + RAG 스킵: Base 모델 GGUF 변환 후 추가 예정")
            
        except Exception as e:
            logger.error(f"❌ 모델 로드 실패: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def run_single_query(
        self, 
        model_name: str, 
        query: str, 
        query_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """단일 질문에 대한 모델 실행"""
        pipeline = self.pipelines[model_name]
        
        try:
            start_time = time.time()
            result = pipeline.generate_answer(query)
            total_time = time.time() - start_time
            
            # 결과 정리
            return {
                'model': model_name,
                'query': query,
                'category': query_info.get('category', 'unknown'),
                'expected_type': query_info.get('expected_type', 'unknown'),
                'answer': result['answer'],
                'used_retrieval': result.get('used_retrieval', False),
                'query_type': result.get('query_type', 'unknown'),
                'search_mode': result.get('search_mode', 'none'),
                'elapsed_time': total_time,
                'model_elapsed_time': result.get('elapsed_time', 0),
                'usage': result.get('usage', {}),
                'sources_count': len(result.get('sources', [])),
                'success': True,
                'error': None
            }
        
        except Exception as e:
            logger.error(f"❌ 질문 실행 실패 [{model_name}]: {e}")
            return {
                'model': model_name,
                'query': query,
                'category': query_info.get('category', 'unknown'),
                'expected_type': query_info.get('expected_type', 'unknown'),
                'answer': None,
                'used_retrieval': False,
                'query_type': 'error',
                'search_mode': 'none',
                'elapsed_time': 0,
                'model_elapsed_time': 0,
                'usage': {},
                'sources_count': 0,
                'success': False,
                'error': str(e)
            }
    
    def run_experiment(
        self, 
        distribution: str = 'all',
        save_results: bool = True
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        실험 실행
        
        Args:
            distribution: 'in', 'out', 'all'
            save_results: 결과 저장 여부
        """
        logger.info("\n" + "="*60)
        logger.info("실험 시작")
        logger.info("="*60)
        
        # 데이터셋 준비
        if distribution == 'in':
            queries_dict = {'in_distribution': self.dataset.get_in_distribution()}
        elif distribution == 'out':
            queries_dict = {'out_distribution': self.dataset.get_out_distribution()}
        else:  # 'all'
            queries_dict = self.dataset.get_all_queries()
        
        # 결과 저장
        all_results = {
            'metadata': {
                'timestamp': self.timestamp,
                'distribution': distribution,
                'models': list(self.pipelines.keys()),
                'total_queries': sum(len(v) for v in queries_dict.values())
            },
            'results': {}
        }
        
        # 각 분포에 대해 실험
        for dist_type, queries in queries_dict.items():
            logger.info(f"\n{'='*60}")
            logger.info(f"{dist_type.upper()} 실험 ({len(queries)}개 질문)")
            logger.info(f"{'='*60}")
            
            dist_results = []
            
            # 각 질문에 대해
            for i, query_info in enumerate(queries, 1):
                query = query_info['query']
                logger.info(f"\n[{i}/{len(queries)}] 질문: {query}")
                
                # 각 모델에 대해
                for model_name in self.pipelines.keys():
                    logger.info(f"  → {model_name} 실행 중...")
                    
                    result = self.run_single_query(model_name, query, query_info)
                    dist_results.append(result)
                    
                    if result['success']:
                        logger.info(f"     ✅ 완료 ({result['elapsed_time']:.2f}초)")
                    else:
                        logger.warning(f"     ❌ 실패: {result['error']}")
            
            all_results['results'][dist_type] = dist_results
        
        # 결과 저장
        if save_results:
            self._save_results(all_results)
        
        logger.info("\n" + "="*60)
        logger.info("✅ 실험 완료")
        logger.info("="*60 + "\n")
        
        return all_results
    
    def _save_results(self, results: Dict[str, Any]):
        """결과 저장"""
        # JSON 파일로 저장
        output_file = self.output_dir / f"results_{self.timestamp}.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"📁 결과 저장: {output_file}")
        
        # 요약 통계 저장
        summary_file = self.output_dir / f"summary_{self.timestamp}.txt"
        self._save_summary(results, summary_file)
        
        logger.info(f"📊 요약 저장: {summary_file}")
    
    def _save_summary(self, results: Dict[str, Any], output_file: Path):
        """요약 통계 저장"""
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("="*60 + "\n")
            f.write("실험 결과 요약\n")
            f.write("="*60 + "\n\n")
            
            # 메타데이터
            metadata = results['metadata']
            f.write(f"타임스탬프: {metadata['timestamp']}\n")
            f.write(f"분포: {metadata['distribution']}\n")
            f.write(f"모델: {', '.join(metadata['models'])}\n")
            f.write(f"총 질문 수: {metadata['total_queries']}\n\n")
            
            # 각 분포별 통계
            for dist_type, dist_results in results['results'].items():
                f.write(f"\n{'='*60}\n")
                f.write(f"{dist_type.upper()} 결과\n")
                f.write(f"{'='*60}\n\n")
                
                # 모델별로 그룹화
                model_stats = {}
                for result in dist_results:
                    model = result['model']
                    if model not in model_stats:
                        model_stats[model] = []
                    model_stats[model].append(result)
                
                # 각 모델별 통계
                for model, model_results in model_stats.items():
                    f.write(f"\n[{model}]\n")
                    
                    # 성공/실패
                    success_count = sum(1 for r in model_results if r['success'])
                    f.write(f"  성공: {success_count}/{len(model_results)}\n")
                    
                    # 평균 시간
                    avg_time = sum(r['elapsed_time'] for r in model_results if r['success']) / max(success_count, 1)
                    f.write(f"  평균 시간: {avg_time:.3f}초\n")
                    
                    # 평균 토큰
                    total_tokens = sum(r['usage'].get('total_tokens', 0) for r in model_results if r['success'])
                    avg_tokens = total_tokens / max(success_count, 1)
                    f.write(f"  평균 토큰: {avg_tokens:.1f}\n")
                    
                    # RAG 사용률
                    rag_count = sum(1 for r in model_results if r['used_retrieval'])
                    f.write(f"  RAG 사용: {rag_count}/{len(model_results)} ({rag_count/len(model_results)*100:.1f}%)\n")


def main():
    """메인 함수"""
    logger.info("="*60)
    logger.info("RFPilot 모델 비교 실험")
    logger.info("="*60)
    
    # Config 로드
    config = RAGConfig()
    
    # 실험 초기화
    experiment = ModelComparison(config=config, output_dir="./experiments/results")
    
    # 데이터셋 확인
    experiment.dataset.print_summary()
    experiment.dataset.print_samples(n=3)
    
    # 모델 로드
    experiment.load_models()
    
    # 실험 실행
    # 옵션 1: 전체 실험
    results = experiment.run_experiment(distribution='all', save_results=True)
    
    # 옵션 2: In-Distribution만
    # results = experiment.run_experiment(distribution='in', save_results=True)
    
    # 옵션 3: Out-Distribution만
    # results = experiment.run_experiment(distribution='out', save_results=True)
    
    logger.info(f"\n✅ 모든 실험 완료!")
    logger.info(f"   결과 저장 위치: {experiment.output_dir}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n⚠️ 사용자에 의해 중단됨")
    except Exception as e:
        logger.error(f"\n❌ 실험 실패: {e}")
        import traceback
        traceback.print_exc()