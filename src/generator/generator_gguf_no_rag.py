from llama_cpp import Llama
from typing import Optional, Dict, Any, List
import logging
import time
import os

from src.utils.config import RAGConfig
from src.router.query_router import QueryRouter
from src.prompts.dynamic_prompts import PromptManager

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GGUFGenerator:
    """
    GGUF 기반 Llama-3 생성기
    
    llama.cpp를 사용하여 GGUF 포맷 모델을 로드하고
    입찰 관련 질의응답을 수행합니다.
    """
    
    def __init__(
        self,
        model_path: str,
        n_gpu_layers: int = 0,
        n_ctx: int = 8192,
        n_threads: int = 8,
        config = None,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        system_prompt: str = "당신은 RFP(제안요청서) 분석 및 요약 전문가입니다."
    ):
        """생성기 초기화"""
        self.config = config or RAGConfig() 
        self.model_path = model_path
        self.n_gpu_layers = n_gpu_layers
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.system_prompt = system_prompt
        
        # 모델 (나중에 로드)
        self.model = None
        
        logger.info(f"GGUFGenerator 초기화 완료")
    
    def load_model(self) -> None:
        """GGUF 모델 로드"""
        
        # 중복 로드 방지
        if self.model is not None:
            logger.info("모델이 이미 로드되어 있습니다.")
            return
        
        try:
            # Config에서 USE_MODEL_HUB 확인
            use_model_hub = getattr(self.config, 'USE_MODEL_HUB', True)
            
            # Model Hub 사용 여부에 따라 경로 결정
            if use_model_hub:
                # === Model Hub에서 다운로드 ===
                model_hub_repo = getattr(self.config, 'MODEL_HUB_REPO', 'Dongjin1203/RFP_Documents_chatbot')
                model_hub_filename = getattr(self.config, 'MODEL_HUB_FILENAME', 'Llama-3-Open-Ko-8B.Q4_K_M.gguf')
                model_cache_dir = getattr(self.config, 'MODEL_CACHE_DIR', '.cache/models')
                
                logger.info(f"📥 Model Hub에서 다운로드: {model_hub_repo}")
                
                from huggingface_hub import hf_hub_download
                
                model_path = hf_hub_download(
                    repo_id=model_hub_repo,
                    filename=model_hub_filename,
                    cache_dir=model_cache_dir,
                    local_dir=model_cache_dir,
                    local_dir_use_symlinks=False
                )
                
                logger.info(f"✅ 다운로드 완료: {model_path}")
                
            else:
                # === 로컬 파일 사용 ===
                model_path = self.model_path
                
                if not os.path.exists(model_path):
                    raise FileNotFoundError(
                        f"❌ 로컬 모델 파일을 찾을 수 없습니다: {model_path}\n"
                        f"   USE_MODEL_HUB=true로 설정하거나 모델 파일을 준비하세요."
                    )
                
                logger.info(f"📂 로컬 모델 사용: {model_path}")
            
            # === 공통: 모델 로드 ===
            logger.info(f"🚀 GGUF 모델 로드 중...")
            logger.info(f"   GPU 레이어: {self.n_gpu_layers}")
            logger.info(f"   컨텍스트: {self.n_ctx}")
            
            self.model = Llama(
                model_path=model_path,
                n_gpu_layers=self.n_gpu_layers,
                n_ctx=self.n_ctx,
                n_threads=self.n_threads,
                verbose=True,
            )
            
            # 실제 적용된 n_ctx 확인
            actual_n_ctx = self.model.n_ctx()
            logger.info("✅ GGUF 모델 로드 완료!")
            logger.info(f"   - 설정한 n_ctx: {self.n_ctx}")
            logger.info(f"   - 실제 n_ctx: {actual_n_ctx}")
            
            if actual_n_ctx < self.n_ctx:
                logger.warning(f"⚠️ n_ctx가 예상보다 작습니다: {actual_n_ctx} < {self.n_ctx}")
                logger.warning(f"   메모리 부족일 수 있습니다. n_gpu_layers를 줄여보세요.")
            
        except FileNotFoundError as e:
            logger.error(f"❌ 모델 파일을 찾을 수 없습니다: {e}")
            raise
        except Exception as e:
            logger.error(f"❌ 모델 로드 실패: {e}")
            raise RuntimeError(f"모델 로드 중 오류 발생: {e}")
    
    def format_prompt(
        self,
        question: str,
        context: Optional[str] = None,
        system_prompt: Optional[str] = None
    ) -> str:
        """GGUF 모델용 간단한 프롬프트 포맷팅"""
        # 시스템 프롬프트 설정
        if system_prompt is None:
            system_prompt = self.system_prompt
        
        # 컨텍스트 포함 여부
        if context is not None:
            user_message = f"참고 문서:\n{context}\n\n질문: {question}"
        else:
            user_message = question
        
        # 간단한 한국어 템플릿
        formatted_prompt = f"""### 시스템
{system_prompt}

### 사용자
{user_message}

### 답변
"""
        
        return formatted_prompt
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> str:
        """프롬프트를 입력받아 응답 생성"""
        # 모델 로드 확인
        if self.model is None:
            raise RuntimeError(
                "모델이 로드되지 않았습니다. load_model()을 먼저 호출하세요."
            )
        
        # 파라미터 설정
        if max_new_tokens is None:
            max_new_tokens = self.max_new_tokens
        if temperature is None:
            temperature = self.temperature
        if top_p is None:
            top_p = self.top_p
        
        try:
            logger.info(f"🔄 생성 시작 (max_tokens={max_new_tokens}, temp={temperature})")
            start_time = time.time()
            
            # 생성
            output = self.model(
                prompt,
                max_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                echo=False,
                stop=[
                    "###", "\n\n###", 
                    "### 사용자", "\n사용자:", 
                    "</s>",
                    "한국어 답변", "한국어로 답변", "지침:",
                    "문장", "(문장",
                    "\n\n",
                    "?",
                    "요?", "까?", "나요?", "습니까?"
                ],
            )
            
            elapsed = time.time() - start_time
            logger.info(f"✅ 생성 완료: {elapsed:.2f}초")
            
            # 응답 추출
            response = output['choices'][0]['text'].strip()
            
            logger.info(f"📝 응답 길이: {len(response)} 글자")
            return response
            
        except Exception as e:
            logger.error(f"❌ 생성 중 오류 발생: {e}")
            raise RuntimeError(f"텍스트 생성 실패: {e}")
    
    def chat(
        self,
        question: str,
        context: Optional[str] = None,
        system_prompt=None,
        **kwargs
    ) -> str:
        """질문에 대한 응답 생성 (통합 메서드)"""
        # 프롬프트 포맷팅
        prompt = self.format_prompt(
            question=question,
            context=context,
            system_prompt=system_prompt
        )
        
        # 응답 생성
        response = self.generate(prompt, **kwargs)
        
        return response


class GGUFNoRAGPipeline:
    """
    QLoRA 모델 단독 파이프라인 (RAG 제거)
    
    ✅ Retriever 완전 제거
    ✅ Router만 유지 (greeting/thanks 처리용)
    ✅ 순수 모델 성능만 측정
    """
    
    def __init__(
        self,
        config=None,
        model: str = None,
        top_k: int = None,
        n_gpu_layers: int = None,
        n_ctx: int = None,
        n_threads: int = None,
        max_new_tokens: int = None,
        temperature: float = None,
        top_p: float = None
    ):
        """초기화"""
        self.config = config or RAGConfig()
        
        # GGUF 설정
        gguf_n_gpu_layers = n_gpu_layers if n_gpu_layers is not None else getattr(self.config, 'GGUF_N_GPU_LAYERS', 35)
        gguf_n_ctx = n_ctx if n_ctx is not None else getattr(self.config, 'GGUF_N_CTX', 2048)
        gguf_n_threads = n_threads if n_threads is not None else getattr(self.config, 'GGUF_N_THREADS', 4)
        gguf_max_new_tokens = max_new_tokens if max_new_tokens is not None else getattr(self.config, 'GGUF_MAX_NEW_TOKENS', 512)
        gguf_temperature = temperature if temperature is not None else getattr(self.config, 'GGUF_TEMPERATURE', 0.7)
        gguf_top_p = top_p if top_p is not None else getattr(self.config, 'GGUF_TOP_P', 0.9)
        
        # 모델 경로
        gguf_model_path = getattr(self.config, 'GGUF_MODEL_PATH', '.cache/models/llama-3-ko-8b.gguf')
        
        # 시스템 프롬프트
        system_prompt = getattr(self.config, 'SYSTEM_PROMPT', '당신은 한국 공공기관 사업제안서 분석 전문가입니다.')
        
        # GGUFGenerator 초기화
        logger.info("GGUFGenerator 초기화 중... (RAG 없음)")
        logger.info(f"   GPU 레이어: {gguf_n_gpu_layers}")
        logger.info(f"   컨텍스트: {gguf_n_ctx}")
        logger.info(f"   스레드: {gguf_n_threads}")
        
        self.generator = GGUFGenerator(
            model_path=gguf_model_path,
            n_gpu_layers=gguf_n_gpu_layers,
            n_ctx=gguf_n_ctx,
            n_threads=gguf_n_threads,
            config=self.config,
            max_new_tokens=gguf_max_new_tokens,
            temperature=gguf_temperature,
            top_p=gguf_top_p,
            system_prompt=system_prompt
        )
        
        # 모델 로드
        logger.info("GGUF 모델 로드 중...")
        self.generator.load_model()
        
        # ✅ Retriever 없음 (완전 제거)
        self.retriever = None
        
        # Router (greeting/thanks 처리용)
        self.router = QueryRouter()
        
        # 대화 히스토리
        self.chat_history: List[Dict] = []
        
        logger.info("✅ GGUFNoRAGPipeline 초기화 완료 (RAG 제거)")
        logger.info("   - Retriever: ❌ 없음")
        logger.info("   - Router: ✅ 있음 (greeting/thanks용)")
    
    def _estimate_usage(self, query: str, answer: str) -> dict:
        """토큰 사용량 추정"""
        prompt_tokens = len(query.split()) * 2
        completion_tokens = len(answer.split()) * 2
        
        return {
            'total_tokens': prompt_tokens + completion_tokens,
            'prompt_tokens': prompt_tokens,
            'completion_tokens': completion_tokens
        }
    
    def generate_answer(
        self,
        query: str,
        top_k: int = None,
        search_mode: str = None,
        alpha: float = None
    ) -> dict:
        """
        답변 생성 (RAG 없음)
        
        Args:
            query: 질문
            top_k: 사용 안 함 (호환성용)
            search_mode: 사용 안 함 (호환성용)
            alpha: 사용 안 함 (호환성용)
        
        Returns:
            dict: answer, sources, search_mode, usage, elapsed_time, used_retrieval
        """
        try:
            start_time = time.time()
            
            # Router로 질문 분류
            classification = self.router.classify(query)
            query_type = classification['type']
            
            logger.info(f"📍 분류: {query_type} (신뢰도: {classification['confidence']:.2f})")
            
            # 동적 프롬프트 선택
            if query_type in ['greeting', 'thanks', 'out_of_scope']:
                system_prompt = PromptManager.get_prompt(query_type, model_type="gguf")
            else:
                system_prompt = PromptManager.get_prompt('document', model_type="gguf")
            
            # ✅ 항상 RAG 없이 생성 (context=None)
            answer = self.generator.chat(
                question=query,
                context=None,  # ✅ 컨텍스트 없음
                system_prompt=system_prompt
            )
            
            elapsed_time = time.time() - start_time
            
            # 대화 히스토리 추가
            self.chat_history.append({"role": "user", "content": query})
            self.chat_history.append({"role": "assistant", "content": answer})
            
            # 결과 반환
            return {
                'answer': answer,
                'sources': [],  # ✅ 소스 없음
                'used_retrieval': False,  # ✅ 검색 안 함
                'query_type': query_type,
                'search_mode': 'none',  # ✅ 검색 모드 없음
                'routing_info': classification,
                'elapsed_time': elapsed_time,
                'usage': self._estimate_usage(query, answer)
            }
        
        except Exception as e:
            logger.error(f"❌ 답변 생성 실패: {e}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"답변 생성 실패: {str(e)}") from e
    
    def chat(self, query: str) -> str:
        """간단한 대화 인터페이스"""
        result = self.generate_answer(query)
        return result['answer']
    
    def clear_history(self):
        """대화 히스토리 초기화"""
        self.chat_history = []
        logger.info("🗑️ 대화 히스토리가 초기화되었습니다.")
    
    def get_history(self) -> List[Dict]:
        """대화 히스토리 반환"""
        return self.chat_history.copy()