"""
HuggingFace Space용 실험 앱 (Streamlit)

Streamlit을 사용하여 웹 UI에서 실험을 실행하고 결과를 확인합니다.
"""

import streamlit as st
import os
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
import sys

# 프로젝트 경로 설정
sys.path.insert(0, str(Path(__file__).parent))

st.set_page_config(
    page_title="RFPilot 모델 비교 실험",
    page_icon="🔬",
    layout="wide"
)


class ExperimentApp:
    """실험 앱 클래스"""
    
    @staticmethod
    def setup_environment(api_key: str) -> bool:
        """환경 설정"""
        if not api_key:
            return False
        
        os.environ['OPENAI_API_KEY'] = api_key
        os.environ['USE_MODEL_HUB'] = 'true'
        os.environ['GGUF_N_GPU_LAYERS'] = '35'
        
        return True
    
    @staticmethod
    @st.cache_resource(show_spinner="모델 로딩 중... (5-10분 소요)")
    def load_models():
        """모델 로드 (캐싱)"""
        from src.utils.config import RAGConfig
        from src.compare_models import ModelComparison
        
        config = RAGConfig()
        experiment = ModelComparison(
            config=config,
            output_dir="./experiments/results"
        )
        
        experiment.load_models()
        
        return experiment
    
    @staticmethod
    def generate_summary(results: dict) -> str:
        """요약 생성"""
        summary = "=" * 60 + "\n"
        summary += "실험 결과 요약\n"
        summary += "=" * 60 + "\n\n"
        
        metadata = results['metadata']
        summary += f"타임스탬프: {metadata['timestamp']}\n"
        summary += f"분포: {metadata['distribution']}\n"
        summary += f"모델: {', '.join(metadata['models'])}\n"
        summary += f"총 질문 수: {metadata['total_queries']}\n\n"
        
        return summary
    
    @staticmethod
    def results_to_dataframe(results: dict) -> pd.DataFrame:
        """결과를 DataFrame으로 변환"""
        all_rows = []
        
        for dist_type, dist_results in results['results'].items():
            for result in dist_results:
                row = {
                    'distribution': dist_type,
                    'model': result['model'],
                    'query': result['query'],
                    'success': result['success'],
                    'elapsed_time': result['elapsed_time'],
                    'total_tokens': result.get('usage', {}).get('total_tokens', 0)
                }
                all_rows.append(row)
        
        return pd.DataFrame(all_rows)


def main():
    """메인 앱"""
    
    # 헤더
    st.title("🔬 RFPilot 모델 비교 실험")
    st.markdown("""
    3가지 모델(QLoRA+RAG, QLoRA 단독, Base+RAG)의 성능을 비교합니다.
    
    ⚠️ **첫 실행 시 모델 다운로드로 5-10분 소요됩니다.**
    """)
    
    # 탭 생성
    tab1, tab2, tab3 = st.tabs(["🚀 실험 실행", "📊 결과 분석", "ℹ️ 정보"])
    
    # ===== 탭 1: 실험 실행 =====
    with tab1:
        st.header("실험 실행")
        
        # API 키 입력
        api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            placeholder="sk-..."
        )
        
        # 분포 선택
        distribution = st.radio(
            "분포 선택",
            ["All", "In", "Out"],
            horizontal=True
        )
        
        # 실험 시작 버튼
        if st.button("실험 시작", type="primary"):
            if not api_key:
                st.error("❌ OpenAI API 키를 입력해주세요.")
            else:
                try:
                    # 환경 설정
                    ExperimentApp.setup_environment(api_key)
                    st.success("✅ 환경 설정 완료!")
                    
                    # 모델 로드
                    with st.spinner("모델 로딩 중... (첫 실행 시 5-10분 소요)"):
                        experiment = ExperimentApp.load_models()
                    st.success("✅ 모델 로드 완료!")
                    
                    # 실험 실행
                    with st.spinner("실험 실행 중... (10-20분 소요)"):
                        results = experiment.run_experiment(
                            distribution=distribution.lower(),
                            save_results=True
                        )
                    
                    st.success("✅ 실험 완료!")
                    
                    # 결과 표시
                    st.subheader("실험 결과")
                    
                    # 요약
                    summary = ExperimentApp.generate_summary(results)
                    st.text(summary)
                    
                    # DataFrame
                    df = ExperimentApp.results_to_dataframe(results)
                    st.dataframe(df, use_container_width=True)
                    
                    # 세션에 결과 저장
                    st.session_state['latest_results'] = results
                    st.session_state['result_file'] = experiment.output_dir / f"results_{experiment.timestamp}.json"
                    
                except Exception as e:
                    st.error(f"❌ 실험 실패: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
    
    # ===== 탭 2: 결과 분석 =====
    with tab2:
        st.header("결과 분석")
        
        if st.button("분석 시작", type="primary"):
            if 'result_file' not in st.session_state:
                st.error("❌ 먼저 실험을 실행해주세요.")
            else:
                try:
                    from src.analyze_results import ResultAnalyzer
                    
                    with st.spinner("결과 분석 중..."):
                        analyzer = ResultAnalyzer(str(st.session_state['result_file']))
                        
                        # 그래프 생성
                        analyzer.plot_time_comparison()
                        analyzer.plot_token_comparison()
                        analyzer.plot_rag_usage()
                        analyzer.plot_overfitting_analysis()
                    
                    st.success("✅ 분석 완료!")
                    
                    # 그래프 표시
                    analysis_dir = st.session_state['result_file'].parent / "analysis"
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        time_plot = analysis_dir / "time_comparison.png"
                        if time_plot.exists():
                            st.image(str(time_plot), caption="응답 시간 비교")
                        
                        rag_plot = analysis_dir / "rag_usage.png"
                        if rag_plot.exists():
                            st.image(str(rag_plot), caption="RAG 사용 패턴")
                    
                    with col2:
                        token_plot = analysis_dir / "token_comparison.png"
                        if token_plot.exists():
                            st.image(str(token_plot), caption="토큰 사용량 비교")
                        
                        overfitting_plot = analysis_dir / "overfitting_analysis.png"
                        if overfitting_plot.exists():
                            st.image(str(overfitting_plot), caption="과적합 분석")
                    
                except Exception as e:
                    st.error(f"❌ 분석 실패: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
    
    # ===== 탭 3: 정보 =====
    with tab3:
        st.header("프로젝트 정보")
        
        st.markdown("""
        ## 📋 비교 모델
        
        | 모델 | 설명 |
        |------|------|
        | QLoRA + RAG | 기존 서비스 (QLoRA fine-tuning + RAG) |
        | QLoRA 단독 | RAG 제거 (QLoRA만) |
        | Base + RAG | PEFT 제거 (Base 모델 + RAG) |
        
        ## 📊 측정 지표
        
        - **과적합**: In-Distribution vs Out-Distribution 성능 차이
        - **답변 속도**: 평균 응답 시간
        - **토큰 사용량**: 평균 토큰 소비
        - **RAG 사용 패턴**: RAG 활용도
        
        ## ⏱️ 예상 소요 시간
        
        - 모델 로딩: 5-10분 (첫 실행 시)
        - 실험 실행: 10-20분 (5개 질문 x 3개 모델)
        - 결과 분석: 1-2분
        
        ## 💡 사용 팁
        
        - 모델은 한 번 로드되면 캐시됩니다
        - 페이지 새로고침 시 모델 재로드 필요
        - API 키는 세션에만 저장되며 서버에 저장되지 않습니다
        """)


if __name__ == "__main__":
    main()