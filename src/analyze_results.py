"""
실험 결과 분석 및 시각화

JSON 결과 파일을 읽어서 다양한 그래프를 생성합니다.
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any
import numpy as np

# 한글 폰트 설정 (matplotlib)
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False


class ResultAnalyzer:
    """실험 결과 분석 클래스"""
    
    def __init__(self, result_file_path: str):
        """
        초기화
        
        Args:
            result_file_path: JSON 결과 파일 경로
        """
        self.result_file = Path(result_file_path)
        self.results = self._load_results()
        self.df = self._results_to_dataframe()
        
        # 분석 결과 저장 디렉토리
        self.analysis_dir = self.result_file.parent / "analysis"
        self.analysis_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"✅ ResultAnalyzer 초기화 완료")
        print(f"   결과 파일: {self.result_file}")
        print(f"   분석 저장 경로: {self.analysis_dir}")
    
    def _load_results(self) -> Dict[str, Any]:
        """JSON 결과 파일 로드"""
        with open(self.result_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _results_to_dataframe(self) -> pd.DataFrame:
        """결과를 DataFrame으로 변환"""
        all_rows = []
        
        for dist_type, dist_results in self.results['results'].items():
            for result in dist_results:
                row = {
                    'distribution': dist_type,
                    'model': result['model'],
                    'query': result['query'],
                    'category': result.get('category', 'unknown'),
                    'success': result['success'],
                    'elapsed_time': result['elapsed_time'],
                    'used_retrieval': result.get('used_retrieval', False),
                    'query_type': result.get('query_type', 'unknown'),
                    'search_mode': result.get('search_mode', 'none'),
                    'total_tokens': result.get('usage', {}).get('total_tokens', 0),
                    'prompt_tokens': result.get('usage', {}).get('prompt_tokens', 0),
                    'completion_tokens': result.get('usage', {}).get('completion_tokens', 0),
                }
                all_rows.append(row)
        
        return pd.DataFrame(all_rows)
    
    def plot_time_comparison(self, figsize=(12, 6)):
        """응답 시간 비교 그래프"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # 성공한 결과만 사용
        df_success = self.df[self.df['success'] == True].copy()
        
        # 1. 모델별 평균 응답 시간
        model_time = df_success.groupby('model')['elapsed_time'].agg(['mean', 'std'])
        
        ax1.bar(model_time.index, model_time['mean'], yerr=model_time['std'], 
                capsize=5, alpha=0.7, color=['#2ecc71', '#3498db', '#e74c3c'])
        ax1.set_title('Average Response Time by Model', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Time (seconds)', fontsize=12)
        ax1.set_xlabel('Model', fontsize=12)
        ax1.grid(axis='y', alpha=0.3)
        
        # 2. Distribution별 응답 시간
        pivot_data = df_success.pivot_table(
            values='elapsed_time', 
            index='model', 
            columns='distribution', 
            aggfunc='mean'
        )
        
        x = np.arange(len(pivot_data.index))
        width = 0.35
        
        if 'in_distribution' in pivot_data.columns:
            ax2.bar(x - width/2, pivot_data['in_distribution'], width, 
                   label='In-Distribution', alpha=0.8, color='#2ecc71')
        
        if 'out_distribution' in pivot_data.columns:
            ax2.bar(x + width/2, pivot_data['out_distribution'], width, 
                   label='Out-Distribution', alpha=0.8, color='#e74c3c')
        
        ax2.set_title('Response Time: In vs Out Distribution', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Time (seconds)', fontsize=12)
        ax2.set_xlabel('Model', fontsize=12)
        ax2.set_xticks(x)
        ax2.set_xticklabels(pivot_data.index, rotation=15, ha='right')
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        # 저장
        output_file = self.analysis_dir / "time_comparison.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 응답 시간 그래프 저장: {output_file}")
    
    def plot_token_comparison(self, figsize=(12, 6)):
        """토큰 사용량 비교 그래프"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # 성공한 결과만 사용
        df_success = self.df[self.df['success'] == True].copy()
        
        # 1. 모델별 평균 토큰 사용량
        model_tokens = df_success.groupby('model')['total_tokens'].agg(['mean', 'std'])
        
        ax1.bar(model_tokens.index, model_tokens['mean'], yerr=model_tokens['std'],
                capsize=5, alpha=0.7, color=['#2ecc71', '#3498db', '#e74c3c'])
        ax1.set_title('Average Token Usage by Model', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Tokens', fontsize=12)
        ax1.set_xlabel('Model', fontsize=12)
        ax1.grid(axis='y', alpha=0.3)
        
        # 2. Distribution별 토큰 사용량
        pivot_data = df_success.pivot_table(
            values='total_tokens',
            index='model',
            columns='distribution',
            aggfunc='mean'
        )
        
        x = np.arange(len(pivot_data.index))
        width = 0.35
        
        if 'in_distribution' in pivot_data.columns:
            ax2.bar(x - width/2, pivot_data['in_distribution'], width,
                   label='In-Distribution', alpha=0.8, color='#2ecc71')
        
        if 'out_distribution' in pivot_data.columns:
            ax2.bar(x + width/2, pivot_data['out_distribution'], width,
                   label='Out-Distribution', alpha=0.8, color='#e74c3c')
        
        ax2.set_title('Token Usage: In vs Out Distribution', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Tokens', fontsize=12)
        ax2.set_xlabel('Model', fontsize=12)
        ax2.set_xticks(x)
        ax2.set_xticklabels(pivot_data.index, rotation=15, ha='right')
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        # 저장
        output_file = self.analysis_dir / "token_comparison.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 토큰 사용량 그래프 저장: {output_file}")
    
    def plot_rag_usage(self, figsize=(10, 6)):
        """RAG 사용 패턴 분석"""
        fig, ax = plt.subplots(figsize=figsize)
        
        # 모델별 RAG 사용률 계산
        rag_usage = self.df.groupby('model').agg({
            'used_retrieval': lambda x: (x.sum() / len(x) * 100)
        }).round(2)
        
        colors = ['#2ecc71', '#3498db', '#e74c3c']
        bars = ax.bar(rag_usage.index, rag_usage['used_retrieval'], 
                     alpha=0.7, color=colors)
        
        # 막대 위에 퍼센트 표시
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%',
                   ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        ax.set_title('RAG Usage Rate by Model', fontsize=14, fontweight='bold')
        ax.set_ylabel('Usage Rate (%)', fontsize=12)
        ax.set_xlabel('Model', fontsize=12)
        ax.set_ylim(0, 105)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        # 저장
        output_file = self.analysis_dir / "rag_usage.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ RAG 사용 패턴 그래프 저장: {output_file}")
    
    def plot_overfitting_analysis(self, figsize=(12, 8)):
        """과적합 분석: In-Distribution vs Out-Distribution 성능 차이"""
        # 성공한 결과만 사용
        df_success = self.df[self.df['success'] == True].copy()
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # 1. 응답 시간 차이
        time_pivot = df_success.pivot_table(
            values='elapsed_time',
            index='model',
            columns='distribution',
            aggfunc='mean'
        )
        
        if 'in_distribution' in time_pivot.columns and 'out_distribution' in time_pivot.columns:
            time_diff = time_pivot['out_distribution'] - time_pivot['in_distribution']
            
            axes[0, 0].bar(time_diff.index, time_diff.values, 
                          color=['green' if x < 0 else 'red' for x in time_diff.values],
                          alpha=0.7)
            axes[0, 0].axhline(y=0, color='black', linestyle='-', linewidth=0.8)
            axes[0, 0].set_title('Response Time Gap (Out - In)', fontsize=12, fontweight='bold')
            axes[0, 0].set_ylabel('Time Difference (seconds)', fontsize=10)
            axes[0, 0].grid(axis='y', alpha=0.3)
        
        # 2. 토큰 사용량 차이
        token_pivot = df_success.pivot_table(
            values='total_tokens',
            index='model',
            columns='distribution',
            aggfunc='mean'
        )
        
        if 'in_distribution' in token_pivot.columns and 'out_distribution' in token_pivot.columns:
            token_diff = token_pivot['out_distribution'] - token_pivot['in_distribution']
            
            axes[0, 1].bar(token_diff.index, token_diff.values,
                          color=['green' if x < 0 else 'red' for x in token_diff.values],
                          alpha=0.7)
            axes[0, 1].axhline(y=0, color='black', linestyle='-', linewidth=0.8)
            axes[0, 1].set_title('Token Usage Gap (Out - In)', fontsize=12, fontweight='bold')
            axes[0, 1].set_ylabel('Token Difference', fontsize=10)
            axes[0, 1].grid(axis='y', alpha=0.3)
        
        # 3. 성공률 비교
        success_pivot = self.df.pivot_table(
            values='success',
            index='model',
            columns='distribution',
            aggfunc=lambda x: (x.sum() / len(x) * 100)
        )
        
        x = np.arange(len(success_pivot.index))
        width = 0.35
        
        if 'in_distribution' in success_pivot.columns:
            axes[1, 0].bar(x - width/2, success_pivot['in_distribution'], width,
                          label='In-Distribution', alpha=0.8, color='#2ecc71')
        
        if 'out_distribution' in success_pivot.columns:
            axes[1, 0].bar(x + width/2, success_pivot['out_distribution'], width,
                          label='Out-Distribution', alpha=0.8, color='#e74c3c')
        
        axes[1, 0].set_title('Success Rate Comparison', fontsize=12, fontweight='bold')
        axes[1, 0].set_ylabel('Success Rate (%)', fontsize=10)
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(success_pivot.index, rotation=15, ha='right')
        axes[1, 0].legend()
        axes[1, 0].grid(axis='y', alpha=0.3)
        axes[1, 0].set_ylim(0, 105)
        
        # 4. 과적합 지수 (Performance Gap)
        if 'in_distribution' in time_pivot.columns and 'out_distribution' in time_pivot.columns:
            # 간단한 과적합 지수: (Out 시간 - In 시간) / In 시간 * 100
            overfitting_index = ((time_pivot['out_distribution'] - time_pivot['in_distribution']) / 
                                time_pivot['in_distribution'] * 100)
            
            colors_custom = ['green' if x < 10 else 'orange' if x < 30 else 'red' 
                           for x in overfitting_index.values]
            
            axes[1, 1].bar(overfitting_index.index, overfitting_index.values,
                          color=colors_custom, alpha=0.7)
            axes[1, 1].axhline(y=0, color='black', linestyle='-', linewidth=0.8)
            axes[1, 1].axhline(y=10, color='orange', linestyle='--', linewidth=0.8, alpha=0.5)
            axes[1, 1].axhline(y=30, color='red', linestyle='--', linewidth=0.8, alpha=0.5)
            axes[1, 1].set_title('Overfitting Index (Time-based)', fontsize=12, fontweight='bold')
            axes[1, 1].set_ylabel('Performance Gap (%)', fontsize=10)
            axes[1, 1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        # 저장
        output_file = self.analysis_dir / "overfitting_analysis.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 과적합 분석 그래프 저장: {output_file}")
    
    def generate_summary_report(self):
        """종합 요약 보고서 생성"""
        report_file = self.analysis_dir / "summary_report.txt"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("RFPilot 모델 비교 실험 - 종합 분석 보고서\n")
            f.write("="*70 + "\n\n")
            
            # 메타데이터
            metadata = self.results['metadata']
            f.write(f"실험 일시: {metadata['timestamp']}\n")
            f.write(f"분포: {metadata['distribution']}\n")
            f.write(f"비교 모델: {', '.join(metadata['models'])}\n")
            f.write(f"총 질문 수: {metadata['total_queries']}\n\n")
            
            # 성공한 결과만
            df_success = self.df[self.df['success'] == True]
            
            # 1. 모델별 평균 성능
            f.write("\n" + "="*70 + "\n")
            f.write("1. 모델별 평균 성능\n")
            f.write("="*70 + "\n\n")
            
            for model in df_success['model'].unique():
                model_df = df_success[df_success['model'] == model]
                
                f.write(f"[{model}]\n")
                f.write(f"  - 성공률: {len(model_df)/len(self.df[self.df['model']==model])*100:.1f}%\n")
                f.write(f"  - 평균 응답 시간: {model_df['elapsed_time'].mean():.3f}초\n")
                f.write(f"  - 평균 토큰: {model_df['total_tokens'].mean():.1f}\n")
                f.write(f"  - RAG 사용률: {model_df['used_retrieval'].sum()/len(model_df)*100:.1f}%\n\n")
            
            # 2. Distribution별 성능
            f.write("\n" + "="*70 + "\n")
            f.write("2. Distribution별 성능 비교\n")
            f.write("="*70 + "\n\n")
            
            for dist in df_success['distribution'].unique():
                dist_df = df_success[df_success['distribution'] == dist]
                
                f.write(f"[{dist}]\n")
                f.write(f"  - 평균 응답 시간: {dist_df['elapsed_time'].mean():.3f}초\n")
                f.write(f"  - 평균 토큰: {dist_df['total_tokens'].mean():.1f}\n\n")
            
            # 3. 권장사항
            f.write("\n" + "="*70 + "\n")
            f.write("3. 분석 및 권장사항\n")
            f.write("="*70 + "\n\n")
            
            # 가장 빠른 모델
            fastest_model = df_success.groupby('model')['elapsed_time'].mean().idxmin()
            f.write(f"⚡ 가장 빠른 모델: {fastest_model}\n")
            
            # 가장 토큰을 적게 사용하는 모델
            efficient_model = df_success.groupby('model')['total_tokens'].mean().idxmin()
            f.write(f"💡 가장 효율적인 모델 (토큰): {efficient_model}\n")
            
            # RAG를 가장 많이 사용하는 모델
            rag_model = df_success.groupby('model')['used_retrieval'].sum().idxmax()
            f.write(f"🔍 RAG를 가장 많이 활용하는 모델: {rag_model}\n")
            
            f.write("\n" + "="*70 + "\n")
        
        print(f"✅ 종합 보고서 저장: {report_file}")
    
    def run_all_analysis(self):
        """모든 분석 실행"""
        print("\n" + "="*70)
        print("전체 분석 시작")
        print("="*70 + "\n")
        
        self.plot_time_comparison()
        self.plot_token_comparison()
        self.plot_rag_usage()
        self.plot_overfitting_analysis()
        self.generate_summary_report()
        
        print("\n" + "="*70)
        print("✅ 모든 분석 완료!")
        print(f"   결과 저장 위치: {self.analysis_dir}")
        print("="*70 + "\n")


def main():
    """테스트용 메인 함수"""
    import sys
    
    if len(sys.argv) < 2:
        print("사용법: python analyze_results.py <result_json_path>")
        return
    
    result_file = sys.argv[1]
    
    analyzer = ResultAnalyzer(result_file)
    analyzer.run_all_analysis()


if __name__ == "__main__":
    main()