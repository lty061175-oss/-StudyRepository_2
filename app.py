import streamlit as st
import scipy.io
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import kurtosis

# 1. 페이지 초기 설정 및 라이트 테마 친화적 스타일 적용
st.set_page_config(
    page_title="정상 vs 이상 데이터 특징 시각화 분석기",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 커스텀 화이트 테마 CSS 스타일 적용
st.markdown("""
    <style>
        /* 메인 배경 및 레이아웃 조정 */
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        /* 카드 컴포넌트 스타일 */
        .metric-card {
            background-color: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            padding: 1rem;
            box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
        }
        .metric-title {
            font-size: 0.85rem;
            color: #64748b;
            font-weight: 500;
        }
        .metric-value-n {
            font-size: 1.1rem;
            color: #10b981;
            font-weight: 700;
            font-family: monospace;
        }
        .metric-value-a {
            font-size: 1.1rem;
            color: #f43f5e;
            font-weight: 700;
            font-family: monospace;
        }
        .metric-desc {
            font-size: 0.75rem;
            color: #94a3b8;
            margin-top: 0.25rem;
        }
    </style>
""", unsafe_allow_html=True)

# 2. 통계 피처 추출 함수 정의
def extract_features(signal):
    if signal is None or len(signal) == 0:
        return {"mean": 0, "rms": 0, "std": 0, "p2p": 0, "kurt": 3.0, "cf": 0}
    
    mean_val = np.mean(signal)
    rms_val = np.sqrt(np.mean(signal**2))
    std_val = np.std(signal)
    p2p_val = np.max(signal) - np.min(signal)
    
    # scipy kurtosis는 기본적으로 Fisher 공식(정상 분포=0)을 사용하므로, Pearson 공식(정상 분포=3)으로 보정
    kurt_val = kurtosis(signal, fisher=False)
    
    max_abs = np.max(np.abs(signal))
    cf_val = max_abs / rms_val if rms_val > 0 else 0
    
    return {
        "mean": round(mean_val, 5),
        "rms": round(rms_val, 5),
        "std": round(std_val, 5),
        "p2p": round(p2p_val, 5),
        "kurt": round(kurt_val, 5),
        "cf": round(cf_val, 5)
    }

# 3. FFT 연산 함수 정의
def compute_fft(signal, fs=12000, n_fft=512):
    truncated = signal[:n_fft]
    yf = np.fft.fft(truncated)
    mags = (2.0 / n_fft) * np.abs(yf[:n_fft // 2])
    freqs = np.fft.fftfreq(n_fft, 1 / fs)[:n_fft // 2]
    return freqs, mags

# 4. MATLAB .mat 바이너리 파서 정의
def parse_mat_file(uploaded_file):
    try:
        # scipy.io.loadmat를 사용하여 압축/비압축 매트랩 파일을 통합 파싱
        mat_data = scipy.io.loadmat(uploaded_file)
        # 메타데이터를 제외한 실제 데이터 키 필터링
        keys = [k for k in mat_data.keys() if not k.startswith('__')]
        if not keys:
            st.error("MAT 파일 내에 유효한 데이터 키가 존재하지 않습니다.")
            return None, None
        
        # 가장 긴 길이를 가진 수치 배열 검색
        best_key = keys[0]
        max_len = 0
        for k in keys:
            data_item = mat_data[k]
            if isinstance(data_item, np.ndarray):
                flat_len = data_item.size
                if flat_len > max_len:
                    max_len = flat_len
                    best_key = k
                    
        extracted_data = mat_data[best_key].flatten().astype(float)
        return best_key, extracted_data
    except Exception as e:
        st.error(f"MAT 파일 해석 실패: {str(e)}")
        return None, None

# 5. 데모 데이터 생성 핸들러
def generate_demo_data(is_anomaly=False, fs=12000, length=12000):
    t = np.arange(length) / fs
    if not is_anomaly:
        # 정상 신호: 60Hz + 150Hz 성분 및 낮은 백색잡음
        signal = 1.2 * np.sin(2 * np.pi * 60 * t) + 0.6 * np.sin(2 * np.pi * 150 * t) + np.random.normal(0, 0.2, length)
    else:
        # 이상 신호: 주기적 충격 성분(320Hz 임팩트) 및 높은 불규칙 노이즈
        impact = np.exp(-((np.arange(length) % 600) / 100)) * np.sin(2 * np.pi * 320 * t) * 6.0
        signal = 1.0 * np.sin(2 * np.pi * 60 * t) + impact + np.random.normal(0, 0.9, length)
    return signal

# 6. 헤더 영역 구축
st.markdown("""
    <div style="display: flex; align-items: center; gap: 15px; margin-bottom: 5px;">
        <div style="background-color: #4f46e5; padding: 10px; border-radius: 12px; color: white; display: flex; align-items: center; justify-content: center;">
            <i class="fa-solid fa-chart-line" style="font-size: 24px;"></i>
        </div>
        <div>
            <h1 style="margin: 0; font-size: 1.8rem; font-weight: 700; color: #1e293b;">신호 특징 분석 대시보드</h1>
            <p style="margin: 0; font-size: 0.85rem; color: #64748b;">MATLAB (.mat) 정상 및 이상 파일 특징 비교 시각화</p>
        </div>
    </div>
    <hr style="margin-top: 10px; margin-bottom: 20px; border-color: #e2e8f0;">
""", unsafe_allow_html=True)

# 7. 안내 가이드 및 데모 활성화 영역
guide_col, demo_col = st.columns([3, 1])
with guide_col:
    st.markdown("""
        <div style="font-size: 0.9rem; color: #475569; line-height: 1.6;">
            정상 데이터와 이상 데이터를 드래그 앤 드롭으로 업로드하면 시간 영역의 대표적인 진동 특징값(KPI)과 
            FFT 주파수 분포 패턴을 화이트 테마 기반 인터랙티브 차트로 완벽하게 파싱하여 분석합니다.
        </div>
    """, unsafe_allow_html=True)
with demo_col:
    if st.button("🪄 데모 데이터로 즉시 테스트", use_container_width=True):
        st.session_state["normal_data"] = generate_demo_data(is_anomaly=False)
        st.session_state["normal_filename"] = "Demo_Normal_60Hz.mat"
        st.session_state["normal_var"] = "Demo_Normal"
        st.session_state["anomaly_data"] = generate_demo_data(is_anomaly=True)
        st.session_state["anomaly_filename"] = "Demo_Fault_Bearing_320Hz.mat"
        st.session_state["anomaly_var"] = "Demo_Anomaly"
        st.success("데모 데이터가 성공적으로 탑재되었습니다!")

# 세션 상태 초기화
if "normal_data" not in st.session_state:
    st.session_state["normal_data"] = None
    st.session_state["normal_filename"] = "Normal.mat"
    st.session_state["normal_var"] = "N/A"
if "anomaly_data" not in st.session_state:
    st.session_state["anomaly_data"] = None
    st.session_state["anomaly_filename"] = "B.mat"
    st.session_state["anomaly_var"] = "N/A"

# 8. 파일 업로더 영역 구축 (실시간 파일명 반영 대응)
up_col1, up_col2 = st.columns(2)

with up_col1:
    st.markdown(f"#### Class 1: 정상 데이터 (<span style='color:#10b981;'>{st.session_state['normal_filename']}</span>)", unsafe_allow_html=True)
    file_normal = st.file_uploader("Normal.mat 파일을 여기에 업로드하세요", type=["mat"], label_visibility="collapsed", key="uploader_normal")
    if file_normal is not None:
        var_name, data_arr = parse_mat_file(file_normal)
        if data_arr is not None:
            st.session_state["normal_data"] = data_arr
            st.session_state["normal_filename"] = file_normal.name
            st.session_state["normal_var"] = var_name
            st.info(f"✔️ {file_normal.name} 업로드 완료 (변수명: {var_name}, 크기: {len(data_arr):,}샘플)")

with up_col2:
    st.markdown(f"#### Class 2: 이상 데이터 (<span style='color:#f43f5e;'>{st.session_state['anomaly_filename']}</span>)", unsafe_allow_html=True)
    file_anomaly = st.file_uploader("B.mat 파일을 여기에 업로드하세요", type=["mat"], label_visibility="collapsed", key="uploader_anomaly")
    if file_anomaly is not None:
        var_name, data_arr = parse_mat_file(file_anomaly)
        if data_arr is not None:
            st.session_state["anomaly_data"] = data_arr
            st.session_state["anomaly_filename"] = file_anomaly.name
            st.session_state["anomaly_var"] = var_name
            st.info(f"✔️ {file_anomaly.name} 업로드 완료 (변수명: {var_name}, 크기: {len(data_arr):,}샘플)")

# 9. 분석 수행 및 대시보드 시각화 (두 데이터가 모두 준비되었을 때 노출)
if st.session_state["normal_data"] is not None and st.session_state["anomaly_data"] is not None:
    
    n_data = st.session_state["normal_data"]
    a_data = st.session_state["anomaly_data"]
    
    # 특징 통계치 계산
    feat_n = extract_features(n_data)
    feat_a = extract_features(a_data)
    
    st.markdown("### <i class='fa-solid fa-calculator' style='color:#4f46e5;'></i> 시계열 대표 특징 통계 (Feature Summary)", unsafe_allow_html=True)
    
    # 1x6 KPI 컬럼 생성
    kpi_cols = st.columns(6)
    
    kpis = [
        ("실효값 (RMS)", "rms", "전체 진동 에너지 수준"),
        ("최대-최소 폭 (P2P)", "p2p", "최대 충격 정도"),
        ("첨도 (Kurtosis)", "kurt", "충격성 신호 유무 (정상 ~3)"),
        ("평균값 (Mean)", "mean", "센서 오프셋/편향 정도"),
        ("표준편차 (Std Dev)", "std", "신호의 분산/변동폭"),
        ("크레스트 팩터 (CF)", "cf", "피크치 대 실효치 비율")
    ]
    
    for idx, (title, key, desc) in enumerate(kpis):
        with kpi_cols[idx]:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">{title}</div>
                    <div style="margin-top: 8px;">
                        <span style="color:#64748b; font-size:0.8rem;">N:</span> <span class="metric-value-n">{feat_n[key]}</span>
                    </div>
                    <div>
                        <span style="color:#64748b; font-size:0.8rem;">A:</span> <span class="metric-value-a">{feat_a[key]}</span>
                    </div>
                    <div class="metric-desc">{desc}</div>
                </div>
            """, unsafe_allow_html=True)
            
    st.write("") # 간격 조정
    
    # 10. 메인 차트 그리드 레이아웃 (시간 영역 + 레이더 패턴)
    chart_col1, chart_col2 = st.columns([2, 1])
    
    with chart_col1:
        st.markdown(f"#### <i class='fa-solid fa-wave-square' style='color:#4f46e5;'></i> 시간 영역 파형 비교 ({st.session_state['normal_filename']} vs {st.session_state['anomaly_filename']})", unsafe_allow_html=True)
        limit_options = [500, 1000, 2000, 5000]
        sample_limit = st.selectbox("표시할 샘플 구간 수 설정", options=limit_options, index=1)
        
        sliced_n = n_data[:sample_limit]
        sliced_a = a_data[:sample_limit]
        
        # Plotly를 이용한 부드러운 반응형 라인 차트 생성
        fig_time = go.Figure()
        fig_time.add_trace(go.Scatter(
            y=sliced_n, name=f"정상 ({st.session_state['normal_filename']})",
            line=dict(color="#10b981", width=1.5)
        ))
        fig_time.add_trace(go.Scatter(
            y=sliced_a, name=f"이상 ({st.session_state['anomaly_filename']})",
            line=dict(color="#f43f5e", width=1.5)
        ))
        
        fig_time.update_layout(
            plot_bgcolor="#f8fafc",
            paper_bgcolor="#ffffff",
            margin=dict(l=40, r=20, t=10, b=30),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis=dict(title="샘플 인덱스 (n)", gridcolor="#e2e8f0"),
            yaxis=dict(title="진폭 (Amplitude)", gridcolor="#e2e8f0"),
            height=340
        )
        st.plotly_chart(fig_time, use_container_width=True)

    with chart_col2:
        st.markdown("#### <i class='fa-solid fa-circle-notch' style='color:#4f46e5;'></i> 다차원 패턴 레이더 분석", unsafe_allow_html=True)
        st.markdown("<p style='font-size:0.75rem; color:#94a3b8; margin-top:-5px;'>각 통계 지표는 정상 신호 기준 비율로 상대 정규화</p>", unsafe_allow_html=True)
        
        # 레이더 정규화 작업
        radar_categories = ['실효치(RMS)', '피크폭(P2P)', '첨도(Kurt)', '표준편차(Std)', '크레스트팩터(CF)']
        norm_n = [1.0] * 5
        norm_a = [
            feat_a["rms"] / feat_n["rms"] if feat_n["rms"] > 0 else 1.0,
            feat_a["p2p"] / feat_n["p2p"] if feat_n["p2p"] > 0 else 1.0,
            feat_a["kurt"] / feat_n["kurt"] if feat_n["kurt"] > 0 else 1.0,
            feat_a["std"] / feat_n["std"] if feat_n["std"] > 0 else 1.0,
            feat_a["cf"] / feat_n["cf"] if feat_n["cf"] > 0 else 1.0
        ]
        
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=norm_n, theta=radar_categories, fill='toself',
            name="정상 데이터", line_color="#10b981", fillcolor="rgba(16, 185, 129, 0.1)"
        ))
        fig_radar.add_trace(go.Scatterpolar(
            r=norm_a, theta=radar_categories, fill='toself',
            name="이상 데이터", line_color="#f43f5e", fillcolor="rgba(244, 63, 94, 0.1)"
        ))
        
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, showticklabels=False, gridcolor="#e2e8f0"),
                angularaxis=dict(gridcolor="#e2e8f0")
            ),
            paper_bgcolor="#ffffff",
            margin=dict(l=40, r=40, t=30, b=30),
            legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
            height=340
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    # 11. 주파수 영역 (FFT) 분석 그래프 영역
    st.markdown("#### <i class='fa-solid fa-bolt' style='color:#eab308;'></i> 고속 푸리에 변환 주파수 스펙트럼 (FFT Spectrum)", unsafe_allow_html=True)
    
    freqs_n, mags_n = compute_fft(n_data)
    freqs_a, mags_a = compute_fft(a_data)
    
    fig_fft = go.Figure()
    fig_fft.add_trace(go.Scatter(
        x=freqs_n, y=mags_n, name=f"정상 ({st.session_state['normal_filename']})",
        line=dict(color="#10b981", width=1.5), fill='tozeroy', fillcolor="rgba(16, 185, 129, 0.04)"
    ))
    fig_fft.add_trace(go.Scatter(
        x=freqs_a, y=mags_a, name=f"이상 ({st.session_state['anomaly_filename']})",
        line=dict(color="#f43f5e", width=1.5), fill='tozeroy', fillcolor="rgba(244, 63, 94, 0.04)"
    ))
    
    fig_fft.update_layout(
        plot_bgcolor="#f8fafc",
        paper_bgcolor="#ffffff",
        margin=dict(l=40, r=20, t=10, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(title="주파수 (Frequency, Hz)", gridcolor="#e2e8f0"),
        yaxis=dict(title="가속도/신호 강도 (Magnitude)", gridcolor="#e2e8f0"),
        height=320
    )
    st.plotly_chart(fig_fft, use_container_width=True)

else:
    # 데이터가 로드되지 않았을 때 디폴트 레이아웃 가이드 제시
    st.markdown("---")
    st.info("💡 분석할 정상 데이터와 이상 데이터 파일을 업로드하거나, 우측 상단의 '데모 데이터로 즉시 테스트' 버튼을 클릭하면 대시보드 전체 시각화 및 피처 분석이 부드럽게 펼쳐집니다.")
    
    # 로딩 이전 디폴트 카드 뼈대 노출 (UX 유지용)
    kpi_cols = st.columns(6)
    for title, _, desc in kpis:
        with kpi_cols[0]:
            pass
