import streamlit as st
import numpy as np
import pandas as pd
import struct
import zlib

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

# 2. 전역 통계 지표 정보 정의
kpis = [
    ("실효값 (RMS)", "rms", "전체 진동 에너지 수준"),
    ("최대-최소 폭 (P2P)", "p2p", "최대 충격 정도"),
    ("첨도 (Kurtosis)", "kurt", "충격성 신호 유무 (정상 ~3)"),
    ("평균값 (Mean)", "mean", "센서 오프셋/편향 정도"),
    ("표준편차 (Std Dev)", "std", "신호의 분산/변동폭"),
    ("크레스트 팩터 (CF)", "cf", "피크치 대 실효치 비율")
]

# 3. 순수 NumPy 기반 통계 피처 추출 함수 정의 (scipy 의존성 제거)
def extract_features(signal):
    if signal is None or len(signal) == 0:
        return {"mean": 0, "rms": 0, "std": 0, "p2p": 0, "kurt": 3.0, "cf": 0}
    
    mean_val = np.mean(signal)
    rms_val = np.sqrt(np.mean(signal**2))
    std_val = np.std(signal)
    p2p_val = np.max(signal) - np.min(signal)
    
    # 첨도(Kurtosis) 수동 연산 (Pearson 정의: 정규분포 = 3.0)
    variance = np.var(signal)
    if variance > 0:
        m4 = np.mean((signal - mean_val) ** 4)
        kurt_val = m4 / (variance ** 2)
    else:
        kurt_val = 3.0
    
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

# 4. FFT 연산 함수 정의
def compute_fft(signal, fs=12000, n_fft=512):
    truncated = signal[:n_fft]
    yf = np.fft.fft(truncated)
    mags = (2.0 / n_fft) * np.abs(yf[:n_fft // 2])
    freqs = np.fft.fftfreq(n_fft, 1 / fs)[:n_fft // 2]
    return freqs, mags

# 5. 순수 파이썬 기반 MATLAB .mat v5 파일 바이너리 고성능 파서 (scipy.io 대체)
def parse_mat_file_pure_python(uploaded_file):
    try:
        uploaded_file.seek(0)
        data = uploaded_file.read()
        
        if len(data) < 128:
            return None, None
            
        header_text = data[0:116].decode('utf-8', errors='ignore')
        if "MATLAB 5.0" not in header_text:
            return None, None
            
        # 엔디안 인디케이터 판별 (126-127 바이트)
        endian_indicator = data[126:128]
        is_le = (endian_indicator == b'IM')
        
        # 내부 구조 재귀 파싱 도우미
        def parse_elements(buf, start_offset, end_offset):
            offset = start_offset
            endian_char = '<' if is_le else '>'
            
            while offset < end_offset:
                if offset + 8 > end_offset:
                    break
                
                tag_type, tag_bytes = struct.unpack_from(f'{endian_char}II', buf, offset)
                
                # Small Data Element (SDE) 식별
                is_sde = (tag_type >> 16) != 0
                if is_sde:
                    actual_type = tag_type & 0xFFFF
                    actual_bytes = (tag_type >> 16) & 0xFFFF
                    tag_len = 4
                else:
                    actual_type = tag_type
                    actual_bytes = tag_bytes
                    tag_len = 8
                    
                if offset + tag_len + actual_bytes > end_offset:
                    break
                
                # miCOMPRESSED (Type 15) -> zlib 압축 해제 후 하위 레이어 재귀 파싱
                if actual_type == 15:
                    compressed_data = buf[offset + tag_len : offset + tag_len + actual_bytes]
                    try:
                        decompressed = zlib.decompress(compressed_data)
                        res = parse_elements(decompressed, 0, len(decompressed))
                        if res is not None:
                            return res
                    except:
                        pass
                
                # miMATRIX (Type 14) -> 배열 정보 및 수치 벡터 추출
                elif actual_type == 14:
                    matrix_data = buf[offset + tag_len : offset + tag_len + actual_bytes]
                    sub_offset = 0
                    array_name = "unknown"
                    numerical_data = None
                    
                    while sub_offset < len(matrix_data):
                        if sub_offset + 8 > len(matrix_data):
                            break
                        sub_tag_type, sub_tag_bytes = struct.unpack_from(f'{endian_char}II', matrix_data, sub_offset)
                        
                        sub_is_sde = (sub_tag_type >> 16) != 0
                        if sub_is_sde:
                            sub_act_type = sub_tag_type & 0xFFFF
                            sub_act_bytes = (sub_tag_type >> 16) & 0xFFFF
                            sub_tag_len = 4
                        else:
                            sub_act_type = sub_tag_type
                            sub_act_bytes = sub_tag_bytes
                            sub_tag_len = 8
                            
                        if sub_offset + sub_tag_len + sub_act_bytes > len(matrix_data):
                            break
                            
                        val_bytes = matrix_data[sub_offset + sub_tag_len : sub_offset + sub_tag_len + sub_act_bytes]
                        
                        # miINT8 (1) -> 변수명 추출
                        if sub_act_type == 1 and 0 < sub_act_bytes < 64:
                            try:
                                possible_name = val_bytes.decode('utf-8', errors='ignore').strip('\x00').strip()
                                if possible_name.isidentifier():
                                    array_name = possible_name
                            except:
                                pass
                        
                        # miDOUBLE (9) / miSINGLE (7) -> 대량 부동 소수점 수치 어레이 복원
                        elif sub_act_type in (7, 9) and sub_act_bytes > 100:
                            dtype = np.float64 if sub_act_type == 9 else np.float32
                            np_dtype = dtype if is_le else dtype.newbyteorder('>')
                            numerical_data = np.frombuffer(val_bytes, dtype=np_dtype).copy()
                            
                        sub_padding = 0 if sub_is_sde else ((8 - (sub_act_bytes % 8)) % 8)
                        sub_offset += sub_tag_len + sub_act_bytes + sub_padding
                        
                    if numerical_data is not None and len(numerical_data) > 100:
                        return array_name, numerical_data
                
                padding = 0 if is_sde else ((8 - (actual_bytes % 8)) % 8)
                offset += tag_len + actual_bytes + padding
            return None, None
            
        return parse_elements(data, 128, len(data))
    except Exception as e:
        st.error(f"MAT 파일 파싱 에러: {str(e)}")
        return None, None

# 6. 데모 데이터 생성 핸들러
def generate_demo_data(is_anomaly=False, fs=12000, length=12000):
    t = np.arange(length) / fs
    if not is_anomaly:
        signal = 1.2 * np.sin(2 * np.pi * 60 * t) + 0.6 * np.sin(2 * np.pi * 150 * t) + np.random.normal(0, 0.2, length)
    else:
        impact = np.exp(-((np.arange(length) % 600) / 100)) * np.sin(2 * np.pi * 320 * t) * 6.0
        signal = 1.0 * np.sin(2 * np.pi * 60 * t) + impact + np.random.normal(0, 0.9, length)
    return signal

# 세션 상태 및 파일 트래킹 변수 초기화
if "normal_data" not in st.session_state:
    st.session_state["normal_data"] = None
    st.session_state["normal_filename"] = "Normal.mat"
    st.session_state["normal_var"] = "N/A"
    st.session_state["last_processed_normal"] = None
if "anomaly_data" not in st.session_state:
    st.session_state["anomaly_data"] = None
    st.session_state["anomaly_filename"] = "B.mat"
    st.session_state["anomaly_var"] = "N/A"
    st.session_state["last_processed_anomaly"] = None

# 7. 헤더 영역 구축
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

# 8. 안내 가이드 및 데모 활성화 영역
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
        st.session_state["last_processed_normal"] = "Demo_Normal_60Hz.mat"
        st.session_state["anomaly_data"] = generate_demo_data(is_anomaly=True)
        st.session_state["anomaly_filename"] = "Demo_Fault_Bearing_320Hz.mat"
        st.session_state["anomaly_var"] = "Demo_Anomaly"
        st.session_state["last_processed_anomaly"] = "Demo_Fault_Bearing_320Hz.mat"
        st.success("데모 데이터가 성공적으로 탑재되었습니다!")

# 9. 파일 업로더 레이아웃 구성 및 실시간 동기화 플레이스홀더 연동
up_col1, up_col2 = st.columns(2)

with up_col1:
    header_normal_placeholder = st.empty()
    file_normal = st.file_uploader("Normal.mat 파일을 여기에 업로드하세요", type=["mat"], label_visibility="collapsed", key="uploader_normal")
    
    if file_normal is not None:
        if st.session_state.get("last_processed_normal") != file_normal.name:
            var_name, data_arr = parse_mat_file_pure_python(file_normal)
            if data_arr is not None:
                st.session_state["normal_data"] = data_arr
                st.session_state["normal_filename"] = file_normal.name
                st.session_state["normal_var"] = var_name
                st.session_state["last_processed_normal"] = file_normal.name
    else:
        last_normal = st.session_state.get("last_processed_normal")
        if last_normal is not None and not last_normal.startswith("Demo_"):
            st.session_state["normal_data"] = None
            st.session_state["normal_filename"] = "Normal.mat"
            st.session_state["normal_var"] = "N/A"
            st.session_state["last_processed_normal"] = None
            
    header_normal_placeholder.markdown(f"#### Class 1: 정상 데이터 (<span style='color:#10b981;'>{st.session_state['normal_filename']}</span>)", unsafe_allow_html=True)
    
    if st.session_state["normal_data"] is not None:
        st.info(f"✔️ {st.session_state['normal_filename']} 로드 완료 (변수명: {st.session_state['normal_var']}, 크기: {len(st.session_state['normal_data']):,}샘플)")

with up_col2:
    header_anomaly_placeholder = st.empty()
    file_anomaly = st.file_uploader("B.mat 파일을 여기에 업로드하세요", type=["mat"], label_visibility="collapsed", key="uploader_anomaly")
    
    if file_anomaly is not None:
        if st.session_state.get("last_processed_anomaly") != file_anomaly.name:
            var_name, data_arr = parse_mat_file_pure_python(file_anomaly)
            if data_arr is not None:
                st.session_state["anomaly_data"] = data_arr
                st.session_state["anomaly_filename"] = file_anomaly.name
                st.session_state["anomaly_var"] = var_name
                st.session_state["last_processed_anomaly"] = file_anomaly.name
    else:
        last_anomaly = st.session_state.get("last_processed_anomaly")
        if last_anomaly is not None and not last_anomaly.startswith("Demo_"):
            st.session_state["anomaly_data"] = None
            st.session_state["anomaly_filename"] = "B.mat"
            st.session_state["anomaly_var"] = "N/A"
            st.session_state["last_processed_anomaly"] = None

    header_anomaly_placeholder.markdown(f"#### Class 2: 이상 데이터 (<span style='color:#f43f5e;'>{st.session_state['anomaly_filename']}</span>)", unsafe_allow_html=True)
    
    if st.session_state["anomaly_data"] is not None:
        st.info(f"✔️ {st.session_state['anomaly_filename']} 로드 완료 (변수명: {st.session_state['anomaly_var']}, 크기: {len(st.session_state['anomaly_data']):,}샘플)")

# 10. 분석 수행 및 대시보드 시각화
if st.session_state["normal_data"] is not None and st.session_state["anomaly_data"] is not None:
    
    n_data = st.session_state["normal_data"]
    a_data = st.session_state["anomaly_data"]
    
    feat_n = extract_features(n_data)
    feat_a = extract_features(a_data)
    
    st.markdown("### <i class='fa-solid fa-calculator' style='color:#4f46e5;'></i> 시계열 대표 특징 통계 (Feature Summary)", unsafe_allow_html=True)
    
    kpi_cols = st.columns(6)
    
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
            
    st.write("")
    
    chart_col1, chart_col2 = st.columns([2, 1])
    
    # 시간 영역 그래프 영역 (스트림릿 내장 고성능 st.line_chart 대체)
    with chart_col1:
        st.markdown(f"#### <i class='fa-solid fa-wave-square' style='color:#4f46e5;'></i> 시간 영역 파형 비교 ({st.session_state['normal_filename']} vs {st.session_state['anomaly_filename']})", unsafe_allow_html=True)
        limit_options = [500, 1000, 2000, 5000]
        sample_limit = st.selectbox("표시할 샘플 구간 수 설정", options=limit_options, index=1)
        
        sliced_n = n_data[:sample_limit]
        sliced_a = a_data[:sample_limit]
        
        # Pandas DataFrame 생성 및 선그리기 구현
        df_time_chart = pd.DataFrame({
            f"정상 ({st.session_state['normal_filename']})": sliced_n,
            f"이상 ({st.session_state['anomaly_filename']})": sliced_a
        })
        st.line_chart(df_time_chart, height=340, color=["#10b981", "#f43f5e"])

    # 다차원 패턴 비교 대칭 막대 그래프 영역 (st.bar_chart 대체)
    with chart_col2:
        st.markdown("#### <i class='fa-solid fa-chart-bar' style='color:#4f46e5;'></i> 다차원 패턴 특징 분석 (배율)", unsafe_allow_html=True)
        st.markdown("<p style='font-size:0.75rem; color:#94a3b8; margin-top:-5px;'>각 통계 지표는 정상 신호 기준(=1.0)으로 상대 정규화</p>", unsafe_allow_html=True)
        
        categories = ['실효치(RMS)', '피크폭(P2P)', '첨도(Kurt)', '표준편차(Std)', '크레스트팩터(CF)']
        norm_n = [1.0] * 5
        norm_a = [
            feat_a["rms"] / feat_n["rms"] if feat_n["rms"] > 0 else 1.0,
            feat_a["p2p"] / feat_n["p2p"] if feat_n["p2p"] > 0 else 1.0,
            feat_a["kurt"] / feat_n["kurt"] if feat_n["kurt"] > 0 else 1.0,
            feat_a["std"] / feat_n["std"] if feat_n["std"] > 0 else 1.0,
            feat_a["cf"] / feat_n["cf"] if feat_n["cf"] > 0 else 1.0
        ]
        
        # 사이드 바이 사이드 막대 데이터프레임 구조화
        df_pattern_chart = pd.DataFrame({
            "정상": norm_n,
            "이상": norm_a
        }, index=categories)
        
        st.bar_chart(df_pattern_chart, height=340, color=["#10b981", "#f43f5e"])

    # 주파수 영역 (FFT) 영역 누적 그래프 (스트림릿 내장 고성능 st.area_chart 대체)
    st.markdown("#### <i class='fa-solid fa-bolt' style='color:#eab308;'></i> 고속 푸리에 변환 주파수 스펙트럼 (FFT Spectrum)", unsafe_allow_html=True)
    
    freqs_n, mags_n = compute_fft(n_data)
    freqs_a, mags_a = compute_fft(a_data)
    
    # x축 눈금 매핑을 위해 인덱스를 Hertz 단위 정수로 세팅
    df_fft_chart = pd.DataFrame({
        f"정상 ({st.session_state['normal_filename']})": mags_n,
        f"이상 ({st.session_state['anomaly_filename']})": mags_a
    }, index=np.round(freqs_n).astype(int))
    
    st.area_chart(df_fft_chart, height=320, color=["#10b981", "#f43f5e"])

else:
    st.markdown("---")
    st.info("💡 분석할 정상 데이터와 이상 데이터 파일을 업로드하거나, 우측 상단의 '데모 데이터로 즉시 테스트' 버튼을 클릭하면 대시보드 전체 시각화 및 피처 분석이 부드럽게 펼쳐집니다.")
    
    # 초기 대기 모드에서 빈 수치로 구성된 미려한 투명 스켈레톤 카드 렌더링
    st.markdown("### <i class='fa-solid fa-calculator' style='color:#cbd5e1;'></i> 대표 특징 통계 (대기 중)", unsafe_allow_html=True)
    kpi_cols = st.columns(6)
    for idx, (title, _, desc) in enumerate(kpis):
        with kpi_cols[idx]:
            st.markdown(f"""
                <div class="metric-card" style="opacity: 0.55; border-style: dashed;">
                    <div class="metric-title">{title}</div>
                    <div style="margin-top: 8px;">
                        <span style="color:#94a3b8; font-size:0.8rem;">N:</span> <span class="metric-value-n" style="color:#94a3b8;">-</span>
                    </div>
                    <div>
                        <span style="color:#94a3b8; font-size:0.8rem;">A:</span> <span class="metric-value-a" style="color:#94a3b8;">-</span>
                    </div>
                    <div class="metric-desc">{desc}</div>
                </div>
            """, unsafe_allow_html=True)
