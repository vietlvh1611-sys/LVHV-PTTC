import streamlit as st
import pandas as pd
from google import genai
from google.genai.errors import APIError
import numpy as np

# Tương thích cao nhất: System Instruction được truyền bằng cách ghép vào User Prompt

# --- Khởi tạo State cho Chatbot và Dữ liệu ---
# Lưu trữ lịch sử chat
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Xin chào! Hãy tải lên Báo cáo Tài chính của bạn để bắt đầu phân tích và trò chuyện."}]
# Lưu trữ dữ liệu đã xử lý dưới dạng Markdown để làm bối cảnh (context) cho AI
if "data_for_chat" not in st.session_state:
    st.session_state.data_for_chat = None

# --- Cấu hình Trang Streamlit ---
st.set_page_config(
    page_title="App Phân Tích Báo Cáo Tài Chính",
    layout="wide"
)

st.title("Ứng dụng Phân Tích Báo cáo Tài chính 📊")

# === [V17] ĐỊNH NGHĨA CÁC HÀM ĐỊNH DẠNG TÙY CHỈNH THEO CHUẨN VIỆT NAM (., phân cách) ===
def format_vn_currency(val):
    # Định dạng tiền tệ (hàng đơn vị), dot là ngàn, comma là thập phân. Ẩn 0.
    if pd.isna(val) or (val == 0): 
        return "" 
    val = round(val)
    # Định dạng số nguyên
    return "{:,d}".format(val).replace(",", "TEMP_SEP").replace(".", ",").replace("TEMP_SEP", ".")

def format_vn_percentage(val):
    # Định dạng tỷ lệ (1 chữ số thập phân), dot là ngàn, comma là thập phân. Ẩn 0.
    if pd.isna(val) or (val == 0):
        return ""
    val = round(val, 1)
    # Định dạng Tỷ lệ % từ 1 chữ số thập phân
    formatted_val = "{:,.1f}".format(val).replace(",", "TEMP_SEP").replace(".", ",").replace("TEMP_SEP", ".") + "%"
    return formatted_val

def format_vn_delta_currency(val):
    # Loại bỏ dấu + khi số dương. Chỉ hiển thị dấu - khi số âm.
    if pd.isna(val):
        return ""
    val = round(val)
    
    # Định dạng số nguyên: Chỉ dùng '-' khi âm, không dùng '+' khi dương.
    if val < 0:
        # Sử dụng abs() để định dạng số dương, sau đó thêm dấu '-' thủ công
        formatted_val = "-{:,d}".format(abs(val))
    else:
        formatted_val = "{:,d}".format(val)
        
    return formatted_val.replace(",", "TEMP_SEP").replace(".", ",").replace("TEMP_SEP", ".")

def format_vn_delta_ratio(val):
    # Loại bỏ dấu + khi số dương. Giữ 2 chữ số thập phân (cho độ chính xác so sánh).
    if pd.isna(val) or (val == 0):
        return ""
    val = round(val, 2)
    
    # Định dạng số thập phân: Chỉ dùng '-' khi âm, không dùng '+' khi dương.
    if val < 0:
        # Sử dụng abs() để định dạng số dương, sau đó thêm dấu '-' thủ công
        formatted_val = "-{:.2f}".format(abs(val)).replace(".", ",")
    else:
        formatted_val = "{:.2f}".format(val).replace(".", ",")
        
    # Định dạng lại để dùng dấu phẩy cho thập phân
    return formatted_val
# === KẾT THÚC ĐỊNH NGHĨA FORMATTERS ===

# === [V16] ĐỊNH NGHĨA HÀM STYLING CHO CÁC CHỈ TIÊU CHÍNH/PHỤ ===
def highlight_financial_items(row):
    """Áp dụng in đậm cho mục chính (A, I, TỔNG CỘNG) và in nghiêng cho mục chi tiết (Nguyên giá, Hao mòn)."""
    styles = [''] * len(row)
    item = str(row['Chỉ tiêu']).strip()
    
    # 1. In đậm cho mục chính và tổng cộng
    is_major_section = (
        item.startswith(('A.', 'B.', 'C.')) or 
        item.startswith(('I.', 'II.', 'III.', 'IV.', 'V.', 'VI.', 'VII.', 'VIII.', 'IX.', 'X.')) or
        'TỔNG CỘNG' in item.upper() or
        'TỔNG CỘNG TÀI SẢN' in item.upper() or
        'TỔNG CỘNG NGUỒN VỐN' in item.upper() or
        'NỢ PHẢI TRẢ' in item.upper() or
        'VỐN CHỦ SỞ HỮU' in item.upper() or
        # BỔ SUNG: Cho các tiêu đề chính trong bảng chỉ tiêu tài chính
        item in ['Khả năng thanh toán', 'Chỉ tiêu hoạt động', 'Chỉ tiêu cân nợ', 'Hệ số sinh lời']
    )
    
    # 2. In nghiêng cho mục chi tiết TSCĐ
    is_italic_item = (
        'Nguyên giá' in item or 
        'Giá trị hao mòn lũy kế' in item
    )
    
    if is_major_section:
        styles = ['font-weight: bold'] * len(row)
    
    elif is_italic_item:
        styles = ['font-style: italic'] * len(row)
        
    return styles
# === KẾT THÚC [V16] HÀM STYLING ===

# === [FIX] HÀM HỖ TRỢ TÍNH TOÁN (DI CHUYỂN RA NGOÀI VÀ SỬA LỖI) ===

def get_value(df, keyword, year):
    """Lấy giá trị số (float) từ DataFrame, xử lý NaN và lỗi."""
    row = df[df['Chỉ tiêu'].str.contains(keyword, case=False, na=False)]
    if row.empty:
        return 0
        
    # 1. Lấy giá trị đầu tiên, đảm bảo chuyển nó thành số (numeric)
    value = pd.to_numeric(row[year].iloc[0], errors='coerce') 
    
    # 2. [FIX] Nếu giá trị là NaN, thay bằng 0. Nếu không, giữ nguyên.
    # (pd.isna() hoạt động chính xác trên numpy.float64)
    return 0.0 if pd.isna(value) else float(value)

def safe_div(numerator, denominator):
    """Hàm chia an toàn, xử lý chia cho 0 hoặc NaN."""
    # Trả về 0 nếu mẫu số là 0 hoặc NaN.
    if denominator == 0 or pd.isna(denominator) or denominator == np.nan: 
        return 0.0 
    
    result = float(numerator) / float(denominator)
    
    # Trường hợp chia số âm cho số rất nhỏ, dẫn đến số rất lớn (Inf/-Inf)
    if np.isinf(result) or np.isneginf(result):
         return 0.0 
    return result

# === KẾT THÚC HÀM HỖ TRỢ ===


# --- Hàm tính toán chính (Sử dụng Caching để Tối ưu hiệu suất) ---
@st.cache_data
def process_financial_data(df_balance_sheet, df_income_statement):
    """
    Thực hiện các phép tính Tăng trưởng, So sánh Tuyệt đối, Tỷ trọng Cơ cấu, Tỷ trọng Chi phí/DT thuần và Chỉ số Tài chính.
    [CẬP NHẬT] Bổ sung Vòng quay Phải thu, Vòng quay VLĐ, ROS, ROA, ROE.
    [CẬP NHẬT] Sắp xếp lại df_final_ratios: Thanh toán -> Hoạt động -> Cân nợ -> Sinh lời.
    Trả về tuple (df_bs_processed, df_is_processed, df_ratios_processed, df_final_ratios)
    """
    
    df_bs = df_balance_sheet.copy()
    df_is = df_income_statement.copy()
    years = ['Năm 1', 'Năm 2', 'Năm 3']
    
    # Đảm bảo các giá trị là số để tính toán (trước khi gọi get_value)
    for df in [df_bs, df_is]:
        if not df.empty:
            for col in years:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # -----------------------------------------------------------------
    # PHẦN 1: XỬ LÝ BẢNG CÂN ĐỐI KẾ TOÁN (BALANCE SHEET - BS)
    # -----------------------------------------------------------------
    if not df_bs.empty:
        df_bs['Delta (Y2 vs Y1)'] = df_bs['Năm 2'] - df_bs['Năm 1']
        df_bs['Growth (Y2 vs Y1)'] = ((df_bs['Delta (Y2 vs Y1)'] / df_bs['Năm 1'].replace(0, 1e-9)) * 100)
        df_bs['Delta (Y3 vs Y2)'] = df_bs['Năm 3'] - df_bs['Năm 2']
        df_bs['Growth (Y3 vs Y2)'] = ((df_bs['Delta (Y3 vs Y2)'] / df_bs['Năm 2'].replace(0, 1e-9)) * 100)

        # Tính Tỷ trọng theo Tổng Tài sản
        tong_tai_san_row = df_bs[df_bs['Chỉ tiêu'].str.contains('TỔNG CỘNG TÀI SẢN|TỔNG CỘNG', case=False, na=False)]
        
        tong_tai_san_N1 = tong_tai_san_row['Năm 1'].iloc[0] if not tong_tai_san_row.empty else 1e-9
        tong_tai_san_N2 = tong_tai_san_row['Năm 2'].iloc[0] if not tong_tai_san_row.empty else 1e-9
        tong_tai_san_N3 = tong_tai_san_row['Năm 3'].iloc[0] if not tong_tai_san_row.empty else 1e-9

        divisor_N1 = tong_tai_san_N1 if tong_tai_san_N1 != 0 else 1e-9
        divisor_N2 = tong_tai_san_N2 if tong_tai_san_N2 != 0 else 1e-9
        divisor_N3 = tong_tai_san_N3 if tong_tai_san_N3 != 0 else 1e-9

        df_bs['Tỷ trọng Năm 1 (%)'] = (df_bs['Năm 1'] / divisor_N1) * 100
        df_bs['Tỷ trọng Năm 2 (%)'] = (df_bs['Năm 2'] / divisor_N2) * 100
        df_bs['Tỷ trọng Năm 3 (%)'] = (df_bs['Năm 3'] / divisor_N3) * 100
    
    # -----------------------------------------------------------------
    # PHẦN 2 & 3: XỬ LÝ KQKD & TỶ TRỌNG CHI PHÍ / DOANH THU THUẦN
    # -----------------------------------------------------------------
    if not df_is.empty:
        df_is['S.S Tuyệt đối (Y2 vs Y1)'] = df_is['Năm 2'] - df_is['Năm 1']
        df_is['S.S Tương đối (%) (Y2 vs Y1)'] = ((df_is['S.S Tuyệt đối (Y2 vs Y1)'] / df_is['Năm 1'].replace(0, 1e-9)) * 100)
        
        df_is['S.S Tuyệt đối (Y3 vs Y2)'] = df_is['Năm 3'] - df_is['Năm 2']
        df_is['S.S Tương đối (%) (Y3 vs Y2)'] = ((df_is['S.S Tuyệt đối (Y3 vs Y2)'] / df_is['Năm 2'].replace(0, 1e-9)) * 100)
    
    # Tính Tỷ trọng Chi phí/DT Thuần (df_ratios)
    df_ratios = pd.DataFrame(columns=['Chỉ tiêu', 'Năm 1', 'Năm 2', 'Năm 3'])
    if not df_is.empty:
        dt_thuan_row = df_is[df_is['Chỉ tiêu'].str.contains('Doanh thu thuần về bán hàng', case=False, na=False)]
        
        if not dt_thuan_row.empty:
            DT_thuan_N1 = dt_thuan_row['Năm 1'].iloc[0] if dt_thuan_row['Năm 1'].iloc[0] != 0 else 1e-9
            DT_thuan_N2 = dt_thuan_row['Năm 2'].iloc[0] if dt_thuan_row['Năm 2'].iloc[0] != 0 else 1e-9
            DT_thuan_N3 = dt_thuan_row['Năm 3'].iloc[0] if dt_thuan_row['Năm 3'].iloc[0] != 0 else 1e-9
            divisors = [DT_thuan_N1, DT_thuan_N2, DT_thuan_N3]
            
            ratio_mapping = {
                'Giá vốn hàng bán': 'Giá vốn hàng bán',
                'Chi phí lãi vay': 'Trong đó: Chi phí lãi vay', 
                'Chi phí Bán hàng': 'Chi phí bán hàng', 
                'Chi phí Quản lý doanh nghiệp': 'Chi phí quản lý doanh nghiệp',
                'Lợi nhuận sau thuế': 'Lợi nhuận sau thuế TNDN'
            }
            
            data_ratio_is = []
            for ratio_name, search_keyword in ratio_mapping.items():
                row = df_is[df_is['Chỉ tiêu'].str.contains(search_keyword, case=False, na=False)]
                if not row.empty:
                    ratios = [0, 0, 0]
                    for i, year in enumerate(years):
                        value = row[year].iloc[0]
                        ratios[i] = (value / divisors[i]) * 100
                    data_ratio_is.append([ratio_name] + ratios)

            df_ratios = pd.DataFrame(data_ratio_is, columns=['Chỉ tiêu', 'Năm 1', 'Năm 2', 'Năm 3'])
            df_ratios['S.S Tương đối (%) (Y2 vs Y1)'] = df_ratios['Năm 2'] - df_ratios['Năm 1']

    # -----------------------------------------------------------------
    # PHẦN 4: TÍNH TẤT CẢ CÁC CHỈ SỐ TÀI CHÍNH MỚI/CŨ
    # -----------------------------------------------------------------
    
    # Lấy các giá trị cần thiết từ Bảng CĐKT (BS) và KQKD (IS) - SỬ DỤNG HÀM GET_VALUE ĐÃ FIX
    data = {}
    data['TSNH'] = {y: get_value(df_bs, 'Tài sản ngắn hạn|TS ngắn hạn', y) for y in years}
    data['NO_NGAN_HAN'] = {y: get_value(df_bs, 'Nợ ngắn hạn', y) for y in years} 
    data['HTK'] = {y: get_value(df_bs, 'Hàng tồn kho|HTK', y) for y in years}
    data['GVHB'] = {y: get_value(df_is, 'Giá vốn hàng bán', y) for y in years} 
    data['VCSH'] = {y: get_value(df_bs, 'Vốn chủ sở hữu', y) for y in years}
    data['NPT'] = {y: get_value(df_bs, 'Nợ phải trả', y) for y in years}
    data['TTS'] = {y: get_value(df_bs, 'TỔNG CỘNG TÀI SẢN|TỔNG CỘNG NGUỒN VỐN|TỔNG CỘNG', y) for y in years}
    data['LNST'] = {y: get_value(df_is, 'Lợi nhuận sau thuế TNDN', y) for y in years}
    data['DT_THUAN'] = {y: get_value(df_is, 'Doanh thu thuần về bán hàng', y) for y in years}
    data['PHAI_THU'] = {y: get_value(df_bs, 'Các khoản phải thu ngắn hạn|Phải thu khách hàng', y) for y in years} 
    
    # --- KHỞI TẠO DATAFRAME CHỈ SỐ ---
    ratios_list = []
    
    for i, y in enumerate(years):
        # Lấy giá trị đầu kỳ/cuối kỳ
        tts_current = data['TTS'][y]
        tts_previous = data['TTS'][years[i-1]] if i > 0 else tts_current
        avg_tts = safe_div(tts_current + tts_previous, 2)

        vcsh_current = data['VCSH'][y]
        vcsh_previous = data['VCSH'][years[i-1]] if i > 0 else vcsh_current
        avg_vcsh = safe_div(vcsh_current + vcsh_previous, 2)

        tsnh = data['TSNH'][y]
        nnh = data['NO_NGAN_HAN'][y]
        htk = data['HTK'][y]
        gvhb = data['GVHB'][y]
        lnst = data['LNST'][y]
        dt_thuan = data['DT_THUAN'][y]
        npt = data['NPT'][y]
        
        # Hàng tồn kho BQ
        htk_previous = data['HTK'][years[i-1]] if i > 0 else htk
        avg_inventory = safe_div(htk + htk_previous, 2)
        
        # Phải thu BQ
        pt_current = data['PHAI_THU'][y]
        pt_previous = data['PHAI_THU'][years[i-1]] if i > 0 else pt_current
        avg_receivable = safe_div(pt_current + pt_previous, 2)
        
        # Vốn lưu động BQ
        wl_current = tsnh - nnh
        wl_previous = (data['TSNH'][years[i-1]] - data['NO_NGAN_HAN'][years[i-1]]) if i > 0 else wl_current
        avg_working_capital = safe_div(wl_current + wl_previous, 2)

        # ---------------------------------------------------
        # TÍNH TOÁN CÁC CHỈ SỐ (Sử dụng safe_div đã fix)
        # ---------------------------------------------------

        # Thanh toán
        current_ratio = safe_div(tsnh, nnh)
        quick_ratio = safe_div(tsnh - htk, nnh)

        # Hoạt động
        inv_turnover = safe_div(gvhb, avg_inventory)
        inv_days = safe_div(365, inv_turnover) # (safe_div xử lý inv_turnover = 0)
        
        rcv_turnover = safe_div(dt_thuan, avg_receivable)
        rcv_days = safe_div(365, rcv_turnover) # (safe_div xử lý rcv_turnover = 0)
        
        wcl_turnover = safe_div(dt_thuan, avg_working_capital)

        # Cân nợ (Solvency/Leverage)
        equity_ratio = safe_div(vcsh_current, tts_current) # Sửa VCSH -> vcsh_current
        d_to_e_ratio = safe_div(npt, vcsh_current) # Sửa VCSH -> vcsh_current
        
        # Sinh lời (Profitability)
        ros_ratio = safe_div(lnst, dt_thuan) * 100 
        roa_ratio = safe_div(lnst, avg_tts) * 100
        
        # Xử lý ROE khi VCSH <= 0 (Sử dụng np.nan để format sau)
        if avg_vcsh <= 0:
             roe_ratio = np.nan # Đánh dấu là NaN để hiển thị rõ (format_vn_delta_ratio sẽ xử lý)
        else:
             roe_ratio = safe_div(lnst, avg_vcsh) * 100


        # Thêm dữ liệu vào list (Theo thứ tự mới)
        ratios_list.append({
            'Chỉ tiêu': 'Hệ số Thanh toán ngắn hạn (Current Ratio)', y: current_ratio, 'Type': 'Liquidity'
        })
        ratios_list.append({
