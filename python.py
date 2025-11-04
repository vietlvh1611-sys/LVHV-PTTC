import streamlit as st
import pandas as pd
from google import genai
from google.genai.errors import APIError
import numpy as np
import io

# Tương thích cao nhất: System Instruction được truyền bằng cách ghép vào User Prompt

# --- Khởi tạo State cho Chatbot và Dữ liệu ---
# Lưu trữ lịch sử chat
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Xin chào! Hãy tải lên Báo cáo Tài chính của bạn để bắt đầu phân tích và trò chuyện."}]
# Lưu trữ dữ liệu đã xử lý dưới dạng Markdown để làm bối cảnh (context) cho AI
if "data_for_chat" not in st.session_state:
    st.session_state.data_for_chat = None
# Lưu trữ các DataFrame đã xử lý cho việc tạo báo cáo (Mục 7)
if "processed_dataframes" not in st.session_state:
    st.session_state.processed_dataframes = {}

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

# === HÀM HỖ TRỢ TÍNH TOÁN ===

def get_value(df, keyword, year):
    """Lấy giá trị số (float) từ DataFrame, xử lý NaN và lỗi."""
    row = df[df['Chỉ tiêu'].str.contains(keyword, case=False, na=False)]
    if row.empty:
        return 0
        
    # 1. Lấy giá trị đầu tiên, đảm bảo chuyển nó thành số (numeric)
    value = pd.to_numeric(row[year].iloc[0], errors='coerce') 
    
    # 2. Nếu giá trị là NaN, thay bằng 0.
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

def get_top_items(df, parent_keyword, year, n=2, is_asset=True):
    """Tìm 2 khoản mục lớn nhất trong Tài sản Ngắn hạn/Dài hạn (hoặc nguồn vốn) theo tỷ trọng."""
    
    # Lọc các khoản mục con, loại bỏ các dòng tổng cộng/phụ đề.
    # Đây là logic giả định, cần điều chỉnh dựa trên cấu trúc file Excel thực tế
    
    # 1. Tìm tổng của khoản mục cha (ví dụ: Tổng TSNH)
    parent_row = df[df['Chỉ tiêu'].str.contains(parent_keyword, case=False, na=False)]
    if parent_row.empty or parent_row[year].iloc[0] == 0:
        return [('N/A', 0, '0.0%'), ('N/A', 0, '0.0%')]
    
    parent_value = parent_row[year].iloc[0]
    
    # 2. Lọc các khoản mục con (giả định chúng nằm gần khoản mục cha)
    # Lấy index của khoản mục cha
    parent_index = parent_row.index[0]
    
    # Chỉ lấy các khoản mục con ngay sau đó (giả định khoảng 10-15 dòng)
    df_slice = df.loc[parent_index + 1 : parent_index + 15].copy()
    
    # Loại bỏ các dòng trống/dòng tổng cộng con/các dòng chứa chữ "TỔNG" hoặc chữ cái đầu (A, B, C...)
    df_slice = df_slice[~df_slice['Chỉ tiêu'].str.startswith(('A', 'B', 'C', 'I', 'II', 'III', 'TỔNG'))].copy()
    df_slice = df_slice[df_slice[year] != 0].copy()
    
    if df_slice.empty:
        return [('N/A', 0, '0.0%'), ('N/A', 0, '0.0%')]

    # 3. Tính toán tỷ trọng và sắp xếp
    df_slice['Ratio'] = df_slice[year] / parent_value
    
    # Sắp xếp theo giá trị tuyệt đối giảm dần
    df_slice['AbsValue'] = df_slice[year].abs()
    df_slice = df_slice.sort_values(by='AbsValue', ascending=False)
    
    # 4. Trả về kết quả
    results = []
    for _, row in df_slice.head(n).iterrows():
        item_name = str(row['Chỉ tiêu']).strip()
        item_value = row[year]
        item_pct = "{:.1f}%".format(row['Ratio'] * 100)
        results.append((item_name, format_vn_currency(item_value), item_pct))
        
    # Đảm bảo luôn trả về đủ n mục
    while len(results) < n:
        results.append(('N/A', '', ''))
        
    return results[:n]


# === ÁNH XẠ DỮ LIỆU TỪ DATAFRAME VÀO PLACEHOLDERS ===
def map_data_to_placeholders(df_bs_proc, df_is_proc, y1_name, y2_name, y3_name, col_names):
    
    # 1. Khởi tạo dictionary chứa các cặp {placeholder: value}
    placeholders = {}
    
    # Dùng list years nội bộ để dễ truy cập
    years = ['Năm 1', 'Năm 2', 'Năm 3']
    
    # Lấy giá trị từ Bảng CĐKT
    def get_bs_val(keyword, year_key):
        return get_value(df_bs_proc, keyword, year_key)

    # -----------------------------------------------------
    # Lấy dữ liệu thô (đã được định dạng)
    # -----------------------------------------------------
    
    # TTS
    TTS_Y1 = get_bs_val('TỔNG CỘNG TÀI SẢN|TỔNG CỘNG|TỔNG CỘNG NGUỒN VỐN', years[0])
    TTS_Y2 = get_bs_val('TỔNG CỘNG TÀI SẢN|TỔNG CỘNG|TỔNG CỘNG NGUỒN VỐN', years[1])
    TTS_Y3 = get_bs_val('TỔNG CỘNG TÀI SẢN|TỔNG CỘNG|TỔNG CỘNG NGUỒN VỐN', years[2])
    
    # TSNH
    TSNH_Y1 = get_bs_val('Tài sản ngắn hạn|TS ngắn hạn', years[0])
    TSNH_Y2 = get_bs_val('Tài sản ngắn hạn|TS ngắn hạn', years[1])
    TSNH_Y3 = get_bs_val('Tài sản ngắn hạn|TS ngắn hạn', years[2])
    
    # TSDH
    TSDH_Y1 = get_bs_val('Tài sản dài hạn|TS dài hạn', years[0])
    TSDH_Y2 = get_bs_val('Tài sản dài hạn|TS dài hạn', years[1])
    TSDH_Y3 = get_bs_val('Tài sản dài hạn|TS dài hạn', years[2])
    
    # Tiền
    CASH_Y2 = get_bs_val('Tiền và các khoản tương đương tiền', years[1])
    CASH_Y3 = get_bs_val('Tiền và các khoản tương đương tiền', years[2])

    # Phải thu KH
    AR_Y2 = get_bs_val('Phải thu ngắn hạn của khách hàng', years[1])
    AR_Y3 = get_bs_val('Phải thu ngắn hạn của khách hàng', years[2])
    
    # Hàng tồn kho
    INV_Y2 = get_bs_val('Hàng tồn kho|HTK', years[1])
    INV_Y3 = get_bs_val('Hàng tồn kho|HTK', years[2])

    # TSCĐHH
    PP_E_Y2 = get_bs_val('Tài sản cố định hữu hình', years[1])
    PP_E_Y3 = get_bs_val('Tài sản cố định hữu hình', years[2])

    # -----------------------------------------------------
    # 2. Tính toán và Ánh xạ các giá trị
    # -----------------------------------------------------
    
    # ** A. So sánh Y2 vs Y1 **
    placeholders['{{TTS_Y2_CUR}}'] = format_vn_currency(TTS_Y2)
    placeholders['{{TTS_DELTA_Y2_Y1_CUR}}'] = format_vn_delta_currency(TTS_Y2 - TTS_Y1)
    placeholders['{{TTS_GROWTH_Y2_Y1_PCT}}'] = format_vn_percentage(safe_div(TTS_Y2 - TTS_Y1, TTS_Y1) * 100)

    placeholders['{{TSNH_Y2_CUR}}'] = format_vn_currency(TSNH_Y2)
    placeholders['{{TSNH_DELTA_Y2_Y1_CUR}}'] = format_vn_delta_currency(TSNH_Y2 - TSNH_Y1)
    placeholders['{{TSNH_GROWTH_Y2_Y1_PCT}}'] = format_vn_percentage(safe_div(TSNH_Y2 - TSNH_Y1, TSNH_Y1) * 100)

    placeholders['{{TSDH_Y2_CUR}}'] = format_vn_currency(TSDH_Y2)
    placeholders['{{TSDH_DELTA_Y2_Y1_CUR}}'] = format_vn_delta_currency(TSDH_Y2 - TSDH_Y1)
    placeholders['{{TSDH_GROWTH_Y2_Y1_PCT}}'] = format_vn_percentage(safe_div(TSDH_Y2 - TSDH_Y1, TSDH_Y1) * 100)
    
    # ** B. So sánh Y3 vs Y2 **
    placeholders['{{TTS_Y3_CUR}}'] = format_vn_currency(TTS_Y3)
    placeholders['{{TTS_DELTA_Y3_Y2_CUR}}'] = format_vn_delta_currency(TTS_Y3 - TTS_Y2)
    placeholders['{{TTS_GROWTH_Y3_Y2_PCT}}'] = format_vn_percentage(safe_div(TTS_Y3 - TTS_Y2, TTS_Y2) * 100)

    placeholders['{{TSNH_Y3_CUR}}'] = format_vn_currency(TSNH_Y3)
    placeholders['{{TSNH_DELTA_Y3_Y2_CUR}}'] = format_vn_delta_currency(TSNH_Y3 - TSNH_Y2)
    placeholders['{{TSNH_GROWTH_Y3_Y2_PCT}}'] = format_vn_percentage(safe_div(TSNH_Y3 - TSNH_Y2, TSNH_Y2) * 100)

    placeholders['{{TSDH_Y3_CUR}}'] = format_vn_currency(TSDH_Y3)
    placeholders['{{TSDH_DELTA_Y3_Y2_CUR}}'] = format_vn_delta_currency(TSDH_Y3 - TSDH_Y2)
    placeholders['{{TSDH_GROWTH_Y3_Y2_PCT}}'] = format_vn_percentage(safe_div(TSDH_Y3 - TSDH_Y2, TSDH_Y2) * 100)

    # ** C. Chi tiết Tài sản Ngắn hạn **
    
    # Tiền
    CASH_Y1 = get_bs_val('Tiền và các khoản tương đương tiền', years[0])
    placeholders['{{CASH_Y2_CUR}}'] = format_vn_currency(CASH_Y2)
    placeholders['{{CASH_Y3_CUR}}'] = format_vn_currency(CASH_Y3)
    placeholders['{{CASH_DELTA_Y2_Y1_CUR}}'] = format_vn_delta_currency(CASH_Y2 - CASH_Y1)
    placeholders['{{CASH_DELTA_Y3_Y2_CUR}}'] = format_vn_delta_currency(CASH_Y3 - CASH_Y2)
    
    # Phải thu
    AR_Y1 = get_bs_val('Phải thu ngắn hạn của khách hàng', years[0])
    placeholders['{{AR_Y2_CUR}}'] = format_vn_currency(AR_Y2)
    placeholders['{{AR_Y3_CUR}}'] = format_vn_currency(AR_Y3)
    placeholders['{{AR_DELTA_Y2_Y1_CUR}}'] = format_vn_delta_currency(AR_Y2 - AR_Y1)
    placeholders['{{AR_GROWTH_Y2_Y1_PCT}}'] = format_vn_percentage(safe_div(AR_Y2 - AR_Y1, AR_Y1) * 100)
    placeholders['{{AR_PCT_TSNH_Y2}}'] = format_vn_percentage(safe_div(AR_Y2, TSNH_Y2) * 100)
    
    placeholders['{{AR_DELTA_Y3_Y2_CUR}}'] = format_vn_delta_currency(AR_Y3 - AR_Y2)
    placeholders['{{AR_GROWTH_Y3_Y2_PCT}}'] = format_vn_percentage(safe_div(AR_Y3 - AR_Y2, AR_Y2) * 100)
    placeholders['{{AR_PCT_TSNH_Y3}}'] = format_vn_percentage(safe_div(AR_Y3, TSNH_Y3) * 100)

    # Hàng tồn kho
    INV_Y1 = get_bs_val('Hàng tồn kho|HTK', years[0])
    placeholders['{{INV_Y2_CUR}}'] = format_vn_currency(INV_Y2)
    placeholders['{{INV_Y3_CUR}}'] = format_vn_currency(INV_Y3)
    placeholders['{{INV_DELTA_Y2_Y1_CUR}}'] = format_vn_delta_currency(INV_Y2 - INV_Y1)
    placeholders['{{INV_GROWTH_Y2_Y1_PCT}}'] = format_vn_percentage(safe_div(INV_Y2 - INV_Y1, INV_Y1) * 100)
    placeholders['{{INV_PCT_TSNH_Y2}}'] = format_vn_percentage(safe_div(INV_Y2, TSNH_Y2) * 100)
    
    placeholders['{{INV_DELTA_Y3_Y2_CUR}}'] = format_vn_delta_currency(INV_Y3 - INV_Y2)
    placeholders['{{INV_GROWTH_Y3_Y2_PCT}}'] = format_vn_percentage(safe_div(INV_Y3 - INV_Y2, INV_Y2) * 100)
    placeholders['{{INV_PCT_TSNH_Y3}}'] = format_vn_percentage(safe_div(INV_Y3, TSNH_Y3) * 100)

    # ** D. Chi tiết Tài sản Dài hạn **
    
    # TSCĐHH
    PP_E_Y1 = get_bs_val('Tài sản cố định hữu hình', years[0])
    placeholders['{{PP&E_Y2_CUR}}'] = format_vn_currency(PP_E_Y2)
    placeholders['{{PP&E_Y3_CUR}}'] = format_vn_currency(PP_E_Y3)
    placeholders['{{PP&E_DELTA_Y2_Y1_CUR}}'] = format_vn_delta_currency(PP_E_Y2 - PP_E_Y1)
    placeholders['{{PP&E_GROWTH_Y2_Y1_PCT}}'] = format_vn_percentage(safe_div(PP_E_Y2 - PP_E_Y1, PP_E_Y1) * 100)
    placeholders['{{PP&E_PCT_TSDH_Y2}}'] = format_vn_percentage(safe_div(PP_E_Y2, TSDH_Y2) * 100)
    
    placeholders['{{PP&E_DELTA_Y3_Y2_CUR}}'] = format_vn_delta_currency(PP_E_Y3 - PP_E_Y2)
    placeholders['{{PP&E_GROWTH_Y3_Y2_PCT}}'] = format_vn_percentage(safe_div(PP_E_Y3 - PP_E_Y2, PP_E_Y2) * 100)
    placeholders['{{PP&E_PCT_TSDH_Y3}}'] = format_vn_percentage(safe_div(PP_E_Y3, TSDH_Y3) * 100)
    
    # ** E. Tài sản tập trung (Top Items) - SỬ DỤNG LOGIC GIẢ ĐỊNH TRÊN TSNH VÀ TSDH CHÍNH **
    
    # Top 2 TSNH Y2
    top_tsnh_y2 = get_top_items(df_bs_proc, 'Tài sản ngắn hạn|TS ngắn hạn', years[1], n=2)
    placeholders['{{TSNH_TOP_ITEM_1_NAME}}'] = top_tsnh_y2[0][0]
    placeholders['{{TSNH_TOP_ITEM_1_VALUE_CUR}}'] = top_tsnh_y2[0][1]
    placeholders['{{TSNH_TOP_ITEM_1_PCT}}'] = top_tsnh_y2[0][2]
    placeholders['{{TSNH_TOP_ITEM_2_NAME}}'] = top_tsnh_y2[1][0]
    placeholders['{{TSNH_TOP_ITEM_2_VALUE_CUR}}'] = top_tsnh_y2[1][1]
    placeholders['{{TSNH_TOP_ITEM_2_PCT}}'] = top_tsnh_y2[1][2]

    # Top 2 TSDH Y2 (Dùng khoản mục lớn nhất trong TSDH, thường là TSCDHH)
    # Lưu ý: Hàm get_top_items lấy TOP 2 khoản mục con, không phải khoản mục chính (như TSCDHH, ĐTTC,...)
    top_tsdh_y2 = get_top_items(df_bs_proc, 'Tài sản dài hạn|TS dài hạn', years[1], n=2)
    placeholders['{{TSDH_TOP_ITEM_1_NAME}}'] = top_tsdh_y2[0][0]
    placeholders['{{TSDH_TOP_ITEM_1_VALUE_CUR}}'] = top_tsdh_y2[0][1]
    placeholders['{{TSDH_TOP_ITEM_1_PCT}}'] = top_tsdh_y2[0][2]
    placeholders['{{TSDH_TOP_ITEM_2_NAME}}'] = top_tsdh_y2[1][0]
    placeholders['{{TSDH_TOP_ITEM_2_VALUE_CUR}}'] = top_tsdh_y2[1][1]
    placeholders['{{TSDH_TOP_ITEM_2_PCT}}'] = top_tsdh_y2[1][2]
    
    # Top 2 TSNH Y3
    top_tsnh_y3 = get_top_items(df_bs_proc, 'Tài sản ngắn hạn|TS ngắn hạn', years[2], n=2)
    placeholders['{{TSNH_TOP_ITEM_1_NAME_Y3}}'] = top_tsnh_y3[0][0]
    placeholders['{{TSNH_TOP_ITEM_1_VALUE_CUR_Y3}}'] = top_tsnh_y3[0][1]
    placeholders['{{TSNH_TOP_ITEM_1_PCT_Y3}}'] = top_tsnh_y3[0][2]
    placeholders['{{TSNH_TOP_ITEM_2_NAME_Y3}}'] = top_tsnh_y3[1][0]
    placeholders['{{TSNH_TOP_ITEM_2_VALUE_CUR_Y3}}'] = top_tsnh_y3[1][1]
    placeholders['{{TSNH_TOP_ITEM_2_PCT_Y3}}'] = top_tsnh_y3[1][2]
    
    # Top 2 TSDH Y3
    top_tsdh_y3 = get_top_items(df_bs_proc, 'Tài sản dài hạn|TS dài hạn', years[2], n=2)
    placeholders['{{TSDH_TOP_ITEM_1_NAME_Y3}}'] = top_tsdh_y3[0][0]
    placeholders['{{TSDH_TOP_ITEM_1_VALUE_CUR_Y3}}'] = top_tsdh_y3[0][1]
    placeholders['{{TSDH_TOP_ITEM_1_PCT_Y3}}'] = top_tsdh_y3[0][2]
    placeholders['{{TSDH_TOP_ITEM_2_NAME_Y3}}'] = top_tsdh_y3[1][0]
    placeholders['{{TSDH_TOP_ITEM_2_VALUE_CUR_Y3}}'] = top_tsdh_y3[1][1]
    placeholders['{{TSDH_TOP_ITEM_2_PCT_Y3}}'] = top_tsdh_y3[1][2]

    return placeholders

# --- Hàm tính toán chính (Sử dụng Caching để Tối ưu hiệu suất) ---
@st.cache_data
def process_financial_data(df_balance_sheet, df_income_statement):
    """
    Thực hiện các phép tính Tăng trưởng, So sánh Tuyệt đối, Tỷ trọng Cơ cấu, Tỷ trọng Chi phí/DT thuần và Chỉ số Tài chính.
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
            'Chỉ tiêu': 'Hệ số Thanh toán nhanh (Quick Ratio)', y: quick_ratio, 'Type': 'Liquidity'
        })
        ratios_list.append({
            'Chỉ tiêu': 'Vòng quay Hàng tồn kho (Lần)', y: inv_turnover, 'Type': 'Activity'
        })
        ratios_list.append({
            'Chỉ tiêu': 'Thời gian Tồn kho (Ngày)', y: inv_days, 'Type': 'Activity'
        })
        ratios_list.append({
            'Chỉ tiêu': 'Vòng quay các khoản phải thu (Lần)', y: rcv_turnover, 'Type': 'Activity'
        })
        ratios_list.append({
            'Chỉ tiêu': 'Kỳ phải thu bình quân (Ngày)', y: rcv_days, 'Type': 'Activity'
        })
        ratios_list.append({
            'Chỉ tiêu': 'Vòng quay Vốn lưu động (Lần)', y: wcl_turnover, 'Type': 'Activity'
        })
        ratios_list.append({
            'Chỉ tiêu': 'Hệ số Tự tài trợ (Equity Ratio)', y: equity_ratio, 'Type': 'Solvency'
        })
        ratios_list.append({
            'Chỉ tiêu': 'Hệ số Nợ trên Vốn chủ sở hữu (Debt-to-Equity Ratio)', y: d_to_e_ratio, 'Type': 'Solvency'
        })
        ratios_list.append({
            'Chỉ tiêu': 'Hệ số Sinh lời Doanh thu (ROS) (%)', y: ros_ratio, 'Type': 'Profitability'
        })
        ratios_list.append({
            'Chỉ tiêu': 'Hệ số Sinh lời Tài sản (ROA) (%)', y: roa_ratio, 'Type': 'Profitability'
        })
        ratios_list.append({
            'Chỉ tiêu': 'Hệ số Sinh lời Vốn chủ sở hữu (ROE) (%)', y: roe_ratio, 'Type': 'Profitability'
        })
        
    df_temp_ratios = pd.DataFrame(ratios_list)
    df_final_ratios = df_temp_ratios.pivot_table(index=['Chỉ tiêu', 'Type'], values=years, aggfunc='first').reset_index()
    
    # Sắp xếp theo Type (Thanh toán, Hoạt động, Cân nợ, Sinh lời)
    type_order = ['Liquidity', 'Activity', 'Solvency', 'Profitability']
    df_final_ratios['Type'] = pd.Categorical(df_final_ratios['Type'], categories=type_order, ordered=True)
    df_final_ratios = df_final_ratios.sort_values('Type').drop(columns=['Type']).reset_index(drop=True)
    
    # Tính so sánh (np.nan - number = np.nan, điều này là OK vì format_vn_delta_ratio xử lý được)
    df_final_ratios['S.S Tuyệt đối (Y2 vs Y1)'] = df_final_ratios['Năm 2'] - df_final_ratios['Năm 1']
    
    return df_bs, df_is, df_ratios, df_final_ratios

# --- Hàm gọi API Gemini cho Phân tích Báo cáo (Single-shot analysis) ---
def get_ai_analysis(data_for_ai, api_key):
    """Gửi dữ liệu phân tích đến Gemini API và nhận nhận xét."""
    try:
        # Sử dụng API key đã cung cấp hoặc giả định có sẵn
        client = genai.Client(api_key=api_key)
        model_name = 'gemini-2.5-flash'  
        
        # [CẬP NHẬT] System Instruction (Bổ sung đơn vị tính là triệu đồng)
        system_instruction_text = (
            "Bạn là một chuyên gia phân tích tài chính chuyên nghiệp. Chú ý: Tất cả các số liệu tiền tệ trong dữ liệu được cung cấp đều có đơn vị tính là **triệu đồng**. Hãy luôn đề cập đến đơn vị này khi trả lời các câu hỏi về số liệu tài chính cụ thể. "
            "Dựa trên dữ liệu đã cung cấp, hãy đưa ra một nhận xét khách quan, ngắn gọn (khoảng 3-4 đoạn) về tình hình tài chính của doanh nghiệp. "
            "Đánh giá tập trung vào tốc độ tăng trưởng, thay đổi cơ cấu tài sản, **tỷ trọng chi phí/doanh thu thuần**, **hiệu quả hoạt động (Vòng quay Tồn kho, Phải thu, Vốn lưu động)**, **cấu trúc vốn (Hệ số tự tài trợ và Hệ số nợ/VCSH)**, và **khả năng sinh lời (ROS, ROA, ROE)** trong 3 năm/kỳ."
        )
        
        user_prompt = f"""
        {system_instruction_text}
        
        Dữ liệu thô và chỉ số:<br>
        {data_for_ai}
        """

        response = client.models.generate_content(
            model=model_name,
            contents=user_prompt  
        )
        return response.text

    except APIError as e:
        return f"Lỗi gọi Gemini API: Vui lòng kiểm tra Khóa API hoặc giới hạn sử dụng. Chi tiết lỗi: {e}"
    except KeyError:
        return "Lỗi: Không tìm thấy Khóa API 'GEMINI_API_KEY'."
    except Exception as e:
        return f"Đã xảy ra lỗi không xác định: {e}"

# --- Hàm gọi API Gemini cho CHAT tương tác (có quản lý lịch sử) ---
def get_chat_response(prompt, chat_history_st, context_data, api_key):
    try:
        # Sử dụng API key đã cung cấp hoặc giả định có sẵn
        client = genai.Client(api_key=api_key)
        model_name = 'gemini-2.5-flash'
        
        # 1. Định nghĩa System Instruction
        # [CẬP NHẬT] System Instruction (Bổ sung đơn vị tính là triệu đồng)
        system_instruction_text = (
            "Bạn là một trợ lý phân tích tài chính thông minh (Financial Analyst Assistant). "
            "Chú ý: Tất cả các số liệu tiền tệ trong dữ liệu được cung cấp đều có đơn vị tính là **triệu đồng**. Hãy luôn đề cập đến đơn vị này khi trả lời các câu hỏi về số liệu tài chính cụ thể (ví dụ: 'Tổng tài sản là 31.286 triệu đồng'). "
            "Bạn phải trả lời các câu hỏi của người dùng dựa trên dữ liệu tài chính đã xử lý sau. "
            "Dữ liệu này bao gồm tốc độ tăng trưởng, so sánh tuyệt đối/tương đối, tỷ trọng cơ cấu, tỷ trọng chi phí/doanh thu thuần, và **các chỉ số tài chính chủ chốt (Thanh toán, Hoạt động, Cấu trúc Vốn, Sinh lời)** trong 3 kỳ Báo cáo tài chính. "
            "Nếu người dùng hỏi một câu không liên quan đến dữ liệu tài chính hoặc phân tích, hãy lịch sự từ chối trả lời. "
            "Dữ liệu tài chính đã xử lý (được trình bày dưới dạng Markdown để bạn dễ hiểu): \n\n" + context_data
        )
        
        # 2. Chuyển đổi lịch sử Streamlit sang định dạng Gemini
        gemini_history = []
        for msg in chat_history_st[1:]: 
            role = "user" if msg["role"] == "user" else "model"
            gemini_history.append({"role": role, "parts": [{"text": msg["content"]}]})
        
        # 3. Ghép System Instruction và Prompt mới nhất vào Content cuối cùng
        last_user_prompt = prompt
        
        final_prompt = f"""
        {system_instruction_text}
        
        ---
        
        Câu hỏi của người dùng: {last_user_prompt}
        """

        full_contents = gemini_history
        full_contents.append({"role": "user", "parts": [{"text": final_prompt}]})

        # 4. Gọi API
        response = client.models.generate_content(
            model=model_name,
            contents=full_contents 
        )
        return response.text

    except APIError as e:
        return f"Lỗi gọi Gemini API: Vui lòng kiểm tra Khóa API hoặc giới hạn sử dụng. Chi tiết lỗi: {e}"
    except Exception as e:
        return f"Đã xảy ra lỗi không xác định: {e}"


# --- Chức năng 1: Tải File ---
uploaded_file = st.file_uploader(
    "1. Tải file Excel (Sheet 1: BĐKT và KQKD - Tối thiểu 3 cột năm)",
    type=['xlsx', 'xls']
)

# Khai báo các biến tên cột ở phạm vi toàn cục của script
col_nam_1, col_nam_2, col_nam_3 = None, None, None
Y1_Name, Y2_Name, Y3_Name = None, None, None

def format_date_col_name(col_name):
    col_name = str(col_name) 
    if ' ' in col_name:
        col_name = col_name.split(' ')[0]
    try:
        parts = col_name.split('-')
        if len(parts) == 3:
            return f"{parts[2]}/{parts[1]}/{parts[0]}"
    except Exception:
        pass
    return col_name

# -----------------------------------------------------------------
# HÀM CHUẨN HÓA TÊN CỘT ĐỂ DÙNG LỌC DF (LOẠI BỎ DATETIME OBJECT)
# -----------------------------------------------------------------
def clean_column_names(df):
    new_columns = []
    for col in df.columns:
        col_str = str(col)
        if isinstance(col, pd.Timestamp) or (isinstance(col, str) and ' ' in col_str and col_str.endswith('00:00:00')):
            new_columns.append(col_str)
        else:
            new_columns.append(col_str)
    df.columns = new_columns
    return df
# -----------------------------------------------------------------


if uploaded_file is not None:
    try:
        
        # --- ĐỌC DỮ LIỆU TỪ NHIỀU SHEET ---
        xls = pd.ExcelFile(uploaded_file)
        
        # Đọc Sheet 1 cho Bảng CĐKT
        try:
            df_raw_bs = xls.parse(xls.sheet_names[0], header=0) 
            df_raw_bs = clean_column_names(df_raw_bs) # CHUẨN HÓA CỘT BĐKT
        except Exception:
            raise Exception("Không thể đọc Sheet 1 (Bảng CĐKT). Vui lòng kiểm tra định dạng sheet.")
            
        # === LOGIC ĐỌC FILE CHUNG SHEET VÀ TÁCH KQKD (V12) ===
        st.info("Đang xử lý file... Giả định BĐKT và KQKD nằm chung 1 sheet.")
        
        # 1. Đặt tên cột đầu tiên là 'Chỉ tiêu' (từ df_raw_bs đã đọc)
        df_raw_full = df_raw_bs.rename(columns={df_raw_bs.columns[0]: 'Chỉ tiêu'})
        
        # 2. Tìm điểm chia (index của hàng chứa 'KẾT QUẢ HOẠT ĐỘNG KINH DOANH')
        split_keyword = "KẾT QUẢ HOẠT ĐỘNG KINH DOANH"
        
        df_raw_full['Chỉ tiêu'] = df_raw_full['Chỉ tiêu'].astype(str)
        if len(df_raw_full.columns) > 1:
             search_col = df_raw_full['Chỉ tiêu'] + ' ' + df_raw_full[df_raw_full.columns[1]].astype(str)
        else:
             search_col = df_raw_full['Chỉ tiêu']
        
        split_rows = df_raw_full[search_col.str.contains(split_keyword, case=False, na=False)]
        
        if split_rows.empty:
            st.warning(f"Không tìm thấy từ khóa '{split_keyword}' trong Sheet 1. Chỉ phân tích Bảng CĐKT.")
            df_raw_bs = df_raw_full.copy()
            df_raw_is = pd.DataFrame()
        else:
            split_index = split_rows.index[0]
            
            # Tách DataFrame
            if split_index > 0:
                df_raw_bs = df_raw_full.loc[:split_index-1].copy()
            else:
                df_raw_bs = pd.DataFrame(columns=df_raw_full.columns) # BĐKT rỗng
                
            df_raw_is = df_raw_full.loc[split_index:].copy()
            
            # Reset lại header cho Báo cáo KQKD 
            df_is_str = df_raw_is.apply(lambda col: col.astype(str))
            keyword = "CHỈ TIÊU"
            header_mask = df_is_str.apply(lambda row: row.str.contains(keyword, case=False, na=False).any(), axis=1)
            header_rows = df_raw_is[header_mask]
            
            if header_rows.empty:
                st.warning("Không tìm thấy dòng header 'CHỈ TIÊU' trong phần KQKD. Bỏ qua phân tích KQKD.")
                df_raw_is = pd.DataFrame()
            else:
                header_row_index = header_rows.index[0]
                new_header = df_raw_is.loc[header_row_index] 
                df_raw_is = df_raw_is.loc[header_row_index+1:] # Bỏ hàng header
                
                if df_raw_is.empty:
                    st.warning("Phần KQKD chỉ có duy nhất dòng header 'CHỈ TIÊU' và không có dữ liệu. Bỏ qua phân tích KQKD.")
                    df_raw_is = pd.DataFrame()
                else:
                    df_raw_is.columns = new_header
                    col_to_rename = df_raw_is.columns[0]
                    if pd.isna(col_to_rename) or str(col_to_rename).strip() == '':
                        df_raw_is.rename(columns={col_to_rename: 'Chỉ tiêu'}, inplace=True)
                    else:
                        df_raw_is = df_raw_is.rename(columns={df_raw_is.columns[0]: 'Chỉ tiêu'})
        
        # --- TIỀN XỬ LÝ (PRE-PROCESSING) DỮ LIỆU ---
        
        # 1. Đặt tên cột đầu tiên là 'Chỉ tiêu' 
        if not df_raw_bs.empty and df_raw_bs.columns[0] != 'Chỉ tiêu':
            df_raw_bs = df_raw_bs.rename(columns={df_raw_bs.columns[0]: 'Chỉ tiêu'})
            
        if not df_raw_is.empty:
            df_raw_is.columns = [str(col) for col in df_raw_is.columns]
        
        
        # 2. Xác định cột năm/kỳ gần nhất ('Năm 3', 'Năm 2', 'Năm 1')
        value_cols_unique = {} 
        col_name_map = {} 
        for col in df_raw_bs.columns:
            col_str = str(col)
            def normalize_date_col(name):
                if ' ' in name: name = name.split(' ')[0]
                return name
            
            normalized_name = normalize_date_col(col_str)
            
            if len(normalized_name) >= 10 and normalized_name[4] == '-' and normalized_name[7] == '-' and normalized_name[:4].isdigit():
                 if normalized_name not in value_cols_unique:
                    value_cols_unique[normalized_name] = col 
                    col_name_map[normalized_name] = col_str 
            elif normalized_name.isdigit() and len(normalized_name) == 4 and normalized_name.startswith('20'):
                 if normalized_name not in value_cols_unique:
                    value_cols_unique[normalized_name] = col
                    col_name_map[normalized_name] = col_str 

        normalized_names = list(value_cols_unique.keys())
        
        if len(normalized_names) < 3: 
            st.warning(f"Chỉ tìm thấy {len(normalized_names)} cột năm trong Sheet 1 (Bảng CĐKT). Ứng dụng cần ít nhất 3 năm/kỳ để so sánh.")
            st.stop()
            
        normalized_names.sort(key=lambda x: str(x), reverse=True)
        
        col_nam_3 = col_name_map[normalized_names[0]] 
        col_nam_2 = col_name_map[normalized_names[1]] 
        col_nam_1 = col_name_map[normalized_names[2]] 
        
        
        # 3. Lọc bỏ hàng đầu tiên chứa các chỉ số so sánh (SS) không cần thiết (chỉ BĐKT)
        if not df_raw_bs.empty and len(df_raw_bs) > 1:
            df_raw_bs = df_raw_bs.drop(df_raw_bs.index[0])
        
        # --- LOGIC LÀM SẠCH VÀ ĐIỀN CHỈ TIÊU KQKD (V12) ---
        if not df_raw_is.empty:
            first_data_col = col_nam_1 
            
            # BƯỚC 1: HỢP NHẤT TÊN CHỈ TIÊU BỊ DỊCH CHUYỂN
            if 'Chỉ tiêu' in df_raw_is.columns:
                potential_name_cols = [col for i, col in enumerate(df_raw_is.columns) if i > 0 and i < 4]
                
                for name_col in potential_name_cols:
                    df_raw_is[name_col] = df_raw_is[name_col].astype(str).str.strip()
                    
                    df_raw_is['Chỉ tiêu'] = df_raw_is.apply(
                        lambda row: row[name_col] if pd.isna(row['Chỉ tiêu']) or str(row['Chỉ tiêu']).strip() == '' else row['Chỉ tiêu'], 
                        axis=1
                    )
                
            # BƯỚC 2: CHUẨN HÓA VÀ LOẠI BỎ HÀNG KHÔNG CÓ TÊN CHỈ TIÊU HỢP LỆ
            df_raw_is['Chỉ tiêu'] = df_raw_is['Chỉ tiêu'].astype(str).str.strip()
            df_raw_is = df_raw_is[df_raw_is['Chỉ tiêu'].str.len() > 0].copy()
            df_raw_is = df_raw_is[df_raw_is['Chỉ tiêu'].astype(str) != '0'].copy()
                
            # BƯỚC 3: LOẠI BỎ CÁC HÀNG CHÚ THÍCH/RỖNG BẰNG CÁCH KIỂM TRA GIÁ TRỊ SỐ
            if first_data_col in df_raw_is.columns:
                df_raw_is[first_data_col] = pd.to_numeric(df_raw_is[first_data_col], errors='coerce')
                df_raw_is = df_raw_is[df_raw_is[first_data_col].notnull()].copy()
            else:
                st.warning(f"Lỗi: Không tìm thấy cột dữ liệu đầu tiên '{first_data_col}' trong KQKD để làm sạch. Bỏ qua phân tích KQKD.")
                df_raw_is = pd.DataFrame()


        # 4. Tạo DataFrame Bảng CĐKT và KQKD đã lọc (chỉ giữ lại 4 cột)
        cols_to_keep = ['Chỉ tiêu', col_nam_1, col_nam_2, col_nam_3]

        # Bảng CĐKT
        try:
            df_bs_final = df_raw_bs[cols_to_keep].copy()
            df_bs_final.columns = ['Chỉ tiêu', 'Năm 1', 'Năm 2', 'Năm 3']
            df_bs_final = df_bs_final.dropna(subset=['Chỉ tiêu'])
        except KeyError as ke:
             st.warning(f"Lỗi truy cập cột: {ke}. BĐKT có thể rỗng hoặc bị mất cột 'Chỉ tiêu'. Khởi tạo BĐKT rỗng.")
             df_bs_final = pd.DataFrame(columns=['Chỉ tiêu', 'Năm 1', 'Năm 2', 'Năm 3'])
        
        # Báo cáo KQKD
        if not df_raw_is.empty:
            try:
                df_is_final = df_raw_is[cols_to_keep].copy() 
                df_is_final.columns = ['Chỉ tiêu', 'Năm 1', 'Năm 2', 'Năm 3']
                df_is_final = df_is_final.dropna(subset=['Chỉ tiêu'])
                
            except KeyError as ke:
                 st.warning(f"Các cột năm trong phần KQKD không khớp với BĐKT. Bỏ qua phân tích KQKD. Lỗi chi tiết: Cột {ke} bị thiếu.")
                 df_is_final = pd.DataFrame(columns=['Chỉ tiêu', 'Năm 1', 'Năm 2', 'Năm 3'])
            except Exception:
                 df_is_final = pd.DataFrame(columns=['Chỉ tiêu', 'Năm 1', 'Năm 2', 'Năm 3'])
                 
        else:
            st.info("Không tìm thấy dữ liệu KQKD để phân tích.")
            df_is_final = pd.DataFrame(columns=['Chỉ tiêu', 'Năm 1', 'Năm 2', 'Năm 3'])


        # Xử lý dữ liệu
        df_bs_processed, df_is_processed, df_ratios_processed, df_financial_ratios_processed = process_financial_data(df_bs_final.copy(), df_is_final.copy())

        # === [V15] LỌC BỎ CÁC DÒNG CÓ TẤT CẢ GIÁ TRỊ NĂM BẰNG 0 ===
        def filter_zero_rows(df):
            if df.empty:
                return df
            
            # Lọc các cột số có trong df
            numeric_cols = ['Năm 1', 'Năm 2', 'Năm 3']
            cols_to_sum = [col for col in numeric_cols if col in df.columns]
            
            if not cols_to_sum:
                return df 
                
            mask = (df[cols_to_sum].abs().sum(axis=1)) != 0
            return df[mask].copy()

        df_bs_processed = filter_zero_rows(df_bs_processed)
        df_is_processed = filter_zero_rows(df_is_processed)
        df_ratios_processed = filter_zero_rows(df_ratios_processed)
        df_financial_ratios_processed = filter_zero_rows(df_financial_ratios_processed)
        # === KẾT THÚC [V15] ===


        if not df_bs_processed.empty:
            
            # -----------------------------------------------------
            # CHUẨN HÓA TÊN CỘT ĐỂ HIỂN THỊ (DD/MM/YYYY hoặc YYYY)
            # -----------------------------------------------------
            Y1_Name = format_date_col_name(col_nam_1)
            Y2_Name = format_date_col_name(col_nam_2)
            Y3_Name = format_date_col_name(col_nam_3)
            col_names = {'Năm 1': Y1_Name, 'Năm 2': Y2_Name, 'Năm 3': Y3_Name}
            # -----------------------------------------------------

            # LƯU CÁC DATAFRAME ĐÃ XỬ LÝ VÀO SESSION STATE CHO MỤC TẠO BÁO CÁO
            st.session_state.processed_dataframes = {
                'df_bs_proc': df_bs_processed,
                'df_is_proc': df_is_processed,
                'col_names': col_names,
                'y1_name_raw': col_nam_1,
                'y2_name_raw': col_nam_2,
                'y3_name_raw': col_nam_3
            }
            
            # --- Chức năng 2 & 3: Hiển thị Kết quả theo Tabs ---
            st.subheader("2. Phân tích Bảng Cân đối Kế toán & 3. Phân tích Tỷ trọng Cơ cấu Tài sản")
            
            # 1. TẠO DATAFRAME BẢNG CĐKT TĂNG TRƯỞNG (GHÉP CỘT)
            df_growth = df_bs_processed[['Chỉ tiêu', 'Năm 1', 'Năm 2', 'Năm 3', 
                                         'Delta (Y2 vs Y1)', 'Growth (Y2 vs Y1)', 
                                         'Delta (Y3 vs Y2)', 'Growth (Y3 vs Y2)']].copy()
            
            df_growth.columns = [
                'Chỉ tiêu', Y1_Name, Y2_Name, Y3_Name, 
                f'S.S Tuyệt đối ({Y2_Name} vs {Y1_Name})', 
                f'S.S Tương đối (%) ({Y2_Name} vs {Y1_Name})',
                f'S.S Tuyệt đối ({Y3_Name} vs {Y2_Name})', 
                f'S.S Tương đối (%) ({Y3_Name} vs {Y2_Name})'
            ]
            
            # 2. TẠO DATAFRAME BẢNG CĐKT CƠ CẤU
            df_structure = df_bs_processed[['Chỉ tiêu', 'Năm 1', 'Năm 2', 'Năm 3', 
                                            'Tỷ trọng Năm 1 (%)', 'Tỷ trọng Năm 2 (%)', 'Tỷ trọng Năm 3 (%)']].copy()
            
            df_structure.columns = [
                'Chỉ tiêu', Y1_Name, Y2_Name, Y3_Name, 
                f'Tỷ trọng {Y1_Name} (%)', f'Tỷ trọng {Y2_Name} (%)', f'Tỷ trọng {Y3_Name} (%)'
            ]

            tab1, tab2 = st.tabs(["📈 Tốc độ Tăng trưởng Bảng CĐKT", "🏗️ Tỷ trọng Cơ cấu Tài sản"])
            
            # Format và hiển thị tab 1
            with tab1:
                st.markdown("##### Bảng phân tích Tốc độ Tăng trưởng & So sánh Tuyệt đối (Bảng CĐKT)")
                st.dataframe(df_growth.style.apply(highlight_financial_items, axis=1).format({
                    Y1_Name: format_vn_currency,
                    Y2_Name: format_vn_currency,
                    Y3_Name: format_vn_currency,
                    f'S.S Tuyệt đối ({Y2_Name} vs {Y1_Name})': format_vn_delta_currency,
                    f'S.S Tuyệt đối ({Y3_Name} vs {Y2_Name})': format_vn_delta_currency,
                    f'S.S Tương đối (%) ({Y2_Name} vs {Y1_Name})': format_vn_percentage,
                    f'S.S Tương đối (%) ({Y3_Name} vs {Y2_Name})': format_vn_percentage
                }), use_container_width=True, hide_index=True)
                
            # Format và hiển thị tab 2
            with tab2:
                st.markdown("##### Bảng phân tích Tỷ trọng Cơ cấu Tài sản (%)")
                st.dataframe(df_structure.style.apply(highlight_financial_items, axis=1).format({
                    Y1_Name: format_vn_currency,
                    Y2_Name: format_vn_currency,
                    Y3_Name: format_vn_currency,
                    f'Tỷ trọng {Y1_Name} (%)': format_vn_percentage,
                    f'Tỷ trọng {Y2_Name} (%)': format_vn_percentage,
                    f'Tỷ trọng {Y3_Name} (%)': format_vn_percentage
                }), use_container_width=True, hide_index=True)
                
            # -----------------------------------------------------
            # CHỨC NĂNG 4: BÁO CÁO KẾT QUẢ HOẠT ĐỘNG KINH DOANH
            # -----------------------------------------------------
            st.subheader("4. Phân tích Kết quả hoạt động kinh doanh")
            
            is_context = "Không tìm thấy dữ liệu Báo cáo Kết quả hoạt động kinh doanh."

            if not df_is_processed.empty:
                df_is_display = df_is_processed[['Chỉ tiêu', 'Năm 1', 'Năm 2', 'Năm 3', 
                                                 'S.S Tuyệt đối (Y2 vs Y1)', 'S.S Tương đối (%) (Y2 vs Y1)',
                                                 'S.S Tuyệt đối (Y3 vs Y2)', 'S.S Tương đối (%) (Y3 vs Y2)'
                                                 ]].copy()
                
                df_is_display.columns = [
                    'Chỉ tiêu', Y1_Name, Y2_Name, Y3_Name, 
                    f'S.S Tuyệt đối ({Y2_Name} vs {Y1_Name})', 
                    f'S.S Tương đối (%) ({Y2_Name} vs {Y1_Name})',
                    f'S.S Tuyệt đối ({Y3_Name} vs {Y2_Name})', 
                    f'S.S Tương đối (%) ({Y3_Name} vs {Y2_Name})'
                ]
                
                st.markdown(f"##### Bảng so sánh Kết quả hoạt động kinh doanh ({Y2_Name} vs {Y1_Name} và {Y3_Name} vs {Y2_Name})")
                
                st.dataframe(df_is_display.style.apply(highlight_financial_items, axis=1).format({
                    Y1_Name: format_vn_currency,
                    Y2_Name: format_vn_currency,
                    Y3_Name: format_vn_currency,
                    f'S.S Tuyệt đối ({Y2_Name} vs {Y1_Name})': format_vn_delta_currency,
                    f'S.S Tuyệt đối ({Y3_Name} vs {Y2_Name})': format_vn_delta_currency, 
                    f'S.S Tương đối (%) ({Y2_Name} vs {Y1_Name})': format_vn_percentage, 
                    f'S.S Tương đối (%) ({Y3_Name} vs {Y2_Name})': format_vn_percentage 
                }), use_container_width=True, hide_index=True)
                
                is_context = df_is_processed.copy().rename(columns=col_names).to_markdown(index=False)

            else:
                st.info("Không có dữ liệu Báo cáo Kết quả hoạt động kinh doanh để hiển thị.")

            
            # -----------------------------------------------------
            # [V13] CHỨC NĂNG 5: TỶ TRỌNG CHI PHÍ / DOANH THU THUẦN
            # -----------------------------------------------------
            st.subheader("5. Tỷ trọng Chi phí/Doanh thu thuần (%)")
            
            ratios_context = "Không tìm thấy dữ liệu Tỷ trọng Chi phí/Doanh thu thuần."

            if not df_ratios_processed.empty:
                # Cột so sánh là Năm 2 vs Năm 1
                df_ratios_display = df_ratios_processed.copy()
                df_ratios_display.columns = [
                    'Chỉ tiêu', 
                    Y1_Name, 
                    Y2_Name, 
                    Y3_Name, 
                    f'So sánh Tương đối ({Y2_Name} vs {Y1_Name})'
                ]
                
                st.dataframe(df_ratios_display.style.apply(highlight_financial_items, axis=1).format({
                    Y1_Name: format_vn_percentage,
                    Y2_Name: format_vn_percentage,
                    Y3_Name: format_vn_percentage,
                    f'So sánh Tương đối ({Y2_Name} vs {Y1_Name})': format_vn_delta_ratio
                }), use_container_width=True, hide_index=True)

                ratios_context = df_ratios_processed.copy().rename(columns=col_names).to_markdown(index=False)
                
            else:
                st.info("Không thể tính Tỷ trọng Chi phí/Doanh thu thuần do thiếu dữ liệu KQKD.")
            
            # -----------------------------------------------------
            # [CẬP NHẬT] CHỨC NĂNG 6: CÁC HỆ SỐ TÀI CHÍNH CHỦ CHỐT
            # (Gộp Thanh toán, Hoạt động, Cấu trúc Vốn, Sinh lời)
            # -----------------------------------------------------
            st.subheader("6. Các Hệ số Tài chính Chủ chốt (Thanh toán, Hoạt động, Cấu trúc Vốn, Sinh lời) 🔑")

            key_ratios_context = "Không tìm thấy dữ liệu Chỉ tiêu Tài chính Chủ chốt."
            
            if not df_financial_ratios_processed.empty:
                df_ratios_final_display = df_financial_ratios_processed.copy()
                
                # Chỉ giữ lại các cột cần hiển thị: Chỉ tiêu, Năm 1, Năm 2, Năm 3, So sánh Y2 vs Y1
                cols_to_display = ['Chỉ tiêu', 'Năm 1', 'Năm 2', 'Năm 3', 'S.S Tuyệt đối (Y2 vs Y1)']
                
                # Đảm bảo các cột tồn tại trước khi lọc
                cols_to_display = [col for col in cols_to_display if col in df_ratios_final_display.columns]
                
                df_ratios_final_display = df_ratios_final_display[cols_to_display]
                
                # Đổi tên cột (chỉ đổi nếu cột tồn tại)
                rename_map = {
                    'Năm 1': Y1_Name,
                    'Năm 2': Y2_Name,
                    'Năm 3': Y3_Name,
                    'S.S Tuyệt đối (Y2 vs Y1)': f'So sánh Tuyệt đối ({Y2_Name} vs {Y1_Name})'
                }
                
                df_ratios_final_display.columns = [rename_map.get(col, col) for col in df_ratios_final_display.columns]
                
                st.markdown(f"##### Bảng tính Chỉ số Tài chính Chủ chốt ({Y1_Name} - {Y3_Name})")
                
                # Định dạng tùy chỉnh cho các chỉ tiêu: Tỷ lệ (chỉ số)
                st.dataframe(df_ratios_final_display.style.apply(highlight_financial_items, axis=1).format({
                    Y1_Name: format_vn_delta_ratio, # Tỷ lệ 2 thập phân
                    Y2_Name: format_vn_delta_ratio, # Tỷ lệ 2 thập phân
                    Y3_Name: format_vn_delta_ratio,
                    f'So sánh Tuyệt đối ({Y2_Name} vs {Y1_Name})': format_vn_delta_ratio # Delta Tỷ lệ
                }), use_container_width=True, hide_index=True)
                
                key_ratios_context = df_financial_ratios_processed.copy().rename(columns=col_names).to_markdown(index=False)

            else:
                st.info("Không thể tính các Chỉ số Tài chính Chủ chốt do thiếu dữ liệu.")
            
            # -----------------------------------------------------
            # [CẬP NHẬT] CẬP NHẬT CONTEXT CHO CHATBOT (FIXED)
            # -----------------------------------------------------
            
            # Ánh xạ tên cột nội bộ ('Năm 1', 'Năm 2', 'Năm 3') sang tên kỳ báo cáo thực tế.
            # col_names đã được định nghĩa ở trên
            
            # 1. Chuẩn bị Bảng CĐKT Context (luôn có nếu đã chạy đến đây)
            df_bs_context = df_bs_processed.copy().rename(columns=col_names)
            bs_context_md = df_bs_context.to_markdown(index=False)

            # 2. KQKD Context, Tỷ trọng CP Context, Chỉ số TC Context đã được tạo ở trên

            data_for_chat_context = f"""
            **DỮ LIỆU TÀI CHÍNH ĐÃ XỬ LÝ (Kỳ: {Y1_Name}, {Y2_Name}, {Y3_Name}):**
            
            **BẢNG CÂN ĐỐI KẾ TOÁN (Balance Sheet Analysis):**
            {bs_context_md}
            
            **BÁO CÁO KẾT QUẢ KINH DOANH (Income Statement Analysis):**
            {is_context}

            **TỶ TRỌNG CHI PHÍ/DOANH THU THUẦN (%):**
            {ratios_context}
            
            **CÁC HỆ SỐ TÀI CHÍNH CHỦ CHỐT (Thanh toán, Hoạt động, Cấu trúc Vốn, Sinh lời):**
            {key_ratios_context}
            """
            st.session_state.data_for_chat = data_for_chat_context
            
            # Cập nhật tin nhắn chào mừng
            if st.session_state.messages[0]["content"].startswith("Xin chào!") or st.session_state.messages[0]["content"].startswith("Phân tích"):
                st.session_state.messages[0]["content"] = f"Phân tích 3 kỳ ({Y1_Name} đến {Y3_Name}) đã hoàn tất! Bây giờ bạn có thể hỏi tôi bất kỳ điều gì về Bảng CĐKT, KQKD, tỷ trọng chi phí, **các chỉ số thanh toán**, **hiệu quả sử dụng vốn (tồn kho, phải thu, vốn lưu động)**, **cấu trúc vốn/hệ số nợ**, và **khả năng sinh lời (ROS, ROA, ROE)** của báo cáo này."


    except ValueError as ve:
        st.error(f"Lỗi cấu trúc dữ liệu: {ve}")
        st.session_state.data_for_chat = None # Reset chat context
        st.session_state.processed_dataframes = {}
    except Exception as e:
        if "empty" not in str(e) and "columns" not in str(e) and "cannot index" not in str(e):
             st.error(f"Có lỗi xảy ra khi đọc hoặc xử lý file: {e}.")
        st.session_state.data_for_chat = None # Reset chat context
        st.session_state.processed_dataframes = {}

else:
    st.info("Vui lòng tải lên file Excel (Sheet 1 chứa BĐKT và KQKD) để bắt đầu phân tích.")
    st.session_state.data_for_chat = None # Đảm bảo context được reset khi chưa có file
    st.session_state.processed_dataframes = {}

# -----------------------------------------------------
# CHỨC NĂNG MỚI: MỤC 7 - TẠO BÁO CÁO TỪ TEMPLATE
# -----------------------------------------------------
st.subheader("7. Tạo Báo cáo Phân tích từ Mẫu (Template Generation) 📄") 

if not st.session_state.processed_dataframes:
    st.info("Vui lòng tải lên và xử lý báo cáo tài chính trước khi sử dụng chức năng này.")
else:
    uploaded_template = st.file_uploader(
        "Tải lên file Mẫu Báo cáo phân tích (định dạng Markdown .md)",
        type=['md', 'txt']
    )
    
    if uploaded_template is not None:
        try:
            # Đọc nội dung template
            template_content = uploaded_template.read().decode("utf-8")
            
            # Lấy dữ liệu đã xử lý
            df_bs_proc = st.session_state.processed_dataframes['df_bs_proc']
            df_is_proc = st.session_state.processed_dataframes['df_is_proc']
            col_names = st.session_state.processed_dataframes['col_names']

            # Ánh xạ dữ liệu vào placeholders
            placeholders = map_data_to_placeholders(
                df_bs_proc, 
                df_is_proc, 
                col_names['Năm 1'], 
                col_names['Năm 2'], 
                col_names['Năm 3'], 
                col_names
            )
            
            # Thay thế các placeholders
            report_content = template_content
            for placeholder, value in placeholders.items():
                report_content = report_content.replace(placeholder, str(value))
                
            st.success("Tạo báo cáo thành công!")
            
            # Hiển thị kết quả dưới dạng Markdown
            st.markdown("##### Kết quả Báo cáo Phân tích (Sẵn sàng để Copy/Tải xuống)")
            st.markdown(report_content)
            
            # Thêm nút tải xuống
            st.download_button(
                label="Tải xuống Báo cáo (.md)",
                data=report_content,
                file_name="Bao_cao_phan_tich_hoan_thanh.md",
                mime="text/markdown"
            )

        except Exception as e:
            st.error(f"Lỗi khi xử lý template: {e}")


# -----------------------------------------------------
# CHỨC NĂNG CUỐI: MỤC 8 - KHUNG CHATBOT TƯƠNG TÁC
# -----------------------------------------------------
st.subheader("8. Trò chuyện và Hỏi đáp (Gemini AI) 💬") 
if st.session_state.data_for_chat is None:
    st.info("Vui lòng tải lên và xử lý báo cáo tài chính trước khi bắt đầu trò chuyện với AI.")
else:
    # Hiển thị lịch sử chat
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Xử lý input mới từ người dùng
    if prompt := st.chat_input("Hỏi AI về báo cáo tài chính này..."):
        api_key = st.secrets.get("GEMINI_API_KEY")
        
        if not api_key:
            st.error("Lỗi: Không tìm thấy Khóa API. Vui lòng cấu hình Khóa 'GEMINI_API_KEY' trong Streamlit Secrets.")
        else:
            # Thêm tin nhắn của người dùng vào lịch sử
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Tạo phản hồi từ AI
            with st.chat_message("assistant"):
                with st.spinner("Đang gửi câu hỏi và chờ Gemini trả lời..."):
                    
                    full_response = get_chat_response(
                        prompt, 
                        st.session_state.messages, 
                        st.session_state.data_for_chat, 
                        api_key
                    )
                    
                    st.markdown(full_response)
            
            # FIX: Thêm dấu đóng ngoặc đơn
            st.session_state.messages.append({"role": "assistant", "content": full_response})
