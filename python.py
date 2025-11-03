import streamlit as st
import pandas as pd
from google import genai
from google.genai.errors import APIError
import numpy as np # Thêm numpy để xử lý NaN/array

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


# --- Hàm tính toán chính (Sử dụng Caching để Tối ưu hiệu suất) ---
@st.cache_data
def process_financial_data(df_balance_sheet, df_income_statement):
    """
    Thực hiện các phép tính Tăng trưởng, So sánh Tuyệt đối, Tỷ trọng Cơ cấu, Tỷ trọng Chi phí/DT thuần và Chỉ số Tài chính.
    Bây giờ hỗ trợ 4 năm/kỳ: Năm 1, Năm 2, Năm 3, Năm 4 (gần nhất).
    Trả về tuple (df_bs_processed, df_is_processed, df_ratios_processed, df_final_ratios)
    """
    
    # -----------------------------------------------------------------
    # PHẦN 1: XỬ LÝ BẢNG CÂN ĐỐI KẾ TOÁN (BALANCE SHEET - BS)
    # -----------------------------------------------------------------
    df_bs = df_balance_sheet.copy()
    
    # Đảm bảo các giá trị là số để tính toán
    numeric_cols_bs = ['Năm 1', 'Năm 2', 'Năm 3', 'Năm 4']
    for col in numeric_cols_bs:
        df_bs[col] = pd.to_numeric(df_bs[col], errors='coerce').fillna(0)
    
    # Tính toán Tăng trưởng & So sánh Tuyệt đối (Delta / Growth)
    df_bs['Delta (Y2 vs Y1)'] = df_bs['Năm 2'] - df_bs['Năm 1']
    df_bs['Growth (Y2 vs Y1)'] = ((df_bs['Delta (Y2 vs Y1)'] / df_bs['Năm 1'].replace(0, 1e-9)) * 100)
    
    df_bs['Delta (Y3 vs Y2)'] = df_bs['Năm 3'] - df_bs['Năm 2']
    df_bs['Growth (Y3 vs Y2)'] = ((df_bs['Delta (Y3 vs Y2)'] / df_bs['Năm 2'].replace(0, 1e-9)) * 100)
    
    df_bs['Delta (Y4 vs Y3)'] = df_bs['Năm 4'] - df_bs['Năm 3'] # NEW
    df_bs['Growth (Y4 vs Y3)'] = ((df_bs['Delta (Y4 vs Y3)'] / df_bs['Năm 3'].replace(0, 1e-9)) * 100) # NEW

    # Tính Tỷ trọng theo Tổng Tài sản
    tong_tai_san_row = df_bs[df_bs['Chỉ tiêu'].str.contains('TỔNG CỘNG TÀI SẢN|TỔNG CỘNG', case=False, na=False)]
    
    if tong_tai_san_row.empty:
        tong_tai_san_N1, tong_tai_san_N2, tong_tai_san_N3, tong_tai_san_N4 = 1e-9, 1e-9, 1e-9, 1e-9
        st.warning("Không tìm thấy TỔNG CỘNG TÀI SẢN. Tỷ trọng cơ cấu có thể bị sai hoặc không tính được.")
    else:
        tong_tai_san_N1 = tong_tai_san_row['Năm 1'].iloc[0]
        tong_tai_san_N2 = tong_tai_san_row['Năm 2'].iloc[0]
        tong_tai_san_N3 = tong_tai_san_row['Năm 3'].iloc[0]
        tong_tai_san_N4 = tong_tai_san_row['Năm 4'].iloc[0] # NEW

    divisor_N1 = tong_tai_san_N1 if tong_tai_san_N1 != 0 else 1e-9
    divisor_N2 = tong_tai_san_N2 if tong_tai_san_N2 != 0 else 1e-9
    divisor_N3 = tong_tai_san_N3 if tong_tai_san_N3 != 0 else 1e-9
    divisor_N4 = tong_tai_san_N4 if tong_tai_san_N4 != 0 else 1e-9 # NEW

    df_bs['Tỷ trọng Năm 1 (%)'] = (df_bs['Năm 1'] / divisor_N1) * 100
    df_bs['Tỷ trọng Năm 2 (%)'] = (df_bs['Năm 2'] / divisor_N2) * 100
    df_bs['Tỷ trọng Năm 3 (%)'] = (df_bs['Năm 3'] / divisor_N3) * 100
    df_bs['Tỷ trọng Năm 4 (%)'] = (df_bs['Năm 4'] / divisor_N4) * 100 # NEW
    
    # -----------------------------------------------------------------
    # PHẦN 2: XỬ LÝ BÁO CÁO KẾT QUẢ KINH DOANH (INCOME STATEMENT - IS)
    # -----------------------------------------------------------------
    df_is = df_income_statement.copy()
    
    # Đảm bảo các giá trị là số để tính toán
    numeric_cols_is = ['Năm 1', 'Năm 2', 'Năm 3', 'Năm 4'] # Changed to 4 years
    for col in numeric_cols_is:
        df_is[col] = pd.to_numeric(df_is[col], errors='coerce').fillna(0)
    
    # Tính toán Tăng trưởng & So sánh Tuyệt đối (Delta / Growth)
    df_is['S.S Tuyệt đối (Y2 vs Y1)'] = df_is['Năm 2'] - df_is['Năm 1']
    df_is['S.S Tương đối (%) (Y2 vs Y1)'] = ((df_is['S.S Tuyệt đối (Y2 vs Y1)'] / df_is['Năm 1'].replace(0, 1e-9)) * 100)
    
    df_is['S.S Tuyệt đối (Y3 vs Y2)'] = df_is['Năm 3'] - df_is['Năm 2']
    df_is['S.S Tương đối (%) (Y3 vs Y2)'] = ((df_is['S.S Tuyệt đối (Y3 vs Y2)'] / df_is['Năm 2'].replace(0, 1e-9)) * 100)
    
    df_is['S.S Tuyệt đối (Y4 vs Y3)'] = df_is['Năm 4'] - df_is['Năm 3'] # NEW
    df_is['S.S Tương đối (%) (Y4 vs Y3)'] = ((df_is['S.S Tuyệt đối (Y4 vs Y3)'] / df_is['Năm 3'].replace(0, 1e-9)) * 100) # NEW
    
    # -----------------------------------------------------------------
    # PHẦN 3: TÍNH TỶ TRỌNG CHI PHÍ / DOANH THU THUẦN
    # -----------------------------------------------------------------
    df_ratios = pd.DataFrame(columns=['Chỉ tiêu', 'Năm 1', 'Năm 2', 'Năm 3', 'Năm 4'])

    # 1. Tìm Doanh thu thuần (Mẫu số)
    dt_thuan_row = df_is[df_is['Chỉ tiêu'].str.contains('Doanh thu thuần về bán hàng', case=False, na=False)]
    
    if dt_thuan_row.empty:
        DT_thuan_N1, DT_thuan_N2, DT_thuan_N3, DT_thuan_N4 = 1e-9, 1e-9, 1e-9, 1e-9
    else:
        # Lấy giá trị DT thuần, tránh chia cho 0
        DT_thuan_N1 = dt_thuan_row['Năm 1'].iloc[0] if dt_thuan_row['Năm 1'].iloc[0] != 0 else 1e-9
        DT_thuan_N2 = dt_thuan_row['Năm 2'].iloc[0] if dt_thuan_row['Năm 2'].iloc[0] != 0 else 1e-9
        DT_thuan_N3 = dt_thuan_row['Năm 3'].iloc[0] if dt_thuan_row['Năm 3'].iloc[0] != 0 else 1e-9
        DT_thuan_N4 = dt_thuan_row['Năm 4'].iloc[0] if dt_thuan_row['Năm 4'].iloc[0] != 0 else 1e-9 # NEW
    
    # Tính tỷ trọng (dù có DT thuần hay không, để tránh lỗi)
    if not df_is.empty and not dt_thuan_row.empty:
        divisors = [DT_thuan_N1, DT_thuan_N2, DT_thuan_N3, DT_thuan_N4] # Changed to 4 divisors
        years = ['Năm 1', 'Năm 2', 'Năm 3', 'Năm 4'] # Changed to 4 years
        
        # Mapping các chỉ tiêu cần tính tỷ trọng
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
                ratios = [0, 0, 0, 0] # Changed to 4 items
                for i, year in enumerate(years):
                    value = row[year].iloc[0]
                    ratios[i] = (value / divisors[i]) * 100
                
                data_ratio_is.append([ratio_name] + ratios)

        df_ratios = pd.DataFrame(data_ratio_is, columns=['Chỉ tiêu', 'Năm 1', 'Năm 2', 'Năm 3', 'Năm 4']) # Changed to 4 years
        df_ratios['S.S Tương đối (%) (Y2 vs Y1)'] = df_ratios['Năm 2'] - df_ratios['Năm 1']
        df_ratios['S.S Tương đối (%) (Y4 vs Y3)'] = df_ratios['Năm 4'] - df_ratios['Năm 3'] # NEW
        
    # --- HÀM HỖ TRỢ TÌM GIÁ TRỊ CỦA CHỈ TIÊU (Dùng chung cho Ratios) ---
    def get_value(df, keyword, year):
        row = df[df['Chỉ tiêu'].str.contains(keyword, case=False, na=False)]
        if row.empty:
            return 0
        return row[year].iloc[0]

    def safe_div(numerator, denominator):
        # Trả về 0 nếu mẫu số là 0, nếu không tính toán
        return numerator / denominator if denominator != 0 else 0

    years = ['Năm 1', 'Năm 2', 'Năm 3', 'Năm 4'] # Changed to 4 years
    
    # Lấy các giá trị cần thiết từ Bảng CĐKT (BS) và KQKD (IS)
    data = {}
    data['TSNH'] = {y: get_value(df_bs, 'Tài sản ngắn hạn|TS ngắn hạn', y) for y in years}
    data['NO_NGAN_HAN'] = {y: get_value(df_bs, 'Nợ ngắn hạn', y) for y in years} 
    data['HTK'] = {y: get_value(df_bs, 'Hàng tồn kho|HTK', y) for y in years}
    data['GVHB'] = {y: get_value(df_is, 'Giá vốn hàng bán', y) for y in years} 
    
    # [V26] Lấy Nguồn vốn chủ sở hữu, Nợ phải trả và Tổng tài sản
    data['VCSH'] = {y: get_value(df_bs, 'Vốn chủ sở hữu', y) for y in years}
    data['NPT'] = {y: get_value(df_bs, 'Nợ phải trả', y) for y in years}
    data['TTS'] = {y: get_value(df_bs, 'TỔNG CỘNG TÀI SẢN|TỔNG CỘNG NGUỒN VỐN|TỔNG CỘNG', y) for y in years}


    # -----------------------------------------------------------------
    # [V25] PHẦN 4: TÍNH CHỈ SỐ HOẠT ĐỘNG (Vòng quay & Thời gian tồn kho)
    # -----------------------------------------------------------------
    avg_inventory = {}
    inventory_turnover = {}
    inventory_days = {}

    # Hàng tồn kho bình quân: (HTK_kỳ này + HTK_kỳ trước) / 2
    for i, y in enumerate(years): 
        htk_current = data['HTK'][y]
        
        if i == 0:
            # Năm 1: Giả định HTK kỳ trước = HTK năm 1 
            htk_previous = htk_current
        else:
            # Lấy kỳ trước đó
            htk_previous = data['HTK'][years[i-1]]
            
        avg_inventory[y] = safe_div(htk_current + htk_previous, 2)

        # 1. Vòng quay hàng tồn kho = GVHB / HTK BQ
        inventory_turnover[y] = safe_div(data['GVHB'][y], avg_inventory[y])
        
        # 2. Thời gian tồn kho = 365 / Vòng quay
        inventory_days[y] = safe_div(365, inventory_turnover[y]) if inventory_turnover[y] != 0 else 0

    # Tạo DF cho chỉ số hoạt động (Inventory Ratios)
    inventory_ratios_data = {
        'Chỉ tiêu': ['Vòng quay hàng tồn kho (Lần)', 'Thời gian tồn kho (Ngày)'],
        'Năm 1': [inventory_turnover['Năm 1'], inventory_days['Năm 1']],
        'Năm 2': [inventory_turnover['Năm 2'], inventory_days['Năm 2']],
        'Năm 3': [inventory_turnover['Năm 3'], inventory_days['Năm 3']],
        'Năm 4': [inventory_turnover['Năm 4'], inventory_days['Năm 4']], # NEW
    }
    df_inventory_ratios = pd.DataFrame(inventory_ratios_data)
    df_inventory_ratios['S.S Tuyệt đối (Y2 vs Y1)'] = df_inventory_ratios['Năm 2'] - df_inventory_ratios['Năm 1']
    df_inventory_ratios['S.S Tuyệt đối (Y4 vs Y3)'] = df_inventory_ratios['Năm 4'] - df_inventory_ratios['Năm 3'] # NEW


    # -----------------------------------------------------------------
    # [V25] PHẦN 5: TÍNH CHỈ SỐ CÂN ĐỐI (Thanh toán)
    # -----------------------------------------------------------------
    ratios_data_liquidity = {
        'Chỉ tiêu': [
            'Hệ số thanh toán ngắn hạn (Current Ratio)', 
            'Hệ số thanh toán nhanh (Quick Ratio)' 
        ],
        'Năm 1': [
            safe_div(data['TSNH']['Năm 1'], data['NO_NGAN_HAN']['Năm 1']),
            safe_div(data['TSNH']['Năm 1'] - data['HTK']['Năm 1'], data['NO_NGAN_HAN']['Năm 1']) 
        ],
        'Năm 2': [
            safe_div(data['TSNH']['Năm 2'], data['NO_NGAN_HAN']['Năm 2']),
            safe_div(data['TSNH']['Năm 2'] - data['HTK']['Năm 2'], data['NO_NGAN_HAN']['Năm 2'])
        ],
        'Năm 3': [
            safe_div(data['TSNH']['Năm 3'], data['NO_NGAN_HAN']['Năm 3']),
            safe_div(data['TSNH']['Năm 3'] - data['HTK']['Năm 3'], data['NO_NGAN_HAN']['Năm 3'])
        ],
        'Năm 4': [ # NEW
            safe_div(data['TSNH']['Năm 4'], data['NO_NGAN_HAN']['Năm 4']),
            safe_div(data['TSNH']['Năm 4'] - data['HTK']['Năm 4'], data['NO_NGAN_HAN']['Năm 4'])
        ],
    }
    df_liquidity_ratios = pd.DataFrame(ratios_data_liquidity)
    df_liquidity_ratios['S.S Tuyệt đối (Y2 vs Y1)'] = df_liquidity_ratios['Năm 2'] - df_liquidity_ratios['Năm 1']
    df_liquidity_ratios['S.S Tuyệt đối (Y4 vs Y3)'] = df_liquidity_ratios['Năm 4'] - df_liquidity_ratios['Năm 3'] # NEW


    # -----------------------------------------------------------------
    # [V26] PHẦN 6: TÍNH CHỈ SỐ CẤU TRÚC VỐN (SOLVENCY/LEVERAGE)
    # -----------------------------------------------------------------
    solvency_data = {
        'Chỉ tiêu': [
            'Hệ số tự tài trợ (Equity Ratio)', # VCSH / TTS
            'Hệ số nợ trên vốn chủ sở hữu (Debt-to-Equity Ratio)' # NPT / VCSH
        ],
        'Năm 1': [
            safe_div(data['VCSH']['Năm 1'], data['TTS']['Năm 1']),
            safe_div(data['NPT']['Năm 1'], data['VCSH']['Năm 1']),
        ],
        'Năm 2': [
            safe_div(data['VCSH']['Năm 2'], data['TTS']['Năm 2']),
            safe_div(data['NPT']['Năm 2'], data['VCSH']['Năm 2']),
        ],
        'Năm 3': [
            safe_div(data['VCSH']['Năm 3'], data['TTS']['Năm 3']),
            safe_div(data['NPT']['Năm 3'], data['VCSH']['Năm 3']),
        ],
        'Năm 4': [ # NEW
            safe_div(data['VCSH']['Năm 4'], data['TTS']['Năm 4']),
            safe_div(data['NPT']['Năm 4'], data['VCSH']['Năm 4']),
        ],
    }
    df_solvency_ratios = pd.DataFrame(solvency_data)
    df_solvency_ratios['S.S Tuyệt đối (Y2 vs Y1)'] = df_solvency_ratios['Năm 2'] - df_solvency_ratios['Năm 1']
    df_solvency_ratios['S.S Tuyệt đối (Y4 vs Y3)'] = df_solvency_ratios['Năm 4'] - df_solvency_ratios['Năm 3'] # NEW

    # Hợp nhất: Liquidity (Thanh toán) + Inventory (Hoạt động) + Solvency (Cấu trúc vốn)
    df_final_ratios = pd.concat([df_liquidity_ratios, df_inventory_ratios, df_solvency_ratios], ignore_index=True)
    
    return df_bs, df_is, df_ratios, df_final_ratios

# --- Hàm gọi API Gemini cho Phân tích Báo cáo (Single-shot analysis) ---
def get_ai_analysis(data_for_ai, api_key):
    """Gửi dữ liệu phân tích đến Gemini API và nhận nhận xét."""
    try:
        client = genai.Client(api_key=api_key)
        model_name = 'gemini-2.5-flash'  
        
        # [V26] Cập nhật System Instruction
        system_instruction_text = (
            "Bạn là một chuyên gia phân tích tài chính chuyên nghiệp. "
            "Dựa trên dữ liệu đã cung cấp, hãy đưa ra một nhận xét khách quan, ngắn gọn (khoảng 3-4 đoạn) về tình hình tài chính của doanh nghiệp. "
            "Đánh giá tập trung vào tốc độ tăng trưởng qua các chu kỳ, thay đổi cơ cấu tài sản, **tỷ trọng chi phí/doanh thu thuần**, **hiệu quả quản lý hàng tồn kho**, và **cấu trúc vốn (Hệ số tự tài trợ và Hệ số nợ/VCSH)** trong 4 năm/kỳ." # Updated instruction
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
        client = genai.Client(api_key=api_key)
        model_name = 'gemini-2.5-flash'
        
        # 1. Định nghĩa System Instruction
        # [V26] Cập nhật System Instruction
        system_instruction_text = (
            "Bạn là một trợ lý phân tích tài chính thông minh (Financial Analyst Assistant). "
            "Bạn phải trả lời các câu hỏi của người dùng dựa trên dữ liệu tài chính đã xử lý sau. "
            "Dữ liệu này bao gồm tốc độ tăng trưởng, so sánh tuyệt đối/tương đối, tỷ trọng cơ cấu, tỷ trọng chi phí/doanh thu thuần, **các chỉ số thanh toán ngắn hạn và nhanh**, **hiệu quả hàng tồn kho (Vòng quay và Thời gian tồn kho)**, và **cấu trúc vốn (Hệ số tự tài trợ và Hệ số nợ/VCSH)** trong 4 kỳ Báo cáo tài chính. " # Updated instruction
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
    "1. Tải file Excel (Sheet 1: BĐKT và KQKD - Tối thiểu 4 cột năm)", # Updated instruction
    type=['xlsx', 'xls']
)

if uploaded_file is not None:
    try:
        
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
        
        
        # 2. Xác định cột năm/kỳ gần nhất ('Năm 4', 'Năm 3', 'Năm 2', 'Năm 1')
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
        
        if len(normalized_names) < 4: # Changed from 3 to 4
            st.warning(f"Chỉ tìm thấy {len(normalized_names)} cột năm trong Sheet 1 (Bảng CĐKT). Ứng dụng cần ít nhất 4 năm/kỳ để so sánh.")
            st.stop()
            
        normalized_names.sort(key=lambda x: str(x), reverse=True)
        
        col_nam_4 = col_name_map[normalized_names[0]] # Năm 4 (Mới nhất)
        col_nam_3 = col_name_map[normalized_names[1]] # Năm 3
        col_nam_2 = col_name_map[normalized_names[2]] # Năm 2
        col_nam_1 = col_name_map[normalized_names[3]] # Năm 1 (Lâu nhất)
        
        
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


        # 4. Tạo DataFrame Bảng CĐKT và KQKD đã lọc (chỉ giữ lại 5 cột)
        cols_to_keep = ['Chỉ tiêu', col_nam_1, col_nam_2, col_nam_3, col_nam_4] # Changed to 5 columns

        # Bảng CĐKT
        try:
            df_bs_final = df_raw_bs[cols_to_keep].copy()
            df_bs_final.columns = ['Chỉ tiêu', 'Năm 1', 'Năm 2', 'Năm 3', 'Năm 4'] # Changed to 4 years
            df_bs_final = df_bs_final.dropna(subset=['Chỉ tiêu'])
        except KeyError as ke:
             st.warning(f"Lỗi truy cập cột: {ke}. BĐKT có thể rỗng hoặc bị mất cột 'Chỉ tiêu'. Khởi tạo BĐKT rỗng.")
             df_bs_final = pd.DataFrame(columns=['Chỉ tiêu', 'Năm 1', 'Năm 2', 'Năm 3', 'Năm 4']) # Changed to 4 years
        

        # Báo cáo KQKD
        if not df_raw_is.empty:
            try:
                df_is_final = df_raw_is[cols_to_keep].copy() 
                df_is_final.columns = ['Chỉ tiêu', 'Năm 1', 'Năm 2', 'Năm 3', 'Năm 4'] # Changed to 4 years
                df_is_final = df_is_final.dropna(subset=['Chỉ tiêu'])
                
            except KeyError as ke:
                 st.warning(f"Các cột năm trong phần KQKD không khớp với BĐKT. Bỏ qua phân tích KQKD. Lỗi chi tiết: Cột {ke} bị thiếu.")
                 df_is_final = pd.DataFrame(columns=['Chỉ tiêu', 'Năm 1', 'Năm 2', 'Năm 3', 'Năm 4']) # Changed to 4 years
            except Exception:
                 df_is_final = pd.DataFrame(columns=['Chỉ tiêu', 'Năm 1', 'Năm 2', 'Năm 3', 'Năm 4']) # Changed to 4 years
                
        else:
            st.info("Không tìm thấy dữ liệu KQKD để phân tích.")
            df_is_final = pd.DataFrame(columns=['Chỉ tiêu', 'Năm 1', 'Năm 2', 'Năm 3', 'Năm 4']) # Changed to 4 years


        # Xử lý dữ liệu
        df_bs_processed, df_is_processed, df_ratios_processed, df_financial_ratios_processed = process_financial_data(df_bs_final.copy(), df_is_final.copy())

        # === [V15] LỌC BỎ CÁC DÒNG CÓ TẤT CẢ GIÁ TRỊ NĂM BẰNG 0 ===
        def filter_zero_rows(df):
            if df.empty:
                return df
            
            # Lọc các cột số có trong df
            numeric_cols = ['Năm 1', 'Năm 2', 'Năm 3', 'Năm 4'] # Changed to 4 years
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
            def format_col_name(col_name):
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

            Y1_Name = format_col_name(col_nam_1)
            Y2_Name = format_col_name(col_nam_2)
            Y3_Name = format_col_name(col_nam_3)
            Y4_Name = format_col_name(col_nam_4) # NEW
            # -----------------------------------------------------
            
            # --- Chức năng 2 & 3: Hiển thị Kết quả theo Tabs ---
            st.subheader("2. Phân tích Bảng Cân đối Kế toán & 3. Phân tích Tỷ trọng Cơ cấu Tài sản")
            
            # 1. TẠO DATAFRAME BẢNG CĐKT TĂNG TRƯỞNG (GHÉP CỘT)
            df_growth = df_bs_processed[['Chỉ tiêu', 'Năm 1', 'Năm 2', 'Năm 3', 'Năm 4', # Added Năm 4
                                         'Delta (Y2 vs Y1)', 'Growth (Y2 vs Y1)', 
                                         'Delta (Y3 vs Y2)', 'Growth (Y3 vs Y2)',
                                         'Delta (Y4 vs Y3)', 'Growth (Y4 vs Y3)']].copy() # Added Y4 vs Y3
            
            df_growth.columns = [
                'Chỉ tiêu', Y1_Name, Y2_Name, Y3_Name, Y4_Name, # Added Y4_Name
                f'S.S Tuyệt đối ({Y2_Name} vs {Y1_Name})', 
                f'S.S Tương đối (%) ({Y2_Name} vs {Y1_Name})',
                f'S.S Tuyệt đối ({Y3_Name} vs {Y2_Name})', 
                f'S.S Tương đối (%) ({Y3_Name} vs {Y2_Name})',
                f'S.S Tuyệt đối ({Y4_Name} vs {Y3_Name})', # Added Y4 vs Y3
                f'S.S Tương đối (%) ({Y4_Name} vs {Y3_Name})' # Added Y4 vs Y3
            ]
            
            # 2. TẠO DATAFRAME BẢNG CĐKT CƠ CẤU
            df_structure = df_bs_processed[['Chỉ tiêu', 'Năm 1', 'Năm 2', 'Năm 3', 'Năm 4', # Added Năm 4
                                            'Tỷ trọng Năm 1 (%)', 'Tỷ trọng Năm 2 (%)', 'Tỷ trọng Năm 3 (%)', 'Tỷ trọng Năm 4 (%)']].copy() # Added Năm 4
            
            df_structure.columns = [
                'Chỉ tiêu', Y1_Name, Y2_Name, Y3_Name, Y4_Name, # Added Y4_Name
                f'Tỷ trọng {Y1_Name} (%)', f'Tỷ trọng {Y2_Name} (%)', f'Tỷ trọng {Y3_Name} (%)', f'Tỷ trọng {Y4_Name} (%)' # Added Y4_Name
            ]

            tab1, tab2 = st.tabs(["📈 Tốc độ Tăng trưởng Bảng CĐKT", "🏗️ Tỷ trọng Cơ cấu Tài sản"])
            
            # Format và hiển thị tab 1
            with tab1:
                st.markdown("##### Bảng phân tích Tốc độ Tăng trưởng & So sánh Tuyệt đối (Bảng CĐKT)")
                st.dataframe(df_growth.style.apply(highlight_financial_items, axis=1).format({
                    Y1_Name: format_vn_currency,
                    Y2_Name: format_vn_currency,
                    Y3_Name: format_vn_currency,
                    Y4_Name: format_vn_currency, # NEW
                    f'S.S Tuyệt đối ({Y2_Name} vs {Y1_Name})': format_vn_delta_currency,
                    f'S.S Tuyệt đối ({Y3_Name} vs {Y2_Name})': format_vn_delta_currency,
                    f'S.S Tuyệt đối ({Y4_Name} vs {Y3_Name})': format_vn_delta_currency, # NEW
                    f'S.S Tương đối (%) ({Y2_Name} vs {Y1_Name})': format_vn_percentage,
                    f'S.S Tương đối (%) ({Y3_Name} vs {Y2_Name})': format_vn_percentage,
                    f'S.S Tương đối (%) ({Y4_Name} vs {Y3_Name})': format_vn_percentage # NEW
                }), use_container_width=True, hide_index=True)
                
            # Format và hiển thị tab 2
            with tab2:
                st.markdown("##### Bảng phân tích Tỷ trọng Cơ cấu Tài sản (%)")
                st.dataframe(df_structure.style.apply(highlight_financial_items, axis=1).format({
                    Y1_Name: format_vn_currency,
                    Y2_Name: format_vn_currency,
                    Y3_Name: format_vn_currency,
                    Y4_Name: format_vn_currency, # NEW
                    f'Tỷ trọng {Y1_Name} (%)': format_vn_percentage,
                    f'Tỷ trọng {Y2_Name} (%)': format_vn_percentage,
                    f'Tỷ trọng {Y3_Name} (%)': format_vn_percentage,
                    f'Tỷ trọng {Y4_Name} (%)': format_vn_percentage # NEW
                }), use_container_width=True, hide_index=True)
                
            # -----------------------------------------------------
            # CHỨC NĂNG 4: BÁO CÁO KẾT QUẢ HOẠT ĐỘNG KINH DOANH
            # -----------------------------------------------------
            st.subheader("4. Phân tích Kết quả hoạt động kinh doanh")

            if not df_is_processed.empty:
                df_is_display = df_is_processed[['Chỉ tiêu', 'Năm 1', 'Năm 2', 'Năm 3', 'Năm 4', # Added Năm 4
                                                 'S.S Tuyệt đối (Y2 vs Y1)', 'S.S Tương đối (%) (Y2 vs Y1)',
                                                 'S.S Tuyệt đối (Y3 vs Y2)', 'S.S Tương đối (%) (Y3 vs Y2)',
                                                 'S.S Tuyệt đối (Y4 vs Y3)', 'S.S Tương đối (%) (Y4 vs Y3)' # Added Y4 vs Y3
                                                 ]].copy()
                
                df_is_display.columns = [
                    'Chỉ tiêu', Y1_Name, Y2_Name, Y3_Name, Y4_Name, # Added Y4_Name
                    f'S.S Tuyệt đối ({Y2_Name} vs {Y1_Name})', 
                    f'S.S Tương đối (%) ({Y2_Name} vs {Y1_Name})',
                    f'S.S Tuyệt đối ({Y3_Name} vs {Y2_Name})', 
                    f'S.S Tương đối (%) ({Y3_Name} vs {Y2_Name})',
                    f'S.S Tuyệt đối ({Y4_Name} vs {Y3_Name})', # Added Y4 vs Y3
                    f'S.S Tương đối (%) ({Y4_Name} vs {Y3_Name})' # Added Y4 vs Y3
                ]
                
                st.markdown(f"##### Bảng so sánh Kết quả hoạt động kinh doanh ({Y2_Name} vs {Y1_Name}, {Y3_Name} vs {Y2_Name} và {Y4_Name} vs {Y3_Name})") # Updated Title
                
                st.dataframe(df_is_display.style.apply(highlight_financial_items, axis=1).format({
                    Y1_Name: format_vn_currency,
                    Y2_Name: format_vn_currency,
                    Y3_Name: format_vn_currency,
                    Y4_Name: format_vn_currency, # NEW
                    f'S.S Tuyệt đối ({Y2_Name} vs {Y1_Name})': format_vn_delta_currency,
                    f'S.S Tương đối (%) ({Y2_Name} vs {Y1_Name})': format_vn_percentage,
                    f'S.S Tuyệt đối ({Y3_Name} vs {Y2_Name})': format_vn_delta_currency, 
                    f'S.S Tương đối (%) ({Y3_Name} vs {Y2_Name})': format_vn_percentage, 
                    f'S.S Tuyệt đối ({Y4_Name} vs {Y3_Name})': format_vn_delta_currency, # NEW
                    f'S.S Tương đối (%) ({Y4_Name} vs {Y3_Name})': format_vn_percentage # NEW
                }), use_container_width=True, hide_index=True)

                is_context = df_is_processed.to_markdown(index=False)
            else:
                st.info("Không có dữ liệu Báo cáo Kết quả hoạt động kinh doanh để hiển thị.")
                is_context = "Không tìm thấy dữ liệu Báo cáo Kết quả hoạt động kinh doanh."

            
            # -----------------------------------------------------
            # [V13] CHỨC NĂNG 5: TỶ TRỌNG CHI PHÍ / DOANH THU THUẦN
            # -----------------------------------------------------
            st.subheader("5. Tỷ trọng Chi phí/Doanh thu thuần (%)")
            
            if not df_ratios_processed.empty:
                # SỬA: Bao gồm cột Năm 4 và So sánh Y4 vs Y3
                df_ratios_display = df_ratios_processed[['Chỉ tiêu', 'Năm 1', 'Năm 2', 'Năm 3', 'Năm 4', 
                                                         'S.S Tương đối (%) (Y2 vs Y1)', 'S.S Tương đối (%) (Y4 vs Y3)']].copy()
                
                df_ratios_display.columns = [
                    'Chỉ tiêu', 
                    Y1_Name, 
                    Y2_Name, 
                    Y3_Name, 
                    Y4_Name, # NEW
                    f'So sánh Tương đối ({Y2_Name} vs {Y1_Name})',
                    f'So sánh Tương đối ({Y4_Name} vs {Y3_Name})' # NEW
                ]
                
                st.dataframe(df_ratios_display.style.apply(highlight_financial_items, axis=1).format({
                    Y1_Name: format_vn_percentage,
                    Y2_Name: format_vn_percentage,
                    Y3_Name: format_vn_percentage,
                    Y4_Name: format_vn_percentage, # NEW
                    f'So sánh Tương đối ({Y2_Name} vs {Y1_Name})': format_vn_delta_ratio,
                    f'So sánh Tương đối ({Y4_Name} vs {Y3_Name})': format_vn_delta_ratio # NEW
                }), use_container_width=True, hide_index=True)
                
                ratios_context = df_ratios_processed.to_markdown(index=False)
            else:
                st.info("Không thể tính Tỷ trọng Chi phí/Doanh thu thuần do thiếu dữ liệu KQKD.")
                ratios_context = "Không tìm thấy dữ liệu Tỷ trọng Chi phí/Doanh thu thuần."
            
            # -----------------------------------------------------
            # [V25] CHỨC NĂNG 6: HIỆU QUẢ HOẠT ĐỘNG (Vòng quay Tồn kho)
            # -----------------------------------------------------
            st.subheader("6. Hiệu quả Hoạt động: Vòng quay và Thời gian Tồn kho")
            
            # Lọc chỉ các chỉ tiêu liên quan đến Tồn kho/Vòng quay (Mục đích hiển thị riêng)
            df_inventory_ratios_processed = df_financial_ratios_processed[
                df_financial_ratios_processed['Chỉ tiêu'].str.contains('Vòng quay|Thời gian tồn kho', case=False, na=False)
            ].copy()

            if not df_inventory_ratios_processed.empty:
                df_inv_display = df_inventory_ratios_processed[['Chỉ tiêu', 'Năm 1', 'Năm 2', 'Năm 3', 'Năm 4', # Added Năm 4
                                                                'S.S Tuyệt đối (Y2 vs Y1)', 'S.S Tuyệt đối (Y4 vs Y3)']].copy() # Added Y4 vs Y3
                
                df_inv_display.columns = [
                    'Chỉ tiêu', 
                    Y1_Name, 
                    Y2_Name, 
                    Y3_Name, 
                    Y4_Name, # NEW
                    f'So sánh Tuyệt đối ({Y2_Name} vs {Y1_Name})',
                    f'So sánh Tuyệt đối ({Y4_Name} vs {Y3_Name})' # NEW
                ]
                
                st.markdown(f"##### Bảng tính Chỉ số Hoạt động ({Y1_Name} - {Y4_Name})") # Updated Title
                
                # Định dạng tùy chỉnh cho các chỉ tiêu: Tỷ lệ (chỉ số)
                st.dataframe(df_inv_display.style.apply(highlight_financial_items, axis=1).format({
                    Y1_Name: format_vn_delta_ratio, # Tỷ lệ 2 thập phân cho Lần/Ngày
                    Y2_Name: format_vn_delta_ratio, 
                    Y3_Name: format_vn_delta_ratio,
                    Y4_Name: format_vn_delta_ratio, # NEW
                    f'So sánh Tuyệt đối ({Y2_Name} vs {Y1_Name})': format_vn_delta_ratio, # Delta Lần/Ngày
                    f'So sánh Tuyệt đối ({Y4_Name} vs {Y3_Name})': format_vn_delta_ratio # NEW
                }), use_container_width=True, hide_index=True)
                
                inv_context = df_inventory_ratios_processed.to_markdown(index=False)
            else:
                st.info("Không thể tính Vòng quay và Thời gian tồn kho do thiếu dữ liệu Giá vốn hàng bán hoặc Hàng tồn kho.")
                inv_context = "Không tìm thấy dữ liệu Vòng quay/Thời gian tồn kho."
            
            # -----------------------------------------------------
            # [V27 - ĐÃ SỬA] CHỨC NĂNG 7: CÁC CHỈ SỐ TÀI CHÍNH CHỦ CHỐT (Thanh toán & Cấu trúc Vốn)
            # -----------------------------------------------------
            st.subheader("7. Các Chỉ số Tài chính Chủ chốt (Thanh toán và Cấu trúc Vốn)")

            # Lọc chỉ các chỉ tiêu KHÔNG liên quan đến Tồn kho/Vòng quay
            df_combined_key_ratios = df_financial_ratios_processed[
                ~df_financial_ratios_processed['Chỉ tiêu'].str.contains('Vòng quay|Thời gian tồn kho', case=False, na=False)
            ].copy()

            if not df_combined_key_ratios.empty:
                df_ratios_final_display = df_combined_key_ratios.copy()
                
                # SỬA: Bao gồm cột Năm 4 (kỳ gần nhất) và so sánh Y4 vs Y3
                cols_to_display = ['Chỉ tiêu', 'Năm 1', 'Năm 2', 'Năm 3', 'Năm 4', 
                                   'S.S Tuyệt đối (Y2 vs Y1)', 'S.S Tuyệt đối (Y4 vs Y3)'] # Added Y4 and Y4 vs Y3 comparison
                df_ratios_final_display = df_ratios_final_display[cols_to_display]
                
                # SỬA: Cập nhật tên cột hiển thị
                df_ratios_final_display.columns = [
                    'Chỉ tiêu', 
                    Y1_Name, 
                    Y2_Name, 
                    Y3_Name, 
                    Y4_Name, # NEW
                    f'So sánh Tuyệt đối ({Y2_Name} vs {Y1_Name})',
                    f'So sánh Tuyệt đối ({Y4_Name} vs {Y3_Name})' # NEW
                ]
                
                st.markdown(f"##### Bảng tính Chỉ số Thanh toán và Cấu trúc Vốn ({Y1_Name} - {Y4_Name})") # Sửa tiêu đề

                # SỬA: Thêm định dạng cho cột Y4_Name và so sánh Y4 vs Y3
                st.dataframe(df_ratios_final_display.style.apply(highlight_financial_items, axis=1).format({
                    Y1_Name: format_vn_delta_ratio, # Tỷ lệ 2 thập phân
                    Y2_Name: format_vn_delta_ratio, # Tỷ lệ 2 thập phân
                    Y3_Name: format_vn_delta_ratio,
                    Y4_Name: format_vn_delta_ratio, # NEW
                    f'So sánh Tuyệt đối ({Y2_Name} vs {Y1_Name})': format_vn_delta_ratio, # Delta Tỷ lệ
                    f'So sánh Tuyệt đối ({Y4_Name} vs {Y3_Name})': format_vn_delta_ratio # NEW
                }), use_container_width=True, hide_index=True)
                
                # Context chỉ chứa các chỉ số Thanh toán và Cấu trúc vốn
                combined_key_ratios_context = df_combined_key_ratios.to_markdown(index=False)
            else:
                st.info("Không thể tính các Chỉ số Tài chính Chủ chốt (Thanh toán và Cấu trúc Vốn) do thiếu dữ liệu.")
                combined_key_ratios_context = "Không tìm thấy dữ liệu Chỉ tiêu Tài chính Chủ chốt."
            
            # -----------------------------------------------------
            # [V27] MỤC 8 LÀ KHUNG CHATBOT
            # -----------------------------------------------------
            
            # -----------------------------------------------------
            # [V27] CẬP NHẬT CONTEXT CHO CHATBOT 
            # -----------------------------------------------------
            data_for_chat_context = f"""
            **BẢNG CÂN ĐỐI KẾ TOÁN (Balance Sheet Analysis):**
            {df_bs_processed.to_markdown(index=False)}
            
            **BÁO CÁO KẾT QUẢ KINH DOANH (Income Statement Analysis):**
            {is_context}

            **TỶ TRỌNG CHI PHÍ/DOANH THU THUẦN (%):**
            {ratios_context}
            
            **HIỆU QUẢ HOẠT ĐỘNG (HÀNG TỒN KHO):**
            {inv_context}
            
            **CHỈ SỐ TÀI CHÍNH CHỦ CHỐT (THANH TOÁN & CẤU TRÚC VỐN):**
            {combined_key_ratios_context}
            """
            st.session_state.data_for_chat = data_for_chat_context
            
            # Cập nhật tin nhắn chào mừng
            if st.session_state.messages[0]["content"].startswith("Xin chào!") or st.session_state.messages[0]["content"].startswith("Phân tích"):
                       st.session_state.messages[0]["content"] = f"Phân tích 4 kỳ ({Y1_Name} đến {Y4_Name}) đã hoàn tất! Bây giờ bạn có thể hỏi tôi bất kỳ điều gì về Bảng CĐKT, KQKD, tỷ trọng chi phí, **các chỉ số thanh toán**, **hiệu quả sử dụng hàng tồn kho**, và **cấu trúc vốn/hệ số nợ** của báo cáo này."

            # -----------------------------------------------------
            # MỤC 8 LÀ KHUNG CHATBOT (Thay thế Mục 9 cũ)
            # -----------------------------------------------------

    except ValueError as ve:
        st.error(f"Lỗi cấu trúc dữ liệu: {ve}")
        st.session_state.data_for_chat = None # Reset chat context
    except Exception as e:
        if "empty" not in str(e) and "columns" not in str(e) and "cannot index" not in str(e):
             st.error(f"Có lỗi xảy ra khi đọc hoặc xử lý file: {e}.")
        st.session_state.data_for_chat = None # Reset chat context

else:
    st.info("Vui lòng tải lên file Excel (Sheet 1 chứa BĐKT và KQKD) để bắt đầu phân tích.")
    st.session_state.data_for_chat = None # Đảm bảo context được reset khi chưa có file

# --- Chức năng 8: Khung Chatbot tương tác (Thay thế Mục 9 cũ) ---
st.subheader("8. Trò chuyện và Hỏi đáp (Gemini AI)") 
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
            
            # Thêm phản hồi của AI vào lịch sử
            st.session_state.messages.append({"role": "assistant", "content": full_response})
