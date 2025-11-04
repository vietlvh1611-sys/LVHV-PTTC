import streamlit as st
import pandas as pd
from google import genai
from google.genai.errors import APIError
import numpy as np
import io # Thư viện IO (Mặc dù không dùng docxtpl nữa, giữ lại cũng không sao)
# try:
#     from docxtpl import DocxTemplate 
# except ImportError:
#     # Không cần báo lỗi nữa vì đã xóa chức năng
#     pass 


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
    if pd.isna(val) or (val == 0): 
        return "" 
    val = round(val)
    return "{:,d}".format(val).replace(",", "TEMP_SEP").replace(".", ",").replace("TEMP_SEP", ".")

def format_vn_percentage(val):
    if pd.isna(val) or (val == 0):
        return ""
    val = round(val, 1)
    formatted_val = "{:,.1f}".format(val).replace(",", "TEMP_SEP").replace(".", ",").replace("TEMP_SEP", ".") + "%"
    return formatted_val

def format_vn_delta_currency(val):
    if pd.isna(val):
        return ""
    val = round(val)
    if val < 0:
        formatted_val = "-{:,d}".format(abs(val))
    else:
        formatted_val = "{:,d}".format(val)
    return formatted_val.replace(",", "TEMP_SEP").replace(".", ",").replace("TEMP_SEP", ".")

def format_vn_delta_ratio(val):
    if pd.isna(val) or (val == 0):
        return ""
    val = round(val, 2)
    if val < 0:
        formatted_val = "-{:.2f}".format(abs(val)).replace(".", ",")
    else:
        formatted_val = "{:.2f}".format(val).replace(".", ",")
    return formatted_val
# === KẾT THÚC ĐỊNH NGHĨA FORMATTERS ===

# === [V16] ĐỊNH NGHĨA HÀM STYLING CHO CÁC CHỈ TIÊU CHÍNH/PHỤ ===
def highlight_financial_items(row):
    """Áp dụng in đậm cho mục chính (A, I, TỔNG CỘNG) và in nghiêng cho mục chi tiết (Nguyên giá, Hao mòn)."""
    styles = [''] * len(row)
    item = str(row['Chỉ tiêu']).strip()
    
    is_major_section = (
        item.startswith(('A.', 'B.', 'C.')) or 
        item.startswith(('I.', 'II.', 'III.', 'IV.', 'V.', 'VI.', 'VII.', 'VIII.', 'IX.', 'X.')) or
        'TỔNG CỘNG' in item.upper() or
        'TỔNG CỘNG TÀI SẢN' in item.upper() or
        'TỔNG CỘNG NGUỒN VỐN' in item.upper() or
        'NỢ PHẢI TRẢ' in item.upper() or
        'VỐN CHỦ SỞ HỮU' in item.upper() or
        item in ['Khả năng thanh toán', 'Chỉ tiêu hoạt động', 'Chỉ tiêu cân nợ', 'Hệ số sinh lời']
    )
    
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
    if df.empty: return 0.0
    row = df[df['Chỉ tiêu'].str.contains(keyword, case=False, na=False)]
    if row.empty:
        return 0.0
        
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
    
    ratios_list = []
    
    for i, y in enumerate(years):
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
        
        htk_previous = data['HTK'][years[i-1]] if i > 0 else htk
        avg_inventory = safe_div(htk + htk_previous, 2)
        
        pt_current = data['PHAI_THU'][y]
        pt_previous = data['PHAI_THU'][years[i-1]] if i > 0 else pt_current
        avg_receivable = safe_div(pt_current + pt_previous, 2)
        
        wl_current = tsnh - nnh
        wl_previous = (data['TSNH'][years[i-1]] - data['NO_NGAN_HAN'][years[i-1]]) if i > 0 else wl_current
        avg_working_capital = safe_div(wl_current + wl_previous, 2)

        current_ratio = safe_div(tsnh, nnh)
        quick_ratio = safe_div(tsnh - htk, nnh)

        inv_turnover = safe_div(gvhb, avg_inventory)
        inv_days = safe_div(365, inv_turnover)
        
        rcv_turnover = safe_div(dt_thuan, avg_receivable)
        rcv_days = safe_div(365, rcv_turnover)
        
        wcl_turnover = safe_div(dt_thuan, avg_working_capital)

        equity_ratio = safe_div(vcsh_current, tts_current)
        d_to_e_ratio = safe_div(npt, vcsh_current)
        
        ros_ratio = safe_div(lnst, dt_thuan) * 100 
        roa_ratio = safe_div(lnst, avg_tts) * 100
        
        if avg_vcsh <= 0:
             roe_ratio = np.nan
        else:
             roe_ratio = safe_div(lnst, avg_vcsh) * 100

        ratios_list.append({'Chỉ tiêu': 'Hệ số Thanh toán ngắn hạn (Current Ratio)', y: current_ratio, 'Type': 'Liquidity'})
        ratios_list.append({'Chỉ tiêu': 'Hệ số Thanh toán nhanh (Quick Ratio)', y: quick_ratio, 'Type': 'Liquidity'})
        ratios_list.append({'Chỉ tiêu': 'Vòng quay Hàng tồn kho (Lần)', y: inv_turnover, 'Type': 'Activity'})
        ratios_list.append({'Chỉ tiêu': 'Thời gian Tồn kho (Ngày)', y: inv_days, 'Type': 'Activity'})
        ratios_list.append({'Chỉ tiêu': 'Vòng quay các khoản phải thu (Lần)', y: rcv_turnover, 'Type': 'Activity'})
        ratios_list.append({'Chỉ tiêu': 'Kỳ phải thu bình quân (Ngày)', y: rcv_days, 'Type': 'Activity'})
        ratios_list.append({'Chỉ tiêu': 'Vòng quay Vốn lưu động (Lần)', y: wcl_turnover, 'Type': 'Activity'})
        ratios_list.append({'Chỉ tiêu': 'Hệ số Tự tài trợ (Equity Ratio)', y: equity_ratio, 'Type': 'Solvency'})
        ratios_list.append({'Chỉ tiêu': 'Hệ số Nợ trên Vốn chủ sở hữu (Debt-to-Equity Ratio)', y: d_to_e_ratio, 'Type': 'Solvency'})
        ratios_list.append({'Chỉ tiêu': 'Hệ số Sinh lời Doanh thu (ROS) (%)', y: ros_ratio, 'Type': 'Profitability'})
        ratios_list.append({'Chỉ tiêu': 'Hệ số Sinh lời Tài sản (ROA) (%)', y: roa_ratio, 'Type': 'Profitability'})
        ratios_list.append({'Chỉ tiêu': 'Hệ số Sinh lời Vốn chủ sở hữu (ROE) (%)', y: roe_ratio, 'Type': 'Profitability'})
        
    df_temp_ratios = pd.DataFrame(ratios_list)
    df_final_ratios = df_temp_ratios.pivot_table(index=['Chỉ tiêu', 'Type'], values=years, aggfunc='first').reset_index()
    
    type_order = ['Liquidity', 'Activity', 'Solvency', 'Profitability']
    df_final_ratios['Type'] = pd.Categorical(df_final_ratios['Type'], categories=type_order, ordered=True)
    df_final_ratios = df_final_ratios.sort_values('Type').drop(columns=['Type']).reset_index(drop=True)
    
    df_final_ratios['S.S Tuyệt đối (Y2 vs Y1)'] = df_final_ratios['Năm 2'] - df_final_ratios['Năm 1']
    
    return df_bs, df_is, df_ratios, df_final_ratios

# --- Hàm gọi API Gemini cho Phân tích Báo cáo (Single-shot analysis) ---
def get_ai_analysis(data_for_ai, api_key):
    """Gửi dữ liệu phân tích đến Gemini API và nhận nhận xét."""
    try:
        client = genai.Client(api_key=api_key)
        model_name = 'gemini-2.5-flash'  
        
        system_instruction_text = (
            "Bạn là một chuyên gia phân tích tài chính chuyên nghiệp. "
            "Dựa trên dữ liệu đã cung cấp, hãy đưa ra một nhận xét khách quan, ngắn gọn (khoảng 3-4 đoạn) về tình hình tài chính của doanh nghiệp. "
            "Đánh giá tập trung vào tốc độ tăng trưởng, thay đổi cơ cấu tài sản, **tỷ trọng chi phí/doanh thu thuần**, **hiệu quả hoạt động (Vòng quay Tồn kho, Phải thu, Vốn lưu động)**, **cấu trúc vốn (Hệ số tự tài trợ và Hệ số nợ/VCSH)**, và **khả năng sinh lời (ROS, ROA, ROE)** trong 3 năm/kỳ."
        )
        
        user_prompt = f"""
        {system_instruction_text}
        Dữ liệu thô và chỉ số:<br>
        {data_for_ai}
        """
        response = client.models.generate_content(model=model_name, contents=user_prompt)
        return response.text
    except Exception as e:
        return f"Đã xảy ra lỗi không xác định khi gọi Gemini (Phân tích): {e}"

# --- Hàm gọi API Gemini cho CHAT tương tác (có quản lý lịch sử) ---
def get_chat_response(prompt, chat_history_st, context_data, api_key):
    try:
        client = genai.Client(api_key=api_key)
        model_name = 'gemini-2.5-flash'
        
        system_instruction_text = (
            "Bạn là một trợ lý phân tích tài chính thông minh (Financial Analyst Assistant). "
            "Bạn phải trả lời các câu hỏi của người dùng dựa trên dữ liệu tài chính đã xử lý sau. "
            "Dữ liệu này bao gồm tốc độ tăng trưởng, so sánh tuyệt đối/tương đối, tỷ trọng cơ cấu, tỷ trọng chi phí/doanh thu thuần, và **các chỉ số tài chính chủ chốt (Thanh toán, Hoạt động, Cấu trúc Vốn, Sinh lời)** trong 3 kỳ Báo cáo tài chính. "
            "Nếu người dùng hỏi một câu không liên quan đến dữ liệu tài chính hoặc phân tích, hãy lịch sự từ chối trả lời. "
            "Dữ liệu tài chính đã xử lý (được trình bày dưới dạng Markdown để bạn dễ hiểu): \n\n" + context_data
        )
        
        gemini_history = []
        for msg in chat_history_st[1:]: 
            role = "user" if msg["role"] == "user" else "model"
            gemini_history.append({"role": role, "parts": [{"text": msg["content"]}]})
        
        last_user_prompt = prompt
        
        final_prompt = f"""
        {system_instruction_text}
        ---
        Câu hỏi của người dùng: {last_user_prompt}
        """

        full_contents = gemini_history
        full_contents.append({"role": "user", "parts": [{"text": final_prompt}]})

        response = client.models.generate_content(model=model_name, contents=full_contents)
        return response.text
    except Exception as e:
        return f"Đã xảy ra lỗi không xác định khi gọi Gemini (Chat): {e}"


# --- Chức năng 1: Tải File ---
uploaded_file = st.file_uploader(
    "1. Tải file Excel (Sheet 1: BĐKT và KQKD - Tối thiểu 3 cột năm)",
    type=['xlsx', 'xls']
)

if uploaded_file is not None:
    try: # <--- KHỐI TRY CHÍNH BẮT ĐẦU TỪ ĐÂY
        
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

        xls = pd.ExcelFile(uploaded_file)
        
        try:
            df_raw_bs = xls.parse(xls.sheet_names[0], header=0) 
            df_raw_bs = clean_column_names(df_raw_bs)
        except Exception:
            raise Exception("Không thể đọc Sheet 1 (Bảng CĐKT). Vui lòng kiểm tra định dạng sheet.")
            
        st.info("Đang xử lý file... Giả định BĐKT và KQKD nằm chung 1 sheet.")
        
        df_raw_full = df_raw_bs.rename(columns={df_raw_bs.columns[0]: 'Chỉ tiêu'})
        
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
            
            if split_index > 0:
                df_raw_bs = df_raw_full.loc[:split_index-1].copy()
            else:
                df_raw_bs = pd.DataFrame(columns=df_raw_full.columns)
                
            df_raw_is = df_raw_full.loc[split_index:].copy()
            
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
                df_raw_is = df_raw_is.loc[header_row_index+1:]
                
                if df_raw_is.empty:
                    st.warning("Phần KQKD không có dữ liệu. Bỏ qua phân tích KQKD.")
                    df_raw_is = pd.DataFrame()
                else:
                    df_raw_is.columns = new_header
                    col_to_rename = df_raw_is.columns[0]
                    if pd.isna(col_to_rename) or str(col_to_rename).strip() == '':
                         df_raw_is.rename(columns={col_to_rename: 'Chỉ tiêu'}, inplace=True)
                    else:
                         df_raw_is = df_raw_is.rename(columns={df_raw_is.columns[0]: 'Chỉ tiêu'})
        
        if not df_raw_bs.empty and df_raw_bs.columns[0] != 'Chỉ tiêu':
            df_raw_bs = df_raw_bs.rename(columns={df_raw_bs.columns[0]: 'Chỉ tiêu'})
            
        if not df_raw_is.empty:
            df_raw_is.columns = [str(col) for col in df_raw_is.columns]
        
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
        
        
        if not df_raw_bs.empty and len(df_raw_bs) > 1:
            df_raw_bs = df_raw_bs.drop(df_raw_bs.index[0])
        
        if not df_raw_is.empty:
            first_data_col = col_nam_1 
            
            if 'Chỉ tiêu' in df_raw_is.columns:
                potential_name_cols = [col for i, col in enumerate(df_raw_is.columns) if i > 0 and i < 4]
                
                for name_col in potential_name_cols:
                    df_raw_is[name_col] = df_raw_is[name_col].astype(str).str.strip()
                    
                    df_raw_is['Chỉ tiêu'] = df_raw_is.apply(
                        lambda row: row[name_col] if pd.isna(row['Chỉ tiêu']) or str(row['Chỉ tiêu']).strip() == '' else row['Chỉ tiêu'], 
                        axis=1
                    )
                
            df_raw_is['Chỉ tiêu'] = df_raw_is['Chỉ tiêu'].astype(str).str.strip()
            df_raw_is = df_raw_is[df_raw_is['Chỉ tiêu'].str.len() > 0].copy()
            df_raw_is = df_raw_is[df_raw_is['Chỉ tiêu'].astype(str) != '0'].copy()
                
            if first_data_col in df_raw_is.columns:
                df_raw_is[first_data_col] = pd.to_numeric(df_raw_is[first_data_col], errors='coerce')
                df_raw_is = df_raw_is[df_raw_is[first_data_col].notnull()].copy()
            else:
                st.warning(f"Lỗi: Không tìm thấy cột dữ liệu đầu tiên '{first_data_col}' trong KQKD để làm sạch. Bỏ qua phân tích KQKD.")
                df_raw_is = pd.DataFrame()


        cols_to_keep = ['Chỉ tiêu', col_nam_1, col_nam_2, col_nam_3]

        try:
            df_bs_final = df_raw_bs[cols_to_keep].copy()
            df_bs_final.columns = ['Chỉ tiêu', 'Năm 1', 'Năm 2', 'Năm 3']
            df_bs_final = df_bs_final.dropna(subset=['Chỉ tiêu'])
        except KeyError as ke:
             st.warning(f"Lỗi truy cập cột: {ke}. BĐKT có thể rỗng hoặc bị mất cột 'Chỉ tiêu'. Khởi tạo BĐKT rỗng.")
             df_bs_final = pd.DataFrame(columns=['Chỉ tiêu', 'Năm 1', 'Năm 2', 'Năm 3'])
        
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

        def filter_zero_rows(df):
            if df.empty:
                return df
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

        if not df_bs_processed.empty:
            
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
            
            # --- Chức năng 2 & 3: Hiển thị Kết quả theo Tabs ---
            st.subheader("2. Phân tích Bảng Cân đối Kế toán & 3. Phân tích Tỷ trọng Cơ cấu Tài sản")
            
            df_growth = df_bs_processed[['Chỉ tiêu', 'Năm 1', 'Năm 2', 'Năm 3', 
                                         'Delta (Y2 vs Y1)', 'Growth (Y2 vs Y1)', 
                                         'Delta (Y3 vs Y2)', 'Growth (Y3 vs Y2)']].copy()
            df_growth.columns = ['Chỉ tiêu', Y1_Name, Y2_Name, Y3_Name, 
                                 f'S.S Tuyệt đối ({Y2_Name} vs {Y1_Name})', f'S.S Tương đối (%) ({Y2_Name} vs {Y1_Name})',
                                 f'S.S Tuyệt đối ({Y3_Name} vs {Y2_Name})', f'S.S Tương đối (%) ({Y3_Name} vs {Y2_Name})']
            
            df_structure = df_bs_processed[['Chỉ tiêu', 'Năm 1', 'Năm 2', 'Năm 3', 
                                            'Tỷ trọng Năm 1 (%)', 'Tỷ trọng Năm 2 (%)', 'Tỷ trọng Năm 3 (%)']].copy()
            df_structure.columns = ['Chỉ tiêu', Y1_Name, Y2_Name, Y3_Name, 
                                    f'Tỷ trọng {Y1_Name} (%)', f'Tỷ trọng {Y2_Name} (%)', f'Tỷ trọng {Y3_Name} (%)']

            tab1, tab2 = st.tabs(["📈 Tốc độ Tăng trưởng Bảng CĐKT", "🏗️ Tỷ trọng Cơ cấu Tài sản"])
            
            with tab1:
                st.markdown("##### Bảng phân tích Tốc độ Tăng trưởng & So sánh Tuyệt đối (Bảng CĐKT)")
                st.dataframe(df_growth.style.apply(highlight_financial_items, axis=1).format({
                    Y1_Name: format_vn_currency, Y2_Name: format_vn_currency, Y3_Name: format_vn_currency,
                    f'S.S Tuyệt đối ({Y2_Name} vs {Y1_Name})': format_vn_delta_currency,
                    f'S.S Tuyệt đối ({Y3_Name} vs {Y2_Name})': format_vn_delta_currency,
                    f'S.S Tương đối (%) ({Y2_Name} vs {Y1_Name})': format_vn_percentage,
                    f'S.S Tương đối (%) ({Y3_Name} vs {Y2_Name})': format_vn_percentage
                }), use_container_width=True, hide_index=True)
                
            with tab2:
                st.markdown("##### Bảng phân tích Tỷ trọng Cơ cấu Tài sản (%)")
                st.dataframe(df_structure.style.apply(highlight_financial_items, axis=1).format({
                    Y1_Name: format_vn_currency, Y2_Name: format_vn_currency, Y3_Name: format_vn_currency,
                    f'Tỷ trọng {Y1_Name} (%)': format_vn_percentage,
                    f'Tỷ trọng {Y2_Name} (%)': format_vn_percentage,
                    f'Tỷ trọng {Y3_Name} (%)': format_vn_percentage
                }), use_container_width=True, hide_index=True)
                
            # --- CHỨC NĂNG 4: KQKD ---
            st.subheader("4. Phân tích Kết quả hoạt động kinh doanh")
            if not df_is_processed.empty:
                df_is_display = df_is_processed[['Chỉ tiêu', 'Năm 1', 'Năm 2', 'Năm 3', 
                                                 'S.S Tuyệt đối (Y2 vs Y1)', 'S.S Tương đối (%) (Y2 vs Y1)',
                                                 'S.S Tuyệt đối (Y3 vs Y2)', 'S.S Tương đối (%) (Y3 vs Y2)'
                                                 ]].copy()
                df_is_display.columns = ['Chỉ tiêu', Y1_Name, Y2_Name, Y3_Name, 
                                         f'S.S Tuyệt đối ({Y2_Name} vs {Y1_Name})', f'S.S Tương đối (%) ({Y2_Name} vs {Y1_Name})',
                                         f'S.S Tuyệt đối ({Y3_Name} vs {Y2_Name})', f'S.S Tương đối (%) ({Y3_Name} vs {Y2_Name})']
                
                st.markdown(f"##### Bảng so sánh Kết quả hoạt động kinh doanh ({Y2_Name} vs {Y1_Name} và {Y3_Name} vs {Y2_Name})")
                
                st.dataframe(df_is_display.style.apply(highlight_financial_items, axis=1).format({
                    Y1_Name: format_vn_currency, Y2_Name: format_vn_currency, Y3_Name: format_vn_currency,
                    f'S.S Tuyệt đối ({Y2_Name} vs {Y1_Name})': format_vn_delta_currency,
                    f'S.S Tương đối (%) ({Y2_Name} vs {Y1_Name})': format_vn_percentage,
                    f'S.S Tuyệt đối ({Y3_Name} vs {Y2_Name})': format_vn_delta_currency, 
                    f'S.S Tương đối (%) ({Y3_Name} vs {Y2_Name})': format_vn_percentage 
                }), use_container_width=True, hide_index=True)

                is_context = df_is_processed.to_markdown(index=False)
            else:
                st.info("Không có dữ liệu Báo cáo Kết quả hoạt động kinh doanh để hiển thị.")
                is_context = "Không tìm thấy dữ liệu Báo cáo Kết quả hoạt động kinh doanh."

            
            # --- CHỨC NĂNG 5: TỶ TRỌNG CHI PHÍ / DOANH THU THUẦN ---
            st.subheader("5. Tỷ trọng Chi phí/Doanh thu thuần (%)")
            if not df_ratios_processed.empty:
                df_ratios_display = df_ratios_processed.copy()
                df_ratios_display.columns = ['Chỉ tiêu', Y1_Name, Y2_Name, Y3_Name, 
                                             f'So sánh Tương đối ({Y2_Name} vs {Y1_Name})']
                
                st.dataframe(df_ratios_display.style.apply(highlight_financial_items, axis=1).format({
                    Y1_Name: format_vn_percentage, Y2_Name: format_vn_percentage, Y3_Name: format_vn_percentage,
                    f'So sánh Tương đối ({Y2_Name} vs {Y1_Name})': format_vn_delta_ratio
                }), use_container_width=True, hide_index=True)
                
                ratios_context = df_ratios_processed.to_markdown(index=False)
            else:
                st.info("Không thể tính Tỷ trọng Chi phí/Doanh thu thuần do thiếu dữ liệu KQKD.")
                ratios_context = "Không tìm thấy dữ liệu Tỷ trọng Chi phí/Doanh thu thuần."
            
            # --- CHỨC NĂNG 6: CÁC HỆ SỐ TÀI CHÍNH CHỦ CHỐT ---
            st.subheader("6. Các Hệ số Tài chính Chủ chốt (Thanh toán, Hoạt động, Cấu trúc Vốn, Sinh lời) 🔑")
            if not df_financial_ratios_processed.empty:
                df_ratios_final_display = df_financial_ratios_processed.copy()
                
                cols_to_display = ['Chỉ tiêu', 'Năm 1', 'Năm 2', 'Năm 3', 'S.S Tuyệt đối (Y2 vs Y1)']
                cols_to_display = [col for col in cols_to_display if col in df_ratios_final_display.columns]
                df_ratios_final_display = df_ratios_final_display[cols_to_display]
                
                rename_map = {
                    'Năm 1': Y1_Name, 'Năm 2': Y2_Name, 'Năm 3': Y3_Name,
                    'S.S Tuyệt đối (Y2 vs Y1)': f'So sánh Tuyệt đối ({Y2_Name} vs {Y1_Name})'
                }
                df_ratios_final_display.columns = [rename_map.get(col, col) for col in df_ratios_final_display.columns]
                
                st.markdown(f"##### Bảng tính Chỉ số Tài chính Chủ chốt ({Y1_Name} - {Y3_Name})")
                
                st.dataframe(df_ratios_final_display.style.apply(highlight_financial_items, axis=1).format({
                    Y1_Name: format_vn_delta_ratio, Y2_Name: format_vn_delta_ratio, Y3_Name: format_vn_delta_ratio,
                    f'So sánh Tuyệt đối ({Y2_Name} vs {Y1_Name})': format_vn_delta_ratio
                }), use_container_width=True, hide_index=True)
                
                key_ratios_context = df_financial_ratios_processed.to_markdown(index=False)
            else:
                st.info("Không thể tính các Chỉ số Tài chính Chủ chốt do thiếu dữ liệu.")
                key_ratios_context = "Không tìm thấy dữ liệu Chỉ tiêu Tài chính Chủ chốt."
            
            # --- CẬP NHẬT CONTEXT CHO CHATBOT ---
            data_for_chat_context = f"""
            **BẢNG CÂN ĐỐI KẾ TOÁN (Balance Sheet Analysis):**
            {df_bs_processed.to_markdown(index=False)}
            
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
                 st.session_state.messages[0]["content"] = f"Phân tích 3 kỳ ({Y1_Name} đến {Y3_Name}) đã hoàn tất! Bây giờ bạn có thể hỏi tôi bất kỳ điều gì về BĐKT, KQKD, tỷ trọng chi phí, **các chỉ số thanh toán**, **hiệu quả sử dụng vốn**, **cấu trúc nợ**, và **khả năng sinh lời (ROS, ROA, ROE)**."

            # -----------------------------------------------------------------
            # [SẮP XẾP LẠI] CHỨC NĂNG 7: CHATBOT
            # -----------------------------------------------------------------
            st.subheader("7. Trò chuyện và Hỏi đáp (Gemini AI) 💬") 
            if st.session_state.data_for_chat is None:
                st.info("Vui lòng tải lên và xử lý báo cáo tài chính trước khi bắt đầu trò chuyện với AI.")
            else:
                for message in st.session_state.messages:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])

                if prompt := st.chat_input("Hỏi AI về báo cáo tài chính này..."):
                    api_key = st.secrets.get("GEMINI_API_KEY")
                    
                    if not api_key:
                        st.error("Lỗi: Không tìm thấy Khóa API. Vui lòng cấu hình Khóa 'GEMINI_API_KEY' trong Streamlit Secrets.")
                    else:
                        st.session_state.messages.append({"role": "user", "content": prompt})
                        with st.chat_message("user"):
                            st.markdown(prompt)

                        with st.chat_message("assistant"):
                            with st.spinner("Đang gửi câu hỏi và chờ Gemini trả lời..."):
                                full_response = get_chat_response(
                                    prompt, 
                                    st.session_state.messages, 
                                    st.session_state.data_for_chat, 
                                    api_key
                                )
                                st.markdown(full_response)
                        
                        st.session_state.messages.append({"role": "assistant", "content": full_response})

            # -----------------------------------------------------------------
            # [XÓA] MỤC 8 ĐÃ BỊ XÓA THEO YÊU CẦU
            # -----------------------------------------------------------------
            
            # --- KẾT THÚC MỤC 8 ---

    # [SỬA LỖI] Hai khối 'except' này phải nằm ngang hàng với 'try' (đã sửa)
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
