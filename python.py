import streamlit as st
import pandas as pd
from google import genai
from google.genai.errors import APIError
# ĐÃ SỬA LỖI: Loại bỏ import GenerationConfig và SystemInstruction để tránh lỗi Pydantic
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

# --- Hàm tính toán chính (Sử dụng Caching để Tối ưu hiệu suất) ---
@st.cache_data
def process_financial_data(df_balance_sheet, df_income_statement):
    """
    Thực hiện các phép tính Tăng trưởng, So sánh Tuyệt đối và Tỷ trọng cho cả Bảng Cân đối và Báo cáo Kết quả Kinh doanh.
    Trả về tuple (df_bs_processed, df_is_processed)
    """
    
    # -----------------------------------------------------------------
    # PHẦN 1: XỬ LÝ BẢNG CÂN ĐỐI KẾ TOÁN (BALANCE SHEET - BS)
    # -----------------------------------------------------------------
    df_bs = df_balance_sheet.copy()
    
    # Đảm bảo các giá trị là số để tính toán
    numeric_cols_bs = ['Năm 1', 'Năm 2', 'Năm 3']
    for col in numeric_cols_bs:
        df_bs[col] = pd.to_numeric(df_bs[col], errors='coerce').fillna(0)
    
    # Tính toán Tăng trưởng & So sánh Tuyệt đối (Delta / Growth)
    df_bs['Delta (Y2 vs Y1)'] = df_bs['Năm 2'] - df_bs['Năm 1']
    df_bs['Growth (Y2 vs Y1)'] = ((df_bs['Delta (Y2 vs Y1)'] / df_bs['Năm 1'].replace(0, 1e-9)) * 100)
    df_bs['Delta (Y3 vs Y2)'] = df_bs['Năm 3'] - df_bs['Năm 2']
    df_bs['Growth (Y3 vs Y2)'] = ((df_bs['Delta (Y3 vs Y2)'] / df_bs['Năm 2'].replace(0, 1e-9)) * 100)

    # Tính Tỷ trọng theo Tổng Tài sản
    tong_tai_san_row = df_bs[df_bs['Chỉ tiêu'].str.contains('TỔNG CỘNG TÀI SẢN|TỔNG CỘNG', case=False, na=False)]
    
    if tong_tai_san_row.empty:
        raise ValueError("Không tìm thấy chỉ tiêu 'TỔNG CỘNG TÀI SẢN' hoặc 'TỔNG CỘNG' để tính tỷ trọng. Vui lòng kiểm tra tên chỉ tiêu trong file.")

    tong_tai_san_N1 = tong_tai_san_row['Năm 1'].iloc[0]
    tong_tai_san_N2 = tong_tai_san_row['Năm 2'].iloc[0]
    tong_tai_san_N3 = tong_tai_san_row['Năm 3'].iloc[0]

    divisor_N1 = tong_tai_san_N1 if tong_tai_san_N1 != 0 else 1e-9
    divisor_N2 = tong_tai_san_N2 if tong_tai_san_N2 != 0 else 1e-9
    divisor_N3 = tong_tai_san_N3 if tong_tai_san_N3 != 0 else 1e-9

    df_bs['Tỷ trọng Năm 1 (%)'] = (df_bs['Năm 1'] / divisor_N1) * 100
    df_bs['Tỷ trọng Năm 2 (%)'] = (df_bs['Năm 2'] / divisor_N2) * 100
    df_bs['Tỷ trọng Năm 3 (%)'] = (df_bs['Năm 3'] / divisor_N3) * 100
    
    # -----------------------------------------------------------------
    # PHẦN 2: XỬ LÝ BÁO CÁO KẾT QUẢ KINH DOANH (INCOME STATEMENT - IS)
    # -----------------------------------------------------------------
    df_is = df_income_statement.copy()
    
    # Đảm bảo các giá trị là số để tính toán
    numeric_cols_is = ['Năm 1', 'Năm 2', 'Năm 3']
    for col in numeric_cols_is:
        df_is[col] = pd.to_numeric(df_is[col], errors='coerce').fillna(0)
    
    # Tính toán Tăng trưởng & So sánh Tuyệt đối (Delta / Growth)
    # Y2 vs Y1 (2024 vs 2023)
    df_is['S.S Tuyệt đối (Y2 vs Y1)'] = df_is['Năm 2'] - df_is['Năm 1']
    df_is['S.S Tương đối (%) (Y2 vs Y1)'] = ((df_is['S.S Tuyệt đối (Y2 vs Y1)'] / df_is['Năm 1'].replace(0, 1e-9)) * 100)
    
    # Y3 vs Y2 (6T/2025 vs 2024) - Giữ lại cột Y3 để hiển thị
    # Nếu muốn so sánh tăng trưởng, cần xử lý Chu kỳ 3 (6T) khác Chu kỳ 2 (12T), nhưng ở đây chỉ thực hiện so sánh thô.
    
    return df_bs, df_is

# --- Hàm gọi API Gemini cho Phân tích Báo cáo (Single-shot analysis) ---
def get_ai_analysis(data_for_ai, api_key):
    """Gửi dữ liệu phân tích đến Gemini API và nhận nhận xét."""
    try:
        client = genai.Client(api_key=api_key)
        model_name = 'gemini-2.5-flash' 
        
        system_instruction_text = (
            "Bạn là một chuyên gia phân tích tài chính chuyên nghiệp. "
            "Dựa trên dữ liệu đã cung cấp, hãy đưa ra một nhận xét khách quan, ngắn gọn (khoảng 3-4 đoạn) về tình hình tài chính của doanh nghiệp. "
            "Đánh giá tập trung vào tốc độ tăng trưởng qua các chu kỳ, thay đổi cơ cấu tài sản, khả năng thanh toán và **kết quả hoạt động kinh doanh** trong 3 năm/kỳ."
        )
        
        # SỬA LỖI: Ghép System Instruction vào đầu Prompt để tương thích API
        user_prompt = f"""
        {system_instruction_text}
        
        Dữ liệu thô và chỉ số:
        {data_for_ai}
        """

        # Truyền prompt duy nhất
        response = client.models.generate_content(
            model=model_name,
            contents=user_prompt 
        )
        return response.text

    except APIError as e:
        return f"Lỗi gọi Gemini API: Vui lòng kiểm tra Khóa API hoặc giới hạn sử dụng. Chi tiết lỗi: {e}"
    except KeyError:
        return "Lỗi: Không tìm thấy Khóa API 'GEMINI_API_KEY'. Vui lòng kiểm tra cấu hình Secrets trên Streamlit Cloud."
    except Exception as e:
        return f"Đã xảy ra lỗi không xác định: {e}"


# --- Hàm gọi API Gemini cho CHAT tương tác (có quản lý lịch sử) ---
def get_chat_response(prompt, chat_history_st, context_data, api_key):
    try:
        client = genai.Client(api_key=api_key)
        model_name = 'gemini-2.5-flash'
        
        # 1. Định nghĩa System Instruction
        system_instruction_text = (
            "Bạn là một trợ lý phân tích tài chính thông minh (Financial Analyst Assistant). "
            "Bạn phải trả lời các câu hỏi của người dùng dựa trên dữ liệu tài chính đã xử lý sau. "
            "Dữ liệu này bao gồm tốc độ tăng trưởng, so sánh tuyệt đối/tương đối và tỷ trọng trong 3 kỳ Báo cáo tài chính, cùng với 3 chỉ số thanh toán hiện hành, và **phân tích kết quả hoạt động kinh doanh**. "
            "Nếu người dùng hỏi một câu không liên quan đến dữ liệu tài chính hoặc phân tích, hãy lịch sự từ chối trả lời. "
            "Dữ liệu tài chính đã xử lý (được trình bày dưới dạng Markdown để bạn dễ hiểu): \n\n" + context_data
        )
        
        # 2. Chuyển đổi lịch sử Streamlit sang định dạng Gemini
        gemini_history = []
        # Bắt đầu từ tin nhắn thứ hai trong lịch sử ST (bỏ qua tin nhắn chào mừng đầu tiên)
        for msg in chat_history_st[1:]: 
            # Đảm bảo chỉ có role 'user' và 'model' được sử dụng
            role = "user" if msg["role"] == "user" else "model"
            gemini_history.append({"role": role, "parts": [{"text": msg["content"]}]})
        
        # 3. Ghép System Instruction và Prompt mới nhất vào Content cuối cùng
        last_user_prompt = prompt
        
        # Tạo prompt cuối cùng bằng cách ghép System Instruction, Context Data và Prompt người dùng
        final_prompt = f"""
        {system_instruction_text}
        
        ---
        
        Câu hỏi của người dùng: {last_user_prompt}
        """

        # Thêm prompt cuối cùng (final_prompt) vào cuối lịch sử
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
    "1. Tải file Excel/CSV Báo cáo Tài chính (Bảng CĐKT & KQKD - Tối thiểu 3 cột năm)",
    type=['xlsx', 'xls', 'csv']
)

if uploaded_file is not None:
    try:
        # Xử lý file dựa trên định dạng
        # Dùng header=0 để lấy tên cột là ngày tháng
        if uploaded_file.name.endswith(('.xlsx', '.xls')):
            df_raw = pd.read_excel(uploaded_file, header=0)
        elif uploaded_file.name.endswith('.csv'):
            df_raw = pd.read_csv(uploaded_file, header=0)
        else:
            raise Exception("Định dạng file không được hỗ trợ.")

        # --- TIỀN XỬ LÝ (PRE-PROCESSING) DỮ LIỆU ---
        
        # 1. Đặt tên cột đầu tiên là 'Chỉ tiêu' (Dựa trên snippet 'KHOẢN MỤC')
        df_raw = df_raw.rename(columns={df_raw.columns[0]: 'Chỉ tiêu'})
        
        # 2. Xác định cột năm/kỳ gần nhất ('Năm 3'), 'Năm 2', 'Năm 1'
        
        # TÌM KIẾM CỘT NGÀY THÁNG LINH HOẠT
        value_cols_unique = {} # Dùng dictionary để đảm bảo key (giá trị ngày) là duy nhất
        for col in df_raw.columns:
            col_str = str(col)
            
            # Hàm phụ để chuẩn hóa tên cột (loại bỏ giờ và định dạng YYYY-MM-DD)
            def normalize_date_col(name):
                # Loại bỏ phần giờ nếu có
                if ' ' in name:
                    name = name.split(' ')[0]
                return name
            
            normalized_name = normalize_date_col(col_str)
            
            # Kiểm tra nếu tên chuẩn hóa là ngày tháng (ví dụ: '2023-12-31')
            if len(normalized_name) >= 10 and normalized_name[4] == '-' and normalized_name[7] == '-' and normalized_name[:4].isdigit():
                 # Nếu tên ngày tháng (normalized_name) chưa có trong dict, thêm cột gốc (col) vào
                 if normalized_name not in value_cols_unique:
                    value_cols_unique[normalized_name] = col
            # Hoặc tìm các cột có tên là năm đơn thuần (VD: 2023)
            elif normalized_name.isdigit() and len(normalized_name) == 4 and normalized_name.startswith('20'):
                 if normalized_name not in value_cols_unique:
                    value_cols_unique[normalized_name] = col

        # Lấy danh sách các cột gốc không trùng lặp (Value của dictionary)
        value_cols = list(value_cols_unique.values())
        
        if len(value_cols) < 3: # Yêu cầu 3 năm để tính toán 2 chu kỳ
            st.warning(f"Chỉ tìm thấy {len(value_cols)} cột năm. Ứng dụng cần ít nhất 3 năm/kỳ để so sánh.")
            st.stop()
            
        # Chọn 3 cột năm gần nhất (Sắp xếp theo tên cột/ngày tháng, mới nhất (Y3) lên đầu)
        value_cols.sort(key=lambda x: str(x), reverse=True)
        
        col_nam_3 = value_cols[0] # Newest (Năm 3)
        col_nam_2 = value_cols[1] # Middle (Năm 2)
        col_nam_1 = value_cols[2] # Oldest (Năm 1)
        
        # 3. Lọc bỏ hàng đầu tiên chứa các chỉ số so sánh (SS) không cần thiết
        df_raw = df_raw.drop(df_raw.index[0])
        
        # 4. Tách Bảng cân đối Kế toán (Balance Sheet) và Báo cáo Kết quả Kinh doanh (Income Statement)
        
        # TÌM KIẾM LINH HOẠT HƠN CHO BÁO CÁO KẾT QUẢ KINH DOANH
        # Tìm dòng đầu tiên chứa một trong các chỉ tiêu chính của KQKD: 'Doanh thu', 'Lợi nhuận', 'CHỈ TIÊU'
        is_keywords = ['Doanh thu', 'Lợi nhuận', 'CHỈ TIÊU']
        income_statement_start_index = df_raw[
            df_raw['Chỉ tiêu'].astype(str).str.contains('|'.join(is_keywords), case=False, na=False)
        ].index
        
        if income_statement_start_index.empty:
            st.warning("Không tìm thấy dòng 'Doanh thu', 'Lợi nhuận' hoặc 'CHỈ TIÊU' để phân tách Bảng Cân đối Kế toán và Kết quả Kinh doanh. Chỉ phân tích Bảng CĐKT.")
            # Xử lý toàn bộ là Bảng CĐKT
            df_balance_sheet_raw = df_raw.copy()
            df_income_statement_raw = pd.DataFrame(columns=['Chỉ tiêu', col_nam_1, col_nam_2, col_nam_3]) # DF rỗng
            
        else:
            # Lấy index dòng bắt đầu phần Kết quả Kinh doanh (dòng đầu tiên chứa từ khóa)
            is_start_idx = income_statement_start_index[0]
            
            # Bảng Cân đối Kế toán (từ đầu đến ngay trước dòng chứa từ khóa KQKD)
            df_balance_sheet_raw = df_raw.loc[:is_start_idx-1].copy()
            
            # Báo cáo Kết quả Kinh doanh (từ dòng chứa từ khóa đến hết)
            df_income_statement_raw = df_raw.loc[is_start_idx:].copy()
            
            # **LƯU Ý: Loại bỏ hàng phụ thứ 2 của KQKD nếu có (như hàng 'SS (+/-)')**
            if not df_income_statement_raw.empty and len(df_income_statement_raw) > 0:
                # Nếu hàng đầu tiên của KQKD có vẻ là hàng phụ (chứa NaN nhiều), ta xóa nó đi
                if df_income_statement_raw.iloc[0].astype(str).str.contains('CHỈ TIÊU', case=False, na=False).any():
                     # Nếu dòng đầu tiên là tiêu đề CHỈ TIÊU, ta bắt đầu từ dòng kế tiếp (is_start_idx+1)
                     df_income_statement_raw = df_raw.loc[is_start_idx+1:].copy()


        # 5. Tạo DataFrame Bảng CĐKT (4 cột cần thiết)
        df_bs_final = df_balance_sheet_raw[['Chỉ tiêu', col_nam_1, col_nam_2, col_nam_3]].copy()
        df_bs_final.columns = ['Chỉ tiêu', 'Năm 1', 'Năm 2', 'Năm 3']
        df_bs_final = df_bs_final.dropna(subset=['Chỉ tiêu'])

        # 6. Tạo DataFrame KQKD (4 cột cần thiết)
        # Kiểm tra để tránh lỗi Key Error nếu KQKD không có đủ 4 cột
        if not df_income_statement_raw.empty:
            df_is_final = df_income_statement_raw[['Chỉ tiêu', col_nam_1, col_nam_2, col_nam_3]].copy()
            df_is_final.columns = ['Chỉ tiêu', 'Năm 1', 'Năm 2', 'Năm 3']
            df_is_final = df_is_final.dropna(subset=['Chỉ tiêu'])
        else:
            df_is_final = pd.DataFrame(columns=['Chỉ tiêu', 'Năm 1', 'Năm 2', 'Năm 3'])


        # Xử lý dữ liệu
        df_bs_processed, df_is_processed = process_financial_data(df_bs_final.copy(), df_is_final.copy())

        if df_bs_processed is not None:
            
            # -----------------------------------------------------
            # CHUẨN HÓA TÊN CỘT ĐỂ HIỂN THỊ (DD/MM/YYYY)
            # -----------------------------------------------------
            def format_col_name(col_name):
                col_name = str(col_name)
                # Loại bỏ phần giờ nếu có
                if ' ' in col_name:
                    col_name = col_name.split(' ')[0]
                
                # Chuyển từ YYYY-MM-DD sang DD/MM/YYYY
                try:
                    # Tách YYYY, MM, DD dựa trên dấu '-'
                    parts = col_name.split('-')
                    if len(parts) == 3:
                        return f"{parts[2]}/{parts[1]}/{parts[0]}"
                except Exception:
                    # Nếu không phải định dạng YYYY-MM-DD (ví dụ: chỉ là '2023'), giữ nguyên
                    pass

                return col_name

            Y1_Name = format_col_name(col_nam_1)
            Y2_Name = format_col_name(col_nam_2)
            Y3_Name = format_col_name(col_nam_3)
            # -----------------------------------------------------
            
            # --- Chức năng 2 & 3: Hiển thị Kết quả theo Tabs ---
            st.subheader("2. Phân tích Bảng Cân đối Kế toán & 3. Phân tích Tỷ trọng Cơ cấu Tài sản")
            
            # 1. TẠO DATAFRAME BẢNG CĐKT TĂNG TRƯỞNG (GHÉP CỘT)
            df_growth = df_bs_processed[['Chỉ tiêu', 'Năm 1', 'Năm 2', 'Năm 3', 
                                    'Delta (Y2 vs Y1)', 'Growth (Y2 vs Y1)', 
                                    'Delta (Y3 vs Y2)', 'Growth (Y3 vs Y2)']].copy()
            
            # Đổi tên cột cho trực quan (theo yêu cầu của người dùng)
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
            
            # Đổi tên cột cho trực quan
            df_structure.columns = [
                'Chỉ tiêu', Y1_Name, Y2_Name, Y3_Name, 
                f'Tỷ trọng {Y1_Name} (%)', f'Tỷ trọng {Y2_Name} (%)', f'Tỷ trọng {Y3_Name} (%)'
            ]

            tab1, tab2 = st.tabs(["📈 Tốc độ Tăng trưởng Bảng CĐKT", "🏗️ Tỷ trọng Cơ cấu Tài sản"])
            
            with tab1:
                st.markdown("##### Bảng phân tích Tốc độ Tăng trưởng & So sánh Tuyệt đối (Bảng CĐKT)")
                st.dataframe(df_growth.style.format({
                    Y1_Name: '{:,.0f}',
                    Y2_Name: '{:,.0f}',
                    Y3_Name: '{:,.0f}',
                    f'S.S Tuyệt đối ({Y2_Name} vs {Y1_Name})': '{:,.0f}',
                    f'S.S Tuyệt đối ({Y3_Name} vs {Y2_Name})': '{:,.0f}',
                    f'S.S Tương đối (%) ({Y2_Name} vs {Y1_Name})': '{:.2f}%',
                    f'S.S Tương đối (%) ({Y3_Name} vs {Y2_Name})': '{:.2f}%'
                }), use_container_width=True, hide_index=True)
                
            with tab2:
                st.markdown("##### Bảng phân tích Tỷ trọng Cơ cấu Tài sản (%)")
                st.dataframe(df_structure.style.format({
                    Y1_Name: '{:,.0f}',
                    Y2_Name: '{:,.0f}',
                    Y3_Name: '{:,.0f}',
                    f'Tỷ trọng {Y1_Name} (%)': '{:.2f}%',
                    f'Tỷ trọng {Y2_Name} (%)': '{:.2f}%',
                    f'Tỷ trọng {Y3_Name} (%)': '{:.2f}%'
                }), use_container_width=True, hide_index=True)
            
            # -----------------------------------------------------
            # CHỨC NĂNG 4: BÁO CÁO KẾT QUẢ HOẠT ĐỘNG KINH DOANH
            # -----------------------------------------------------
            st.subheader("4. Phân tích Kết quả hoạt động kinh doanh")

            if not df_is_processed.empty:
                df_is_display = df_is_processed[['Chỉ tiêu', 'Năm 1', 'Năm 2', 'Năm 3', 
                                                'S.S Tuyệt đối (Y2 vs Y1)', 'S.S Tương đối (%) (Y2 vs Y1)']].copy()
                
                # Đổi tên cột cho trực quan
                df_is_display.columns = [
                    'Chỉ tiêu', Y1_Name, Y2_Name, Y3_Name, 
                    f'S.S Tuyệt đối ({Y2_Name} vs {Y1_Name})', 
                    f'S.S Tương đối (%) ({Y2_Name} vs {Y1_Name})'
                ]
                
                st.markdown("##### Bảng so sánh Kết quả hoạt động kinh doanh (2024 so với 2023)")
                st.dataframe(df_is_display.style.format({
                    Y1_Name: '{:,.0f}',
                    Y2_Name: '{:,.0f}',
                    Y3_Name: '{:,.0f}',
                    f'S.S Tuyệt đối ({Y2_Name} vs {Y1_Name})': '{:,.0f}',
                    f'S.S Tương đối (%) ({Y2_Name} vs {Y1_Name})': '{:.2f}%',
                }), use_container_width=True, hide_index=True)

                # Cập nhật context cho Chatbot
                is_context = df_is_processed.to_markdown(index=False)
            else:
                st.info("Không tìm thấy dữ liệu Báo cáo Kết quả hoạt động kinh doanh để phân tích.")
                is_context = "Không tìm thấy dữ liệu Báo cáo Kết quả hoạt động kinh doanh."

            
            # --- CHỨC NĂNG 5: TÍNH CHỈ SỐ THANH TOÁN ---
            st.subheader("5. Các Chỉ số Tài chính Cơ bản (Tính từ Bảng CĐKT)")
            
            # Khởi tạo giá trị mặc định cho chỉ số thanh toán
            thanh_toan_hien_hanh_N1 = "N/A"
            thanh_toan_hien_hanh_N2 = "N/A"
            thanh_toan_hien_hanh_N3 = "N/A"

            try:
                # Lấy Tài sản ngắn hạn (TSNH)
                tsnh_n3 = df_bs_processed[df_bs_processed['Chỉ tiêu'].str.contains('TÀI SẢN NGẮN HẠN', case=False, na=False)]['Năm 3'].iloc[0]
                tsnh_n2 = df_bs_processed[df_bs_processed['Chỉ tiêu'].str.contains('TÀI SẢN NGẮN HẠN', case=False, na=False)]['Năm 2'].iloc[0]
                tsnh_n1 = df_bs_processed[df_bs_processed['Chỉ tiêu'].str.contains('TÀI SẢN NGẮN HẠN', case=False, na=False)]['Năm 1'].iloc[0]

                # Lấy Nợ ngắn hạn (NNH)
                no_ngan_han_N3 = df_bs_processed[df_bs_processed['Chỉ tiêu'].str.contains('NỢ NGẮN HẠN', case=False, na=False)]['Năm 3'].iloc[0]  
                no_ngan_han_N2 = df_bs_processed[df_bs_processed['Chỉ tiêu'].str.contains('NỢ NGẮN HẠN', case=False, na=False)]['Năm 2'].iloc[0]
                no_ngan_han_N1 = df_bs_processed[df_bs_processed['Chỉ tiêu'].str.contains('NỢ NGẮN HẠN', case=False, na=False)]['Năm 1'].iloc[0]

                # Tính toán
                thanh_toan_hien_hanh_N3 = tsnh_n3 / no_ngan_han_N3
                thanh_toan_hien_hanh_N2 = tsnh_n2 / no_ngan_han_N2
                thanh_toan_hien_hanh_N1 = tsnh_n1 / no_ngan_han_N1
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(
                        label=f"Chỉ số Thanh toán Hiện hành ({Y1_Name})",
                        value=f"{thanh_toan_hien_hanh_N1:.2f} lần"
                    )
                with col2:
                    st.metric(
                        label=f"Chỉ số Thanh toán Hiện hành ({Y2_Name})",
                        value=f"{thanh_toan_hien_hanh_N2:.2f} lần",
                        delta=f"{thanh_toan_hien_hanh_N2 - thanh_toan_hien_hanh_N1:.2f}"
                    )
                with col3:
                    st.metric(
                        label=f"Chỉ số Thanh toán Hiện hành ({Y3_Name})",
                        value=f"{thanh_toan_hien_hanh_N3:.2f} lần",
                        delta=f"{thanh_toan_hien_hanh_N3 - thanh_toan_hien_hanh_N2:.2f}"
                    )
                    
            except IndexError:
                st.warning("Thiếu chỉ tiêu 'TÀI SẢN NGẮN HẠN' hoặc 'NỢ NGẮN HẠN' để tính chỉ số.")
            except ZeroDivisionError:
                st.error("Lỗi chia cho 0 khi tính chỉ số thanh toán. Vui lòng kiểm tra dữ liệu 'Nợ Ngắn Hạn'!")
            
            # --- CẬP NHẬT DỮ LIỆU CHO CHATBOT (CONTEXT) ---
            data_for_chat_context = f"""
            **BẢNG CÂN ĐỐI KẾ TOÁN (Balance Sheet Analysis):**
            {df_bs_processed.to_markdown(index=False)}
            
            **BÁO CÁO KẾT QUẢ KINH DOANH (Income Statement Analysis):**
            {is_context}

            **CHỈ SỐ THANH TOÁN HIỆN HÀNH (Current Ratio):**
            | Chỉ tiêu | Giá trị |
            | :--- | :--- |
            | Thanh toán hiện hành ({Y1_Name}) | {thanh_toan_hien_hanh_N1} |
            | Thanh toán hiện hành ({Y2_Name}) | {thanh_toan_hien_hanh_N2} |
            | Thanh toán hiện hành ({Y3_Name}) | {thanh_toan_hien_hanh_N3} |
            """
            st.session_state.data_for_chat = data_for_chat_context
            
            # Cập nhật tin nhắn chào mừng nếu data đã sẵn sàng
            if st.session_state.messages[0]["content"].startswith("Xin chào!") or st.session_state.messages[0]["content"].startswith("Phân tích"):
                 st.session_state.messages[0]["content"] = f"Phân tích 3 kỳ ({Y1_Name} đến {Y3_Name}) đã hoàn tất! Bây giờ bạn có thể hỏi tôi bất kỳ điều gì về Bảng CĐKT, KQKD và các chỉ số tài chính của báo cáo này."

            # --- Chức năng 6: Nhận xét AI ---
            st.subheader("6. Nhận xét Tình hình Tài chính (AI)")
            
            # Chuẩn bị dữ liệu để gửi cho AI (giống hệt logic data_for_chat_context, thêm chỉ số tăng trưởng TSNH)
            try:
                tsnh_growth_y2y1 = f"{df_bs_processed[df_bs_processed['Chỉ tiêu'].str.contains('TÀI SẢN NGẮN HẠN', case=False, na=False)]['Growth (Y2 vs Y1)'].iloc[0]:.2f}%"
                tsnh_growth_y3y2 = f"{df_bs_processed[df_bs_processed['Chỉ tiêu'].str.contains('TÀI SẢN NGẮN HẠN', case=False, na=False)]['Growth (Y3 vs Y2)'].iloc[0]:.2f}%"
            except IndexError:
                tsnh_growth_y2y1 = "N/A"
                tsnh_growth_y3y2 = "N/A"

            data_for_ai = f"""
            **Bảng Cân đối Kế toán:**
            {df_bs_processed.to_markdown(index=False)}
            
            **Báo cáo Kết quả Kinh doanh:**
            {df_is_processed.to_markdown(index=False)}

            **Các Chỉ số Chính:**
            | Chỉ tiêu | Giá trị |
            | :--- | :--- |
            | Tăng trưởng Tài sản ngắn hạn ({Y2_Name} vs {Y1_Name}) | {tsnh_growth_y2y1} |
            | Tăng trưởng Tài sản ngắn hạn ({Y3_Name} vs {Y2_Name}) | {tsnh_growth_y3y2} |
            | Thanh toán hiện hành ({Y1_Name}) | {thanh_toan_hien_hanh_N1} |
            | Thanh toán hiện hành ({Y2_Name}) | {thanh_toan_hien_hanh_N2} |
            | Thanh toán hiện hành ({Y3_Name}) | {thanh_toan_hien_hanh_N3} |
            """
            
            if st.button("Yêu cầu AI Phân tích (Nhận xét Chung)"):
                api_key = st.secrets.get("GEMINI_API_KEY") 
                
                if api_key:
                    with st.spinner('Đang gửi dữ liệu và chờ Gemini phân tích...'):
                        ai_result = get_ai_analysis(data_for_ai, api_key)
                        st.markdown("**Kết quả Phân tích từ Gemini AI:**")
                        st.info(ai_result)
                else:
                    st.error("Lỗi: Không tìm thấy Khóa API. Vui lòng cấu hình Khóa 'GEMINI_API_KEY' trong Streamlit Secrets.")

    except ValueError as ve:
        st.error(f"Lỗi cấu trúc dữ liệu: {ve}")
        st.session_state.data_for_chat = None # Reset chat context
    except Exception as e:
        st.error(f"Có lỗi xảy ra khi đọc hoặc xử lý file: {e}. Vui lòng kiểm tra định dạng file và đảm bảo có đủ 3 cột năm.")
        st.session_state.data_for_chat = None # Reset chat context

else:
    st.info("Vui lòng tải lên file Excel hoặc CSV để bắt đầu phân tích.")
    st.session_state.data_for_chat = None # Đảm bảo context được reset khi chưa có file

# --- Chức năng 7: Khung Chatbot tương tác ---
st.subheader("7. Trò chuyện và Hỏi đáp (Gemini AI)")
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
                    
                    # Gọi hàm chat mới
                    full_response = get_chat_response(
                        prompt, 
                        st.session_state.messages, 
                        st.session_state.data_for_chat, 
                        api_key
                    )
                    
                    st.markdown(full_response)
            
            # Thêm phản hồi của AI vào lịch sử
            st.session_state.messages.append({"role": "assistant", "content": full_response})
