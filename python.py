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
def process_financial_data(df):
    """Thực hiện các phép tính Tăng trưởng, So sánh Tuyệt đối và Tỷ trọng cho 3 năm (Năm 1, Năm 2, Năm 3)."""
    
    # Đảm bảo các giá trị là số để tính toán
    numeric_cols = ['Năm 1', 'Năm 2', 'Năm 3']
    for col in numeric_cols:
        # Chuyển đổi sang dạng số, thay thế lỗi bằng 0
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # 1. So sánh Y2 vs Y1 (31/12/2024 vs 31/12/2023)
    df['Delta (Y2 vs Y1)'] = df['Năm 2'] - df['Năm 1']
    # Dùng .replace(0, 1e-9) cho Series Pandas để tránh lỗi chia cho 0
    df['Growth (Y2 vs Y1)'] = (
        df['Delta (Y2 vs Y1)'] / df['Năm 1'].replace(0, 1e-9)
    ) * 100

    # 2. So sánh Y3 vs Y2 (30/06/2025 vs 31/12/2024)
    df['Delta (Y3 vs Y2)'] = df['Năm 3'] - df['Năm 2']
    # Dùng .replace(0, 1e-9) cho Series Pandas để tránh lỗi chia cho 0
    df['Growth (Y3 vs Y2)'] = (
        df['Delta (Y3 vs Y2)'] / df['Năm 2'].replace(0, 1e-9)
    ) * 100

    # 3. Tính Tỷ trọng theo Tổng Tài sản
    # Lọc chỉ tiêu "TỔNG CỘNG TÀI SẢN" hoặc "TỔNG CỘNG"
    tong_tai_san_row = df[df['Chỉ tiêu'].str.contains('TỔNG CỘNG TÀI SẢN|TỔNG CỘNG', case=False, na=False)]
    
    if tong_tai_san_row.empty:
        raise ValueError("Không tìm thấy chỉ tiêu 'TỔNG CỘNG TÀI SẢN' hoặc 'TỔNG CỘNG' để tính tỷ trọng. Vui lòng kiểm tra tên chỉ tiêu trong file.")

    tong_tai_san_N1 = tong_tai_san_row['Năm 1'].iloc[0]
    tong_tai_san_N2 = tong_tai_san_row['Năm 2'].iloc[0]
    tong_tai_san_N3 = tong_tai_san_row['Năm 3'].iloc[0]

    # Xử lý mẫu số bằng 0
    divisor_N1 = tong_tai_san_N1 if tong_tai_san_N1 != 0 else 1e-9
    divisor_N2 = tong_tai_san_N2 if tong_tai_san_N2 != 0 else 1e-9
    divisor_N3 = tong_tai_san_N3 if tong_tai_san_N3 != 0 else 1e-9

    # Tính tỷ trọng
    df['Tỷ trọng Năm 1 (%)'] = (df['Năm 1'] / divisor_N1) * 100
    df['Tỷ trọng Năm 2 (%)'] = (df['Năm 2'] / divisor_N2) * 100
    df['Tỷ trọng Năm 3 (%)'] = (df['Năm 3'] / divisor_N3) * 100
    
    return df

# --- Hàm gọi API Gemini cho Phân tích Báo cáo (Single-shot analysis) ---
def get_ai_analysis(data_for_ai, api_key):
    """Gửi dữ liệu phân tích đến Gemini API và nhận nhận xét."""
    try:
        client = genai.Client(api_key=api_key)
        model_name = 'gemini-2.5-flash' 
        
        system_instruction_text = (
            "Bạn là một chuyên gia phân tích tài chính chuyên nghiệp. "
            "Dựa trên dữ liệu đã cung cấp, hãy đưa ra một nhận xét khách quan, ngắn gọn (khoảng 3-4 đoạn) về tình hình tài chính của doanh nghiệp. "
            "Đánh giá tập trung vào tốc độ tăng trưởng qua các chu kỳ, thay đổi cơ cấu tài sản và khả năng thanh toán hiện hành trong 3 năm/kỳ."
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
            "Dữ liệu này bao gồm tốc độ tăng trưởng, so sánh tuyệt đối/tương đối và tỷ trọng trong 3 kỳ Báo cáo tài chính, cùng với 3 chỉ số thanh toán hiện hành. "
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
    "1. Tải file Excel/CSV Báo cáo Tài chính (KHOẢN MỤC | YYYY-MM-DD | YYYY-MM-DD | YYYY-MM-DD)",
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

        # --- TIỀN XỬ LÝ (PRE-PROCESSING) DỮ LIỆU ĐỂ PHÙ HỢP VỚI LOGIC CŨ ---
        
        # 1. Đặt tên cột đầu tiên là 'Chỉ tiêu' (Dựa trên snippet 'KHOẢN MỤC')
        df_raw = df_raw.rename(columns={df_raw.columns[0]: 'Chỉ tiêu'})
        
        # 2. Xác định cột năm gần nhất ('Năm 3'), 'Năm 2', 'Năm 1'
        
        # TÌM KIẾM CỘT NGÀY THÁNG LINH HOẠT
        value_cols = []
        for col in df_raw.columns:
            col_str = str(col)
            # Tìm kiếm các chuỗi có dạng ngày tháng yyyy-mm-dd (VD: 2023-12-31)
            if len(col_str) >= 10 and col_str[4] == '-' and col_str[7] == '-' and col_str[:4].isdigit():
                 value_cols.append(col)
            # Hoặc tìm các cột có tên là năm đơn thuần (VD: 2023)
            elif col_str.isdigit() and len(col_str) == 4 and col_str.startswith('20'):
                 value_cols.append(col)
        
        if len(value_cols) < 3: # Yêu cầu 3 năm để tính toán 2 chu kỳ
            st.warning(f"Chỉ tìm thấy {len(value_cols)} cột năm. Ứng dụng cần ít nhất 3 năm/kỳ để so sánh.")
            st.stop()
            
        # Chọn 3 cột năm gần nhất (Sắp xếp theo tên cột/ngày tháng, mới nhất (Y3) lên đầu)
        value_cols.sort(key=lambda x: str(x), reverse=True)
        
        col_nam_3 = value_cols[0] # Newest (Năm 3)
        col_nam_2 = value_cols[1] # Middle (Năm 2)
        col_nam_1 = value_cols[2] # Oldest (Năm 1)
        
        # 3. Xóa các hàng chỉ chứa dữ liệu phụ (hàng phụ của Header gốc)
        # Hàng 0 trong df_raw (hàng thứ hai trong file gốc) thường chứa các giá trị NaN và tiêu đề phụ như "SS (+/-)"
        df_raw = df_raw.drop(df_raw.index[0])
        
        # 4. Tạo DataFrame mới chỉ chứa 4 cột cần thiết (Chỉ tiêu + 3 năm)
        df_final = df_raw[['Chỉ tiêu', col_nam_1, col_nam_2, col_nam_3]].copy()
        
        # 5. Đổi tên cột để phù hợp với hàm process_financial_data
        df_final.columns = ['Chỉ tiêu', 'Năm 1', 'Năm 2', 'Năm 3']
        
        # 6. Lọc bỏ các hàng NaN ở cột 'Chỉ tiêu' (các hàng trống)
        df_final = df_final.dropna(subset=['Chỉ tiêu'])
        
        # Xử lý dữ liệu
        df_processed = process_financial_data(df_final.copy())

        if df_processed is not None:
            
            # Lưu tên cột gốc (ngày tháng) để hiển thị
            Y1_Name = str(col_nam_1)
            Y2_Name = str(col_nam_2)
            Y3_Name = str(col_nam_3)
            
            # --- Chức năng 2 & 3: Hiển thị Kết quả theo Tabs ---
            st.subheader("2. Phân tích Tốc độ Tăng trưởng & 3. Phân tích Tỷ trọng Cơ cấu Tài sản")
            
            # 1. TẠO DATAFRAME TĂNG TRƯỞNG (GHÉP CỘT)
            df_growth = df_processed[['Chỉ tiêu', 'Năm 1', 'Năm 2', 'Năm 3', 
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
            
            # 2. TẠO DATAFRAME CƠ CẤU
            df_structure = df_processed[['Chỉ tiêu', 'Năm 1', 'Năm 2', 'Năm 3', 
                                         'Tỷ trọng Năm 1 (%)', 'Tỷ trọng Năm 2 (%)', 'Tỷ trọng Năm 3 (%)']].copy()
            
            # Đổi tên cột cho trực quan
            df_structure.columns = [
                'Chỉ tiêu', Y1_Name, Y2_Name, Y3_Name, 
                f'Tỷ trọng {Y1_Name} (%)', f'Tỷ trọng {Y2_Name} (%)', f'Tỷ trọng {Y3_Name} (%)'
            ]

            tab1, tab2 = st.tabs(["📈 Tốc độ Tăng trưởng (3 Năm)", "🏗️ Tỷ trọng Cơ cấu (3 Năm)"])
            
            with tab1:
                st.markdown("##### Bảng phân tích Tốc độ Tăng trưởng & So sánh Tuyệt đối (2 chu kỳ)")
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
                st.markdown("##### Bảng phân tích Tỷ trọng Cơ cấu Tài sản (%) (3 Năm)")
                st.dataframe(df_structure.style.format({
                    Y1_Name: '{:,.0f}',
                    Y2_Name: '{:,.0f}',
                    Y3_Name: '{:,.0f}',
                    f'Tỷ trọng {Y1_Name} (%)': '{:.2f}%',
                    f'Tỷ trọng {Y2_Name} (%)': '{:.2f}%',
                    f'Tỷ trọng {Y3_Name} (%)': '{:.2f}%'
                }), use_container_width=True, hide_index=True)
            
            # Khởi tạo giá trị mặc định cho chỉ số thanh toán
            thanh_toan_hien_hanh_N1 = "N/A"
            thanh_toan_hien_hanh_N2 = "N/A"
            thanh_toan_hien_hanh_N3 = "N/A"

            # --- Chức năng 4: Tính Chỉ số Tài chính ---
            st.subheader("4. Các Chỉ số Tài chính Cơ bản")
            
            try:
                # Lấy Tài sản ngắn hạn (TSNH)
                tsnh_n3 = df_processed[df_processed['Chỉ tiêu'].str.contains('TÀI SẢN NGẮN HẠN', case=False, na=False)]['Năm 3'].iloc[0]
                tsnh_n2 = df_processed[df_processed['Chỉ tiêu'].str.contains('TÀI SẢN NGẮN HẠN', case=False, na=False)]['Năm 2'].iloc[0]
                tsnh_n1 = df_processed[df_processed['Chỉ tiêu'].str.contains('TÀI SẢN NGẮN HẠN', case=False, na=False)]['Năm 1'].iloc[0]

                # Lấy Nợ ngắn hạn (NNH)
                no_ngan_han_N3 = df_processed[df_processed['Chỉ tiêu'].str.contains('NỢ NGẮN HẠN', case=False, na=False)]['Năm 3'].iloc[0]  
                no_ngan_han_N2 = df_processed[df_processed['Chỉ tiêu'].str.contains('NỢ NGẮN HẠN', case=False, na=False)]['Năm 2'].iloc[0]
                no_ngan_han_N1 = df_processed[df_processed['Chỉ tiêu'].str.contains('NỢ NGẮN HẠN', case=False, na=False)]['Năm 1'].iloc[0]

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
            data_for_chat_context = pd.DataFrame({
                'Chỉ tiêu': [
                    'Toàn bộ Bảng phân tích (dữ liệu thô)', 
                    f'Thanh toán hiện hành ({Y1_Name})', 
                    f'Thanh toán hiện hành ({Y2_Name})',
                    f'Thanh toán hiện hành ({Y3_Name})'
                ],
                'Giá trị': [
                    df_processed.to_markdown(index=False),
                    f"{thanh_toan_hien_hanh_N1}", 
                    f"{thanh_toan_hien_hanh_N2}",
                    f"{thanh_toan_hien_hanh_N3}"
                ]
            }).to_markdown(index=False)
            st.session_state.data_for_chat = data_for_chat_context
            
            # Cập nhật tin nhắn chào mừng nếu data đã sẵn sàng
            if st.session_state.messages[0]["content"].startswith("Xin chào!") or st.session_state.messages[0]["content"].startswith("Phân tích"):
                 st.session_state.messages[0]["content"] = f"Phân tích 3 kỳ ({Y1_Name} đến {Y3_Name}) đã hoàn tất! Bây giờ bạn có thể hỏi tôi bất kỳ điều gì về 'Tốc độ tăng trưởng', 'Tỷ trọng' và 'Chỉ số thanh toán hiện hành' của báo cáo này."

            # --- Chức năng 5: Nhận xét AI ---
            st.subheader("5. Nhận xét Tình hình Tài chính (AI)")
            
            # Chuẩn bị dữ liệu để gửi cho AI (giống hệt logic data_for_chat_context)
            try:
                tsnh_growth_y2y1 = f"{df_processed[df_processed['Chỉ tiêu'].str.contains('TÀI SẢN NGẮN HẠN', case=False, na=False)]['Growth (Y2 vs Y1)'].iloc[0]:.2f}%"
                tsnh_growth_y3y2 = f"{df_processed[df_processed['Chỉ tiêu'].str.contains('TÀI SẢN NGẮN HẠN', case=False, na=False)]['Growth (Y3 vs Y2)'].iloc[0]:.2f}%"
            except IndexError:
                tsnh_growth_y2y1 = "N/A"
                tsnh_growth_y3y2 = "N/A"

            data_for_ai = pd.DataFrame({
                'Chỉ tiêu': [
                    'Toàn bộ Bảng phân tích (dữ liệu thô)', 
                    f'Tăng trưởng Tài sản ngắn hạn ({Y2_Name} vs {Y1_Name})',
                    f'Tăng trưởng Tài sản ngắn hạn ({Y3_Name} vs {Y2_Name})',
                    f'Thanh toán hiện hành ({Y1_Name})', 
                    f'Thanh toán hiện hành ({Y2_Name})',
                    f'Thanh toán hiện hành ({Y3_Name})'
                ],
                'Giá trị': [
                    df_processed.to_markdown(index=False),
                    tsnh_growth_y2y1,
                    tsnh_growth_y3y2,
                    f"{thanh_toan_hien_hanh_N1}", 
                    f"{thanh_toan_hien_hanh_N2}", 
                    f"{thanh_toan_hien_hanh_N3}"
                ]
            }).to_markdown(index=False)

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

# --- Chức năng 6: Khung Chatbot tương tác ---
st.subheader("6. Trò chuyện và Hỏi đáp (Gemini AI)")
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
