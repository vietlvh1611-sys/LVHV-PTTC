import streamlit as st
import pandas as pd
from google import genai
from google.genai.errors import APIError
# Thêm import cho GenerationConfig để truyền system_instruction
from google.genai.types import GenerationConfig 
# LƯU Ý LỖI: Loại bỏ import SystemInstruction vì không tương thích trong môi trường này
# from google.genai.types import SystemInstruction

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
    """Thực hiện các phép tính Tăng trưởng và Tỷ trọng."""
    
    # Đảm bảo các giá trị là số để tính toán
    numeric_cols = ['Năm trước', 'Năm sau']
    for col in numeric_cols:
        # Sử dụng df[col] = df[col]... thay vì df[col] = pd.to_numeric(col... như lỗi trước đó
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # 1. Tính Tốc độ Tăng trưởng
    # Dùng .replace(0, 1e-9) cho Series Pandas để tránh lỗi chia cho 0
    df['Tốc độ tăng trưởng (%)'] = (
        (df['Năm sau'] - df['Năm trước']) / df['Năm trước'].replace(0, 1e-9)
    ) * 100

    # 2. Tính Tỷ trọng theo Tổng Tài sản
    # Lọc chỉ tiêu "TỔNG CỘNG TÀI SẢN"
    # LƯU Ý: Đảm bảo dữ liệu của bạn có dòng này (hoặc "Tài sản")
    tong_tai_san_row = df[df['Chỉ tiêu'].str.contains('TỔNG CỘNG TÀI SẢN', case=False, na=False)]
    
    if tong_tai_san_row.empty:
        # Thử tìm từ khóa chung hơn
        tong_tai_san_row = df[df['Chỉ tiêu'].str.contains('TỔNG CỘNG', case=False, na=False)]
        if tong_tai_san_row.empty:
            raise ValueError("Không tìm thấy chỉ tiêu 'TỔNG CỘNG TÀI SẢN' hoặc 'TỔNG CỘNG' để tính tỷ trọng. Vui lòng kiểm tra tên chỉ tiêu trong file.")

    # Lấy giá trị của dòng TỔNG CỘNG (có thể có nhiều dòng nếu chỉ tìm 'TỔNG CỘNG', nên dùng .iloc[0])
    tong_tai_san_N_1 = tong_tai_san_row['Năm trước'].iloc[0]
    tong_tai_san_N = tong_tai_san_row['Năm sau'].iloc[0]
    
    divisor_N_1 = tong_tai_san_N_1 if tong_tai_san_N_1 != 0 else 1e-9
    divisor_N = tong_tai_san_N if tong_tai_san_N != 0 else 1e-9

    # Tính tỷ trọng với mẫu số đã được xử lý
    df['Tỷ trọng Năm trước (%)'] = (df['Năm trước'] / divisor_N_1) * 100
    df['Tỷ trọng Năm sau (%)'] = (df['Năm sau'] / divisor_N) * 100
    
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
            "Đánh giá tập trung vào tốc độ tăng trưởng, thay đổi cơ cấu tài sản và khả năng thanh toán hiện hành."
        )
        
        # SỬA LỖI: Loại bỏ role='system', truyền system_instruction qua config
        config = GenerationConfig(system_instruction=system_instruction_text)


        user_prompt = f"""
        Dữ liệu thô và chỉ số:
        {data_for_ai}
        """

        # Truyền prompt duy nhất
        response = client.models.generate_content(
            model=model_name,
            contents=user_prompt, # Chỉ truyền prompt người dùng
            config=config # Truyền hướng dẫn hệ thống qua config
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
            "Dữ liệu này bao gồm 'Tốc độ tăng trưởng (%)' và 'Tỷ trọng Năm trước/sau (%)' của các chỉ tiêu Báo cáo tài chính, cùng với các chỉ số thanh toán. "
            "Nếu người dùng hỏi một câu không liên quan đến dữ liệu tài chính hoặc phân tích, hãy lịch sự từ chối trả lời. "
            "Dữ liệu tài chính đã xử lý (được trình bày dưới dạng Markdown để bạn dễ hiểu): \n\n" + context_data
        )
        
        # SỬA LỖI: Loại bỏ role='system', truyền system_instruction qua config
        config = GenerationConfig(system_instruction=system_instruction_text)
        
        # 2. Chuyển đổi lịch sử Streamlit sang định dạng Gemini
        gemini_history = []
        # Bắt đầu từ tin nhắn thứ hai trong lịch sử ST (bỏ qua tin nhắn chào mừng đầu tiên)
        for msg in chat_history_st[1:]: 
            # Đảm bảo chỉ có role 'user' và 'model' được sử dụng
            role = "user" if msg["role"] == "user" else "model"
            gemini_history.append({"role": role, "parts": [{"text": msg["content"]}]})
        
        # 3. Thêm prompt mới nhất vào cuối contents (Không cần thêm System Content ở đây nữa)
        full_contents = gemini_history
        full_contents.append({"role": "user", "parts": [{"text": prompt}]})

        # 4. Gọi API
        response = client.models.generate_content(
            model=model_name,
            contents=full_contents, # Chỉ truyền lịch sử chat (user/model)
            config=config # Truyền hướng dẫn hệ thống qua config
        )
        return response.text

    except APIError as e:
        return f"Lỗi gọi Gemini API: Vui lòng kiểm tra Khóa API hoặc giới hạn sử dụng. Chi tiết lỗi: {e}"
    except Exception as e:
        return f"Đã xảy ra lỗi không xác định: {e}"


# --- Chức năng 1: Tải File ---
uploaded_file = st.file_uploader(
    "1. Tải file Excel/CSV Báo cáo Tài chính (KHOẢN MỤC | YYYY-MM-DD | YYYY-MM-DD)",
    type=['xlsx', 'xls', 'csv']
)

if uploaded_file is not None:
    try:
        # Xử lý file dựa trên định dạng
        # CHUYỂN SANG DÙNG header=0 VÀ BỎ QUA HÀNG THỨ HAI ĐỂ LẤY ĐÚNG TÊN CỘT NGÀY THÁNG
        if uploaded_file.name.endswith(('.xlsx', '.xls')):
            # Đọc Excel, lấy hàng đầu tiên (index 0) làm header
            df_raw = pd.read_excel(uploaded_file, header=0)
        elif uploaded_file.name.endswith('.csv'):
            # Đọc CSV, lấy hàng đầu tiên (index 0) làm header
            df_raw = pd.read_csv(uploaded_file, header=0)
        else:
            raise Exception("Định dạng file không được hỗ trợ.")

        # --- TIỀN XỬ LÝ (PRE-PROCESSING) DỮ LIỆU ĐỂ PHÙ HỢP VỚI LOGIC CŨ ---
        
        # Hàng 1 (index 1) trong file gốc là hàng phụ (SS (+/-), SS (%)) nên ta xóa nó đi nếu nó đã bị đọc vào DF
        # Nếu dùng header=0, hàng này sẽ trở thành hàng đầu tiên của dữ liệu
        
        # 1. Đặt tên cột đầu tiên là 'Chỉ tiêu' (Dựa trên snippet 'KHOẢN MỤC')
        # Cột đầu tiên trong DF sau khi dùng header=0 là 'KHOẢN MỤC'
        df_raw = df_raw.rename(columns={df_raw.columns[0]: 'Chỉ tiêu'})
        
        # 2. Xác định cột năm gần nhất ('Năm sau') và năm trước đó ('Năm trước')
        
        # TÌM KIẾM CỘT NGÀY THÁNG LINH HOẠT
        value_cols = []
        for col in df_raw.columns:
            col_str = str(col)
            # Tìm kiếm các chuỗi chứa năm 20XX (Ví dụ: '2023-12-31')
            # Cột cần tìm là chuỗi có dạng ngày tháng yyyy-mm-dd
            if len(col_str) >= 10 and col_str[4] == '-' and col_str[7] == '-' and col_str[:4].isdigit():
                 value_cols.append(col)
            # Hoặc tìm các cột có tên là năm đơn thuần (ví dụ: 2023)
            elif col_str.isdigit() and len(col_str) == 4 and col_str.startswith('20'):
                 value_cols.append(col)
        
        if len(value_cols) < 2:
            st.warning(f"Chỉ tìm thấy {len(value_cols)} cột năm. Ứng dụng cần ít nhất 2 năm để so sánh.")
            st.stop()
            
        # Chọn 2 cột năm gần nhất (Sắp xếp theo tên cột/ngày tháng)
        value_cols.sort(key=lambda x: str(x), reverse=True)
        
        col_nam_sau = value_cols[0] 
        col_nam_truoc = value_cols[1]
        
        # 3. Xóa các hàng chỉ chứa dữ liệu phụ (hàng phụ của Header gốc)
        # Hàng 0 trong df_raw (hàng thứ hai trong file gốc) thường chứa các giá trị NaN và tiêu đề phụ như "SS (+/-)"
        df_raw = df_raw.drop(df_raw.index[0])
        
        # 4. Tạo DataFrame mới chỉ chứa 3 cột cần thiết
        df_final = df_raw[['Chỉ tiêu', col_nam_truoc, col_nam_sau]].copy()
        
        # 5. Đổi tên cột để phù hợp với hàm process_financial_data
        df_final.columns = ['Chỉ tiêu', 'Năm trước', 'Năm sau']
        
        # 6. Lọc bỏ các hàng NaN ở cột 'Chỉ tiêu' (các hàng trống)
        df_final = df_final.dropna(subset=['Chỉ tiêu'])
        
        # Xử lý dữ liệu
        df_processed = process_financial_data(df_final.copy())

        if df_processed is not None:
            
            # --- Chức năng 2 & 3: Hiển thị Kết quả ---
            st.subheader("2. Tốc độ Tăng trưởng & 3. Tỷ trọng Cơ cấu Tài sản")
            st.dataframe(df_processed.style.format({
                'Năm trước': '{:,.0f}',
                'Năm sau': '{:,.0f}',
                'Tốc độ tăng trưởng (%)': '{:.2f}%',
                'Tỷ trọng Năm trước (%)': '{:.2f}%',
                'Tỷ trọng Năm sau (%)': '{:.2f}%'
            }), use_container_width=True)
            
            # Khởi tạo giá trị mặc định cho chỉ số thanh toán
            thanh_toan_hien_hanh_N = "N/A"
            thanh_toan_hien_hanh_N_1 = "N/A"

            # --- Chức năng 4: Tính Chỉ số Tài chính ---
            st.subheader("4. Các Chỉ số Tài chính Cơ bản")
            
            try:
                # Lấy Tài sản ngắn hạn
                tsnh_n = df_processed[df_processed['Chỉ tiêu'].str.contains('TÀI SẢN NGẮN HẠN', case=False, na=False)]['Năm sau'].iloc[0]
                tsnh_n_1 = df_processed[df_processed['Chỉ tiêu'].str.contains('TÀI SẢN NGẮN HẠN', case=False, na=False)]['Năm trước'].iloc[0]

                # Lấy Nợ ngắn hạn (Dùng giá trị giả định hoặc lọc từ file nếu có)
                no_ngan_han_N = df_processed[df_processed['Chỉ tiêu'].str.contains('NỢ NGẮN HẠN', case=False, na=False)]['Năm sau'].iloc[0]  
                no_ngan_han_N_1 = df_processed[df_processed['Chỉ tiêu'].str.contains('NỢ NGẮN HẠN', case=False, na=False)]['Năm trước'].iloc[0]

                # Tính toán
                thanh_toan_hien_hanh_N = tsnh_n / no_ngan_han_N
                thanh_toan_hien_hanh_N_1 = tsnh_n_1 / no_ngan_han_N_1
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(
                        label="Chỉ số Thanh toán Hiện hành (Năm trước)",
                        value=f"{thanh_toan_hien_hanh_N_1:.2f} lần"
                    )
                with col2:
                    st.metric(
                        label="Chỉ số Thanh toán Hiện hành (Năm sau)",
                        value=f"{thanh_toan_hien_hanh_N:.2f} lần",
                        delta=f"{thanh_toan_hien_hanh_N - thanh_toan_hien_hanh_N_1:.2f}"
                    )
                    
            except IndexError:
                st.warning("Thiếu chỉ tiêu 'TÀI SẢN NGẮN HẠN' hoặc 'NỢ NGẮN HẠN' để tính chỉ số.")
                thanh_toan_hien_hanh_N = "N/A" 
                thanh_toan_hien_hanh_N_1 = "N/A"
            except ZeroDivisionError:
                st.error("Lỗi chia cho 0 khi tính chỉ số thanh toán. Vui lòng kiểm tra dữ liệu 'Nợ Ngắn Hạn' (Năm trước hoặc Năm sau)!")
                thanh_toan_hien_hanh_N = "N/A"
                thanh_toan_hien_hanh_N_1 = "N/A"
            
            # --- CẬP NHẬT DỮ LIỆU CHO CHATBOT (CONTEXT) ---
            data_for_chat_context = pd.DataFrame({
                'Chỉ tiêu': [
                    'Toàn bộ Bảng phân tích (dữ liệu thô)', 
                    'Thanh toán hiện hành (N-1)', 
                    'Thanh toán hiện hành (N)'
                ],
                'Giá trị': [
                    df_processed.to_markdown(index=False),
                    f"{thanh_toan_hien_hanh_N_1}", 
                    f"{thanh_toan_hien_hanh_N}"
                ]
            }).to_markdown(index=False)
            st.session_state.data_for_chat = data_for_chat_context
            
            # Cập nhật tin nhắn chào mừng nếu data đã sẵn sàng
            if st.session_state.messages[0]["content"].startswith("Xin chào!") or st.session_state.messages[0]["content"].startswith("Phân tích"):
                 st.session_state.messages[0]["content"] = "Phân tích đã hoàn tất! Bây giờ bạn có thể hỏi tôi bất kỳ điều gì về 'Tốc độ tăng trưởng', 'Tỷ trọng' và 'Chỉ số thanh toán hiện hành' của báo cáo này."

            # --- Chức năng 5: Nhận xét AI ---
            st.subheader("5. Nhận xét Tình hình Tài chính (AI)")
            
            # Chuẩn bị dữ liệu để gửi cho AI (giống hệt logic data_for_chat_context)
            try:
                tsnh_growth = f"{df_processed[df_processed['Chỉ tiêu'].str.contains('TÀI SẢN NGẮN HẠN', case=False, na=False)]['Tốc độ tăng trưởng (%)'].iloc[0]:.2f}%"
            except IndexError:
                tsnh_growth = "N/A"

            data_for_ai = pd.DataFrame({
                'Chỉ tiêu': [
                    'Toàn bộ Bảng phân tích (dữ liệu thô)', 
                    'Tăng trưởng Tài sản ngắn hạn (%)',
                    'Thanh toán hiện hành (N-1)', 
                    'Thanh toán hiện hành (N)'
                ],
                'Giá trị': [
                    df_processed.to_markdown(index=False),
                    tsnh_growth,
                    f"{thanh_toan_hien_hanh_N_1}", 
                    f"{thanh_toan_hien_hanh_N}"
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
        st.error(f"Có lỗi xảy ra khi đọc hoặc xử lý file: {e}. Vui lòng kiểm tra định dạng file.")
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
