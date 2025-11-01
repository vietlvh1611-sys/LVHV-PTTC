import streamlit as st
import pandas as pd
# (THAY ĐỔI 1: Sử dụng thư viện Gemini mới nhất)
import google.generativeai as genai
# from google import genai (BỎ)
# from google.genai.errors import APIError (BỎ)

# --- Cấu hình Trang Streamlit ---
st.set_page_config(
    page_title="App Phân Tích Báo Cáo Tài Chính",
    layout="wide"
)

st.title("Ứng dụng Phân Tích Báo Cáo Tài Chính 📊")

# --- (THAY ĐỔI 2: Cấu hình API Key toàn cục) ---
# Lấy API key từ Streamlit Secrets
API_KEY = st.secrets.get("GEMINI_API_KEY")

if not API_KEY:
    # Nếu không có key, hiển thị lỗi và dừng ứng dụng
    st.error("Lỗi: Không tìm thấy Khóa API. Vui lòng cấu hình 'GEMINI_API_KEY' trong Streamlit Secrets để bật tính năng AI.")
    st.stop()

try:
    # Cấu hình API key cho thư viện
    genai.configure(api_key=API_KEY)
except Exception as e:
    # Bắt lỗi nếu key không hợp lệ
    st.error(f"Lỗi cấu hình Gemini API (Key có thể không hợp lệ): {e}")
    st.stop()

# --- Hàm tính toán chính (Giữ nguyên) ---
@st.cache_data
def process_financial_data(df):
    """Thực hiện các phép tính Tăng trưởng và Tỷ trọng."""
    
    # Đảm bảo các giá trị là số để tính toán
    numeric_cols = ['Năm trước', 'Năm sau']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # 1. Tính Tốc độ Tăng trưởng
    df['Tốc độ tăng trưởng (%)'] = (
        (df['Năm sau'] - df['Năm trước']) / df['Năm trước'].replace(0, 1e-9)
    ) * 100

    # 2. Tính Tỷ trọng theo Tổng Tài sản
    tong_tai_san_row = df[df['Chỉ tiêu'].str.contains('TỔNG CỘNG TÀI SẢN', case=False, na=False)]
    
    if tong_tai_san_row.empty:
        raise ValueError("Không tìm thấy chỉ tiêu 'TỔNG CỘNG TÀI SẢN'.")

    tong_tai_san_N_1 = tong_tai_san_row['Năm trước'].iloc[0]
    tong_tai_san_N = tong_tai_san_row['Năm sau'].iloc[0]

    # Xử lý chia cho 0
    divisor_N_1 = tong_tai_san_N_1 if tong_tai_san_N_1 != 0 else 1e-9
    divisor_N = tong_tai_san_N if tong_tai_san_N != 0 else 1e-9

    df['Tỷ trọng Năm trước (%)'] = (df['Năm trước'] / divisor_N_1) * 100
    df['Tỷ trọng Năm sau (%)'] = (df['Năm sau'] / divisor_N) * 100
    
    return df

# --- (THAY ĐỔI 3: Cập nhật hàm gọi API theo cú pháp mới) ---
def get_ai_analysis(data_for_ai):
    """Gửi dữ liệu phân tích đến Gemini API và nhận nhận xét. (Đã cập nhật)"""
    try:
        # 1. Chọn Model (Không cần client. nữa)
        model = genai.GenerativeModel('gemini-2.5-flash') 

        prompt = f"""
        Bạn là một chuyên gia phân tích tài chính chuyên nghiệp. Dựa trên các chỉ số tài chính sau, hãy đưa ra một nhận xét khách quan, ngắn gọn (khoảng 3-4 đoạn) về tình hình tài chính của doanh nghiệp. Đánh giá tập trung vào tốc độ tăng trưởng, thay đổi cơ cấu tài sản và khả năng thanh toán hiện hành.
        
        Dữ liệu thô và chỉ số:
        {data_for_ai}
        """

        # 2. Gọi API (Cú pháp mới)
        response = model.generate_content(prompt)
        return response.text

    except Exception as e:
        # Bắt lỗi chung
        return f"Đã xảy ra lỗi không xác định khi gọi AI: {e}"


# --- Chức năng 1: Tải File ---
uploaded_file = st.file_uploader(
    "1. Tải file Excel Báo cáo Tài chính (Chỉ tiêu | Năm trước | Năm sau)",
    type=['xlsx', 'xls']
)

if uploaded_file is not None:
    try:
        df_raw = pd.read_excel(uploaded_file)
        
        # Tiền xử lý: Đảm bảo chỉ có 3 cột quan trọng
        df_raw.columns = ['Chỉ tiêu', 'Năm trước', 'Năm sau']
        
        # Xử lý dữ liệu
        df_processed = process_financial_data(df_raw.copy())

        if df_processed is not None:
            
            # --- Chức năng 2 & 3: Hiển thị Kết quả (Giữ nguyên) ---
            st.subheader("2. Tốc độ Tăng trưởng & 3. Tỷ trọng Cơ cấu Tài sản")
            st.dataframe(df_processed.style.format({
                'Năm trước': '{:,.0f}',
                'Năm sau': '{:,.0f}',
                'Tốc độ tăng trưởng (%)': '{:.2f}%',
                'Tỷ trọng Năm trước (%)': '{:.2f}%',
                'Tỷ trọng Năm sau (%)': '{:.2f}%'
            }), use_container_width=True)
            
            # --- Chức năng 4: Tính Chỉ số Tài chính (Giữ nguyên) ---
            st.subheader("4. Các Chỉ số Tài chính Cơ bản")
            
            try:
                # Lọc giá trị
                tsnh_n = df_processed[df_processed['Chỉ tiêu'].str.contains('TÀI SẢN NGẮN HẠN', case=False, na=False)]['Năm sau'].iloc[0]
                tsnh_n_1 = df_processed[df_processed['Chỉ tiêu'].str.contains('TÀI SẢN NGẮN HẠN', case=False, na=False)]['Năm trước'].iloc[0]
                no_ngan_han_N = df_processed[df_processed['Chỉ tiêu'].str.contains('NỢ NGẮN HẠN', case=False, na=False)]['Năm sau'].iloc[0]  
                no_ngan_han_N_1 = df_processed[df_processed['Chỉ tiêu'].str.contains('NỢ NGẮN HẠN', case=False, na=False)]['Năm trước'].iloc[0]

                # Tính toán (xử lý chia cho 0)
                thanh_toan_hien_hanh_N = tsnh_n / (no_ngan_han_N if no_ngan_han_N != 0 else 1e-9)
                thanh_toan_hien_hanh_N_1 = tsnh_n_1 / (no_ngan_han_N_1 if no_ngan_han_N_1 != 0 else 1e-9)
                
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
            
            # --- Chức năng 5: Nhận xét AI (Giữ nguyên logic, cập nhật cách gọi) ---
            st.subheader("5. Nhận xét Tình hình Tài chính (AI)")
            
            # Chuẩn bị dữ liệu để gửi cho AI (Giữ nguyên)
            data_for_ai = pd.DataFrame({
                'Chỉ tiêu': [
                    'Toàn bộ Bảng phân tích (dữ liệu thô)', 
                    'Tăng trưởng Tài sản ngắn hạn (%)', 
                    'Thanh toán hiện hành (N-1)', 
                    'Thanh toán hiện hành (N)'
                ],
                'Giá trị': [
                    df_processed.to_markdown(index=False),
                    f"{df_processed[df_processed['Chỉ tiêu'].str.contains('TÀI SẢN NGẮN HẠN', case=False, na=False)]['Tốc độ tăng trưởng (%)'].iloc[0]:.2f}%", 
                    f"{thanh_toan_hien_hanh_N_1}", 
                    f"{thanh_toan_hien_hanh_N}"
                ]
            }).to_markdown(index=False) 

            if st.button("Yêu cầu AI Phân tích (Tóm tắt)"):
                # (THAY ĐỔI 4: Không cần lấy key ở đây nữa)
                with st.spinner('Đang gửi dữ liệu và chờ Gemini phân tích...'):
                    # Chỉ cần gọi hàm (không cần truyền key)
                    ai_result = get_ai_analysis(data_for_ai)
                    st.markdown("**Kết quả Phân tích từ Gemini AI:**")
                    st.info(ai_result)

            # --- (THAY ĐỔI 5: THÊM KHUNG CHAT MỚI) ---
            st.subheader("6. Trò chuyện Tương tác với AI (Gemini)")
            st.markdown("Hỏi AI bất cứ điều gì liên quan đến dữ liệu vừa tải lên hoặc các khái niệm tài chính chung.")

            # Logic để reset chat nếu file mới được tải lên
            current_file_name = uploaded_file.name
            if "current_file" not in st.session_state or st.session_state.current_file != current_file_name:
                # Lưu tên file mới
                st.session_state.current_file = current_file_name
                # Reset lịch sử tin nhắn
                st.session_state.messages = [] 
                
                # Tạo ngữ cảnh ban đầu cho AI
                initial_context = f"""
                Bạn là một trợ lý tài chính. Người dùng vừa tải lên một tệp có tên '{current_file_name}'.
                Dữ liệu đã xử lý (dưới dạng markdown) là:
                {df_processed.to_markdown(index=False)}
                
                Các chỉ số quan trọng:
                Chỉ số thanh toán hiện hành (Năm trước): {thanh_toan_hien_hanh_N_1}
                Chỉ số thanh toán hiện hành (Năm sau): {thanh_toan_hien_hanh_N}
                
                Bây giờ, hãy sẵn sàng trả lời các câu hỏi của người dùng về dữ liệu này. 
                Hãy bắt đầu bằng cách chào họ và xác nhận đã nhận dữ liệu.
                """
                
                # Chọn model cho chat
                model = genai.GenerativeModel('gemini-2.5-flash')
                # Bắt đầu một chat session MỚI với ngữ cảnh
                st.session_state.chat_session = model.start_chat(
                    history=[
                        {"role": "user", "parts": [initial_context]},
                        # Tin nhắn chào tự động
                        {"role": "model", "parts": ["Chào bạn! Tôi đã nhận và phân tích dữ liệu từ tệp của bạn. Bạn muốn hỏi tôi điều gì cụ thể về các chỉ số này hoặc các khái niệm tài chính liên quan?"]}
                    ]
                )
                # Thêm tin nhắn chào vào lịch sử để hiển thị
                st.session_state.messages = [
                    {"role": "assistant", "content": "Chào bạn! Tôi đã nhận và phân tích dữ liệu từ tệp của bạn. Bạn muốn hỏi tôi điều gì cụ thể về các chỉ số này hoặc các khái niệm tài chính liên quan?"}
                ]

            # Hiển thị lịch sử chat
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            # Nhận input từ người dùng
            if prompt := st.chat_input("Hỏi AI về dữ liệu này..."):
                # Thêm tin nhắn của user vào lịch sử
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)

                # Gửi tin nhắn đến Gemini và nhận phản hồi
                with st.chat_message("assistant"):
                    with st.spinner("Gemini đang suy nghĩ..."):
                        try:
                            # Gửi tin nhắn bằng session đã có ngữ cảnh
                            response = st.session_state.chat_session.send_message(prompt)
                            response_text = response.text
                            
                            st.markdown(response_text)
                            # Thêm phản hồi của AI vào lịch sử
                            st.session_state.messages.append({"role": "assistant", "content": response_text})
                            
                        except Exception as e:
                            st.error(f"Lỗi khi gửi tin nhắn: {e}")

    except ValueError as ve:
        st.error(f"Lỗi cấu trúc dữ liệu: {ve}")
    except Exception as e:
        st.error(f"Có lỗi xảy ra khi đọc hoặc xử lý file: {e}. Vui lòng kiểm tra định dạng file.")

else:
    st.info("Vui lòng tải lên file Excel để bắt đầu phân tích và trò chuyện với AI.")
