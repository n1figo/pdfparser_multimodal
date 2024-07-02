from flask import Flask, request, render_template_string
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from datetime import datetime

app = Flask(__name__)

# 6월 데이터 (실제로는 데이터베이스나 파일에서 로드해야 합니다)
june_data = {
    'date': ['2024-06-01', '2024-06-30'],
    'MAU': [47719, 853795]
}

@app.route('/', methods=['GET', 'POST'])
def predict():
    prediction = None
    if request.method == 'POST':
        july_first_mau = float(request.form['july_first_mau'])
        
        # 데이터 준비
        df = pd.DataFrame(june_data)
        df['date'] = pd.to_datetime(df['date'])
        df['month_order'] = range(1, len(df) + 1)
        
        # 7월 1일 데이터 추가
        july_first = pd.DataFrame({
            'date': [datetime(2024, 7, 1)],
            'MAU': [july_first_mau],
            'month_order': [len(df) + 1]
        })
        df = pd.concat([df, july_first])
        
        # 모델 학습
        model = LinearRegression()
        model.fit(df[['month_order']], df['MAU'])
        
        # 7월 말 예측
        july_end_month_order = len(df) + 1
        prediction = model.predict([[july_end_month_order]])[0]
    
    return render_template_string('''
        <h1>MAU 예측 서비스</h1>
        <form method="post">
            7월 1일 MAU: <input type="number" name="july_first_mau" required>
            <input type="submit" value="예측">
        </form>
        {% if prediction %}
        <p>예측된 7월 말 MAU: {{ prediction|round|int }}</p>
        {% endif %}
    ''', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)