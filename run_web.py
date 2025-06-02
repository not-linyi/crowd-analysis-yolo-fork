
from src.app import app

if __name__ == '__main__':
    print("启动BRT人群分析系统Web应用...")
    print("请在浏览器中访问: http://localhost:5000")
    app.run(debug=False, host='0.0.0.0', port=5000)
