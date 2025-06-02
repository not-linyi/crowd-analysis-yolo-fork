
from src.app import app

if __name__ == '__main__':
    print("启动BRT人群分析系统Web应用...")
    app.run(debug=False, host='0.0.0.0', port=5000)
