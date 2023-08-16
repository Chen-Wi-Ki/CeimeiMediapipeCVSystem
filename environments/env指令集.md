#創建環境:
python3 -m venv tutorial-env

#開啟環境
source env/bin/activate

#關閉環境
deactivate

#記錄環境資訊
pip3 freeze > requirements.txt

#透過requirements.txt安裝程式
