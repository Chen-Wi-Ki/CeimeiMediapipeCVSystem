from sanic import Sanic
from sanic.response import text,html,file
import os
app = Sanic("MyWebApp")

@app.route("/")
async def StartEvent(request):
    #搜索Documents資料夾裡的檔案
    dir_path = '/home/wiki/Documents'
    all_file_name = os.listdir(dir_path)
    all_file_name_len = len(all_file_name)
    #print(all_file_name)
    if all_file_name_len==0:
        return text("None Data")
    else:
        TempReq='<!DOCTYPE html><html lang="en"><meta charset="UTF-8"><div>奇美手術縫合訓練系統資料集</div><br>'
        for i in range(0,all_file_name_len,1):
            TempReq = TempReq+'<a href="/download?'+all_file_name[i]+'">'+all_file_name[i]+'</a><br>'
        return html(TempReq)

#下載指定的CSV檔案
@app.route('/download', methods=["GET"])
async def handler(request):
    file_name = request.url.split('?')[1]
    dir_path = '/home/wiki/Documents/'+file_name
    print(request.ip,' -->download file:',file_name)
    return await file(dir_path,filename=file_name)
    #return text('OK!')

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=7777, workers=1, access_log=False)
