from flask import Flask,request
from flask_restful import Api, Resource
from flask_cors import CORS, cross_origin
from video_test import video_test, send_visualization, send_accuracy

app = Flask(__name__)
api = Api(app)
cors = CORS(app)

########################### Login method ###########################

@app.route('/login',methods=['POST'])
def login():
    data = request.get_json()
    password = data['password']
    if isinstance(password,str):
        if password == "OracleLogin":
            return {"data":"SUCCESS","mode":"Oracle"}
        elif password == "NormalLogin":
            return {"data": "SUCCESS","mode":"Normal"}
    else:
        return {"data":"FAILURE","mode":"none"}

########################### Video prediction method ###########################

@app.route('/video',methods=['POST'])    
def upload():
    url = request.get_json()
    print(url)
    data = url['link']
    print(type(data))
    if isinstance(data,str):
        res = video_test(data)
        print(res)
        return {'res':res}
    else:
        return {'result':'Please send proper link to the video file'}
    
########################### Accuracy method ###########################

@app.route('/accuracy',methods=['GET'])
def accuracy():
    response = send_accuracy()
    print(response)
    return {'res':response}

########################### Visualization image method ###########################

@app.route('/visualization',methods=['GET'])
def visualization():
    response = send_visualization()
    print(response)
    return {'res':response}

if __name__ == "__main__":
    app.run(debug=False)